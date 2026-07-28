//! Tests for the three recall levers of [`DiskIndex`]: distance-bounded
//! adaptive probing (`SearchOptions::probe_epsilon`), exact rescoring over
//! stored full-precision vectors (`store_vectors` + `rescore_k`), and
//! SPANN-style boundary multi-assignment (`set_replication`).
//!
//! Oracle discipline: every flat oracle replays the identical add/remove/
//! write history as the index under test, so quantized codes (incl. the
//! TQ+ calibration) match exactly and recall deltas measure only the lever
//! under test.

use std::collections::HashSet;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::{DiskIndex, IdMapIndex, SearchOptions};

const DIM: usize = 64;
const BITS: usize = 4;
const TARGET: usize = 256;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-lever-{}-{}", nonce, name));
    p
}

/// Unit-normalized gaussian-mixture vectors (same recipe as the
/// partitioned-mode tests): clusterable by construction, with enough noise
/// that cluster boundaries carry real mass.
fn mixture_vectors(n_clusters: usize, per_cluster: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut draw = |scale: f32| -> Vec<f32> {
        (0..DIM)
            .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng) * scale)
            .collect()
    };
    let anchors: Vec<Vec<f32>> = (0..n_clusters).map(|_| draw(1.0)).collect();

    let mut vectors = Vec::with_capacity(n_clusters * per_cluster * DIM);
    for anchor in &anchors {
        for _ in 0..per_cluster {
            let noise = draw(0.25);
            let row: Vec<f32> = anchor.iter().zip(&noise).map(|(&a, &e)| a + e).collect();
            let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            vectors.extend(row.iter().map(|&v| v / norm));
        }
    }
    vectors
}

fn ids(range: std::ops::Range<u64>) -> Vec<u64> {
    range.collect()
}

/// A clustered corpus with genuine boundary mass: `per_cluster` points
/// around each anchor plus `per_bridge` points interpolated between each
/// adjacent anchor pair at `t in [0.4, 0.6]` (in-plane, near-equidistant
/// from both anchors). Bridge points are what closure assignment exists
/// for: SPANN's RNG rule deliberately does NOT replicate points whose
/// centroid distances are near-tied merely because both centroids are far
/// away (the orthogonal-noise case `mixture_vectors` produces) — those are
/// the query-side `probe_epsilon`'s job.
///
/// Returns `(corpus, bridge_queries)`: queries are fresh draws from the
/// same bridge regions, so their true neighbors straddle partitions.
fn bridged_corpus(
    n_clusters: usize,
    per_cluster: usize,
    per_bridge: usize,
    n_queries: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut draw = |scale: f32| -> Vec<f32> {
        (0..DIM)
            .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng) * scale)
            .collect()
    };
    let anchors: Vec<Vec<f32>> = (0..n_clusters).map(|_| draw(1.0)).collect();
    let normalize_into = |row: Vec<f32>, out: &mut Vec<f32>| {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        out.extend(row.iter().map(|&v| v / norm));
    };

    let mut corpus = Vec::new();
    for anchor in &anchors {
        for _ in 0..per_cluster {
            let noise = draw(0.25);
            let row: Vec<f32> = anchor.iter().zip(&noise).map(|(&a, &e)| a + e).collect();
            normalize_into(row, &mut corpus);
        }
    }
    let mut bridge_point = |pair: usize, t: f32, noise_scale: f32| -> Vec<f32> {
        let a = &anchors[pair];
        let b = &anchors[(pair + 1) % n_clusters];
        let noise = draw(noise_scale);
        a.iter()
            .zip(b)
            .zip(&noise)
            .map(|((&va, &vb), &e)| (1.0 - t) * va + t * vb + e)
            .collect()
    };
    let mut t_steps = {
        let mut step = 0usize;
        move || {
            step += 1;
            0.4 + 0.2 * ((step * 7919) % 1000) as f32 / 1000.0
        }
    };
    for pair in 0..n_clusters {
        for _ in 0..per_bridge {
            let t = t_steps();
            let row = bridge_point(pair, t, 0.05);
            normalize_into(row, &mut corpus);
        }
    }

    let mut queries = Vec::new();
    for qi in 0..n_queries {
        let t = t_steps();
        let row = bridge_point(qi % n_clusters, t, 0.05);
        normalize_into(row, &mut queries);
    }
    (corpus, queries)
}

/// Exact f32 inner-product top-k over the raw vectors.
fn exact_top_k(vectors: &[f32], all_ids: &[u64], queries: &[f32], k: usize) -> Vec<u64> {
    let n = all_ids.len();
    let nq = queries.len() / DIM;
    let mut out = Vec::with_capacity(nq * k);
    for qi in 0..nq {
        let query = &queries[qi * DIM..(qi + 1) * DIM];
        let mut scored: Vec<(f32, u64)> = (0..n)
            .map(|i| {
                let row = &vectors[i * DIM..(i + 1) * DIM];
                let score: f32 = query.iter().zip(row).map(|(&q, &v)| q * v).sum();
                (score, all_ids[i])
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        out.extend(scored.iter().take(k).map(|&(_, id)| id));
    }
    out
}

fn recall_against(oracle_ids: &[u64], candidate_ids: &[u64], nq: usize, k: usize) -> f64 {
    let mut hits = 0usize;
    for qi in 0..nq {
        let oracle: HashSet<u64> =
            oracle_ids[qi * k..(qi + 1) * k].iter().copied().collect();
        hits += candidate_ids[qi * k..(qi + 1) * k]
            .iter()
            .filter(|id| oracle.contains(id))
            .count();
    }
    hits as f64 / (nq * k) as f64
}

// ---------------------------------------------------------------------------
// Adaptive probing
// ---------------------------------------------------------------------------

#[test]
fn huge_probe_epsilon_equals_full_probe_and_cap_is_respected() {
    let part_path = temp_path("eps-part.tvdm");
    let vectors = mixture_vectors(8, 200, 21);
    let all_ids = ids(0..1600);
    let queries = mixture_vectors(8, 2, 22);
    let k = 10;

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&vectors, &all_ids).unwrap();
    partitioned.write(&part_path).unwrap();
    let nlist = partitioned.nlist();
    assert!(nlist >= 2);

    // An epsilon admitting every partition must reproduce the full probe
    // when the cap allows it.
    let (full_scores, full_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(nlist));
    let (eps_scores, eps_ids) = partitioned.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(nlist),
            probe_epsilon: Some(1e6),
            ..SearchOptions::default()
        },
    );
    assert_eq!(eps_ids, full_ids);
    assert_eq!(eps_scores, full_scores);

    // Without an explicit nprobe the epsilon rule is capped at
    // max(4, nlist / 2) — the guard that keeps out-of-distribution queries
    // (near-tied distances to every centroid) from degenerating into full
    // scans. A huge epsilon must therefore reproduce the capped probe, not
    // the full one.
    let default_cap = (nlist / 2).max(4).min(nlist);
    let (capped_default_scores, capped_default_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(default_cap));
    let (eps_default_scores, eps_default_ids) = partitioned.search_with_options(
        &queries,
        k,
        SearchOptions {
            probe_epsilon: Some(1e6),
            ..SearchOptions::default()
        },
    );
    assert_eq!(eps_default_ids, capped_default_ids);
    assert_eq!(eps_default_scores, capped_default_scores);

    // With an nprobe cap, the same epsilon must reproduce the capped probe.
    let (capped_scores, capped_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(2));
    let (eps_capped_scores, eps_capped_ids) = partitioned.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(2),
            probe_epsilon: Some(1e6),
            ..SearchOptions::default()
        },
    );
    assert_eq!(eps_capped_ids, capped_ids);
    assert_eq!(eps_capped_scores, capped_scores);

    std::fs::remove_file(&part_path).ok();
}

#[test]
fn probe_epsilon_beats_fixed_nprobe_at_matched_average_cost() {
    let flat_path = temp_path("eps-recall-flat.tvdm");
    let part_path = temp_path("eps-recall-part.tvdm");
    let vectors = mixture_vectors(16, 200, 23); // 3200 vectors
    let all_ids = ids(0..3200);
    let nq = 64;
    let queries = mixture_vectors(16, 4, 24);
    let k = 10;

    let mut flat = DiskIndex::new(Some(DIM), BITS).unwrap();
    flat.add_with_ids(&vectors, &all_ids).unwrap();
    flat.write(&flat_path).unwrap();
    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&vectors, &all_ids).unwrap();
    partitioned.write(&part_path).unwrap();

    let (_, oracle_ids) = flat.search(&queries, k);
    let (_, fixed_ids) = partitioned.search_with_nprobe(&queries, k, Some(2));
    let fixed_recall = recall_against(&oracle_ids, &fixed_ids, nq, k);

    // Sweep epsilon to roughly match nprobe=2's average probe budget, then
    // compare recall: spending the same scans adaptively (more partitions
    // for boundary queries, fewer for confident ones) must not lose.
    let (_, adaptive_ids) = partitioned.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(4),
            probe_epsilon: Some(0.15),
            ..SearchOptions::default()
        },
    );
    let adaptive_recall = recall_against(&oracle_ids, &adaptive_ids, nq, k);
    println!(
        "fixed nprobe=2 recall {fixed_recall:.3}, adaptive eps=0.15 cap=4 \
         recall {adaptive_recall:.3} (nlist={})",
        partitioned.nlist(),
    );
    assert!(
        adaptive_recall >= fixed_recall,
        "adaptive probing ({adaptive_recall}) lost to fixed nprobe=2 ({fixed_recall})",
    );

    std::fs::remove_file(&flat_path).ok();
    std::fs::remove_file(&part_path).ok();
}

// ---------------------------------------------------------------------------
// Exact rescoring (store_vectors)
// ---------------------------------------------------------------------------

#[test]
fn full_depth_rescore_returns_exact_float_ranking() {
    let path = temp_path("rescore-exact.tvdm");
    let vectors = mixture_vectors(4, 125, 31); // 500 vectors
    let all_ids = ids(0..500);
    let queries = mixture_vectors(4, 4, 32);
    let nq = 16;
    let k = 10;
    let oracle = exact_top_k(&vectors, &all_ids, &queries, k);

    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &all_ids).unwrap();

    // Rescoring at full depth must reproduce the exact f32 ranking — first
    // from the delta alone, then from the compacted base, then mixed.
    let full_depth = SearchOptions {
        rescore_k: Some(500),
        ..SearchOptions::default()
    };
    let (scores, result_ids) = index.search_with_options(&queries, k, full_depth);
    assert_eq!(result_ids, oracle, "delta-only rescore diverges from exact");

    index.write(&path).unwrap();
    let (scores_base, ids_base) = index.search_with_options(&queries, k, full_depth);
    assert_eq!(ids_base, oracle, "base rescore diverges from exact");
    assert_eq!(scores_base, scores, "exact scores must not depend on residence");

    // Returned scores are the exact inner products.
    for qi in 0..nq {
        let query = &queries[qi * DIM..(qi + 1) * DIM];
        for j in 0..k {
            let id = ids_base[qi * k + j] as usize;
            let row = &vectors[id * DIM..(id + 1) * DIM];
            let exact: f32 = query.iter().zip(row).map(|(&q, &v)| q * v).sum();
            assert_eq!(scores_base[qi * k + j], exact);
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn default_rescore_improves_recall_vs_quantized_only() {
    let path = temp_path("rescore-recall.tvdm");
    let vectors = mixture_vectors(8, 250, 33); // 2000 vectors
    let all_ids = ids(0..2000);
    let queries = mixture_vectors(8, 8, 34);
    let nq = 64;
    let k = 10;
    let oracle = exact_top_k(&vectors, &all_ids, &queries, k);

    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &all_ids).unwrap();
    index.write(&path).unwrap();

    let (_, quantized_ids) = index.search_with_options(
        &queries,
        k,
        SearchOptions {
            rescore_k: Some(0), // off
            ..SearchOptions::default()
        },
    );
    let (_, rescored_ids) = index.search(&queries, k); // default: rescore at 4k
    let quantized_recall = recall_against(&oracle, &quantized_ids, nq, k);
    let rescored_recall = recall_against(&oracle, &rescored_ids, nq, k);
    println!("quantized {quantized_recall:.3} -> rescored {rescored_recall:.3}");
    assert!(
        rescored_recall > quantized_recall,
        "rescoring ({rescored_recall}) did not improve on quantized ({quantized_recall})",
    );
    assert!(
        rescored_recall >= 0.97,
        "default rescore recall {rescored_recall} below 0.97",
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn get_vector_roundtrips_and_respects_liveness() {
    let path = temp_path("getvec.tvdm");
    let vectors = mixture_vectors(2, 100, 35); // 200 vectors
    let all_ids = ids(0..200);

    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &all_ids).unwrap();

    // From the delta (exact originals)...
    assert_eq!(index.get_vector(7).unwrap(), vectors[7 * DIM..8 * DIM]);
    assert_eq!(index.get_vector(999), None);

    // ...and from the mmap-backed base after compaction.
    index.write(&path).unwrap();
    for &id in &[0u64, 7, 199] {
        assert_eq!(
            index.get_vector(id).unwrap(),
            vectors[id as usize * DIM..(id as usize + 1) * DIM],
            "base vector for id {id} does not round-trip",
        );
    }

    // Tombstoned ids stop resolving immediately, and a re-add shadows the
    // hidden base copy with the new original.
    assert!(index.remove(7));
    assert_eq!(index.get_vector(7), None);
    let replacement = vec![0.125f32; DIM];
    index.add_with_ids(&replacement, &[7]).unwrap();
    assert_eq!(index.get_vector(7).unwrap(), replacement);

    std::fs::remove_file(&path).ok();
}

#[test]
fn store_vectors_persists_across_reopen() {
    let path = temp_path("storevec-reopen.tvdm");
    let vectors = mixture_vectors(2, 150, 36);
    let all_ids = ids(0..300);
    let queries = mixture_vectors(2, 2, 37);
    let k = 5;

    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &all_ids).unwrap();
    index.write(&path).unwrap();
    let (scores_before, ids_before) = index.search(&queries, k);

    let reopened = DiskIndex::open(&path).unwrap();
    assert!(reopened.stores_vectors());
    assert_eq!(reopened.get_vector(11).unwrap(), vectors[11 * DIM..12 * DIM]);
    let (scores_after, ids_after) = reopened.search(&queries, k);
    assert_eq!(ids_after, ids_before);
    assert_eq!(scores_after, scores_before);

    std::fs::remove_file(&path).ok();
}

#[test]
#[should_panic(expected = "store_vectors must be set while the index is empty")]
fn store_vectors_cannot_be_enabled_after_adds() {
    let vectors = mixture_vectors(1, 10, 38);
    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &ids(0..10)).unwrap();
    index.set_store_vectors(true);
}

#[test]
#[should_panic(expected = "rescore_k requires an index built with store_vectors")]
fn explicit_rescore_without_stored_vectors_panics() {
    let vectors = mixture_vectors(1, 10, 39);
    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &ids(0..10)).unwrap();
    index.search_with_options(
        &vectors[0..DIM],
        5,
        SearchOptions {
            rescore_k: Some(8),
            ..SearchOptions::default()
        },
    );
}

// ---------------------------------------------------------------------------
// Boundary multi-assignment (closure assignment)
// ---------------------------------------------------------------------------

/// Build (flat oracle, plain partitioned, replicated partitioned) over the
/// identical single-batch history.
fn build_replication_trio(
    vectors: &[f32],
    all_ids: &[u64],
    epsilon: f32,
    tag: &str,
) -> (DiskIndex, DiskIndex, DiskIndex, Vec<PathBuf>) {
    let paths = vec![
        temp_path(&format!("{tag}-flat.tvdm")),
        temp_path(&format!("{tag}-plain.tvdm")),
        temp_path(&format!("{tag}-repl.tvdm")),
    ];
    let mut flat = DiskIndex::new(Some(DIM), BITS).unwrap();
    flat.add_with_ids(vectors, all_ids).unwrap();
    flat.write(&paths[0]).unwrap();

    let mut plain = DiskIndex::new(Some(DIM), BITS).unwrap();
    plain.set_partitioning(Some(TARGET));
    plain.add_with_ids(vectors, all_ids).unwrap();
    plain.write(&paths[1]).unwrap();

    let mut replicated = DiskIndex::new(Some(DIM), BITS).unwrap();
    replicated.set_partitioning(Some(TARGET));
    replicated.set_replication(Some(epsilon));
    replicated.add_with_ids(vectors, all_ids).unwrap();
    replicated.write(&paths[2]).unwrap();

    (flat, plain, replicated, paths)
}

#[test]
fn replication_creates_copies_without_changing_len_or_membership() {
    let (vectors, _) = bridged_corpus(8, 200, 50, 0, 41); // 2000 vectors
    let all_ids = ids(0..2000);
    let (_, _, replicated, paths) =
        build_replication_trio(&vectors, &all_ids, 1.0, "membership");

    assert_eq!(replicated.len(), 2000);
    assert!(
        replicated.base_replica_count() > 0,
        "epsilon 1.0 on bridged data produced no replicas",
    );
    assert_eq!(
        replicated.base_len(),
        2000 + replicated.base_replica_count(),
    );
    for &id in &[0u64, 799, 1999] {
        assert!(replicated.contains(id));
    }
    assert!(!replicated.contains(2000));

    for path in &paths {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn replication_lifts_small_nprobe_recall() {
    let (vectors, queries) = bridged_corpus(16, 200, 50, 64, 42); // 4000 vectors
    let all_ids = ids(0..4000);
    let nq = 64;
    let k = 10;
    let (flat, plain, replicated, paths) =
        build_replication_trio(&vectors, &all_ids, 1.0, "recall");

    let (_, oracle_ids) = flat.search(&queries, k);
    for nprobe in [1usize, 2] {
        let (_, plain_ids) = plain.search_with_nprobe(&queries, k, Some(nprobe));
        let (_, replicated_ids) =
            replicated.search_with_nprobe(&queries, k, Some(nprobe));
        let plain_recall = recall_against(&oracle_ids, &plain_ids, nq, k);
        let replicated_recall = recall_against(&oracle_ids, &replicated_ids, nq, k);
        println!(
            "nprobe={nprobe}: plain {plain_recall:.3} -> replicated \
             {replicated_recall:.3} (replicas={}, nlist={}/{})",
            replicated.base_replica_count(),
            plain.nlist(),
            replicated.nlist(),
        );
        assert!(
            replicated_recall >= plain_recall,
            "replication lost recall at nprobe={nprobe}: \
             {replicated_recall} < {plain_recall}",
        );
    }
    // At nprobe=1 the lift must be real, not a tie — boundary vectors are
    // findable from the adjacent partition only via their replicas.
    let (_, plain_ids) = plain.search_with_nprobe(&queries, k, Some(1));
    let (_, replicated_ids) = replicated.search_with_nprobe(&queries, k, Some(1));
    assert!(
        recall_against(&oracle_ids, &replicated_ids, nq, k)
            > recall_against(&oracle_ids, &plain_ids, nq, k),
        "replication produced no strict lift at nprobe=1",
    );

    for path in &paths {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn replicated_full_probe_dedups_to_flat_results() {
    let (vectors, queries) = bridged_corpus(8, 200, 50, 16, 44); // 2000 vectors
    let all_ids = ids(0..2000);
    let k = 10;
    let nq = 16;
    let (flat, _, replicated, paths) =
        build_replication_trio(&vectors, &all_ids, 1.0, "dedup");
    assert!(replicated.base_replica_count() > 0);

    let (flat_scores, flat_ids) = flat.search(&queries, k);
    let (repl_scores, repl_ids) =
        replicated.search_with_nprobe(&queries, k, Some(replicated.nlist()));

    // No id may appear twice in a result row, and the (score, id) sets must
    // match the flat scan's (replicas carry identical codes, so identical
    // scores; only tie order may differ).
    for qi in 0..nq {
        let row: Vec<u64> = repl_ids[qi * k..(qi + 1) * k].to_vec();
        let unique: HashSet<u64> = row.iter().copied().collect();
        assert_eq!(unique.len(), k, "duplicate id in result row {qi}: {row:?}");

        let mut flat_pairs: Vec<(u64, u64)> = (0..k)
            .map(|j| (flat_scores[qi * k + j].to_bits() as u64, flat_ids[qi * k + j]))
            .collect();
        let mut repl_pairs: Vec<(u64, u64)> = (0..k)
            .map(|j| (repl_scores[qi * k + j].to_bits() as u64, repl_ids[qi * k + j]))
            .collect();
        flat_pairs.sort_unstable();
        repl_pairs.sort_unstable();
        assert_eq!(repl_pairs, flat_pairs, "full-probe row {qi} diverges from flat");
    }

    for path in &paths {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn replication_survives_churn_and_remove_hides_every_copy() {
    let path = temp_path("repl-churn.tvdm");
    let (first, queries) = bridged_corpus(8, 200, 50, 16, 46); // 2000
    let second = mixture_vectors(4, 100, 47); // 400 fresh
    let k = 10;

    let mut index = DiskIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    index.set_replication(Some(1.0));
    index.add_with_ids(&first, &ids(0..2000)).unwrap();
    index.write(&path).unwrap();
    assert!(index.base_replica_count() > 0);

    // Remove a third of the corpus (pre-compaction: tombstones must hide
    // every replica of a removed id, since filtering is id-based).
    let removed: Vec<u64> = (0..2000u64).filter(|id| id % 3 == 0).collect();
    for &id in &removed {
        assert!(index.remove(id));
        assert!(!index.contains(id));
    }
    let (_, result_ids) =
        index.search_with_nprobe(&queries, 50, Some(index.nlist()));
    for id in result_ids.iter().filter(|&&id| id != 0) {
        assert_ne!(id % 3, 0, "removed id {id} surfaced via a replica");
    }

    // Compact with fresh adds: replicas are recomputed, uniqueness holds.
    index.add_with_ids(&second, &ids(10_000..10_400)).unwrap();
    index.write(&path).unwrap();
    let expected_len = 2000 - removed.len() + 400;
    assert_eq!(index.len(), expected_len);
    assert_eq!(index.tombstone_count(), 0);

    // Reopen and re-verify integrity (the open-time validators check the
    // bitmap/n_unique invariants on the file just written).
    let reopened = DiskIndex::open(&path).unwrap();
    assert_eq!(reopened.len(), expected_len);
    assert_eq!(reopened.replica_epsilon(), Some(1.0));
    let (_, reopened_ids) =
        reopened.search_with_nprobe(&queries, k, Some(reopened.nlist()));
    let (_, current_ids) = index.search_with_nprobe(&queries, k, Some(index.nlist()));
    assert_eq!(reopened_ids, current_ids);

    std::fs::remove_file(&path).ok();
}

#[test]
fn replicated_file_converts_to_id_map_with_primaries_only() {
    let tvim_path = temp_path("repl-convert.tvim");
    let (vectors, queries) = bridged_corpus(8, 200, 50, 16, 49); // 2000 vectors
    let all_ids = ids(0..2000);
    let k = 10;
    let (flat, _, replicated, paths) =
        build_replication_trio(&vectors, &all_ids, 1.0, "convert");
    assert!(replicated.base_replica_count() > 0);

    DiskIndex::convert_to_id_map_file(replicated.path().unwrap(), &tvim_path).unwrap();
    let id_map = IdMapIndex::load(&tvim_path).unwrap();
    assert_eq!(id_map.len(), 2000, "conversion must emit each id exactly once");

    // The primaries carry the same codes/scales as the flat oracle, so the
    // flat scans agree exactly up to tie order.
    let (flat_scores, _) = flat.search(&queries, k);
    let (tvim_scores, _) = id_map.search(&queries, k);
    let sort = |scores: &[f32]| {
        let mut sorted: Vec<u32> = scores.iter().map(|s| s.to_bits()).collect();
        sorted.sort_unstable();
        sorted
    };
    assert_eq!(sort(&tvim_scores), sort(&flat_scores));

    std::fs::remove_file(&tvim_path).ok();
    for path in &paths {
        std::fs::remove_file(path).ok();
    }
}

// ---------------------------------------------------------------------------
// All three levers composed
// ---------------------------------------------------------------------------

#[test]
fn composed_levers_beat_plain_routing_against_exact_oracle() {
    let plain_path = temp_path("composed-plain.tvdm");
    let full_path = temp_path("composed-full.tvdm");
    let (vectors, queries) = bridged_corpus(16, 200, 50, 64, 51); // 4000 vectors
    let all_ids = ids(0..4000);
    let nq = 64;
    let k = 10;
    let oracle = exact_top_k(&vectors, &all_ids, &queries, k);

    let mut plain = DiskIndex::new(Some(DIM), BITS).unwrap();
    plain.set_partitioning(Some(TARGET));
    plain.add_with_ids(&vectors, &all_ids).unwrap();
    plain.write(&plain_path).unwrap();

    let mut tuned = DiskIndex::new(Some(DIM), BITS).unwrap();
    tuned.set_partitioning(Some(TARGET));
    tuned.set_replication(Some(1.0));
    tuned.set_store_vectors(true);
    tuned.add_with_ids(&vectors, &all_ids).unwrap();
    tuned.write(&full_path).unwrap();

    let nprobe = 2;
    let (_, plain_ids) = plain.search_with_nprobe(&queries, k, Some(nprobe));
    let (_, tuned_ids) = tuned.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(nprobe + 2),
            probe_epsilon: Some(0.15),
            rescore_k: None, // default 4k
        },
    );
    let plain_recall = recall_against(&oracle, &plain_ids, nq, k);
    let tuned_recall = recall_against(&oracle, &tuned_ids, nq, k);
    println!(
        "vs exact oracle: plain nprobe={nprobe} {plain_recall:.3} -> composed \
         {tuned_recall:.3} (replicas={})",
        tuned.base_replica_count(),
    );
    assert!(
        tuned_recall > plain_recall,
        "composed levers ({tuned_recall}) did not beat plain routing ({plain_recall})",
    );

    std::fs::remove_file(&plain_path).ok();
    std::fs::remove_file(&full_path).ok();
}
