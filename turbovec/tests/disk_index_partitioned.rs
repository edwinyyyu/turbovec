//! Tests for the partitioned (SPFresh-lite) mode of [`DiskIndex`].
//!
//! The anchor property: probing **all** partitions must return exactly the
//! flat scan's results — partitioning changes data layout, not scoring —
//! so the flat index is an exact oracle. Routing quality (default nprobe)
//! is then asserted as recall against that oracle on clustered data, plus
//! the split/merge lifecycle and persistence of the partitioning config.

use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::DiskIndex;

const DIM: usize = 64;
const BITS: usize = 4;
const TARGET: usize = 256;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-part-{}-{}", nonce, name));
    p
}

/// Unit-normalized gaussian-mixture vectors: `per_cluster` draws around
/// each of `n_clusters` random unit anchors, with small noise — realistic
/// shape for embedding corpora, and clusterable by construction.
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

/// Build (flat, partitioned) DiskIndexes over identical data, both
/// compacted to disk.
fn build_pair(
    vectors: &[f32],
    all_ids: &[u64],
    flat_path: &PathBuf,
    part_path: &PathBuf,
) -> (DiskIndex, DiskIndex) {
    let mut flat = DiskIndex::new(Some(DIM), BITS).unwrap();
    flat.add_with_ids(vectors, all_ids).unwrap();
    flat.write(flat_path).unwrap();

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(vectors, all_ids).unwrap();
    partitioned.write(part_path).unwrap();

    (flat, partitioned)
}

fn recall_against(
    oracle_ids: &[u64],
    candidate_ids: &[u64],
    nq: usize,
    k: usize,
) -> f64 {
    let mut hits = 0usize;
    for qi in 0..nq {
        let oracle: std::collections::HashSet<u64> =
            oracle_ids[qi * k..(qi + 1) * k].iter().copied().collect();
        hits += candidate_ids[qi * k..(qi + 1) * k]
            .iter()
            .filter(|id| oracle.contains(id))
            .count();
    }
    hits as f64 / (nq * k) as f64
}

#[test]
fn probing_all_partitions_equals_flat_scan_exactly() {
    let flat_path = temp_path("full-flat.tvdm");
    let part_path = temp_path("full-part.tvdm");
    let vectors = mixture_vectors(8, 200, 1); // n = 1600 -> several partitions
    let n = 8 * 200;
    let all_ids = ids(0..n as u64);
    let queries = mixture_vectors(8, 2, 2); // 16 queries from the same mixture
    let k = 10;

    let (flat, partitioned) = build_pair(&vectors, &all_ids, &flat_path, &part_path);
    assert!(
        partitioned.nlist() >= 2,
        "expected multiple partitions, got {}",
        partitioned.nlist(),
    );

    let (flat_scores, flat_ids) = flat.search(&queries, k);
    let (part_scores, part_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(partitioned.nlist()));
    assert_eq!(part_ids, flat_ids);
    assert_eq!(part_scores, flat_scores);

    std::fs::remove_file(&flat_path).ok();
    std::fs::remove_file(&part_path).ok();
}

#[test]
fn default_nprobe_recall_is_high_on_clustered_data() {
    let flat_path = temp_path("recall-flat.tvdm");
    let part_path = temp_path("recall-part.tvdm");
    let vectors = mixture_vectors(16, 200, 3); // n = 3200
    let n = 16 * 200;
    let all_ids = ids(0..n as u64);
    let nq = 32;
    let queries = mixture_vectors(16, 2, 4);
    let k = 10;

    let (flat, partitioned) = build_pair(&vectors, &all_ids, &flat_path, &part_path);

    let (_, oracle_ids) = flat.search(&queries, k);
    let (_, routed_ids) = partitioned.search(&queries, k);
    let recall = recall_against(&oracle_ids, &routed_ids, nq, k);
    assert!(
        recall >= 0.9,
        "default-nprobe recall {recall} below 0.9 (nlist={})",
        partitioned.nlist(),
    );

    std::fs::remove_file(&flat_path).ok();
    std::fs::remove_file(&part_path).ok();
}

#[test]
fn reopened_partitioned_index_keeps_config_and_results() {
    let part_path = temp_path("reopen-part.tvdm");
    let vectors = mixture_vectors(8, 150, 5);
    let n = 8 * 150;
    let all_ids = ids(0..n as u64);
    let queries = mixture_vectors(8, 1, 6);
    let k = 10;

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&vectors, &all_ids).unwrap();
    partitioned.write(&part_path).unwrap();
    let (scores_before, ids_before) = partitioned.search(&queries, k);

    let reopened = DiskIndex::open(&part_path).unwrap();
    assert_eq!(reopened.partition_target(), Some(TARGET));
    assert_eq!(reopened.nlist(), partitioned.nlist());
    let (scores_after, ids_after) = reopened.search(&queries, k);
    assert_eq!(ids_after, ids_before);
    assert_eq!(scores_after, scores_before);

    std::fs::remove_file(&part_path).ok();
}

#[test]
fn growth_triggers_splits_and_results_stay_exact_under_full_probe() {
    let part_path = temp_path("split-part.tvdm");
    let flat_path = temp_path("split-flat.tvdm");
    let first = mixture_vectors(4, 200, 7); // 800 vectors
    let second = mixture_vectors(4, 200, 8); // 800 more, fresh clusters
    let first_ids = ids(0..800);
    let second_ids = ids(10_000..10_800);
    let queries = mixture_vectors(4, 2, 9);
    let k = 10;

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&first, &first_ids).unwrap();
    partitioned.write(&part_path).unwrap();
    let nlist_before = partitioned.nlist();

    // Incremental growth: appended vectors land in existing partitions and
    // oversized ones split at the next compaction.
    partitioned.add_with_ids(&second, &second_ids).unwrap();
    partitioned.write(&part_path).unwrap();
    let nlist_after = partitioned.nlist();
    assert!(
        nlist_after > nlist_before,
        "expected splits to raise nlist ({nlist_before} -> {nlist_after})",
    );
    assert_eq!(partitioned.len(), 1600);

    // Same two-batch history through the flat oracle.
    let mut flat = DiskIndex::new(Some(DIM), BITS).unwrap();
    flat.add_with_ids(&first, &first_ids).unwrap();
    flat.write(&flat_path).unwrap();
    flat.add_with_ids(&second, &second_ids).unwrap();
    flat.write(&flat_path).unwrap();

    let (flat_scores, flat_ids) = flat.search(&queries, k);
    let (part_scores, part_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(partitioned.nlist()));
    assert_eq!(part_ids, flat_ids);
    assert_eq!(part_scores, flat_scores);

    std::fs::remove_file(&part_path).ok();
    std::fs::remove_file(&flat_path).ok();
}

#[test]
fn shrinkage_triggers_merges_and_tombstones_stay_hidden() {
    let part_path = temp_path("merge-part.tvdm");
    let vectors = mixture_vectors(8, 200, 10); // 1600
    let n = 1600usize;
    let all_ids = ids(0..n as u64);
    let queries = mixture_vectors(8, 1, 11);

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&vectors, &all_ids).unwrap();
    partitioned.write(&part_path).unwrap();
    let nlist_before = partitioned.nlist();

    // Remove 90% of vectors; partitions shrink below target/4 and merge.
    let removed: Vec<u64> = (0..n as u64).filter(|id| id % 10 != 0).collect();
    for &id in &removed {
        assert!(partitioned.remove(id));
    }
    assert_eq!(partitioned.len(), n / 10);

    // Tombstoned ids never surface, pre-compaction...
    let (_, result_ids) = partitioned.search_with_nprobe(
        &queries,
        20,
        Some(partitioned.nlist()),
    );
    for id in result_ids.iter().filter(|&&id| id != 0) {
        assert_eq!(id % 10, 0, "tombstoned id {id} surfaced");
    }

    // ...and compaction merges undersized partitions away.
    partitioned.write(&part_path).unwrap();
    let nlist_after = partitioned.nlist();
    assert!(
        nlist_after < nlist_before,
        "expected merges to lower nlist ({nlist_before} -> {nlist_after})",
    );
    assert_eq!(partitioned.len(), n / 10);
    assert_eq!(partitioned.tombstone_count(), 0);
    let (_, result_ids) = partitioned.search_with_nprobe(
        &queries,
        20,
        Some(partitioned.nlist()),
    );
    for id in result_ids.iter().filter(|&&id| id != 0) {
        assert_eq!(id % 10, 0, "removed id {id} surfaced after compaction");
    }

    std::fs::remove_file(&part_path).ok();
}

#[test]
fn delta_and_partitioned_base_merge_in_search() {
    let part_path = temp_path("delta-part.tvdm");
    let base_vectors = mixture_vectors(8, 150, 12);
    let base_ids = ids(0..1200);
    let queries = mixture_vectors(8, 1, 13);
    let k = 10;

    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    partitioned.add_with_ids(&base_vectors, &base_ids).unwrap();
    partitioned.write(&part_path).unwrap();

    // A delta vector identical to a query must win rank 1 even though the
    // base is partitioned (the delta is scanned exhaustively).
    let probe_query = &queries[0..DIM];
    partitioned.add_with_ids(probe_query, &[99_999]).unwrap();
    let (_, result_ids) = partitioned.search(&queries[0..DIM], k);
    assert_eq!(result_ids[0], 99_999, "delta self-match must rank first");

    std::fs::remove_file(&part_path).ok();
}

/// Sustained churn with distribution drift: each generation removes old
/// vectors and adds vectors from brand-new clusters, compacting every
/// time. The LIRE maintenance pass (delta assignment + neighbor
/// reassignment + centroid refresh + split/merge) must keep routed recall
/// near the flat oracle's despite centroids having been bootstrapped on a
/// distribution that no longer exists.
#[test]
fn routed_recall_survives_churn_with_drift() {
    let part_path = temp_path("churn-part.tvdm");
    let k = 10;
    let generations = 6;
    let per_generation = 400;

    let flat_path = temp_path("churn-flat.tvdm");
    let mut partitioned = DiskIndex::new(Some(DIM), BITS).unwrap();
    partitioned.set_partitioning(Some(TARGET));
    // The flat oracle follows the IDENTICAL add/remove/write history so its
    // quantized codes (incl. the generation-0 TQ+ calibration) match the
    // churned index's exactly — anything else confounds routing loss with
    // calibration differences.
    let mut flat = DiskIndex::new(Some(DIM), BITS).unwrap();

    // Live id set so the test can sweep removals oldest-first.
    let mut live: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();

    // Generation 0: bootstrap distribution.
    let initial = mixture_vectors(8, 200, 100);
    live.extend(0..1600u64);
    partitioned
        .add_with_ids(&initial, &ids(0..1600))
        .unwrap();
    partitioned.write(&part_path).unwrap();
    flat.add_with_ids(&initial, &ids(0..1600)).unwrap();
    flat.write(&flat_path).unwrap();

    let mut next_id = 1600u64;
    for generation in 0..generations {
        // Remove ~25% of the oldest live ids.
        let to_remove: Vec<u64> = live.iter().copied().take(per_generation).collect();
        for id in to_remove {
            assert!(partitioned.remove(id));
            assert!(flat.remove(id));
            live.remove(&id);
        }
        // Add vectors from clusters never seen at bootstrap.
        let fresh = mixture_vectors(4, per_generation / 4, 200 + generation);
        let fresh_ids: Vec<u64> =
            (next_id..next_id + (fresh.len() / DIM) as u64).collect();
        live.extend(fresh_ids.iter().copied());
        partitioned.add_with_ids(&fresh, &fresh_ids).unwrap();
        flat.add_with_ids(&fresh, &fresh_ids).unwrap();
        next_id += fresh_ids.len() as u64;
        partitioned.write(&part_path).unwrap();
        flat.write(&flat_path).unwrap();
    }
    assert_eq!(partitioned.len(), live.len());
    assert_eq!(flat.len(), live.len());

    // Queries from the drifted (current) distribution: the last two
    // generations' clusters plus the surviving originals.
    let queries: Vec<f32> = [
        mixture_vectors(4, 4, 200 + generations - 1),
        mixture_vectors(4, 4, 200 + generations - 2),
        mixture_vectors(8, 2, 100),
    ]
    .concat();
    let nq = queries.len() / DIM;

    // File-content sanity: probing all partitions must equal the flat scan
    // regardless of how mangled the routing structure is.
    let (full_scores, full_ids) =
        partitioned.search_with_nprobe(&queries, k, Some(partitioned.nlist()));
    let (flat_scores, flat_ids) = flat.search(&queries, k);
    assert_eq!(full_ids, flat_ids, "full-probe diverges from flat after churn");
    assert_eq!(full_scores, flat_scores);

    let (_, oracle_ids) = flat.search(&queries, k);
    let (_, routed_ids) = partitioned.search(&queries, k);
    let churned_recall = recall_against(&oracle_ids, &routed_ids, nq, k);

    // Control: a fresh bootstrap over the same final corpus AND the same
    // quantized codes (re-cluster the flat oracle's file) — the best this
    // partition count can do. The churned index must stay close to it; the
    // gap is what the LIRE maintenance is responsible for closing.
    let control_path = temp_path("churn-control.tvdm");
    std::fs::copy(&flat_path, &control_path).unwrap();
    let mut control = DiskIndex::open(&control_path).unwrap();
    control.set_partitioning(Some(TARGET));
    control.write(&control_path).unwrap();
    let (_, control_ids) = control.search(&queries, k);
    let control_recall = recall_against(&oracle_ids, &control_ids, nq, k);

    println!(
        "churned recall {churned_recall:.3} (nlist={}) vs fresh-bootstrap control \
         {control_recall:.3} (nlist={})",
        partitioned.nlist(),
        control.nlist(),
    );
    assert!(
        churned_recall >= control_recall - 0.05,
        "churned recall {churned_recall} lags fresh-bootstrap control {control_recall} \
         by more than 0.05 after {generations} drift generations \
         (nlist={} vs {}, len={})",
        partitioned.nlist(),
        control.nlist(),
        partitioned.len(),
    );

    std::fs::remove_file(&part_path).ok();
    std::fs::remove_file(&flat_path).ok();
    std::fs::remove_file(&control_path).ok();
}
