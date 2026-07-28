//! Tests for [`FreshIndex`] — the incrementally-updatable (SPFresh-style)
//! disk index.
//!
//! Anchor discipline: a flat [`DiskIndex`] replaying the IDENTICAL
//! add/remove/save history is the exact oracle (same first-batch TQ+
//! calibration, same codes); FreshIndex full-probe results must match it
//! up to tie order. Recall under routing is then asserted against that
//! oracle, and the storage model's own guarantees (WAL durability, crash
//! cleanup, file stability for untouched partitions) are tested directly.

use std::collections::HashSet;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::{DiskIndex, FreshIndex, IdMapIndex, SearchOptions};

const DIM: usize = 64;
const BITS: usize = 4;
const TARGET: usize = 256;

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-fresh-{}-{}", nonce, name));
    p
}

fn cleanup(path: &PathBuf) {
    std::fs::remove_dir_all(path).ok();
    std::fs::remove_file(path).ok();
}

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

/// Per-query result rows as multisets of (score bits, id) — equality up to
/// tie order.
fn result_multisets(scores: &[f32], result_ids: &[u64], nq: usize) -> Vec<Vec<(u32, u64)>> {
    let k = scores.len() / nq.max(1);
    (0..nq)
        .map(|qi| {
            let mut row: Vec<(u32, u64)> = (0..k)
                .map(|j| (scores[qi * k + j].to_bits(), result_ids[qi * k + j]))
                .collect();
            row.sort_unstable();
            row
        })
        .collect()
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

fn full_probe(index: &FreshIndex, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
    index.search_with_options(
        queries,
        k,
        SearchOptions {
            nprobe: Some(index.nlist().max(1)),
            ..SearchOptions::default()
        },
    )
}

// ---------------------------------------------------------------------------
// Core lifecycle
// ---------------------------------------------------------------------------

#[test]
fn memtable_only_lifecycle_before_any_save() {
    let vectors = mixture_vectors(4, 50, 1);
    let all_ids = ids(0..200);
    let queries = &vectors[0..DIM];

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &all_ids).unwrap();
    assert_eq!(index.len(), 200);
    assert!(index.contains(7));
    let (_, result_ids) = index.search(queries, 1);
    assert_eq!(result_ids[0], 0, "self-match must rank first");

    assert!(index.remove(0));
    assert!(!index.contains(0));
    assert_eq!(index.len(), 199);
    let (_, result_ids) = index.search(queries, 1);
    assert_ne!(result_ids[0], 0);
}

#[test]
fn empty_index_saves_and_reopens() {
    let dir = temp_dir("empty");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.save(&dir).unwrap();
    let reopened = FreshIndex::open(&dir).unwrap();
    assert_eq!(reopened.len(), 0);
    assert!(reopened.search(&vec![0.5; DIM], 5).1.is_empty());
    cleanup(&dir);
}

#[test]
fn flat_results_match_disk_index_exactly() {
    let dir = temp_dir("flat-parity");
    let disk_path = temp_dir("flat-parity-oracle.tvdm");
    let first = mixture_vectors(4, 100, 2); // 400
    let second = mixture_vectors(4, 50, 3); // 200 more
    let queries = mixture_vectors(4, 4, 4);
    let nq = 16;
    let k = 10;

    // Identical histories: add, save, add, save.
    let mut fresh = FreshIndex::new(Some(DIM), BITS).unwrap();
    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    fresh.add_with_ids(&first, &ids(0..400)).unwrap();
    disk.add_with_ids(&first, &ids(0..400)).unwrap();
    fresh.save(&dir).unwrap();
    disk.write(&disk_path).unwrap();
    fresh.add_with_ids(&second, &ids(1000..1200)).unwrap();
    disk.add_with_ids(&second, &ids(1000..1200)).unwrap();
    fresh.save(&dir).unwrap();
    disk.write(&disk_path).unwrap();

    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    let (disk_scores, disk_ids) = disk.search(&queries, k);
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&disk_scores, &disk_ids, nq),
        "fresh full scan diverges from the flat DiskIndex oracle",
    );

    // Mixed memtable + base must also match (same pending delta on both).
    let third = mixture_vectors(4, 25, 5);
    fresh.add_with_ids(&third, &ids(2000..2100)).unwrap();
    disk.add_with_ids(&third, &ids(2000..2100)).unwrap();
    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    let (disk_scores, disk_ids) = disk.search(&queries, k);
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&disk_scores, &disk_ids, nq),
    );

    cleanup(&dir);
    cleanup(&disk_path);
}

#[test]
fn reopen_serves_identical_results_and_config() {
    let dir = temp_dir("reopen");
    let vectors = mixture_vectors(4, 100, 6);
    let queries = mixture_vectors(4, 2, 7);
    let k = 10;

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &ids(0..400)).unwrap();
    index.save(&dir).unwrap();
    let (scores_before, ids_before) = index.search(&queries, k);

    let reopened = FreshIndex::open(&dir).unwrap();
    assert_eq!(reopened.len(), 400);
    assert_eq!(reopened.partition_target(), Some(TARGET));
    assert!(reopened.stores_vectors());
    assert_eq!(reopened.nlist(), index.nlist());
    let (scores_after, ids_after) = reopened.search(&queries, k);
    assert_eq!(ids_after, ids_before);
    assert_eq!(scores_after, scores_before);

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Durability / crash consistency
// ---------------------------------------------------------------------------

#[test]
fn wal_recovers_unsaved_mutations() {
    let dir = temp_dir("wal");
    let vectors = mixture_vectors(4, 100, 8);
    let extra = mixture_vectors(1, 10, 9);
    let queries = &extra[0..DIM];

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &ids(0..400)).unwrap();
    index.save(&dir).unwrap();

    // Mutations after the save, never flushed: an add batch and a remove.
    index.add_with_ids(&extra, &ids(5000..5010)).unwrap();
    assert!(index.remove(3));
    let (_, expected_ids) = index.search(queries, 5);
    let expected_len = index.len();
    drop(index); // "crash": memtable and dead-marks lost, WAL survives

    let recovered = FreshIndex::open(&dir).unwrap();
    assert_eq!(recovered.len(), expected_len);
    assert!(recovered.contains(5000));
    assert!(!recovered.contains(3));
    let (_, recovered_ids) = recovered.search(queries, 5);
    assert_eq!(recovered_ids, expected_ids);
    // store_vectors originals also recovered from the log.
    assert_eq!(
        recovered.get_vector(5000).unwrap(),
        extra[0..DIM].to_vec(),
    );

    cleanup(&dir);
}

#[test]
fn corrupt_wal_tail_stops_replay_cleanly() {
    let dir = temp_dir("wal-corrupt");
    let vectors = mixture_vectors(4, 100, 10);
    let extra = mixture_vectors(1, 4, 11);

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &ids(0..400)).unwrap();
    index.save(&dir).unwrap();
    index.add_with_ids(&extra, &ids(900..904)).unwrap();
    drop(index);

    // Corrupt the last record's payload byte; replay must keep the intact
    // prefix and stop at the bad record instead of erroring.
    let wal_path = dir.join("wal");
    let mut bytes = std::fs::read(&wal_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&wal_path, &bytes).unwrap();

    let recovered = FreshIndex::open(&dir).unwrap();
    assert!(recovered.contains(900));
    assert!(recovered.contains(902));
    assert!(!recovered.contains(903), "corrupted record must not replay");
    assert_eq!(recovered.len(), 403);

    cleanup(&dir);
}

#[test]
fn crashed_append_tail_is_truncated_on_open() {
    let dir = temp_dir("crash-tail");
    let vectors = mixture_vectors(4, 100, 12);
    let queries = mixture_vectors(4, 2, 13);
    let k = 10;

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &ids(0..400)).unwrap();
    index.save(&dir).unwrap();
    let (scores_before, ids_before) = index.search(&queries, k);
    drop(index);

    // Simulate a crash mid-append: garbage past the manifest-recorded end
    // of a segment file.
    let segment = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().starts_with("segment-"))
        .expect("at least one segment");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(segment.path())
        .unwrap();
    use std::io::Write;
    file.write_all(&vec![0xAB; 4096]).unwrap();
    drop(file);

    let recovered = FreshIndex::open(&dir).unwrap();
    let (scores_after, ids_after) = recovered.search(&queries, k);
    assert_eq!(ids_after, ids_before);
    assert_eq!(scores_after, scores_before);

    // And the index keeps working: a new add + save lands cleanly.
    let mut recovered = recovered;
    let extra = mixture_vectors(1, 4, 14);
    recovered.add_with_ids(&extra, &ids(7000..7004)).unwrap();
    recovered.save(&dir).unwrap();
    assert!(FreshIndex::open(&dir).unwrap().contains(7000));

    cleanup(&dir);
}

#[test]
fn stale_wal_from_before_a_flush_is_discarded() {
    let dir = temp_dir("stale-wal");
    let vectors = mixture_vectors(4, 100, 15);
    let extra = mixture_vectors(1, 4, 16);

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.add_with_ids(&vectors, &ids(0..400)).unwrap();
    index.save(&dir).unwrap();
    index.add_with_ids(&extra, &ids(900..904)).unwrap();
    // Snapshot the WAL holding the 4 unsaved adds, then flush them.
    let stale_wal = std::fs::read(dir.join("wal")).unwrap();
    index.save(&dir).unwrap();
    assert_eq!(index.len(), 404);
    drop(index);

    // A crash that resurrects the pre-flush WAL must not replay it (its
    // records are already in the manifest state).
    std::fs::write(dir.join("wal"), &stale_wal).unwrap();
    let recovered = FreshIndex::open(&dir).unwrap();
    assert_eq!(recovered.len(), 404, "stale WAL replayed double");

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Partitioning lifecycle on the incremental substrate
// ---------------------------------------------------------------------------

#[test]
fn incremental_growth_clusters_and_full_probe_stays_exact() {
    let dir = temp_dir("grow");
    let disk_path = temp_dir("grow-oracle.tvdm");
    let queries = mixture_vectors(6, 3, 20);
    let nq = 18;
    let k = 10;

    let mut fresh = FreshIndex::new(Some(DIM), BITS).unwrap();
    fresh.set_partitioning(Some(TARGET));
    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();

    // Grow through the clustering threshold in several flushes.
    let mut next_id = 0u64;
    for generation in 0..4 {
        let batch = mixture_vectors(6, 50, 30 + generation); // 300 per flush
        let batch_ids = ids(next_id..next_id + 300);
        next_id += 300;
        fresh.add_with_ids(&batch, &batch_ids).unwrap();
        disk.add_with_ids(&batch, &batch_ids).unwrap();
        fresh.save(&dir).unwrap();
        disk.write(&disk_path).unwrap();
    }
    assert_eq!(fresh.len(), 1200);
    assert!(
        fresh.nlist() > 1,
        "expected clustering to engage (nlist = {})",
        fresh.nlist(),
    );

    // Full probe == flat oracle, exactly (same codes, same calibration).
    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    let (disk_scores, disk_ids) = disk.search_with_nprobe(&queries, k, Some(disk.nlist()));
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&disk_scores, &disk_ids, nq),
    );

    // Routed recall against the oracle.
    let (_, routed_ids) = fresh.search(&queries, k);
    let recall = recall_against(&disk_ids, &routed_ids, nq, k);
    assert!(recall >= 0.9, "routed recall {recall} below 0.9");

    cleanup(&dir);
    cleanup(&disk_path);
}

#[test]
fn removals_dissolve_partitions_and_gc_reclaims_dead_rows() {
    let dir = temp_dir("shrink");
    let vectors = mixture_vectors(8, 200, 40); // 1600

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    index.add_with_ids(&vectors, &ids(0..1600)).unwrap();
    index.save(&dir).unwrap();
    let nlist_before = index.nlist();
    assert!(nlist_before > 1);

    for id in (0..1600u64).filter(|id| id % 10 != 0) {
        assert!(index.remove(id));
    }
    assert_eq!(index.len(), 160);
    // Dead-marked but not yet reclaimed; queries must not surface them.
    let queries = mixture_vectors(8, 1, 41);
    let (_, result_ids) = full_probe(&index, &queries, 20);
    for id in result_ids.iter().filter(|&&id| id != 0) {
        assert_eq!(id % 10, 0, "removed id {id} surfaced");
    }

    index.save(&dir).unwrap();
    assert!(
        index.nlist() < nlist_before,
        "expected dissolves to lower nlist ({nlist_before} -> {})",
        index.nlist(),
    );
    assert!(
        index.dead_count() == 0,
        "garbage collection left {} dead rows after a 90% purge",
        index.dead_count(),
    );
    assert_eq!(index.len(), 160);

    let reopened = FreshIndex::open(&dir).unwrap();
    assert_eq!(reopened.len(), 160);

    cleanup(&dir);
}

/// The cold-cache guarantee: a flush that only touches some partitions
/// must leave every other partition's segment file untouched (same inode,
/// same length — the page cache for those files stays valid).
#[cfg(unix)]
#[test]
fn flush_leaves_untouched_partitions_files_alone() {
    use std::os::unix::fs::MetadataExt;

    let dir = temp_dir("inode");
    let vectors = mixture_vectors(8, 200, 50); // 1600 -> several partitions

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    index.add_with_ids(&vectors, &ids(0..1600)).unwrap();
    index.save(&dir).unwrap();
    assert!(index.nlist() >= 4);

    let snapshot = |dir: &PathBuf| -> std::collections::HashMap<String, (u64, u64)> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("segment-"))
            .map(|e| {
                let meta = e.metadata().unwrap();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    (meta.ino(), meta.len()),
                )
            })
            .collect()
    };
    let before = snapshot(&dir);

    // A tiny add lands in ONE partition (one cluster's direction).
    let one_cluster = mixture_vectors(1, 3, 50); // same seed -> same first anchor
    index
        .add_with_ids(&one_cluster[0..3 * DIM], &ids(9000..9003))
        .unwrap();
    index.save(&dir).unwrap();

    let after = snapshot(&dir);
    let unchanged = before
        .iter()
        .filter(|(name, state)| after.get(*name) == Some(state))
        .count();
    assert!(
        unchanged >= before.len() - 2,
        "flush of 3 vectors rewrote {} of {} segment files",
        before.len() - unchanged,
        before.len(),
    );

    cleanup(&dir);
}

/// Sustained churn with distribution drift, against a flat oracle with the
/// identical history (the same protocol as the DiskIndex churn test).
#[test]
fn routed_recall_survives_churn_with_drift() {
    let dir = temp_dir("churn");
    let disk_path = temp_dir("churn-oracle.tvdm");
    let k = 10;

    let mut fresh = FreshIndex::new(Some(DIM), BITS).unwrap();
    fresh.set_partitioning(Some(TARGET));
    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    let mut live: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();

    let initial = mixture_vectors(8, 200, 100);
    live.extend(0..1600u64);
    fresh.add_with_ids(&initial, &ids(0..1600)).unwrap();
    disk.add_with_ids(&initial, &ids(0..1600)).unwrap();
    fresh.save(&dir).unwrap();
    disk.write(&disk_path).unwrap();

    let generations = 6;
    let per_generation = 400;
    let mut next_id = 1600u64;
    for generation in 0..generations {
        let to_remove: Vec<u64> = live.iter().copied().take(per_generation).collect();
        for id in to_remove {
            assert!(fresh.remove(id));
            assert!(disk.remove(id));
            live.remove(&id);
        }
        let added = mixture_vectors(4, per_generation / 4, 200 + generation);
        let added_ids: Vec<u64> = (next_id..next_id + (added.len() / DIM) as u64).collect();
        live.extend(added_ids.iter().copied());
        fresh.add_with_ids(&added, &added_ids).unwrap();
        disk.add_with_ids(&added, &added_ids).unwrap();
        next_id += added_ids.len() as u64;
        fresh.save(&dir).unwrap();
        disk.write(&disk_path).unwrap();
    }
    assert_eq!(fresh.len(), live.len());
    assert_eq!(disk.len(), live.len());

    let queries: Vec<f32> = [
        mixture_vectors(4, 4, 200 + generations - 1),
        mixture_vectors(4, 4, 200 + generations - 2),
        mixture_vectors(8, 2, 100),
    ]
    .concat();
    let nq = queries.len() / DIM;

    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    let (disk_scores, disk_ids) = disk.search(&queries, k);
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&disk_scores, &disk_ids, nq),
        "full probe diverges from the flat oracle after churn",
    );

    let (_, routed_ids) = fresh.search(&queries, k);
    let recall = recall_against(&disk_ids, &routed_ids, nq, k);
    println!(
        "fresh churned recall {recall:.3} (nlist={}, dead={}, chunks={}, runs={})",
        fresh.nlist(),
        fresh.dead_count(),
        fresh.chunk_count(),
        fresh.run_count(),
    );
    assert!(
        recall >= 0.9,
        "churned routed recall {recall} below 0.9 (nlist={})",
        fresh.nlist(),
    );

    // The directory keeps a bounded shape under churn.
    assert!(fresh.run_count() <= MAX_RUNS_BOUND);
    cleanup(&dir);
    cleanup(&disk_path);
}

/// Mirror of the crate's MAX_RUNS + 1 (a freshly-written run may push the
/// count one past the merge threshold within a single flush).
const MAX_RUNS_BOUND: usize = 8;

// ---------------------------------------------------------------------------
// Levers
// ---------------------------------------------------------------------------

#[test]
fn rescore_returns_exact_ranking_and_get_vector_roundtrips() {
    let dir = temp_dir("rescore");
    let vectors = mixture_vectors(4, 125, 60); // 500
    let all_ids = ids(0..500);
    let queries = mixture_vectors(4, 4, 61);
    let nq = 16;
    let k = 10;

    // Exact float oracle.
    let mut exact: Vec<Vec<u64>> = Vec::new();
    for qi in 0..nq {
        let query = &queries[qi * DIM..(qi + 1) * DIM];
        let mut scored: Vec<(f32, u64)> = (0..500)
            .map(|i| {
                let row = &vectors[i * DIM..(i + 1) * DIM];
                (query.iter().zip(row).map(|(&q, &v)| q * v).sum(), i as u64)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        exact.push(scored.iter().take(k).map(|&(_, id)| id).collect());
    }

    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_store_vectors(true);
    index.add_with_ids(&vectors, &all_ids).unwrap();
    index.save(&dir).unwrap();

    let (_, result_ids) = index.search_with_options(
        &queries,
        k,
        SearchOptions {
            rescore_k: Some(500),
            ..SearchOptions::default()
        },
    );
    for qi in 0..nq {
        assert_eq!(
            result_ids[qi * k..(qi + 1) * k].to_vec(),
            exact[qi],
            "full-depth rescore diverges from exact float ranking (query {qi})",
        );
    }

    assert_eq!(
        index.get_vector(7).unwrap(),
        vectors[7 * DIM..8 * DIM].to_vec(),
    );
    assert!(index.remove(7));
    assert_eq!(index.get_vector(7), None);

    cleanup(&dir);
}

#[test]
fn replication_on_boundary_corpus_lifts_small_nprobe_recall() {
    // Bridged corpus: in-plane interpolations between anchors are genuine
    // closure-assignment targets (see the DiskIndex lever tests).
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(70);
    let mut draw = |scale: f32| -> Vec<f32> {
        (0..DIM)
            .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng) * scale)
            .collect()
    };
    let n_clusters = 8;
    let anchors: Vec<Vec<f32>> = (0..n_clusters).map(|_| draw(1.0)).collect();
    let mut corpus: Vec<f32> = Vec::new();
    let normalize_into = |row: Vec<f32>, out: &mut Vec<f32>| {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        out.extend(row.iter().map(|&v| v / norm));
    };
    for anchor in &anchors {
        for _ in 0..200 {
            let noise = draw(0.25);
            let row: Vec<f32> = anchor.iter().zip(&noise).map(|(&a, &e)| a + e).collect();
            normalize_into(row, &mut corpus);
        }
    }
    let mut step = 0usize;
    let mut bridge = |pair: usize, noise: &[f32]| -> Vec<f32> {
        step += 1;
        let t = 0.4 + 0.2 * ((step * 7919) % 1000) as f32 / 1000.0;
        let a = &anchors[pair];
        let b = &anchors[(pair + 1) % n_clusters];
        a.iter()
            .zip(b)
            .zip(noise)
            .map(|((&va, &vb), &e)| (1.0 - t) * va + t * vb + e)
            .collect()
    };
    for pair in 0..n_clusters {
        for _ in 0..50 {
            let noise = draw(0.05);
            let row = bridge(pair, &noise);
            normalize_into(row, &mut corpus);
        }
    }
    let n = corpus.len() / DIM; // 2000
    let mut queries: Vec<f32> = Vec::new();
    for qi in 0..32 {
        let noise = draw(0.05);
        let row = bridge(qi % n_clusters, &noise);
        normalize_into(row, &mut queries);
    }
    let nq = 32;
    let k = 10;
    let all_ids = ids(0..n as u64);

    let plain_dir = temp_dir("rep-plain");
    let replicated_dir = temp_dir("rep-on");
    let mut plain = FreshIndex::new(Some(DIM), BITS).unwrap();
    plain.set_partitioning(Some(TARGET));
    plain.add_with_ids(&corpus, &all_ids).unwrap();
    plain.save(&plain_dir).unwrap();
    let mut replicated = FreshIndex::new(Some(DIM), BITS).unwrap();
    replicated.set_partitioning(Some(TARGET));
    replicated.set_replication(Some(1.0));
    replicated.add_with_ids(&corpus, &all_ids).unwrap();
    replicated.save(&replicated_dir).unwrap();
    assert!(
        replicated.replica_count() > 0,
        "bridged corpus produced no replicas",
    );
    assert_eq!(replicated.len(), n, "replicas must not count toward len");

    let (_, oracle_ids) = full_probe(&plain, &queries, k);
    let (_, plain_ids) = plain.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(1),
            ..SearchOptions::default()
        },
    );
    let (_, replicated_ids) = replicated.search_with_options(
        &queries,
        k,
        SearchOptions {
            nprobe: Some(1),
            ..SearchOptions::default()
        },
    );
    let plain_recall = recall_against(&oracle_ids, &plain_ids, nq, k);
    let replicated_recall = recall_against(&oracle_ids, &replicated_ids, nq, k);
    println!(
        "fresh nprobe=1: plain {plain_recall:.3} -> replicated {replicated_recall:.3} \
         (replicas={})",
        replicated.replica_count(),
    );
    assert!(
        replicated_recall > plain_recall,
        "replication produced no lift at nprobe=1",
    );

    // Removing an id hides every copy.
    let victim = (n - 1) as u64; // a bridge vector (replicated region)
    assert!(replicated.remove(victim));
    assert!(!replicated.contains(victim));
    let (_, post_ids) = full_probe(&replicated, &queries, 50);
    assert!(
        !post_ids.contains(&victim),
        "removed id surfaced via a replica copy",
    );

    cleanup(&plain_dir);
    cleanup(&replicated_dir);
}

// ---------------------------------------------------------------------------
// Membership runs
// ---------------------------------------------------------------------------

#[test]
fn many_flushes_merge_runs_and_membership_stays_correct() {
    let dir = temp_dir("runs");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(64));

    let mut next_id = 0u64;
    for flush in 0..12 {
        let batch = mixture_vectors(2, 40, 80 + flush); // 80 per flush
        let batch_ids = ids(next_id..next_id + 80);
        index.add_with_ids(&batch, &batch_ids).unwrap();
        next_id += 80;
        index.save(&dir).unwrap();
    }
    assert_eq!(index.len(), 960);
    assert!(
        index.run_count() <= MAX_RUNS_BOUND,
        "run merging did not bound the run count ({})",
        index.run_count(),
    );
    // Membership across the whole history, including re-add after remove.
    assert!(index.contains(0));
    assert!(index.contains(959));
    assert!(!index.contains(960));
    assert!(index.remove(5));
    assert!(!index.contains(5));
    let replacement = mixture_vectors(1, 1, 99);
    index.add_with_ids(&replacement, &[5]).unwrap();
    assert!(index.contains(5));
    index.save(&dir).unwrap();
    let reopened = FreshIndex::open(&dir).unwrap();
    assert!(reopened.contains(5));
    assert_eq!(reopened.len(), 960);

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Migration
// ---------------------------------------------------------------------------

#[test]
fn import_from_disk_index_is_lossless() {
    let disk_path = temp_dir("import-src.tvdm");
    let dir = temp_dir("import-dst");
    let vectors = mixture_vectors(8, 200, 90); // 1600
    let queries = mixture_vectors(8, 2, 91);
    let nq = 16;
    let k = 10;

    let mut disk = DiskIndex::new(Some(DIM), BITS).unwrap();
    disk.set_partitioning(Some(TARGET));
    disk.set_store_vectors(true);
    disk.add_with_ids(&vectors, &ids(0..1600)).unwrap();
    disk.write(&disk_path).unwrap();

    let fresh = FreshIndex::import_disk_index_file(&disk_path, &dir).unwrap();
    assert_eq!(fresh.len(), 1600);
    assert_eq!(fresh.nlist(), disk.nlist());
    assert!(fresh.stores_vectors());

    // Identical codes + identical partitioning => identical full-probe
    // results (and identical exact rescoring).
    let (disk_scores, disk_ids) =
        disk.search_with_nprobe(&queries, k, Some(disk.nlist()));
    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&disk_scores, &disk_ids, nq),
    );
    assert_eq!(
        fresh.get_vector(11).unwrap(),
        vectors[11 * DIM..12 * DIM].to_vec(),
    );

    // The imported directory is a fully functional index.
    let reopened = FreshIndex::open(&dir).unwrap();
    assert_eq!(reopened.len(), 1600);

    cleanup(&disk_path);
    cleanup(&dir);
}

#[test]
fn export_id_map_file_round_trips_codes() {
    let dir = temp_dir("export");
    let tvim_path = temp_dir("export.tvim");
    let vectors = mixture_vectors(4, 150, 95); // 600
    let queries = mixture_vectors(4, 2, 96);
    let nq = 8;
    let k = 10;

    let mut fresh = FreshIndex::new(Some(DIM), BITS).unwrap();
    fresh.set_partitioning(Some(TARGET));
    fresh.add_with_ids(&vectors, &ids(0..600)).unwrap();
    fresh.save(&dir).unwrap();

    fresh.export_id_map_file(&tvim_path).unwrap();
    let id_map = IdMapIndex::load(&tvim_path).unwrap();
    assert_eq!(id_map.len(), 600);
    let (fresh_scores, fresh_ids) = full_probe(&fresh, &queries, k);
    let (tvim_scores, tvim_ids) = id_map.search(&queries, k);
    assert_eq!(
        result_multisets(&fresh_scores, &fresh_ids, nq),
        result_multisets(&tvim_scores, &tvim_ids, nq),
    );

    cleanup(&dir);
    cleanup(&tvim_path);
}
