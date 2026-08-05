//! Queries run against a [`FreshReader`] while the writer ingests, maintains
//! and compacts.
//!
//! These are the tests the snapshot design exists for, and each one fails
//! against a plausible-looking implementation of it:
//!
//! * **Reclamation.** A segment file may only be unlinked once no live
//!   snapshot names it. Retiring a file against the snapshot that last
//!   referenced it is *not* enough — a query holding an older snapshot can
//!   outlive that one. `reader_holding_an_old_snapshot_keeps_its_files` pins
//!   a snapshot across several flushes and then reads it.
//! * **Generation keying.** The segment mmap cache must key on
//!   `(partition_id, generation)`. Keyed on partition alone, a reader on an
//!   older snapshot is handed the *newer* file and reads rows at offsets its
//!   snapshot does not describe.
//! * **Visibility.** Every published snapshot must be internally consistent:
//!   a row is in the memtable or in a partition, never briefly in neither.
//!
//! The oracle throughout is a full-probe self-query of a vector that is live
//! in every snapshot ever published: it must come back in the top 10. Top-1
//! identity would NOT do -- at 4 bits it misses about 0.2% of the time even
//! single-threaded, so it measures quantization noise rather than snapshot
//! consistency. Top-10 membership is exact serially (0 misses in 500), which
//! makes any miss here attributable to a torn or stale read.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::{FreshIndex, SearchOptions};

const DIM: usize = 32;
const BITS: usize = 4;
const TARGET: usize = 64;

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-conc-{}-{}", nonce, name));
    p
}

fn vectors(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n * DIM)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect()
}

fn full_probe(nlist: usize) -> SearchOptions {
    SearchOptions {
        nprobe: Some(nlist.max(1)),
        ..SearchOptions::default()
    }
}

/// A reader that grabs a snapshot and then takes its time must still be able
/// to read every segment that snapshot names, however many generations the
/// writer has published since.
///
/// This is the case that a retire-against-the-previous-snapshot scheme gets
/// wrong: a partition untouched for several flushes is named by a whole run
/// of snapshots, and the newest of those can die while an older one is still
/// being read.
#[test]
fn reader_holding_an_old_snapshot_keeps_its_files() {
    let dir = temp_dir("old-snapshot");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(2_000, 7);
    let ids: Vec<u64> = (0..2_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    let reader = index.reader();
    // Pin a snapshot, then churn the index hard underneath it: inserts,
    // deletes, splits, compactions and run merges all rewrite segments and
    // queue the superseded files for deletion.
    let pinned = reader.snapshot();
    let pinned_nlist = pinned.nlist();

    for round in 0..6 {
        let extra = vectors(400, 100 + round);
        let extra_ids: Vec<u64> = (2_000 + round * 400..2_000 + (round + 1) * 400).collect();
        index.add_with_ids(&extra, &extra_ids).unwrap();
        for id in (round * 200..(round + 1) * 200).step_by(2) {
            index.remove(id as u64);
        }
        index.save(&dir).unwrap();
    }

    // Now use the pinned snapshot. Every segment it names must still be
    // readable -- if any were unlinked, the scan silently drops that
    // partition and recall collapses.
    let queries = &base[..20 * DIM];
    let (_, got) = pinned.search_with_options(queries, 10, full_probe(pinned_nlist));
    let found: HashSet<u64> = got.iter().copied().collect();
    assert!(
        found.len() >= 100,
        "pinned snapshot lost partitions: only {} distinct ids over 20 queries",
        found.len(),
    );
    for q in 0..20 {
        assert!(
            got[q * 10..(q + 1) * 10].contains(&(q as u64)),
            "pinned snapshot lost the query vector itself for query {q}: got {:?}",
            &got[q * 10..(q + 1) * 10],
        );
    }
    // A pinned snapshot freezes the PARTITIONS, not the write buffer: the
    // memtable cell it names goes on taking rows until the next flush swaps
    // in a fresh one. So round 0's inserts (2000..2400) land in the pinned
    // cell and are visible, and everything flushed after that is not.
    assert!(
        found.iter().all(|&id| id < 2_400),
        "pinned snapshot saw ids flushed after it was taken: {:?}",
        found.iter().filter(|&&id| id >= 2_400).collect::<Vec<_>>(),
    );
    drop(pinned);
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// Files superseded while a reader holds an old snapshot must be unlinked
/// once that reader lets go — deferring reclamation must not leak them.
#[test]
fn retired_files_are_reclaimed_once_readers_finish() {
    let dir = temp_dir("reclaim");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    let base = vectors(1_500, 11);
    let ids: Vec<u64> = (0..1_500).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    let reader = index.reader();
    let pinned = reader.snapshot();

    for round in 0..4u64 {
        let extra = vectors(300, 200 + round);
        let extra_ids: Vec<u64> = (1_500 + round * 300..1_500 + (round + 1) * 300).collect();
        index.add_with_ids(&extra, &extra_ids).unwrap();
        index.save(&dir).unwrap();
    }

    let held = index.retired_pending();
    drop(pinned);
    // One more publish sweeps the queue now that nothing old is live.
    index.save(&dir).unwrap();
    assert_eq!(
        index.retired_pending(),
        0,
        "retired files leaked: {held} pending while pinned, still \
         {} after the reader finished",
        index.retired_pending(),
    );

    // And the directory really is clean: no superseded generations left.
    let dropped: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let nlist = index.nlist();
    let segments = dropped.iter().filter(|n| n.starts_with("segment-")).count();
    assert_eq!(
        segments, nlist,
        "expected one segment file per partition, found {segments} for {nlist} partitions: \
         {dropped:?}",
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// The headline claim: a query thread makes progress while the writer is
/// inside save/maintain, and every result it gets is correct.
///
/// Correctness here is "never returns a removed id and never misses a
/// never-removed one". A torn snapshot -- partitions from after a flush with
/// a memtable from before it, or vice versa -- shows up as a vector that is
/// briefly in neither.
#[test]
fn queries_are_correct_while_the_writer_ingests() {
    let dir = temp_dir("concurrent");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let total = 6_000usize;
    let base = Arc::new(vectors(total, 3));
    // The first 500 ids are added up front and never removed: they are the
    // invariant the query thread checks on every single search.
    let anchors: Vec<u64> = (0..500).collect();
    index.add_with_ids(&base[..500 * DIM], &anchors).unwrap();
    index.save(&dir).unwrap();

    let reader = index.reader();
    let stop = Arc::new(AtomicBool::new(false));
    let searches = Arc::new(AtomicUsize::new(0));
    let misses = Arc::new(AtomicUsize::new(0));

    let probe = {
        let base = Arc::clone(&base);
        let stop = Arc::clone(&stop);
        let searches = Arc::clone(&searches);
        let misses = Arc::clone(&misses);
        let reader = reader.clone();
        std::thread::spawn(move || {
            let mut q = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let anchor = q % 500;
                let query = &base[anchor * DIM..(anchor + 1) * DIM];
                let nlist = reader.nlist();
                let (_, got) = reader.search_with_options(query, 10, full_probe(nlist));
                // An anchor is live in every snapshot ever published, so its
                // own self-query must surface it. Anything else means the
                // snapshot the scan ran against was internally inconsistent.
                if !got.contains(&(anchor as u64)) {
                    misses.fetch_add(1, Ordering::Relaxed);
                }
                searches.fetch_add(1, Ordering::Relaxed);
                q += 1;
            }
        })
    };

    for round in 0..10usize {
        let lo = 500 + round * 550;
        let hi = lo + 550;
        let add_ids: Vec<u64> = (lo as u64..hi as u64).collect();
        index
            .add_with_ids(&base[lo * DIM..hi * DIM], &add_ids)
            .unwrap();
        // Delete from the previous round's batch, so partitions shrink as
        // well as grow and dissolve/compaction get a chance to fire.
        if round > 0 {
            for id in (lo - 550..lo - 275).step_by(3) {
                index.remove(id as u64);
            }
        }
        index.save(&dir).unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    probe.join().unwrap();

    let n = searches.load(Ordering::Relaxed);
    assert!(
        n > 50,
        "query thread barely ran ({n} searches): it was blocked by the writer",
    );
    assert_eq!(
        misses.load(Ordering::Relaxed),
        0,
        "{} of {n} concurrent searches saw an inconsistent snapshot",
        misses.load(Ordering::Relaxed),
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// Same, but with the writer doing DEFERRED maintenance through `maintain()`
/// rather than inside `save()`. This is the path the caps were built for, and
/// it exercises split/reassign/compaction landing while queries are in
/// flight.
#[test]
fn queries_are_correct_while_maintenance_runs() {
    let dir = temp_dir("maintain");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    let mut tuning = index.tuning();
    tuning.defer_maintenance = true;
    tuning.max_rewrites_per_flush = 4;
    tuning.max_reassign_partitions = 4;
    index.set_tuning(tuning);

    let total = 5_000usize;
    let base = Arc::new(vectors(total, 5));
    let anchors: Vec<u64> = (0..400).collect();
    index.add_with_ids(&base[..400 * DIM], &anchors).unwrap();
    index.save(&dir).unwrap();

    let reader = index.reader();
    let stop = Arc::new(AtomicBool::new(false));
    let searches = Arc::new(AtomicUsize::new(0));
    let misses = Arc::new(AtomicUsize::new(0));
    let probe = {
        let base = Arc::clone(&base);
        let stop = Arc::clone(&stop);
        let searches = Arc::clone(&searches);
        let misses = Arc::clone(&misses);
        let reader = reader.clone();
        std::thread::spawn(move || {
            let mut q = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let anchor = q % 400;
                let query = &base[anchor * DIM..(anchor + 1) * DIM];
                let nlist = reader.nlist();
                let (_, got) = reader.search_with_options(query, 10, full_probe(nlist));
                if !got.contains(&(anchor as u64)) {
                    misses.fetch_add(1, Ordering::Relaxed);
                }
                searches.fetch_add(1, Ordering::Relaxed);
                q += 1;
            }
        })
    };

    for round in 0..8usize {
        let lo = 400 + round * 550;
        let hi = lo + 550;
        let add_ids: Vec<u64> = (lo as u64..hi as u64).collect();
        index
            .add_with_ids(&base[lo * DIM..hi * DIM], &add_ids)
            .unwrap();
        index.save(&dir).unwrap();
        // Drain the deferred work in bounded units, exactly as a caller
        // scheduling maintenance would.
        let mut guard = 0;
        while index.maintain().unwrap() && guard < 40 {
            guard += 1;
        }
    }

    stop.store(true, Ordering::Relaxed);
    probe.join().unwrap();

    let n = searches.load(Ordering::Relaxed);
    assert!(n > 50, "query thread barely ran ({n} searches)");
    assert_eq!(
        misses.load(Ordering::Relaxed),
        0,
        "{} of {n} searches saw an inconsistent snapshot during maintenance",
        misses.load(Ordering::Relaxed),
    );

    // Deferred maintenance, once drained, must still hold the size bound.
    let max = index.partition_sizes().into_iter().max().unwrap_or(0);
    assert!(
        max <= 2 * TARGET,
        "size bound breached after draining maintenance: max posting {max} > {}",
        2 * TARGET,
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// Many readers at once, no writer: snapshots are shared, so this catches
/// anything that mutates through a shared reference during a scan.
#[test]
fn many_readers_share_one_snapshot() {
    let dir = temp_dir("shared");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));
    let base = Arc::new(vectors(3_000, 13));
    let ids: Vec<u64> = (0..3_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    let reader = index.reader();
    let nlist = reader.nlist();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let reader = reader.clone();
            let base = Arc::clone(&base);
            std::thread::spawn(move || {
                let mut hits = 0;
                for i in 0..100 {
                    let q = (t * 100 + i) % 3_000;
                    let (_, got) = reader.search_with_options(
                        &base[q * DIM..(q + 1) * DIM],
                        10,
                        full_probe(nlist),
                    );
                    if got.contains(&(q as u64)) {
                        hits += 1;
                    }
                }
                hits
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), 100, "concurrent readers disagreed");
    }
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}
