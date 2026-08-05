//! Staging holds arrivals in the memtable across saves and distributes them to
//! partitions in one pass, because distribution cost is dominated by how many
//! rows each partition receives at once (a 10k batch over 810 partitions writes
//! ~12-row chunks, and a chunk pads to a whole 32-row block).
//!
//! It adds no storage: the memtable is already durable via the WAL and already
//! scanned by every query. But that is exactly what makes it dangerous. A save
//! that skips the drain must ALSO skip resetting the WAL, because the log is
//! then the only durable copy of those rows. Reset it and a crash loses every
//! row added since the last real drain -- silently, and only on the recovery
//! path, which is the one nobody exercises.
//!
//! `staged_rows_survive_a_reopen` is the test for that, and it fails against
//! the obvious implementation (gate the drain, leave the rest of flush alone).

use std::collections::HashSet;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use turbovec::{FreshIndex, SearchOptions};

const DIM: usize = 32;
const BITS: usize = 4;
const TARGET: usize = 64;
const STAGE: usize = 2_000;

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-stage-{}-{}", nonce, name));
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

fn staged_index(dir: &PathBuf) -> FreshIndex {
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    let mut t = ix.tuning();
    t.staging_threshold = STAGE;
    ix.set_tuning(t);
    let _ = dir;
    ix
}

/// The recovery case. Rows added after the last drain live only in the WAL and
/// the memtable; if the staging save reset the log, reopening loses them.
#[test]
fn staged_rows_survive_a_reopen() {
    let dir = temp_dir("reopen");
    let mut ix = staged_index(&dir);

    // Enough to force one real drain, so the index has partitions...
    let bulk = vectors(3_000, 3);
    let ids: Vec<u64> = (0..3_000).collect();
    ix.add_with_ids(&bulk, &ids).unwrap();
    ix.save(&dir).unwrap();

    // ...then a trickle that stays staged: several saves, none of them big
    // enough to trip the threshold.
    let extra = vectors(900, 5);
    let extra_ids: Vec<u64> = (10_000..10_900).collect();
    for round in 0..3 {
        let lo = round * 300;
        ix.add_with_ids(
            &extra[lo * DIM..(lo + 300) * DIM],
            &extra_ids[lo..lo + 300],
        )
        .unwrap();
        ix.save(&dir).unwrap();
    }
    assert!(
        ix.maintenance_stats().staged_saves > 0,
        "no save skipped the drain, so this test proves nothing",
    );
    drop(ix);

    // Reopen: the staged rows must come back via WAL replay.
    let reopened = FreshIndex::open(&dir).unwrap();
    let nlist = reopened.nlist();
    let (_, got) = reopened.search_with_options(&extra[..200 * DIM], 10, full_probe(nlist));
    let mut lost = Vec::new();
    for q in 0..200 {
        let want = 10_000 + q as u64;
        if !got[q * 10..(q + 1) * 10].contains(&want) {
            lost.push(want);
        }
    }
    assert!(
        lost.is_empty(),
        "{} staged rows lost across a reopen (WAL reset while undrained?): {:?}",
        lost.len(),
        &lost[..lost.len().min(10)],
    );
    drop(reopened);
    std::fs::remove_dir_all(&dir).ok();
}

/// Staged rows must be searchable before they are ever distributed -- they are
/// in the memtable, which the scan already covers.
#[test]
fn staged_rows_are_searchable_before_distribution() {
    let dir = temp_dir("visible");
    let mut ix = staged_index(&dir);
    let bulk = vectors(3_000, 7);
    let ids: Vec<u64> = (0..3_000).collect();
    ix.add_with_ids(&bulk, &ids).unwrap();
    ix.save(&dir).unwrap();

    let extra = vectors(500, 11);
    let extra_ids: Vec<u64> = (20_000..20_500).collect();
    ix.add_with_ids(&extra, &extra_ids).unwrap();
    ix.save(&dir).unwrap();
    assert!(ix.maintenance_stats().staged_saves > 0);

    let nlist = ix.nlist();
    let (_, got) = ix.search_with_options(&extra[..150 * DIM], 10, full_probe(nlist));
    for q in 0..150 {
        assert!(
            got[q * 10..(q + 1) * 10].contains(&(20_000 + q as u64)),
            "staged row {} not searchable before distribution",
            20_000 + q,
        );
    }
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}

/// Crossing the threshold must actually distribute, or staging is just an
/// unbounded memtable and the query cost grows without limit.
#[test]
fn crossing_the_threshold_distributes() {
    let dir = temp_dir("drain");
    let mut ix = staged_index(&dir);
    let bulk = vectors(3_000, 13);
    let ids: Vec<u64> = (0..3_000).collect();
    ix.add_with_ids(&bulk, &ids).unwrap();
    ix.save(&dir).unwrap();
    assert_eq!(ix.memtable_len(), 0, "the bulk load should have drained");

    // Below the threshold: stays staged.
    let a = vectors(500, 17);
    let a_ids: Vec<u64> = (30_000..30_500).collect();
    ix.add_with_ids(&a, &a_ids).unwrap();
    ix.save(&dir).unwrap();
    assert_eq!(ix.memtable_len(), 500, "rows should still be staged");

    // Over the threshold: drains.
    let b = vectors(STAGE, 19);
    let b_ids: Vec<u64> = (40_000..40_000 + STAGE as u64).collect();
    ix.add_with_ids(&b, &b_ids).unwrap();
    ix.save(&dir).unwrap();
    assert_eq!(
        ix.memtable_len(),
        0,
        "crossing the threshold should have distributed everything",
    );

    // And everything is still findable afterwards.
    let nlist = ix.nlist();
    let (_, got) = ix.search_with_options(&a[..100 * DIM], 10, full_probe(nlist));
    for q in 0..100 {
        assert!(
            got[q * 10..(q + 1) * 10].contains(&(30_000 + q as u64)),
            "row {} lost in the distribution",
            30_000 + q,
        );
    }
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}

/// A removal of a staged row must stick across the distribution -- the delete
/// applies to the memtable, and the drain must not resurrect it.
#[test]
fn removing_a_staged_row_survives_distribution() {
    let dir = temp_dir("del");
    let mut ix = staged_index(&dir);
    let bulk = vectors(3_000, 23);
    let ids: Vec<u64> = (0..3_000).collect();
    ix.add_with_ids(&bulk, &ids).unwrap();
    ix.save(&dir).unwrap();

    let a = vectors(600, 29);
    let a_ids: Vec<u64> = (50_000..50_600).collect();
    ix.add_with_ids(&a, &a_ids).unwrap();
    ix.save(&dir).unwrap();

    let mut removed = HashSet::new();
    for id in (50_000..50_600).step_by(3) {
        ix.remove(id);
        removed.insert(id);
    }
    // Force the distribution.
    let b = vectors(STAGE, 31);
    let b_ids: Vec<u64> = (60_000..60_000 + STAGE as u64).collect();
    ix.add_with_ids(&b, &b_ids).unwrap();
    ix.save(&dir).unwrap();

    let nlist = ix.nlist();
    let (_, got) = ix.search_with_options(&a[..200 * DIM], 10, full_probe(nlist));
    let mut resurrected = Vec::new();
    for q in 0..200 {
        let id = 50_000 + q as u64;
        if removed.contains(&id) && got[q * 10..(q + 1) * 10].contains(&id) {
            resurrected.push(id);
        }
    }
    assert!(
        resurrected.is_empty(),
        "removed staged rows came back after distribution: {:?}",
        &resurrected[..resurrected.len().min(10)],
    );
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}
