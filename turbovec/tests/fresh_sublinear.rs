//! Two per-save costs that were O(N), and the invariants their fixes put at
//! risk.
//!
//! `publish` grew 16x over an 8.2x rise in nlist while every other ingest
//! phase grew sublinearly, and it contains exactly two O(N) operations:
//!
//!   * the manifest carries every centroid (`nlist * dim * 4`) and is rewritten
//!     whole every save, though 73% of saves change no centroid;
//!   * a run merge rewrites every live id->location entry.
//!
//! The centroid fix moves the blob to `centroids-<gen>` and rewrites it only
//! on change. That introduces a file the manifest REFERENCES, and three ways
//! to get it wrong: overwrite it in place (an unreplaced manifest then resolves
//! to the wrong bytes), let the orphan sweep delete it (the sweep removes every
//! filename it does not recognise), or fail to bump the generation.
//!
//! The run fix merges in size tiers rather than collapsing everything. Lookup
//! scans runs newest-first and takes the first hit, so a newer run shadows an
//! older one; merging a non-contiguous set, or reinserting the merged run in
//! the wrong position, resurrects superseded rows.

use std::collections::HashSet;
use std::path::PathBuf;

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
    p.push(format!("turbovec-sublin-{}-{}", nonce, name));
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

fn centroid_blobs(dir: &PathBuf) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().to_string();
            n.starts_with("centroids-").then_some(n)
        })
        .collect();
    v.sort();
    v
}

/// The point of the split: a save that changes no centroid must not rewrite
/// them. Without the dirty flag this rewrites `nlist * dim * 4` bytes every
/// save -- ~30 MB at 10M x 768d, against ~4 MB of new row data.
#[test]
fn unchanged_centroids_are_not_rewritten() {
    let dir = temp_dir("nochange");
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    ix.add_with_ids(&vectors(4_000, 3), &(0..4_000).collect::<Vec<u64>>())
        .unwrap();
    ix.save(&dir).unwrap();

    // Saves that add nothing cannot change a centroid.
    let after_build = centroid_blobs(&dir);
    for _ in 0..5 {
        ix.save(&dir).unwrap();
    }
    assert_eq!(
        centroid_blobs(&dir),
        after_build,
        "a no-op save rewrote the centroid blob",
    );
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}

/// Exactly one blob may survive a run of saves that DO change centroids --
/// superseded generations must be retired, or the directory grows without
/// bound at `nlist * dim * 4` a time.
#[test]
fn superseded_centroid_blobs_are_retired() {
    let dir = temp_dir("retire");
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    for round in 0..8 {
        let lo = round * 1_000;
        ix.add_with_ids(
            &vectors(1_000, 100 + round as u64),
            &(lo..lo + 1_000).collect::<Vec<u64>>(),
        )
        .unwrap();
        ix.save(&dir).unwrap();
    }
    let blobs = centroid_blobs(&dir);
    assert_eq!(
        blobs.len(),
        1,
        "expected one live centroid blob, found {}: {:?}",
        blobs.len(),
        blobs,
    );
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}

/// The orphan sweep at open deletes every filename it does not recognise. If
/// it does not know about the centroid blob, the FIRST open still works (the
/// manifest is read before the sweep runs) and the SECOND fails -- so this
/// reopens twice on purpose.
#[test]
fn the_orphan_sweep_keeps_the_referenced_blob() {
    let dir = temp_dir("sweep");
    let base = vectors(3_000, 11);
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    ix.add_with_ids(&base, &(0..3_000).collect::<Vec<u64>>())
        .unwrap();
    ix.save(&dir).unwrap();
    let nlist = ix.nlist();
    drop(ix);

    for pass in 0..2 {
        let ix = FreshIndex::open(&dir).unwrap();
        assert_eq!(ix.nlist(), nlist, "pass {pass}: partition count changed");
        assert!(
            !centroid_blobs(&dir).is_empty(),
            "pass {pass}: the sweep deleted the blob the manifest references",
        );
        let (_, got) = ix.search_with_options(&base[..50 * DIM], 10, full_probe(nlist));
        for q in 0..50 {
            assert!(
                got[q * 10..(q + 1) * 10].contains(&(q as u64)),
                "pass {pass}: vector {q} unreachable after reopen",
            );
        }
        drop(ix);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Tiered merging must keep newer entries shadowing older ones. A row that
/// moved partitions has a stale entry in an older run; merge the ranges wrong
/// and the stale location wins, returning the wrong vector or a dead row.
#[test]
fn tiered_run_merge_preserves_shadowing() {
    let dir = temp_dir("shadow");
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    let base = vectors(4_000, 17);
    ix.add_with_ids(&base, &(0..4_000).collect::<Vec<u64>>())
        .unwrap();
    ix.save(&dir).unwrap();

    // Many saves with deletes: rows move, partitions split and compact, so
    // older runs accumulate entries that newer runs supersede.
    let mut removed: HashSet<u64> = HashSet::new();
    for round in 0..14u64 {
        let lo = 10_000 + round * 400;
        ix.add_with_ids(
            &vectors(400, 200 + round),
            &(lo..lo + 400).collect::<Vec<u64>>(),
        )
        .unwrap();
        for id in (round * 120..(round + 1) * 120).step_by(4) {
            ix.remove(id);
            removed.insert(id);
        }
        ix.save(&dir).unwrap();
    }
    assert!(
        ix.maintenance_stats().run_merges > 0,
        "no run merge fired, so this test proves nothing",
    );

    let nlist = ix.nlist();
    let (_, got) = ix.search_with_options(&base[..400 * DIM], 10, full_probe(nlist));
    let (mut resurrected, mut lost) = (Vec::new(), Vec::new());
    for q in 0..400 {
        let id = q as u64;
        let hit = got[q * 10..(q + 1) * 10].contains(&id);
        if removed.contains(&id) {
            if hit {
                resurrected.push(id);
            }
        } else if !hit {
            lost.push(id);
        }
    }
    assert!(
        resurrected.is_empty(),
        "removed ids came back through a stale run entry: {:?}",
        &resurrected[..resurrected.len().min(10)],
    );
    assert!(
        lost.is_empty(),
        "live ids lost across tiered run merges: {:?}",
        &lost[..lost.len().min(10)],
    );
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}

/// Tiering trades merge cost for run count, so the count must stay bounded --
/// lookup walks every run, and an unbounded list moves the cost to the read
/// path, which is the thing this must not do.
#[test]
fn tiered_run_merge_bounds_the_run_count() {
    let dir = temp_dir("bound");
    let mut ix = FreshIndex::new(Some(DIM), BITS).unwrap();
    ix.set_partitioning(Some(TARGET));
    ix.add_with_ids(&vectors(2_000, 23), &(0..2_000).collect::<Vec<u64>>())
        .unwrap();
    ix.save(&dir).unwrap();

    let mut peak = 0usize;
    for round in 0..40u64 {
        let lo = 50_000 + round * 200;
        ix.add_with_ids(
            &vectors(200, 300 + round),
            &(lo..lo + 200).collect::<Vec<u64>>(),
        )
        .unwrap();
        ix.save(&dir).unwrap();
        peak = peak.max(ix.run_count());
    }
    assert!(
        peak <= 24,
        "run count reached {peak} over 40 saves; tiering is not bounding it",
    );
    drop(ix);
    std::fs::remove_dir_all(&dir).ok();
}
