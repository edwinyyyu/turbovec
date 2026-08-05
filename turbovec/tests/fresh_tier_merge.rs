//! Tier merging folds a partition's trailing small chunks into one by
//! APPENDING the merged chunk and abandoning the originals in place.
//!
//! That is the whole point — a whole-partition rewrite to absorb a handful of
//! trickle-ingested rows was 87.7% of all segment bytes — but it puts three
//! invariants at risk, and each test here fails against a plausible-looking
//! implementation:
//!
//! * **Row renumbering.** Rows are addressed by their index across the chunk
//!   table in order. A merge drops trailing chunks and appends one, so the
//!   merged rows get new indices while the retained prefix keeps its own. Get
//!   the base row wrong and lookups return the wrong vector rather than
//!   failing loudly.
//! * **Abandoned bytes must stay readable.** A reader on an older snapshot
//!   still addresses rows inside the chunks the merge dropped. Truncating them
//!   away — which is what the pre-merge append offset (`end of last live
//!   chunk`) would do on the next append — hands that reader garbage, or
//!   SIGBUS.
//! * **The dead bitmap is a bitmap.** Truncating it to a byte boundary leaves
//!   stale high bits inside the final byte, which the next `resize` reads back
//!   as rows that are dead but should not be.
//!
//! The oracle is a full-probe self-query: a vector that is live must come back
//! in its own top 10. Top-1 would measure 4-bit quantization noise instead.
//!
//! Both fixes were verified to be load-bearing by reverting them in place:
//! restoring the chunk-table append offset fails
//! `pinned_snapshot_still_reads_chunks_a_merge_abandoned`, and dropping the
//! bitmap high-bit clear fails `rows_renumbered_by_a_merge_are_still_addressable`
//! -- NOT `removals_and_merges_agree_on_which_rows_are_dead`, which was written
//! for it. Stale dead-bits make merged rows unreachable before they resurrect a
//! removed one, so the renumbering test trips first. Both are kept; the naming
//! reflects intent, not which assertion happens to fire.

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
    p.push(format!("turbovec-tier-{}-{}", nonce, name));
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

/// Drive the shape that causes merging: many small appends spread over many
/// partitions, which is trickle ingest at scale.
fn trickle(index: &mut FreshIndex, dir: &PathBuf, rounds: usize, per_round: usize, base_id: u64) {
    for round in 0..rounds {
        let lo = base_id + (round * per_round) as u64;
        let v = vectors(per_round, 900 + round as u64);
        let ids: Vec<u64> = (lo..lo + per_round as u64).collect();
        index.add_with_ids(&v, &ids).unwrap();
        index.save(dir).unwrap();
    }
}

/// Every id ever inserted and not removed must still be findable by its own
/// vector, after enough trickle saves to force many merge cascades.
#[test]
fn every_live_id_survives_repeated_tier_merges() {
    let dir = temp_dir("survives");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(4_000, 11);
    let ids: Vec<u64> = (0..4_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    trickle(&mut index, &dir, 12, 300, 4_000);
    assert!(
        index.maintenance_stats().tier_merges > 0,
        "workload never triggered a tier merge, so this test proves nothing",
    );

    // Self-query every vector from the ORIGINAL bulk load: those live in the
    // large base chunk that merging must never disturb.
    let nlist = index.nlist();
    let queries = &base[..200 * DIM];
    let (_, got) = index.search_with_options(queries, 10, full_probe(nlist));
    let mut missing = Vec::new();
    for q in 0..200 {
        if !got[q * 10..(q + 1) * 10].contains(&(q as u64)) {
            missing.push(q);
        }
    }
    assert!(
        missing.is_empty(),
        "{} of 200 base vectors lost after {} tier merges: {:?}",
        missing.len(),
        index.maintenance_stats().tier_merges,
        &missing[..missing.len().min(10)],
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// The merged rows themselves — the ones that got renumbered — must be
/// findable too. This is the test that catches a wrong base row: the base
/// chunk is untouched, so `every_live_id_survives` can pass while every
/// recently-ingested row points at the wrong vector.
#[test]
fn rows_renumbered_by_a_merge_are_still_addressable() {
    let dir = temp_dir("renumber");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(2_000, 13);
    let ids: Vec<u64> = (0..2_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    // Keep the trickled vectors so they can be queried by content.
    let mut trickled: Vec<f32> = Vec::new();
    for round in 0..10 {
        let v = vectors(200, 500 + round);
        let lo = 2_000 + round * 200;
        let ids: Vec<u64> = (lo..lo + 200).collect();
        index.add_with_ids(&v, &ids).unwrap();
        index.save(&dir).unwrap();
        trickled.extend_from_slice(&v);
    }
    assert!(index.maintenance_stats().tier_merges > 0);

    let nlist = index.nlist();
    let n_probe = 150usize;
    let queries = &trickled[..n_probe * DIM];
    let (_, got) = index.search_with_options(queries, 10, full_probe(nlist));
    let mut missing = Vec::new();
    for q in 0..n_probe {
        let want = 2_000 + q as u64;
        if !got[q * 10..(q + 1) * 10].contains(&want) {
            missing.push(want);
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {} merged rows unreachable (renumbering bug): {:?}",
        missing.len(),
        n_probe,
        &missing[..missing.len().min(10)],
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// A snapshot pinned BEFORE a merge still addresses rows inside the chunks the
/// merge abandoned. Those bytes must remain in the file and remain correct.
///
/// This is the one that fails if the append offset comes from the chunk table
/// rather than `file_bytes`: the next append truncates to the end of the last
/// live chunk, cutting the abandoned bytes off underneath the pinned reader.
#[test]
fn pinned_snapshot_still_reads_chunks_a_merge_abandoned() {
    let dir = temp_dir("abandoned");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(2_000, 17);
    let ids: Vec<u64> = (0..2_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    // A few small appends WITHOUT letting a merge cascade collapse them yet,
    // then pin. The pinned snapshot names those small chunks directly.
    trickle(&mut index, &dir, 2, 150, 2_000);
    let reader = index.reader();
    let pinned = reader.snapshot();
    let pinned_nlist = pinned.nlist();
    let before = index.maintenance_stats().tier_merges;

    // Now churn hard enough to merge those chunks away and to keep appending
    // afterwards -- the append is what would truncate them.
    trickle(&mut index, &dir, 10, 300, 2_300);
    assert!(
        index.maintenance_stats().tier_merges > before,
        "no merge happened after pinning, so nothing was abandoned",
    );

    // The pinned snapshot must still find its own vectors.
    let queries = &base[..100 * DIM];
    let (_, got) = pinned.search_with_options(queries, 10, full_probe(pinned_nlist));
    let mut missing = Vec::new();
    for q in 0..100 {
        if !got[q * 10..(q + 1) * 10].contains(&(q as u64)) {
            missing.push(q);
        }
    }
    assert!(
        missing.is_empty(),
        "pinned snapshot lost {} vectors across a merge: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)],
    );
    let found: HashSet<u64> = got.iter().copied().collect();
    assert!(
        found.iter().all(|&id| id < 2_300 + 300),
        "pinned snapshot saw ids flushed long after it was taken",
    );
    drop(pinned);
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}

/// Merging must survive a save/load round trip: `file_bytes` is what tells the
/// loader where the real end of file is, and the loader truncates anything
/// past it as a crashed append. Persist it wrong and the abandoned bytes are
/// either cut off (losing the prefix's addressing) or the append point lands
/// on top of live data.
#[test]
fn merged_layout_round_trips_through_disk() {
    let dir = temp_dir("roundtrip");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(2_000, 19);
    let ids: Vec<u64> = (0..2_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();
    trickle(&mut index, &dir, 10, 250, 2_000);
    assert!(index.maintenance_stats().tier_merges > 0);
    let nlist_before = index.nlist();
    drop(index);

    let mut reloaded = FreshIndex::open(&dir).unwrap();
    assert_eq!(reloaded.nlist(), nlist_before, "partition count changed");

    let queries = &base[..150 * DIM];
    let (_, got) = reloaded.search_with_options(queries, 10, full_probe(nlist_before));
    let mut missing = Vec::new();
    for q in 0..150 {
        if !got[q * 10..(q + 1) * 10].contains(&(q as u64)) {
            missing.push(q);
        }
    }
    assert!(
        missing.is_empty(),
        "{} vectors unreachable after reload: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)],
    );

    // And the reloaded index must still be writable: this is where a wrong
    // append offset corrupts live rows rather than just losing them.
    trickle(&mut reloaded, &dir, 4, 250, 5_000);
    let (_, got2) = reloaded.search_with_options(queries, 10, full_probe(reloaded.nlist()));
    for q in 0..150 {
        assert!(
            got2[q * 10..(q + 1) * 10].contains(&(q as u64)),
            "vector {q} lost after appending to a reloaded merged layout",
        );
    }
    drop(reloaded);
    std::fs::remove_dir_all(&dir).ok();
}

/// Removals interact with merging through the dead bitmap, which a merge
/// truncates to a byte boundary. Stale high bits in the final byte would
/// resurrect as "dead" rows that were never removed.
#[test]
fn removals_and_merges_agree_on_which_rows_are_dead() {
    let dir = temp_dir("dead-bits");
    let mut index = FreshIndex::new(Some(DIM), BITS).unwrap();
    index.set_partitioning(Some(TARGET));

    let base = vectors(3_000, 23);
    let ids: Vec<u64> = (0..3_000).collect();
    index.add_with_ids(&base, &ids).unwrap();
    index.save(&dir).unwrap();

    // Interleave odd-id removals with trickle appends so merges run against a
    // partially-dead bitmap whose live count is not a multiple of 8.
    let mut removed: HashSet<u64> = HashSet::new();
    for round in 0..10 {
        let lo = 3_000 + round * 200;
        let v = vectors(200, 700 + round);
        let ids: Vec<u64> = (lo..lo + 200).collect();
        index.add_with_ids(&v, &ids).unwrap();
        for id in (round * 100..(round + 1) * 100).step_by(3) {
            index.remove(id as u64);
            removed.insert(id as u64);
        }
        index.save(&dir).unwrap();
    }
    assert!(index.maintenance_stats().tier_merges > 0);

    let nlist = index.nlist();
    let queries = &base[..300 * DIM];
    let (_, got) = index.search_with_options(queries, 10, full_probe(nlist));

    // A removed id must never come back, and a surviving one must be findable.
    let mut resurrected = Vec::new();
    let mut lost = Vec::new();
    for q in 0..300 {
        let window = &got[q * 10..(q + 1) * 10];
        let id = q as u64;
        if removed.contains(&id) {
            if window.contains(&id) {
                resurrected.push(id);
            }
        } else if !window.contains(&id) {
            lost.push(id);
        }
    }
    assert!(
        resurrected.is_empty(),
        "removed ids came back after a merge: {:?}",
        &resurrected[..resurrected.len().min(10)],
    );
    assert!(
        lost.is_empty(),
        "live ids marked dead by a merge's bitmap truncation: {:?}",
        &lost[..lost.len().min(10)],
    );
    drop(index);
    std::fs::remove_dir_all(&dir).ok();
}
