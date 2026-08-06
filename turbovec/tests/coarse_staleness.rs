//! The coarse level must survive the centroid table shrinking under it.
//!
//! Member lists hold centroid ids up to `built_for - 1`, and a dissolve
//! renumbers partitions. A level built for N centroids and then used against
//! fewer indexes out of bounds on the assignment path. The staleness rule this
//! guards once compared `max/min > 1.2`, which called a 1,000 -> 900 shrink
//! "fresh" -- so the shrink here is kept deliberately small, INSIDE that old
//! tolerance. A large delete rebuilds anyway and would prove nothing.
//!
//! Two guards now cover this: the staleness check inside assignment and the
//! end-of-flush refresh, either of which alone is sufficient. Verified by
//! removing BOTH, which panics at `kmeans.rs` on the out-of-range centroid
//! slice; with either present it passes. So this test fails only on a
//! regression that removes both, not on one -- weaker than ideal, and recorded
//! rather than papered over.

use rand::{Rng, SeedableRng};
use turbovec::FreshIndex;

const TARGET: usize = 16;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-coarsestale-{nonce}-{name}"));
    p
}

/// Clustered, not uniform: uniform vectors make every centroid equidistant and
/// produce evenly-sized supers, which is the case that hides router defects.
fn corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_clusters = 128;
    let centers: Vec<f32> = (0..n_clusters * dim)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = (i % n_clusters) * dim;
        for d in 0..dim {
            out.push(centers[c + d] + rng.gen_range(-0.25f32..0.25));
        }
    }
    out
}

#[test]
fn assignment_survives_a_shrinking_centroid_table() {
    let (n, dim) = (40_000usize, 64usize);
    let data = corpus(n, dim, 5);
    let path = temp_dir("shrink");

    let mut ix = FreshIndex::new(Some(dim), 4).expect("construct");
    ix.set_partitioning(Some(TARGET));
    let mut tuning = ix.tuning();
    tuning.hierarchical_assign = true;
    tuning.rebootstrap_enabled = false;
    ix.set_tuning(tuning);

    // Several saves: the level is built during assignment, and assignment
    // returns early until the index is clustered -- which happens in the
    // maintenance pass AFTER the first batch. One save never builds a level.
    let chunk = n / 4;
    for lo in (0..n).step_by(chunk) {
        let hi = (lo + chunk).min(n);
        let ids: Vec<u64> = (lo as u64..hi as u64).collect();
        ix.add_with_ids(&data[lo * dim..hi * dim], &ids).expect("add");
        ix.save(&path).expect("save");
    }
    let before = ix.nlist();
    assert!(before >= 1024, "need nlist >= 1024 for the level to exist, got {before}");

    let doomed: Vec<u64> = (0..n as u64).filter(|i| i % 16 == 0).collect();
    ix.remove_many(&doomed);
    ix.save(&path).expect("save");
    let after = ix.nlist();
    let ratio = before as f64 / after.max(1) as f64;
    println!("nlist {before} -> {after} (shrink ratio {ratio:.3})");
    assert!(after < before, "expected partitions to dissolve");
    assert!(
        ratio <= 1.2,
        "shrink ratio {ratio:.3} exceeds the old rule's tolerance, so this case \
         would have rebuilt anyway and proves nothing",
    );

    // The next save ASSIGNS new rows through a level built for a larger table.
    let more = corpus(4_000, dim, 21);
    let ids: Vec<u64> = (n as u64..n as u64 + 4_000).collect();
    ix.add_with_ids(&more, &ids).expect("add");
    ix.save(&path).expect("save");

    let queries = corpus(50, dim, 99);
    for q in 0..50 {
        let (_, ids) = ix.search(&queries[q * dim..(q + 1) * dim], 10);
        for id in ids {
            assert!(id % 16 != 0 || id >= n as u64, "returned a deleted id {id}");
        }
    }
    let _ = std::fs::remove_dir_all(&path);
}
