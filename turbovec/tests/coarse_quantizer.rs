//! The two-level coarse quantizer must agree with the flat scan often enough
//! to be worth its speedup, and must never produce an invalid assignment.
//!
//! It is APPROXIMATE by construction — that is the whole trade — so the test
//! cannot demand equality. What it can demand is that every assignment is a
//! real partition, and that agreement with exact assignment is high enough
//! that reassignment has little to repair. A hierarchy that returns valid but
//! near-random partitions would pass a correctness check and destroy recall,
//! so agreement is asserted, not just validity.

use rand::{Rng, SeedableRng};
use turbovec::kmeans_test_api as km;

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    // Clustered, not uniform: uniform data makes every centroid equidistant
    // and would flatter any router.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_clusters = 64;
    let centers: Vec<f32> = (0..n_clusters * dim)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = (i % n_clusters) * dim;
        for d in 0..dim {
            out.push(centers[c + d] + rng.gen_range(-0.15f32..0.15));
        }
    }
    out
}

#[test]
fn hierarchical_assignment_agrees_with_the_flat_scan() {
    let dim = 64;
    for &nlist in &[256usize, 1024] {
        let n_rows = 4_000;
        let centroids = corpus(nlist, dim, 7);
        let rows = corpus(n_rows, dim, 99);

        let exact = km::assign_flat(&rows, n_rows, dim, &centroids, nlist);
        let coarse = km::CoarseIndex::build(&centroids, nlist, dim, 8, 42);
        for &probe in &[1usize, 2, 4, 8] {
            let got = coarse.assign(&rows, n_rows, &centroids, probe);
            for (i, &c) in got.iter().enumerate() {
                assert!((c as usize) < nlist, "nlist {nlist} probe {probe}: row {i} -> {c}");
            }
            let agree = got.iter().zip(&exact).filter(|(a, b)| a == b).count();
            println!("nlist={nlist} probe={probe}: agreement {:.1}%", agree as f64 / n_rows as f64 * 100.0);
        }
        let got = coarse.assign(&rows, n_rows, &centroids, 8);

        for (i, &c) in got.iter().enumerate() {
            assert!(
                (c as usize) < nlist,
                "nlist {nlist}: row {i} assigned to partition {c}, out of range",
            );
        }
        let agree = got.iter().zip(&exact).filter(|(a, b)| a == b).count();
        let rate = agree as f64 / n_rows as f64;
        println!("nlist={nlist}: agreement with exact assignment {:.1}%", rate * 100.0);
        assert!(
            rate > 0.60,
            "nlist {nlist}: only {:.1}% agreement with exact assignment — the \
             hierarchy is routing near-randomly, which would cost recall even \
             though every assignment is a valid partition",
            rate * 100.0,
        );
    }
}

/// The hierarchy must actually PRUNE, not merely answer accurately.
///
/// This is the assertion whose absence let a two-level quantizer ship that
/// did 98% of the flat scan's arithmetic. Level-2 cost is
/// `sum over groups of rows_g * members_s`, which equals the intended
/// `n * nlist/n_super` only when supers own comparable numbers of centroids.
/// Plain Lloyd over the centroids of a CLUSTERED corpus does not deliver
/// that -- measured 1..582 members (mean 39) on a real 1,561-partition index
/// -- and because the crowded supers also attract the most rows, the two
/// skews multiply. Agreement stays fine throughout, so only a work assertion
/// catches it.
///
/// Random centroids cannot expose this: they are balanced by construction.
/// The centroids here are clustered, like real ones.
///
/// The sizes start at `HIERARCHY_MIN_NLIST`. Below it the hierarchy cannot
/// prune whatever the balance -- worst-case coverage is `1.5 * probe /
/// sqrt(nlist)`, which at nlist 256 is 8 probes over 16 supers, i.e. most of
/// the index -- which is why that constant exists and why it sits where it
/// does.
#[test]
fn hierarchical_assignment_actually_prunes() {
    let dim = 64;
    for &nlist in &[1024usize, 4096, 16384] {
        let centroids = corpus(nlist, dim, 7);
        let coarse = km::CoarseIndex::build(&centroids, nlist, dim, 8, 42);
        let counts = coarse.member_counts();
        let n_super = counts.len();
        let total: usize = counts.iter().sum();
        assert_eq!(total, nlist, "every centroid must belong to exactly one super");

        let mean = total as f64 / n_super as f64;
        let max = *counts.iter().max().expect("at least one super");
        // The cap is 1.5x the even share; allow one centroid of rounding.
        let bound = (mean * 1.5).ceil() as usize + 1;
        assert!(
            max <= bound,
            "nlist {nlist}: largest super holds {max} centroids, cap is {bound} \
             (mean {mean:.1} over {n_super} supers) -- unbalanced supers make the \
             hierarchy scan nearly everything",
        );

        // With probe=8 the fraction of centroids reachable in level 2 is
        // bounded by 8 * max/nlist. Require real pruning, not just a cap.
        let probed = (8 * max) as f64 / nlist as f64;
        println!(
            "nlist={nlist} nsuper={n_super} mean={mean:.1} max={max} \
             worst-case probed fraction {:.1}%",
            probed * 100.0
        );
        assert!(
            probed < 0.75,
            "nlist {nlist}: worst-case probe reaches {:.0}% of centroids -- \
             the hierarchy is not pruning",
            probed * 100.0
        );
    }
}
