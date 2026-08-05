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
        let coarse = km::CoarseIndex::build(&centroids, nlist, dim, 42);
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
