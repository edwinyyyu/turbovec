//! `Rotation::apply_inverse` must undo `Rotation::apply`.
//!
//! A partitioned index decodes stored codes back to approximate vectors --
//! k-means over a partition, split centroids, reassignment -- and all of that
//! works in the ORIGINAL coordinate space. Upstream never needed an inverse
//! because it only rotates into the encoded space, so this is new code and it
//! is the piece of the port most able to be subtly wrong: a permutation
//! inverted in the wrong direction still produces plausible-looking vectors
//! with the right norm, and would surface only as quietly degraded recall.
//!
//! Norm preservation alone would NOT catch that -- any orthogonal map
//! preserves norms, including a wrong one. So this asserts coordinate-wise
//! recovery.

use rand::{Rng, SeedableRng};
use turbovec::rotation::Rotation;

/// Dims chosen to cover the block cases the transform special-cases: powers of
/// two, `8 * odd` (the weak-block case the permutation-first order exists to
/// fix), and the product's own 768.
const DIMS: &[usize] = &[8, 16, 24, 32, 64, 96, 128, 200, 256, 768, 1536];

#[test]
fn inverse_recovers_the_original_row() {
    for &dim in DIMS {
        let rot = Rotation::new(dim);
        let mut rng = rand::rngs::StdRng::seed_from_u64(dim as u64);
        let original: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();

        let mut row = original.clone();
        rot.apply(&mut row);
        assert_ne!(
            row, original,
            "dim {dim}: rotation was a no-op, so the round trip proves nothing",
        );
        rot.apply_inverse(&mut row);

        for (i, (got, want)) in row.iter().zip(&original).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "dim {dim} coord {i}: round trip gave {got}, want {want}",
            );
        }
    }
}

/// The inverse must be a true inverse, not merely norm-preserving. A wrongly
/// directed permutation passes a norm check and fails this.
#[test]
fn inverse_is_not_merely_norm_preserving() {
    let dim = 768;
    let rot = Rotation::new(dim);
    let mut basis = vec![0.0f32; dim];
    basis[42] = 1.0;

    let mut row = basis.clone();
    rot.apply(&mut row);
    rot.apply_inverse(&mut row);

    let (peak, peak_val) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    assert_eq!(peak, 42, "energy landed on coord {peak}, not 42");
    assert!((peak_val - 1.0).abs() < 1e-4, "peak {peak_val}, want 1.0");
}
