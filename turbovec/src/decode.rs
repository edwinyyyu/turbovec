//! Approximate reconstruction of vectors from quantized codes.
//!
//! Inverts the encode pipeline (see `encode.rs`): per-coordinate codes map
//! back to Lloyd-Max centroid values, TQ+ calibration is undone, the random
//! rotation is inverted (it is orthogonal, so its inverse is its transpose),
//! and the stored per-vector scale recovers the magnitude:
//!
//! ```text
//! x_hat[d] = centroids[code[d]] / scale_tq[d] - shift[d]   (≈ u_rot)
//! v_hat    = (x_hat · R) * scale                            (≈ v)
//! ```
//!
//! The reconstruction is lossy (that's the point of quantization) but more
//! than accurate enough for clustering: the quantization error is small
//! relative to inter-cluster distances.

use rayon::prelude::*;

/// Reconstruct `n` approximate vectors from bit-plane packed codes.
/// Returns a flat `n * dim` array.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode(
    packed_codes: &[u8],
    scales: &[f32],
    n: usize,
    dim: usize,
    bits: usize,
    rotation: &crate::rotation::Rotation,
    centroids: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> Vec<f32> {
    assert_eq!(scales.len(), n);
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    assert_eq!(packed_codes.len(), n * bytes_per_row);
    let identity_calibration = tqplus_shift.is_empty();
    if !identity_calibration {
        assert_eq!(tqplus_shift.len(), dim);
        assert_eq!(tqplus_scale.len(), dim);
    }

    // Codes -> centroid values in (approximate) rotated space.
    let mut x_hat = vec![0.0f32; n * dim];
    x_hat.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let packed_row = &packed_codes[i * bytes_per_row..(i + 1) * bytes_per_row];
        for (d, value) in row.iter_mut().enumerate() {
            let byte_in_plane = d / 8;
            let bit_in_byte = 7 - (d % 8);
            let mut code = 0usize;
            for p in 0..bits {
                if packed_row[p * bytes_per_plane + byte_in_plane] >> bit_in_byte & 1 == 1 {
                    code |= 1 << p;
                }
            }
            let centroid = centroids[code];
            *value = if identity_calibration {
                centroid
            } else {
                centroid / tqplus_scale[d] - tqplus_shift[d]
            };
        }
    });

    // Invert the rotation. Upstream's transform is block-Hadamard rather than
    // a dense matrix, so this is `apply_inverse` per row instead of a GEMM --
    // O(d log d) and with no matrix to hold.
    let mut reconstructed = x_hat;
    reconstructed.par_chunks_mut(dim).for_each_init(
        || vec![0.0f32; dim],
        |scratch, row| rotation.apply_inverse_with_scratch(row, scratch),
    );

    // Recover magnitude. scale = ||v|| / <u_rot, x_hat>, and since
    // u_hat has the same norm as x_hat (orthogonal rotation) with
    // <u_rot, x_hat> ~= ||x_hat||^2 for a good reconstruction, scaling the
    // un-normalized u_hat by `scale` lands close to the original v.
    reconstructed
        .par_chunks_mut(dim)
        .zip(scales.par_iter())
        .for_each(|(row, &scale)| {
            for value in row.iter_mut() {
                *value *= scale;
            }
        });

    reconstructed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{codebook, rotation as rotation_mod, TurboQuantIndex};

    fn pseudo_random_unit_vectors(n: usize, dim: usize, mut state: u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Roughly uniform in [-1, 1); good enough as gaussian stand-in.
            out.push(((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0);
        }
        out
    }

    #[test]
    fn decode_reconstructs_directions_well() {
        let n = 1200; // above TQPLUS_MIN_SAMPLES so calibration is exercised
        let dim = 64;
        let bits = 4;
        let vectors = pseudo_random_unit_vectors(n, dim, 7);

        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
        index.add(&vectors);

        let rotation = rotation_mod::Rotation::new(dim);
        let (_, centroids) = codebook::codebook(bits, dim);
        let reconstructed = decode(
            index.packed_codes(),
            index.scales(),
            n,
            dim,
            bits,
            &rotation,
            &centroids,
            index.tqplus_shift(),
            index.tqplus_scale(),
        );

        // Reconstruction quality target: mean cosine similarity well above
        // anything cluster assignment needs.
        let mut total_cosine = 0.0f64;
        for i in 0..n {
            let original = &vectors[i * dim..(i + 1) * dim];
            let decoded = &reconstructed[i * dim..(i + 1) * dim];
            let dot: f64 = original
                .iter()
                .zip(decoded)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let norm_a: f64 = original
                .iter()
                .map(|&a| (a as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm_b: f64 = decoded
                .iter()
                .map(|&b| (b as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            total_cosine += dot / (norm_a * norm_b);
        }
        let mean_cosine = total_cosine / n as f64;
        assert!(
            mean_cosine > 0.9,
            "mean reconstruction cosine {mean_cosine} too low",
        );
    }
}
