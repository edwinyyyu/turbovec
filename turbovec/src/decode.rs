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
    // Walk BYTES of the bit-plane layout, not dimensions.
    //
    // `byte_in_plane` only changes every 8 dimensions, so a per-dimension loop
    // re-loads each plane byte eight times -- 3,072 loads per row at dim 768,
    // bits 4, for the 384 bytes that exist. Hoisting the load out and peeling
    // the 8 dimensions it covers keeps the arithmetic identical and touches
    // each byte once. The per-bit branch also becomes a shift-and-or, which
    // has no misprediction to pay on essentially random code bits.
    debug_assert!(bits <= 8, "bit-plane decode holds one byte per plane");
    x_hat.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let packed_row = &packed_codes[i * bytes_per_row..(i + 1) * bytes_per_row];
        let mut planes = [0u8; 8];
        for byte_in_plane in 0..bytes_per_plane {
            for (p, slot) in planes[..bits].iter_mut().enumerate() {
                *slot = packed_row[p * bytes_per_plane + byte_in_plane];
            }
            let d0 = byte_in_plane * 8;
            for k in 0..8 {
                let d = d0 + k;
                if d >= dim {
                    break;
                }
                let bit_in_byte = 7 - k;
                let mut code = 0usize;
                for (p, &plane) in planes[..bits].iter().enumerate() {
                    code |= (((plane >> bit_in_byte) & 1) as usize) << p;
                }
                let centroid = centroids[code];
                row[d] = if identity_calibration {
                    centroid
                } else {
                    centroid / tqplus_scale[d] - tqplus_shift[d]
                };
            }
        }
    });

    finish(x_hat, scales, dim, rotation)
}

/// Reconstruct `n` approximate vectors straight from GROUP-BYTE rows.
///
/// `decode` takes the bit-plane layout, which exists for the SIMD distance
/// kernels -- but a caller holding group bytes had to run
/// `pack::packed_from_group_bytes` first, and that scatters every code into
/// four bit-planes only for `decode` to gather it straight back out. The round
/// trip cancels: measured at n=10,000, dim=768, bits=4, the scatter cost
/// 68.6 ms against 6.4 ms for the decode it was preparing, i.e. 91% of the
/// work was undoing itself.
///
/// Reading the code directly is one load, one shift, one mask per dimension.
/// Output is bit-identical to `decode(packed_from_group_bytes(rows), ..)`,
/// which `decode_groups_matches_packed_path` asserts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_groups(
    rows: &[u8],
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
    let codes_per_byte = 8 / bits;
    let n_byte_groups = crate::pack::n_byte_groups(bits, dim);
    assert_eq!(rows.len(), n * n_byte_groups);
    let identity_calibration = tqplus_shift.is_empty();
    if !identity_calibration {
        assert_eq!(tqplus_shift.len(), dim);
        assert_eq!(tqplus_scale.len(), dim);
    }
    let mask = (1u8 << bits) - 1;

    let mut x_hat = vec![0.0f32; n * dim];
    x_hat.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let group_row = &rows[i * n_byte_groups..(i + 1) * n_byte_groups];
        for (d, value) in row.iter_mut().enumerate() {
            let g = d / codes_per_byte;
            let c = d % codes_per_byte;
            // 3-bit codes are stored two per byte on 4-bit boundaries, so the
            // stride is 4 rather than `bits` -- mirroring `group_bytes`.
            let shift = if bits == 3 {
                (codes_per_byte - 1 - c) * 4
            } else {
                (codes_per_byte - 1 - c) * bits
            };
            let code = ((group_row[g] >> shift) & mask) as usize;
            let centroid = centroids[code];
            *value = if identity_calibration {
                centroid
            } else {
                centroid / tqplus_scale[d] - tqplus_shift[d]
            };
        }
    });

    finish(x_hat, scales, dim, rotation)
}

/// Shared tail: undo the rotation, then restore magnitude.
fn finish(
    x_hat: Vec<f32>,
    scales: &[f32],
    dim: usize,
    rotation: &crate::rotation::Rotation,
) -> Vec<f32> {
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
