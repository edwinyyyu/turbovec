//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

#[cfg(target_arch = "x86_64")]
const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Number of group bytes per vector in the SIMD-blocked layout.
pub(crate) fn n_byte_groups(bits: usize, dim: usize) -> usize {
    dim / (8 / bits)
}

/// Extract per-vector "group bytes" — the nibble-packed bytes the SIMD
/// scoring kernel consumes — from bit-plane packed codes. Returns a flat
/// row-major array of `n_vectors * n_byte_groups(bits, dim)` bytes.
pub(crate) fn group_bytes(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = n_byte_groups(bits, dim);
    let bytes_per_row = bits * bytes_per_plane;

    let mut rows = vec![0u8; n_vectors * n_byte_groups];
    for vec_idx in 0..n_vectors {
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                byte_val |= code << shift;
            }
            rows[vec_idx * n_byte_groups + g] = byte_val;
        }
    }
    rows
}

/// Inverse of [`group_bytes`]: reconstruct bit-plane packed codes from
/// per-vector group-byte rows. `rows` is `n_vectors * n_byte_groups(bits,
/// dim)` bytes; returns `n_vectors * dim * bits / 8` bit-plane bytes.
pub(crate) fn packed_from_group_bytes(
    rows: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = n_byte_groups(bits, dim);
    let bytes_per_row = bits * bytes_per_plane;
    assert_eq!(rows.len(), n_vectors * n_byte_groups);

    let mut packed = vec![0u8; n_vectors * bytes_per_row];
    for vec_idx in 0..n_vectors {
        for g in 0..n_byte_groups {
            let byte_val = rows[vec_idx * n_byte_groups + g];
            let dim_start = g * codes_per_byte;
            for c in 0..codes_per_byte {
                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                let code = (byte_val >> shift) & ((1u8 << bits) - 1);

                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;
                for p in 0..bits {
                    if code & (1 << p) != 0 {
                        packed[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane] |= mask;
                    }
                }
            }
        }
    }
    packed
}

/// Pack up to [`BLOCK`] row-major group-byte rows into one blocked block.
///
/// `rows` is `n_rows * n_byte_groups` bytes (`n_rows <= BLOCK`); `out` must
/// be `n_byte_groups * BLOCK` bytes and is fully overwritten (missing lanes
/// pack as zero, matching the original repack padding).
pub(crate) fn pack_block_rows(rows: &[u8], n_rows: usize, n_byte_groups: usize, out: &mut [u8]) {
    assert!(n_rows <= BLOCK);
    assert_eq!(rows.len(), n_rows * n_byte_groups);
    assert_eq!(out.len(), n_byte_groups * BLOCK);

    let row = |lane: usize, g: usize| -> u8 {
        if lane < n_rows {
            rows[lane * n_byte_groups + g]
        } else {
            0
        }
    };

    #[cfg(target_arch = "x86_64")]
    {
        // FAISS layout: split each byte into hi/lo nibbles, interleave with PERM0.
        for g in 0..n_byte_groups {
            let out_offset = g * BLOCK;
            for j in 0..16 {
                let ba = row(PERM0[j], g);
                let bb = row(PERM0[j] + 16, g);
                out[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                out[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Sequential layout: each byte stored as-is, vectors in order.
        for g in 0..n_byte_groups {
            let out_offset = g * BLOCK;
            for (lane, out_byte) in out[out_offset..out_offset + BLOCK].iter_mut().enumerate() {
                *out_byte = row(lane, g);
            }
        }
    }
}

/// Inverse of [`pack_block_rows`]: extract the [`BLOCK`] row-major
/// group-byte rows from one blocked block. `block` is
/// `n_byte_groups * BLOCK` bytes; `out` must be `BLOCK * n_byte_groups`
/// bytes and is fully overwritten (padding lanes come back as zero rows).
pub(crate) fn unpack_block_rows(block: &[u8], n_byte_groups: usize, out: &mut [u8]) {
    assert_eq!(block.len(), n_byte_groups * BLOCK);
    assert_eq!(out.len(), BLOCK * n_byte_groups);

    #[cfg(target_arch = "x86_64")]
    {
        for g in 0..n_byte_groups {
            let in_offset = g * BLOCK;
            for j in 0..16 {
                let hi = block[in_offset + j];
                let lo = block[in_offset + 16 + j];
                let ba = ((hi & 0x0F) << 4) | (lo & 0x0F);
                let bb = (hi & 0xF0) | (lo >> 4);
                out[PERM0[j] * n_byte_groups + g] = ba;
                out[(PERM0[j] + 16) * n_byte_groups + g] = bb;
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for g in 0..n_byte_groups {
            let in_offset = g * BLOCK;
            for lane in 0..BLOCK {
                out[lane * n_byte_groups + g] = block[in_offset + lane];
            }
        }
    }
}

/// Repack bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
pub fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let n_byte_groups = n_byte_groups(bits, dim);
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let block_bytes = n_byte_groups * BLOCK;

    // Step 1: Extract packed nibble bytes per vector per group
    let rows = group_bytes(packed_codes, n_vectors, bits, dim);

    // Step 2: Pack into platform-specific layout, block by block
    let mut blocked = vec![0u8; n_blocks * block_bytes];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        let n_rows = (n_vectors - base_vec).min(BLOCK);
        pack_block_rows(
            &rows[base_vec * n_byte_groups..(base_vec + n_rows) * n_byte_groups],
            n_rows,
            n_byte_groups,
            &mut blocked[block_idx * block_bytes..(block_idx + 1) * block_bytes],
        );
    }
    (blocked, n_blocks)
}

/// Repack 3-bit codes into two blocked arrays:
/// - sub_codes: 2-bit nibble format from planes 0,1
/// - plane2: packed bits blocked by 32 vectors
pub fn repack_3bit(
    packed_codes: &[u8],
    n_vectors: usize,
    dim: usize,
) -> (Vec<u8>, Vec<u8>, usize) {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = 3 * bytes_per_plane;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;

    let sub_byte_groups = dim / 4;
    let mut sub_codes = vec![0u8; n_blocks * sub_byte_groups * BLOCK];

    let plane2_byte_groups = bytes_per_plane;
    let mut plane2_blocked = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;

        for g in 0..sub_byte_groups {
            let out_offset = (block_idx * sub_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }

                let mut byte_val = 0u8;
                let dim_start = g * 4;
                for c in 0..4usize {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..2usize {
                        let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                        if plane_byte & mask != 0 { code |= 1 << p; }
                    }
                    byte_val |= code << ((3 - c) * 2);
                }
                sub_codes[out_offset + lane] = byte_val;
            }
        }

        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }
                plane2_blocked[out_offset + lane] = packed_codes[vec_idx * bytes_per_row + 2 * bytes_per_plane + g];
            }
        }
    }

    (sub_codes, plane2_blocked, n_blocks)
}

#[cfg(test)]
mod block_roundtrip_tests {
    use super::*;

    /// Deterministic pseudo-random bytes (xorshift) — no RNG dependency.
    fn pseudo_random_bytes(len: usize, mut state: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn pack_then_unpack_block_rows_is_identity() {
        for &(n_rows, n_byte_groups) in &[(BLOCK, 16usize), (BLOCK, 384), (7, 16), (1, 4)] {
            let rows = pseudo_random_bytes(n_rows * n_byte_groups, 42);
            let mut block = vec![0u8; n_byte_groups * BLOCK];
            pack_block_rows(&rows, n_rows, n_byte_groups, &mut block);

            let mut out = vec![0u8; BLOCK * n_byte_groups];
            unpack_block_rows(&block, n_byte_groups, &mut out);

            assert_eq!(&out[..n_rows * n_byte_groups], rows.as_slice());
            // Padding lanes must come back as zero rows.
            assert!(out[n_rows * n_byte_groups..].iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn unpack_inverts_repack_blocks() {
        // repack() and the per-block helpers must agree: unpacking every
        // block of repack's output reproduces group_bytes' rows.
        let bits = 4;
        let dim = 64;
        let n_vectors = 70; // 2 full blocks + 1 partial
        let packed = pseudo_random_bytes(n_vectors * dim * bits / 8, 7);

        let rows = group_bytes(&packed, n_vectors, bits, dim);
        let (blocked, n_blocks) = repack(&packed, n_vectors, bits, dim);
        let n_byte_groups = n_byte_groups(bits, dim);
        let block_bytes = n_byte_groups * BLOCK;
        assert_eq!(blocked.len(), n_blocks * block_bytes);

        let mut out = vec![0u8; BLOCK * n_byte_groups];
        for block_idx in 0..n_blocks {
            unpack_block_rows(
                &blocked[block_idx * block_bytes..(block_idx + 1) * block_bytes],
                n_byte_groups,
                &mut out,
            );
            let base_vec = block_idx * BLOCK;
            let lanes = (n_vectors - base_vec).min(BLOCK);
            assert_eq!(
                &out[..lanes * n_byte_groups],
                &rows[base_vec * n_byte_groups..(base_vec + lanes) * n_byte_groups],
            );
        }
    }
}
