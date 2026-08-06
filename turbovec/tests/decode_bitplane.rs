//! Does walking the bit-plane layout by BYTE beat walking it by dimension?
//!
//! Decode is 56% of an assignment at 768d (94.6 ms against the GEMM's 73.9 ms
//! in situ), so it is the larger half of a cost the whole coarse-quantizer
//! effort only attacks the smaller half of.
//!
//! `byte_in_plane` advances only every 8 dimensions, so a per-dimension loop
//! reloads each plane byte eight times: 3,072 loads per row at dim 768 / bits 4
//! for the 384 bytes that exist, each followed by a branch on an essentially
//! random bit. Hoisting the load and peeling the 8 dimensions it covers keeps
//! the arithmetic identical.
//!
//! Both loops run in ONE process, back to back, on the same data: this machine
//! may be busy, and a ratio measured under shared load is still meaningful
//! where two separate runs would not be. Equality is asserted, because a
//! faster decode that changes a single reconstructed value is a silent
//! corruption of every vector in the index.
//!
//! Run with: cargo test --release --test decode_bitplane -- --nocapture

use rayon::prelude::*;

/// The original: one pass per dimension, re-deriving the byte each time.
fn decode_by_dimension(
    packed: &[u8],
    n: usize,
    dim: usize,
    bits: usize,
    centroids: &[f32],
) -> Vec<f32> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut out = vec![0.0f32; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let packed_row = &packed[i * bytes_per_row..(i + 1) * bytes_per_row];
        for (d, value) in row.iter_mut().enumerate() {
            let byte_in_plane = d / 8;
            let bit_in_byte = 7 - (d % 8);
            let mut code = 0usize;
            for p in 0..bits {
                if packed_row[p * bytes_per_plane + byte_in_plane] >> bit_in_byte & 1 == 1 {
                    code |= 1 << p;
                }
            }
            *value = centroids[code];
        }
    });
    out
}

/// The replacement: one pass per byte, peeling its 8 dimensions.
fn decode_by_byte(
    packed: &[u8],
    n: usize,
    dim: usize,
    bits: usize,
    centroids: &[f32],
) -> Vec<f32> {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = bits * bytes_per_plane;
    let mut out = vec![0.0f32; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let packed_row = &packed[i * bytes_per_row..(i + 1) * bytes_per_row];
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
                row[d] = centroids[code];
            }
        }
    });
    out
}

#[test]
fn byte_major_decode_matches_and_is_faster() {
    let (dim, bits) = (768usize, 4usize);
    let centroids: Vec<f32> = (0..1 << bits).map(|c| c as f32 * 0.125 - 1.0).collect();
    for &n in &[10_000usize, 20_000] {
        let bytes_per_row = bits * (dim / 8);
        let mut packed = vec![0u8; n * bytes_per_row];
        let mut x = 0x243F6A8885A308D3u64;
        for b in packed.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 24) as u8;
        }

        // Warm both, so neither pays first-touch on its output buffer.
        let _ = decode_by_dimension(&packed[..bytes_per_row * 64], 64, dim, bits, &centroids);
        let _ = decode_by_byte(&packed[..bytes_per_row * 64], 64, dim, bits, &centroids);

        let t = std::time::Instant::now();
        let a = decode_by_dimension(&packed, n, dim, bits, &centroids);
        let old_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let b = decode_by_byte(&packed, n, dim, bits, &centroids);
        let new_ms = t.elapsed().as_secs_f64() * 1e3;

        assert_eq!(a, b, "byte-major decode must reproduce the values exactly");
        println!(
            "n={n} dim={dim} bits={bits}: by-dimension {old_ms:.1} ms, \
             by-byte {new_ms:.1} ms, {:.2}x",
            old_ms / new_ms
        );
    }
}
