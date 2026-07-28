//! Small k-means used for partitioning the disk index.
//!
//! Plain Lloyd iterations with random initialization, GEMM-based
//! assignment, and farthest-point reseeding for emptied clusters.
//! Deterministic for a given seed. This is deliberately minimal: partition
//! quality only needs to be roughly balanced — the disk index's split/merge
//! loop repairs imbalance incrementally, so a fancy init buys little.

use ndarray::ArrayView2;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Assign each of `n` vectors to its nearest centroid by squared L2.
/// Returns (assignments, squared distances to the assigned centroid).
pub(crate) fn assign(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(data.len(), n * dim);
    assert_eq!(centroids.len(), k * dim);
    assert!(k > 0);

    // argmin_c ||x - c||^2 = argmin_c (||c||^2 - 2 x . c); ||x||^2 is added
    // back only for the returned distances.
    let centroid_sq_norms: Vec<f32> = (0..k)
        .map(|c| {
            centroids[c * dim..(c + 1) * dim]
                .iter()
                .map(|&v| v * v)
                .sum()
        })
        .collect();

    let data_mat = ArrayView2::from_shape((n, dim), data).unwrap();
    let centroid_mat = ArrayView2::from_shape((k, dim), centroids).unwrap();
    let products = data_mat.dot(&centroid_mat.t()); // (n, k)

    let mut assignments = vec![0u32; n];
    let mut distances = vec![0.0f32; n];
    assignments
        .par_iter_mut()
        .zip(distances.par_iter_mut())
        .enumerate()
        .for_each(|(i, (assignment, distance))| {
            let row = products.row(i);
            let mut best = 0usize;
            let mut best_score = f32::INFINITY;
            for c in 0..k {
                let score = centroid_sq_norms[c] - 2.0 * row[c];
                if score < best_score {
                    best_score = score;
                    best = c;
                }
            }
            *assignment = best as u32;
            let x_sq: f32 = data[i * dim..(i + 1) * dim].iter().map(|&v| v * v).sum();
            *distance = (x_sq + best_score).max(0.0);
        });

    (assignments, distances)
}

/// Lloyd k-means. Returns (centroids `k * dim`, assignments `n`).
/// `k` is clamped to `n`; deterministic for a given `seed`.
pub(crate) fn kmeans(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iterations: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u32>) {
    assert_eq!(data.len(), n * dim);
    assert!(n > 0);
    let k = k.clamp(1, n);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(k * dim);
    for picked in sample(&mut rng, n, k) {
        centroids.extend_from_slice(&data[picked * dim..(picked + 1) * dim]);
    }

    let mut assignments = vec![0u32; n];
    for _ in 0..iterations {
        let (new_assignments, distances) = assign(data, n, dim, &centroids, k);
        assignments = new_assignments;

        let mut sums = vec![0.0f64; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += data[i * dim + d] as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Reseed an emptied cluster at the point farthest from its
                // assigned centroid, the standard fix for collapse.
                let farthest = distances
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i);
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&data[farthest * dim..(farthest + 1) * dim]);
                continue;
            }
            for d in 0..dim {
                centroids[c * dim + d] = (sums[c * dim + d] / counts[c] as f64) as f32;
            }
        }
    }

    let (final_assignments, _) = assign(data, n, dim, &centroids, k);
    (centroids, final_assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated clusters in 8-d; k-means must recover them.
    fn clustered_data(per_cluster: usize) -> (Vec<f32>, usize) {
        let dim = 8;
        let anchors: [[f32; 8]; 3] = [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mut data = Vec::new();
        let mut state = 5u64;
        for anchor in &anchors {
            for _ in 0..per_cluster {
                for &a in anchor {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let noise = ((state >> 11) as f32 / (1u64 << 53) as f32) - 0.5;
                    data.push(a + noise);
                }
            }
        }
        (data, dim)
    }

    #[test]
    fn kmeans_recovers_separated_clusters() {
        let per_cluster = 50;
        let (data, dim) = clustered_data(per_cluster);
        let n = 3 * per_cluster;
        let (_, assignments) = kmeans(&data, n, dim, 3, 10, 42);

        // All members of a ground-truth cluster share one label, and the
        // three labels are distinct.
        let mut labels = Vec::new();
        for cluster in 0..3 {
            let first = assignments[cluster * per_cluster];
            for i in 0..per_cluster {
                assert_eq!(assignments[cluster * per_cluster + i], first);
            }
            labels.push(first);
        }
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn assign_matches_brute_force() {
        let (data, dim) = clustered_data(20);
        let n = 60;
        let centroids: Vec<f32> = data[..3 * dim].to_vec(); // arbitrary 3 points
        let (assignments, _) = assign(&data, n, dim, &centroids, 3);
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..3 {
                let d: f32 = (0..dim)
                    .map(|j| (data[i * dim + j] - centroids[c * dim + j]).powi(2))
                    .sum();
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assert_eq!(assignments[i] as usize, best, "row {i}");
        }
    }
}
