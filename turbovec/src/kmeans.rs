//! Small k-means used for partitioning the disk index.
//!
//! Plain Lloyd iterations with random initialization, GEMM-based
//! assignment, and farthest-point reseeding for emptied clusters.
//! Deterministic for a given seed. This is deliberately minimal: partition
//! quality only needs to be roughly balanced — the disk index's split/merge
//! loop repairs imbalance incrementally, so a fancy init buys little.

use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Assign each of `n` vectors to its nearest centroid by squared L2.
/// Returns (assignments, squared distances to the assigned centroid).
use std::sync::atomic::{AtomicU64, Ordering as AtOrd};

/// Wall microseconds inside the assignment GEMM, summed over every call.
///
/// This is the multiply that removing BLAS made slowest, and locating it took
/// three wrong guesses, so it stays measured.
static GEMM_US: AtomicU64 = AtomicU64::new(0);

pub(crate) fn assign_gemm_micros() -> u64 {
    GEMM_US.load(AtOrd::Relaxed)
}

pub(crate) fn assign(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assign_ex(data, n, dim, centroids, k, None)
}

/// `assign` with the GEMM's parallelism chosen by the caller. Ingest assigns
/// one large batch from the top of a flush and wants every core; maintenance
/// assigns per partition from inside a `par_iter` and must stay serial.
pub(crate) fn assign_ex(
    data: &[f32],
    n: usize,
    dim: usize,
    centroids: &[f32],
    k: usize,
    parallel: Option<bool>,
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

    let t_gemm = std::time::Instant::now();
    let products = crate::linalg::matmul_nt_ex(data, n, dim, centroids, k, parallel); // (n, k)
    let gemm_us = t_gemm.elapsed().as_micros() as u64;
    GEMM_US.fetch_add(gemm_us, AtOrd::Relaxed);

    let mut assignments = vec![0u32; n];
    let mut distances = vec![0.0f32; n];
    assignments
        .par_iter_mut()
        .zip(distances.par_iter_mut())
        .enumerate()
        .for_each(|(i, (assignment, distance))| {
            let row = &products[i * k..(i + 1) * k];
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

/// Two-level coarse quantizer over the partition centroids.
///
/// Assignment scores every row against every centroid, so per-row cost is
/// O(nlist·dim) and, because nlist grows linearly with N at a fixed partition
/// size, bulk construction is O(N²). That is the shape every billion-scale
/// system avoids by making the coarse quantizer an index rather than a linear
/// scan (FAISS `IVF..._HNSW`, SPANN, DiskANN).
///
/// A GEMM hierarchy is used here rather than a graph, because the workload is
/// BATCH assignment of many rows, not one lookup at a time. Both levels stay
/// dense matrix multiplies — SIMD, cache-friendly, amortized across the batch —
/// where a graph would give one pointer-chasing walk per row.
///
/// Cost per row falls from `nlist` to `n_super + probe·members ≈ 2·√nlist`,
/// i.e. a factor of about `√nlist / 2`: ~14x at nlist 781, ~39x at 6,300,
/// ~156x at 97,656. RAM is `√nlist·dim·4` — 958 KB at nlist 97,656 × 768d,
/// against 300 MB of centroids — and the search path is untouched.
///
/// The cost is that assignment becomes APPROXIMATE: a row can land in a
/// partition that is not its true nearest. `probe_super` trades that back, and
/// LIRE's reassignment pass corrects residual error over time.
pub struct CoarseIndex {
    /// nlist this was built for; rebuilt when the partition count has moved
    /// materially, not on every centroid touch.
    pub built_for: usize,
    n_super: usize,
    dim: usize,
    supers: Vec<f32>,
    /// Centroid ids owned by each super, flattened with offsets so the hot
    /// path indexes a slice rather than chasing a `Vec<Vec<_>>`.
    member_offsets: Vec<u32>,
    members: Vec<u32>,
}

impl CoarseIndex {
    pub fn build(centroids: &[f32], nlist: usize, dim: usize, seed: u64) -> Self {
        let n_super = (nlist as f64).sqrt().ceil().max(1.0) as usize;
        let n_super = n_super.min(nlist);
        // Few iterations on purpose: this clusters CENTROIDS, which are
        // already a summary, and a better super level buys accuracy the
        // `probe_super` dial buys more cheaply.
        let (supers, owner) = kmeans(centroids, nlist, dim, n_super, 4, seed);
        let n_super = supers.len() / dim.max(1);

        let mut counts = vec![0u32; n_super];
        for &s in &owner {
            counts[s as usize] += 1;
        }
        let mut member_offsets = Vec::with_capacity(n_super + 1);
        let mut acc = 0u32;
        for &c in &counts {
            member_offsets.push(acc);
            acc += c;
        }
        member_offsets.push(acc);
        let mut cursor = member_offsets.clone();
        let mut members = vec![0u32; nlist];
        for (cid, &s) in owner.iter().enumerate() {
            let slot = &mut cursor[s as usize];
            members[*slot as usize] = cid as u32;
            *slot += 1;
        }
        Self {
            built_for: nlist,
            n_super,
            dim,
            supers,
            member_offsets,
            members,
        }
    }

    /// Assign `n` rows to centroids through the hierarchy.
    ///
    /// Rows are grouped by their chosen super before the second level so that
    /// stage stays a GEMM per group rather than a per-row gather.
    pub fn assign(
        &self,
        data: &[f32],
        n: usize,
        centroids: &[f32],
        probe_super: usize,
    ) -> Vec<u32> {
        let dim = self.dim;
        if n == 0 || self.n_super == 0 {
            return vec![0u32; n];
        }
        let probe = probe_super.clamp(1, self.n_super);

        // Level 1: every row against every super. One GEMM.
        let sup_scores = crate::linalg::matmul_nt_ex(
            data,
            n,
            dim,
            &self.supers,
            self.n_super,
            Some(true),
        );
        let sup_sq: Vec<f32> = (0..self.n_super)
            .map(|s| self.supers[s * dim..(s + 1) * dim].iter().map(|v| v * v).sum())
            .collect();

        // Rank each row's supers, keeping the top `probe`.
        //
        // probe=1 is not enough: a row near a super boundary has its true
        // nearest centroid in the NEIGHBOURING super, and measured agreement
        // with exact assignment was only 37% at nlist 256. Probing several
        // supers is the accuracy dial, and it stays cheap because each round
        // is still one GEMM per group.
        let mut ranked = vec![0u32; n * probe];
        {
            let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(self.n_super);
            for i in 0..n {
                let row = &sup_scores[i * self.n_super..(i + 1) * self.n_super];
                scratch.clear();
                scratch.extend((0..self.n_super).map(|s| (sup_sq[s] - 2.0 * row[s], s as u32)));
                if probe < self.n_super {
                    scratch.select_nth_unstable_by(probe - 1, |a, b| a.0.total_cmp(&b.0));
                }
                scratch[..probe].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                for p in 0..probe {
                    ranked[i * probe + p] = scratch[p].1;
                }
            }
        }

        let cent_sq: Vec<f32> = (0..centroids.len() / dim)
            .map(|c| centroids[c * dim..(c + 1) * dim].iter().map(|v| v * v).sum())
            .collect();

        // Level 2, one ROUND per probed super. Within a round, rows are grouped
        // by which super they are visiting, so the scoring stays a GEMM per
        // group rather than a per-row gather -- the reason a hierarchy beats a
        // graph for batch assignment. Rounds keep a running best.
        let mut out = vec![0u32; n];
        let mut best_score = vec![f32::INFINITY; n];
        let mut order: Vec<u32> = (0..n as u32).collect();
        for p in 0..probe {
            order.sort_unstable_by_key(|&i| ranked[i as usize * probe + p]);
            let mut pos = 0usize;
            while pos < n {
                let s = ranked[order[pos] as usize * probe + p] as usize;
                let mut end = pos;
                while end < n && ranked[order[end] as usize * probe + p] as usize == s {
                    end += 1;
                }
                let rows = &order[pos..end];
                let lo = self.member_offsets[s] as usize;
                let hi = self.member_offsets[s + 1] as usize;
                let member_ids = &self.members[lo..hi];
                if member_ids.is_empty() {
                    pos = end;
                    continue;
                }

                let mut a = Vec::with_capacity(rows.len() * dim);
                for &i in rows {
                    a.extend_from_slice(&data[i as usize * dim..(i as usize + 1) * dim]);
                }
                let mut b = Vec::with_capacity(member_ids.len() * dim);
                for &cid in member_ids {
                    b.extend_from_slice(&centroids[cid as usize * dim..(cid as usize + 1) * dim]);
                }
                let prod = crate::linalg::matmul_nt_ex(
                    &a,
                    rows.len(),
                    dim,
                    &b,
                    member_ids.len(),
                    Some(true),
                );
                for (r, &i) in rows.iter().enumerate() {
                    let row = &prod[r * member_ids.len()..(r + 1) * member_ids.len()];
                    for (m, &cid) in member_ids.iter().enumerate() {
                        let score = cent_sq[cid as usize] - 2.0 * row[m];
                        if score < best_score[i as usize] {
                            best_score[i as usize] = score;
                            out[i as usize] = cid;
                        }
                    }
                }
                pos = end;
            }
        }
        out
    }
}
