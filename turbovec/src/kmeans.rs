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
    kmeans_ex(data, n, dim, k, iterations, seed, None)
}

/// `kmeans` with the inner assignment's parallelism chosen by the caller.
///
/// The ambient check reads false inside an installed pool, so a caller that
/// runs during a save gets a SERIAL Lloyd loop unless it says otherwise --
/// measured at ~800 ms for one `CoarseIndex::build` at nlist 1,561, against
/// a 244 ms budget for the assignment it exists to accelerate.
pub(crate) fn kmeans_ex(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iterations: usize,
    seed: u64,
    parallel: Option<bool>,
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
        let (new_assignments, distances) = assign_ex(data, n, dim, &centroids, k, parallel);
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

/// Assign each centroid to a super under a hard per-super capacity.
///
/// Greedy by regret: a centroid whose best and second-best supers are far
/// apart loses the most by being displaced, so it picks first. Each centroid
/// then takes its nearest super that still has room, which always exists
/// because total capacity exceeds `nlist`.
fn balanced_owners(
    centroids: &[f32],
    nlist: usize,
    dim: usize,
    supers: &[f32],
    n_super: usize,
) -> Vec<u32> {
    if n_super <= 1 {
        return vec![0u32; nlist];
    }
    let scores = crate::linalg::matmul_nt_ex(centroids, nlist, dim, supers, n_super, Some(true));
    let sup_sq: Vec<f32> = (0..n_super)
        .map(|s| supers[s * dim..(s + 1) * dim].iter().map(|v| v * v).sum())
        .collect();

    // Rank every super per centroid once; the greedy pass then walks the
    // preference list instead of re-scanning for the nearest open super.
    let mut prefs: Vec<Vec<u32>> = Vec::with_capacity(nlist);
    let mut regret: Vec<(f32, u32)> = Vec::with_capacity(nlist);
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_super);
    for c in 0..nlist {
        let row = &scores[c * n_super..(c + 1) * n_super];
        scratch.clear();
        scratch.extend((0..n_super).map(|s| (sup_sq[s] - 2.0 * row[s], s as u32)));
        scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        regret.push((scratch[1].0 - scratch[0].0, c as u32));
        prefs.push(scratch.iter().map(|&(_, s)| s).collect());
    }
    // Largest regret first: those centroids have the strongest preference.
    regret.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    // Experiment hook: the cap trades assignment accuracy against pruning, and
    // both sides are measurable, so the constant is swept rather than argued.
    let slack = std::env::var("TURBOVEC_MEMBER_SLACK")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(CoarseIndex::MEMBER_SLACK);
    let cap = ((nlist as f64 / n_super as f64) * slack).ceil() as usize;
    let cap = cap.max(1);
    let mut counts = vec![0usize; n_super];
    let mut owner = vec![0u32; nlist];
    for &(_, c) in &regret {
        let choice = prefs[c as usize]
            .iter()
            .copied()
            .find(|&s| counts[s as usize] < cap)
            .unwrap_or(prefs[c as usize][0]);
        counts[choice as usize] += 1;
        owner[c as usize] = choice;
    }
    owner
}

impl CoarseIndex {
    /// How far above the even share one super may go, as a multiple of
    /// `nlist / n_super`.
    ///
    /// The cap is what makes level-2 work bounded, so it cannot be loose; but
    /// a cap of exactly 1.0 would force the last centroids into supers they
    /// have no affinity for, which costs assignment accuracy for no speed. 1.5
    /// bounds the worst group at 1.5x the mean while leaving the geometry room
    /// to express itself.
    const MEMBER_SLACK: f64 = 1.5;

    /// Build the level over `centroids`, sized for the `probe` the caller will
    /// use.
    ///
    /// `n_super = sqrt(probe * nlist)`, NOT `sqrt(nlist)`. Level-2 work per row
    /// is `n_super + probe * nlist/n_super`, which is minimised at
    /// `sqrt(probe * nlist)` -- the naive `sqrt(nlist)` ignores that each row
    /// visits `probe` supers, and so builds far too few, far too large. The
    /// difference is not marginal: measured on real centroids at matched
    /// accuracy (~99% agreement with the exact scan), assigning a 10,000-row
    /// batch took
    ///
    ///   nlist    sqrt(nlist)    sqrt(8*nlist)
    ///    6,048       220.1 ms         112.7 ms   (1.95x)
    ///   12,617       365.0 ms         156.7 ms   (2.33x)
    ///
    /// so the constant was worth about 2x on its own, and the gap widens with
    /// nlist.
    pub fn build(centroids: &[f32], nlist: usize, dim: usize, probe: usize, seed: u64) -> Self {
        let probe = probe.max(1);
        let n_super = ((probe * nlist) as f64).sqrt().ceil().max(1.0) as usize;
        let n_super = n_super.min(nlist);
        // Few iterations on purpose: this clusters CENTROIDS, which are
        // already a summary, and a better super level buys accuracy the
        // `probe_super` dial buys more cheaply.
        // Explicitly parallel: `build` runs from inside the save's installed
        // pool, where the ambient check reads false and would leave the Lloyd
        // loop single-threaded.
        let (supers, _) = kmeans_ex(centroids, nlist, dim, n_super, 4, seed, Some(true));
        let n_super = supers.len() / dim.max(1);

        // Ownership is CAPACITY-CONSTRAINED, not nearest-super.
        //
        // Level-2 work is `sum over groups of rows_g * members_s`, which equals
        // the intended `n * nlist/n_super` only when members are balanced. Plain
        // Lloyd over the centroids of a clustered corpus does not balance them,
        // and the failure is not mild: measured at nlist 1,561 on a real index,
        // supers held between 1 and 582 centroids (mean 39), and because dense
        // supers also attract the most ROWS the two skews multiply. The result
        // did 11.74 GMAC against the flat scan's 11.99 GMAC -- 98%, i.e. the
        // hierarchy pruned nothing while still paying for two levels. A random
        // -centroid benchmark cannot see this, because random centroids are
        // balanced by construction (measured 13..27 members, 0.44x).
        //
        // A hard cap restores the bound the design assumes. Centroids are
        // placed in order of REGRET -- how much worse their second choice is --
        // so the ones with a real preference choose first and the indifferent
        // ones absorb the displacement.
        let owner = balanced_owners(centroids, nlist, dim, &supers, n_super);

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

    /// Centroids owned by each super, in super order.
    ///
    /// Exposed because the hierarchy's SPEED depends entirely on this being
    /// balanced, and a skewed partition passes every accuracy assertion while
    /// pruning nothing: level-2 work is `sum_g rows_g * members_s`, so one
    /// super owning a third of the centroids -- and attracting rows in
    /// proportion -- reproduces the flat scan at two levels' cost.
    pub fn member_counts(&self) -> Vec<usize> {
        (0..self.n_super)
            .map(|s| (self.member_offsets[s + 1] - self.member_offsets[s]) as usize)
            .collect()
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

        // Centroids are regrouped ONCE per call into super-major order, so each
        // of the `probe` rounds SLICES its group's operand instead of gathering
        // it. Gathering per round cost `probe * nlist * dim` of copying; this
        // costs `nlist * dim` once. The buffer is transient rather than owned
        // by the index, because owning it would double resident centroid RAM
        // (300 MB at nlist 97,656 x 768d) for a build-path scratch.
        let mut grouped = Vec::with_capacity(self.members.len() * dim);
        for &cid in &self.members {
            grouped.extend_from_slice(&centroids[cid as usize * dim..(cid as usize + 1) * dim]);
        }
        // Norms in GROUPED order, so the reduce indexes the same slice as the
        // GEMM rather than hopping back through `members`.
        let grouped_sq: Vec<f32> = grouped
            .chunks_exact(dim)
            .map(|c| c.iter().map(|v| v * v).sum())
            .collect();

        // Level 2, one ROUND per probed super. Within a round, rows are grouped
        // by which super they are visiting, so the scoring stays a GEMM per
        // group rather than a per-row gather -- the reason a hierarchy beats a
        // graph for batch assignment. Rounds keep a running best.
        //
        // Parallelism is ACROSS groups with a SERIAL GEMM inside each, not one
        // parallel GEMM per group. Measured at n=20k / nlist=16,384: the
        // per-group multiplies are ~156 rows each and 1,024 of them ran, so a
        // parallel GEMM per group spent its time in fork/join -- 350 ms against
        // a 135 ms FLOP budget. Across groups every core gets whole groups,
        // and the gather and reduce parallelise with them for free.
        let mut out = vec![0u32; n];
        let mut best_score = vec![f32::INFINITY; n];
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut bounds: Vec<(usize, usize, usize)> = Vec::new();
        for p in 0..probe {
            order.sort_unstable_by_key(|&i| ranked[i as usize * probe + p]);

            bounds.clear();
            let mut pos = 0usize;
            while pos < n {
                let s = ranked[order[pos] as usize * probe + p] as usize;
                let mut end = pos;
                while end < n && ranked[order[end] as usize * probe + p] as usize == s {
                    end += 1;
                }
                let lo = self.member_offsets[s] as usize;
                let hi = self.member_offsets[s + 1] as usize;
                if hi > lo {
                    bounds.push((s, pos, end));
                }
                pos = end;
            }

            // Rows are disjoint across groups within a round (each row visits
            // exactly one super per round), so each group can own its slice of
            // the answer and the merge afterwards is a straight copy.
            let per_group: Vec<Vec<(u32, f32)>> = bounds
                .par_iter()
                .map(|&(s, gstart, gend)| {
                    let rows = &order[gstart..gend];
                    let lo = self.member_offsets[s] as usize;
                    let hi = self.member_offsets[s + 1] as usize;
                    let members = hi - lo;
                    let b = &grouped[lo * dim..hi * dim];
                    let sq = &grouped_sq[lo..hi];

                    let mut a = Vec::with_capacity(rows.len() * dim);
                    for &i in rows {
                        a.extend_from_slice(&data[i as usize * dim..(i as usize + 1) * dim]);
                    }
                    let prod = crate::linalg::matmul_nt_ex(
                        &a,
                        rows.len(),
                        dim,
                        b,
                        members,
                        Some(false),
                    );

                    let mut best = Vec::with_capacity(rows.len());
                    for r in 0..rows.len() {
                        let row = &prod[r * members..(r + 1) * members];
                        let mut bs = f32::INFINITY;
                        let mut bc = 0u32;
                        for m in 0..members {
                            let score = sq[m] - 2.0 * row[m];
                            if score < bs {
                                bs = score;
                                bc = self.members[lo + m];
                            }
                        }
                        best.push((bc, bs));
                    }
                    best
                })
                .collect();

            for (g, &(_, gstart, gend)) in bounds.iter().enumerate() {
                for (r, &i) in order[gstart..gend].iter().enumerate() {
                    let (cid, score) = per_group[g][r];
                    if score < best_score[i as usize] {
                        best_score[i as usize] = score;
                        out[i as usize] = cid;
                    }
                }
            }
        }
        out
    }
}
