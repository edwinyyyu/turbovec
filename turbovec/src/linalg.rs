//! The linear algebra this crate needs, with no external system dependency.
//!
//! Replaces `ndarray`'s BLAS backend (Accelerate on macOS, OpenBLAS on Linux)
//! and `faer`. Both are fine libraries; neither is something a phone-class
//! deployment can rely on being present, and a build that silently falls back
//! to a slow path on the platforms where the link fails is worse than one that
//! never had the option.
//!
//! Two operations are needed:
//!
//! * [`matmul_nt`] — `C[m,n] = A[m,k] · B[n,k]ᵀ`, both operands row-major.
//!   This is the shape almost every call site already wanted (routing, encode,
//!   query rotation), and it is the *friendly* one: `C[i][j]` is the dot
//!   product of two contiguous rows, so the reduction axis is contiguous in
//!   both operands and neither has to be transposed.
//! * [`orthonormal_from_gaussian`] — Householder QR of a seeded Gaussian,
//!   which is how the rotation matrix is built.
//!
//! Neither uses `unsafe`, platform intrinsics, or a build script. The
//! micro-kernel is written so LLVM's autovectoriser has an obvious job: fixed
//! small loop bounds over contiguous packed panels.

use rayon::prelude::*;

/// Rows of `C` per micro-kernel invocation.
const MR: usize = 4;
/// Columns of `C` per micro-kernel invocation.
///
/// `MR * NR` accumulators have to live in registers, and the tile has to be
/// big enough to hide FMA latency. 4x8 is not: measured on the reassignment
/// shape it reached 76 GF/s, where 4x16 and 8x8 both reach ~290 GF/s. Thirty-two
/// accumulators leave the pipeline waiting on its own results; sixty-four keep
/// it fed.
const NR: usize = 16;
/// Minimum rows of `C` per parallel task.
const M_BLOCK: usize = 64;
/// Target multiply-accumulates per parallel task. A fixed row count per task
/// starves the scheduler when `n` is small -- at n=8 a 64-row task is only
/// ~49k operations, and rayon's hand-off costs more than the work -- so the
/// row count is chosen from the shape instead.
const TASK_WORK: usize = 1 << 18;
/// Below this much work, thread hand-off costs more than the whole multiply.
const PARALLEL_THRESHOLD: usize = 1 << 16;

/// Should this multiply spread itself across threads?
///
/// Only if it is big enough AND we are not already inside a rayon worker.
/// Maintenance runs `par_iter` over partitions and calls a GEMM per partition;
/// parallelising that GEMM again nests rayon inside rayon, and the workers
/// spend their time in the scheduler rather than in the kernel. Measured on a
/// 400k first save: maintenance 2.12 s with nesting, against 0.97 s for the
/// BLAS build it replaced -- and `kmeans::assign`, which is the call site
/// inside the parallel reassignment loop, accounted for 88% of that gap.
///
/// The split cascade calls in from the main thread and still parallelises.
fn should_parallelise(work: usize) -> bool {
    work >= PARALLEL_THRESHOLD && rayon::current_thread_index().is_none()
}

/// `C[m,n] = A[m,k] · B[n,k]ᵀ` for row-major `a` (m×k) and `b` (n×k).
///
/// Note `b` is indexed by ROW: `C[i][j] = dot(a_row_i, b_row_j)`. Callers that
/// think in terms of "multiply by Bᵀ" and callers that think "score every row
/// of A against every row of B" want the same function.
pub fn matmul_nt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a is not m*k");
    assert_eq!(b.len(), n * k, "b is not n*k");
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return c;
    }

    // Small operands take the direct path. The packed kernel below computes a
    // fixed MR x NR tile whatever the real shape, and pays to pack B before it
    // starts, so on a short or narrow problem it does mostly wasted work:
    //
    //   * a 2-means split assigns against n = 2 centroids, so 6 of every 8
    //     accumulator columns are padding -- and splitting is most of what
    //     maintenance does. Measured: maintenance 0.98 s -> 2.15 s when this
    //     path was missing, against a GEMM that is otherwise 0.74x of BLAS.
    //   * a single query rotates with m = 1, where packing the dim x dim
    //     rotation costs as much as the multiply it is preparing for.
    //
    // Here each output element is one dot product over contiguous memory,
    // which vectorises on its own and needs no packing at all.
    if m < MR || n < NR {
        direct(a, m, k, b, n, &mut c);
        return c;
    }

    // Pack B once for every A block to reuse: for each block of NR rows,
    // interleave so the NR values sharing a k index are adjacent. That turns
    // the inner loop's B access into one contiguous vector load.
    let n_blocks = n.div_ceil(NR);
    let mut bp = vec![0.0f32; n_blocks * NR * k];
    for jb in 0..n_blocks {
        let j0 = jb * NR;
        let rows = NR.min(n - j0);
        let panel = &mut bp[jb * NR * k..(jb + 1) * NR * k];
        for kk in 0..k {
            for j in 0..rows {
                panel[kk * NR + j] = b[(j0 + j) * k + kk];
            }
        }
    }

    let per_row = (n * k).max(1);
    let mblock = M_BLOCK
        .max(TASK_WORK / per_row)
        .next_multiple_of(MR)
        .min(m.max(1));
    let work = m * n * k;
    if should_parallelise(work) {
        c.par_chunks_mut(mblock * n)
            .enumerate()
            .for_each(|(blk, c_block)| {
                let i0 = blk * mblock;
                let rows = (m - i0).min(mblock);
                block(a, i0, rows, k, &bp, n, n_blocks, c_block);
            });
    } else {
        for (blk, c_block) in c.chunks_mut(mblock * n).enumerate() {
            let i0 = blk * mblock;
            let rows = (m - i0).min(mblock);
            block(a, i0, rows, k, &bp, n, n_blocks, c_block);
        }
    }
    c
}

/// Unpacked path: every output element is a dot product of two contiguous
/// rows. Used when the packed kernel's fixed tile would be mostly padding.
fn direct(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    let row_work = (n * k).max(1);
    // Rows per task, not one task per row: at n=2 a single output row is 192
    // multiply-accumulates, and handing that to a worker costs far more than
    // doing it.
    let rows_per_task = (TASK_WORK / row_work).max(1);
    let each = |(t, out): (usize, &mut [f32])| {
        for (r, out) in out.chunks_mut(n).enumerate() {
            let i = t * rows_per_task + r;
            let arow = &a[i * k..(i + 1) * k];
            for (j, o) in out.iter_mut().enumerate() {
                let brow = &b[j * k..(j + 1) * k];
                // Four independent accumulators: the reduction is associative
                // enough for f32 here and this breaks the dependency chain that
                // would otherwise serialise one FMA per cycle.
                let mut s = [0.0f32; 4];
                let chunks = k / 4;
                for ci in 0..chunks {
                    let ab = &arow[ci * 4..ci * 4 + 4];
                    let bb = &brow[ci * 4..ci * 4 + 4];
                    for l in 0..4 {
                        s[l] += ab[l] * bb[l];
                    }
                }
                let mut tail = 0.0f32;
                for idx in chunks * 4..k {
                    tail += arow[idx] * brow[idx];
                }
                *o = s[0] + s[1] + s[2] + s[3] + tail;
            }
        }
    };
    if m * row_work >= PARALLEL_THRESHOLD {
        c.par_chunks_mut(rows_per_task * n)
            .enumerate()
            .for_each(each);
    } else {
        c.chunks_mut(rows_per_task * n).enumerate().for_each(each);
    }
}

/// One `M_BLOCK`-tall horizontal strip of `C`.
fn block(
    a: &[f32],
    i0: usize,
    rows: usize,
    k: usize,
    bp: &[f32],
    n: usize,
    n_blocks: usize,
    c: &mut [f32],
) {
    let mut ap = vec![0.0f32; MR * k];
    let mut ib = 0;
    while ib < rows {
        let mr = MR.min(rows - ib);
        // Pack MR rows of A the same way, so the inner loop broadcasts from
        // contiguous memory instead of striding by k.
        for kk in 0..k {
            for i in 0..mr {
                ap[kk * MR + i] = a[(i0 + ib + i) * k + kk];
            }
        }
        for jb in 0..n_blocks {
            let j0 = jb * NR;
            let nr = NR.min(n - j0);
            let panel = &bp[jb * NR * k..(jb + 1) * NR * k];
            let mut acc = [[0.0f32; NR]; MR];
            for kk in 0..k {
                let av = &ap[kk * MR..kk * MR + MR];
                let bv = &panel[kk * NR..kk * NR + NR];
                // Fixed bounds: the compiler unrolls both and emits FMAs.
                for i in 0..MR {
                    let x = av[i];
                    for j in 0..NR {
                        acc[i][j] += x * bv[j];
                    }
                }
            }
            for i in 0..mr {
                let row = &mut c[(ib + i) * n..(ib + i + 1) * n];
                row[j0..j0 + nr].copy_from_slice(&acc[i][..nr]);
            }
        }
        ib += MR;
    }
}

/// `C[m,n] = A[m,k] · B[k,n]` for row-major `a` and `b`.
///
/// Transposes `b` and defers to [`matmul_nt`]. Only used where `b` is a small
/// square rotation matrix, so the transpose is cheap next to the multiply;
/// anything hot should be expressed as `matmul_nt` directly.
pub fn matmul_nn(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(b.len(), k * n, "b is not k*n");
    let mut bt = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            bt[j * k + i] = b[i * n + j];
        }
    }
    matmul_nt(a, m, k, &bt, n)
}

/// Orthonormalise a column-major `n`×`n` matrix by Householder QR, returning
/// `Q` in column-major order with the sign convention `Q · diag(sign(diag R))`.
///
/// That convention is what makes the result a deterministic function of the
/// input rather than one of `2^n` sign-equivalent answers, so it is what keeps
/// the rotation matrix — and therefore every encoded vector — stable.
///
/// `f64` throughout: the caller narrows to `f32` only at the end, and losing
/// orthogonality here would show up as quantization error later.
pub fn orthonormal_from_gaussian(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    let at = |m: &[f64], i: usize, j: usize| m[i + j * n];

    // Householder vectors, one per column, each stored full-length with
    // leading zeros so the back-application below can index uniformly.
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r_diag = vec![0.0f64; n];

    for j in 0..n {
        let mut norm = 0.0f64;
        for i in j..n {
            let x = at(&a, i, j);
            norm += x * x;
        }
        norm = norm.sqrt();
        let x0 = at(&a, j, j);
        // alpha carries the OPPOSITE sign to x0, which avoids cancellation
        // when forming v and fixes R[j][j] = alpha.
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        r_diag[j] = alpha;

        let mut v = vec![0.0f64; n];
        if norm == 0.0 {
            vs.push(v);
            continue;
        }
        for i in j..n {
            v[i] = a[i + j * n];
        }
        v[j] -= alpha;
        let vnorm = v[j..].iter().map(|x| x * x).sum::<f64>().sqrt();
        if vnorm > 0.0 {
            for x in v[j..].iter_mut() {
                *x /= vnorm;
            }
        }

        // A[j:, j:] -= 2 v (vᵀ A[j:, j:]). Column-major, so each column of the
        // trailing submatrix is contiguous.
        for c in j..n {
            let col = &mut a[c * n..(c + 1) * n];
            let mut dot = 0.0f64;
            for i in j..n {
                dot += v[i] * col[i];
            }
            let two_dot = 2.0 * dot;
            for i in j..n {
                col[i] -= two_dot * v[i];
            }
        }
        vs.push(v);
    }

    // Q = H_0 H_1 ... H_{n-1} applied to the identity, accumulated backwards.
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        q[i + i * n] = 1.0;
    }
    for j in (0..n).rev() {
        let v = &vs[j];
        for c in j..n {
            let col = &mut q[c * n..(c + 1) * n];
            let mut dot = 0.0f64;
            for i in j..n {
                dot += v[i] * col[i];
            }
            let two_dot = 2.0 * dot;
            for i in j..n {
                col[i] -= two_dot * v[i];
            }
        }
    }

    // Sign correction, so the answer does not depend on which of the two
    // valid Householder reflections the loop above happened to pick.
    for j in 0..n {
        if r_diag[j] < 0.0 {
            for i in 0..n {
                q[i + j * n] = -q[i + j * n];
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_nt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        // xorshift, so the test carries no dependency of its own
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// Every shape that exercises an edge: m and n not multiples of MR/NR,
    /// k of 1, and sizes either side of the parallel threshold.
    #[test]
    fn matmul_nt_matches_the_definition() {
        for &(m, k, n) in &[
            (1, 1, 1),
            (3, 5, 7),
            (4, 8, 8),
            (5, 96, 13),
            (64, 96, 65),
            (129, 33, 17),
            (200, 128, 200),
        ] {
            let a = seeded(m * k, 1);
            let b = seeded(n * k, 2);
            let got = matmul_nt(&a, m, k, &b, n);
            let want = naive_nt(&a, m, k, &b, n);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                    "({m},{k},{n}) element {i}: {g} vs {w}",
                );
            }
        }
    }

    #[test]
    fn matmul_nn_matches_matmul_nt_on_a_transposed_operand() {
        let (m, k, n) = (17, 33, 21);
        let a = seeded(m * k, 3);
        let b = seeded(k * n, 4); // k x n row-major
        let mut bt = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let got = matmul_nn(&a, m, k, &b, n);
        let want = matmul_nt(&a, m, k, &bt, n);
        assert_eq!(got, want);
    }

    /// The property the rotation actually depends on. A matrix that is merely
    /// close to orthogonal would leave the rotated coordinates off their
    /// assumed Beta distribution and quietly cost recall.
    #[test]
    fn householder_qr_produces_an_orthonormal_matrix() {
        for &n in &[1usize, 2, 7, 32, 96] {
            let a: Vec<f64> = seeded(n * n, 5).into_iter().map(|x| x as f64).collect();
            let q = orthonormal_from_gaussian(a, n);
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0f64;
                    for r in 0..n {
                        dot += q[r + i * n] * q[r + j * n];
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-9,
                        "n={n}: columns {i}.{j} dot to {dot}, want {want}",
                    );
                }
            }
        }
    }

    /// Deterministic: the same input must give the same matrix on every run,
    /// or stored indexes stop decoding.
    #[test]
    fn householder_qr_is_deterministic() {
        let a: Vec<f64> = seeded(64 * 64, 6).into_iter().map(|x| x as f64).collect();
        let first = orthonormal_from_gaussian(a.clone(), 64);
        let second = orthonormal_from_gaussian(a, 64);
        assert_eq!(first, second);
    }
}
