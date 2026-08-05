//! What does `matmul_nt` achieve at the SHAPE assignment actually uses?
//!
//! `assign` measured ~25 GMAC/s in situ against a ~290 GF/s figure recorded
//! for this primitive earlier in the project. That gap would mean a large
//! constant-factor win sitting on the floor -- but the historical number was
//! taken on a quiet machine and a different shape, so comparing to it proves
//! nothing on its own. This runs the same primitive, at assignment's shape, on
//! whatever machine is running the test.
//!
//! Run with: cargo test --release --test assign_gemm_bench -- --nocapture

use turbovec::linalg::matmul_nt;

#[test]
fn matmul_at_assignment_shape() {
    let dim = 768;
    for (rows, nlist) in [(20_000usize, 195usize), (20_000, 390), (20_000, 781)] {
        // Unit-norm rows, like decoded vectors -- patterned data could differ
        // in ways (denormals, cache behaviour) that flatter the benchmark.
        let mk = |n: usize, seed: u64| -> Vec<f32> {
            let mut v = vec![0.0f32; n * dim];
            let mut x = seed;
            for row in v.chunks_mut(dim) {
                for e in row.iter_mut() {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *e = ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                }
                let nrm = row.iter().map(|&z| z * z).sum::<f32>().sqrt().max(1e-12);
                for e in row.iter_mut() {
                    *e /= nrm;
                }
            }
            v
        };
        let a = mk(rows, 12345);
        let b = mk(nlist, 999);
        // Warm the allocator and caches so the first shape is not penalised.
        let _ = matmul_nt(&a[..dim * 64], 64, dim, &b, nlist);

        let t = std::time::Instant::now();
        let c = matmul_nt(&a, rows, dim, &b, nlist);
        let el = t.elapsed().as_secs_f64();
        std::hint::black_box(&c);

        let gmac = (rows * nlist * dim) as f64 / 1e9;
        println!(
            "rows={rows} nlist={nlist} dim={dim}: {:.1} ms  {:.1} GMAC/s  ({:.0} GFLOP/s)",
            el * 1e3,
            gmac / el,
            2.0 * gmac / el,
        );
    }
}
