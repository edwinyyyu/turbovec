//! Where does the two-level assign lose to the flat scan?
//!
//! The hierarchy does ~10x fewer FLOPs at nlist 8,725 and measured 23% SLOWER
//! in situ, so the loss is constant factors, not arithmetic. This runs both at
//! the shapes a real build hits and prints the ratio, so a fix can be checked
//! against a number instead of a theory.
//!
//! Run with: cargo test --release --test coarse_bench -- --nocapture

use turbovec::kmeans_test_api::{assign_flat, CoarseIndex};

fn unit_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut x = seed | 1;
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
}

#[test]
fn flat_vs_hierarchy_at_build_shapes() {
    let dim = 768;
    let probe = 8;
    println!(
        "{:>7} {:>7} {:>10} {:>10} {:>11} {:>8} {:>9}",
        "n", "nlist", "flat_ms", "hier_ms", "inpool_ms", "speedup", "agree"
    );
    for &nlist in &[390usize, 781, 1561, 4096, 16384] {
        let centroids = unit_rows(nlist, dim, 0xC0FFEE ^ nlist as u64);
        // The rebuild is charged to the same gauge as the assignment it
        // accelerates, so it has to be separated or it reads as a slow assign.
        let t = std::time::Instant::now();
        let coarse = CoarseIndex::build(&centroids, nlist, dim, 7);
        println!("  build(nlist={nlist}) = {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        for &n in &[10_000usize, 20_000] {
            let data = unit_rows(n, dim, 0xBEEF ^ n as u64);

            // Warm both paths so neither pays first-touch or allocator costs.
            let _ = assign_flat(&data[..dim * 64], 64, dim, &centroids, nlist);
            let _ = coarse.assign(&data[..dim * 64], 64, &centroids, probe);

            let t = std::time::Instant::now();
            let flat = assign_flat(&data, n, dim, &centroids, nlist);
            let flat_ms = t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            let hier = coarse.assign(&data, n, &centroids, probe);
            let hier_ms = t.elapsed().as_secs_f64() * 1e3;

            // The extension runs every save inside an installed pool, so the
            // real call site is a rayon WORKER, not a bare thread. Measure
            // that too: a par_iter behaves differently depending on which it
            // is, and the in-situ number is the one that matters.
            let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
            let t = std::time::Instant::now();
            let inpool = pool.install(|| coarse.assign(&data, n, &centroids, probe));
            let inpool_ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(hier, inpool, "pool context must not change the answer");

            let agree =
                flat.iter().zip(&hier).filter(|(a, b)| a == b).count() as f64 / n as f64;
            println!(
                "{n:>7} {nlist:>7} {flat_ms:>10.1} {hier_ms:>10.1} {inpool_ms:>11.1} {:>7.2}x {:>8.1}%",
                flat_ms / inpool_ms,
                agree * 100.0
            );
        }
    }
}
