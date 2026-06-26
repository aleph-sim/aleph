//! Q3-03 oracle: the on-device Monte-Carlo logical-error rate ([`CudaThreshold`]) must agree with
//! the CPU harness ([`run_dem_experiment`] + [`UnionFindDecoder`]) within statistical error on every
//! cell — the two sample the same DEM (different RNG streams) and decode with the same algorithm, so
//! their rate estimates must coincide within the combined confidence interval.
//!
//! Runs only on a Linux box with `--features cuda` and a real GPU; skips cleanly otherwise.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_cuda::CudaThreshold;
use aleph_qec::{build_dem, run_dem_experiment, SurfaceCode, UnionFindDecoder};

/// GPU and CPU logical-error rates must agree within a generous (4σ) combined CI — robustly
/// non-flaky, while a broken sampler or decoder (which would differ by ≫ 0.01) is still caught.
#[test]
fn gpu_montecarlo_rate_matches_cpu() {
    let shots = 60_000u64;
    let seed = 2024u64;
    for &d in &[3usize, 5, 7, 9, 11] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        for &p in &[0.02, 0.03, 0.04] {
            let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();

            let gpu = match CudaThreshold::new(&dem) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping GPU Monte-Carlo oracle (no device): {e}");
                    return;
                }
            };
            let g = gpu.run(shots, seed).expect("gpu run");
            let cpu = run_dem_experiment(&dem, shots, &UnionFindDecoder::new(&dem).unwrap(), seed)
                .expect("cpu run");

            let delta = (g.rate() - cpu.rate).abs();
            let bound = 2.0 * (g.ci95() + cpu.ci95); // ~4σ combined
            assert!(
                delta <= bound,
                "d={d} p={p}: GPU rate {:.5} vs CPU {:.5} differ by {delta:.5} > {bound:.5}",
                g.rate(),
                cpu.rate
            );
        }
    }
}
