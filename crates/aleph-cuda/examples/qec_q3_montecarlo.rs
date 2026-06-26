//! Q3-03 benchmark: on-device Monte-Carlo threshold sweep (sample + decode + score entirely on the
//! GPU) vs the CPU harness (`run_dem_experiment` + Union-Find), for `d ∈ {3..13}`. Reports, per
//! cell, the two logical-error rates (which must agree within CI) and the GPU-vs-CPU wall-clock
//! speed-up, plus the aggregate sweep speed-up.
//!
//! Requires a Linux box with `--features cuda` and a real GPU. Usage:
//!
//! ```text
//! cargo run --release -p aleph-cuda --features cuda --example qec_q3_montecarlo -- [shots] [seed]
//! # defaults: shots=100000 seed=2024
//! ```

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() {
    use std::time::Instant;

    use aleph_cuda::CudaThreshold;
    use aleph_qec::{build_dem, run_dem_experiment, SurfaceCode, UnionFindDecoder};

    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    const DISTANCES: &[usize] = &[3, 5, 7, 9, 11, 13];
    const PROBS: &[f64] = &[0.020, 0.025, 0.030, 0.035, 0.040];

    eprintln!(
        "# Q3-03 on-device Monte-Carlo vs CPU harness: shots={shots} seed={seed}, UF decoder"
    );
    println!("d,p,shots,gpu_rate,gpu_ci,cpu_rate,cpu_ci,within_ci,gpu_s,cpu_s,speedup");

    let (mut tot_gpu, mut tot_cpu) = (0.0f64, 0.0f64);
    for &d in DISTANCES {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        for &p in PROBS {
            let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();

            let gpu = match CudaThreshold::new(&dem) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("no GPU ({e}); aborting");
                    return;
                }
            };
            // GPU end-to-end: sample + decode + score on device (only the count returns). Best of 2.
            let mut gpu_best = f64::INFINITY;
            let mut g = gpu.run(shots, seed).expect("gpu run");
            for _ in 0..2 {
                let t = Instant::now();
                g = gpu.run(shots, seed).expect("gpu run");
                gpu.synchronize().ok();
                gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
            }

            // CPU harness (Q0-04): parallel sampling + Union-Find decode. Best of 2.
            let dec = UnionFindDecoder::new(&dem).unwrap();
            let mut cpu_best = f64::INFINITY;
            let mut c = run_dem_experiment(&dem, shots, &dec, seed).expect("cpu run");
            for _ in 0..2 {
                let t = Instant::now();
                c = run_dem_experiment(&dem, shots, &dec, seed).expect("cpu run");
                cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
            }

            tot_gpu += gpu_best;
            tot_cpu += cpu_best;
            let delta = (g.rate() - c.rate).abs();
            let within = delta <= 2.0 * (g.ci95() + c.ci95);
            println!(
                "{d},{p},{shots},{:.6},{:.6},{:.6},{:.6},{within},{gpu_best:.4},{cpu_best:.4},{:.2}",
                g.rate(),
                g.ci95(),
                c.rate,
                c.ci95,
                cpu_best / gpu_best,
            );
            eprintln!(
                "  d={d} p={p:.3}: gpu={:.4e} cpu={:.4e} within_ci={within} | gpu={gpu_best:.3}s cpu={cpu_best:.3}s ({:.1}x)",
                g.rate(),
                c.rate,
                cpu_best / gpu_best,
            );
        }
    }
    eprintln!(
        "# aggregate sweep wall-clock: gpu={tot_gpu:.2}s cpu={tot_cpu:.2}s  speedup={:.2}x",
        tot_cpu / tot_gpu
    );
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("qec_q3_montecarlo requires a Linux host built with --features cuda");
}
