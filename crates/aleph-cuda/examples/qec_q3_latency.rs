//! Q3-04 latency micro-benchmark: GPU Union-Find decode **per-shot latency** as a function of batch
//! size, against the CPU per-shot decode time. The point is to expose the throughput/latency split:
//! the GPU decoder is batch-throughput-optimised, so at small batch its per-shot latency is dominated
//! by kernel-launch + transfer overhead (worse than the CPU), and only at large batch does the
//! per-shot cost fall below the CPU. This is the honest limit of the GPU decoder (latency → FPGA,
//! Phase Q4/Q6).
//!
//! Requires a Linux box with `--features cuda` and a real GPU. Usage:
//!
//! ```text
//! cargo run --release -p aleph-cuda --features cuda --example qec_q3_latency -- [seed]
//! ```

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() {
    use std::time::Instant;

    use aleph_cuda::CudaUnionFind;
    use aleph_qec::{
        build_dem, Decoder, DetectorErrorModel, SurfaceCode, Syndrome, UnionFindDecoder,
    };

    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2024);

    const BATCHES: &[usize] = &[1, 10, 100, 1_000, 10_000, 100_000];

    eprintln!("# Q3-04 GPU UF decode latency vs batch size, uniform p=3%");
    println!("d,batch,gpu_total_ms,gpu_per_shot_us,cpu_per_shot_us,gpu_over_cpu");

    for &d in &[5usize, 11] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let cpu = UnionFindDecoder::new(&dem).unwrap();
        let gpu = match CudaUnionFind::new(&cpu.graph()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU ({e}); aborting");
                return;
            }
        };

        // CPU per-shot latency: average a single serial decode over a fixed sample.
        let probe = sample_dem(&dem, 20_000, seed ^ d as u64);
        let mut cpu_best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let mut sink = 0u64;
            for s in &probe {
                sink ^= cpu
                    .decode(s)
                    .observable_flips
                    .first()
                    .copied()
                    .unwrap_or(false) as u64;
            }
            std::hint::black_box(sink);
            cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
        }
        let cpu_per_shot_us = cpu_best / probe.len() as f64 * 1e6;

        for &b in BATCHES {
            let syns = sample_dem(&dem, b, seed ^ (d as u64) ^ ((b as u64) << 8));
            let packed = gpu.pack(&syns);
            // GPU whole-batch decode (upload + launch + download + sync), best of 5.
            let mut gpu_best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let _ = gpu.decode_packed(&packed, b).expect("gpu decode");
                gpu.synchronize().ok();
                gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
            }
            let gpu_total_ms = gpu_best * 1e3;
            let gpu_per_shot_us = gpu_best / b as f64 * 1e6;
            println!(
                "{d},{b},{gpu_total_ms:.4},{gpu_per_shot_us:.3},{cpu_per_shot_us:.3},{:.2}",
                gpu_per_shot_us / cpu_per_shot_us
            );
            eprintln!(
                "  d={d} batch={b}: gpu={gpu_total_ms:.3}ms ({gpu_per_shot_us:.2}us/shot) cpu={cpu_per_shot_us:.2}us/shot  gpu/cpu={:.2}x",
                gpu_per_shot_us / cpu_per_shot_us
            );
        }
    }

    fn sample_dem(dem: &DetectorErrorModel, shots: usize, seed: u64) -> Vec<Syndrome> {
        (0..shots as u64)
            .map(|s| {
                let mut z = seed.wrapping_add(s.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut next = || {
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    (z >> 11) as f64 / (1u64 << 53) as f64
                };
                let mut det = vec![false; dem.detectors];
                for e in &dem.errors {
                    if next() < e.prob {
                        for &dd in &e.dets {
                            det[dd as usize] ^= true;
                        }
                    }
                }
                Syndrome::from_bits(&det)
            })
            .collect()
    }
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("qec_q3_latency requires a Linux host built with --features cuda");
}
