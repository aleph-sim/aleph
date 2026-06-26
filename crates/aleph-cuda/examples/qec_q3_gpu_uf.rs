//! Q3-01 benchmark: GPU Union-Find decoder vs the CPU `UnionFindDecoder`, syndromes/second per
//! distance, plus a correctness guard (every GPU mask must equal the CPU correction). The GPU path
//! decodes a whole batch in one launch (one thread per shot); the CPU path is single-thread decode
//! (the directly comparable core, matching the Q1-05/Q2-03 methodology).
//!
//! Requires a Linux box with `--features cuda` and a real GPU. Usage:
//!
//! ```text
//! cargo run --release -p aleph-cuda --features cuda --example qec_q3_gpu_uf -- [shots] [seed]
//! # defaults: shots=100000 seed=2024
//! ```

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() {
    use std::time::Instant;

    use aleph_cuda::{mask_to_flips, CudaUnionFind};
    use aleph_qec::{build_dem, Decoder, SurfaceCode, Syndrome, UnionFindDecoder};

    let args: Vec<String> = std::env::args().collect();
    let shots: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!("# Q3-01 GPU UF vs CPU UF: shots={shots} seed={seed}, uniform p=3%, unweighted");
    println!("d,detectors,edges,shots,avg_defects,cpu_syn_per_s,gpu_syn_per_s,speedup,mismatches");

    for &d in &[3usize, 5, 7, 9, 11] {
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
        let nd = cpu.graph().num_detectors;
        let no = cpu.graph().num_observables;
        let n_edges = cpu.graph().edge_a.len();

        // Deterministic syndrome batch (DEM-Bernoulli sampling, like the harness).
        let syns = sample_dem(&dem, shots, seed ^ d as u64);
        let avg_def =
            syns.iter().map(|s| s.weight()).sum::<usize>() as f64 / syns.len().max(1) as f64;

        // CPU: single-thread decode, best of 3.
        let mut cpu_best = f64::INFINITY;
        let mut cpu_masks: Vec<u64> = Vec::new();
        for r in 0..3 {
            let t = Instant::now();
            let mut sink = 0u64;
            let mut local = Vec::with_capacity(shots);
            for s in &syns {
                let flips = cpu.decode(s).observable_flips;
                let mut m = 0u64;
                for (o, f) in flips.iter().enumerate() {
                    if *f {
                        m |= 1 << o;
                    }
                }
                sink ^= m;
                if r == 0 {
                    local.push(m);
                }
            }
            std::hint::black_box(sink);
            cpu_best = cpu_best.min(t.elapsed().as_secs_f64());
            if r == 0 {
                cpu_masks = local;
            }
        }

        // GPU: whole-batch decode (incl. upload + launch + download + sync), best of 3.
        let packed = gpu.pack(&syns);
        let mut gpu_best = f64::INFINITY;
        let mut gpu_masks: Vec<u64> = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let masks = gpu.decode_packed(&packed, shots).expect("gpu decode");
            gpu.synchronize().ok();
            gpu_best = gpu_best.min(t.elapsed().as_secs_f64());
            gpu_masks = masks;
        }

        // Correctness guard: GPU must equal CPU bit-for-bit.
        let mismatches = cpu_masks
            .iter()
            .zip(&gpu_masks)
            .filter(|(a, b)| mask_to_flips(**a, no) != mask_to_flips(**b, no))
            .count();

        let cpu_s = shots as f64 / cpu_best;
        let gpu_s = shots as f64 / gpu_best;
        println!(
            "{d},{nd},{n_edges},{shots},{avg_def:.2},{cpu_s:.0},{gpu_s:.0},{:.2},{mismatches}",
            gpu_s / cpu_s
        );
        eprintln!(
            "  d={d}: cpu={cpu_s:.0}/s gpu={gpu_s:.0}/s speedup={:.2}x defects={avg_def:.1} mism={mismatches}",
            gpu_s / cpu_s
        );
    }

    /// DEM-Bernoulli syndrome sampler (one coin per mechanism), deterministic per seed.
    fn sample_dem(dem: &aleph_qec::DetectorErrorModel, shots: usize, seed: u64) -> Vec<Syndrome> {
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
    eprintln!("qec_q3_gpu_uf requires a Linux host built with --features cuda");
}
