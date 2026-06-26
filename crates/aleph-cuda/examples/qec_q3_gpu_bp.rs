//! Q3-02 benchmark: GPU min-sum BP decoder vs the CPU `BpDecoder`, syndromes/second per distance,
//! plus a correctness guard (every GPU mask must equal the CPU correction). GPU decodes a whole
//! batch in one launch (one thread per shot); CPU is single-thread decode.
//!
//! Requires a Linux box with `--features cuda` and a real GPU. Usage:
//!
//! ```text
//! cargo run --release -p aleph-cuda --features cuda --example qec_q3_gpu_bp -- [shots] [seed]
//! # defaults: shots=50000 seed=2024
//! ```

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() {
    use std::time::Instant;

    use aleph_cuda::{mask_to_flips, CudaBp};
    use aleph_qec::{build_dem, BpDecoder, Decoder, DetectorErrorModel, SurfaceCode, Syndrome};

    let args: Vec<String> = std::env::args().collect();
    let shots: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    eprintln!("# Q3-02 GPU BP vs CPU BP: shots={shots} seed={seed}, uniform p=3%, min-sum α=0.875, 64 iters");
    println!(
        "d,detectors,vars,edges,shots,avg_defects,cpu_syn_per_s,gpu_syn_per_s,speedup,mismatches"
    );

    for &d in &[3usize, 5, 7, 9, 11] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let cpu = BpDecoder::with_params(&dem, 64, 0.875);
        let gpu = match CudaBp::new(&cpu.tanner()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU ({e}); aborting");
                return;
            }
        };
        let t = cpu.tanner();
        let (nd, no, nv, ne) = (t.num_detectors, t.num_observables, t.n_vars, t.n_edges);

        let syns = sample_dem(&dem, shots, seed ^ d as u64);
        let avg_def =
            syns.iter().map(|s| s.weight()).sum::<usize>() as f64 / syns.len().max(1) as f64;

        // CPU: single-thread decode, best of 3.
        let mut cpu_best = f64::INFINITY;
        let mut cpu_masks: Vec<u64> = Vec::new();
        for r in 0..3 {
            let t0 = Instant::now();
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
            cpu_best = cpu_best.min(t0.elapsed().as_secs_f64());
            if r == 0 {
                cpu_masks = local;
            }
        }

        // GPU: whole-batch decode (upload + launch + download + sync), best of 3.
        let packed = gpu.pack(&syns);
        let mut gpu_best = f64::INFINITY;
        let mut gpu_masks: Vec<u64> = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let masks = gpu.decode_packed(&packed, shots).expect("gpu bp decode");
            gpu.synchronize().ok();
            gpu_best = gpu_best.min(t0.elapsed().as_secs_f64());
            gpu_masks = masks;
        }

        let mismatches = cpu_masks
            .iter()
            .zip(&gpu_masks)
            .filter(|(a, b)| mask_to_flips(**a, no) != mask_to_flips(**b, no))
            .count();

        let cpu_s = shots as f64 / cpu_best;
        let gpu_s = shots as f64 / gpu_best;
        println!(
            "{d},{nd},{nv},{ne},{shots},{avg_def:.2},{cpu_s:.0},{gpu_s:.0},{:.2},{mismatches}",
            gpu_s / cpu_s
        );
        eprintln!(
            "  d={d}: cpu={cpu_s:.0}/s gpu={gpu_s:.0}/s speedup={:.2}x vars={nv} edges={ne} mism={mismatches}",
            gpu_s / cpu_s
        );
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
    eprintln!("qec_q3_gpu_bp requires a Linux host built with --features cuda");
}
