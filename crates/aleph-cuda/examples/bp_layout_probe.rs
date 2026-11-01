//! P8-09 bounded BP profiling input. One fresh process per cell; fail on missing CUDA.
//! Usage: bp_layout_probe DISTANCE SHOTS MECHANISM_RATE REPEATS [shot|edge|warp]
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_cuda::{BpScratchLayout, CudaBp};
    use aleph_qec::{build_dem, BpDecoder, SurfaceCode, Syndrome};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::time::Instant;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: DISTANCE SHOTS RATE REPEATS [shot|edge|warp]".into());
    }
    let d: usize = args[0].parse()?;
    let shots: usize = args[1].parse()?;
    let p: f64 = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    if shots == 0 || repeats == 0 || !p.is_finite() || !(0.0..=0.5).contains(&p) {
        return Err("invalid endpoint".into());
    }
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p))?;
    let cpu = BpDecoder::with_params(&dem, 64, 0.875);
    let graph = cpu.tanner();
    let layout = match args.get(4).map(String::as_str).unwrap_or("shot") {
        "shot" => BpScratchLayout::ShotMajor,
        "edge" => BpScratchLayout::EdgeMajor,
        "warp" => BpScratchLayout::WarpTiled,
        _ => return Err("layout must be shot, edge or warp".into()),
    };
    let gpu = CudaBp::with_scratch_layout(&graph, layout)?;
    eprintln!(
        "layout={layout:?} per_shot_bytes={} max_iter=64 alpha=0.875 seed=559",
        gpu.per_shot_bytes()
    );
    let mut rng = StdRng::seed_from_u64(559);
    let syndromes: Vec<_> = (0..shots)
        .map(|_| {
            let mut bits = vec![false; dem.detectors];
            for error in &dem.errors {
                if rng.gen::<f64>() < error.prob {
                    for &det in &error.dets {
                        bits[det as usize] ^= true;
                    }
                }
            }
            Syndrome::from_bits(&bits)
        })
        .collect();
    let packed = gpu.pack(&syndromes);
    let reference = gpu.decode_packed(&packed, shots)?;
    gpu.synchronize()?;
    eprintln!("retained_scratch_bytes={}", gpu.retained_scratch_bytes());
    println!("distance,shots,rate,repeat,ns,vars,edges,checks,checksum");
    for repeat in 0..repeats {
        let start = Instant::now();
        let got = gpu.decode_packed(&packed, shots)?;
        gpu.synchronize()?;
        let ns = start.elapsed().as_nanos();
        assert_eq!(got, reference, "nondeterministic decode");
        let checksum = got
            .iter()
            .fold(0u64, |acc, &x| acc.wrapping_mul(31).wrapping_add(x));
        println!(
            "{d},{shots},{p},{repeat},{ns},{},{},{},{checksum}",
            graph.n_vars, graph.n_edges, graph.num_detectors
        );
    }
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux --features cuda and a real GPU");
    std::process::exit(1);
}
