//! P8-10 same-input accuracy and latency probe. Requires a real GPU; no skip success.
//! Usage: bp_cooperative_probe surface7|surface11|gross SHOTS RATE REPEATS shot|warp|coop [ORACLE.json] [--trace]
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_cuda::{BpScratchLayout, CudaBp};
    use aleph_qec::{build_dem, BBCode, BpDecoder, SurfaceCode, Syndrome};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::time::Instant;
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: MODEL SHOTS RATE REPEATS shot|warp|coop [ORACLE.json] [--trace]".into(),
        );
    }
    let trace = match args.get(6).map(String::as_str) {
        None => false,
        Some("--trace") => true,
        _ => return Err("unknown trace option".into()),
    };
    let model = &args[0];
    let shots: usize = args[1].parse()?;
    let p: f64 = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    if shots == 0 || repeats == 0 || !p.is_finite() || !(0.0..=0.5).contains(&p) {
        return Err("invalid endpoint".into());
    }
    let dem = match model.as_str() {
        "surface7" | "surface11" => {
            let d = if model == "surface7" { 7 } else { 11 };
            let e = SurfaceCode::new(d).memory_z_experiment(d);
            build_dem(&e.annotated, &e.phenomenological_mechanisms(p, p))?
        }
        "gross" => BBCode::gross().code_capacity_dem(p),
        _ => return Err("unsupported model".into()),
    };
    let cpu = BpDecoder::with_params(&dem, 64, 0.875);
    let graph = cpu.tanner();
    let gpu = match args[4].as_str() {
        "shot" => CudaBp::new(&graph)?,
        "warp" => CudaBp::with_scratch_layout(&graph, BpScratchLayout::WarpTiled)?,
        "coop" => CudaBp::warp_cooperative(&graph)?,
        _ => return Err("unsupported mode".into()),
    };
    let mut rng = StdRng::seed_from_u64(560);
    let mut truths = Vec::with_capacity(shots);
    let syndromes: Vec<_> = (0..shots)
        .map(|_| {
            let mut bits = vec![false; dem.detectors];
            let mut truth = 0u64;
            for error in &dem.errors {
                if rng.gen::<f64>() < error.prob {
                    for &d in &error.dets {
                        bits[d as usize] ^= true;
                    }
                    for &o in &error.obs {
                        truth ^= 1u64 << o;
                    }
                }
            }
            truths.push(truth);
            Syndrome::from_bits(&bits)
        })
        .collect();
    let packed = gpu.pack(&syndromes);
    let reference = if trace {
        None
    } else {
        Some(gpu.decode_packed(&packed, shots)?)
    };
    gpu.synchronize()?;
    // CPU oracle is cached only outside timers, with exact DEM/input hashes checked each use.
    let dem_sha = format!("{:x}", Sha256::digest(dem.to_dem_string().as_bytes()));
    let mut hasher = Sha256::new();
    for word in &packed {
        hasher.update(word.to_le_bytes());
    }
    let input_sha = format!("{:x}", hasher.finalize());
    let cache = args.get(5).map(std::path::Path::new);
    let oracle: serde_json::Value = if let Some(path) = cache.filter(|p| p.exists()) {
        let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if saved["dem_sha256"] != dem_sha
            || saved["input_sha256"] != input_sha
            || saved["shots"] != shots
            || saved["seed"] != 560
            || saved["alpha"] != 0.875
            || saved["max_iter"] != 64
        {
            return Err("stale oracle input/configuration".into());
        }
        saved
    } else {
        let mut invalid = 0;
        let mut logical_failures = 0;
        let mut expected_masks = Vec::with_capacity(shots);
        let mut convergence = Vec::with_capacity(shots);
        let mut failures = Vec::with_capacity(shots);
        for (i, syn) in syndromes.iter().enumerate() {
            let (correction, converged) = cpu.decode_bp(syn);
            let expected = correction
                .observable_flips
                .iter()
                .enumerate()
                .fold(0u64, |m, (o, &f)| m | (u64::from(f) << o));
            invalid += usize::from(!converged);
            logical_failures += usize::from(expected != truths[i]);
            expected_masks.push(expected);
            convergence.push(converged);
            failures.push(expected != truths[i]);
        }
        let result = json!({"dem_sha256":dem_sha,"input_sha256":input_sha,"shots":shots,"seed":560,"alpha":0.875,"max_iter":64,"masks":expected_masks,"invalid":invalid,"logical_failures":logical_failures,"converged":convergence,"logical_failure":failures});
        // No unchecked GPU result can seed the shared CPU oracle cache.
        let expected: Vec<u64> = serde_json::from_value(result["masks"].clone())?;
        if let Some(reference) = &reference {
            assert_eq!(reference, &expected, "initial GPU/CPU mismatch");
        }
        if let Some(path) = cache {
            std::fs::write(path, serde_json::to_string(&result)?)?;
        }
        result
    };
    let expected_masks: Vec<u64> = serde_json::from_value(oracle["masks"].clone())?;
    if let Some(reference) = reference {
        assert_eq!(
            reference, expected_masks,
            "GPU/CPU mismatch mode={}",
            args[4]
        );
    }
    // CPU validation can cool the GPU; give every arm the same final untimed warmup.
    let words = (dem.detectors as u32).div_ceil(32).max(1) as usize;
    if trace {
        if oracle["converged"].as_array().map(Vec::len) != Some(shots)
            || oracle["logical_failure"].as_array().map(Vec::len) != Some(shots)
        {
            return Err("trace needs a fresh per-input CPU oracle".into());
        }
        assert_eq!(gpu.decode_packed(&packed[..words], 1)?, expected_masks[..1]);
    } else {
        assert_eq!(gpu.decode_packed(&packed, shots)?, expected_masks);
    }
    gpu.synchronize()?;
    eprintln!("vars={} edges={} checks={} observables={} per_shot_bytes={} retained_scratch_bytes={} seed=560 max_iter=64 alpha=.875 invalid={} logical_failures={} oracle_shots={shots} dem_sha256={} input_sha256={}",graph.n_vars,graph.n_edges,graph.num_detectors,graph.num_observables,gpu.per_shot_bytes(),gpu.retained_scratch_bytes(),oracle["invalid"],oracle["logical_failures"],dem_sha,input_sha);
    println!("model,shots,rate,mode,repeat,ns,checksum");
    if trace {
        for (shot, expected) in expected_masks.iter().enumerate() {
            let start = Instant::now();
            let masks = gpu.decode_packed(&packed[shot * words..(shot + 1) * words], 1)?;
            gpu.synchronize()?;
            let ns = start.elapsed().as_nanos();
            assert_eq!(
                masks[0], *expected,
                "trace changed CPU correction at {shot}"
            );
            println!("{model},1,{p},{},{shot},{ns},{}", args[4], masks[0]);
        }
        return Ok(());
    }
    for repeat in 0..repeats {
        let start = Instant::now();
        let masks = gpu.decode_packed(&packed, shots)?;
        gpu.synchronize()?;
        let ns = start.elapsed().as_nanos();
        assert_eq!(masks, expected_masks, "repeat changed correction");
        let checksum = masks
            .iter()
            .fold(0u64, |a, &x| a.wrapping_mul(31).wrapping_add(x));
        println!("{model},{shots},{p},{},{repeat},{ns},{checksum}", args[4]);
    }
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux --features cuda and an actual GPU");
    std::process::exit(1);
}
