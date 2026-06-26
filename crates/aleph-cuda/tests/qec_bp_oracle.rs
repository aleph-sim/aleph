//! Q3-02 oracle: the GPU min-sum BP decoder must produce corrections **numerically identical** to
//! the CPU [`BpDecoder`] on the same syndromes (same `double` schedule in the same edge order → the
//! hard-decision error vector, and thus the correction, matches bit-for-bit).
//!
//! Runs only on a Linux box with `--features cuda` and a real GPU; skips cleanly otherwise.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_cuda::{mask_to_flips, CudaBp};
use aleph_qec::{
    build_dem, BpDecoder, Decoder, DemError, DetectorErrorModel, SurfaceCode, Syndrome,
};

fn splitmix(seed: u64) -> impl FnMut() -> u64 {
    let mut z = seed;
    move || {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    }
}

fn sample(nd: usize, n: usize, q: f64, seed: u64) -> Vec<Syndrome> {
    let mut rng = splitmix(seed);
    (0..n)
        .map(|_| {
            let bits: Vec<bool> = (0..nd)
                .map(|_| ((rng() >> 11) as f64 / (1u64 << 53) as f64) < q)
                .collect();
            Syndrome::from_bits(&bits)
        })
        .collect()
}

fn repetition_dem(n: usize, p: f64) -> DetectorErrorModel {
    let n_checks = n - 1;
    let mut errors = Vec::new();
    for i in 0..n {
        let mut dets = Vec::new();
        if i > 0 {
            dets.push((i - 1) as u32);
        }
        if i < n_checks {
            dets.push(i as u32);
        }
        let obs = if i == 0 { vec![0u32] } else { vec![] };
        errors.push(DemError::new(p, dets, obs));
    }
    DetectorErrorModel {
        detectors: n_checks,
        observables: 1,
        errors,
    }
}

/// GPU masks must equal CPU corrections on every shot.
fn assert_match(bp: &BpDecoder, nd: usize, shots: usize, seed: u64) {
    let gpu = match CudaBp::new(&bp.tanner()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU BP oracle (no device): {e}");
            return;
        }
    };
    let no = bp.num_observables();
    for (i, &q) in [0.02, 0.05, 0.10, 0.18].iter().enumerate() {
        let syns = sample(nd, shots, q, seed ^ ((i as u64) << 20));
        let masks = gpu.decode(&syns).expect("gpu bp decode");
        for (s, syn) in syns.iter().enumerate() {
            let want = bp.decode(syn).observable_flips;
            let got = mask_to_flips(masks[s], no);
            assert_eq!(
                got, want,
                "BP mismatch q={q} shot={s} fired={:?}",
                syn.fired
            );
        }
    }
}

#[test]
fn gpu_bp_matches_cpu_repetition() {
    for &n in &[8usize, 16, 32] {
        let bp = BpDecoder::new(&repetition_dem(n, 0.05));
        assert_match(&bp, n - 1, 4_000, 0x12E9 ^ n as u64);
    }
}

#[test]
fn gpu_bp_matches_cpu_surface() {
    for &d in &[3usize, 5, 7] {
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let bp = BpDecoder::with_params(&dem, 64, 0.875);
        assert_match(&bp, dem.detectors, 4_000, 0x5_FACE ^ d as u64);
    }
}

/// Acceptance: ≥1e5 syndromes decoded GPU vs CPU BP with zero disagreements.
#[test]
fn gpu_bp_matches_cpu_large_batch() {
    let d = 7;
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
    let bp = BpDecoder::with_params(&dem, 64, 0.875);
    let gpu = match CudaBp::new(&bp.tanner()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU BP large-batch oracle (no device): {e}");
            return;
        }
    };
    let no = bp.num_observables();
    let syns = sample(dem.detectors, 100_000, 0.05, 0xB1A5);
    let masks = gpu.decode(&syns).expect("gpu bp decode");
    let mismatches = syns
        .iter()
        .enumerate()
        .filter(|(s, syn)| mask_to_flips(masks[*s], no) != bp.decode(syn).observable_flips)
        .count();
    assert_eq!(
        mismatches, 0,
        "{mismatches}/100000 GPU/CPU BP corrections disagreed"
    );
}
