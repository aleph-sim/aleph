//! Q3-01 oracle: the GPU Union-Find decoder must produce **bit-identical** corrections to the CPU
//! [`UnionFindDecoder`] on the same syndromes, across distances and both growth modes.
//!
//! Runs only on a Linux box with `--features cuda` and a real GPU; skips cleanly otherwise (so the
//! GPU-less CI runner passes). Acceptance criterion: GPU vs CPU corrections match on ≥1e5 syndromes.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_cuda::{mask_to_flips, CudaUnionFind};
use aleph_qec::{build_dem, Decoder, SurfaceCode, Syndrome, UnionFindDecoder};

/// Build the surface-code memory-Z UF decoder at distance `d`, phys error `(p_data, p_meas)`.
fn cpu_decoder(d: usize, p_data: f64, p_meas: f64, weighted: bool) -> UnionFindDecoder {
    let exp = SurfaceCode::new(d).memory_z_experiment(d);
    let dem = build_dem(
        &exp.annotated,
        &exp.phenomenological_mechanisms(p_data, p_meas),
    )
    .unwrap();
    UnionFindDecoder::new(&dem).unwrap().weighted(weighted)
}

/// Deterministic SplitMix64 stream.
fn splitmix(seed: u64) -> impl FnMut() -> u64 {
    let mut z = seed;
    move || {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    }
}

/// Sample `n` random detector-pattern syndromes for `nd` detectors at fire-probability `q`.
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

/// Core check: GPU masks equal CPU corrections on every shot, in the given mode.
fn assert_gpu_matches_cpu(d: usize, p_data: f64, p_meas: f64, weighted: bool, shots: usize) {
    let cpu = cpu_decoder(d, p_data, p_meas, weighted);
    let gpu = match CudaUnionFind::new(&cpu.graph()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU UF oracle (no device): {e}");
            return;
        }
    };
    let no = cpu.graph().num_observables;
    // Sweep fire-probability so we cover sparse and dense syndromes; vary seed per cell.
    for (i, &q) in [0.02, 0.06, 0.12, 0.20].iter().enumerate() {
        let syns = sample(
            cpu.graph().num_detectors,
            shots,
            q,
            0xC0DE ^ (d as u64) ^ ((i as u64) << 20) ^ ((weighted as u64) << 40),
        );
        let masks = gpu.decode(&syns).expect("gpu decode");
        for (s, syn) in syns.iter().enumerate() {
            let want = cpu.decode(syn).observable_flips;
            let got = mask_to_flips(masks[s], no);
            assert_eq!(
                got, want,
                "mismatch d={d} weighted={weighted} q={q} shot={s} fired={:?}",
                syn.fired
            );
        }
    }
}

#[test]
fn gpu_uf_matches_cpu_unweighted() {
    for &d in &[3usize, 5, 7, 9, 11] {
        assert_gpu_matches_cpu(d, 0.03, 0.03, false, 6_000);
    }
}

#[test]
fn gpu_uf_matches_cpu_weighted() {
    for &d in &[3usize, 5, 7, 9, 11] {
        // Asymmetric noise → heterogeneous edge weights → exercises the jump-growth path.
        assert_gpu_matches_cpu(d, 0.02, 0.06, true, 6_000);
    }
}

/// Acceptance: ≥1e5 syndromes decoded GPU vs CPU with zero disagreements, at a large distance.
#[test]
fn gpu_uf_matches_cpu_large_batch() {
    let d = 9;
    let cpu = cpu_decoder(d, 0.03, 0.03, false);
    let gpu = match CudaUnionFind::new(&cpu.graph()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU UF large-batch oracle (no device): {e}");
            return;
        }
    };
    let no = cpu.graph().num_observables;
    let syns = sample(cpu.graph().num_detectors, 100_000, 0.05, 0xBEEF);
    let masks = gpu.decode(&syns).expect("gpu decode");
    let mut mismatches = 0usize;
    for (s, syn) in syns.iter().enumerate() {
        let want = cpu.decode(syn).observable_flips;
        let got = mask_to_flips(masks[s], no);
        if got != want {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches}/100000 GPU/CPU corrections disagreed"
    );
}
