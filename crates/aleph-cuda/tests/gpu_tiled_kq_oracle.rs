//! P5.10-01: warp-cooperative register-tiled fused-block kernel (`apply_kq_tiled`).
//!
//! The generic `apply_kq` puts one thread on a whole 2^k group, holding the
//! block in `v[32]`/`gidx[32]` thread-local arrays that spill at k=4,5.
//! `apply_kq_tiled` instead spreads the block across a group of 2^k warp lanes
//! (one amplitude per lane) and does the matvec as an intra-warp shuffle
//! reduction. This file proves the tiled kernel is **bit-for-bit oracle-equal**
//! to the unfused CPU SV at 1e-10 across k=2..5 — forced on for every dense
//! block (`with_tiled_min_k(2)`) so the new code path is exercised end-to-end,
//! and with k=4 and k=5 blocks asserted present so the spill-prone regime is hit.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::collections::BTreeMap;

use aleph_backend::run;
use aleph_core::Gate;
use aleph_cuda::CudaSvBackend;
use aleph_ir::passes::{FuseKq, Pass};
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A width-5 entangling chain whose `FuseKq{max_qubits:5}` collapse
/// deterministically yields both k=4 and k=5 `UnitaryKq` blocks (mirrors the
/// P5.9-02a `dense_chain` fixture).
fn dense_chain(n: u32, rng: &mut StdRng) -> Circuit {
    assert!(n >= 5);
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for base in (0..=n - 5).step_by(5) {
        for off in 0..4 {
            c.cnot(base + off, base + off + 1).unwrap();
            c.rz(rng.gen::<f64>() * std::f64::consts::TAU, base + off + 1)
                .unwrap();
        }
        c.rx(rng.gen::<f64>() * std::f64::consts::TAU, base)
            .unwrap();
    }
    c
}

fn random_brickwall(rng: &mut StdRng, n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3) {
                0 => c.rx(theta, q),
                1 => c.ry(theta, q),
                _ => c.rz(theta, q),
            }
            .unwrap();
        }
        for q in (0..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

fn vqe(n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut t = 0.1_f64;
    for _ in 0..layers {
        for q in 0..n {
            c.ry(t, q).unwrap();
            t += 0.017;
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in 0..n {
            c.rz(t, q).unwrap();
            t += 0.013;
        }
    }
    c
}

/// Histogram of fused `UnitaryKq` blocks by `k`.
fn kq_histogram(c: &Circuit) -> BTreeMap<u8, usize> {
    let mut h = BTreeMap::new();
    for inst in c.instructions() {
        if let Instruction::Gate(g) = inst {
            if let Gate::UnitaryKq { k, .. } = &g.gate {
                *h.entry(*k).or_default() += 1;
            }
        }
    }
    h
}

fn fuse_kq(circ: &Circuit, max_qubits: usize) -> Circuit {
    let mut fused = circ.clone();
    FuseKq { max_qubits }.run(&mut fused).expect("FuseKq");
    fused
}

/// The tiled kernel forced on for every dense block (k≥2) must equal the unfused
/// CPU oracle at 1e-10, with k=4 and k=5 blocks present so the spill regime the
/// kernel targets is actually exercised.
#[test]
fn gpu_tiled_kq_matches_cpu_unfused() {
    let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_tiled_min_k(2)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tiled-kq oracle: {e}");
            return;
        }
    };
    // n=11 spans whole blocks (n=8 thresholds) and the small-tail block path
    // (the last grid block is partially out of range): both must be correct.
    let n = 11;
    let mut rng = StdRng::seed_from_u64(0x51001);

    let mut total: BTreeMap<u8, usize> = BTreeMap::new();
    let workloads: Vec<(&str, Circuit)> = vec![
        ("dense_chain", dense_chain(n, &mut rng)),
        ("random(d20)", random_brickwall(&mut rng, n, 20)),
        ("vqe(8L)", vqe(n, 8)),
    ];

    for (name, circ) in &workloads {
        let fused = fuse_kq(circ, 5);
        let hist = kq_histogram(&fused);
        for (&k, &cnt) in &hist {
            *total.entry(k).or_default() += cnt;
        }

        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu unfused"));
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("gpu tiled fused"));
        assert_eq!(got.len(), want.len(), "{name}: len");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(d <= 1e-10, "{name} i={i}: |Δ|={d:.2e} (kq hist {hist:?})");
        }
    }

    let k4 = total.get(&4).copied().unwrap_or(0);
    let k5 = total.get(&5).copied().unwrap_or(0);
    assert!(
        k4 > 0 && k5 > 0,
        "expected k=4 and k=5 tiled blocks to be exercised; got histogram {total:?}"
    );
}

/// Small-n exhaustiveness: at n=5 a single fused k=5 block over a known state is
/// the partial-grid-block worst case (2^5=32 threads < BLOCK=256), where the
/// out-of-range tail and the warp shuffle mask must still agree with the oracle.
#[test]
fn gpu_tiled_kq_small_n_partial_block() {
    let mut gpu = match CudaSvBackend::with_seed(0).map(|b| b.with_tiled_min_k(2)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tiled-kq small-n: {e}");
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0xabc51001);
    // n=5: one window fuses to exactly one k=5 block; n=6,7 add k=4,5 mixes whose
    // groups straddle whole + tail grid blocks.
    for n in 5..=7u32 {
        let circ = dense_chain(n, &mut rng);
        let fused = fuse_kq(&circ, 5);
        assert!(
            kq_histogram(&fused).keys().any(|&k| k >= 4),
            "n={n}: expected a k≥4 block"
        );
        let mut cpu = NaiveSvBackend::with_seed(0);
        let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("gpu tiled"));
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(d <= 1e-10, "n={n} i={i}: |Δ|={d:.2e}");
        }
    }
}
