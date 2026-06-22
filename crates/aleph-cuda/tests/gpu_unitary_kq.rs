//! P5.9-02a: apply fused `UnitaryKq` (k=4,5) dense blocks on the GPU.
//!
//! `FuseKq` (P2-07) collapses adjacent gates spanning ≤ `max_qubits` into one
//! dense `2^k × 2^k` `UnitaryKq`. The GPU backends previously rejected these
//! (no fixed-size `GateMatrix`; the enum stops at 8×8) — now `CudaSvBackend`'s
//! `apply_kq` kernel (k≤5) and `CuStateVecBackend`'s `custatevecApplyMatrix`
//! take the raw row-major slice. This file proves the fused GPU run equals the
//! unfused CPU oracle at 1e-10, and asserts the fused circuit actually contains
//! k=4 and k=5 blocks so the new path is genuinely exercised.
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

/// A 5-wide entangling chain: H + a CNOT ladder + rotations confined to a
/// sliding window. `FuseKq{max_qubits:5}` merges each window into one dense
/// k≤5 block, deterministically producing both k=4 and k=5 `UnitaryKq`s.
fn dense_chain(n: u32, rng: &mut StdRng) -> Circuit {
    assert!(n >= 5);
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    // Walk a width-5 window; inside it, ladder CNOTs + a sprinkle of rotations
    // keep the whole window in one fused block (support never exceeds 5).
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

/// Correctness: the FuseKq-fused circuit on the GPU must equal the unfused
/// circuit on the CPU oracle, proving the GPU applies k=4,5 `UnitaryKq` blocks.
#[test]
fn gpu_unitary_kq_matches_cpu_unfused() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping UnitaryKq oracle: {e}");
            return;
        }
    };
    let n = 11;
    let mut rng = StdRng::seed_from_u64(0x5909_02a0);

    // Aggregate k-histogram across all workloads so we can assert the new
    // k=4 and k=5 paths are both hit at least once.
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
        let got = HasAmplitudes::amplitudes(&run(&mut gpu, &fused).expect("gpu fused"));
        assert_eq!(got.len(), want.len(), "{name}: len");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = (a - b).norm();
            assert!(d <= 1e-10, "{name} i={i}: |Δ|={d:.2e} (kq hist {hist:?})");
        }
    }

    // The whole point of P5.9-02a: blocks larger than the M8x8 (k=3) ceiling
    // must actually reach the GPU. If fusion produced none, the oracle above
    // proves nothing about the new path.
    let k4 = total.get(&4).copied().unwrap_or(0);
    let k5 = total.get(&5).copied().unwrap_or(0);
    assert!(
        k4 > 0 && k5 > 0,
        "expected the new k=4 and k=5 GPU path to be exercised; got histogram {total:?}"
    );
}
