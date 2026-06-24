//! P5.11-03: multi-gate tile-residency paging correctness.
//!
//! `run_paged_batched` collapses a maximal run of consecutive low-only (`< m`)
//! gates into a single PCIe pass: each tile is gathered once, the whole run is
//! applied on its `2^m`-amplitude device sub-state, and it is scattered once.
//! High-qubit gates flush the run and take the proven per-gate path. This file
//! forces paging on at **small n with a small tile** (so the high/low split and
//! the residency batching are genuinely exercised) and pins the final state
//! **bit-for-bit against both the CPU `NaiveSvBackend` and the per-gate paged
//! path** (`run_paged`) at 1e-10, across circuits with deep low-qubit runs and
//! interleaved high-qubit gates. It also checks that the batched schedule makes
//! strictly fewer full-state passes (the P5.11-03 lever) on a locality-rich
//! circuit via [`paged_pass_counts`].
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{paged_pass_counts, CudaSvBackend};
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Locality-rich: many layers of 1q rotations + nearest-neighbour CNOTs **on the
/// low qubits only** (`< split`), with a single entangling CNOT into a high qubit
/// per layer. The long low-only runs are exactly what residency batching folds
/// into one pass; the high CNOT forces a flush + per-gate fallback. `split` is
/// the low/high boundary the test will also use as the tile size `m`.
fn locality_rich(rng: &mut StdRng, n: u32, split: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for _ in 0..depth {
        // Dense low-qubit run: 1q rotations on every low qubit …
        for q in 0..split {
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3) {
                0 => c.rx(theta, q),
                1 => c.ry(theta, q),
                _ => c.rz(theta, q),
            }
            .unwrap();
        }
        // … + a nearest-neighbour CNOT brick, still within the low block.
        for q in (0..split.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        for q in (1..split.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1).unwrap();
        }
        // One entangler crossing into the high block: ends the low run.
        if n > split {
            c.cnot(split - 1, split).unwrap();
        }
    }
    c
}

/// GHZ ladder — the worst case for batching: every CNOT climbs into a new high
/// qubit, so there are no low-only runs to fold (batched == per-gate here). Still
/// must reproduce the oracle exactly.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

/// Textbook QFT — controlled phases pair distant qubits (hh=2 high↔high), so it
/// mixes high-qubit gates with low-only H/phase runs.
fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.h(j).unwrap();
        for (offset, k) in ((j + 1)..n).enumerate() {
            let theta = std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![k],
                vec![j],
            ))
            .unwrap();
        }
    }
    c
}

/// The batched state must equal both the CPU oracle and the per-gate paged path.
fn assert_batched_matches(name: &str, circ: &Circuit, m: u32) {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping batched oracle ({name}, m={m}): {e}");
            return;
        }
    };
    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, circ).expect("cpu in-core"));

    let per_gate = gpu
        .run_paged(circ, m)
        .expect("per-gate paged")
        .amplitudes_vec();
    let batched = gpu
        .run_paged_batched(circ, m)
        .expect("batched paged")
        .amplitudes_vec();

    assert_eq!(batched.len(), want.len(), "{name} m={m}: len");
    for (i, ((b, p), w)) in batched
        .iter()
        .zip(per_gate.iter())
        .zip(want.iter())
        .enumerate()
    {
        let dw = (b - w).norm();
        let dp = (b - p).norm();
        assert!(dw <= 1e-10, "{name} m={m} i={i}: |Δ cpu|={dw:.2e}");
        assert!(dp <= 1e-10, "{name} m={m} i={i}: |Δ per-gate|={dp:.2e}");
    }
}

/// Across circuits and tile splits, the batched path is bit-for-bit identical to
/// the CPU oracle and to per-gate paging.
#[test]
fn batched_matches_oracle_and_per_gate() {
    let n = 8;
    let mut rng = StdRng::seed_from_u64(0x511_03);
    let workloads: Vec<(&str, Circuit)> = vec![
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("locality(split5,d6)", locality_rich(&mut rng, n, 5, 6)),
        ("locality(split6,d6)", locality_rich(&mut rng, n, 6, 6)),
    ];
    // m=5 → h=3 ; m=6 → h=2 ; m=7 → h=1. Each forces a different split.
    for (name, circ) in &workloads {
        for m in [5u32, 6, 7] {
            assert_batched_matches(name, circ, m);
        }
    }
}

/// The locality-rich split must give a real pass reduction (the P5.11-03 metric).
/// At the matching tile size `m = split`, every per-layer low run folds into one
/// pass while only the single cross-block CNOT stays per-gate, so per-gate makes
/// many more full-state passes than batched.
#[test]
fn batched_cuts_pcie_passes() {
    let mut rng = StdRng::seed_from_u64(0x511_03a);
    let split = 6;
    let circ = locality_rich(&mut rng, 8, split, 6);
    let (per_gate, batched) = paged_pass_counts(&circ, split);
    assert!(
        per_gate >= 2 * batched,
        "expected ≥2× fewer passes: per_gate={per_gate}, batched={batched}"
    );
}
