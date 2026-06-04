//! P2-09: relabelling transparency (AC #3, 1e-12).
//!
//! `RelabelQubits` permutes qubit indices for cache locality, leaving the
//! simulated state in PHYSICAL-bit order and recording mid-circuit `Measure`
//! outcomes against PHYSICAL qubits. The `run_optimized` driver must make this
//! invisible: a single final gather (`Backend::unpermute_state`) reorders the
//! state to logical order, and outcome qubits are mapped back to logical.
//!
//! `run` (raw, gate-by-gate, no optimization, no relabelling) is the trusted
//! logical-order oracle. We:
//!
//!   (A) Force relabel to fire (a high-qubit-heavy fixture + small tile width),
//!       run the relabelled+tiled physical circuit, then un-permute via the
//!       backend hook, and assert the result equals the raw reference at 1e-12.
//!       This exercises the real `RelabelQubits` rewrite, the tiled executor on
//!       the relabelled circuit, AND the `unpermute_state` gather.
//!
//!   (B) Drive a normal Tier-1 circuit through `run_optimized` and assert it
//!       equals raw `run`, guarding the orchestration against regressions in the
//!       (common) no-relabel case as well as the relabel case.
//!
//!   (C) Force relabel on a circuit with a `Measure` on a high qubit and assert
//!       the reported `MeasurementRecord.qubit` is the LOGICAL qubit, not the
//!       physical one the pass rewrote it to.

use aleph_backend::{run, run_with_outcomes, Backend};
use aleph_ir::passes::{Pass, RelabelQubits, TileBlock};
use aleph_ir::Circuit;
use aleph_sv::{CpuState, NaiveSvBackend};

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

/// A high-qubit-heavy circuit: n qubits, all gate traffic concentrated on the
/// top three qubits (n-3, n-2, n-1). With a small tile width those targets are
/// all ≥ tile_bits (nothing confinable), so relabelling them to the low bits
/// strictly increases the confinable count and the guard fires.
fn high_qubit_heavy(n: u32) -> Circuit {
    assert!(n >= 4);
    let (a, b, c_) = (n - 3, n - 2, n - 1);
    let mut c = Circuit::new(n, 0);
    // A non-trivial entangling/rotating block, all on the top qubits.
    c.h(a).unwrap();
    c.h(b).unwrap();
    c.h(c_).unwrap();
    c.cnot(a, b).unwrap();
    c.rz(0.7, b).unwrap();
    c.cnot(b, c_).unwrap();
    c.ry(1.3, a).unwrap();
    c.cz(a, c_).unwrap();
    c.rx(-0.4, c_).unwrap();
    c.cnot(c_, a).unwrap();
    c.phase(0.9, b).unwrap();
    c
}

/// GHZ-n (low-qubit, normal Tier-1): `H q0; CX q0,q1; …`.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for i in 0..n - 1 {
        c.cnot(i, i + 1).unwrap();
    }
    c
}

// ---------------------------------------------------------------------------
// Oracle.
// ---------------------------------------------------------------------------

/// Assert two states agree elementwise within the 1e-12 spec bound.
fn assert_close(a: &CpuState, b: &CpuState, ctx: &str) {
    let xa = a.amplitudes();
    let xb = b.amplitudes();
    assert_eq!(
        xa.len(),
        xb.len(),
        "{ctx}: len {} vs {}",
        xa.len(),
        xb.len()
    );
    let mut max_err = 0.0f64;
    let mut worst = 0usize;
    for (i, (x, y)) in xa.iter().zip(xb.iter()).enumerate() {
        let e = (x - y).norm();
        if e > max_err {
            max_err = e;
            worst = i;
        }
    }
    assert!(
        max_err < 1e-12,
        "{ctx}: max abs err {max_err:e} >= 1e-12 at amp[{worst}] ({:?} vs {:?})",
        xa[worst],
        xb[worst]
    );
}

/// `inv[perm[l]] = l` — invert `perm[logical] = physical` to `inv[physical] =
/// logical`. Mirror of `aleph_backend::invert_perm` (private there).
fn invert_perm(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (logical, &physical) in perm.iter().enumerate() {
        inv[physical as usize] = logical as u32;
    }
    inv
}

// ---------------------------------------------------------------------------
// (A) Forced relabel + tile + un-permute, end-to-end at 1e-12.
// ---------------------------------------------------------------------------

#[test]
fn relabel_tile_unpermute_transparent() {
    const TILE_BITS: u8 = 3;
    let mut saw_relabel = false;

    for n in [6u32, 8, 10] {
        let c = high_qubit_heavy(n);

        // Trusted logical-order truth: raw, un-optimized, no relabelling.
        let reference = run(&mut NaiveSvBackend::with_seed(0), &c).unwrap();

        // Manually compose the relabel + tile passes at a small width so the
        // guard fires. (The default pipeline uses tile_bits=15, inert at small
        // n; forcing the width is the faithful way to exercise the path.)
        let mut opt = c.clone();
        RelabelQubits::new(TILE_BITS).run(&mut opt).unwrap();
        assert!(
            opt.qubit_permutation().is_some(),
            "relabel must fire for high-qubit-heavy n={n}"
        );
        saw_relabel = true;
        TileBlock::new(TILE_BITS).run(&mut opt).unwrap();

        // Drive it the way `run_optimized`'s tail does: run the physical-order
        // circuit, then invert via the backend hook → logical order.
        let perm = opt.qubit_permutation().unwrap().to_vec();
        let mut backend = NaiveSvBackend::with_seed(0);
        let mut state = run(&mut backend, &opt).unwrap();
        backend.unpermute_state(&mut state, &perm).unwrap();

        assert_close(&reference, &state, &format!("relabel+tile+unpermute n={n}"));
    }

    assert!(
        saw_relabel,
        "relabel never fired — fixture failed to force it"
    );
}

// ---------------------------------------------------------------------------
// (B) Driver integration through the default pipeline (no-relabel common case).
// ---------------------------------------------------------------------------

#[test]
fn run_optimized_matches_raw_default_pipeline() {
    for n in [4u32, 6, 8] {
        let c = ghz(n);
        let reference = run(&mut NaiveSvBackend::with_seed(0), &c).unwrap();
        let optimized =
            aleph_backend::run_optimized(&mut NaiveSvBackend::with_seed(0), &c).unwrap();
        assert_close(&reference, &optimized, &format!("run_optimized ghz n={n}"));
    }
}

// ---------------------------------------------------------------------------
// (C) Measure-qubit reported logical, not physical, when relabel fires.
// ---------------------------------------------------------------------------

#[test]
fn measure_qubit_reported_logical_after_relabel() {
    const TILE_BITS: u8 = 3;
    let n = 8u32;
    let high_q = n - 1; // logical qubit we measure (traffic-heavy, top bit)

    // High-qubit-heavy circuit (rebuilt with a classical bit) plus a
    // measurement on the top logical qubit. `high_qubit_heavy` declares 0
    // clbits, so rebuild it over a 1-clbit circuit for the measure.
    let base = high_qubit_heavy(n);
    let mut c = Circuit::new(n, 1);
    for inst in base.instructions() {
        c.add_instruction(inst.clone()).unwrap();
    }
    c.measure(high_q, 0).unwrap();

    // Force relabel; the pass rewrites the Measure qubit to a physical index.
    let mut opt = c.clone();
    RelabelQubits::new(TILE_BITS).run(&mut opt).unwrap();
    let perm = opt.qubit_permutation().expect("relabel must fire").to_vec();
    let physical_q = perm[high_q as usize];
    // Sanity: relabel actually moved the measured qubit to a low physical bit.
    assert_ne!(
        physical_q, high_q,
        "fixture must move the measured qubit (else the test is vacuous)"
    );

    // Run the relabelled circuit: outcomes come back against PHYSICAL qubits.
    let (_state, physical_outcomes) =
        run_with_outcomes(&mut NaiveSvBackend::with_seed(0), &opt).unwrap();
    let phys_rec = physical_outcomes
        .iter()
        .find(|r| r.clbit == 0)
        .expect("measurement recorded");
    assert_eq!(
        phys_rec.qubit, physical_q,
        "raw run records the PHYSICAL measure qubit"
    );

    // The driver tail maps it back to logical via invert_perm.
    let logical_of = invert_perm(&perm);
    assert_eq!(
        logical_of[phys_rec.qubit as usize], high_q,
        "driver must report the LOGICAL measure qubit ({high_q}), not physical ({physical_q})"
    );
}
