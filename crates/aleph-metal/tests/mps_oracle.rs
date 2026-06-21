//! P5.5-06: `MetalMpsBackend` dense-statevector oracle.
//!
//! Runs NN circuits through the scaffold GPU MPS backend and asserts the dense
//! statevector matches (a) the CPU MPS backend `aleph_mps::MpsBackend` — the
//! issue's acceptance criterion — and (b) the exact FP64 `NaiveSvBackend`, both
//! within 1e-5. With the bond cap above the circuits' entanglement, neither MPS
//! truncates, so all three agree to fp32 precision. Device-or-skip so
//! headless/Linux CI stays green.
//!
//! Run: `cargo test -p aleph-metal --features metal --test mps_oracle`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::{run, BackendError};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_metal::MetalMpsBackend;
use aleph_mps::MpsBackend;
use aleph_sv::NaiveSvBackend;

/// Bond cap large enough that the small NN test circuits never truncate.
const MAX_BOND: usize = 64;
const TOL: f64 = 1e-5;

fn assert_close(label: &str, actual: &[Complex<f64>], expected: &[Complex<f64>]) {
    assert_eq!(actual.len(), expected.len(), "{label}: dim mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.re.is_finite() && a.im.is_finite() && e.re.is_finite() && e.im.is_finite(),
            "{label}: non-finite amplitude at {i}: got {a:?} ref {e:?}"
        );
        let d = ((a.re - e.re).powi(2) + (a.im - e.im).powi(2)).sqrt();
        assert!(
            d <= TOL,
            "{label}: amp {i} |Δ|={d:.3e} > {TOL:.0e}\n  got {a:?}\n  ref {e:?}"
        );
    }
}

/// Dense statevector from the CPU MPS backend (the AC reference).
fn cpu_mps_dense(circuit: &Circuit) -> Vec<Complex<f64>> {
    let mut be = MpsBackend::new().with_max_bond(MAX_BOND);
    run(&mut be, circuit)
        .expect("cpu mps run")
        .dense_statevector()
}

/// Dense statevector from the exact FP64 CPU statevector backend.
fn naive_dense(circuit: &Circuit) -> Vec<Complex<f64>> {
    let mut be = NaiveSvBackend::with_seed(0);
    let state = run(&mut be, circuit).expect("naive run");
    aleph_oracle::HasAmplitudes::amplitudes(&state)
}

/// Run `circuit` on the GPU MPS scaffold and compare to both references — via the
/// gate-by-gate `run` *and* the layer-batched `run_batched` (P5.7-04), so both
/// execution paths are oracle-checked against the CPU MPS and the exact FP64 SV.
fn check(label: &str, gpu: &mut MetalMpsBackend, circuit: &Circuit) {
    let cpu = cpu_mps_dense(circuit);
    let naive = naive_dense(circuit);

    let got = gpu.run(circuit).expect("gpu mps run").dense_statevector();
    assert_close(&format!("{label} vs cpu-mps"), &got, &cpu);
    assert_close(&format!("{label} vs naive-sv"), &got, &naive);

    let got_b = gpu
        .run_batched(circuit)
        .expect("gpu mps run_batched")
        .dense_statevector();
    assert_close(&format!("{label} batched vs cpu-mps"), &got_b, &cpu);
    assert_close(&format!("{label} batched vs naive-sv"), &got_b, &naive);
}

#[test]
fn mps_metal_matches_cpu_on_nn_circuits() {
    let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping MPS Metal oracle: no Metal device available");
            return;
        }
    };

    // 1q-only sanity: H/S/T-style rotations spread across the chain (no 2q gate,
    // so this exercises only the 1q site kernel and the dense readout).
    let mut oneq = Circuit::new(4, 0);
    oneq.h(0).unwrap();
    oneq.h(1).unwrap();
    oneq.s(1).unwrap();
    oneq.t(2).unwrap();
    oneq.h(3).unwrap();
    check("1q-only", &mut gpu, &oneq);

    // GHZ is a NN CNOT chain — the canonical low-entanglement MPS case (bond 2).
    for n in [3u32, 5, 8, 10] {
        check(
            &format!("ghz_{n}"),
            &mut gpu,
            &aleph_benches::ghz_circuit(n),
        );
    }

    // NN brickwall: dense adjacent 2q gates → genuine bond growth (capped < 64),
    // the real test of contract + gate-apply + SVD split.
    for (n, depth) in [(4u32, 6usize), (6, 8), (8, 6)] {
        check(
            &format!("brickwall_{n}x{depth}"),
            &mut gpu,
            &aleph_benches::random_brickwall_circuit(n, depth),
        );
    }
}

/// P5.7-04 AC: a layer of disjoint NN 2q gates factored in one batched dispatch
/// must equal the gate-by-gate result. The circuit has a genuine disjoint 2q layer
/// — `CX(0,1)` and `CX(2,3)` with no shared site — plus a following `CX(1,2)` that
/// conflicts (so it lands in its own layer), exercising both the batch and the
/// flush-on-conflict path.
#[test]
fn batched_layer_matches_sequential() {
    let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping batched-vs-sequential: no Metal device available");
            return;
        }
    };
    let mut c = Circuit::new(4, 0);
    c.h(0).unwrap();
    c.h(2).unwrap();
    c.cnot(0, 1).unwrap(); // ┐ disjoint 2q layer → one batched dispatch
    c.cnot(2, 3).unwrap(); // ┘
    c.ry(0.3, 1).unwrap(); // 1q gates flush the layer
    c.ry(0.7, 2).unwrap();
    c.cnot(1, 2).unwrap(); // shares sites with both above → its own layer

    let seq = gpu.run(&c).expect("gate-by-gate").dense_statevector();
    let bat = gpu.run_batched(&c).expect("batched").dense_statevector();
    assert_close("batched-vs-sequential", &bat, &seq);
}

/// P5.7-06 AC: non-nearest-neighbour 2q gates (SWAP-routed) match the references.
/// A long-range, QAOA-style circuit — `H` layer, then 2q gates spanning gaps of
/// 2–5 sites, with interleaved 1q rotations — checked via both `run` and
/// `run_batched` against the CPU MPS and the exact FP64 SV.
#[test]
fn swap_routed_non_nn_matches_references() {
    let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping SWAP-router oracle: no Metal device available");
            return;
        }
    };
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.h(q).unwrap();
    }
    c.cnot(0, 5).unwrap(); // gap 5
    c.rz(0.4, 2).unwrap();
    c.cnot(1, 4).unwrap(); // gap 3
    c.cz(0, 3).unwrap(); // gap 3, CZ
    c.ry(0.9, 5).unwrap();
    c.cnot(2, 5).unwrap(); // gap 3
    c.cnot(0, 2).unwrap(); // gap 2
    check("swap_routed", &mut gpu, &c);
}

/// P5.7-06: SWAP routing is its own inverse — applying a non-NN CNOT twice returns
/// the exact pre-gate state (property reversibility; the SWAP network must unwind).
#[test]
fn swap_routed_is_reversible() {
    let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut prep = Circuit::new(5, 0);
    prep.h(0).unwrap();
    prep.ry(0.7, 2).unwrap();
    prep.h(4).unwrap();
    let before = gpu.run(&prep).expect("prep").dense_statevector();

    let mut twice = prep.clone();
    twice.cnot(0, 4).unwrap(); // non-NN
    twice.cnot(0, 4).unwrap(); // self-inverse ⇒ identity
    let after = gpu.run(&twice).expect("twice").dense_statevector();
    assert_close("cnot(0,4) twice == identity", &after, &before);
}

/// P5.6-02 AC: the truncating regime must be **refused**, not silently applied.
/// GHZ needs bond 2 (the CNOT entangles); with `max_bond = 1` the first two-site
/// split drops half the Schmidt weight, which this non-canonical scaffold cannot
/// renormalize, so `apply_gate` must return `MpsTruncationUnsupported` rather than
/// return a wrong state. The discarded weight is still surfaced via `trunc_error()`.
#[test]
fn truncation_below_required_bond_is_refused() {
    let mut gpu = match MetalMpsBackend::with_max_bond(1) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping MPS truncation guard: no Metal device available");
            return;
        }
    };
    // GHZ(3): H(0), CX(0,1), CX(1,2). The first CX forces bond 2 > cap 1.
    // (`MetalMpsState` isn't `Debug`, so match the Result rather than `expect_err`.)
    match gpu.run(&aleph_benches::ghz_circuit(3)) {
        Err(BackendError::MpsTruncationUnsupported {
            max_bond,
            trunc_error,
        }) => {
            assert_eq!(max_bond, 1);
            // A maximally-entangling CNOT drops ~half the squared Schmidt weight.
            assert!(
                trunc_error > 0.4,
                "expected a large dropped weight, got {trunc_error}"
            );
        }
        Err(other) => panic!("expected MpsTruncationUnsupported, got {other:?}"),
        Ok(_) => panic!("bond-1 GHZ must be refused, not truncated"),
    }
    // AC #3: the loss is surfaced, not silently dropped.
    assert!(
        gpu.trunc_error() > 0.4,
        "trunc_error accessor must reflect the refused split: {}",
        gpu.trunc_error()
    );

    // P5.7-04: the layer-batched path must refuse identically (a single-block
    // layer is still subject to the truncation guard before mutating state).
    let mut gpu_b = MetalMpsBackend::with_max_bond(1).expect("device");
    match gpu_b.run_batched(&aleph_benches::ghz_circuit(3)) {
        Err(BackendError::MpsTruncationUnsupported {
            max_bond,
            trunc_error,
        }) => {
            assert_eq!(max_bond, 1);
            assert!(trunc_error > 0.4, "batched dropped weight {trunc_error}");
        }
        Err(other) => panic!("expected MpsTruncationUnsupported (batched), got {other:?}"),
        Ok(_) => panic!("bond-1 GHZ must be refused on the batched path too"),
    }
}

/// AC #3 (positive side): a within-cap run stays exact and leaves the cumulative
/// truncation weight negligible, so the accessor cleanly distinguishes "exact"
/// from "truncated". GHZ(5) at bond 64 never truncates.
#[test]
fn exact_run_records_negligible_truncation() {
    let mut gpu = match MetalMpsBackend::with_max_bond(MAX_BOND) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping MPS exact-truncation check: no Metal device available");
            return;
        }
    };
    gpu.run(&aleph_benches::ghz_circuit(5))
        .expect("exact GHZ run");
    assert!(
        gpu.trunc_error() < 1e-9,
        "exact run must record ≈0 truncation, got {}",
        gpu.trunc_error()
    );
}

/// AC #2: report the CPU-SVD round-trip cost. Times the GPU contract+apply phase
/// against the host SVD split across an NN brickwall, printing the split. Ignored
/// (timing, not a pass/fail assertion); run with:
///   cargo test -p aleph-metal --features metal --test mps_oracle -- \
///     --ignored --nocapture report_svd_roundtrip_cost
#[test]
#[ignore = "timing report, not a correctness gate"]
fn report_svd_roundtrip_cost() {
    let mut gpu = match MetalMpsBackend::with_max_bond(128) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping SVD round-trip report: no Metal device");
            return;
        }
    };
    let (n, depth) = (12u32, 24usize);
    let circuit = aleph_benches::random_brickwall_circuit(n, depth);

    // Gate-by-gate (P5.7-03): one SVD dispatch + GPU sync per 2q gate.
    gpu.reset_timing();
    let _ = gpu.run(&circuit).expect("gpu mps run");
    let (gpu_ns, svd_ns) = gpu.timing_ns();
    let total = (gpu_ns + svd_ns) as f64;
    let pct = |x: u128| 100.0 * x as f64 / total;
    eprintln!(
        "SVD round-trip cost (NN brickwall n={n} depth={depth}):\n  \
         gate-by-gate (P5.7-03):\n    \
         GPU contract+apply: {:.2} ms ({:.1}%)\n    \
         GPU SVD split:      {:.2} ms ({:.1}%)\n    \
         split/contract:     {:.1}×",
        gpu_ns as f64 / 1e6,
        pct(gpu_ns),
        svd_ns as f64 / 1e6,
        pct(svd_ns),
        svd_ns as f64 / gpu_ns.max(1) as f64,
    );

    // Layer-batched (P5.7-04): one batched SVD dispatch per brickwall layer.
    gpu.reset_timing();
    let _ = gpu.run_batched(&circuit).expect("gpu mps run_batched");
    let (gpu_b, svd_b) = gpu.timing_ns();
    let total_b = (gpu_b + svd_b) as f64;
    let pct_b = |x: u128| 100.0 * x as f64 / total_b;
    eprintln!(
        "  layer-batched (P5.7-04):\n    \
         GPU contract+apply: {:.2} ms ({:.1}%)\n    \
         GPU SVD split:      {:.2} ms ({:.1}%)\n    \
         split/contract:     {:.1}×\n  \
         split speedup (gate-by-gate ÷ batched): {:.2}×",
        gpu_b as f64 / 1e6,
        pct_b(gpu_b),
        svd_b as f64 / 1e6,
        pct_b(svd_b),
        svd_b as f64 / gpu_b.max(1) as f64,
        svd_ns as f64 / svd_b.max(1) as f64,
    );
}
