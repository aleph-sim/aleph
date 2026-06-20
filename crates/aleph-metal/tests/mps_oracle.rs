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

/// Run `circuit` on the GPU MPS scaffold and compare to both references.
fn check(label: &str, gpu: &mut MetalMpsBackend, circuit: &Circuit) {
    let got = gpu.run(circuit).expect("gpu mps run").dense_statevector();
    assert_close(
        &format!("{label} vs cpu-mps"),
        &got,
        &cpu_mps_dense(circuit),
    );
    assert_close(&format!("{label} vs naive-sv"), &got, &naive_dense(circuit));
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
    gpu.reset_timing();
    let _ = gpu.run(&circuit).expect("gpu mps run");
    let (gpu_ns, svd_ns) = gpu.timing_ns();
    let total = (gpu_ns + svd_ns) as f64;
    let pct = |x: u128| 100.0 * x as f64 / total;
    eprintln!(
        "SVD round-trip cost (NN brickwall n={n} depth={depth}):\n  \
         GPU contract+apply: {:.2} ms ({:.1}%)\n  \
         host SVD split:     {:.2} ms ({:.1}%)\n  \
         host/GPU ratio:     {:.1}×",
        gpu_ns as f64 / 1e6,
        pct(gpu_ns),
        svd_ns as f64 / 1e6,
        pct(svd_ns),
        svd_ns as f64 / gpu_ns.max(1) as f64,
    );
}
