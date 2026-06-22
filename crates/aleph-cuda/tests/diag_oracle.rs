//! Oracle tests for the custom diagonal-gate kernels (P5-06).
//!
//! Every diagonal-gate path — `apply_diag_1q` (Z/S/T/Rz/Phase and their
//! controlled forms) and the general `apply_diag` (Cz/CRz at k=2, Ccz at k=3) —
//! must match the CPU `NaiveSvBackend` amplitude-for-amplitude at the full FP64
//! tolerance, *and* must agree with the dense fallback path so the routing is a
//! pure optimisation. Both GPU backends (hand-written + cuStateVec) are pinned.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips cleanly
//! when no CUDA device is present, so a GPU-less host (CI) is a pass.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-10;

/// A circuit that exercises *every* diagonal kernel path plus non-diagonal gates
/// (which must keep falling through to the dense path). Non-uniform rotations up
/// front so amplitudes are all distinct — a uniform state would mask an
/// operand-order bug in the multi-qubit diagonal mapping.
fn diag_mix(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    // Distinct seed amplitudes (non-diagonal: dense path).
    for q in 0..n {
        c.ry(0.3 + 0.37 * q as f64, q).unwrap();
    }
    // 1q diagonal: apply_diag_1q.
    for q in 0..n {
        c.z(q).unwrap();
        c.s(q).unwrap();
        c.t(q).unwrap();
        c.rz(0.2 + 0.11 * q as f64, q).unwrap();
        c.add_gate(GateInstance::new(
            Gate::Phase(Param::Concrete(0.45)),
            vec![q],
        ))
        .unwrap();
    }
    // 2q diagonal both-operand: Cz and CRz → general apply_diag (k=2).
    let mut q = 0;
    while q + 1 < n {
        c.add_gate(GateInstance::new(Gate::Cz, vec![q, q + 1]))
            .unwrap();
        c.add_gate(GateInstance::new(
            Gate::CRz(Param::Concrete(0.6)),
            vec![q, q + 1],
        ))
        .unwrap();
        q += 1;
    }
    // 3q diagonal: Ccz → general apply_diag (k=3).
    if n >= 3 {
        c.add_gate(GateInstance::new(Gate::Ccz, vec![0, 1, 2]))
            .unwrap();
    }
    // Controlled 1q diagonal: controlled-Phase (the QFT workhorse) and a
    // multi-controlled Z (Grover diffusion core) → apply_diag_1q + ctrl_mask.
    if n >= 2 {
        c.add_gate(GateInstance::controlled(
            Gate::Phase(Param::Concrete(0.8)),
            vec![n - 1],
            vec![0],
        ))
        .unwrap();
    }
    // Multi-controlled Z (Grover diffusion core). The IR caps controls at 8, so
    // this exercises the `ctrl_mask` path only up to n=9.
    if (2..=9).contains(&n) {
        c.add_gate(GateInstance::controlled(
            Gate::Z,
            vec![n - 1],
            (0..n - 1).collect::<Vec<_>>(),
        ))
        .unwrap();
    }
    // Non-diagonal trailer (must stay on the dense path): Cnot + H.
    if n >= 2 {
        c.cnot(0, 1).unwrap();
    }
    c.h(0).unwrap();
    c
}

/// Full QFT — a controlled-Phase-dominated circuit; the diagonal routing must
/// reproduce it exactly.
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

fn cpu_amps(circuit: &Circuit) -> Vec<aleph_core::Complex> {
    let mut cpu = NaiveSvBackend::with_seed(0);
    HasAmplitudes::amplitudes(&run(&mut cpu, circuit).expect("cpu run"))
}

fn assert_match(name: &str, got: &[aleph_core::Complex], want: &[aleph_core::Complex]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).norm();
        assert!(
            d <= TOL,
            "{name} i={i}: |Δ|={d:.3e} > {TOL:e}\n got={a} want={b}"
        );
    }
}

/// Build the hand-written backend with custom-diag on/off. Returns `None` when no
/// CUDA device (caller skips the suite).
fn sv_backend(custom: bool) -> Option<CudaSvBackend> {
    match CudaSvBackend::with_seed(0) {
        Ok(b) => Some(b.with_custom_kernels(custom)),
        Err(e) => {
            eprintln!("skipping P5-06 diag oracle: {e}");
            None
        }
    }
}

#[test]
fn cuda_sv_diag_matches_cpu_and_dense() {
    for n in 2..=10u32 {
        for circuit in [diag_mix(n), qft(n)] {
            let want = cpu_amps(&circuit);

            let Some(mut on) = sv_backend(true) else {
                return;
            };
            let got_on = HasAmplitudes::amplitudes(&run(&mut on, &circuit).expect("gpu on"));
            assert_match(&format!("sv custom-on n={n}"), &got_on, &want);

            // Dense fallback must be bit-for-bit identical to the custom path
            // (same FP64 math), proving the routing is a pure optimisation.
            let mut off = sv_backend(false).expect("device present above");
            let got_off = HasAmplitudes::amplitudes(&run(&mut off, &circuit).expect("gpu off"));
            assert_match(&format!("sv custom-off n={n}"), &got_off, &want);
        }
    }
}

/// cuStateVec backend: the same diagonal gates must match the CPU whether routed
/// to the custom kernel (default) or through `custatevecApplyMatrix`.
#[cfg(feature = "cuquantum")]
#[test]
fn custatevec_diag_matches_cpu_and_cuquantum() {
    use aleph_cuda::CuStateVecBackend;

    fn backend(custom: bool) -> Option<CuStateVecBackend> {
        match CuStateVecBackend::with_seed(0) {
            Ok(b) => Some(b.with_custom_kernels(custom)),
            Err(e) => {
                eprintln!("skipping P5-06 cuStateVec diag oracle: {e}");
                None
            }
        }
    }

    for n in 2..=10u32 {
        for circuit in [diag_mix(n), qft(n)] {
            let want = cpu_amps(&circuit);

            let Some(mut on) = backend(true) else { return };
            let got_on = HasAmplitudes::amplitudes(&run(&mut on, &circuit).expect("cusv on"));
            assert_match(&format!("cusv custom-on n={n}"), &got_on, &want);

            let mut off = backend(false).expect("device present above");
            let got_off = HasAmplitudes::amplitudes(&run(&mut off, &circuit).expect("cusv off"));
            assert_match(&format!("cusv custom-off n={n}"), &got_off, &want);
        }
    }
}
