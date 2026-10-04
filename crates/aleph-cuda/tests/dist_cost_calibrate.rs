//! P6-05 PR 2: per-kernel-kind calibration for `GpuCostModel` (spec §6.2).
//! Times 32 launches of one kind on a 2^m_ref state, best of 5, and prints the
//! `KindTimes` literal to paste into `src/dist/cost.rs`.
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_cuda::{CudaContext, CudaSvBackend, CudaSvBackendF32};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};

const LAUNCHES: usize = 32;

/// `LAUNCHES` copies of `make(i)` after one H layer (non-trivial amplitudes).
fn circuit(n: u32, make: impl Fn(usize) -> Instruction) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for i in 0..LAUNCHES {
        c.add_instruction(make(i)).unwrap();
    }
    c
}

fn g(gate: Gate, q: &[u32]) -> Instruction {
    Instruction::Gate(GateInstance::new(gate, q.to_vec()))
}

fn dp(n: u32, terms: usize) -> Instruction {
    Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: n,
        terms: (0..terms)
            .map(|t| PhaseTerm {
                conds: [(0b11u64 << (t as u32 % (n - 1))) & ((1u64 << n) - 1)]
                    .into_iter()
                    .collect(),
                angle: 0.01 * (t as f64 + 1.0),
            })
            .collect(),
    }))
}

/// Best-of-5 seconds per launch of the payload (H-layer baseline subtracted).
fn per_launch<F: FnMut(&Circuit) -> f64>(
    n: u32,
    mut time: F,
    make: impl Fn(usize) -> Instruction,
) -> f64 {
    let base = {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.h(q).unwrap();
        }
        c
    };
    let full = circuit(n, make);
    let t_base = (0..5).map(|_| time(&base)).fold(f64::INFINITY, f64::min);
    let t_full = (0..5).map(|_| time(&full)).fold(f64::INFINITY, f64::min);
    (t_full - t_base) / LAUNCHES as f64
}

fn kinds(n: u32, time: &mut dyn FnMut(&Circuit) -> f64) -> [f64; 9] {
    let p = Param::Concrete;
    let u2 = |i: usize| {
        // Non-diagonal, non-Cnot dense 2q: Iswap.
        g(Gate::Iswap, &[(i % 3) as u32, 3 + (i % 3) as u32])
    };
    let mut kq = vec![Complex::new(0.0, 0.0); 64];
    for r in 0..8 {
        kq[r * 8 + (r ^ 1)] = Complex::new(1.0, 0.0); // permutation, not diagonal
    }
    let dense1 = per_launch(n, &mut *time, |i| g(Gate::H, &[(i % 8) as u32]));
    let dense2 = per_launch(n, &mut *time, u2);
    let dense3 = per_launch(n, &mut *time, |i| {
        g(
            Gate::UnitaryKq {
                k: 3,
                data: kq.clone().into_boxed_slice(),
            },
            &[(i % 4) as u32, 5, 6],
        )
    });
    let diag1 = per_launch(n, &mut *time, |i| {
        g(Gate::Rz(p(0.1 + i as f64)), &[(i % 8) as u32])
    });
    let diag_k = per_launch(n, &mut *time, |i| g(Gate::Cz, &[(i % 4) as u32, 5]));
    let cnot = per_launch(n, &mut *time, |i| g(Gate::Cnot, &[(i % 4) as u32, 5]));
    let ph1 = per_launch(n, &mut *time, |_| dp(n, 1));
    let ph64 = per_launch(n, &mut *time, |_| dp(n, 64));
    let phase_term = (ph64 - ph1) / 63.0;
    let phase_base = ph1 - phase_term;
    [
        dense1, dense2, dense3, diag1, diag_k, cnot, phase_base, phase_term, 0.0,
    ]
}

fn print(name: &str, m_ref: u32, k: [f64; 9]) {
    println!(
        "const {name}: KindTimes = KindTimes {{ m_ref: {m_ref}, dense1: {:.6e}, dense2: {:.6e}, dense3: {:.6e}, \
         diag1: {:.6e}, diag_k: {:.6e}, cnot: {:.6e}, phase_base: {:.6e}, phase_term: {:.6e} }};",
        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]
    );
}

#[test]
#[ignore]
fn calibrate_kind_times() {
    let Ok(sync) = CudaContext::new(0) else {
        return;
    };
    let Ok(mut b64) = CudaSvBackend::with_seed(0) else {
        return;
    };
    let mut t64 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b64, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    print("RTX4000_FP64", 27, kinds(27, &mut t64));
    let Ok(mut b32) = CudaSvBackendF32::with_seed(0) else {
        return;
    };
    let mut t32 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b32, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    print("RTX4000_FP32", 28, kinds(28, &mut t32));
}
