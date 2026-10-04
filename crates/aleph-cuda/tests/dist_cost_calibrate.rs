//! P6-05 PR 2: per-kernel-kind calibration for `GpuCostModel` (spec §6.2).
//! Times 32 launches of one kind on a 2^m_ref state, each interleaved with an
//! `H` (real programs mix kernels; 32 identical launches back to back at the
//! card's power cap read 2–10 % slow), best of 5, and prints the `KindTimes`
//! literal to paste into `src/dist/cost.rs`.
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_cuda::{CudaContext, CudaSvBackend, CudaSvBackendF32};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};

const LAUNCHES: usize = 32;

/// One H layer (non-trivial amplitudes), optionally a scrambling prefix (one
/// `Rx` and one `Rz` per qubit, distinct angles: generic complex amplitudes),
/// then `LAUNCHES` × (`H` on qubit `i % 8`, then `payload(i)` if any). With no
/// payload this is the baseline.
fn circuit(n: u32, scramble: bool, payload: Option<&dyn Fn(usize) -> Instruction>) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    if scramble {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 + 0.11 * f64::from(q), q).unwrap();
        }
    }
    for i in 0..LAUNCHES {
        c.h((i % 8) as u32).unwrap();
        if let Some(make) = payload {
            c.add_instruction(make(i)).unwrap();
        }
    }
    c
}

fn g(gate: Gate, q: &[u32]) -> Instruction {
    Instruction::Gate(GateInstance::new(gate, q.to_vec()))
}

fn dp(n: u32, terms: impl Fn(u32) -> PhaseTerm, count: usize) -> Instruction {
    Instruction::DiagonalPhase(Box::new(DiagonalPhase {
        n_qubits: n,
        terms: (0..count as u32).map(terms).collect(),
    }))
}

/// Single-cond term: one 2-bit parity cond (fires on 1/2 of amplitudes).
fn single_term(n: u32) -> impl Fn(u32) -> PhaseTerm {
    move |t| PhaseTerm {
        conds: [(0b11u64 << (t % (n - 1))) & ((1u64 << n) - 1)]
            .into_iter()
            .collect(),
        angle: 0.01 * (f64::from(t) + 1.0),
    }
}

/// Multi-cond term: AND of two 1-bit conds `[1 << (n-1), 1 << s]`, `s ≠ n-1`
/// (fires on 1/4; the QFT controlled-phase shape).
fn multi_term(n: u32) -> impl Fn(u32) -> PhaseTerm {
    move |t| PhaseTerm {
        conds: [1u64 << (n - 1), 1u64 << (t % (n - 1))]
            .into_iter()
            .collect(),
        angle: 0.01 * (f64::from(t) + 1.0),
    }
}

/// Best-of-5 seconds per launch of the payload (interleaved-H baseline subtracted).
fn per_launch<F: FnMut(&Circuit) -> f64>(
    n: u32,
    time: F,
    make: impl Fn(usize) -> Instruction,
) -> f64 {
    per_launch_on(n, false, time, make)
}

/// [`per_launch`], with the scrambling prefix in both full and baseline when `scramble`.
fn per_launch_on<F: FnMut(&Circuit) -> f64>(
    n: u32,
    scramble: bool,
    mut time: F,
    make: impl Fn(usize) -> Instruction,
) -> f64 {
    let base = circuit(n, scramble, None);
    let full = circuit(n, scramble, Some(&make));
    let t_base = (0..5).map(|_| time(&base)).fold(f64::INFINITY, f64::min);
    let t_full = (0..5).map(|_| time(&full)).fold(f64::INFINITY, f64::min);
    (t_full - t_base) / LAUNCHES as f64
}

/// `[dense1, dense2, dense3, diag1, diag_k, cnot, phase_base, phase_term, phase_term_multi]`.
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
    // Dense2 is timed on a scrambled (generic complex) state; every other kind
    // on the uniform H state. Why (random-diagnosis H8): at the card's 70 W cap
    // FP64 kernel time depends on the amplitude data. Dense2 costs x1.196 on a
    // generic state vs the uniform one (Iswap 18.4 -> 22.1 ms), and real
    // Dense2 traffic (random's fused 2q blocks: 23.3 ms in situ) runs on such
    // states. Dense1 is bandwidth-bound and moves ~1.5 %. Caveat: this per-kind
    // choice was made AFTER seeing the §6.3 gate results. Scrambling Dense3 too
    // (x1.12) would push GHZ, whose state stays low-entropy, to ~1.11, so Dense3
    // stays uniform and carries a known state-dependent residual.
    let dense2 = per_launch_on(n, true, &mut *time, u2);
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
    // phase_base includes the per-launch device allocation + upload of the terms.
    let ph1 = per_launch(n, &mut *time, |_| dp(n, single_term(n), 1));
    let ph64 = per_launch(n, &mut *time, |_| dp(n, single_term(n), 64));
    let phase_term = (ph64 - ph1) / 63.0;
    let phase_base = ph1 - phase_term;
    // Multi-cond slope by the same 1-vs-64 fit; the model reuses `phase_base`.
    let pm1 = per_launch(n, &mut *time, |_| dp(n, multi_term(n), 1));
    let pm64 = per_launch(n, &mut *time, |_| dp(n, multi_term(n), 64));
    let phase_term_multi = (pm64 - pm1) / 63.0;
    [
        dense1,
        dense2,
        dense3,
        diag1,
        diag_k,
        cnot,
        phase_base,
        phase_term,
        phase_term_multi,
    ]
}

fn print(name: &str, m_ref: u32, k: [f64; 9]) {
    println!(
        "const {name}: KindTimes = KindTimes {{ m_ref: {m_ref}, dense1: {:.6e}, dense2: {:.6e}, dense3: {:.6e}, \
         diag1: {:.6e}, diag_k: {:.6e}, cnot: {:.6e}, phase_base: {:.6e}, phase_term: {:.6e}, \
         phase_term_multi: {:.6e} }};",
        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8]
    );
}

#[test]
#[ignore]
fn calibrate_kind_times() {
    let Ok(sync) = CudaContext::new(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let Ok(mut b64) = CudaSvBackend::with_seed(0) else {
        eprintln!("skipped: no CUDA");
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
        eprintln!("skipped: no CUDA");
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
