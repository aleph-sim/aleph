//! Shared per-kernel-kind timing for the state microbench (#538 Stage A):
//! the interleaved-with-H method of P6-05 PR 2, on a chosen prepared state.
#![allow(dead_code)]

use aleph_core::{Complex, Gate, GateInstance, Param};
use aleph_ir::{Circuit, DiagonalPhase, Instruction, PhaseTerm};

use super::dist::clifford_layers;

const LAUNCHES: usize = 32;

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

/// The six prepared states of spec §2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Uniform: H on every qubit.
    A,
    /// A + Rz(0.7 + 0.11q): equal magnitudes, varied phases.
    B,
    /// A + Ry(0.3 + 0.17q): varied magnitudes, real.
    C,
    /// A + Rx(0.3 + 0.17q) + Rz(0.7 + 0.11q): generic complex.
    D,
    /// GHZ-like: H(0), CNOT chain.
    E,
    /// Clifford: 4 `clifford_layers`.
    F,
}

pub const STATES: [State; 6] = [State::A, State::B, State::C, State::D, State::E, State::F];

/// The state's preparation from |0…0⟩, in both payload and baseline circuits.
pub fn prefix(state: State, n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    match state {
        State::E => {
            c.h(0).unwrap();
            for q in 0..n - 1 {
                c.cnot(q, q + 1).unwrap();
            }
        }
        State::F => clifford_layers(&mut c, n, 4),
        _ => {
            for q in 0..n {
                c.h(q).unwrap();
            }
            for q in 0..n {
                let f = f64::from(q);
                match state {
                    State::B => {
                        c.rz(0.7 + 0.11 * f, q).unwrap();
                    }
                    State::C => {
                        c.ry(0.3 + 0.17 * f, q).unwrap();
                    }
                    State::D => {
                        c.rx(0.3 + 0.17 * f, q).unwrap();
                        c.rz(0.7 + 0.11 * f, q).unwrap();
                    }
                    _ => {}
                }
            }
        }
    }
    c
}

/// `prefix(state)`, then `LAUNCHES` × (`H` on qubit `i % 8`, then `payload(i)`
/// if any). With no payload this is the baseline.
fn circuit(n: u32, state: State, payload: Option<&dyn Fn(usize) -> Instruction>) -> Circuit {
    let mut c = prefix(state, n);
    for i in 0..LAUNCHES {
        c.h((i % 8) as u32).unwrap();
        if let Some(make) = payload {
            c.add_instruction(make(i)).unwrap();
        }
    }
    c
}

/// Best-of-5 seconds per launch of the payload, the interleaved-H baseline on
/// the same state subtracted.
fn per_launch(
    n: u32,
    state: State,
    time: &mut dyn FnMut(&Circuit) -> f64,
    make: impl Fn(usize) -> Instruction,
) -> f64 {
    let base = circuit(n, state, None);
    let full = circuit(n, state, Some(&make));
    let t_base = (0..5).map(|_| time(&base)).fold(f64::INFINITY, f64::min);
    let t_full = (0..5).map(|_| time(&full)).fold(f64::INFINITY, f64::min);
    (t_full - t_base) / LAUNCHES as f64
}

pub const KIND_NAMES: [&str; 9] = [
    "dense1",
    "dense2",
    "dense3",
    "diag1",
    "diag_k",
    "cnot",
    "phase_base",
    "phase_term",
    "phase_term_multi",
];

/// `[dense1, dense2, dense3, diag1, diag_k, cnot, phase_base, phase_term, phase_term_multi]`.
pub fn kinds(n: u32, state: State, time: &mut dyn FnMut(&Circuit) -> f64) -> [f64; 9] {
    let p = Param::Concrete;
    let u2 = |i: usize| {
        // Non-diagonal, non-Cnot dense 2q: Iswap.
        g(Gate::Iswap, &[(i % 3) as u32, 3 + (i % 3) as u32])
    };
    let mut kq = vec![Complex::new(0.0, 0.0); 64];
    for r in 0..8 {
        kq[r * 8 + (r ^ 1)] = Complex::new(1.0, 0.0); // permutation, not diagonal
    }
    let dense1 = per_launch(n, state, &mut *time, |i| g(Gate::H, &[(i % 8) as u32]));
    let dense2 = per_launch(n, state, &mut *time, u2);
    let dense3 = per_launch(n, state, &mut *time, |i| {
        g(
            Gate::UnitaryKq {
                k: 3,
                data: kq.clone().into_boxed_slice(),
            },
            &[(i % 4) as u32, 5, 6],
        )
    });
    let diag1 = per_launch(n, state, &mut *time, |i| {
        g(Gate::Rz(p(0.1 + i as f64)), &[(i % 8) as u32])
    });
    let diag_k = per_launch(n, state, &mut *time, |i| g(Gate::Cz, &[(i % 4) as u32, 5]));
    let cnot = per_launch(n, state, &mut *time, |i| {
        g(Gate::Cnot, &[(i % 4) as u32, 5])
    });
    // phase_base includes the per-launch device allocation + upload of the terms.
    let ph1 = per_launch(n, state, &mut *time, |_| dp(n, single_term(n), 1));
    let ph64 = per_launch(n, state, &mut *time, |_| dp(n, single_term(n), 64));
    let phase_term = (ph64 - ph1) / 63.0;
    let phase_base = ph1 - phase_term;
    // Multi-cond slope by the same 1-vs-64 fit; the model reuses `phase_base`.
    let pm1 = per_launch(n, state, &mut *time, |_| dp(n, multi_term(n), 1));
    let pm64 = per_launch(n, state, &mut *time, |_| dp(n, multi_term(n), 64));
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
