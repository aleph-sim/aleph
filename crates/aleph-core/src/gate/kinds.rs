//! `Gate` enum and its core methods (`matrix`, `is_diagonal`,
//! `is_clifford`, `inverse`, `arity`).
//!
//! Qubit ordering convention: every variant's documentation specifies
//! the order of qubits in `GateInstance::qubits`. Backends rely on this
//! contract — violating it silently mis-applies the gate.

use crate::Complex;
use crate::gate::Param;

/// Canonical quantum gate representation used by the IR and backends.
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    // --- 1q standard ---
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,

    // --- 1q parametric ---
    /// `Rx(θ) = exp(-i θ X / 2)`.
    Rx(Param),
    /// `Ry(θ) = exp(-i θ Y / 2)`.
    Ry(Param),
    /// `Rz(θ) = exp(-i θ Z / 2)`.
    Rz(Param),
    /// `Phase(θ) = diag(1, e^(iθ))`.
    Phase(Param),
    /// `U3(θ, φ, λ)` — generic single-qubit rotation (Qiskit convention).
    U3(Param, Param, Param),

    // --- 2q standard ---
    /// `Cnot` — controlled-X. `qubits = [control, target]`.
    Cnot,
    /// `Cz` — controlled-Z. `qubits = [q0, q1]` (symmetric).
    Cz,
    /// `Swap`. `qubits = [q0, q1]`.
    Swap,
    /// `Iswap` — `|01⟩ ↔ i|10⟩`. `qubits = [q0, q1]`.
    Iswap,

    // --- 2q parametric ---
    /// `CRx(θ)`. `qubits = [control, target]`.
    CRx(Param),
    /// `CRy(θ)`. `qubits = [control, target]`.
    CRy(Param),
    /// `CRz(θ)`. `qubits = [control, target]`.
    CRz(Param),

    // --- 3q standard ---
    /// `Toffoli` (CCX). `qubits = [c0, c1, target]`.
    Toffoli,
    /// `Ccz`. `qubits = [c0, c1, c2]` (symmetric).
    Ccz,

    // --- arbitrary unitary, owned ---
    /// 1-qubit arbitrary unitary. Boxed to keep the enum small in the
    /// common case of standard gates.
    Unitary1q(Box<[[Complex; 2]; 2]>),
    /// 2-qubit arbitrary unitary. Boxed for the same reason.
    Unitary2q(Box<[[Complex; 4]; 4]>),
}

impl Gate {
    /// Number of target qubits (1, 2, or 3). Does not count generic
    /// external controls carried on a `GateInstance`.
    pub fn arity(&self) -> usize {
        match self {
            Gate::H
            | Gate::X
            | Gate::Y
            | Gate::Z
            | Gate::S
            | Gate::Sdg
            | Gate::T
            | Gate::Tdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::Phase(_)
            | Gate::U3(_, _, _)
            | Gate::Unitary1q(_) => 1,

            Gate::Cnot
            | Gate::Cz
            | Gate::Swap
            | Gate::Iswap
            | Gate::CRx(_)
            | Gate::CRy(_)
            | Gate::CRz(_)
            | Gate::Unitary2q(_) => 2,

            Gate::Toffoli | Gate::Ccz => 3,
        }
    }

    /// Unitary matrix of the gate in the computational basis.
    ///
    /// Returns `Err(GateError::SymbolicParam)` if any parameter is
    /// `Param::Symbolic`. In Phase 0 this branch is unreachable through
    /// the public API.
    pub fn matrix(&self) -> Result<crate::gate::GateMatrix, crate::gate::GateError> {
        use crate::gate::GateMatrix;
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);

        Ok(match self {
            Gate::H => {
                let s = Complex::new(FRAC_1_SQRT_2, 0.0);
                GateMatrix::M2x2([[s, s], [s, -s]])
            }
            Gate::X => GateMatrix::M2x2([[zero, one], [one, zero]]),
            Gate::Y => {
                let pi = Complex::new(0.0, 1.0);
                let ni = Complex::new(0.0, -1.0);
                GateMatrix::M2x2([[zero, ni], [pi, zero]])
            }
            Gate::Z => GateMatrix::M2x2([[one, zero], [zero, -one]]),
            Gate::S => {
                let i = Complex::new(0.0, 1.0);
                GateMatrix::M2x2([[one, zero], [zero, i]])
            }
            Gate::Sdg => {
                let ni = Complex::new(0.0, -1.0);
                GateMatrix::M2x2([[one, zero], [zero, ni]])
            }
            Gate::T => {
                let p = Complex::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
                GateMatrix::M2x2([[one, zero], [zero, p]])
            }
            Gate::Tdg => {
                let p = Complex::new(FRAC_PI_4.cos(), -FRAC_PI_4.sin());
                GateMatrix::M2x2([[one, zero], [zero, p]])
            }

            // Remaining variants get implemented in subsequent tasks.
            _ => unimplemented!("matrix() arm for {self:?} not yet wired up"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AMPLITUDE_TOL, gate::GateMatrix};
    use std::f64::consts::FRAC_1_SQRT_2;

    fn cc(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    fn approx_eq_m2(actual: &[[Complex; 2]; 2], expected: &[[Complex; 2]; 2]) {
        for i in 0..2 {
            for j in 0..2 {
                let a = actual[i][j];
                let e = expected[i][j];
                assert!(
                    (a.re - e.re).abs() < AMPLITUDE_TOL
                        && (a.im - e.im).abs() < AMPLITUDE_TOL,
                    "mismatch at [{i}][{j}]: actual={a:?} expected={e:?}"
                );
            }
        }
    }

    fn unwrap_m2(g: &Gate) -> [[Complex; 2]; 2] {
        match g.matrix().expect("concrete") {
            GateMatrix::M2x2(m) => m,
            other => panic!("expected M2x2, got {other:?}"),
        }
    }

    fn boxed_id1() -> Box<[[Complex; 2]; 2]> {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        Box::new([[one, zero], [zero, one]])
    }

    fn boxed_id2() -> Box<[[Complex; 4]; 4]> {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        Box::new([
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [zero, zero, zero, one],
        ])
    }

    #[test]
    fn arity_1q_standard() {
        for g in [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::Sdg, Gate::T, Gate::Tdg] {
            assert_eq!(g.arity(), 1, "{g:?}");
        }
    }

    #[test]
    fn arity_1q_parametric() {
        let p = Param::Concrete(0.0);
        for g in [
            Gate::Rx(p), Gate::Ry(p), Gate::Rz(p), Gate::Phase(p),
            Gate::U3(p, p, p),
        ] {
            assert_eq!(g.arity(), 1, "{g:?}");
        }
        assert_eq!(Gate::Unitary1q(boxed_id1()).arity(), 1);
    }

    #[test]
    fn arity_2q() {
        let p = Param::Concrete(0.0);
        for g in [
            Gate::Cnot, Gate::Cz, Gate::Swap, Gate::Iswap,
            Gate::CRx(p), Gate::CRy(p), Gate::CRz(p),
        ] {
            assert_eq!(g.arity(), 2, "{g:?}");
        }
        assert_eq!(Gate::Unitary2q(boxed_id2()).arity(), 2);
    }

    #[test]
    fn arity_3q() {
        assert_eq!(Gate::Toffoli.arity(), 3);
        assert_eq!(Gate::Ccz.arity(), 3);
    }

    #[test]
    fn matrix_h() {
        let s = FRAC_1_SQRT_2;
        let expected = [[cc(s, 0.0), cc(s, 0.0)], [cc(s, 0.0), cc(-s, 0.0)]];
        approx_eq_m2(&unwrap_m2(&Gate::H), &expected);
    }

    #[test]
    fn matrix_x() {
        let expected = [[cc(0.0, 0.0), cc(1.0, 0.0)], [cc(1.0, 0.0), cc(0.0, 0.0)]];
        approx_eq_m2(&unwrap_m2(&Gate::X), &expected);
    }

    #[test]
    fn matrix_y() {
        let expected = [[cc(0.0, 0.0), cc(0.0, -1.0)], [cc(0.0, 1.0), cc(0.0, 0.0)]];
        approx_eq_m2(&unwrap_m2(&Gate::Y), &expected);
    }

    #[test]
    fn matrix_z() {
        let expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(-1.0, 0.0)]];
        approx_eq_m2(&unwrap_m2(&Gate::Z), &expected);
    }

    #[test]
    fn matrix_s_and_sdg() {
        let s_expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(0.0, 1.0)]];
        let sdg_expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(0.0, -1.0)]];
        approx_eq_m2(&unwrap_m2(&Gate::S), &s_expected);
        approx_eq_m2(&unwrap_m2(&Gate::Sdg), &sdg_expected);
    }

    #[test]
    fn matrix_t_and_tdg() {
        let phase = std::f64::consts::FRAC_PI_4;
        let t_expected = [
            [cc(1.0, 0.0), cc(0.0, 0.0)],
            [cc(0.0, 0.0), cc(phase.cos(), phase.sin())],
        ];
        let tdg_expected = [
            [cc(1.0, 0.0), cc(0.0, 0.0)],
            [cc(0.0, 0.0), cc(phase.cos(), -phase.sin())],
        ];
        approx_eq_m2(&unwrap_m2(&Gate::T), &t_expected);
        approx_eq_m2(&unwrap_m2(&Gate::Tdg), &tdg_expected);
    }
}
