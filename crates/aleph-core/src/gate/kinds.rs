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
        use crate::gate::{GateError, GateMatrix};
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

        fn concrete(p: Param) -> Result<f64, GateError> {
            match p {
                Param::Concrete(v) => Ok(v),
                Param::Symbolic(_) => Err(GateError::SymbolicParam),
            }
        }

        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);

        match self {
            Gate::H => {
                let s = Complex::new(FRAC_1_SQRT_2, 0.0);
                Ok(GateMatrix::M2x2([[s, s], [s, -s]]))
            }
            Gate::X => Ok(GateMatrix::M2x2([[zero, one], [one, zero]])),
            Gate::Y => Ok(GateMatrix::M2x2([
                [zero, Complex::new(0.0, -1.0)],
                [Complex::new(0.0, 1.0), zero],
            ])),
            Gate::Z => Ok(GateMatrix::M2x2([[one, zero], [zero, -one]])),
            Gate::S => Ok(GateMatrix::M2x2([
                [one, zero],
                [zero, Complex::new(0.0, 1.0)],
            ])),
            Gate::Sdg => Ok(GateMatrix::M2x2([
                [one, zero],
                [zero, Complex::new(0.0, -1.0)],
            ])),
            Gate::T => Ok(GateMatrix::M2x2([
                [one, zero],
                [zero, Complex::new(FRAC_PI_4.cos(), FRAC_PI_4.sin())],
            ])),
            Gate::Tdg => Ok(GateMatrix::M2x2([
                [one, zero],
                [zero, Complex::new(FRAC_PI_4.cos(), -FRAC_PI_4.sin())],
            ])),

            Gate::Rx(p) => {
                let t = concrete(*p)?;
                let c = Complex::new((t / 2.0).cos(), 0.0);
                let nis = Complex::new(0.0, -(t / 2.0).sin());
                Ok(GateMatrix::M2x2([[c, nis], [nis, c]]))
            }
            Gate::Ry(p) => {
                let t = concrete(*p)?;
                let c = Complex::new((t / 2.0).cos(), 0.0);
                let s = Complex::new((t / 2.0).sin(), 0.0);
                Ok(GateMatrix::M2x2([[c, -s], [s, c]]))
            }
            Gate::Rz(p) => {
                let t = concrete(*p)?;
                let neg = Complex::new((t / 2.0).cos(), -(t / 2.0).sin());
                let pos = Complex::new((t / 2.0).cos(), (t / 2.0).sin());
                Ok(GateMatrix::M2x2([[neg, zero], [zero, pos]]))
            }
            Gate::Phase(p) => {
                let t = concrete(*p)?;
                let e = Complex::new(t.cos(), t.sin());
                Ok(GateMatrix::M2x2([[one, zero], [zero, e]]))
            }
            Gate::U3(theta, phi, lambda) => {
                let t = concrete(*theta)?;
                let f = concrete(*phi)?;
                let l = concrete(*lambda)?;
                let c = Complex::new((t / 2.0).cos(), 0.0);
                let s = Complex::new((t / 2.0).sin(), 0.0);
                let e_l = Complex::new(l.cos(), l.sin());
                let e_f = Complex::new(f.cos(), f.sin());
                let e_fl = Complex::new((f + l).cos(), (f + l).sin());
                Ok(GateMatrix::M2x2([[c, -e_l * s], [e_f * s, e_fl * c]]))
            }

            Gate::Cnot => Ok(GateMatrix::M4x4([
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
            ])),
            Gate::Cz => Ok(GateMatrix::M4x4([
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, -one],
            ])),
            Gate::Swap => Ok(GateMatrix::M4x4([
                [one, zero, zero, zero],
                [zero, zero, one, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
            ])),
            Gate::Iswap => {
                let i = Complex::new(0.0, 1.0);
                Ok(GateMatrix::M4x4([
                    [one, zero, zero, zero],
                    [zero, zero, i, zero],
                    [zero, i, zero, zero],
                    [zero, zero, zero, one],
                ]))
            }

            Gate::CRx(p) => {
                let t = concrete(*p)?;
                let c = Complex::new((t / 2.0).cos(), 0.0);
                let nis = Complex::new(0.0, -(t / 2.0).sin());
                Ok(GateMatrix::M4x4([
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, c, nis],
                    [zero, zero, nis, c],
                ]))
            }
            Gate::CRy(p) => {
                let t = concrete(*p)?;
                let c = Complex::new((t / 2.0).cos(), 0.0);
                let s = Complex::new((t / 2.0).sin(), 0.0);
                Ok(GateMatrix::M4x4([
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, c, -s],
                    [zero, zero, s, c],
                ]))
            }
            Gate::CRz(p) => {
                let t = concrete(*p)?;
                let neg = Complex::new((t / 2.0).cos(), -(t / 2.0).sin());
                let pos = Complex::new((t / 2.0).cos(), (t / 2.0).sin());
                Ok(GateMatrix::M4x4([
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, neg, zero],
                    [zero, zero, zero, pos],
                ]))
            }

            Gate::Toffoli => {
                let mut m = [[zero; 8]; 8];
                for i in 0..6 {
                    m[i][i] = one;
                }
                m[6][7] = one;
                m[7][6] = one;
                Ok(GateMatrix::M8x8(m))
            }
            Gate::Ccz => {
                let mut m = [[zero; 8]; 8];
                for i in 0..7 {
                    m[i][i] = one;
                }
                m[7][7] = -one;
                Ok(GateMatrix::M8x8(m))
            }

            _ => unimplemented!("matrix() arm for {self:?} not yet wired up"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AMPLITUDE_TOL, gate::{GateError, GateMatrix}};
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

    #[test]
    fn matrix_rx_zero_is_identity() {
        let m = unwrap_m2(&Gate::Rx(Param::Concrete(0.0)));
        let expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(1.0, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_rx_pi_is_minus_i_x() {
        // Rx(π) = [[0, -i], [-i, 0]]
        let m = unwrap_m2(&Gate::Rx(Param::Concrete(std::f64::consts::PI)));
        let expected = [[cc(0.0, 0.0), cc(0.0, -1.0)], [cc(0.0, -1.0), cc(0.0, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_ry_half_pi() {
        // Ry(π/2) = (1/√2) [[1, -1], [1, 1]]
        let s = FRAC_1_SQRT_2;
        let m = unwrap_m2(&Gate::Ry(Param::Concrete(std::f64::consts::FRAC_PI_2)));
        let expected = [[cc(s, 0.0), cc(-s, 0.0)], [cc(s, 0.0), cc(s, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_rz_pi_is_iz_minus_phase() {
        // Rz(π) = diag(e^(-iπ/2), e^(iπ/2)) = diag(-i, i)
        let m = unwrap_m2(&Gate::Rz(Param::Concrete(std::f64::consts::PI)));
        let expected = [[cc(0.0, -1.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(0.0, 1.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_phase_pi_is_z() {
        let m = unwrap_m2(&Gate::Phase(Param::Concrete(std::f64::consts::PI)));
        let expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(-1.0, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_u3_zero_is_identity() {
        let m = unwrap_m2(&Gate::U3(
            Param::Concrete(0.0),
            Param::Concrete(0.0),
            Param::Concrete(0.0),
        ));
        let expected = [[cc(1.0, 0.0), cc(0.0, 0.0)], [cc(0.0, 0.0), cc(1.0, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_symbolic_param_errors() {
        let g = Gate::Rx(Param::Symbolic(crate::gate::SymbolId(0)));
        assert_eq!(g.matrix(), Err(GateError::SymbolicParam));
    }

    fn approx_eq_m4(actual: &[[Complex; 4]; 4], expected: &[[Complex; 4]; 4]) {
        for i in 0..4 {
            for j in 0..4 {
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

    fn unwrap_m4(g: &Gate) -> [[Complex; 4]; 4] {
        match g.matrix().expect("concrete") {
            GateMatrix::M4x4(m) => m,
            other => panic!("expected M4x4, got {other:?}"),
        }
    }

    #[test]
    fn matrix_cnot() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let expected = [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, z, o],
            [z, z, o, z],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::Cnot), &expected);
    }

    #[test]
    fn matrix_cz() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let no = cc(-1.0, 0.0);
        let expected = [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, o, z],
            [z, z, z, no],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::Cz), &expected);
    }

    #[test]
    fn matrix_swap() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let expected = [
            [o, z, z, z],
            [z, z, o, z],
            [z, o, z, z],
            [z, z, z, o],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::Swap), &expected);
    }

    #[test]
    fn matrix_iswap() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let i = cc(0.0, 1.0);
        let expected = [
            [o, z, z, z],
            [z, z, i, z],
            [z, i, z, z],
            [z, z, z, o],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::Iswap), &expected);
    }

    #[test]
    fn matrix_crx_zero_is_identity() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let id4 = [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]];
        approx_eq_m4(&unwrap_m4(&Gate::CRx(Param::Concrete(0.0))), &id4);
        approx_eq_m4(&unwrap_m4(&Gate::CRy(Param::Concrete(0.0))), &id4);
        approx_eq_m4(&unwrap_m4(&Gate::CRz(Param::Concrete(0.0))), &id4);
    }

    #[test]
    fn matrix_crx_pi_acts_as_minus_ix_on_target() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let ni = cc(0.0, -1.0);
        let expected = [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, z, ni],
            [z, z, ni, z],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::CRx(Param::Concrete(std::f64::consts::PI))), &expected);
    }

    #[test]
    fn matrix_crz_pi() {
        // CRz(π) sets target sub-block to diag(-i, i)
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let expected = [
            [o, z, z, z],
            [z, o, z, z],
            [z, z, cc(0.0, -1.0), z],
            [z, z, z, cc(0.0, 1.0)],
        ];
        approx_eq_m4(&unwrap_m4(&Gate::CRz(Param::Concrete(std::f64::consts::PI))), &expected);
    }

    fn approx_eq_m8(actual: &[[Complex; 8]; 8], expected: &[[Complex; 8]; 8]) {
        for i in 0..8 {
            for j in 0..8 {
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

    fn unwrap_m8(g: &Gate) -> [[Complex; 8]; 8] {
        match g.matrix().expect("concrete") {
            GateMatrix::M8x8(m) => m,
            other => panic!("expected M8x8, got {other:?}"),
        }
    }

    fn identity_m8() -> [[Complex; 8]; 8] {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..8 {
            m[i][i] = o;
        }
        m
    }

    #[test]
    fn matrix_toffoli() {
        let mut expected = identity_m8();
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        // Swap rows/cols 6 and 7 (|110⟩ ↔ |111⟩).
        expected[6][6] = z;
        expected[7][7] = z;
        expected[6][7] = o;
        expected[7][6] = o;
        approx_eq_m8(&unwrap_m8(&Gate::Toffoli), &expected);
    }

    #[test]
    fn matrix_ccz() {
        let mut expected = identity_m8();
        expected[7][7] = cc(-1.0, 0.0);
        approx_eq_m8(&unwrap_m8(&Gate::Ccz), &expected);
    }
}
