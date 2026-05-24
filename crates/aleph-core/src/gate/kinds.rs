//! `Gate` enum and its core methods (`matrix`, `is_diagonal`,
//! `is_clifford`, `inverse`, `arity`).
//!
//! Qubit ordering convention: every variant's documentation specifies
//! the order of qubits in `GateInstance::qubits`. Backends rely on this
//! contract — violating it silently mis-applies the gate.

use crate::gate::Param;
use crate::Complex;

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
    /// `IswapDg` — adjoint of `Iswap`: `|01⟩ ↔ -i|10⟩`. `qubits = [q0, q1]`.
    /// Added so that `Iswap.inverse()` stays inside the Clifford group
    /// instead of falling back to `Unitary2q` (which would defeat
    /// stabilizer-backend dispatch on `is_clifford()`).
    IswapDg,

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
    ///
    /// Matrix convention: the inner `[[Complex; 2]; 2]` is the gate's
    /// matrix in the computational basis `{|0⟩, |1⟩}`, applied to the
    /// single qubit at `GateInstance::qubits[0]`. Row `i`, column `j`
    /// is `⟨i|U|j⟩`. No unitarity check is performed at construction
    /// — supplying a non-unitary matrix produces a backend that
    /// violates `||ψ||² = 1` invariants.
    Unitary1q(Box<[[Complex; 2]; 2]>),
    /// 2-qubit arbitrary unitary. Boxed for the same reason.
    ///
    /// Matrix convention: the inner `[[Complex; 4]; 4]` is the gate's
    /// matrix in the computational basis `{|q0 q1⟩}` indexed `00, 01,
    /// 10, 11` (i.e. `qubits[0]` is the **most significant bit** of
    /// the row/column index, matching the [`Gate::Cnot`] layout in
    /// this file where `qubits = [control, target]` and the matrix
    /// permutes rows/cols 2 ↔ 3). Row `i`, column `j` is `⟨i|U|j⟩`.
    /// No unitarity check is performed at construction.
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
            | Gate::IswapDg
            | Gate::CRx(_)
            | Gate::CRy(_)
            | Gate::CRz(_)
            | Gate::Unitary2q(_) => 2,

            Gate::Toffoli | Gate::Ccz => 3,
        }
    }

    /// Unitary matrix of the gate in the computational basis.
    ///
    /// Returns:
    /// - `Err(GateError::SymbolicParam)` if any parameter is
    ///   `Param::Symbolic` (unreachable through the public API in
    ///   Phase 0; reserved for Phase 4 VQE work).
    /// - `Err(GateError::NonFiniteParam)` if any parameter is NaN
    ///   or infinite — these would otherwise propagate as silent
    ///   NaN entries through `cos`/`sin`.
    ///
    /// # Precision notes
    ///
    /// Parametric angles are passed straight into `cos`/`sin` with no
    /// `mod 2π` reduction. For `|θ|` in the typical `[-2π, 2π]` range
    /// the resulting matrix is unitary to well within `AMPLITUDE_TOL`
    /// (`1e-10`). For very large `|θ|` (e.g. `>= 1e10`) argument-
    /// reduction precision loss makes `cos²+sin²` deviate measurably
    /// from `1`; callers that synthesize such angles should reduce
    /// them themselves before calling `matrix()`.
    ///
    /// `Unitary1q`/`Unitary2q` copy their `[[Complex; N]; N]` payload
    /// out of the box on each call (256 B for `Unitary2q`). Backends
    /// that read the same matrix many times should cache the result
    /// rather than re-calling `matrix()` in a tight loop.
    pub fn matrix(&self) -> Result<crate::gate::GateMatrix, crate::gate::GateError> {
        use crate::gate::{GateError, GateMatrix};
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

        fn concrete(p: Param) -> Result<f64, GateError> {
            match p {
                Param::Concrete(v) if v.is_finite() => Ok(v),
                Param::Concrete(_) => Err(GateError::NonFiniteParam),
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
            Gate::IswapDg => {
                let ni = Complex::new(0.0, -1.0);
                Ok(GateMatrix::M4x4([
                    [one, zero, zero, zero],
                    [zero, zero, ni, zero],
                    [zero, ni, zero, zero],
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
                for (i, row) in m.iter_mut().enumerate().take(6) {
                    row[i] = one;
                }
                m[6][7] = one;
                m[7][6] = one;
                Ok(GateMatrix::M8x8(m))
            }
            Gate::Ccz => {
                let mut m = [[zero; 8]; 8];
                for (i, row) in m.iter_mut().enumerate().take(7) {
                    row[i] = one;
                }
                m[7][7] = -one;
                Ok(GateMatrix::M8x8(m))
            }

            Gate::Unitary1q(m) => Ok(GateMatrix::M2x2(**m)),
            Gate::Unitary2q(m) => Ok(GateMatrix::M4x4(**m)),
        }
    }

    /// Whether the matrix is diagonal in the computational basis.
    ///
    /// Reports the **algebraic structure**, not the numerical case —
    /// `Rx(0.0).is_diagonal()` is `false` even though that particular
    /// matrix is the identity. Backends that special-case diagonal
    /// gates can therefore trust this flag without per-angle checks.
    ///
    /// Written as an exhaustive `match` (not `matches!`) so the
    /// compiler forces every new `Gate` variant to declare its
    /// diagonality explicitly.
    pub fn is_diagonal(&self) -> bool {
        match self {
            Gate::Z
            | Gate::S
            | Gate::Sdg
            | Gate::T
            | Gate::Tdg
            | Gate::Rz(_)
            | Gate::Phase(_)
            | Gate::CRz(_)
            | Gate::Cz
            | Gate::Ccz => true,

            Gate::H
            | Gate::X
            | Gate::Y
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::U3(_, _, _)
            | Gate::Cnot
            | Gate::Swap
            | Gate::Iswap
            | Gate::IswapDg
            | Gate::CRx(_)
            | Gate::CRy(_)
            | Gate::Toffoli
            | Gate::Unitary1q(_)
            | Gate::Unitary2q(_) => false,
        }
    }

    /// Whether the gate belongs to the Clifford group.
    ///
    /// All parametric variants return `false` even for Clifford-equivalent
    /// angles (e.g. `Rx(π/2)`). See `docs/decisions/0002-gate-clifford-detection.md`
    /// — Phase 2 stabilizer work will revisit angle-aware detection.
    ///
    /// Written as an exhaustive `match` (not `matches!`) so the
    /// compiler forces every new `Gate` variant to declare its
    /// Clifford-ness explicitly.
    pub fn is_clifford(&self) -> bool {
        match self {
            Gate::H
            | Gate::X
            | Gate::Y
            | Gate::Z
            | Gate::S
            | Gate::Sdg
            | Gate::Cnot
            | Gate::Cz
            | Gate::Swap
            | Gate::Iswap
            | Gate::IswapDg => true,

            Gate::T
            | Gate::Tdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::Phase(_)
            | Gate::U3(_, _, _)
            | Gate::CRx(_)
            | Gate::CRy(_)
            | Gate::CRz(_)
            | Gate::Toffoli
            | Gate::Ccz
            | Gate::Unitary1q(_)
            | Gate::Unitary2q(_) => false,
        }
    }

    /// Inverse gate: same arity, conjugate-transpose of the matrix.
    ///
    /// Self-inverse variants are returned unchanged. Adjoint pairs swap.
    /// Parametric rotations negate their angle (for `U3(θ, φ, λ)` the
    /// result is `U3(-θ, -λ, -φ)`). `Iswap` ↔ `IswapDg` form an
    /// adjoint pair so the Clifford classification is closed under
    /// inverse (both report `is_clifford() == true`).
    pub fn inverse(&self) -> Gate {
        match self {
            // self-inverse
            Gate::H
            | Gate::X
            | Gate::Y
            | Gate::Z
            | Gate::Cnot
            | Gate::Cz
            | Gate::Swap
            | Gate::Toffoli
            | Gate::Ccz => self.clone(),

            Gate::S => Gate::Sdg,
            Gate::Sdg => Gate::S,
            Gate::T => Gate::Tdg,
            Gate::Tdg => Gate::T,
            Gate::Iswap => Gate::IswapDg,
            Gate::IswapDg => Gate::Iswap,

            Gate::Rx(p) => Gate::Rx(negate(*p)),
            Gate::Ry(p) => Gate::Ry(negate(*p)),
            Gate::Rz(p) => Gate::Rz(negate(*p)),
            Gate::Phase(p) => Gate::Phase(negate(*p)),
            Gate::CRx(p) => Gate::CRx(negate(*p)),
            Gate::CRy(p) => Gate::CRy(negate(*p)),
            Gate::CRz(p) => Gate::CRz(negate(*p)),

            Gate::U3(t, f, l) => Gate::U3(negate(*t), negate(*l), negate(*f)),

            Gate::Unitary1q(m) => Gate::Unitary1q(Box::new(conj_transpose_2(m))),
            Gate::Unitary2q(m) => Gate::Unitary2q(Box::new(conj_transpose_4(m))),
        }
    }
}

fn negate(p: Param) -> Param {
    match p {
        Param::Concrete(v) => Param::Concrete(-v),
        // Phase 0 has no public way to construct Symbolic, but if a future
        // refactor exposes it we want negation to round-trip the symbol
        // unchanged — the consumer can substitute later.
        Param::Symbolic(s) => Param::Symbolic(s),
    }
}

fn conj_transpose_2(m: &[[Complex; 2]; 2]) -> [[Complex; 2]; 2] {
    let mut out = [[Complex::new(0.0, 0.0); 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = m[j][i].conj();
        }
    }
    out
}

fn conj_transpose_4(m: &[[Complex; 4]; 4]) -> [[Complex; 4]; 4] {
    let mut out = [[Complex::new(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = m[j][i].conj();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gate::{GateError, GateMatrix},
        AMPLITUDE_TOL,
    };
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
                    (a.re - e.re).abs() < AMPLITUDE_TOL && (a.im - e.im).abs() < AMPLITUDE_TOL,
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
        for g in [
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
        ] {
            assert_eq!(g.arity(), 1, "{g:?}");
        }
    }

    #[test]
    fn arity_1q_parametric() {
        let p = Param::Concrete(0.0);
        for g in [
            Gate::Rx(p),
            Gate::Ry(p),
            Gate::Rz(p),
            Gate::Phase(p),
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
            Gate::Cnot,
            Gate::Cz,
            Gate::Swap,
            Gate::Iswap,
            Gate::IswapDg,
            Gate::CRx(p),
            Gate::CRy(p),
            Gate::CRz(p),
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
    fn matrix_u3_pi_0_pi_is_x() {
        // Qiskit convention: U3(π, 0, π) == X (no global phase).
        let m = unwrap_m2(&Gate::U3(
            Param::Concrete(std::f64::consts::PI),
            Param::Concrete(0.0),
            Param::Concrete(std::f64::consts::PI),
        ));
        let expected = [[cc(0.0, 0.0), cc(1.0, 0.0)], [cc(1.0, 0.0), cc(0.0, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_u3_halfpi_0_pi_is_h() {
        // Qiskit convention: U3(π/2, 0, π) == H (no global phase).
        let s = FRAC_1_SQRT_2;
        let m = unwrap_m2(&Gate::U3(
            Param::Concrete(std::f64::consts::FRAC_PI_2),
            Param::Concrete(0.0),
            Param::Concrete(std::f64::consts::PI),
        ));
        let expected = [[cc(s, 0.0), cc(s, 0.0)], [cc(s, 0.0), cc(-s, 0.0)]];
        approx_eq_m2(&m, &expected);
    }

    #[test]
    fn matrix_u3_phi_lambda_order_distinguishable() {
        // Guard against silently swapped φ↔λ: U3(π/2, π/3, 0) and
        // U3(π/2, 0, π/3) must differ. With the Qiskit formula,
        // the bottom-left of U3(θ, φ, λ) is e^(iφ)·sin(θ/2) and the
        // top-right is -e^(iλ)·sin(θ/2). Swapping φ↔λ would swap
        // which entry carries the e^(iπ/3) phase.
        let theta = std::f64::consts::FRAC_PI_2;
        let pi3 = std::f64::consts::FRAC_PI_3;
        let m1 = unwrap_m2(&Gate::U3(
            Param::Concrete(theta),
            Param::Concrete(pi3),
            Param::Concrete(0.0),
        ));
        let m2 = unwrap_m2(&Gate::U3(
            Param::Concrete(theta),
            Param::Concrete(0.0),
            Param::Concrete(pi3),
        ));
        // m1 has the phase on bottom-left (e_phi · s), m2 on top-right.
        let s = (theta / 2.0).sin();
        let cos_pi3 = pi3.cos();
        let sin_pi3 = pi3.sin();
        // m1[1][0] = e^(iπ/3) · s, m2[1][0] = 1 · s (real).
        let m1_bottom_left = m1[1][0];
        let m2_bottom_left = m2[1][0];
        assert!((m1_bottom_left.re - cos_pi3 * s).abs() < AMPLITUDE_TOL);
        assert!((m1_bottom_left.im - sin_pi3 * s).abs() < AMPLITUDE_TOL);
        assert!((m2_bottom_left.re - s).abs() < AMPLITUDE_TOL);
        assert!(m2_bottom_left.im.abs() < AMPLITUDE_TOL);
    }

    fn expect_err(g: &Gate, want: GateError) {
        match g.matrix() {
            Err(e) => assert_eq!(e, want, "matrix() error mismatch for {g:?}"),
            Ok(_) => panic!("matrix() returned Ok for {g:?}, expected {want:?}"),
        }
    }

    #[test]
    fn matrix_symbolic_param_errors() {
        let g = Gate::Rx(Param::Symbolic(crate::gate::SymbolId(0)));
        expect_err(&g, GateError::SymbolicParam);
    }

    #[test]
    fn matrix_nan_param_errors() {
        let g = Gate::Rx(Param::Concrete(f64::NAN));
        expect_err(&g, GateError::NonFiniteParam);
    }

    #[test]
    fn matrix_infinite_param_errors() {
        for inf in [f64::INFINITY, f64::NEG_INFINITY] {
            let g = Gate::Phase(Param::Concrete(inf));
            expect_err(&g, GateError::NonFiniteParam);
        }
    }

    #[test]
    fn matrix_u3_propagates_nonfinite_from_any_param() {
        // Validate that all three U3 params are checked, not just θ.
        let nan = Param::Concrete(f64::NAN);
        let ok = Param::Concrete(0.0);
        expect_err(&Gate::U3(nan, ok, ok), GateError::NonFiniteParam);
        expect_err(&Gate::U3(ok, nan, ok), GateError::NonFiniteParam);
        expect_err(&Gate::U3(ok, ok, nan), GateError::NonFiniteParam);
    }

    fn approx_eq_m4(actual: &[[Complex; 4]; 4], expected: &[[Complex; 4]; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                let a = actual[i][j];
                let e = expected[i][j];
                assert!(
                    (a.re - e.re).abs() < AMPLITUDE_TOL && (a.im - e.im).abs() < AMPLITUDE_TOL,
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
        let expected = [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]];
        approx_eq_m4(&unwrap_m4(&Gate::Cnot), &expected);
    }

    #[test]
    fn matrix_cz() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let no = cc(-1.0, 0.0);
        let expected = [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, no]];
        approx_eq_m4(&unwrap_m4(&Gate::Cz), &expected);
    }

    #[test]
    fn matrix_swap() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let expected = [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, o]];
        approx_eq_m4(&unwrap_m4(&Gate::Swap), &expected);
    }

    #[test]
    fn matrix_iswap() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let i = cc(0.0, 1.0);
        let expected = [[o, z, z, z], [z, z, i, z], [z, i, z, z], [z, z, z, o]];
        approx_eq_m4(&unwrap_m4(&Gate::Iswap), &expected);
    }

    #[test]
    fn matrix_iswap_dg() {
        // iSWAP† = [[1,0,0,0],[0,0,-i,0],[0,-i,0,0],[0,0,0,1]]
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let ni = cc(0.0, -1.0);
        let expected = [[o, z, z, z], [z, z, ni, z], [z, ni, z, z], [z, z, z, o]];
        approx_eq_m4(&unwrap_m4(&Gate::IswapDg), &expected);
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
        let expected = [[o, z, z, z], [z, o, z, z], [z, z, z, ni], [z, z, ni, z]];
        approx_eq_m4(
            &unwrap_m4(&Gate::CRx(Param::Concrete(std::f64::consts::PI))),
            &expected,
        );
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
        approx_eq_m4(
            &unwrap_m4(&Gate::CRz(Param::Concrete(std::f64::consts::PI))),
            &expected,
        );
    }

    fn approx_eq_m8(actual: &[[Complex; 8]; 8], expected: &[[Complex; 8]; 8]) {
        for i in 0..8 {
            for j in 0..8 {
                let a = actual[i][j];
                let e = expected[i][j];
                assert!(
                    (a.re - e.re).abs() < AMPLITUDE_TOL && (a.im - e.im).abs() < AMPLITUDE_TOL,
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
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = o;
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

    #[test]
    fn matrix_unitary1q_roundtrip() {
        let s = FRAC_1_SQRT_2;
        let h = [[cc(s, 0.0), cc(s, 0.0)], [cc(s, 0.0), cc(-s, 0.0)]];
        let g = Gate::Unitary1q(Box::new(h));
        approx_eq_m2(&unwrap_m2(&g), &h);
    }

    #[test]
    fn matrix_unitary2q_roundtrip() {
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let m = [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]];
        let g = Gate::Unitary2q(Box::new(m));
        approx_eq_m4(&unwrap_m4(&g), &m);
    }

    #[test]
    fn inverse_self_inverse_set() {
        for g in [
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::Cnot,
            Gate::Cz,
            Gate::Swap,
            Gate::Toffoli,
            Gate::Ccz,
        ] {
            assert_eq!(g.clone().inverse(), g, "{g:?}");
        }
    }

    #[test]
    fn inverse_adjoint_pairs() {
        assert_eq!(Gate::S.inverse(), Gate::Sdg);
        assert_eq!(Gate::Sdg.inverse(), Gate::S);
        assert_eq!(Gate::T.inverse(), Gate::Tdg);
        assert_eq!(Gate::Tdg.inverse(), Gate::T);
    }

    #[test]
    fn inverse_parametric_negates() {
        let theta = 0.7;
        let p = Param::Concrete(theta);
        let np = Param::Concrete(-theta);
        assert_eq!(Gate::Rx(p).inverse(), Gate::Rx(np));
        assert_eq!(Gate::Ry(p).inverse(), Gate::Ry(np));
        assert_eq!(Gate::Rz(p).inverse(), Gate::Rz(np));
        assert_eq!(Gate::Phase(p).inverse(), Gate::Phase(np));
        assert_eq!(Gate::CRx(p).inverse(), Gate::CRx(np));
        assert_eq!(Gate::CRy(p).inverse(), Gate::CRy(np));
        assert_eq!(Gate::CRz(p).inverse(), Gate::CRz(np));
    }

    #[test]
    fn inverse_u3_swaps_phi_lambda() {
        let theta = Param::Concrete(0.3);
        let phi = Param::Concrete(0.5);
        let lambda = Param::Concrete(0.7);
        let expected = Gate::U3(
            Param::Concrete(-0.3),
            Param::Concrete(-0.7),
            Param::Concrete(-0.5),
        );
        assert_eq!(Gate::U3(theta, phi, lambda).inverse(), expected);
    }

    #[test]
    fn inverse_iswap_pair() {
        // Iswap ↔ IswapDg form an adjoint pair (closed under inverse,
        // both Clifford). Multiplying their matrices must give identity.
        assert_eq!(Gate::Iswap.inverse(), Gate::IswapDg);
        assert_eq!(Gate::IswapDg.inverse(), Gate::Iswap);

        let m = unwrap_m4(&Gate::Iswap);
        let m_dg = unwrap_m4(&Gate::IswapDg);
        // (m · m_dg) should equal the 4x4 identity.
        let mut prod = [[cc(0.0, 0.0); 4]; 4];
        for (i, row) in prod.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for k in 0..4 {
                    *cell += m[i][k] * m_dg[k][j];
                }
            }
        }
        let z = cc(0.0, 0.0);
        let o = cc(1.0, 0.0);
        let id4 = [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]];
        approx_eq_m4(&prod, &id4);
    }

    #[test]
    fn is_clifford_truth_table() {
        let p = Param::Concrete(std::f64::consts::FRAC_PI_2);
        let cliff = [
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::Cnot,
            Gate::Cz,
            Gate::Swap,
            Gate::Iswap,
            Gate::IswapDg,
        ];
        for g in &cliff {
            assert!(g.is_clifford(), "{g:?} should be Clifford");
        }

        // Parametric always false in Phase 0 (see ADR-0002), even at π/2.
        let non_cliff = [
            Gate::T,
            Gate::Tdg,
            Gate::Rx(p),
            Gate::Ry(p),
            Gate::Rz(p),
            Gate::Phase(p),
            Gate::U3(p, p, p),
            Gate::CRx(p),
            Gate::CRy(p),
            Gate::CRz(p),
            Gate::Toffoli,
            Gate::Ccz,
            Gate::Unitary1q(Box::new([[Complex::new(0.0, 0.0); 2]; 2])),
            Gate::Unitary2q(Box::new([[Complex::new(0.0, 0.0); 4]; 4])),
        ];
        for g in &non_cliff {
            assert!(!g.is_clifford(), "{g:?} should not be Clifford");
        }
    }

    #[test]
    fn is_diagonal_truth_table() {
        let p = Param::Concrete(0.5);
        let diag_gates = [
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::Rz(p),
            Gate::Phase(p),
            Gate::CRz(p),
            Gate::Cz,
            Gate::Ccz,
        ];
        for g in &diag_gates {
            assert!(g.is_diagonal(), "{g:?} should be diagonal");
        }

        let nondiag_gates = [
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Rx(p),
            Gate::Ry(p),
            Gate::U3(p, p, p),
            Gate::Cnot,
            Gate::Swap,
            Gate::Iswap,
            Gate::IswapDg,
            Gate::CRx(p),
            Gate::CRy(p),
            Gate::Toffoli,
            Gate::Unitary1q(Box::new([[Complex::new(0.0, 0.0); 2]; 2])),
            Gate::Unitary2q(Box::new([[Complex::new(0.0, 0.0); 4]; 4])),
        ];
        for g in &nondiag_gates {
            assert!(!g.is_diagonal(), "{g:?} should not be diagonal");
        }
    }
}
