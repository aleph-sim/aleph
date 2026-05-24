//! Quantum gate representation: `Gate`, `GateInstance`, `GateMatrix`,
//! `Param`. See `docs/superpowers/specs/2026-05-24-p0-06-gate-enum-design.md`.

mod error;
mod instance;
mod kinds;
mod matrix;
mod param;

pub use error::GateError;
pub use instance::GateInstance;
pub use kinds::Gate;
pub use matrix::GateMatrix;
pub use param::{Param, SymbolId};

#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    use super::{Gate, GateMatrix, Param};
    use crate::{Complex, AMPLITUDE_TOL};

    // --- matrix helpers ---

    fn mul2(a: &[[Complex; 2]; 2], b: &[[Complex; 2]; 2]) -> [[Complex; 2]; 2] {
        let mut out = [[Complex::new(0.0, 0.0); 2]; 2];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for k in 0..2 {
                    *cell += a[i][k] * b[k][j];
                }
            }
        }
        out
    }

    fn mul4(a: &[[Complex; 4]; 4], b: &[[Complex; 4]; 4]) -> [[Complex; 4]; 4] {
        let mut out = [[Complex::new(0.0, 0.0); 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for k in 0..4 {
                    *cell += a[i][k] * b[k][j];
                }
            }
        }
        out
    }

    fn mul8(a: &[[Complex; 8]; 8], b: &[[Complex; 8]; 8]) -> [[Complex; 8]; 8] {
        let mut out = [[Complex::new(0.0, 0.0); 8]; 8];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for k in 0..8 {
                    *cell += a[i][k] * b[k][j];
                }
            }
        }
        out
    }

    fn conj_t<const N: usize>(m: &[[Complex; N]; N]) -> [[Complex; N]; N] {
        let mut out = [[Complex::new(0.0, 0.0); N]; N];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = m[j][i].conj();
            }
        }
        out
    }

    fn assert_identity<const N: usize>(m: &[[Complex; N]; N]) {
        for (i, row) in m.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                let want_re = if i == j { 1.0 } else { 0.0 };
                let want_im = 0.0;
                assert!(
                    (c.re - want_re).abs() < AMPLITUDE_TOL
                        && (c.im - want_im).abs() < AMPLITUDE_TOL,
                    "non-identity at [{i}][{j}]: {c:?}"
                );
            }
        }
    }

    fn mul_gm_inv(m: &GateMatrix, inv: &GateMatrix) {
        match (m, inv) {
            (GateMatrix::M2x2(a), GateMatrix::M2x2(b)) => {
                assert_identity(&mul2(a, b));
                assert_identity(&mul2(b, a));
            }
            (GateMatrix::M4x4(a), GateMatrix::M4x4(b)) => {
                assert_identity(&mul4(a, b));
                assert_identity(&mul4(b, a));
            }
            (GateMatrix::M8x8(a), GateMatrix::M8x8(b)) => {
                assert_identity(&mul8(a, b));
                assert_identity(&mul8(b, a));
            }
            _ => panic!("arity mismatch between gate and its inverse"),
        }
    }

    fn unitary_check(m: &GateMatrix) {
        match m {
            GateMatrix::M2x2(a) => assert_identity(&mul2(a, &conj_t(a))),
            GateMatrix::M4x4(a) => assert_identity(&mul4(a, &conj_t(a))),
            GateMatrix::M8x8(a) => assert_identity(&mul8(a, &conj_t(a))),
        }
    }

    fn arity_matches(g: &Gate, m: &GateMatrix) {
        let n = g.arity();
        match m {
            GateMatrix::M2x2(_) => assert_eq!(n, 1, "{g:?}"),
            GateMatrix::M4x4(_) => assert_eq!(n, 2, "{g:?}"),
            GateMatrix::M8x8(_) => assert_eq!(n, 3, "{g:?}"),
        }
    }

    // --- generators ---

    fn arb_angle() -> impl Strategy<Value = f64> {
        // Generous range covers Clifford-equivalent angles + a few cycles.
        -(2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI)
    }

    fn arb_param() -> impl Strategy<Value = Param> {
        arb_angle().prop_map(Param::Concrete)
    }

    fn standard_gates() -> Vec<Gate> {
        vec![
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::Cnot,
            Gate::Cz,
            Gate::Swap,
            Gate::Iswap,
            Gate::IswapDg,
            Gate::Toffoli,
            Gate::Ccz,
        ]
    }

    // --- properties ---

    #[test]
    fn standard_gates_are_unitary_and_match_arity() {
        for g in standard_gates() {
            let m = g.matrix().expect("standard gate");
            unitary_check(&m);
            arity_matches(&g, &m);
        }
    }

    #[test]
    fn standard_gates_inverse_round_trip() {
        for g in standard_gates() {
            let m = g.matrix().expect("standard");
            let inv = g.inverse().matrix().expect("inverse of standard");
            mul_gm_inv(&m, &inv);
        }
    }

    proptest! {
        #[test]
        fn parametric_1q_unitary(p in arb_param()) {
            for g in [Gate::Rx(p), Gate::Ry(p), Gate::Rz(p), Gate::Phase(p)] {
                let m = g.matrix().unwrap();
                unitary_check(&m);
                arity_matches(&g, &m);
            }
        }

        #[test]
        fn u3_unitary(a in arb_param(), b in arb_param(), c in arb_param()) {
            let g = Gate::U3(a, b, c);
            let m = g.matrix().unwrap();
            unitary_check(&m);
            arity_matches(&g, &m);
        }

        #[test]
        fn parametric_2q_unitary(p in arb_param()) {
            for g in [Gate::CRx(p), Gate::CRy(p), Gate::CRz(p)] {
                let m = g.matrix().unwrap();
                unitary_check(&m);
                arity_matches(&g, &m);
            }
        }

        #[test]
        fn parametric_inverse_round_trip(p in arb_param()) {
            for g in [
                Gate::Rx(p), Gate::Ry(p), Gate::Rz(p), Gate::Phase(p),
                Gate::CRx(p), Gate::CRy(p), Gate::CRz(p),
            ] {
                let m = g.matrix().unwrap();
                let inv = g.inverse().matrix().unwrap();
                mul_gm_inv(&m, &inv);
            }
        }

        #[test]
        fn u3_inverse_round_trip(a in arb_param(), b in arb_param(), c in arb_param()) {
            let g = Gate::U3(a, b, c);
            let m = g.matrix().unwrap();
            let inv = g.inverse().matrix().unwrap();
            mul_gm_inv(&m, &inv);
        }

        #[test]
        fn parametric_negate_equals_inverse(p in arb_param()) {
            let np = match p { Param::Concrete(v) => Param::Concrete(-v), _ => unreachable!() };
            for (g, expected) in [
                (Gate::Rx(p), Gate::Rx(np)),
                (Gate::Ry(p), Gate::Ry(np)),
                (Gate::Rz(p), Gate::Rz(np)),
                (Gate::Phase(p), Gate::Phase(np)),
                (Gate::CRx(p), Gate::CRx(np)),
                (Gate::CRy(p), Gate::CRy(np)),
                (Gate::CRz(p), Gate::CRz(np)),
            ] {
                prop_assert_eq!(g.inverse(), expected);
            }
        }

        // --- Coverage for the Unitary*.inverse() paths (conj_transpose_2/4) ---
        //
        // We need actual unitary matrices to exercise inverse round-trip;
        // a random complex matrix isn't unitary. Strategy: build U from
        // a 1-parameter family that is provably unitary
        // (U = cos(t)·I + i·sin(t)·H, where H is a Hermitian — here Pauli-X
        // for the 1q case and X⊗X for the 2q case). The matrix isn't a
        // standard named variant, so it exercises the Unitary* arm.
        #[test]
        fn unitary1q_inverse_round_trip(t in arb_angle()) {
            let c = Complex::new(t.cos(), 0.0);
            let is = Complex::new(0.0, t.sin());
            // cos(t)·I + i·sin(t)·X = [[cos, i·sin], [i·sin, cos]] — unitary.
            let m = [[c, is], [is, c]];
            let g = Gate::Unitary1q(Box::new(m));
            let fwd = g.matrix().unwrap();
            let inv = g.inverse().matrix().unwrap();
            mul_gm_inv(&fwd, &inv);
        }

        #[test]
        fn unitary2q_inverse_round_trip(t in arb_angle()) {
            let c = Complex::new(t.cos(), 0.0);
            let is = Complex::new(0.0, t.sin());
            let z = Complex::new(0.0, 0.0);
            // cos(t)·I + i·sin(t)·(X⊗X) — unitary 4x4.
            // X⊗X has 1s on the anti-diagonal.
            let m = [
                [c, z, z, is],
                [z, c, is, z],
                [z, is, c, z],
                [is, z, z, c],
            ];
            let g = Gate::Unitary2q(Box::new(m));
            let fwd = g.matrix().unwrap();
            let inv = g.inverse().matrix().unwrap();
            mul_gm_inv(&fwd, &inv);
        }
    }
}
