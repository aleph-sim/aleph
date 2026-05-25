//! Shared per-gate validation helpers used by both state-vector
//! backends (`NaiveSvBackend`, `SoaSvBackend`).
//!
//! Hoisted out of `backend.rs` / `soa_backend.rs` in the P1-01
//! review pass: the function carries NaN-propagation discipline
//! (`is_nan()` reject before any `>` comparison — see ADR 0006)
//! that took three review rounds to harden in P0-09. With two
//! copies a future fix is statistically likely to drift between
//! backends; consolidating here removes the duplication and gives
//! both call sites a single test surface.

use aleph_core::{Complex, GateMatrix};

/// Compute `max_{i,j} |(U·U†)_{i,j} - δ_{i,j}|` for a 2×2, 4×4,
/// or 8×8 matrix. Used to reject non-unitary matrices before they
/// corrupt the state vector.
///
/// Propagates NaN: any NaN entry in `m` produces NaN output rather
/// than 0, so the caller's `!deviation.is_finite()` check rejects
/// NaN-bearing matrices instead of letting them through. Both
/// `f64::max` and `if dev > worst` *swallow* NaN (the former by
/// IEEE-754-2008 minNum/maxNum semantics, the latter because all
/// NaN comparisons return false). The explicit `dev.is_nan()`
/// reject below is therefore load-bearing — see ADR 0006.
pub(crate) fn unitarity_deviation(matrix: &GateMatrix) -> f64 {
    fn max_dev<const N: usize>(m: &[[Complex; N]; N]) -> f64 {
        let mut worst = 0.0_f64;
        for (i, row_i) in m.iter().enumerate() {
            for (j, row_j) in m.iter().enumerate() {
                let mut acc = Complex::new(0.0, 0.0);
                for (a, b) in row_i.iter().zip(row_j.iter()) {
                    acc += a * b.conj();
                }
                let want = if i == j { 1.0 } else { 0.0 };
                let dev = (acc - Complex::new(want, 0.0)).norm();
                if dev.is_nan() {
                    return f64::NAN;
                }
                if dev > worst {
                    worst = dev;
                }
            }
        }
        worst
    }
    match matrix {
        GateMatrix::M2x2(m) => max_dev::<2>(m),
        GateMatrix::M4x4(m) => max_dev::<4>(m),
        GateMatrix::M8x8(m) => max_dev::<8>(m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_2x2() -> GateMatrix {
        let o = Complex::new(1.0, 0.0);
        let z = Complex::new(0.0, 0.0);
        GateMatrix::M2x2([[o, z], [z, o]])
    }

    #[test]
    fn identity_is_unitary() {
        let dev = unitarity_deviation(&identity_2x2());
        assert!(dev < aleph_core::AMPLITUDE_TOL);
    }

    #[test]
    fn scaled_identity_is_not_unitary() {
        let two = Complex::new(2.0, 0.0);
        let z = Complex::new(0.0, 0.0);
        let m = GateMatrix::M2x2([[two, z], [z, two]]);
        let dev = unitarity_deviation(&m);
        assert!(dev > 0.1, "expected non-trivial deviation, got {dev}");
        assert!(dev.is_finite());
    }

    #[test]
    fn nan_entry_propagates_to_output() {
        // ADR 0006 NaN discipline — a NaN entry must make the
        // deviation NaN so the caller's `is_finite` guard rejects.
        let nan = Complex::new(f64::NAN, 0.0);
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let m = GateMatrix::M2x2([[nan, z], [z, o]]);
        let dev = unitarity_deviation(&m);
        assert!(dev.is_nan(), "expected NaN, got {dev}");
    }
}
