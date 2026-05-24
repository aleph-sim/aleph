//! Stack-allocated return type for `Gate::matrix()`.
//! See `docs/decisions/0003-gate-matrix-representation.md`.

use crate::Complex;

/// Unitary matrix for a quantum gate, sized by arity.
///
/// `M2x2` for 1-qubit gates, `M4x4` for 2-qubit, `M8x8` for 3-qubit.
/// Backends pattern-match on the variant to dispatch to a kernel.
///
/// **No `PartialEq` derive.** Comparing nested `Complex<f64>` arrays by
/// bitwise equality is the float-equality footgun CLAUDE.md § Common
/// Mistakes flags. Tests compare matrices with `approx_eq_*` helpers
/// at `AMPLITUDE_TOL` tolerance instead.
//
// `large_enum_variant` is intentional: the 1024 B M8x8 variant is the
// hot-path return type for 3-qubit gates, and the whole point of the
// enum (see docs/decisions/0003-gate-matrix-representation.md) is to
// stay heap-free. Boxing M8x8 would defeat that design.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum GateMatrix {
    M2x2([[Complex; 2]; 2]),
    M4x4([[Complex; 4]; 4]),
    M8x8([[Complex; 8]; 8]),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_sizes_are_stack_friendly() {
        // 2x2 = 4 * 16B = 64B, 4x4 = 16 * 16B = 256B, 8x8 = 64 * 16B = 1024B.
        // The enum itself reserves the largest variant; check it stays
        // bounded so the design assumption in the spec holds.
        assert!(core::mem::size_of::<GateMatrix>() <= 1024 + 16);
    }

    #[test]
    fn variant_pattern_match() {
        // Smoke-check that backends can pattern-match on the variant.
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        let m = GateMatrix::M2x2([[one, zero], [zero, one]]);
        match m {
            GateMatrix::M2x2(_) => {}
            other => panic!("expected M2x2, got {other:?}"),
        }
    }
}
