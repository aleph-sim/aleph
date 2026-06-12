//! Explicit-`Par` faer wrappers and the size-threshold parallelism policy
//! (P3-13). faer's high-level `thin_svd()`/`qr()` hard-read the process-global
//! parallelism, which the `parallel` cargo feature flips to rayon for every
//! caller in the build graph (feature unification) — a 1.5×–19× pessimization
//! at χ ≤ 256 (docs/perf/mps_parallel.md). These wrappers take `Par` per call
//! so each operation chooses from its own operand size instead.

use faer::Par;

/// Minimum operand element count (`rows · cols`) for the rayon pool to pay
/// off. Calibrated from the P3-09 EPYC sweep (docs/perf/mps_parallel.md):
/// χ=256 ops (up to 512×1024 = 524 288 elements) are a rayon pessimization,
/// while χ=512 ops (1024×2048 = 2 097 152) win 1.57× @16T. 2^20 is the
/// geometric midpoint of that interval; the P3-13 EPYC sweep validates both
/// sides.
const PAR_MIN_ELEMS: usize = 1 << 20;

/// Whether a `rows × cols` operand is large enough to amortize fork-join
/// overhead (feature-independent threshold arithmetic, unit-tested directly).
#[inline]
pub(crate) fn wants_parallel(rows: usize, cols: usize) -> bool {
    rows.saturating_mul(cols) >= PAR_MIN_ELEMS
}

/// `Par` for one operation on a `rows × cols` operand: the global setting
/// (rayon when the `parallel` feature is compiled in) above the threshold,
/// `Par::Seq` below it. Without the feature the global is always `Par::Seq`,
/// so this degrades to a no-op.
// Call sites are rewired in a later P3-13 task; until then only tests use it.
#[allow(dead_code)]
pub(crate) fn par_for(rows: usize, cols: usize) -> Par {
    if wants_parallel(rows, cols) {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_boundaries() {
        // Largest χ=256 operand (the measured pessimization) stays sequential.
        assert!(!wants_parallel(512, 1024));
        // Saturating, not panicking.
        assert!(!wants_parallel(0, usize::MAX));
        // 1024×1024 (= PAR_MIN_ELEMS exactly) and up may parallelize.
        assert!(wants_parallel(1024, 1024));
        assert!(wants_parallel(1024, 2048));
        assert!(wants_parallel(usize::MAX, usize::MAX));
    }

    #[test]
    fn par_for_below_threshold_is_seq() {
        assert_eq!(par_for(512, 512), Par::Seq);
    }

    #[test]
    fn par_for_above_threshold_follows_global() {
        assert_eq!(par_for(2048, 2048), faer::get_global_parallelism());
    }
}
