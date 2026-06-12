//! Explicit-`Par` faer wrappers and the size-threshold parallelism policy
//! (P3-13). faer's high-level `thin_svd()`/`qr()` hard-read the process-global
//! parallelism, which the `parallel` cargo feature flips to rayon for every
//! caller in the build graph (feature unification) — a 1.5×–19× pessimization
//! at χ ≤ 256 (docs/perf/mps_parallel.md). These wrappers take `Par` per call
//! so each operation chooses from its own operand size instead.

use crate::MpsError;
use aleph_core::Complex;
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::{Mat, MatRef, Par};

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

/// `(U, S, V)` factors of a thin SVD (named to satisfy clippy's
/// type-complexity lint without obscuring the tuple shape).
pub(crate) type ThinSvd = (Mat<Complex>, Diag<Complex>, Mat<Complex>);

/// Thin SVD with an explicit `Par`: `A = U · diag(S) · Vᴴ` with
/// `U: m × size`, `V: n × size`, `size = min(m, n)`. Mirrors
/// `faer::linalg::solvers::Svd::new_thin` — which hard-reads the global
/// parallelism (faer-0.24.0 solvers.rs:1344) — for the canonical c64 element
/// type (no conjugation pass needed: `aleph_core::Complex == faer::c64`).
// Call sites are rewired in a later P3-13 task; until then only tests use it.
#[allow(dead_code)]
pub(crate) fn thin_svd_par(a: MatRef<'_, Complex>, par: Par) -> Result<ThinSvd, MpsError> {
    let (m, n) = a.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::<Complex>::zeros(m, size);
    let mut v = Mat::<Complex>::zeros(n, size);
    let mut s = Diag::<Complex>::zeros(size);
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut MemBuffer::new(svd_scratch::<Complex>(
            m,
            n,
            ComputeSvdVectors::Thin,
            ComputeSvdVectors::Thin,
            par,
            Default::default(),
        ))),
        Default::default(),
    )
    .map_err(|_| MpsError::SvdFailed)?;
    Ok((u, s, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic full-rank-ish complex test matrix.
    fn test_matrix(m: usize, n: usize) -> Mat<Complex> {
        Mat::from_fn(m, n, |i, j| {
            Complex::new(
                ((i * 7 + j * 3) % 11) as f64 * 0.37 - 1.1,
                ((i * 5 + j) % 7) as f64 * 0.23 - 0.6,
            )
        })
    }

    /// The replica must be BIT-EXACT vs faer's high-level thin_svd when given
    /// the same Par the high-level call reads from the global — any divergence
    /// means the low-level invocation differs (wrong compute mode, params, or
    /// conj handling), which a tolerance test could mask. (The crate's "no
    /// float equality" rule targets tolerance-requiring math; this verifies
    /// replication of an identical deterministic code path.)
    #[test]
    fn thin_svd_par_matches_high_level_bit_exact() {
        for (m, n) in [(8usize, 5usize), (5, 8), (6, 6)] {
            let a = test_matrix(m, n);
            let hl = a.thin_svd().unwrap();
            let (u, s, v) = thin_svd_par(a.as_ref(), faer::get_global_parallelism()).unwrap();
            let size = Ord::min(m, n);
            assert_eq!(u.shape(), (m, size));
            assert_eq!(v.shape(), (n, size));
            for t in 0..size {
                assert_eq!(s.as_ref()[t], hl.S()[t], "S[{t}] ({m}x{n})");
            }
            for r in 0..m {
                for c in 0..size {
                    assert_eq!(u[(r, c)], hl.U()[(r, c)], "U[({r},{c})] ({m}x{n})");
                }
            }
            for r in 0..n {
                for c in 0..size {
                    assert_eq!(v[(r, c)], hl.V()[(r, c)], "V[({r},{c})] ({m}x{n})");
                }
            }
        }
    }

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
