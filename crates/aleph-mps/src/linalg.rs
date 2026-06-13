//! Explicit-`Par` faer wrappers and the size-threshold parallelism policy
//! (P3-13). faer's high-level `thin_svd()`/`qr()` hard-read the process-global
//! parallelism, which the `parallel` cargo feature flips to rayon for every
//! caller in the build graph (feature unification) — a 1.5×–19× pessimization
//! at χ ≤ 256 (docs/perf/mps_parallel.md). These wrappers take `Par` per call
//! so each operation chooses from its own operand size instead.
//!
//! The replicas are pinned bit-exact against faer's high-level path by the
//! tests below — re-verify on any faer version bump (Cargo.lock pins 0.24.0).
//! They should be deleted in favor of the upstream API if faer grows per-call
//! `Par` on its high-level solvers.

use crate::MpsError;
use aleph_core::Complex;
use faer::diag::{Diag, DiagMut};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::no_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::{Conj, Mat, MatMut, MatRef, Par};

/// Minimum operand element count (`rows · cols`) for the rayon pool to pay
/// off: strictly above the largest measured-pessimization operand. EPYC 16c
/// calibration (docs/perf/mps_parallel.md): every op in the χ=256 cell
/// (largest 512×512 = 2^18 elements) is a measured rayon pessimization and
/// must stay sequential, while the χ=512 win needs the full family above
/// it — saturated theta/SVD (1024×1024), thin-QR/absorption (1024×512,
/// 512×1024), AND the bond-ramp band in (2^18, 2^19) (e.g. 768×512): the
/// 2^20 threshold retained only 1.23× and 2^19 only 1.48× of P3-09's
/// all-parallel 1.57× @16T.
const PAR_MIN_ELEMS: usize = (1 << 18) + 1;

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
// The one sanctioned production read of faer's global: par_for IS the policy
// the crate-level clippy.toml disallowed-methods fence funnels everything into.
#[allow(clippy::disallowed_methods)]
pub(crate) fn par_for(rows: usize, cols: usize) -> Par {
    if wants_parallel(rows, cols) {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

/// `(U, S, V)` factors of a thin SVD (named to satisfy clippy's
/// type-complexity lint without obscuring the tuple shape).
// P3-14: only `truncated_svd` (now tests-only) consumes this allocating wrapper.
#[allow(dead_code)]
pub(crate) type ThinSvd = (Mat<Complex>, Diag<Complex>, Mat<Complex>);

/// Grow `mem` so a subsequent `MemStack::new(mem)` can satisfy `req`.
/// Effectively monotonic: only rebuilds when the current buffer cannot hold
/// the request, so it never shrinks below a size it already serves.
pub(crate) fn ensure_mem(mem: &mut MemBuffer, req: StackReq) {
    if !MemStack::new(mem).can_hold(req) {
        *mem = MemBuffer::new(req);
    }
}

/// Thin SVD writing factors into caller-provided buffers, using `mem` as scratch
/// (grown as needed). `u_out` must be `m × size`, `v_out` `n × size`, `s_out`
/// `size`, with `size = min(m, n)` — typically `submatrix_mut` views of larger
/// pooled `Mat`s (the arena, P3-14). No allocation in steady state.
pub(crate) fn svd_into(
    a: MatRef<'_, Complex>,
    par: Par,
    u_out: MatMut<'_, Complex>,
    v_out: MatMut<'_, Complex>,
    s_out: DiagMut<'_, Complex>,
    mem: &mut MemBuffer,
) -> Result<(), MpsError> {
    let (m, n) = a.shape();
    let req = svd_scratch::<Complex>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        Default::default(),
    );
    ensure_mem(mem, req);
    svd(
        a,
        s_out,
        Some(u_out),
        Some(v_out),
        par,
        MemStack::new(mem),
        Default::default(),
    )
    .map_err(|_| MpsError::SvdFailed)
}

/// Thin SVD with an explicit `Par`: `A = U · diag(S) · Vᴴ` with
/// `U: m × size`, `V: n × size`, `size = min(m, n)`. Delegates to
/// `svd_into` (allocating wrapper; bit-exact). Mirrors
/// `faer::linalg::solvers::Svd::new_thin` — which hard-reads the global
/// parallelism (faer-0.24.0 solvers.rs:1344) — for the canonical c64 element
/// type (no conjugation pass needed: `aleph_core::Complex == faer::c64`).
#[allow(dead_code)] // P3-14: only the tests-only `truncated_svd` calls this now.
pub(crate) fn thin_svd_par(a: MatRef<'_, Complex>, par: Par) -> Result<ThinSvd, MpsError> {
    let (m, n) = a.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::<Complex>::zeros(m, size);
    let mut v = Mat::<Complex>::zeros(n, size);
    let mut s = Diag::<Complex>::zeros(size);
    let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
    svd_into(a, par, u.as_mut(), v.as_mut(), s.as_mut(), &mut mem)?;
    Ok((u, s, v))
}

/// Thin QR writing `Q` (m × size) into `thin_q` and `R` (size × n) into
/// `thin_r`, factoring in place over `qr_in` (caller copies the source matrix
/// into it first). `q_coeff` is the householder block-T scratch sized
/// `block_size × size` where `block_size = recommended_block_size(m, n)`. `mem`
/// is grown as needed. Mirrors `thin_qr_par` with zero allocation in steady
/// state (P3-14). `size = min(m, n)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qr_into(
    mut qr_in: MatMut<'_, Complex>,
    par: Par,
    mut q_coeff: MatMut<'_, Complex>,
    mut thin_q: MatMut<'_, Complex>,
    mut thin_r: MatMut<'_, Complex>,
    mem: &mut MemBuffer,
) {
    let (m, n) = qr_in.shape();
    let size = Ord::min(m, n);
    let block_size = recommended_block_size::<Complex>(m, n);
    ensure_mem(
        mem,
        qr_in_place_scratch::<Complex>(m, n, block_size, par, Default::default()),
    );
    let _ = qr_in_place(
        qr_in.as_mut(),
        q_coeff.as_mut(),
        par,
        MemStack::new(mem),
        Default::default(),
    );
    // R: upper trapezoid of the factored qr_in. Column-major source → j-outer.
    for j in 0..n {
        for i in 0..Ord::min(j + 1, size) {
            thin_r[(i, j)] = qr_in[(i, j)];
        }
    }
    // Reuse qr_in's first `size` columns as the householder basis (faer
    // split_LU convention: strict-upper zeroed, unit diagonal). No extra alloc.
    for j in 0..size {
        for i in 0..j {
            qr_in[(i, j)] = Complex::new(0.0, 0.0);
        }
        qr_in[(j, j)] = Complex::new(1.0, 0.0);
    }
    // thin_q := identity, then apply the householder sequence.
    thin_q.fill(Complex::new(0.0, 0.0));
    for d in 0..size {
        thin_q[(d, d)] = Complex::new(1.0, 0.0);
    }
    ensure_mem(
        mem,
        apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex>(
            m, block_size, size,
        ),
    );
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        qr_in.as_ref().subcols(0, size),
        q_coeff.as_ref(),
        Conj::No,
        thin_q.as_mut(),
        par,
        MemStack::new(mem),
    );
}

/// Thin QR with an explicit `Par`: returns `(thin_Q, thin_R)` with
/// `thin_Q: m × size` (orthonormal columns), `thin_R: size × n` upper
/// trapezoidal, `size = min(m, n)`. Mirrors `faer::linalg::solvers::Qr::new`
/// plus `compute_thin_Q()`/`thin_R()` — which hard-read the global parallelism
/// (faer-0.24.0 solvers.rs:1115,1196). Takes the input by value: it doubles
/// as the in-place factorization workspace (the high-level path makes the
/// same `to_owned()` copy internally). Delegates to `qr_into` (allocating
/// wrapper; bit-exact).
pub(crate) fn thin_qr_par(qr: Mat<Complex>, par: Par) -> (Mat<Complex>, Mat<Complex>) {
    let (m, n) = qr.shape();
    let size = Ord::min(m, n);
    let block_size = recommended_block_size::<Complex>(m, n);
    let mut qr_in = qr; // consumed as the in-place workspace
    let mut q_coeff = Mat::<Complex>::zeros(block_size, size);
    let mut thin_q = Mat::<Complex>::zeros(m, size);
    let mut thin_r = Mat::<Complex>::zeros(size, n);
    let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
    qr_into(
        qr_in.as_mut(),
        par,
        q_coeff.as_mut(),
        thin_q.as_mut(),
        thin_r.as_mut(),
        &mut mem,
    );
    (thin_q, thin_r)
}

// Tests intentionally read the global / call the high-level solvers: they pin
// the replicas bit-exact against faer's own path under the same Par.
#[allow(clippy::disallowed_methods)]
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
        // (160, 128) / (128, 160) push past the blocked-householder cutoff so
        // the production-sized (large-bond) code paths are compared too.
        for (m, n) in [
            (8usize, 5usize),
            (5, 8),
            (6, 6),
            (1, 2),
            (2, 1),
            (1, 1),
            (160, 128),
            (128, 160),
        ] {
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
        assert!(!wants_parallel(512, 512));
        // Saturating, not panicking.
        assert!(!wants_parallel(0, usize::MAX));
        // Strictly above the χ=256 family — the bond-ramp band parallelizes.
        assert!(wants_parallel(513, 512));
        // χ=512 QR/absorb operands and up may parallelize.
        assert!(wants_parallel(1024, 512));
        assert!(wants_parallel(1024, 1024));
        assert!(wants_parallel(1024, 2048));
        assert!(wants_parallel(usize::MAX, usize::MAX));
    }

    /// Same bit-exactness rationale as the SVD test. Covers m>n, m<n, m=n —
    /// the m<n case exercises the trapezoidal (not triangular) R and the
    /// size×size householder basis.
    #[test]
    fn thin_qr_par_matches_high_level_bit_exact() {
        // (160, 128) / (128, 160) push past the blocked-householder cutoff so
        // the production-sized (large-bond) code paths are compared too.
        for (m, n) in [
            (8usize, 5usize),
            (5, 8),
            (6, 6),
            (1, 2),
            (2, 1),
            (1, 1),
            (160, 128),
            (128, 160),
        ] {
            let a = test_matrix(m, n);
            let hl = a.qr();
            let hq = hl.compute_thin_Q();
            let hr = hl.thin_R();
            let (q, r) = thin_qr_par(a.to_owned(), faer::get_global_parallelism());
            let size = Ord::min(m, n);
            assert_eq!(q.shape(), (m, size));
            assert_eq!(r.shape(), (size, n));
            assert_eq!(q.shape(), hq.shape());
            assert_eq!((r.nrows(), r.ncols()), (hr.nrows(), hr.ncols()));
            for i in 0..m {
                for j in 0..size {
                    assert_eq!(q[(i, j)], hq[(i, j)], "Q[({i},{j})] ({m}x{n})");
                }
            }
            for i in 0..size {
                for j in 0..n {
                    assert_eq!(r[(i, j)], hr[(i, j)], "R[({i},{j})] ({m}x{n})");
                }
            }
        }
    }

    /// Both helpers must produce a valid factorization under either Par —
    /// reconstructions (which are unique, unlike the factors' phases/signs)
    /// must match the input to 1e-12. Run via:
    /// cargo test -p aleph-mps --features parallel
    #[cfg(feature = "parallel")]
    #[test]
    fn helpers_reconstruct_under_seq_and_rayon() {
        let (m, n) = (48usize, 32usize);
        let a = test_matrix(m, n);
        let size = Ord::min(m, n);
        for par in [Par::Seq, Par::rayon(0)] {
            let (u, s, v) = thin_svd_par(a.as_ref(), par).unwrap();
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..size {
                        acc += u[(i, k)] * s.as_ref()[k] * v[(j, k)].conj();
                    }
                    assert!(
                        (acc - a[(i, j)]).norm() < 1e-12,
                        "SVD reconstruction ({i},{j}) under {par:?}"
                    );
                }
            }
            let (q, r) = thin_qr_par(a.to_owned(), par);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..size {
                        acc += q[(i, k)] * r[(k, j)];
                    }
                    assert!(
                        (acc - a[(i, j)]).norm() < 1e-12,
                        "QR reconstruction ({i},{j}) under {par:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn par_for_below_threshold_is_seq() {
        assert_eq!(par_for(512, 512), Par::Seq);
    }

    #[test]
    fn par_for_above_threshold_follows_global() {
        // Only discriminates under `--features parallel`: without it the global
        // is always Par::Seq, so both arms of par_for return Seq and the
        // assertion is vacuously true (still a useful smoke test).
        assert_eq!(par_for(2048, 2048), faer::get_global_parallelism());
    }

    #[test]
    fn svd_into_pooled_matches_high_level_bit_exact() {
        for (m, n) in [(8usize, 5usize), (5, 8), (6, 6), (160, 128)] {
            let size = Ord::min(m, n);
            let a = test_matrix(m, n);
            let hl = a.thin_svd().unwrap();
            // Oversized pooled backing + strided sub-views.
            let mut u = Mat::<Complex>::zeros(m + 3, size + 3);
            let mut v = Mat::<Complex>::zeros(n + 3, size + 3);
            let mut s = Diag::<Complex>::zeros(size);
            let mut mem = MemBuffer::new(StackReq::new::<Complex>(0));
            svd_into(
                a.as_ref(),
                faer::get_global_parallelism(),
                u.as_mut().submatrix_mut(0, 0, m, size),
                v.as_mut().submatrix_mut(0, 0, n, size),
                s.as_mut(),
                &mut mem,
            )
            .unwrap();
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
}
