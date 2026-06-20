//! Host-side truncated SVD split for one NN 2q gate (P5.5-06). The GPU produces
//! the gated two-site block Θ′ (rows = `li·2`, cols = `2·ri`); this factorizes it
//! on the CPU via `faer` into the two new adjacent site tensors. `aleph_core::Complex`
//! is `faer::c64`, so the f64 SVD reads the widened block with no type juggling.
//!
//! This is the documented CPU round-trip (AC #2): Θ′ lives in unified memory, so
//! the host read is zero-copy, but the SVD runs single-threaded on the CPU while
//! the GPU is idle, then the two factor tensors are uploaded into fresh buffers.

use aleph_backend::BackendError;
use aleph_core::Complex;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::{Mat, Par};

/// Minimum `min(rows, cols)` of the two-site block Θ′ at which the SVD switches
/// from `Par::Seq` to rayon-parallel (P5.6-07). Below it the factorization is too
/// small for fork/join to pay off; at/above it (bond χ ≳ 16, so rows/cols = 2·χ ≳
/// 32) the parallel SVD overlaps the cost that dominated per-gate MPS time.
const PAR_SVD_MIN_DIM: usize = 32;

/// Narrow an f64 SVD factor entry back to the stored f32 site precision.
#[inline]
fn narrow(z: Complex) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

/// Pure χ-selection for a fixed bond cap. Returns `(chi, discarded)`: `chi`
/// singular values kept, `discarded` = Σ_{j≥chi} σ_j² (dropped Schmidt weight).
/// Null directions below `1e-7·σ_max` are pruned before the cap so the bond is
/// not inflated with Gram noise.
///
/// Unlike the CPU MPS, the scaffold does **not** renormalize the kept σ. The CPU
/// MPS moves the orthogonality centre onto the active block before factorizing,
/// so Θ′ has unit Frobenius norm and `scale = 1/√(kept weight)` corrects only the
/// truncation loss. This naive (non-canonical) MPS keeps each block's own norm,
/// so the kept singular values must be the *exact* σ (`s_kept[t] = σ_t`) — any
/// global rescale would corrupt a block whose norm is not 1.
///
/// The consequence: this scaffold is correct **only when nothing is truncated**.
/// When the bond cap forces dropping a real singular value there is no orthogonality
/// centre to absorb `scale`, so the amplitudes would silently drift. The backend
/// therefore treats a non-negligible `discarded` as an error (P5.6-02): the caller
/// converts the relative weight `svd_split` reports into a refusal. `discarded`
/// here is the absolute dropped Schmidt weight Σ_{j≥chi} σ_j²; `svd_split`
/// normalizes it by the total Frobenius weight before returning it.
fn truncation_plan(sigmas: &[f64], max_bond: usize) -> (usize, f64) {
    let s_max = sigmas.first().copied().unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = sigmas.iter().filter(|&&s| s > eps).count().max(1);
    let chi = significant.min(max_bond.max(1));
    let discarded: f64 = sigmas[chi..].iter().map(|s| s * s).sum();
    (chi, discarded)
}

/// Factor result: `(chi, site_i_data, site_j_data, trunc_rel)`.
/// - `site_i_data` is row-major `(li, 2, chi)`: `U[:, :chi]` (left isometry).
/// - `site_j_data` is row-major `(chi, 2, ri)`: `diag(σ_kept)·Vᴴ` (S absorbed right).
/// - `trunc_rel` is the **relative** truncation weight `Σ_{j≥chi} σ_j² / Σ_j σ_j²`
///   — the fraction of squared Schmidt weight discarded by the bond cap. `0` (to
///   null-direction rounding) means the split is exact; a non-negligible value
///   means this naive scaffold would corrupt the state (see `truncation_plan`).
pub(crate) type SplitResult = (usize, Vec<Complex<f32>>, Vec<Complex<f32>>, f64);

/// Truncated SVD of Θ′ (row-major, `rows × cols`, f32) → the two new site tensors.
/// `max_bond` caps the kept bond dimension χ.
pub(crate) fn svd_split(
    theta: &[Complex<f32>],
    rows: usize,
    cols: usize,
    max_bond: usize,
) -> Result<SplitResult, BackendError> {
    debug_assert_eq!(theta.len(), rows * cols);
    // Θ′ widened f32 → f64 (exact) into a faer matrix.
    let a = Mat::<Complex>::from_fn(rows, cols, |r, c| {
        let z = theta[r * cols + c];
        Complex::new(z.re as f64, z.im as f64)
    });
    let size = rows.min(cols);
    // Parallelise the factorization only once the block is big enough to pay for
    // rayon's fork/join (P5.6-07). Small early-circuit blocks stay `Par::Seq`;
    // deep-entanglement blocks (large bond ⇒ rows/cols up to 2·χ) fan out, where
    // the host SVD used to be 94.5% of per-gate time. `Par::rayon(0)` uses
    // rayon's default thread count.
    let par = if size >= PAR_SVD_MIN_DIM {
        Par::rayon(0)
    } else {
        Par::Seq
    };
    let mut u = Mat::<Complex>::zeros(rows, size);
    let mut v = Mat::<Complex>::zeros(cols, size);
    let mut s = faer::diag::Diag::<Complex>::zeros(size);
    let req = svd_scratch::<Complex>(
        rows,
        cols,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        Default::default(),
    );
    let mut mem = MemBuffer::new(req);
    svd(
        a.as_ref(),
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|_| BackendError::InvalidState {
        reason: "MPS two-site SVD failed to converge",
    })?;

    let sigmas: Vec<f64> = (0..size).map(|t| s.as_ref()[t].re).collect();
    let (chi, discarded) = truncation_plan(&sigmas, max_bond);
    // Normalize the dropped weight by the block's own Frobenius weight so the
    // backend can compare it against a precision tolerance regardless of the
    // (non-unit) block norm. `total == 0` only for an all-zero block ⇒ no loss.
    let total: f64 = sigmas.iter().map(|s| s * s).sum();
    let trunc_rel = if total > 0.0 { discarded / total } else { 0.0 };
    // Exact σ (no renormalization) — see `truncation_plan` rationale.
    let s_kept: &[f64] = &sigmas[..chi];

    let u_ref = u.as_ref();
    let v_ref = v.as_ref();
    // Site i ← U[:, :chi]: row-major (li,2,chi) = rows×chi, data[row*chi + t].
    let mut site_i = vec![Complex::<f32>::new(0.0, 0.0); rows * chi];
    for row in 0..rows {
        for t in 0..chi {
            site_i[row * chi + t] = narrow(u_ref[(row, t)]);
        }
    }
    // Site j ← diag(σ_kept)·Vᴴ: row-major (chi,2,ri) = chi×cols, data[t*cols + col]
    // = σ_kept[t]·conj(V[col, t]).
    let mut site_j = vec![Complex::<f32>::new(0.0, 0.0); chi * cols];
    for t in 0..chi {
        for col in 0..cols {
            let vh = v_ref[(col, t)].conj() * s_kept[t];
            site_j[t * cols + col] = narrow(vh);
        }
    }
    Ok((chi, site_i, site_j, trunc_rel))
}

#[cfg(test)]
mod tests {
    use super::*;

    // truncation_plan keeps min(significant, max_bond) σ and reports the dropped
    // squared weight. With max_bond above the significant count nothing real is
    // dropped; below it, the smallest σ² are discarded.
    #[test]
    fn truncation_plan_drops_below_cap() {
        let sigmas = [1.0, 0.5, 1e-9]; // third is below the 1e-7·σ_max null floor
        let (chi, discarded) = truncation_plan(&sigmas, 64);
        assert_eq!(chi, 2, "null direction pruned, two significant kept");
        assert!(
            discarded <= 1e-12,
            "no real weight dropped, got {discarded}"
        );

        let (chi, discarded) = truncation_plan(&sigmas, 1);
        assert_eq!(chi, 1, "cap binds to 1");
        assert!(
            (discarded - 0.25).abs() < 1e-12,
            "drops 0.5² = 0.25, got {discarded}"
        );
    }

    // svd_split on a maximally-entangled 2-qubit block (Bell Θ, rows=cols=2):
    // singular values [1/√2, 1/√2]. Cap 2 ⇒ exact (trunc_rel ≈ 0); cap 1 ⇒ drops
    // half the squared weight ⇒ trunc_rel ≈ 0.5. Pure CPU (faer), no device.
    #[test]
    fn svd_split_reports_relative_truncation() {
        let inv_sqrt2 = (0.5f32).sqrt();
        // Bell Θ in the (li=1,2)×(2,ri=1) = 2×2 layout: diag(1/√2, 1/√2).
        let theta = vec![
            Complex::<f32>::new(inv_sqrt2, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(0.0, 0.0),
            Complex::<f32>::new(inv_sqrt2, 0.0),
        ];
        let (_, _, _, trunc) = svd_split(&theta, 2, 2, 2).expect("svd cap=2");
        assert!(trunc < 1e-6, "no truncation at cap=2, got {trunc}");

        let (chi, _, _, trunc) = svd_split(&theta, 2, 2, 1).expect("svd cap=1");
        assert_eq!(chi, 1);
        assert!(
            (trunc - 0.5).abs() < 1e-5,
            "half the weight dropped, got {trunc}"
        );
    }
}
