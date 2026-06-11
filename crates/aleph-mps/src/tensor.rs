//! Rank-3 MPS site tensor `(left, 2, right)` and its faer matrix views.

use crate::MpsError;
use aleph_core::Complex;

/// A single MPS site tensor of shape `(left, 2, right)`.
/// Row-major flat storage: `data[(l*2 + p)*right + r]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub left: usize,
    pub right: usize,
    pub data: Vec<Complex>,
}

impl Site {
    /// Returns a `(left, 2, right)` tensor filled with zeros.
    pub fn zeros(left: usize, right: usize) -> Self {
        Site {
            left,
            right,
            data: vec![Complex::new(0.0, 0.0); left * 2 * right],
        }
    }

    /// Returns the `(1, 2, 1)` tensor `[1, 0]` — a qubit in state |0⟩.
    pub fn ket0() -> Self {
        let mut s = Site::zeros(1, 1);
        s.data[0] = Complex::new(1.0, 0.0);
        s
    }

    /// Flat row-major index for `(l, p, r)`.
    #[inline]
    pub fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * 2 + p) * self.right + r
    }

    /// Read element `(l, p, r)`.
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> Complex {
        self.data[self.idx(l, p, r)]
    }

    /// Mutable reference to element `(l, p, r)`.
    #[inline]
    pub fn get_mut(&mut self, l: usize, p: usize, r: usize) -> &mut Complex {
        let i = self.idx(l, p, r);
        &mut self.data[i]
    }

    /// Zero-copy faer view of the grouped-left matrix `(left·2) × right`
    /// (row `l·2+p`, col `r`) — identical to the row-major layout of `data`.
    pub fn group_left_view(&self) -> faer::MatRef<'_, Complex> {
        faer::MatRef::from_row_major_slice(&self.data, self.left * 2, self.right)
    }

    /// Zero-copy faer view of the grouped-right matrix `left × (2·right)`
    /// (row `l`, col `p·right + r`) — the same bytes, regrouped.
    pub fn group_right_view(&self) -> faer::MatRef<'_, Complex> {
        faer::MatRef::from_row_major_slice(&self.data, self.left, 2 * self.right)
    }

    /// Build a `Site` from a faer `(left·2) × right` grouped-left matrix.
    pub fn from_group_left_faer(m: faer::MatRef<'_, Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (row → (l, p)) split.
        #[allow(clippy::needless_range_loop)]
        for row in 0..left * 2 {
            for r in 0..right {
                s.data[row * right + r] = m[(row, r)];
            }
        }
        s
    }

    /// Build a `Site` from a faer `χ × (2·right)` grouped-right matrix.
    pub fn from_group_right_faer(m: faer::MatRef<'_, Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (col → (p, r)) split.
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            for col in 0..2 * right {
                s.data[l * 2 * right + col] = m[(l, col)];
            }
        }
        s
    }
}

/// How `truncated_svd` chooses how many singular values to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationPolicy {
    /// Keep at most `χ` singular values (the largest).
    FixedBond(usize),
    /// Keep the fewest singular values whose discarded squared weight is `≤ ε`,
    /// never exceeding `max_bond`.
    ErrorBounded { epsilon: f64, max_bond: usize },
}

/// `(u_kept, s_kept, vt_kept, discarded_weight)` returned by [`truncated_svd`].
pub type TruncatedSvd = (faer::Mat<Complex>, Vec<f64>, faer::Mat<Complex>, f64);

/// SVD of `m` truncated according to `policy` (fixed-χ or error-bounded),
/// renormalized to preserve unit weight (input must come from a normalized
/// state). Returns `(u_kept, s_kept, vt_kept, discarded_weight)` where:
/// - `u_kept` has shape `rows × χ` with orthonormal columns (left isometry)
/// - `s_kept` is the χ kept (renormalized) singular values, descending
/// - `vt_kept` has shape `χ × cols` with orthonormal rows
/// - `discarded_weight` is the sum of squares of discarded singular values
///
/// # Why `faer`
///
/// `nalgebra`'s complex `svd()` AND its Hermitian `SymmetricEigen` both return
/// orthonormal but *incorrect* vectors for certain complex matrices (degenerate
/// two-site MPS blocks), so neither reconstructs `M` — a single `CNOT` could
/// silently drop half the state norm (root-caused via the SWAP-network oracle
/// proptest). We use `faer`'s `thin_svd`, which is reliable for complex inputs
/// (verified to reconstruct the offending blocks to ~1e-16).
pub fn truncated_svd(
    m: faer::MatRef<'_, Complex>,
    policy: &TruncationPolicy,
) -> Result<TruncatedSvd, MpsError> {
    let rows = m.nrows();
    let cols = m.ncols();

    // Reliable complex SVD via faer (singular values nonnegative, nonincreasing).
    let svd = m.thin_svd().map_err(|_| MpsError::SvdFailed)?;
    let fu = svd.U();
    let fv = svd.V();
    let fs = svd.S();
    let k = fs.column_vector().nrows(); // = min(rows, cols)
    let sigmas: Vec<f64> = (0..k).map(|t| fs[t].re).collect();

    // Drop singular values numerically zero relative to the largest (null
    // directions). faer's spectrum is accurate, so this only prunes genuine
    // zeros and avoids inflating the bond with noise.
    let s_max = sigmas.first().copied().unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = sigmas.iter().filter(|&&s| s > eps).count().max(1);

    // Suffix sums of σ²: suffix_sq[t] = Σ_{j≥t} σ_j² (non-increasing in t).
    let mut suffix_sq = vec![0.0_f64; k + 1];
    for t in (0..k).rev() {
        suffix_sq[t] = suffix_sq[t + 1] + sigmas[t] * sigmas[t];
    }
    let chi = match *policy {
        TruncationPolicy::FixedBond(max_bond) => significant.min(max_bond.max(1)),
        TruncationPolicy::ErrorBounded { epsilon, max_bond } => {
            let cap = significant.min(max_bond.max(1));
            let mut chosen = cap;
            // Smallest keep ∈ [1, cap] with discarded tail Σ_{j≥keep} σ_j² ≤ ε.
            #[allow(clippy::needless_range_loop)]
            for keep in 1..=cap {
                if suffix_sq[keep] <= epsilon {
                    chosen = keep;
                    break;
                }
            }
            chosen
        }
    };
    let discarded: f64 = suffix_sq[chi];
    let kept_weight: f64 = suffix_sq[0] - suffix_sq[chi];
    let scale = if kept_weight > 0.0 {
        (1.0 / kept_weight).sqrt()
    } else {
        1.0
    };

    let u_kept = faer::Mat::from_fn(rows, chi, |r, t| fu[(r, t)]);
    // vt row t = (t-th right singular vector)ᴴ = conjugate of V's column t.
    let vt_kept = faer::Mat::from_fn(chi, cols, |t, c| fv[(c, t)].conj());
    let s_kept: Vec<f64> = (0..chi).map(|t| sigmas[t] * scale).collect();
    Ok((u_kept, s_kept, vt_kept, discarded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Complex;

    fn c(re: f64) -> Complex {
        Complex::new(re, 0.0)
    }

    #[test]
    fn ket0_site_shape() {
        let s = Site::ket0();
        assert_eq!((s.left, s.right), (1, 1));
        assert_eq!(s.get(0, 0, 0), c(1.0));
        assert_eq!(s.get(0, 1, 0), c(0.0));
    }

    #[test]
    fn truncated_svd_reconstructs_complex_full_rank() {
        // Generic complex 4×4: U·diag(s)·Vt must reconstruct M (renorm scale=1
        // only if ‖M‖=1; here ‖M‖≠1, so compare to scale·M via re-deriving).
        let m = faer::Mat::from_fn(4, 4, |i, j| {
            Complex::new(
                (i as f64 - j as f64) * 0.3 + 1.0,
                (i * 2 + j) as f64 * 0.17 - 0.5,
            )
        });
        let fro: f64 = m.as_ref().norm_l2();
        let (u, s, vt, _disc) =
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64)).unwrap();
        // reconstruction = U·diag(s)·Vt = (1/fro)·M  (renormalized to unit weight)
        let mut maxd = 0.0_f64;
        for r in 0..4 {
            for col in 0..4 {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..s.len() {
                    acc += u[(r, k)] * Complex::new(s[k], 0.0) * vt[(k, col)];
                }
                maxd = maxd.max((acc - m[(r, col)] / Complex::new(fro, 0.0)).norm());
            }
        }
        assert!(maxd < 1e-10, "complex SVD reconstruction err {maxd:e}");
    }

    #[test]
    fn truncated_svd_rank1_complex_collapses_to_chi1() {
        // Rank-1 complex matrix (outer product a·bᴴ) must yield χ=1, not an
        // inflated bond padded with Gram-noise singular values.
        let a = [
            Complex::new(0.5, 0.3),
            Complex::new(-0.2, 0.7),
            Complex::new(0.1, -0.4),
            Complex::new(0.6, 0.0),
        ];
        let b = [
            Complex::new(0.4, -0.1),
            Complex::new(0.2, 0.5),
            Complex::new(-0.3, 0.2),
            Complex::new(0.1, 0.1),
        ];
        let m = faer::Mat::from_fn(4, 4, |i, j| a[i] * b[j].conj());
        let (u, s, vt, _disc) =
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64)).unwrap();
        assert_eq!(
            s.len(),
            1,
            "rank-1 block must collapse to χ=1, got χ={}",
            s.len()
        );
        // And it must still reconstruct (1/‖M‖)·M.
        let fro: f64 = m.as_ref().norm_l2();
        let mut maxd = 0.0_f64;
        for r in 0..4 {
            for col in 0..4 {
                let acc = u[(r, 0)] * Complex::new(s[0], 0.0) * vt[(0, col)];
                maxd = maxd.max((acc - m[(r, col)] / Complex::new(fro, 0.0)).norm());
            }
        }
        assert!(maxd < 1e-10, "rank-1 reconstruction err {maxd:e}");
    }

    fn diag_sigma() -> faer::Mat<Complex> {
        let s = [1.0, 0.1, 0.01, 0.001];
        faer::Mat::from_fn(4, 4, |i, j| {
            if i == j {
                Complex::new(s[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
    }

    #[test]
    fn error_bounded_keeps_minimal_chi() {
        let m = diag_sigma();
        let (_, s, _, disc) = truncated_svd(
            m.as_ref(),
            &TruncationPolicy::ErrorBounded {
                epsilon: 1e-3,
                max_bond: 64,
            },
        )
        .unwrap();
        assert_eq!(s.len(), 2, "expected χ=2");
        assert!(disc <= 1e-3 + 1e-15, "discarded {disc} exceeds ε");
    }

    #[test]
    fn error_bounded_tiny_eps_keeps_all() {
        let m = diag_sigma();
        let (_, s, _, disc) = truncated_svd(
            m.as_ref(),
            &TruncationPolicy::ErrorBounded {
                epsilon: 0.0,
                max_bond: 64,
            },
        )
        .unwrap();
        assert_eq!(s.len(), 4, "ε=0 must keep full rank");
        assert!(disc < 1e-12);
    }

    #[test]
    fn error_bounded_cap_overrides_eps() {
        let m = diag_sigma();
        let (_, s, _, _) = truncated_svd(
            m.as_ref(),
            &TruncationPolicy::ErrorBounded {
                epsilon: 10.0,
                max_bond: 1,
            },
        )
        .unwrap();
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fixed_bond_matches_legacy() {
        let m = diag_sigma();
        let (_, s, _, _) = truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(2)).unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn faer_views_match_element_access() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    *s.get_mut(l, p, r) = Complex::new((l * 100 + p * 10 + r) as f64, 0.5);
                }
            }
        }
        let gl = s.group_left_view(); // (left*2) x right
        assert_eq!((gl.nrows(), gl.ncols()), (4, 3));
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    assert_eq!(gl[(l * 2 + p, r)], s.get(l, p, r));
                }
            }
        }
        let gr = s.group_right_view(); // left x (2*right)
        assert_eq!((gr.nrows(), gr.ncols()), (2, 6));
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    assert_eq!(gr[(l, p * 3 + r)], s.get(l, p, r));
                }
            }
        }
    }

    #[test]
    fn faer_from_group_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for (k, v) in s.data.iter_mut().enumerate() {
            *v = Complex::new(k as f64, -(k as f64));
        }
        let back_l = Site::from_group_left_faer(s.group_left_view(), 2, 3);
        assert_eq!(back_l, s);
        let back_r = Site::from_group_right_faer(s.group_right_view(), 2, 3);
        assert_eq!(back_r, s);
    }
}
