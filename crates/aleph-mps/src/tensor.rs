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

    /// Overwrite this site in place as a left-canonical tensor of shape
    /// `(left, 2, right)` whose grouped-left matrix `(left·2) × right` is `m`.
    /// Reuses the existing `data` allocation (resized), avoiding a fresh `Site`.
    /// `m` must be at least `(left·2) × right`; only that top-left block is read.
    // NOTE: `Site` is currently an internal type (not re-exported from lib.rs).
    // These three fillers will be called by the scratch-arena hot path in a later
    // P3-14 task; the allow below is intentional until that wiring lands.
    #[allow(dead_code)]
    pub fn fill_left_from(&mut self, m: faer::MatRef<'_, Complex>, left: usize, right: usize) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for row in 0..left * 2 {
            for r in 0..right {
                self.data[row * right + r] = m[(row, r)];
            }
        }
    }

    /// Overwrite this site in place from a `left × (2·right)` grouped-right
    /// matrix `m` (row `l`, col `p·right + r`) — the in-place equivalent of
    /// `from_group_right_faer`. Reuses the existing `data` allocation.
    #[allow(dead_code)]
    pub fn fill_from_grouped_right(
        &mut self,
        m: faer::MatRef<'_, Complex>,
        left: usize,
        right: usize,
    ) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            for col in 0..2 * right {
                self.data[l * 2 * right + col] = m[(l, col)];
            }
        }
    }

    /// Overwrite this site in place as a right-canonical tensor of shape
    /// `(left, 2, right)` whose grouped-right matrix `left × (2·right)` is the
    /// scaled conjugate `conj(v[(col, l)]) · sv[l]` — i.e. the singular-value
    /// folding `s·Vᴴ` for the V factor (or the bare conjugate when `sv` is all
    /// ones, e.g. a right-canonical Qᴴ). `v` is read as `cols × left` (its row
    /// = grouped-right column index `col`, its col = bond index `l`). `sv` has
    /// length `left`. Reuses the existing `data` allocation.
    #[allow(dead_code)]
    pub fn fill_right_from_scaled_conj(
        &mut self,
        v: faer::MatRef<'_, Complex>,
        sv: &[f64],
        left: usize,
        right: usize,
    ) {
        self.left = left;
        self.right = right;
        self.data.clear();
        self.data.resize(left * 2 * right, Complex::new(0.0, 0.0));
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            let s = Complex::new(sv[l], 0.0);
            for col in 0..2 * right {
                // grouped-right entry (l, col) = conj(V[col, l]) · s
                self.data[l * 2 * right + col] = v[(col, l)].conj() * s;
            }
        }
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

/// Pure χ-selection + renormalization for a truncated SVD given the (descending,
/// nonnegative) singular values `sigmas`. Returns `(chi, discarded, scale)`:
/// - `chi` ∈ [1, len] singular values to keep,
/// - `discarded` = Σ_{j≥chi} σ_j² (the dropped Schmidt weight),
/// - `scale` = renormalization factor for the kept σ so the state stays unit
///   weight (input must come from a normalized state).
///
/// Null directions numerically zero relative to σ_max (1e-7·σ_max) are pruned
/// before applying the policy, so the bond is never inflated with Gram noise.
pub(crate) fn svd_truncation_plan(sigmas: &[f64], policy: &TruncationPolicy) -> (usize, f64, f64) {
    let k = sigmas.len();
    let s_max = sigmas.first().copied().unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = sigmas.iter().filter(|&&s| s > eps).count().max(1);

    // Suffix sums of σ²: suffix_sq[t] = Σ_{j≥t} σ_j².
    let mut suffix_sq = vec![0.0_f64; k + 1];
    for t in (0..k).rev() {
        suffix_sq[t] = suffix_sq[t + 1] + sigmas[t] * sigmas[t];
    }
    let chi = match *policy {
        TruncationPolicy::FixedBond(max_bond) => significant.min(max_bond.max(1)),
        TruncationPolicy::ErrorBounded { epsilon, max_bond } => {
            let cap = significant.min(max_bond.max(1));
            let mut chosen = cap;
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
    let discarded = suffix_sq[chi];
    let kept_weight = suffix_sq[0] - suffix_sq[chi];
    let scale = if kept_weight > 0.0 {
        (1.0 / kept_weight).sqrt()
    } else {
        1.0
    };
    (chi, discarded, scale)
}

/// SVD of `m` truncated according to `policy` (fixed-χ or error-bounded),
/// renormalized to preserve unit weight (input must come from a normalized
/// state). Returns `(u_kept, s_kept, vt_kept, discarded_weight)` where:
/// - `u_kept` has shape `rows × χ` with orthonormal columns (left isometry)
/// - `s_kept` is the χ kept (renormalized) singular values, descending
/// - `vt_kept` has shape `χ × cols` with orthonormal rows
/// - `discarded_weight` is the sum of squares of discarded singular values
///
/// The SVD runs under the caller-chosen `par` (P3-13 size-thresholded
/// parallelism).
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
    par: faer::Par,
) -> Result<TruncatedSvd, MpsError> {
    let rows = m.nrows();
    let cols = m.ncols();

    // Reliable complex SVD via faer (singular values nonnegative, nonincreasing).
    let (fu, fs, fv) = crate::linalg::thin_svd_par(m, par)?;
    let fs = fs.as_ref();
    let k = fs.column_vector().nrows(); // = min(rows, cols)
    let sigmas: Vec<f64> = (0..k).map(|t| fs[t].re).collect();

    let (chi, discarded, scale) = svd_truncation_plan(&sigmas, policy);

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
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64), faer::Par::Seq).unwrap();
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
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64), faer::Par::Seq).unwrap();
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
            faer::Par::Seq,
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
            faer::Par::Seq,
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
            faer::Par::Seq,
        )
        .unwrap();
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fixed_bond_matches_legacy() {
        let m = diag_sigma();
        let (_, s, _, _) =
            truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(2), faer::Par::Seq).unwrap();
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

    #[test]
    fn plan_fixed_bond_caps_chi() {
        let s = vec![1.0, 0.1, 0.01, 0.001];
        let (chi, _, _) = svd_truncation_plan(&s, &TruncationPolicy::FixedBond(2));
        assert_eq!(chi, 2);
    }

    #[test]
    fn plan_error_bounded_keeps_minimal_chi() {
        let s = vec![1.0, 0.1, 0.01, 0.001];
        let (chi, disc, _) = svd_truncation_plan(
            &s,
            &TruncationPolicy::ErrorBounded {
                epsilon: 1e-3,
                max_bond: 64,
            },
        );
        assert_eq!(chi, 2);
        assert!(disc <= 1e-3 + 1e-15);
    }

    #[test]
    fn plan_tiny_eps_keeps_all() {
        let s = vec![1.0, 0.1, 0.01, 0.001];
        let (chi, disc, _) = svd_truncation_plan(
            &s,
            &TruncationPolicy::ErrorBounded {
                epsilon: 0.0,
                max_bond: 64,
            },
        );
        assert_eq!(chi, 4);
        assert!(disc < 1e-12);
    }

    #[test]
    fn plan_cap_overrides_eps() {
        let s = vec![1.0, 0.1, 0.01, 0.001];
        let (chi, _, _) = svd_truncation_plan(
            &s,
            &TruncationPolicy::ErrorBounded {
                epsilon: 10.0,
                max_bond: 1,
            },
        );
        assert_eq!(chi, 1);
    }

    #[test]
    fn plan_prunes_null_directions() {
        // A rank-1 spectrum padded with numerical zeros must collapse to χ=1.
        let s = vec![1.0, 1e-15, 1e-16, 0.0];
        let (chi, _, scale) = svd_truncation_plan(&s, &TruncationPolicy::FixedBond(64));
        assert_eq!(chi, 1);
        assert!((scale - 1.0).abs() < 1e-9, "unit-weight input → scale≈1");
    }

    #[test]
    fn fill_left_matches_from_group_left() {
        let m = faer::Mat::from_fn(4, 3, |i, j| Complex::new(i as f64 + 1.0, j as f64 - 0.5));
        let reference = Site::from_group_left_faer(m.as_ref(), 2, 3);
        let mut s = Site::ket0(); // wrong shape on purpose; filler must resize
        s.fill_left_from(m.as_ref(), 2, 3);
        assert_eq!(s, reference);
    }

    #[test]
    fn fill_from_grouped_right_matches_builder() {
        let m = faer::Mat::from_fn(2, 6, |i, j| Complex::new(i as f64 - 0.5, j as f64 + 0.25));
        let reference = Site::from_group_right_faer(m.as_ref(), 2, 3);
        let mut s = Site::ket0();
        s.fill_from_grouped_right(m.as_ref(), 2, 3);
        assert_eq!(s, reference);
    }

    #[test]
    fn fill_right_scaled_conj_matches_manual() {
        // V is (cols=2·right) × (left); here left=2, right=3 → V is 6×2.
        let left = 2usize;
        let right = 3usize;
        let v = faer::Mat::from_fn(2 * right, left, |i, j| {
            Complex::new(i as f64 * 0.1 + 1.0, j as f64 * 0.2 - 0.3)
        });
        let sv = [2.0_f64, 0.5];
        let mut s = Site::ket0();
        s.fill_right_from_scaled_conj(v.as_ref(), &sv, left, right);
        assert_eq!((s.left, s.right), (left, right));
        for l in 0..left {
            for col in 0..2 * right {
                let expected = v[(col, l)].conj() * Complex::new(sv[l], 0.0);
                assert_eq!(s.data[l * 2 * right + col], expected);
            }
        }
    }
}
