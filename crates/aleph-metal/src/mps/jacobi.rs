//! Complex one-sided Jacobi thin SVD (P5.7-01) — the CPU reference for the
//! GPU-resident Metal kernel to come (P5.7-02/03).
//!
//! One-sided Jacobi orthogonalizes the *columns* of `A` by right-multiplying 2×2
//! unitary rotations, so the singular values fall out as the converged column norms
//! (`σ_t = ‖A_:,t‖`) and `U_:,t = A_:,t / σ_t`. Unlike forming the Gram matrix
//! `AᴴA` and taking √eigenvalues, the column norms keep full precision — the method
//! never squares the condition number. That is exactly why cuSOLVER's `gesvdj` (and
//! this future Metal port) use one-sided Jacobi for SVD on the GPU at single
//! precision rather than a Gram-then-eig route.
//!
//! Reference: Golub & Van Loan, *Matrix Computations* 4th ed. §8.5 (Jacobi SVD);
//! Drmač & Veselić, "New fast and accurate Jacobi SVD algorithm", SIAM J. Matrix
//! Anal. 29(4), 2008. The complex 2×2 Hermitian sub-problem is real-symmetrized by a
//! `diag(1, e^{-iφ})` phase pre-rotation (φ = arg of the column inner product) before
//! the standard real Jacobi angle.
//!
//! Everything runs in f64 here (the two-site block is widened f32→f64 before
//! factoring, matching the pre-existing faer path), so this reference is at least as
//! accurate as faer's full SVD; the f32 accuracy question belongs to the GPU port.

use aleph_core::Complex;
use faer::{Mat, MatRef};

/// Thin SVD factors: `A(m×n) = U·diag(σ)·Vᴴ` with `U(m×k)`, `V(n×k)`,
/// `k = min(m, n)`, `σ` sorted descending.
pub(crate) struct ThinSvd {
    pub u: Mat<Complex>,
    pub sigma: Vec<f64>,
    pub v: Mat<Complex>,
}

/// Sweep cap. Each sweep annihilates `O(k²)` off-diagonals and converges
/// quadratically; well-conditioned small blocks need ≤8. Past this we declare
/// non-convergence and the caller falls back to faer's full SVD.
const MAX_SWEEPS: usize = 60;

/// A pair `(p, q)` is converged / skippable when
/// `|⟨A_p, A_q⟩| ≤ TOL·‖A_p‖·‖A_q‖`. One-sided Jacobi drives the relative
/// off-diagonal to ~machine eps (~1e-16); this sits safely above that plateau yet
/// far below the 1e-10 reconstruction / 1e-5 oracle tolerances.
const OFFDIAG_TOL: f64 = 1e-14;

/// Apply the 2×2 right rotation `R = [[cs, -sn], [e^{-iφ}·sn, e^{-iφ}·cs]]` to
/// columns `p`, `q` of `mat` in place: `new_p = cs·col_p + (e^{-iφ}sn)·col_q`,
/// `new_q = -sn·col_p + (e^{-iφ}cs)·col_q`. `(es, ec)` carry the phased scalars
/// `e^{-iφ}·sn` and `e^{-iφ}·cs`; the explicit re/im arithmetic mirrors the MSL
/// port (no `Complex` ops in the inner loop).
#[inline]
#[allow(clippy::too_many_arguments)]
fn rotate_cols(
    mat: &mut Mat<Complex>,
    p: usize,
    q: usize,
    cs: f64,
    sn: f64,
    es_re: f64,
    es_im: f64,
    ec_re: f64,
    ec_im: f64,
) {
    let rows = mat.nrows();
    for i in 0..rows {
        let ap = mat[(i, p)];
        let aq = mat[(i, q)];
        let np_re = cs * ap.re + (es_re * aq.re - es_im * aq.im);
        let np_im = cs * ap.im + (es_re * aq.im + es_im * aq.re);
        let nq_re = -sn * ap.re + (ec_re * aq.re - ec_im * aq.im);
        let nq_im = -sn * ap.im + (ec_re * aq.im + ec_im * aq.re);
        mat[(i, p)] = Complex::new(np_re, np_im);
        mat[(i, q)] = Complex::new(nq_re, nq_im);
    }
}

/// One-sided Jacobi for a tall/square `w` (m ≥ n): orthogonalize the columns of
/// `w` in place, accumulating the right rotations into `v` (n×n, pre-seeded to the
/// identity). Returns `false` on non-convergence within [`MAX_SWEEPS`].
fn jacobi_tall(w: &mut Mat<Complex>, v: &mut Mat<Complex>) -> bool {
    let m = w.nrows();
    let n = w.ncols();
    for _sweep in 0..MAX_SWEEPS {
        let mut max_rel = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                // 2×2 column Gram: α = ‖w_p‖², β = ‖w_q‖², γ = w_pᴴ w_q = Σ conj(a)·b.
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut g_re = 0.0;
                let mut g_im = 0.0;
                for i in 0..m {
                    let a = w[(i, p)];
                    let b = w[(i, q)];
                    alpha += a.re * a.re + a.im * a.im;
                    beta += b.re * b.re + b.im * b.im;
                    g_re += a.re * b.re + a.im * b.im;
                    g_im += a.re * b.im - a.im * b.re;
                }
                if alpha <= 0.0 || beta <= 0.0 {
                    continue; // a null column is already orthogonal to everything
                }
                let gabs = (g_re * g_re + g_im * g_im).sqrt();
                let scale = (alpha.sqrt()) * (beta.sqrt());
                let rel = gabs / scale;
                if rel > max_rel {
                    max_rel = rel;
                }
                if gabs <= OFFDIAG_TOL * scale {
                    continue;
                }
                // Phase pre-rotation: e^{-iφ} = conj(γ)/|γ| with γ = |γ|e^{iφ}.
                let inv = 1.0 / gabs;
                let ephi_re = g_re * inv; //  cos φ
                let ephi_im = -g_im * inv; // -sin φ  ⇒ e^{-iφ} = (cos φ, -sin φ)
                                           // Real Jacobi angle that diagonalizes [[α, |γ|], [|γ|, β]] under the
                                           // right-rotation `A·R` (new_p = c·p + s·q): cot 2θ = (α−β)/(2|γ|).
                let tau = (alpha - beta) / (2.0 * gabs);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * cs;
                // R = diag(1, e^{-iφ}) · [[cs, -sn], [sn, cs]].
                let es_re = ephi_re * sn;
                let es_im = ephi_im * sn;
                let ec_re = ephi_re * cs;
                let ec_im = ephi_im * cs;
                rotate_cols(w, p, q, cs, sn, es_re, es_im, ec_re, ec_im);
                rotate_cols(v, p, q, cs, sn, es_re, es_im, ec_re, ec_im);
            }
        }
        if max_rel <= OFFDIAG_TOL {
            return true;
        }
    }
    false
}

/// `n×n` identity over `Complex`.
fn identity(n: usize) -> Mat<Complex> {
    let mut m = Mat::<Complex>::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = Complex::new(1.0, 0.0);
    }
    m
}

/// Recover `(U, σ, V)` from an orthogonalized `w` (m×n, m ≥ n, columns mutually
/// orthogonal) and the accumulated right-rotation `v` (n×n): `σ_t = ‖w_:,t‖`,
/// `U_:,t = w_:,t/σ_t`, with columns reordered to descending σ.
fn finish(w: &Mat<Complex>, v: &Mat<Complex>) -> ThinSvd {
    let m = w.nrows();
    let k = w.ncols();
    let n = v.nrows();
    let mut col_norm = vec![0.0f64; k];
    for (t, norm) in col_norm.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..m {
            let z = w[(i, t)];
            s += z.re * z.re + z.im * z.im;
        }
        *norm = s.sqrt();
    }
    let mut order: Vec<usize> = (0..k).collect();
    // Descending σ; partial_cmp is safe — norms are finite sums of squares.
    order.sort_by(|&x, &y| col_norm[y].partial_cmp(&col_norm[x]).unwrap());

    let mut u = Mat::<Complex>::zeros(m, k);
    let mut vout = Mat::<Complex>::zeros(n, k);
    let mut sigma = vec![0.0f64; k];
    for (t_new, &t_old) in order.iter().enumerate() {
        let s = col_norm[t_old];
        sigma[t_new] = s;
        // A null direction (σ ≈ 0) leaves a zero U column; it carries no weight in
        // A = UΣVᴴ and the caller's truncation prunes it below the null floor.
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for i in 0..m {
            let z = w[(i, t_old)];
            u[(i, t_new)] = Complex::new(z.re * inv, z.im * inv);
        }
        for i in 0..n {
            vout[(i, t_new)] = v[(i, t_old)];
        }
    }
    ThinSvd { u, sigma, v: vout }
}

/// Thin one-sided Jacobi SVD of `a` (m×n). Returns `None` on non-convergence so the
/// caller can fall back to a direct SVD. Wide inputs (m < n) are factored via the
/// adjoint `Aᴴ` (tall) and the U/V roles swapped: `Aᴴ = Ũ Σ Ṽᴴ ⇒ A = Ṽ Σ Ũᴴ`.
pub(crate) fn jacobi_thin_svd(a: MatRef<'_, Complex>) -> Option<ThinSvd> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Some(ThinSvd {
            u: Mat::zeros(m, 0),
            sigma: Vec::new(),
            v: Mat::zeros(n, 0),
        });
    }
    if m >= n {
        let mut w = a.to_owned();
        let mut v = identity(n);
        if !jacobi_tall(&mut w, &mut v) {
            return None;
        }
        Some(finish(&w, &v))
    } else {
        // Aᴴ is n×m (tall); factor it, then swap singular-vector roles.
        let mut w = Mat::<Complex>::from_fn(n, m, |i, j| a[(j, i)].conj());
        let mut v = identity(m);
        if !jacobi_tall(&mut w, &mut v) {
            return None;
        }
        let fin = finish(&w, &v);
        Some(ThinSvd {
            u: fin.v,
            sigma: fin.sigma,
            v: fin.u,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Deterministic complex test matrix (same generator family as the CPU MPS
    /// linalg tests so the two reference paths exercise comparable data).
    fn test_matrix(m: usize, n: usize) -> Mat<Complex> {
        Mat::from_fn(m, n, |i, j| {
            Complex::new(
                ((i * 7 + j * 3) % 11) as f64 * 0.37 - 1.1,
                ((i * 5 + j) % 7) as f64 * 0.23 - 0.6,
            )
        })
    }

    /// Reconstruction `A ≈ U diag(σ) Vᴴ` to 1e-10, σ non-increasing and ≥ 0.
    fn assert_reconstructs(a: &Mat<Complex>) {
        let (m, n) = a.shape();
        let svd = jacobi_thin_svd(a.as_ref()).expect("jacobi converged");
        let k = m.min(n);
        assert_eq!(svd.sigma.len(), k);
        assert_eq!(svd.u.shape(), (m, k));
        assert_eq!(svd.v.shape(), (n, k));
        for t in 1..k {
            assert!(
                svd.sigma[t - 1] >= svd.sigma[t] - 1e-12,
                "σ must be descending: {:?}",
                svd.sigma
            );
            assert!(svd.sigma[t] >= -1e-12, "σ ≥ 0");
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for t in 0..k {
                    acc += svd.u[(i, t)] * svd.sigma[t] * svd.v[(j, t)].conj();
                }
                let d = ((acc.re - a[(i, j)].re).powi(2) + (acc.im - a[(i, j)].im).powi(2)).sqrt();
                assert!(d < 1e-10, "reconstruct ({i},{j}) |Δ|={d:.2e}");
            }
        }
    }

    /// U and V columns must be orthonormal (for the non-null singular values).
    fn assert_isometry(a: &Mat<Complex>) {
        let (m, n) = a.shape();
        let svd = jacobi_thin_svd(a.as_ref()).expect("converged");
        let k = m.min(n);
        let dot = |mat: &Mat<Complex>, rows: usize, c0: usize, c1: usize| {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..rows {
                let x = mat[(i, c0)];
                let y = mat[(i, c1)];
                re += x.re * y.re + x.im * y.im; // conj(x)·y
                im += x.re * y.im - x.im * y.re;
            }
            (re, im)
        };
        for c0 in 0..k {
            if svd.sigma[c0] < 1e-9 {
                continue; // null U column is zero by construction
            }
            let (re, im) = dot(&svd.u, m, c0, c0);
            assert!(
                (re - 1.0).abs() < 1e-9 && im.abs() < 1e-9,
                "U col {c0} norm"
            );
            for c1 in (c0 + 1)..k {
                let (re, im) = dot(&svd.u, m, c0, c1);
                assert!(re.abs() < 1e-9 && im.abs() < 1e-9, "U {c0}⊥{c1}");
                let (vre, vim) = dot(&svd.v, n, c0, c1);
                assert!(vre.abs() < 1e-9 && vim.abs() < 1e-9, "V {c0}⊥{c1}");
            }
        }
    }

    #[test]
    fn reconstructs_tall_square_wide() {
        for (m, n) in [
            (8, 5),
            (5, 8),
            (6, 6),
            (1, 1),
            (2, 1),
            (1, 2),
            (10, 3),
            (3, 10),
        ] {
            assert_reconstructs(&test_matrix(m, n));
            assert_isometry(&test_matrix(m, n));
        }
    }

    /// Singular values must match faer's thin SVD (the σ are unique and sortable,
    /// so this is a strong cross-check independent of singular-vector phase/sign).
    #[test]
    fn singular_values_match_faer() {
        for (m, n) in [(8, 5), (5, 8), (6, 6), (12, 4), (4, 12), (7, 7)] {
            let a = test_matrix(m, n);
            let hl = a.thin_svd().unwrap();
            let svd = jacobi_thin_svd(a.as_ref()).expect("converged");
            let k = m.min(n);
            for t in 0..k {
                let f = hl.S()[t].re;
                let d = (f - svd.sigma[t]).abs();
                assert!(
                    d < 1e-10,
                    "σ[{t}] jacobi {} vs faer {f} (Δ={d:.2e})",
                    svd.sigma[t]
                );
            }
        }
    }

    /// Rank-deficient input: two identical columns ⇒ one zero singular value. The
    /// null direction must sort last and reconstruction must still hold.
    #[test]
    fn rank_deficient_block() {
        let mut a = test_matrix(6, 4);
        for i in 0..6 {
            a[(i, 2)] = a[(i, 1)]; // duplicate a column → rank ≤ 3
        }
        assert_reconstructs(&a);
        let svd = jacobi_thin_svd(a.as_ref()).expect("converged");
        assert!(
            svd.sigma[3] < 1e-9,
            "smallest σ should be ≈0 for a rank-deficient block, got {}",
            svd.sigma[3]
        );
    }

    /// Bell two-site block (the truncation-test fixture): σ = {1/√2, 1/√2}.
    #[test]
    fn bell_block_singular_values() {
        let r = (0.5f64).sqrt();
        let a = Mat::from_fn(2, 2, |i, j| {
            if i == j {
                Complex::new(r, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        });
        let svd = jacobi_thin_svd(a.as_ref()).expect("converged");
        assert!((svd.sigma[0] - r).abs() < 1e-12);
        assert!((svd.sigma[1] - r).abs() < 1e-12);
        assert_reconstructs(&a);
    }
}
