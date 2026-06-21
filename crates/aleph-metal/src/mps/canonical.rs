//! Mixed-canonical form for the Metal MPS scaffold (P5.7-07).
//!
//! Tracks an orthogonality centre: every site left of it is left-canonical
//! (`Σ_{l,p} Aᴴ[l,p,r₁]·A[l,p,r₂] = δ`), every site right of it is right-canonical,
//! and the centre site carries the state norm. With the centre on the active bond,
//! a two-site split's gated block has Frobenius² = ⟨ψ|ψ⟩, so the truncated SVD can
//! renormalise the kept σ (`scale = 1/√Σ_kept σ²`) and **apply** a bond-cap
//! truncation with controlled error — instead of refusing it (P5.6-02).
//!
//! The centre is moved one site at a time via a thin SVD of the grouped view (an
//! SVD-based QR/LQ): the orthogonal factor stays on the site, the rest is absorbed
//! into the neighbour. All math is f64 (sites widened f32→f64 then narrowed back),
//! matching the per-gate SVD path's precision profile. A fresh product state is
//! trivially canonical (every bond is dimension 1).

use aleph_backend::BackendError;
use aleph_core::Complex;
use faer::Mat;

use super::state::SiteTensor;
use super::svd::factor;
use crate::MetalContext;

#[inline]
fn widen(z: Complex<f32>) -> Complex {
    Complex::new(z.re as f64, z.im as f64)
}

#[inline]
fn narrow(z: Complex) -> Complex<f32> {
    Complex::<f32>::new(z.re as f32, z.im as f32)
}

/// Move the orthogonality centre to `target`, stepping one site at a time.
pub(crate) fn move_center_to(
    ctx: &MetalContext,
    sites: &mut [SiteTensor],
    center: &mut usize,
    target: usize,
) -> Result<(), BackendError> {
    while *center < target {
        move_center_right(ctx, sites, *center)?;
        *center += 1;
    }
    while *center > target {
        move_center_left(ctx, sites, *center)?;
        *center -= 1;
    }
    Ok(())
}

/// Shift the centre right from site `i` to `i+1`: SVD the grouped-left view of
/// site `i`, leave `U` (left-canonical) on site `i`, and absorb `diag(σ)·Vᴴ` into
/// site `i+1`'s left bond. The shared bond becomes `min(left·2, mid)`.
fn move_center_right(
    ctx: &MetalContext,
    sites: &mut [SiteTensor],
    i: usize,
) -> Result<(), BackendError> {
    let li = sites[i].left;
    let mid = sites[i].right;
    let nr = sites[i + 1].right;
    debug_assert_eq!(mid, sites[i + 1].left, "bond mismatch at {i}");
    let rows = li * 2;
    let size = rows.min(mid);

    // Grouped-left M (rows × mid): M[l*2+p, c] = A_i[(l*2+p)*mid + c].
    let data_i = sites[i].buf.as_slice();
    let m = Mat::<Complex>::from_fn(rows, mid, |row, c| widen(data_i[row * mid + c]));
    let svd = factor(m.as_ref(), size)?;

    // Site i ← U (left-canonical), reshaped (li, 2, size).
    let mut new_i = vec![Complex::<f32>::new(0.0, 0.0); rows * size];
    for row in 0..rows {
        for t in 0..size {
            new_i[row * size + t] = narrow(svd.u[(row, t)]);
        }
    }

    // carry[t, c] = σ_t · conj(V[c, t])   (size × mid).
    // absorbed[t, p*nr+r] = Σ_c carry[t,c] · GR_{i+1}[c, p*nr+r],
    // where GR_{i+1}[c, p*nr+r] = A_{i+1}[(c*2+p)*nr + r].
    let data_j = sites[i + 1].buf.as_slice();
    let mut new_j = vec![Complex::<f32>::new(0.0, 0.0); size * 2 * nr];
    for t in 0..size {
        for p in 0..2 {
            for r in 0..nr {
                let mut acc = Complex::new(0.0, 0.0);
                for c in 0..mid {
                    let carry = svd.v[(c, t)].conj() * svd.sigma[t];
                    acc += carry * widen(data_j[(c * 2 + p) * nr + r]);
                }
                new_j[(t * 2 + p) * nr + r] = narrow(acc);
            }
        }
    }

    // In-place rebuild (P5.8-02): `new_i`/`new_j` are owned, so the `sites[..]`
    // slices read above are no longer borrowed — reuse each site's device buffer.
    sites[i].set_from_host(ctx, li, size, &new_i);
    sites[i + 1].set_from_host(ctx, size, nr, &new_j);
    Ok(())
}

/// Shift the centre left from site `i` to `i-1`: SVD the grouped-right view of
/// site `i`, leave `Vᴴ` (right-canonical) on site `i`, and absorb `U·diag(σ)` into
/// site `i-1`'s right bond. The shared bond becomes `min(li, 2·right)`.
fn move_center_left(
    ctx: &MetalContext,
    sites: &mut [SiteTensor],
    i: usize,
) -> Result<(), BackendError> {
    let li = sites[i].left;
    let ri = sites[i].right;
    let pl = sites[i - 1].left;
    debug_assert_eq!(li, sites[i - 1].right, "bond mismatch at {i}");
    let cols = 2 * ri;
    let size = li.min(cols);

    // Grouped-right GR (li × 2ri): GR[l, p*ri+r] = A_i[(l*2+p)*ri + r].
    let data_i = sites[i].buf.as_slice();
    let gr = Mat::<Complex>::from_fn(li, cols, |l, pc| {
        let p = pc / ri;
        let r = pc % ri;
        widen(data_i[(l * 2 + p) * ri + r])
    });
    let svd = factor(gr.as_ref(), size); // GR = U·diag(σ)·Vᴴ
    let svd = svd?;

    // Site i ← Vᴴ (right-canonical), reshaped (size, 2, ri):
    // A_i[(t*2+p)*ri + r] = conj(V[p*ri+r, t]).
    let mut new_i = vec![Complex::<f32>::new(0.0, 0.0); size * 2 * ri];
    for t in 0..size {
        for p in 0..2 {
            for r in 0..ri {
                new_i[(t * 2 + p) * ri + r] = narrow(svd.v[(p * ri + r, t)].conj());
            }
        }
    }

    // carry[c, t] = U[c, t]·σ_t   (li × size).
    // absorbed[l*2+p, t] = Σ_c GL_{i-1}[l*2+p, c] · carry[c, t],
    // where GL_{i-1}[l*2+p, c] = A_{i-1}[(l*2+p)*li + c].
    let data_h = sites[i - 1].buf.as_slice();
    let mut new_h = vec![Complex::<f32>::new(0.0, 0.0); pl * 2 * size];
    for row in 0..(pl * 2) {
        for t in 0..size {
            let mut acc = Complex::new(0.0, 0.0);
            for c in 0..li {
                acc += widen(data_h[row * li + c]) * (svd.u[(c, t)] * svd.sigma[t]);
            }
            new_h[row * size + t] = narrow(acc);
        }
    }

    // In-place rebuild (P5.8-02): reuse each site's device buffer.
    sites[i - 1].set_from_host(ctx, pl, size, &new_h);
    sites[i].set_from_host(ctx, size, ri, &new_i);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MetalMpsBackend;
    use aleph_backend::Backend;
    use aleph_core::{Gate, GateInstance};

    /// Moving the orthogonality centre across the chain (and back) must leave the
    /// dense statevector unchanged — canonicalisation is norm- and state-preserving.
    #[test]
    fn move_center_preserves_state() {
        let Ok(mut be) = MetalMpsBackend::with_max_bond(64) else {
            eprintln!("skipping canonical test: no Metal device");
            return;
        };
        // Build an entangled 5-qubit state (bonds > 1 so canonicalisation is real).
        let mut s = be.allocate(5).unwrap();
        for q in 0..5u32 {
            be.apply_gate(&mut s, &GateInstance::new(Gate::H, vec![q]))
                .unwrap();
        }
        for q in 0..4u32 {
            be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, vec![q, q + 1]))
                .unwrap();
        }
        let before = s.dense_statevector();

        // Sweep the centre all the way right, then all the way left.
        let ctx = be.ctx();
        move_center_to(ctx, &mut s.sites, &mut s.center, 4).unwrap();
        assert_eq!(s.center, 4);
        move_center_to(ctx, &mut s.sites, &mut s.center, 0).unwrap();
        assert_eq!(s.center, 0);

        let after = s.dense_statevector();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            let d = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
            assert!(
                d < 1e-4,
                "state changed under canonicalisation: |Δ|={d:.2e}"
            );
        }
    }
}
