//! Host f64 SVD mixed-canonical centre move (P5.7-07) — the **fallback** for the
//! fused GPU-resident QR move (P5.8-04, `super::move_gpu`). Used when a block exceeds
//! the GPU QR kernel's size cap. Moves the orthogonality centre one site at a time:
//! the orthogonal factor stays on the site, `diag(σ)·Vᴴ` (or `U·diag(σ)`) is absorbed
//! into the neighbour. All math is f64. A fresh product state is trivially canonical.

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

/// Move the orthogonality centre to `target`, stepping one site at a time (host SVD).
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

fn move_center_right(
    ctx: &MetalContext,
    sites: &mut [SiteTensor],
    i: usize,
) -> Result<(), BackendError> {
    let li = sites[i].left;
    let mid = sites[i].right;
    let nr = sites[i + 1].right;
    let rows = li * 2;
    let size = rows.min(mid);

    let data_i = sites[i].buf.as_slice();
    let m = Mat::<Complex>::from_fn(rows, mid, |row, c| widen(data_i[row * mid + c]));
    let svd = factor(m.as_ref(), size)?;

    let mut new_i = vec![Complex::<f32>::new(0.0, 0.0); rows * size];
    for row in 0..rows {
        for t in 0..size {
            new_i[row * size + t] = narrow(svd.u[(row, t)]);
        }
    }
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
    sites[i].set_from_host(ctx, li, size, &new_i);
    sites[i + 1].set_from_host(ctx, size, nr, &new_j);
    Ok(())
}

fn move_center_left(
    ctx: &MetalContext,
    sites: &mut [SiteTensor],
    i: usize,
) -> Result<(), BackendError> {
    let li = sites[i].left;
    let ri = sites[i].right;
    let pl = sites[i - 1].left;
    let cols = 2 * ri;
    let size = li.min(cols);

    let data_i = sites[i].buf.as_slice();
    let gr = Mat::<Complex>::from_fn(li, cols, |l, pc| {
        let p = pc / ri;
        let r = pc % ri;
        widen(data_i[(l * 2 + p) * ri + r])
    });
    let svd = factor(gr.as_ref(), size)?;

    let mut new_i = vec![Complex::<f32>::new(0.0, 0.0); size * 2 * ri];
    for t in 0..size {
        for p in 0..2 {
            for r in 0..ri {
                new_i[(t * 2 + p) * ri + r] = narrow(svd.v[(p * ri + r, t)].conj());
            }
        }
    }
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

    /// Host-SVD centre sweep must leave the dense statevector unchanged (the GPU-fused
    /// move path is exercised by the oracle/canonical tests via `run`).
    #[test]
    fn move_center_preserves_state() {
        let Ok(mut be) = MetalMpsBackend::with_max_bond(64) else {
            eprintln!("skipping canonical test: no Metal device");
            return;
        };
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

        let ctx = be.ctx();
        move_center_to(ctx, &mut s.sites, &mut s.center, 4).unwrap();
        assert_eq!(s.center, 4);
        move_center_to(ctx, &mut s.sites, &mut s.center, 0).unwrap();
        assert_eq!(s.center, 0);

        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            let d = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
            assert!(
                d < 1e-4,
                "state changed under canonicalisation: |Δ|={d:.2e}"
            );
        }
    }
}
