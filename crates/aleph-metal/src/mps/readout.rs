//! MPS-on-Metal readout (P5.7-05): `measure` / `sample` / `probabilities` /
//! `expectation_value` for the scaffold `MetalMpsState`, without ever forming the
//! dense `2^n` vector.
//!
//! The scaffold is **non-canonical** (no orthogonality centre — that is P5.7-07),
//! so these cannot lean on the canonical-form shortcuts the CPU MPS uses
//! (`aleph_mps`: right-canonicalise, then a trivial environment). Instead every
//! quantity is a *doubled* (bra⊗ket) transfer-matrix sweep that is exact for any
//! MPS: each environment is a `bond × bond` matrix, accumulated site by site.
//!
//! The site tensors live in unified memory, so the host reads them zero-copy
//! (`SiteTensor::buf.as_slice()`); each f32 entry is widened to f64 for the
//! contraction accuracy the 1e-5 oracle needs. The contractions are `O(n·χ³)` (or
//! `O(shots·n·χ²)` for sampling) on `bond × bond` matrices — tiny next to the
//! circuit evolution — so they run on the host; offloading the per-site transfer
//! to the GPU is a future optimisation, not needed for correctness.
//!
//! Site order ≡ qubit order in the scaffold (NN-only, no SWAP router), so logical
//! qubit `q` is physical site `q` throughout.

use aleph_backend::BackendError;
use aleph_core::{Complex, Pauli, PauliString};
use rand::Rng;

use super::state::{MetalMpsState, SiteTensor};

/// Largest subset size for [`probabilities`] (2^k output), mirroring the CPU MPS
/// cap so the joint marginal never allocates an unbounded vector.
pub(crate) const MAX_PROB_QUBITS: usize = 20;

/// One bra⊗ket environment matrix, row-major `dim × dim`: `e[a*dim + b]` is the
/// (bra bond `a`, ket bond `b`) entry. `dim` is the bond the environment sits on.
struct Env {
    dim: usize,
    data: Vec<Complex<f64>>,
}

impl Env {
    /// The trivial 1×1 boundary environment `[1]` (left of site 0 / right of the
    /// last site: both open bonds are dimension 1).
    fn unit() -> Self {
        Env {
            dim: 1,
            data: vec![Complex::new(1.0, 0.0)],
        }
    }

    #[inline]
    fn get(&self, a: usize, b: usize) -> Complex<f64> {
        self.data[a * self.dim + b]
    }
}

/// Widened f64 view of a site entry `A[l, p, r]` (row-major `(left, 2, right)`).
#[inline]
fn site_get(
    site: &SiteTensor,
    data: &[Complex<f32>],
    l: usize,
    p: usize,
    r: usize,
) -> Complex<f64> {
    let z = data[(l * 2 + p) * site.right + r];
    Complex::new(z.re as f64, z.im as f64)
}

/// Advance a *left* environment across one site with a 2×2 local operator `op`
/// acting on the ket physical leg:
///   `E'[rb,rk] = Σ_{lb,lk,p,p'} conj(A[lb,p,rb]) · E[lb,lk] · op[p][p'] · A[lk,p',rk]`.
/// `op = I` gives the plain norm-transfer (trace over the physical leg).
fn transfer_op(site: &SiteTensor, e: &Env, op: &[[Complex<f64>; 2]; 2]) -> Env {
    debug_assert_eq!(e.dim, site.left);
    let data = site.buf.as_slice();
    let (din, dout) = (site.left, site.right);
    // ket_p[lb, rk] = Σ_lk E[lb,lk] · A[lk, p, rk], for p = 0,1.
    let mut ket = [
        vec![Complex::new(0.0, 0.0); din * dout],
        vec![Complex::new(0.0, 0.0); din * dout],
    ];
    #[allow(clippy::needless_range_loop)]
    for p in 0..2 {
        for lb in 0..din {
            for rk in 0..dout {
                let mut acc = Complex::new(0.0, 0.0);
                for lk in 0..din {
                    acc += e.get(lb, lk) * site_get(site, data, lk, p, rk);
                }
                ket[p][lb * dout + rk] = acc;
            }
        }
    }
    // E'[rb,rk] = Σ_p Σ_p' op[p][p'] · Σ_lb conj(A[lb,p,rb]) · ket_p'[lb,rk].
    let mut out = vec![Complex::new(0.0, 0.0); dout * dout];
    #[allow(clippy::needless_range_loop)]
    for p in 0..2 {
        for pp in 0..2 {
            let o = op[p][pp];
            if o.re == 0.0 && o.im == 0.0 {
                continue;
            }
            for rb in 0..dout {
                for rk in 0..dout {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lb in 0..din {
                        acc += site_get(site, data, lb, p, rb).conj() * ket[pp][lb * dout + rk];
                    }
                    out[rb * dout + rk] += o * acc;
                }
            }
        }
    }
    Env {
        dim: dout,
        data: out,
    }
}

/// Advance a left environment with a *single* physical index `p` on both bra and
/// ket (the diagonal projector branch used by [`probabilities`]):
///   `E'[rb,rk] = Σ_{lb,lk} conj(A[lb,p,rb]) · E[lb,lk] · A[lk,p,rk]`.
fn transfer_p(site: &SiteTensor, e: &Env, p: usize) -> Env {
    let mut op = [[Complex::new(0.0, 0.0); 2]; 2];
    op[p][p] = Complex::new(1.0, 0.0);
    transfer_op(site, e, &op)
}

/// `⟨ψ|ψ⟩` via a plain norm-transfer sweep (op = I at every site). Real and ≥ 0.
pub(crate) fn norm_sq(state: &MetalMpsState) -> f64 {
    let mut e = Env::unit();
    for site in &state.sites {
        e = transfer_op(site, &e, &IDENTITY_OP);
    }
    debug_assert_eq!(e.dim, 1);
    e.get(0, 0).re
}

const IDENTITY_OP: [[Complex<f64>; 2]; 2] = [
    [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
];

/// Exact joint marginal over `qubits` (output length `2^k`, bit `pos` ↔
/// `qubits[pos]`), normalised by `⟨ψ|ψ⟩`. Empty subset → `[1.0]`. The doubled
/// sweep branches into a separate environment per measured-bit assignment and
/// traces over every other site — no `2^n` intermediate.
pub(crate) fn probabilities(
    state: &MetalMpsState,
    qubits: &[u32],
) -> Result<Vec<f64>, BackendError> {
    let n = state.sites.len();
    if qubits.is_empty() {
        return Ok(vec![1.0]);
    }
    if qubits.len() > MAX_PROB_QUBITS {
        return Err(BackendError::TooManyQubits {
            requested: qubits.len() as u32,
            limit: MAX_PROB_QUBITS as u32,
        });
    }
    let mut seen = Vec::new();
    let mut out_bit_for_site: Vec<Option<usize>> = vec![None; n];
    for (pos, &q) in qubits.iter().enumerate() {
        if (q as usize) >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits: n as u32,
            });
        }
        if seen.contains(&q) {
            return Err(BackendError::DuplicateQubit { qubit: q });
        }
        seen.push(q);
        out_bit_for_site[q as usize] = Some(pos); // identity site↔qubit map
    }

    // (output index so far, environment). Starts as the unit boundary.
    let mut envs: Vec<(usize, Env)> = vec![(0usize, Env::unit())];
    for (i, out_bit) in out_bit_for_site.iter().enumerate() {
        let site = &state.sites[i];
        match out_bit {
            None => {
                for (_, e) in envs.iter_mut() {
                    let a = transfer_p(site, e, 0);
                    let b = transfer_p(site, e, 1);
                    let mut sum = vec![Complex::new(0.0, 0.0); a.data.len()];
                    for (s, (x, y)) in sum.iter_mut().zip(a.data.iter().zip(b.data.iter())) {
                        *s = *x + *y;
                    }
                    *e = Env {
                        dim: a.dim,
                        data: sum,
                    };
                }
            }
            Some(pos) => {
                let mut next = Vec::with_capacity(envs.len() * 2);
                for (idx, e) in &envs {
                    next.push((*idx, transfer_p(site, e, 0)));
                    next.push((*idx | (1 << pos), transfer_p(site, e, 1)));
                }
                envs = next;
            }
        }
    }

    let norm = norm_sq(state);
    let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    let dim = 1usize << qubits.len();
    let mut out = vec![0.0; dim];
    for (idx, e) in envs {
        debug_assert_eq!(e.dim, 1);
        out[idx] = (e.get(0, 0).re * inv).max(0.0);
    }
    Ok(out)
}

/// `⟨ψ|P|ψ⟩ / ⟨ψ|ψ⟩ · coefficient` for a Pauli string. The expectation of a
/// Hermitian observable is real; the (≈0) imaginary part is discarded. A single
/// doubled sweep applies each listed Pauli to the ket leg at its site.
pub(crate) fn expectation(state: &MetalMpsState, p: &PauliString) -> Result<f64, BackendError> {
    let n = state.sites.len();
    // Per-site 2×2 operator (identity unless the qubit is in the string).
    let mut ops = vec![IDENTITY_OP; n];
    for (q, pauli) in &p.terms {
        if (*q as usize) >= n {
            return Err(BackendError::QubitOutOfRange {
                qubit: *q,
                num_qubits: n as u32,
            });
        }
        if let Pauli::I = pauli {
            continue;
        }
        let m = pauli.matrix();
        ops[*q as usize] = [[m[0][0], m[0][1]], [m[1][0], m[1][1]]];
    }

    let mut e = Env::unit();
    for (site, op) in state.sites.iter().zip(ops.iter()) {
        e = transfer_op(site, &e, op);
    }
    let num = e.get(0, 0);
    let denom = norm_sq(state);
    if denom <= 0.0 {
        return Ok(0.0);
    }
    Ok(p.coefficient * (num.re / denom))
}

/// Perfect (Ferris–Vidal) sampling without canonical form: precompute the right
/// environments `R[i]` (norm of sites `i..n` traced over their physical legs),
/// then sweep left→right drawing each bit from its exact conditional probability
/// `wᵦ† R[i+1] wᵦ`. Each shot packs qubit `q` into bit `q`. Does not mutate state.
pub(crate) fn sample<R: Rng>(state: &MetalMpsState, shots: u32, rng: &mut R) -> Vec<u64> {
    let n = state.sites.len();
    if n == 0 {
        return vec![0u64; shots as usize];
    }
    // right[i] = environment on the left bond of site i (dim = site_i.left),
    // contracting sites i..n over both physical legs. right[n] is the unit 1×1.
    let mut right: Vec<Env> = Vec::with_capacity(n + 1);
    right.push(Env::unit()); // right[n]
    for i in (0..n).rev() {
        let site = &state.sites[i];
        right.push(transfer_right(site, right.last().unwrap()));
    }
    right.reverse(); // now right[i] indexes site i.left; right[n] is unit

    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        let mut c = vec![Complex::new(1.0, 0.0)]; // ket boundary, left bond of site 0
        let mut bits = 0u64;
        for i in 0..n {
            let site = &state.sites[i];
            let data = site.buf.as_slice();
            // w_b[r] = Σ_l c[l] · A[l, b, r].
            let mut w = [
                vec![Complex::new(0.0, 0.0); site.right],
                vec![Complex::new(0.0, 0.0); site.right],
            ];
            #[allow(clippy::needless_range_loop)]
            for b in 0..2 {
                for r in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for l in 0..site.left {
                        acc += c[l] * site_get(site, data, l, b, r);
                    }
                    w[b][r] = acc;
                }
            }
            // weight_b = w_b† · R[i+1] · w_b  (real, ≥ 0).
            let renv = &right[i + 1];
            let weight = |wb: &[Complex<f64>]| -> f64 {
                let mut acc = Complex::new(0.0, 0.0);
                for rb in 0..renv.dim {
                    for rk in 0..renv.dim {
                        acc += wb[rb].conj() * renv.get(rb, rk) * wb[rk];
                    }
                }
                acc.re
            };
            let w0 = weight(&w[0]).max(0.0);
            let w1 = weight(&w[1]).max(0.0);
            let total = w0 + w1;
            let outcome = total > 0.0 && rng.gen::<f64>() * total >= w0;
            let b = if outcome { 1usize } else { 0usize };
            if outcome {
                bits |= 1u64 << i; // identity site↔qubit map
            }
            let wb = if outcome { w1 } else { w0 };
            let scale = if wb > 0.0 { (1.0 / wb).sqrt() } else { 0.0 };
            c = w[b].iter().map(|z| *z * Complex::new(scale, 0.0)).collect();
        }
        out.push(bits);
    }
    out
}

/// Build the right environment on `site.left` from the one on `site.right`:
///   `R[i][lb,lk] = Σ_{p,rb,rk} conj(A[lb,p,rb]) · R[i+1][rb,rk] · A[lk,p,rk]`.
fn transfer_right(site: &SiteTensor, r_next: &Env) -> Env {
    debug_assert_eq!(r_next.dim, site.right);
    let data = site.buf.as_slice();
    let (din, dout) = (site.left, site.right);
    let mut out = vec![Complex::new(0.0, 0.0); din * din];
    // ket_p[lk, rb] = Σ_rk R[i+1][rb,rk] · A[lk,p,rk].
    #[allow(clippy::needless_range_loop)]
    for p in 0..2 {
        let mut ket = vec![Complex::new(0.0, 0.0); din * dout];
        for lk in 0..din {
            for rb in 0..dout {
                let mut acc = Complex::new(0.0, 0.0);
                for rk in 0..dout {
                    acc += r_next.get(rb, rk) * site_get(site, data, lk, p, rk);
                }
                ket[lk * dout + rb] = acc;
            }
        }
        for lb in 0..din {
            for lk in 0..din {
                let mut acc = Complex::new(0.0, 0.0);
                for rb in 0..dout {
                    acc += site_get(site, data, lb, p, rb).conj() * ket[lk * dout + rb];
                }
                out[lb * din + lk] += acc;
            }
        }
    }
    Env {
        dim: din,
        data: out,
    }
}

/// Measure qubit `q` in the Z basis, collapsing the state, and return the bit.
/// The single-qubit marginal comes from the doubled sweep; the collapse projects
/// site `q`'s physical leg to the outcome and rescales it so the post-measurement
/// state keeps its norm (exactly the CPU MPS contract).
pub(crate) fn measure<R: Rng>(
    state: &mut MetalMpsState,
    q: usize,
    rng: &mut R,
) -> Result<bool, BackendError> {
    let n = state.sites.len();
    if q >= n {
        return Err(BackendError::QubitOutOfRange {
            qubit: q as u32,
            num_qubits: n as u32,
        });
    }
    let probs = probabilities(state, &[q as u32])?;
    let (p0, p1) = (probs[0], probs[1]);
    let total = p0 + p1;
    if total <= 0.0 {
        return Err(BackendError::InvalidState {
            reason: "degenerate measurement: zero-probability qubit",
        });
    }
    let outcome = rng.gen::<f64>() * total >= p0;
    let keep = if outcome { 1usize } else { 0usize };
    let pk = if outcome { p1 } else { p0 };
    // Rescale the projected state back to the pre-measurement norm (= `total`).
    let scale = (total / pk).sqrt() as f32;

    let site = &mut state.sites[q];
    let (left, right) = (site.left, site.right);
    let buf = site.buf.as_mut_slice();
    let drop = 1 - keep;
    for l in 0..left {
        for r in 0..right {
            buf[(l * 2 + drop) * right + r] = Complex::new(0.0, 0.0);
            let v = buf[(l * 2 + keep) * right + r];
            buf[(l * 2 + keep) * right + r] = Complex::new(v.re * scale, v.im * scale);
        }
    }
    Ok(outcome)
}
