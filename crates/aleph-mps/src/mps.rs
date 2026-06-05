//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application, expectation, measurement, sampling, probabilities.

use crate::tensor::{thin_qr, truncated_svd, Site, TruncationPolicy};
use crate::MpsError;
use aleph_core::{Complex, Gate, GateInstance, PauliString};
use nalgebra::DMatrix;
use rand::Rng;

/// Largest subset size `probabilities` will materialize (output is 2^k).
pub(crate) const MAX_PROB_QUBITS: usize = 20;

/// Mixed-canonical MPS. Sites left of `center` are left-canonical, sites right
/// are right-canonical; the center site carries the norm.
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) policy: TruncationPolicy,
    pub(crate) trunc_error: f64,
    pub(crate) max_bond_seen: usize,
}

impl MpsState {
    /// Allocate |0…0⟩ on `n` qubits with a fixed bond cap `max_bond`.
    pub fn new(n: usize, max_bond: usize) -> Self {
        Self::with_policy(n, TruncationPolicy::FixedBond(max_bond.max(1)))
    }

    /// Allocate |0…0⟩ on `n` qubits with an explicit truncation policy.
    pub fn with_policy(n: usize, policy: TruncationPolicy) -> Self {
        let sites = (0..n).map(|_| Site::ket0()).collect();
        MpsState {
            sites,
            center: 0,
            policy,
            trunc_error: 0.0,
            max_bond_seen: 1,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.sites.len()
    }

    /// Accumulated discarded Schmidt weight from all SVD truncations so far.
    pub fn truncation_error(&self) -> f64 {
        self.trunc_error
    }

    /// The largest bond dimension reached by any 2q truncation so far.
    pub fn max_bond_reached(&self) -> usize {
        self.max_bond_seen
    }

    /// Apply a 1q unitary to site `i` (qubit `i`). Preserves canonical form,
    /// so neither the center nor any SVD is touched.
    pub(crate) fn apply_1q(&mut self, i: usize, u: &[[Complex; 2]; 2]) {
        let site = &mut self.sites[i];
        for l in 0..site.left {
            for r in 0..site.right {
                let a0 = site.get(l, 0, r);
                let a1 = site.get(l, 1, r);
                *site.get_mut(l, 0, r) = u[0][0] * a0 + u[0][1] * a1;
                *site.get_mut(l, 1, r) = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }

    /// Apply a 2q gate (4×4 matrix `u`) on the qubits named by `g`
    /// (`g.qubits[0]`=MSB, ADR-0004). Adjacent pairs apply directly; non-adjacent
    /// pairs are brought together by a nearest-neighbor SWAP network, the gate is
    /// applied, then the SWAPs are undone (always-swap-back), so site = qubit is
    /// preserved. `MpsError::NonNearestNeighbor` is retained as a defensive
    /// invariant guard but is no longer reached on the normal 2q path.
    pub(crate) fn apply_2q(
        &mut self,
        g: &GateInstance,
        u: &[[Complex; 4]; 4],
    ) -> Result<(), MpsError> {
        let qa = g.qubits[0];
        let qb = g.qubits[1];
        if qa.abs_diff(qb) == 1 {
            return self.apply_2q_adjacent(qa, qb, u);
        }
        let lo = qa.min(qb);
        let hi = qa.max(qb);
        // Forward ladder: move the qubit at site `hi` down to site `lo+1`.
        for k in (lo as usize + 1..=hi as usize - 1).rev() {
            self.swap_adjacent(k)?;
        }
        // qubit `lo` is at site `lo`, qubit `hi` is now at site `lo+1`.
        // Apply on the adjacent pair preserving original control/target order.
        let (s0, s1) = if qa < qb { (lo, lo + 1) } else { (lo + 1, lo) };
        self.apply_2q_adjacent(s0, s1, u)?;
        // Reverse ladder: undo the SWAPs, restoring site = qubit.
        for k in lo as usize + 1..=hi as usize - 1 {
            self.swap_adjacent(k)?;
        }
        Ok(())
    }

    /// Apply a 2q unitary `u` to the nearest-neighbor pair `(q0, q1)`.
    ///
    /// Caller must ensure `q0.abs_diff(q1) == 1`. The MSB convention (ADR-0004)
    /// is preserved: `q0` maps to the most-significant bit of the matrix index.
    fn apply_2q_adjacent(
        &mut self,
        q0: u32,
        q1: u32,
        u: &[[Complex; 4]; 4],
    ) -> Result<(), MpsError> {
        let i = q0.min(q1) as usize;
        let j = i + 1;

        // Move the orthogonality center to site i so that the two-site
        // contraction preserves normalization after re-factorization.
        self.move_center_to(i);

        let li = self.sites[i].left;
        let mi = self.sites[i].right; // shared bond between site i and j
        let ri = self.sites[j].right;

        // Build the two-site tensor Θ[l, a, b, r] = Σ_m sites[i][l,a,m] · sites[j][m,b,r]
        // Flat index: ((l * 2 + a) * 2 + b) * ri + r
        let theta_len = li * 2 * 2 * ri;
        let mut theta = vec![Complex::new(0.0, 0.0); theta_len];

        // Explicit loops — nested-tensor contractions are clearer than iterators here.
        #[allow(clippy::needless_range_loop)]
        for l in 0..li {
            for a in 0..2usize {
                for b in 0..2usize {
                    for r in 0..ri {
                        let mut acc = Complex::new(0.0, 0.0);
                        for m in 0..mi {
                            acc += self.sites[i].get(l, a, m) * self.sites[j].get(m, b, r);
                        }
                        theta[((l * 2 + a) * 2 + b) * ri + r] = acc;
                    }
                }
            }
        }

        // Helper: given the physical indices of site i (phys_i) and site j (phys_j),
        // return the 2q matrix row/column index following the MSB convention:
        // q0 is the MSB, q1 is the LSB (ADR-0004 / P0-06 convention).
        let out = |phys_i: usize, phys_j: usize| -> usize {
            // Identify which physical index maps to q0 (MSB) and q1 (LSB).
            let bit_q0 = if q0 as usize == i { phys_i } else { phys_j };
            let bit_q1 = if q1 as usize == i { phys_i } else { phys_j };
            (bit_q0 << 1) | bit_q1
        };

        // Apply the gate: Θ'[l,a',b',r] = Σ_{a,b} U[out(a',b')][out(a,b)] · Θ[l,a,b,r]
        let mut theta2 = vec![Complex::new(0.0, 0.0); theta_len];

        #[allow(clippy::needless_range_loop)]
        for l in 0..li {
            for ap in 0..2usize {
                for bp in 0..2usize {
                    let row = out(ap, bp);
                    for a in 0..2usize {
                        for b in 0..2usize {
                            let col = out(a, b);
                            let u_entry = u[row][col];
                            if u_entry == Complex::new(0.0, 0.0) {
                                continue;
                            }
                            for r in 0..ri {
                                theta2[((l * 2 + ap) * 2 + bp) * ri + r] +=
                                    u_entry * theta[((l * 2 + a) * 2 + b) * ri + r];
                            }
                        }
                    }
                }
            }
        }

        // Reshape Θ' to matrix M of shape (li*2) × (2*ri):
        //   row = l*2 + a'  (group left bond and physical of site i)
        //   col = b'*ri + r (group physical of site j and right bond)
        // This matches from_group_left / from_group_right conventions.
        let m = DMatrix::from_fn(li * 2, 2 * ri, |row, col| {
            let l = row / 2;
            let ap = row % 2;
            let bp = col / ri;
            let r = col % ri;
            theta2[((l * 2 + ap) * 2 + bp) * ri + r]
        });

        // Truncated SVD: M = U · diag(s) · Vt, truncated according to the state's policy.
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, &self.policy)?;
        self.trunc_error += discarded;
        let chi = s_kept.len();
        self.max_bond_seen = self.max_bond_seen.max(chi);

        // New site i: left-canonical from the U factor, shape (li, chi).
        self.sites[i] = Site::from_group_left(&u_s, li, chi);

        // New site j: multiply singular values into Vt rows, shape (chi, ri) physical-grouped.
        // sv[(row, col)] = s_kept[row] * vt_s[(row, col)]
        let mut sv = vt_s;
        for row in 0..chi {
            for col in 0..sv.ncols() {
                sv[(row, col)] *= s_kept[row];
            }
        }
        // sv has shape chi × (2*ri) — matches from_group_right(left=chi, right=ri).
        self.sites[j] = Site::from_group_right(&sv, chi, ri);
        self.center = j;

        Ok(())
    }

    /// Swap the qubit states on adjacent sites `(k, k+1)` via a SWAP gate.
    fn swap_adjacent(&mut self, k: usize) -> Result<(), MpsError> {
        let g = GateInstance::new(Gate::Swap, vec![k as u32, (k + 1) as u32]);
        let u = crate::gate::matrix_4x4(&g)?;
        self.apply_2q_adjacent(k as u32, (k + 1) as u32, &u)
    }

    /// Multiply matrix `r` into site `i`'s LEFT bond:
    /// A'[l',p,r2] = Σ_l r[l',l] · A[l,p,r2].
    fn absorb_into_left(&mut self, i: usize, r: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_left = r.nrows();
        let mut out = Site::zeros(new_left, site.right);
        // Explicit index arithmetic for clarity of the bond contraction.
        #[allow(clippy::needless_range_loop)]
        for lp in 0..new_left {
            for p in 0..2 {
                for r2 in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for l in 0..site.left {
                        acc += r[(lp, l)] * site.get(l, p, r2);
                    }
                    *out.get_mut(lp, p, r2) = acc;
                }
            }
        }
        self.sites[i] = out;
    }

    /// Multiply matrix `l` into site `i`'s RIGHT bond:
    /// A'[l2,p,r'] = Σ_r A[l2,p,r] · l[r,r'].
    fn absorb_into_right(&mut self, i: usize, l: &DMatrix<Complex>) {
        let site = &self.sites[i];
        let new_right = l.ncols();
        let mut out = Site::zeros(site.left, new_right);
        // Explicit index arithmetic for clarity of the bond contraction.
        #[allow(clippy::needless_range_loop)]
        for l2 in 0..site.left {
            for p in 0..2 {
                for rp in 0..new_right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for r in 0..site.right {
                        acc += site.get(l2, p, r) * l[(r, rp)];
                    }
                    *out.get_mut(l2, p, rp) = acc;
                }
            }
        }
        self.sites[i] = out;
    }

    /// Shift center right from `i` to `i+1` using thin QR on the grouped-left
    /// matrix. Site `i` becomes left-canonical; the R factor is absorbed into
    /// site `i+1`'s left bond.
    fn move_center_right(&mut self) {
        let i = self.center;
        let m = self.sites[i].to_group_left(); // (left*2) × right
        let (q, r) = thin_qr(&m); // q:(left*2)×k, r:k×right
        let k = q.ncols();
        let left = self.sites[i].left;
        self.sites[i] = Site::from_group_left(&q, left, k);
        self.absorb_into_left(i + 1, &r);
        self.center += 1;
    }

    /// Shift center left from `i` to `i-1` using thin QR on the adjoint of the
    /// grouped-right matrix (LQ decomposition). Site `i` becomes right-canonical;
    /// the Rᴴ factor is absorbed into site `i-1`'s right bond.
    fn move_center_left(&mut self) {
        let i = self.center;
        let m = self.sites[i].to_group_right(); // left × (2*right)
        let mh = m.adjoint(); // (2*right) × left
        let (q, r) = thin_qr(&mh); // q:(2*right)×k, r:k×left
        let k = q.ncols();
        let right = self.sites[i].right;
        let site_mat = q.adjoint(); // k × (2*right) — right-canonical
        self.sites[i] = Site::from_group_right(&site_mat, k, right);
        let r_into = r.adjoint(); // left × k — absorbed into left neighbor's right bond
        self.absorb_into_right(i - 1, &r_into);
        self.center -= 1;
    }

    /// ⟨self|other⟩ via a left-to-right transfer sweep. Both MPS must have the
    /// same qubit count and physical dim 2.
    fn overlap(&self, other: &MpsState) -> Complex {
        // E: bra_bond × ket_bond, start as the 1×1 identity [1].
        let mut e = DMatrix::<Complex>::from_element(1, 1, Complex::new(1.0, 0.0));
        for i in 0..self.sites.len() {
            let bra = &self.sites[i];
            let ket = &other.sites[i];
            // E_new[rb, rk] = Σ_p Σ_{lb,lk} conj(bra[lb,p,rb]) · E[lb,lk] · ket[lk,p,rk]
            let mut e_new = DMatrix::<Complex>::zeros(bra.right, ket.right);
            for p in 0..2 {
                // tmp[lb, rk] = Σ_lk E[lb,lk] · ket[lk,p,rk]
                let mut tmp = DMatrix::<Complex>::zeros(bra.left, ket.right);
                // Explicit index loops — multi-index transfer contraction is clearest
                // expressed this way; no meaningful iterator abstraction available.
                #[allow(clippy::needless_range_loop)]
                for lb in 0..bra.left {
                    for rk in 0..ket.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for lk in 0..ket.left {
                            acc += e[(lb, lk)] * ket.get(lk, p, rk);
                        }
                        tmp[(lb, rk)] = acc;
                    }
                }
                // E_new[rb, rk] += Σ_lb conj(bra[lb,p,rb]) · tmp[lb,rk]
                #[allow(clippy::needless_range_loop)]
                for rb in 0..bra.right {
                    for rk in 0..ket.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for lb in 0..bra.left {
                            acc += bra.get(lb, p, rb).conj() * tmp[(lb, rk)];
                        }
                        e_new[(rb, rk)] += acc;
                    }
                }
            }
            e = e_new;
        }
        e[(0, 0)]
    }

    /// ⟨ψ|P|ψ⟩ for a Pauli string. Returns `coefficient · Re⟨ψ|Pψ⟩`.
    ///
    /// The expectation value of a Hermitian observable is real; the imaginary
    /// part is discarded (it vanishes exactly for a normalised state and an
    /// exact Pauli string, and is O(truncation_error) for a compressed MPS).
    pub(crate) fn expectation(&self, p: &PauliString) -> Result<f64, MpsError> {
        let n = self.sites.len() as u32;
        // Validate qubit indices before cloning.
        for (q, _) in &p.terms {
            if *q >= n {
                return Err(MpsError::QubitOutOfRange {
                    qubit: *q,
                    num_qubits: n,
                });
            }
        }
        let mut pp = self.clone();
        for (q, pauli) in &p.terms {
            // Pauli::I has been stripped by PauliString::new; guard anyway.
            if let aleph_core::Pauli::I = pauli {
                continue;
            }
            let m = (*pauli).matrix();
            pp.apply_1q(*q as usize, &m);
        }
        let ov = self.overlap(&pp);
        Ok(p.coefficient * ov.re)
    }

    /// Move the orthogonality center to `target` by stepping one site at a time.
    pub(crate) fn move_center_to(&mut self, target: usize) {
        while self.center < target {
            self.move_center_right();
        }
        while self.center > target {
            self.move_center_left();
        }
    }

    /// Contract the whole chain into a dense `2^n` amplitude vector.
    /// TEST/SMALL-n ONLY (allocates 2^n). Amplitude index uses the ADR-0004
    /// convention: qubit `q` (== site `q`) occupies bit `q`.
    pub fn dense_statevector(&self) -> Vec<Complex> {
        let n = self.sites.len();
        // amps is laid out as [basis_prefix * left_dim + l]:
        //   basis_prefix is the partial basis index accumulated so far (bits 0..q-1),
        //   l is the left-bond index of the current site.
        //
        // We start with a single virtual "left bond = 1" amplitude of value 1.
        let mut amps: Vec<Complex> = vec![Complex::new(1.0, 0.0)]; // left bond of site 0 = 1
        let mut left_dim = 1usize;

        for (q, site) in self.sites.iter().enumerate() {
            debug_assert_eq!(site.left, left_dim);
            let prefix_count = amps.len() / left_dim;
            // next is laid out as [new_prefix * site.right + r],
            // where new_prefix = old_prefix | (p << q).
            let mut next = vec![Complex::new(0.0, 0.0); prefix_count * 2 * site.right];

            // Allow explicit index arithmetic — the multi-index contraction is
            // clearer expressed as nested loops than as iterator gymnastics.
            #[allow(clippy::needless_range_loop)]
            for prefix in 0..prefix_count {
                for p in 0..2usize {
                    // Bit q of the basis index is the physical index p of site q.
                    let new_prefix = prefix | (p << q);
                    for r in 0..site.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for l in 0..left_dim {
                            acc += amps[prefix * left_dim + l] * site.get(l, p, r);
                        }
                        next[new_prefix * site.right + r] += acc;
                    }
                }
            }
            amps = next;
            left_dim = site.right;
        }

        debug_assert_eq!(left_dim, 1, "right bond of the last site must be 1");
        // At this point amps[i] == amplitude of basis state i (all n bits set).
        let _ = n; // used only for the initial capacity reasoning; length is 2^n
        amps
    }

    /// Perfect sampling (Ferris–Vidal 2012). Does not mutate `self`.
    /// Each shot packs qubit `q` into bit `q`.
    ///
    /// Canonicalize a working clone to right-canonical (center=0) so the right
    /// environment is the identity at every site during the left→right sweep.
    pub(crate) fn sample<R: Rng>(&self, shots: u32, rng: &mut R) -> Vec<u64> {
        let n = self.sites.len();
        // Right-canonical clone (center = 0): right environment is identity at
        // every site during the left→right sweep.
        let mut work = self.clone();
        work.move_center_to(0);
        let mut out = Vec::with_capacity(shots as usize);
        for _ in 0..shots {
            let mut bnd = vec![Complex::new(1.0, 0.0)]; // left bond of site 0 is 1
            let mut bits = 0u64;
            // Explicit range loops — multi-index tensor contraction has no cleaner iterator form.
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let site = &work.sites[i];
                let mut w = [
                    vec![Complex::new(0.0, 0.0); site.right],
                    vec![Complex::new(0.0, 0.0); site.right],
                ];
                for b in 0..2 {
                    for r in 0..site.right {
                        let mut acc = Complex::new(0.0, 0.0);
                        for l in 0..site.left {
                            acc += bnd[l] * site.get(l, b, r);
                        }
                        w[b][r] = acc;
                    }
                }
                let p0: f64 = w[0].iter().map(|c| c.norm_sqr()).sum();
                let p1: f64 = w[1].iter().map(|c| c.norm_sqr()).sum();
                let total = p0 + p1;
                // outcome=true (|1⟩) with probability p1/total
                let outcome = rng.gen::<f64>() * total >= p0;
                let b = if outcome { 1usize } else { 0usize };
                if outcome {
                    bits |= 1u64 << i;
                }
                let pk = if outcome { p1 } else { p0 };
                let scale = if pk > 0.0 { (1.0 / pk).sqrt() } else { 0.0 };
                bnd = w[b].iter().map(|c| *c * Complex::new(scale, 0.0)).collect();
            }
            out.push(bits);
        }
        out
    }

    /// Exact joint marginal over `qubits` (length 2^k). Matches the SV backend
    /// contract: empty → [1.0]; output bit `pos` corresponds to `qubits[pos]`.
    ///
    /// Uses a doubled transfer-matrix sweep: each environment tracks both a bra
    /// and a ket copy of the MPS tensor, accumulating `bra_bond × ket_bond`
    /// matrices. At sites in `qubits` the environment branches into two (p=0 and
    /// p=1); at all other sites it contracts over both physical indices.
    pub(crate) fn probabilities(&self, qubits: &[u32]) -> Result<Vec<f64>, MpsError> {
        let n = self.sites.len();
        if qubits.is_empty() {
            return Ok(vec![1.0]);
        }
        if qubits.len() > MAX_PROB_QUBITS {
            return Err(MpsError::UnsupportedGate {
                kind: "probabilities(subset too large)",
            });
        }
        // Map site index → output bit position (None if site not in subset).
        let mut out_bit_for_site: Vec<Option<usize>> = vec![None; n];
        for (pos, &q) in qubits.iter().enumerate() {
            if (q as usize) >= n {
                return Err(MpsError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n as u32,
                });
            }
            out_bit_for_site[q as usize] = Some(pos);
        }

        // contract_p: advance the transfer matrix for physical index `p`.
        // E_new[rb, rk] = Σ_{lb,lk,p} conj(A[lb,p,rb]) · E[lb,lk] · A[lk,p,rk]
        // (for a single p here; caller sums over p for non-measured sites).
        let contract_p = |site: &Site, e: &DMatrix<Complex>, p: usize| -> DMatrix<Complex> {
            // tmp[lb, rk] = Σ_lk E[lb,lk] · A[lk,p,rk]
            let mut tmp = DMatrix::<Complex>::zeros(site.left, site.right);
            // Explicit index loops — multi-index transfer contraction is clearest
            // expressed this way; no meaningful iterator abstraction available.
            #[allow(clippy::needless_range_loop)]
            for lb in 0..site.left {
                for rk in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lk in 0..site.left {
                        acc += e[(lb, lk)] * site.get(lk, p, rk);
                    }
                    tmp[(lb, rk)] = acc;
                }
            }
            // e_new[rb, rk] = Σ_lb conj(A[lb,p,rb]) · tmp[lb,rk]
            let mut e_new = DMatrix::<Complex>::zeros(site.right, site.right);
            #[allow(clippy::needless_range_loop)]
            for rb in 0..site.right {
                for rk in 0..site.right {
                    let mut acc = Complex::new(0.0, 0.0);
                    for lb in 0..site.left {
                        acc += site.get(lb, p, rb).conj() * tmp[(lb, rk)];
                    }
                    e_new[(rb, rk)] += acc;
                }
            }
            e_new
        };

        // envs: list of (output_index_so_far, transfer_matrix).
        // Starts as a single 1×1 identity environment.
        let mut envs: Vec<(usize, DMatrix<Complex>)> =
            vec![(0usize, DMatrix::from_element(1, 1, Complex::new(1.0, 0.0)))];

        for (i, out_bit) in out_bit_for_site.iter().enumerate() {
            let site = &self.sites[i];
            match out_bit {
                None => {
                    // Traced-out site: contract over both physical indices.
                    for (_, e) in envs.iter_mut() {
                        *e = &contract_p(site, e, 0) + &contract_p(site, e, 1);
                    }
                }
                Some(pos) => {
                    // Measured site: branch into p=0 (bit=0) and p=1 (bit=1<<pos).
                    let mut next = Vec::with_capacity(envs.len() * 2);
                    for (idx, e) in &envs {
                        next.push((*idx, contract_p(site, e, 0)));
                        next.push((*idx | (1 << pos), contract_p(site, e, 1)));
                    }
                    envs = next;
                }
            }
        }

        let dim = 1usize << qubits.len();
        let mut out = vec![0.0; dim];
        for (idx, e) in envs {
            debug_assert_eq!((e.nrows(), e.ncols()), (1, 1));
            out[idx] = e[(0, 0)].re;
        }
        Ok(out)
    }

    /// Measure qubit `q` in the Z basis, collapsing the state. Returns the bit.
    ///
    /// Moves the orthogonality center to `q` so that the environment is trivial
    /// and p(b) = Σ_{l,r} |A[l,b,r]|² is the exact single-qubit marginal.
    /// After the measurement, the center stays at `q`.
    pub(crate) fn measure<R: Rng>(&mut self, q: usize, rng: &mut R) -> Result<bool, MpsError> {
        let n = self.sites.len();
        if q >= n {
            return Err(MpsError::QubitOutOfRange {
                qubit: q as u32,
                num_qubits: n as u32,
            });
        }
        self.move_center_to(q);
        let site = &self.sites[q];
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        // Explicit range loops — multi-index tensor access has no cleaner iterator form.
        #[allow(clippy::needless_range_loop)]
        for l in 0..site.left {
            for r in 0..site.right {
                p0 += site.get(l, 0, r).norm_sqr();
                p1 += site.get(l, 1, r).norm_sqr();
            }
        }
        let total = p0 + p1;
        if total <= 0.0 {
            return Err(MpsError::DegenerateMeasurement {
                qubit: q as u32,
                probability: total,
            });
        }
        let p0n = p0 / total;
        // Sample: outcome=true (|1⟩) with probability p1/total.
        let outcome = rng.gen::<f64>() >= p0n;
        let keep = if outcome { 1usize } else { 0usize };
        let pk = if outcome { p1 } else { p0 };
        // Rescale so the post-collapse MPS remains unit-norm.
        let scale = (total / pk).sqrt();
        let site = &mut self.sites[q];
        let drop = 1 - keep;
        // Explicit range loops — multi-index tensor mutation has no cleaner iterator form.
        #[allow(clippy::needless_range_loop)]
        for l in 0..site.left {
            for r in 0..site.right {
                *site.get_mut(l, drop, r) = Complex::new(0.0, 0.0);
                let v = site.get(l, keep, r);
                *site.get_mut(l, keep, r) = v * Complex::new(scale, 0.0);
            }
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Site;
    use crate::MpsError;
    use aleph_core::{Gate, GateInstance};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use smallvec::smallvec;

    fn norm_sq(v: &[Complex]) -> f64 {
        v.iter().map(|c| c.norm_sqr()).sum()
    }

    /// Left-canonical check: Σ_{l,p} conj(A[l,p,r1]) A[l,p,r2] == δ(r1,r2).
    fn is_left_canonical(site: &Site) -> bool {
        for r1 in 0..site.right {
            for r2 in 0..site.right {
                let mut acc = aleph_core::Complex::new(0.0, 0.0);
                for l in 0..site.left {
                    for p in 0..2 {
                        acc += site.get(l, p, r1).conj() * site.get(l, p, r2);
                    }
                }
                let expect = if r1 == r2 { 1.0 } else { 0.0 };
                if (acc.re - expect).abs() > 1e-9 || acc.im.abs() > 1e-9 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn move_center_right_makes_left_canonical_and_preserves_state() {
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        s.apply_1q(1, &h);
        let before = s.dense_statevector();
        s.move_center_to(2);
        assert_eq!(s.center, 2);
        assert!(is_left_canonical(&s.sites[0]));
        assert!(is_left_canonical(&s.sites[1]));
        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(
                (a - b).norm() < 1e-9,
                "state changed under canonicalization"
            );
        }
    }

    #[test]
    fn move_center_left_preserves_state() {
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        s.apply_1q(1, &h);
        s.apply_1q(2, &h);
        s.move_center_to(2);
        let before = s.dense_statevector();
        s.move_center_to(0);
        assert_eq!(s.center, 0);
        let after = s.dense_statevector();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).norm() < 1e-9, "state changed moving center left");
        }
    }

    #[test]
    fn x_on_zero_is_one() {
        let mut s = MpsState::new(1, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        let v = s.dense_statevector();
        assert!(v[0].norm() < 1e-12);
        assert!((v[1].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn h_on_zero_is_plus() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-12); // |00>
        assert!((v[1].re - inv).abs() < 1e-12); // |01> (q0=1)
        assert!(v[2].norm() < 1e-12);
        assert!(v[3].norm() < 1e-12);
    }

    #[test]
    fn ket0_dense_is_e0() {
        let s = MpsState::new(3, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 8);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!((norm_sq(&v) - 1.0).abs() < 1e-12);
        for amp in &v[1..] {
            assert!(amp.norm() < 1e-12);
        }
    }

    #[test]
    fn single_qubit_dense() {
        let s = MpsState::new(1, 64);
        let v = s.dense_statevector();
        assert_eq!(v.len(), 2);
        assert!((v[0].re - 1.0).abs() < 1e-12);
        assert!(v[1].norm() < 1e-12);
    }

    #[test]
    fn bell_via_h_cnot() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-10); // |00>
        assert!(v[1].norm() < 1e-10);
        assert!(v[2].norm() < 1e-10);
        assert!((v[3].re - inv).abs() < 1e-10); // |11>
        assert!(s.truncation_error() < 1e-12);
    }

    #[test]
    fn ghz_via_nonadjacent_cnots() {
        // |0000>; H(0); CNOT(0,2); CNOT(0,3) → (|0000> + |1101>)/√2 (q0=q2=q3).
        let mut s = MpsState::new(4, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for tgt in [2u32, 3u32] {
            let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, tgt]);
            let cnot = crate::gate::matrix_4x4(&gi).unwrap();
            s.apply_2q(&gi, &cnot).unwrap();
        }
        let v = s.dense_statevector();
        let inv = 1.0 / 2f64.sqrt();
        assert!((v[0].re - inv).abs() < 1e-10, "|0000>");
        assert!((v[0b1101].re - inv).abs() < 1e-10, "|1101>");
        for (k, amp) in v.iter().enumerate() {
            if k != 0 && k != 0b1101 {
                assert!(amp.norm() < 1e-10, "idx {k}");
            }
        }
    }

    #[test]
    fn swap_via_nonadjacent() {
        // X(0); SWAP(0,3) → q3=1, q0=0 → |1000> (bit3) = index 8.
        let mut s = MpsState::new(4, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        let gi = GateInstance::new(Gate::Swap, smallvec![0u32, 3u32]);
        let sw = crate::gate::matrix_4x4(&gi).unwrap();
        s.apply_2q(&gi, &sw).unwrap();
        let v = s.dense_statevector();
        assert!((v[0b1000].re - 1.0).abs() < 1e-10, "expected |1000> (q3=1)");
    }

    #[test]
    fn cnot_reversed_qubit_order() {
        // CNOT qubits [1,0]: control=q1, target=q0. Prep q1=|1>, then CNOT → |11>.
        let mut s = MpsState::new(2, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![1u32])).unwrap();
        s.apply_1q(1, &x);
        let g = GateInstance::new(Gate::Cnot, smallvec![1u32, 0u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        let v = s.dense_statevector();
        assert!((v[3].re - 1.0).abs() < 1e-10); // |11> index 3
    }

    fn bell_state(max_bond: usize) -> MpsState {
        let mut s = MpsState::new(2, max_bond);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let g = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        let cnot = crate::gate::matrix_4x4(&g).unwrap();
        s.apply_2q(&g, &cnot).unwrap();
        s
    }

    #[test]
    fn expectation_bell() {
        use aleph_core::{Pauli, PauliString};
        let s = bell_state(64);
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zi = PauliString::new(1.0, vec![(0, Pauli::Z)]).unwrap();
        assert!((s.expectation(&zz).unwrap() - 1.0).abs() < 1e-10);
        assert!((s.expectation(&xx).unwrap() - 1.0).abs() < 1e-10);
        assert!(s.expectation(&zi).unwrap().abs() < 1e-10);
        let half = PauliString::new(0.5, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        assert!((s.expectation(&half).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn expectation_oor() {
        use aleph_core::{Pauli, PauliString};
        let s = bell_state(64);
        let p = PauliString::new(1.0, vec![(5, Pauli::Z)]).unwrap();
        assert!(matches!(
            s.expectation(&p),
            Err(MpsError::QubitOutOfRange { qubit: 5, .. })
        ));
    }

    #[test]
    fn measure_zero_is_zero() {
        let mut s = MpsState::new(1, 64);
        let mut rng = StdRng::seed_from_u64(1);
        assert!(!s.measure(0, &mut rng).unwrap());
    }

    #[test]
    fn measure_ghz_correlated() {
        // GHZ-3: H(0), CNOT(0,1), CNOT(1,2). Measuring all qubits → all equal.
        let mut s = MpsState::new(3, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..2u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        let mut rng = StdRng::seed_from_u64(7);
        let b0 = s.measure(0, &mut rng).unwrap();
        let b1 = s.measure(1, &mut rng).unwrap();
        let b2 = s.measure(2, &mut rng).unwrap();
        assert_eq!(b0, b1);
        assert_eq!(b1, b2);
    }

    fn ghz(n: usize) -> MpsState {
        let mut s = MpsState::new(n, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..(n as u32 - 1) {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        s
    }

    #[test]
    fn sample_ghz_all_equal() {
        let s = ghz(4);
        let mut rng = StdRng::seed_from_u64(3);
        let shots = s.sample(500, &mut rng);
        for sh in shots {
            assert!(sh == 0b0000 || sh == 0b1111, "bad GHZ shot {sh:04b}");
        }
    }

    #[test]
    fn probabilities_plus_state() {
        let mut s = MpsState::new(2, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        let p = s.probabilities(&[0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-10);
        assert!((p[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn probabilities_bell_joint() {
        let s = ghz(2);
        let p = s.probabilities(&[0, 1]).unwrap();
        assert!((p[0b00] - 0.5).abs() < 1e-10);
        assert!(p[0b01].abs() < 1e-10);
        assert!(p[0b10].abs() < 1e-10);
        assert!((p[0b11] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn probabilities_empty_subset_is_one() {
        let s = ghz(2);
        assert_eq!(s.probabilities(&[]).unwrap(), vec![1.0]);
    }

    #[test]
    fn sample_matches_probabilities() {
        // GHZ-3 sampling distribution must match probabilities over all 3 qubits.
        let s = ghz(3);
        let mut rng = StdRng::seed_from_u64(11);
        let shots = s.sample(20000, &mut rng);
        let mut counts = [0u32; 8];
        for sh in &shots {
            counts[*sh as usize] += 1;
        }
        let probs = s.probabilities(&[0, 1, 2]).unwrap();
        for idx in 0..8 {
            let emp = counts[idx] as f64 / 20000.0;
            assert!(
                (emp - probs[idx]).abs() < 0.02,
                "idx {idx}: emp {emp} vs {}",
                probs[idx]
            );
        }
    }

    #[test]
    fn max_bond_reached_tracks_growth() {
        let mut s = MpsState::new(4, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..3u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        assert!(s.max_bond_reached() >= 2, "got {}", s.max_bond_reached());
    }

    #[test]
    fn error_bounded_policy_respects_bound() {
        use crate::tensor::TruncationPolicy;
        let mut s = MpsState::with_policy(
            4,
            TruncationPolicy::ErrorBounded {
                epsilon: 0.3,
                max_bond: 64,
            },
        );
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..3u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        // GHZ Schmidt values are 1/√2 each (squared 0.5); dropping any discards
        // 0.5 > 0.3, so nothing is dropped → error stays ~0, bond stays 2.
        assert!(s.truncation_error() <= 0.3 + 1e-12);
    }
}
