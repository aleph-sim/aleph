//! Mixed-canonical MPS state: init, dense reconstruction, canonicalization,
//! gate application with lazy SWAP permutation routing (P3-09), expectation,
//! measurement, sampling, probabilities.

use crate::tensor::{Site, TruncationPolicy};
use crate::MpsError;
use aleph_core::{Complex, Gate, GateInstance, PauliString};
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, StackReq};
use faer::linalg::matmul::matmul;
use faer::Accum;
use faer::Mat;
use nalgebra::DMatrix;
use rand::Rng;

/// Largest subset size `probabilities` will materialize (output is 2^k).
pub(crate) const MAX_PROB_QUBITS: usize = 20;

/// Mixed-canonical MPS. Sites left of `center` are left-canonical, sites right
/// are right-canonical; the center site carries the norm.
///
/// Sites hold logical qubits per the `qubit_of_site`/`site_of_qubit`
/// permutation (lazy SWAP routing, P3-09); `site == qubit` only until the
/// first long-range 2q gate.
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) policy: TruncationPolicy,
    pub(crate) trunc_error: f64,
    pub(crate) max_bond_seen: usize,
    /// qubit_of_site[s] = the logical qubit currently stored at site s (P3-09).
    pub(crate) qubit_of_site: Vec<u32>,
    /// site_of_qubit[q] = the site currently holding logical qubit q (P3-09).
    pub(crate) site_of_qubit: Vec<usize>,
    /// Physical nearest-neighbor SWAPs applied so far (lazy-router evidence).
    pub(crate) swaps_applied: u64,
    /// User-level `Gate::Swap`s discharged as O(1) permutation relabels (P3-12),
    /// touching no tensors. Counted separately from `swaps_applied`, which stays
    /// reserved for *physical* router SWAPs (gemm + truncated SVD).
    pub(crate) relabels: u64,
    /// Reusable hot-path workspace (P3-14). Not part of the logical state; see
    /// `Scratch`'s clone-as-empty.
    scratch: Scratch,
    /// Test-only override forcing every faer op down one `Par` regardless of
    /// the size threshold — lets the Par-invariance oracle compare `Seq` vs
    /// `rayon` as plain arguments instead of toggling faer's process global
    /// (P3-13). cfg(test)-gated so production code cannot bypass the
    /// size-threshold policy.
    #[cfg(test)]
    pub(crate) par_override: Option<faer::Par>,
}

/// Reusable per-state workspace for the 2q hot path (P3-14). Each `Mat` grows
/// monotonically to the largest operand seen; ops take `submatrix_mut` views at
/// their exact shape. `mem` is the faer scratch shared sequentially across the
/// SVD/QR ops within one gate.
///
/// NOTE: peak scratch memory rises vs the alloc-per-gate code (≈100–150 MB at
/// χ=512); buffers used at disjoint times (absorbed↔theta) could be unified —
/// documented follow-up, not done in v1 for clarity.
struct Scratch {
    theta: Mat<Complex>,
    theta2: Mat<Complex>,
    svd_u: Mat<Complex>,
    svd_v: Mat<Complex>,
    qr_in: Mat<Complex>,
    q_coeff: Mat<Complex>,
    thin_q: Mat<Complex>,
    thin_r: Mat<Complex>,
    absorbed: Mat<Complex>,
    mem: MemBuffer,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            theta: Mat::new(),
            theta2: Mat::new(),
            svd_u: Mat::new(),
            svd_v: Mat::new(),
            qr_in: Mat::new(),
            q_coeff: Mat::new(),
            thin_q: Mat::new(),
            thin_r: Mat::new(),
            absorbed: Mat::new(),
            mem: MemBuffer::new(StackReq::new::<Complex>(0)),
        }
    }
}

// Cloning a state must NOT copy transient workspace: scratch holds no semantic
// state (always written before read, regrown on demand), so a clone starts
// empty. This keeps `#[derive(Clone)]` on MpsState cheap and correct for the
// expectation()/sampling clone paths.
impl Clone for Scratch {
    fn clone(&self) -> Self {
        Scratch::default()
    }
}

impl std::fmt::Debug for Scratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratch")
            .field("theta", &self.theta.shape())
            .field("theta2", &self.theta2.shape())
            .finish_non_exhaustive()
    }
}

impl Scratch {
    /// Ensure `buf` is at least `rows × cols`, regrowing monotonically (keeps the
    /// larger of each dim so it never shrinks below a size it already serves).
    fn grow(buf: &mut Mat<Complex>, rows: usize, cols: usize) {
        if buf.nrows() < rows || buf.ncols() < cols {
            let nr = buf.nrows().max(rows);
            let nc = buf.ncols().max(cols);
            *buf = Mat::zeros(nr, nc);
        }
    }

    /// Grow `buf` to cover `rows × cols` and return that top-left submatrix view.
    fn view_mut(buf: &mut Mat<Complex>, rows: usize, cols: usize) -> faer::MatMut<'_, Complex> {
        Self::grow(buf, rows, cols);
        buf.as_mut().submatrix_mut(0, 0, rows, cols)
    }
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
            qubit_of_site: (0..n as u32).collect(),
            site_of_qubit: (0..n).collect(),
            swaps_applied: 0,
            relabels: 0,
            scratch: Scratch::default(),
            #[cfg(test)]
            par_override: None,
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

    /// Number of physical nearest-neighbor SWAP gates applied by the lazy
    /// permutation router so far (P3-09).
    pub fn swaps_applied(&self) -> u64 {
        self.swaps_applied
    }

    /// Number of user-level `Gate::Swap`s discharged as O(1) permutation
    /// relabels — pure map updates with zero tensor work, bond growth, or
    /// truncation error (P3-12).
    pub fn relabels(&self) -> u64 {
        self.relabels
    }

    /// Per-operation parallelism for a `rows × cols` faer operand: the
    /// size-threshold policy, unless a test override pins it.
    ///
    /// gemm call sites key the threshold on the OUTPUT m×n (ignoring the
    /// contraction dim k), consistent with how the P3-09 sweep counted operand
    /// elements — a future re-tune on gemm cost m·n·k should revisit this.
    fn choose_par(&self, rows: usize, cols: usize) -> faer::Par {
        #[cfg(test)]
        if let Some(par) = self.par_override {
            return par;
        }
        crate::linalg::par_for(rows, cols)
    }

    /// Apply a 1q unitary to logical qubit `q` (routed to its current site).
    /// Preserves canonical form, so neither the center nor any SVD is touched.
    pub(crate) fn apply_1q(&mut self, q: usize, u: &[[Complex; 2]; 2]) {
        let site = &mut self.sites[self.site_of_qubit[q]];
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
    /// (`g.qubits[0]`=MSB, ADR-0004). The qubits' current sites come from the
    /// lazy permutation: non-adjacent sites are brought together by moving the
    /// qubit at the higher site down with nearest-neighbor SWAPs, and the
    /// permutation is left in place afterwards — no swap-back (P3-09). Reads
    /// route through the permutation, so `site == qubit` is no longer an
    /// invariant. A violated routing invariant (e.g. duplicate qubits reaching
    /// the 2q path in a release build) surfaces as
    /// `MpsError::NonNearestNeighbor` from [`Self::apply_2q_adjacent`] instead
    /// of silently corrupting the state.
    pub(crate) fn apply_2q(
        &mut self,
        g: &GateInstance,
        u: &[[Complex; 4]; 4],
    ) -> Result<(), MpsError> {
        let qa = g.qubits[0] as usize;
        let qb = g.qubits[1] as usize;
        let sa = self.site_of_qubit[qa];
        let sb = self.site_of_qubit[qb];
        if sa.abs_diff(sb) != 1 {
            // Ladder: walk the occupant of the higher site down to lo+1.
            // Site `lo` is untouched, so the pair ends adjacent.
            let lo = sa.min(sb);
            let hi = sa.max(sb);
            for k in (lo + 1..hi).rev() {
                self.swap_adjacent(k)?;
            }
        }
        // Re-resolve sites: the ladder moved one of the qubits.
        self.apply_2q_adjacent(self.site_of_qubit[qa], self.site_of_qubit[qb], u)
    }

    /// Apply a 2q unitary `u` to the adjacent sites `(s_msb, s_lsb)`, where
    /// `s_msb` is the site whose physical index forms the most-significant bit
    /// of the 4×4 matrix row/column index (ADR-0004) and `s_lsb` the
    /// least-significant.
    ///
    /// Caller must ensure `s_msb.abs_diff(s_lsb) == 1`; a violation is a
    /// router-invariant bug and is rejected in ALL build profiles (a
    /// `debug_assert` alone would let release builds silently apply a
    /// non-unitary contraction — verified empirically with a duplicate-qubit
    /// CNOT, whose corrupted state even re-normalizes to 1).
    fn apply_2q_adjacent(
        &mut self,
        s_msb: usize,
        s_lsb: usize,
        u: &[[Complex; 4]; 4],
    ) -> Result<(), MpsError> {
        if s_msb.abs_diff(s_lsb) != 1 {
            return Err(MpsError::NonNearestNeighbor {
                a: s_msb as u32,
                b: s_lsb as u32,
            });
        }
        let i = s_msb.min(s_lsb);
        let j = i + 1;

        // Move the orthogonality center to site i so that the two-site
        // contraction preserves normalization after re-factorization.
        self.move_center_to(i);

        let li = self.sites[i].left;
        let ri = self.sites[j].right;
        let par = self.choose_par(li * 2, 2 * ri);
        let rows = li * 2;
        let cols = 2 * ri;
        let size = rows.min(cols);

        // Θ as a (li·2) × (2·ri) matrix: grouped-left × grouped-right, one gemm
        // into the pooled buffer. No memset — Accum::Replace overwrites all
        // entries (the pooled buffer may hold stale data outside the submatrix,
        // but only the rows×cols submatrix is read downstream).
        {
            let theta = Scratch::view_mut(&mut self.scratch.theta, rows, cols);
            matmul(
                theta,
                Accum::Replace,
                self.sites[i].group_left_view(),
                self.sites[j].group_right_view(),
                Complex::new(1.0, 0.0),
                par,
            );
        }

        // Helper: physical indices → 2q matrix row/col index (s_msb=MSB).
        let out = |phys_i: usize, phys_j: usize| -> usize {
            let bit_msb = if s_msb == i { phys_i } else { phys_j };
            let bit_lsb = if s_lsb == i { phys_i } else { phys_j };
            (bit_msb << 1) | bit_lsb
        };

        // Θ' = U·Θ. theta2 is += accumulated → zero it first. theta and theta2
        // are distinct Scratch fields (disjoint borrows).
        {
            let mut theta2 = Scratch::view_mut(&mut self.scratch.theta2, rows, cols);
            theta2.fill(Complex::new(0.0, 0.0));
            let theta = self.scratch.theta.as_ref().submatrix(0, 0, rows, cols);
            for ap in 0..2usize {
                for bp in 0..2usize {
                    let row_u = out(ap, bp);
                    for a in 0..2usize {
                        for b in 0..2usize {
                            let u_entry = u[row_u][out(a, b)];
                            if u_entry == Complex::new(0.0, 0.0) {
                                continue;
                            }
                            for r in 0..ri {
                                for l in 0..li {
                                    theta2[(l * 2 + ap, bp * ri + r)] +=
                                        u_entry * theta[(l * 2 + a, b * ri + r)];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Truncated SVD of Θ' into pooled u/v/s buffers.
        let mut s_diag = Diag::<Complex>::zeros(size);
        {
            let theta2 = self.scratch.theta2.as_ref().submatrix(0, 0, rows, cols);
            let u_out = Scratch::view_mut(&mut self.scratch.svd_u, rows, size);
            let v_out = Scratch::view_mut(&mut self.scratch.svd_v, cols, size);
            crate::linalg::svd_into(
                theta2,
                par,
                u_out,
                v_out,
                s_diag.as_mut(),
                &mut self.scratch.mem,
            )?;
        }
        let sigmas: Vec<f64> = (0..size).map(|t| s_diag.as_ref()[t].re).collect();
        let (chi, discarded, scale) = crate::tensor::svd_truncation_plan(&sigmas, &self.policy);
        self.trunc_error += discarded;
        self.max_bond_seen = self.max_bond_seen.max(chi);
        let s_kept: Vec<f64> = (0..chi).map(|t| sigmas[t] * scale).collect();

        // Site i ← left-canonical from U[:, 0..chi]  (grouped-left li·2 × chi).
        {
            let u_view = self.scratch.svd_u.as_ref().submatrix(0, 0, rows, chi);
            self.sites[i].fill_left_from(u_view, li, chi);
        }
        // Site j ← right-canonical s·Vᴴ. svd_v is (cols × size); read its first
        // chi columns. fill_right_from_scaled_conj reads V[col, t] and folds
        // conj + s_kept[t] into the grouped-right layout (chi × ri).
        {
            let v_view = self.scratch.svd_v.as_ref().submatrix(0, 0, cols, chi);
            self.sites[j].fill_right_from_scaled_conj(v_view, &s_kept, chi, ri);
        }
        self.center = j;

        Ok(())
    }

    /// Exchange which logical qubits occupy sites `s0` and `s1`, keeping the
    /// `qubit_of_site`/`site_of_qubit` bijection consistent (the single place
    /// that maintains the inverse-map invariant). Touches only the maps, never
    /// tensors — both the physical router and the P3-12 relabel layer it onto
    /// their own tensor/bookkeeping work.
    fn exchange_site_labels(&mut self, s0: usize, s1: usize) {
        let q0 = self.qubit_of_site[s0];
        let q1 = self.qubit_of_site[s1];
        self.qubit_of_site[s0] = q1;
        self.qubit_of_site[s1] = q0;
        self.site_of_qubit[q0 as usize] = s1;
        self.site_of_qubit[q1 as usize] = s0;
    }

    /// Discharge a user-level `Gate::Swap(qa, qb)` as an O(1) relabel of the
    /// lazy permutation (P3-12): exchange the sites that hold logical qubits
    /// `qa` and `qb`. No tensor is touched, so there is no bond growth and no
    /// truncation error. `qa == qb` is a no-op (rejected upstream in the backend
    /// dispatch, but handled harmlessly here for direct callers).
    ///
    /// Contrast [`Self::swap_adjacent`], which physically swaps adjacent tensor
    /// content (gemm + truncated SVD) to *preserve* the logical state during
    /// routing. Here no tensor moves, so the labels swapping *is* the SWAP.
    pub(crate) fn relabel_swap(&mut self, qa: usize, qb: usize) {
        self.exchange_site_labels(self.site_of_qubit[qa], self.site_of_qubit[qb]);
        self.relabels += 1;
    }

    /// Swap the qubit states on adjacent sites `(k, k+1)` via a SWAP gate and
    /// update the site↔qubit permutation accordingly.
    fn swap_adjacent(&mut self, k: usize) -> Result<(), MpsError> {
        let g = GateInstance::new(Gate::Swap, vec![k as u32, (k + 1) as u32]);
        let u = crate::gate::matrix_4x4(&g)?;
        self.apply_2q_adjacent(k, k + 1, &u)?;
        self.exchange_site_labels(k, k + 1);
        self.swaps_applied += 1;
        Ok(())
    }

    /// Shift center right from `i` to `i+1` using thin QR on the grouped-left
    /// view. Site `i` becomes left-canonical; the R factor is absorbed into
    /// site `i+1`'s left bond via a size-thresholded gemm.
    fn move_center_right(&mut self) {
        let i = self.center;
        let left = self.sites[i].left;
        let right = self.sites[i].right;
        let m = left * 2;
        let n = right;
        let size = m.min(n);
        let k = size;
        let next_right = self.sites[i + 1].right;
        let block_size = crate::linalg::recommended_block_size_complex(m, n);
        // choose_par values hoisted before any &mut self.scratch borrow.
        let par_qr = self.choose_par(m, n);
        let par_absorb = self.choose_par(k, 2 * next_right);

        Scratch::grow(&mut self.scratch.qr_in, m, n);
        Scratch::grow(&mut self.scratch.q_coeff, block_size, size);
        Scratch::grow(&mut self.scratch.thin_q, m, size);
        Scratch::grow(&mut self.scratch.thin_r, size, n);
        Scratch::grow(&mut self.scratch.absorbed, k, 2 * next_right);

        // Copy grouped-left view into the pooled QR workspace (vectorized faer
        // copy; shapes match m×n).
        {
            let mut qr_in = self.scratch.qr_in.as_mut().submatrix_mut(0, 0, m, n);
            qr_in.copy_from(self.sites[i].group_left_view());
        }
        // QR into pooled buffers (5 disjoint &mut fields via destructure).
        {
            let Scratch {
                qr_in,
                q_coeff,
                thin_q,
                thin_r,
                mem,
                ..
            } = &mut self.scratch;
            crate::linalg::qr_into(
                qr_in.as_mut().submatrix_mut(0, 0, m, n),
                par_qr,
                q_coeff.as_mut().submatrix_mut(0, 0, block_size, size),
                thin_q.as_mut().submatrix_mut(0, 0, m, size),
                thin_r.as_mut().submatrix_mut(0, 0, size, n),
                mem,
            );
        }
        // absorbed = R · group_right(site[i+1])  (k × 2·next_right).
        {
            let r_view = self.scratch.thin_r.as_ref().submatrix(0, 0, k, n);
            let mut absorbed =
                self.scratch
                    .absorbed
                    .as_mut()
                    .submatrix_mut(0, 0, k, 2 * next_right);
            matmul(
                absorbed.as_mut(),
                Accum::Replace,
                r_view,
                self.sites[i + 1].group_right_view(),
                Complex::new(1.0, 0.0),
                par_absorb,
            );
        }
        // Site i+1 ← right-canonical from absorbed; site i ← left-canonical from Q.
        {
            let absorbed = self
                .scratch
                .absorbed
                .as_ref()
                .submatrix(0, 0, k, 2 * next_right);
            self.sites[i + 1].fill_from_grouped_right(absorbed, k, next_right);
        }
        {
            let q_view = self.scratch.thin_q.as_ref().submatrix(0, 0, m, k);
            self.sites[i].fill_left_from(q_view, left, k);
        }
        self.center += 1;
    }

    /// Shift center left from `i` to `i-1` using thin QR on the adjoint of the
    /// grouped-right view (LQ decomposition). Site `i` becomes right-canonical;
    /// the Rᴴ factor is absorbed into site `i-1`'s right bond via a
    /// size-thresholded gemm.
    fn move_center_left(&mut self) {
        let i = self.center;
        let right = self.sites[i].right;
        let left = self.sites[i].left;
        // LQ via QR of the adjoint of the grouped-right view: (2·right) × left.
        let m = 2 * right;
        let n = left;
        let size = m.min(n);
        let k = size;
        let prev_left = self.sites[i - 1].left;
        let block_size = crate::linalg::recommended_block_size_complex(m, n);
        let par_qr = self.choose_par(m, n);
        let par_absorb = self.choose_par(prev_left * 2, k);

        Scratch::grow(&mut self.scratch.qr_in, m, n);
        Scratch::grow(&mut self.scratch.q_coeff, block_size, size);
        Scratch::grow(&mut self.scratch.thin_q, m, size);
        Scratch::grow(&mut self.scratch.thin_r, size, n);
        Scratch::grow(&mut self.scratch.absorbed, prev_left * 2, k);

        // qr_in := adjoint(group_right(site[i])): entry (r, c) = conj(gr[c, r]).
        // gr is group_right_view = left × (2·right) = n × m; .adjoint() is the
        // m×n conjugate-transpose view, conjugation folded into copy_from.
        {
            let mut qr_in = self.scratch.qr_in.as_mut().submatrix_mut(0, 0, m, n);
            qr_in.copy_from(self.sites[i].group_right_view().adjoint());
        }
        {
            let Scratch {
                qr_in,
                q_coeff,
                thin_q,
                thin_r,
                mem,
                ..
            } = &mut self.scratch;
            crate::linalg::qr_into(
                qr_in.as_mut().submatrix_mut(0, 0, m, n),
                par_qr,
                q_coeff.as_mut().submatrix_mut(0, 0, block_size, size),
                thin_q.as_mut().submatrix_mut(0, 0, m, size),
                thin_r.as_mut().submatrix_mut(0, 0, size, n),
                mem,
            );
        }
        // absorbed = group_left(site[i-1]) · Rᴴ   (prev_left·2 × k).
        {
            let r_view = self.scratch.thin_r.as_ref().submatrix(0, 0, k, n);
            let mut absorbed = self
                .scratch
                .absorbed
                .as_mut()
                .submatrix_mut(0, 0, prev_left * 2, k);
            matmul(
                absorbed.as_mut(),
                Accum::Replace,
                self.sites[i - 1].group_left_view(),
                r_view.adjoint(),
                Complex::new(1.0, 0.0),
                par_absorb,
            );
        }
        {
            let absorbed = self
                .scratch
                .absorbed
                .as_ref()
                .submatrix(0, 0, prev_left * 2, k);
            self.sites[i - 1].fill_left_from(absorbed, prev_left, k);
        }
        // Site i ← right-canonical Qᴴ: grouped-right (t,col) = conj(Q[col,t]).
        // fill_right_from_scaled_conj with sv = all-ones reads exactly that.
        {
            let q_view = self.scratch.thin_q.as_ref().submatrix(0, 0, m, k);
            let ones = vec![1.0_f64; k];
            self.sites[i].fill_right_from_scaled_conj(q_view, &ones, k, right);
        }
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
    /// convention: site `s` contributes the bit of the logical qubit it
    /// currently holds (`qubit_of_site[s]`).
    pub fn dense_statevector(&self) -> Vec<Complex> {
        let n = self.sites.len();
        // Phase 1: contract in site order, producing raw_amps indexed by site bits
        // (bit s = physical index of site s). The incremental layout only works when
        // the bits are introduced in order 0, 1, 2, …, so we always contract with
        // site-index bits here.
        //
        // amps is laid out as [basis_prefix * left_dim + l]:
        //   basis_prefix is the partial basis index accumulated so far (bits 0..s-1),
        //   l is the left-bond index of the current site.
        let mut amps: Vec<Complex> = vec![Complex::new(1.0, 0.0)]; // left bond of site 0 = 1
        let mut left_dim = 1usize;

        for (s, site) in self.sites.iter().enumerate() {
            debug_assert_eq!(site.left, left_dim);
            let prefix_count = amps.len() / left_dim;
            // next is laid out as [new_prefix * site.right + r],
            // where new_prefix = old_prefix | (p << s)  (site-index bit s).
            let mut next = vec![Complex::new(0.0, 0.0); prefix_count * 2 * site.right];

            // Allow explicit index arithmetic — the multi-index contraction is
            // clearer expressed as nested loops than as iterator gymnastics.
            #[allow(clippy::needless_range_loop)]
            for prefix in 0..prefix_count {
                for p in 0..2usize {
                    // Bit s of the raw index is the physical index p of site s.
                    let new_prefix = prefix | (p << s);
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

        // Phase 2: permute raw_amps (site-order bits) into logical-qubit order.
        // raw_amps[raw_idx] has bit s = physical index of site s.
        // The logical index has bit qubit_of_site[s] = same physical index of site s.
        // When the permutation is identity (qubit_of_site[s] == s for all s),
        // the loop below would be an element-wise copy; we skip it and return
        // `amps` directly.
        if self
            .qubit_of_site
            .iter()
            .enumerate()
            .all(|(s, &q)| q as usize == s)
        {
            // Identity permutation: raw layout already matches logical layout.
            return amps;
        }
        let dim = 1usize << n;
        let mut out = vec![Complex::new(0.0, 0.0); dim];
        // Explicit index loop — the permuted bit-shuffling has no cleaner iterator form.
        #[allow(clippy::needless_range_loop)]
        for raw_idx in 0..dim {
            // Build the logical index by mapping each site's bit to the qubit it holds.
            let mut logical_idx = 0usize;
            for s in 0..n {
                let bit = (raw_idx >> s) & 1;
                logical_idx |= bit << self.qubit_of_site[s] as usize;
            }
            out[logical_idx] = amps[raw_idx];
        }
        out
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
                    bits |= 1u64 << work.qubit_of_site[i];
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
            out_bit_for_site[self.site_of_qubit[q as usize]] = Some(pos);
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
    /// Moves the orthogonality center to the site holding `q` (`site_of_qubit[q]`)
    /// so that the environment is trivial and p(b) = Σ_{l,r} |A[l,b,r]|² is the
    /// exact single-qubit marginal.  After the measurement, the center stays at
    /// that site.
    pub(crate) fn measure<R: Rng>(&mut self, q: usize, rng: &mut R) -> Result<bool, MpsError> {
        let n = self.sites.len();
        if q >= n {
            return Err(MpsError::QubitOutOfRange {
                qubit: q as u32,
                num_qubits: n as u32,
            });
        }
        let s = self.site_of_qubit[q];
        self.move_center_to(s);
        let site = &self.sites[s];
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
        let site = &mut self.sites[s];
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
    fn swap_adjacent_updates_permutation_maps() {
        let mut s = MpsState::new(3, 64);
        assert_eq!(s.qubit_of_site, vec![0, 1, 2]);
        assert_eq!(s.site_of_qubit, vec![0, 1, 2]);
        assert_eq!(s.swaps_applied(), 0);
        s.swap_adjacent(1).unwrap();
        assert_eq!(s.qubit_of_site, vec![0, 2, 1]);
        assert_eq!(s.site_of_qubit, vec![0, 2, 1]);
        assert_eq!(s.swaps_applied(), 1);
        s.swap_adjacent(1).unwrap();
        assert_eq!(s.qubit_of_site, vec![0, 1, 2]);
        assert_eq!(s.swaps_applied(), 2);
        assert_eq!(s.site_of_qubit, vec![0, 1, 2]);
    }

    #[test]
    fn duplicate_qubit_2q_errors_in_all_profiles() {
        // The router-invariant guard must be a real error, not a debug_assert:
        // in release a duplicate-qubit gate previously produced a non-unitary
        // contraction that re-normalized to 1 — strictly silent corruption.
        let mut s = MpsState::new(2, 64);
        let mut gi = GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]);
        gi.qubits[1] = 0;
        let u = crate::gate::matrix_4x4(&gi).unwrap();
        let err = s.apply_2q(&gi, &u).unwrap_err();
        assert!(matches!(err, MpsError::NonNearestNeighbor { .. }));
        let v = s.dense_statevector();
        assert!((v[0].re - 1.0).abs() < 1e-12, "state must be untouched");
    }

    #[test]
    fn lazy_swap_counts_amortize() {
        // CNOT(0,4) on n=5: the ladder is 3 SWAPs (always-swap-back paid 6).
        let mut s = MpsState::new(5, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h); // non-trivial state so the SWAPs move real amplitude
        let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, 4u32]);
        let u = crate::gate::matrix_4x4(&gi).unwrap();
        s.apply_2q(&gi, &u).unwrap();
        assert_eq!(s.swaps_applied(), 3);
        assert_eq!(s.site_of_qubit[4], 1);
        // Qubit 4 stayed next to qubit 0 → repeating the gate costs 0 SWAPs.
        s.apply_2q(&gi, &u).unwrap();
        assert_eq!(s.swaps_applied(), 3);
    }

    #[test]
    fn user_swap_relabels_without_physical_swap() {
        // P3-12: a user-level Gate::Swap(0,4) on n=5 is a pure relabel — zero
        // physical SWAPs (no ladder, no SVD), one relabel counted, and the
        // permutation maps reflect the exchange.
        let mut s = MpsState::new(5, 64);
        s.relabel_swap(0, 4);
        assert_eq!(s.swaps_applied(), 0, "relabel must apply no physical SWAPs");
        assert_eq!(s.relabels(), 1);
        // qubit 0 now lives where qubit 4 was and vice versa.
        assert_eq!(s.site_of_qubit[0], 4);
        assert_eq!(s.site_of_qubit[4], 0);
        assert_eq!(s.qubit_of_site[0], 4);
        assert_eq!(s.qubit_of_site[4], 0);
        assert_eq!(s.trunc_error, 0.0, "relabel adds no truncation error");
        assert_eq!(s.max_bond_seen, 1, "relabel grows no bond");
        // Permutation stays a valid bijection.
        let mut seen: Vec<u32> = s.qubit_of_site.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn user_swap_relabel_is_self_inverse() {
        // Two SWAPs on the same pair restore the identity permutation.
        let mut s = MpsState::new(3, 64);
        s.relabel_swap(0, 2);
        assert_eq!(s.site_of_qubit, vec![2, 1, 0]);
        s.relabel_swap(0, 2);
        assert_eq!(s.site_of_qubit, vec![0, 1, 2]);
        assert_eq!(s.qubit_of_site, vec![0, 1, 2]);
        assert_eq!(s.relabels(), 2);
        assert_eq!(s.swaps_applied(), 0);
    }

    #[test]
    fn reads_route_through_permutation() {
        // X(0), then a raw physical swap of sites 0,1: qubit 0 (|1>) now lives
        // at site 1. Every read must still report in logical-qubit order.
        let mut s = MpsState::new(2, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        s.swap_adjacent(0).unwrap();
        // dense: qubit 0 occupies bit 0 → index 0b01.
        let v = s.dense_statevector();
        assert!((v[0b01].re - 1.0).abs() < 1e-10, "dense not routed");
        // probabilities over qubit 0: [0, 1].
        let p = s.probabilities(&[0]).unwrap();
        assert!((p[1] - 1.0).abs() < 1e-10, "probabilities not routed");
        // sample: qubit 0 packs into bit 0.
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(
            s.sample(3, &mut rng),
            vec![0b01, 0b01, 0b01],
            "sample not routed"
        );
        // apply_1q routes: a second X on qubit 0 returns it to |0>.
        s.apply_1q(0, &x);
        let v = s.dense_statevector();
        assert!((v[0b00].re - 1.0).abs() < 1e-10, "apply_1q not routed");
        // measure(0) must read site 1's data: re-flip then measure.
        s.apply_1q(0, &x);
        assert!(s.measure(0, &mut rng).unwrap(), "measure not routed");
    }

    #[test]
    fn reads_route_through_three_cycle_permutation() {
        // A 3-cycle makes qubit_of_site != site_of_qubit, so a map-direction
        // mix-up in any read path fails here (a transposition cannot catch it).
        let mut s = MpsState::new(3, 64);
        let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
        s.apply_1q(0, &x);
        s.swap_adjacent(0).unwrap();
        s.swap_adjacent(1).unwrap();
        assert_eq!(s.qubit_of_site, vec![1, 2, 0]);
        assert_eq!(s.site_of_qubit, vec![2, 0, 1]);
        // Qubit 0 is |1>; logical index 0b001.
        let v = s.dense_statevector();
        assert!(
            (v[0b001] - Complex::new(1.0, 0.0)).norm() < 1e-10,
            "dense not routed"
        );
        let p = s.probabilities(&[0]).unwrap();
        assert!((p[1] - 1.0).abs() < 1e-10, "probabilities(0) not routed");
        let p01 = s.probabilities(&[1, 0]).unwrap();
        // Output bit 0 ↔ qubit 1 (=0), bit 1 ↔ qubit 0 (=1) → index 0b10.
        assert!(
            (p01[0b10] - 1.0).abs() < 1e-10,
            "probabilities subset ordering not routed"
        );
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(
            s.sample(3, &mut rng),
            vec![0b001, 0b001, 0b001],
            "sample not routed"
        );
        // apply_1q routes to site 2; X returns qubit 0 to |0>.
        s.apply_1q(0, &x);
        let v = s.dense_statevector();
        assert!(
            (v[0] - Complex::new(1.0, 0.0)).norm() < 1e-10,
            "apply_1q not routed"
        );
        // measure(0) must read the site holding qubit 0 (site 2): flip back first.
        s.apply_1q(0, &x);
        assert!(s.measure(0, &mut rng).unwrap(), "measure not routed");
        assert!(!s.measure(1, &mut rng).unwrap(), "measure(1) not routed");
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

    proptest::proptest! {
        /// After any random long-range circuit the two maps stay mutually inverse.
        #[test]
        fn permutation_maps_stay_inverse(seq in proptest::collection::vec((0u8..5, 0u8..5, 0u8..5), 0..20)) {
            let n = 5u32;
            let mut s = MpsState::new(n as usize, 64);
            let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
            for (op, x, y) in seq {
                let a = (x as u32) % n;
                match op {
                    0 | 1 => s.apply_1q(a as usize, &h),
                    _ => {
                        let b = (y as u32) % n;
                        if a != b {
                            let gi = GateInstance::new(Gate::Cnot, smallvec![a, b]);
                            let u = crate::gate::matrix_4x4(&gi).unwrap();
                            s.apply_2q(&gi, &u).unwrap();
                        }
                    }
                }
            }
            for q in 0..n as usize {
                proptest::prop_assert_eq!(s.qubit_of_site[s.site_of_qubit[q]] as usize, q);
            }
            for site in 0..n as usize {
                proptest::prop_assert_eq!(s.site_of_qubit[s.qubit_of_site[site] as usize], site);
            }
        }
    }

    /// Same circuit under sequential and rayon-parallel faer must agree to
    /// 1e-10 (not bit-exact: parallel kernels may round differently).
    ///
    /// Replaces the former tests/sv_equivalence.rs global-toggle test
    /// (P3-09): `par_override` forces EVERY op down the chosen path
    /// regardless of the size threshold — Seq vs rayon as plain arguments,
    /// no process-global mutation, no cross-test isolation hazard (P3-13).
    ///
    /// n=10 with 10 brickwall layers grows the central bond to χ = 16
    /// (measured; only every second layer crosses the middle cut), so the
    /// rayon branch sees real multi-column SVD/QR/gemm work.
    #[cfg(feature = "parallel")]
    #[test]
    fn state_invariant_seq_vs_rayon() {
        use aleph_core::Param;
        let run = |par: faer::Par| -> Vec<Complex> {
            let n = 10usize;
            let mut s = MpsState::new(n, 128);
            s.par_override = Some(par);
            let h_of = |q: u32| {
                crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![q])).unwrap()
            };
            for q in 0..n as u32 {
                s.apply_1q(q as usize, &h_of(q));
            }
            for layer in 0..10u32 {
                let mut q = layer % 2;
                while (q as usize) + 1 < n {
                    let ry = crate::gate::matrix_2x2(&GateInstance::new(
                        Gate::Ry(Param::Concrete(0.3 + (q + layer * n as u32) as f64 * 0.11)),
                        smallvec![q],
                    ))
                    .unwrap();
                    s.apply_1q(q as usize, &ry);
                    let gi = GateInstance::new(Gate::Cnot, smallvec![q, q + 1]);
                    let u = crate::gate::matrix_4x4(&gi).unwrap();
                    s.apply_2q(&gi, &u).unwrap();
                    q += 2;
                }
            }
            // Exercise the lazy router too.
            let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, 9u32]);
            let u = crate::gate::matrix_4x4(&gi).unwrap();
            s.apply_2q(&gi, &u).unwrap();
            s.dense_statevector()
        };
        let a = run(faer::Par::Seq);
        let b = run(faer::Par::rayon(0));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).norm() < 1e-10, "parallelism changed the state");
        }
    }
}
