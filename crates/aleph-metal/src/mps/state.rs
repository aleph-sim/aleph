//! Device-resident MPS state: a chain of rank-3 site tensors in FP32, plus the
//! host dense contraction used by the oracle.

use aleph_backend::BackendError;
use aleph_core::{Complex, PauliString};

use crate::mps::readout;
use crate::{DeviceBuffer, MetalContext};

/// One rank-3 MPS site tensor, shape `(left, 2, right)`, row-major in a shared
/// (unified-memory) GPU buffer: `data[(l*2 + p)*right + r]`. FP32 to match the
/// Metal SV core; the host SVD widens to f64 for accuracy.
pub struct SiteTensor {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) buf: DeviceBuffer<Complex<f32>>,
}

impl SiteTensor {
    /// The `|0⟩` site `(1, 2, 1)`: amplitude 1 on physical 0, 0 on physical 1.
    fn ket0(ctx: &MetalContext) -> Self {
        let data = [Complex::<f32>::new(1.0, 0.0), Complex::<f32>::new(0.0, 0.0)];
        Self {
            left: 1,
            right: 1,
            buf: DeviceBuffer::from_slice(ctx, &data),
        }
    }

    /// Overwrite this site's tensor **in place**, reusing its device buffer's
    /// capacity (P5.8-02): no `new_buffer*` unless the new shape exceeds the buffer's
    /// high-water mark. Used for the per-gate two-site split and the canonical
    /// centre-move rebuilds, which previously allocated a fresh `MTLBuffer` every time.
    pub(crate) fn set_from_host(
        &mut self,
        ctx: &MetalContext,
        left: usize,
        right: usize,
        data: &[Complex<f32>],
    ) {
        debug_assert_eq!(data.len(), left * 2 * right);
        self.buf.write(ctx, data);
        self.left = left;
        self.right = right;
    }
}

/// A device-resident matrix-product state over `num_qubits` sites. Site order need
/// not equal qubit order: a **lazy permutation** (P5.8-05) tracks which logical qubit
/// lives on each site, so a non-NN 2q gate routes with a forward SWAP network that is
/// *not* unwound (≈ half the cost), and a user `Swap` is an O(1) relabel.
pub struct MetalMpsState {
    pub(crate) num_qubits: u32,
    pub(crate) sites: Vec<SiteTensor>,
    /// Orthogonality centre for the mixed-canonical form (P5.7-07): sites `<
    /// center` are left-canonical, sites `> center` right-canonical. Maintained by
    /// the gate-by-gate (`run`) path so a two-site split can apply truncation;
    /// `run_batched` leaves the state non-canonical (exact-only). A product state
    /// is trivially canonical (every bond is dimension 1), so the initial value is
    /// arbitrary — 0.
    pub(crate) center: usize,
    /// `qubit_of_site[s]` = the logical qubit whose bit physically lives on site `s`.
    pub(crate) qubit_of_site: Vec<u32>,
    /// Inverse: `site_of_qubit[q]` = the site holding qubit `q`. Both start identity.
    pub(crate) site_of_qubit: Vec<u32>,
}

impl MetalMpsState {
    /// The product state `|0…0⟩`: each site is a `|0⟩` ket; identity permutation.
    pub(crate) fn product(ctx: &MetalContext, num_qubits: u32) -> Self {
        let sites = (0..num_qubits).map(|_| SiteTensor::ket0(ctx)).collect();
        Self {
            num_qubits,
            sites,
            center: 0,
            qubit_of_site: (0..num_qubits).collect(),
            site_of_qubit: (0..num_qubits).collect(),
        }
    }

    /// After a *physical* SWAP of adjacent sites `s` and `s+1`, exchange their qubit
    /// labels (keeps the lazy permutation consistent with the moved tensors).
    pub(crate) fn swap_site_labels(&mut self, s: usize) {
        self.qubit_of_site.swap(s, s + 1);
        self.site_of_qubit[self.qubit_of_site[s] as usize] = s as u32;
        self.site_of_qubit[self.qubit_of_site[s + 1] as usize] = (s + 1) as u32;
    }

    /// A user `Swap(qa, qb)`: exchange the two qubits as a pure O(1) relabel — the
    /// physical state is identical up to which qubit is which, so no tensor moves.
    pub(crate) fn relabel_swap(&mut self, qa: usize, qb: usize) {
        self.site_of_qubit.swap(qa, qb);
        self.qubit_of_site[self.site_of_qubit[qa] as usize] = qa as u32;
        self.qubit_of_site[self.site_of_qubit[qb] as usize] = qb as u32;
    }

    /// Number of qubits (sites) in the chain.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Peak bond dimension currently stored (diagnostic; the scaffold caps it via
    /// the truncation policy).
    pub fn max_bond(&self) -> usize {
        self.sites.iter().map(|s| s.right).max().unwrap_or(1)
    }

    /// State norm `√⟨ψ|ψ⟩` via a doubled transfer-matrix sweep — `O(n·χ³)`, no
    /// `2^n` allocation (P5.7-05 readout). A correctly evolved MPS has norm 1;
    /// this is the cheapest large-`n` correctness invariant for the Phase 5.8
    /// large-χ harness, where the dense oracle is out of reach.
    pub fn norm(&self) -> f64 {
        readout::norm_sq(self).max(0.0).sqrt()
    }

    /// `⟨ψ|P|ψ⟩ / ⟨ψ|ψ⟩ · coefficient` for a Pauli string `P`, via the same
    /// `2^n`-free doubled sweep as [`MetalMpsState::norm`]. Gives an analytic,
    /// dense-free correctness check at large `n` (e.g. GHZ `Z`-string stabilisers).
    pub fn expectation(&self, p: &PauliString) -> Result<f64, BackendError> {
        readout::expectation(self, p)
    }

    /// Contract the chain into a dense `2^n` amplitude vector (TEST/SMALL-n
    /// only — allocates `2^n`). The output is indexed by **qubit** (ADR-0004): site
    /// `s` carries qubit `qubit_of_site[s]`, so its physical bit lands at bit position
    /// `qubit_of_site[s]` of the amplitude index (the lazy permutation, P5.8-05).
    pub fn dense_statevector(&self) -> Vec<Complex<f64>> {
        let mut amps = vec![Complex::<f64>::new(1.0, 0.0)]; // left bond of site 0 = 1
        let mut left_dim = 1usize;
        for (s, site) in self.sites.iter().enumerate() {
            let data = site.buf.as_slice();
            let right = site.right;
            let prefix_count = amps.len() / left_dim;
            let mut next = vec![Complex::<f64>::new(0.0, 0.0); prefix_count * 2 * right];
            for prefix in 0..prefix_count {
                for p in 0..2usize {
                    let new_prefix = prefix | (p << s);
                    for r in 0..right {
                        let mut acc = Complex::<f64>::new(0.0, 0.0);
                        for l in 0..left_dim {
                            let z = data[(l * 2 + p) * right + r];
                            acc += amps[prefix * left_dim + l]
                                * Complex::<f64>::new(z.re as f64, z.im as f64);
                        }
                        next[new_prefix * right + r] += acc;
                    }
                }
            }
            amps = next;
            left_dim = right;
        }
        // `amps` is indexed by *site* bits (bit `s` = site `s`). Permute to qubit
        // order: bit `s` of the site index moves to bit `qubit_of_site[s]` (P5.8-05).
        // Identity permutation ⇒ this is a no-op copy.
        if self
            .qubit_of_site
            .iter()
            .enumerate()
            .all(|(s, &q)| s as u32 == q)
        {
            return amps;
        }
        let n = self.sites.len();
        let mut out = vec![Complex::<f64>::new(0.0, 0.0); amps.len()];
        for (site_idx, amp) in amps.iter().enumerate() {
            let mut qubit_idx = 0usize;
            for s in 0..n {
                if (site_idx >> s) & 1 == 1 {
                    qubit_idx |= 1 << (self.qubit_of_site[s] as usize);
                }
            }
            out[qubit_idx] = *amp;
        }
        out
    }
}

impl aleph_oracle::HasAmplitudes for MetalMpsState {
    fn amplitudes(&self) -> Vec<Complex<f64>> {
        self.dense_statevector()
    }
}
