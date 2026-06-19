//! Device-resident MPS state: a chain of rank-3 site tensors in FP32, plus the
//! host dense contraction used by the oracle.

use aleph_core::Complex;

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

    /// Build a site from host data (length must be `left*2*right`).
    pub(crate) fn from_host(
        ctx: &MetalContext,
        left: usize,
        right: usize,
        data: &[Complex<f32>],
    ) -> Self {
        debug_assert_eq!(data.len(), left * 2 * right);
        Self {
            left,
            right,
            buf: DeviceBuffer::from_slice(ctx, data),
        }
    }
}

/// A device-resident matrix-product state over `num_qubits` sites. Scaffold
/// invariant (P5.5-06): NN-only, so site order ≡ qubit order throughout — no
/// permutation bookkeeping. Site `s` carries the bit of qubit `s`.
pub struct MetalMpsState {
    pub(crate) num_qubits: u32,
    pub(crate) sites: Vec<SiteTensor>,
}

impl MetalMpsState {
    /// The product state `|0…0⟩`: each site is a `|0⟩` ket.
    pub(crate) fn product(ctx: &MetalContext, num_qubits: u32) -> Self {
        let sites = (0..num_qubits).map(|_| SiteTensor::ket0(ctx)).collect();
        Self { num_qubits, sites }
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

    /// Contract the chain into a dense `2^n` amplitude vector (TEST/SMALL-n
    /// only — allocates `2^n`). Mirrors the CPU MPS `dense_statevector` Phase 1:
    /// bit `s` of the index is the physical index of site `s`, which (identity
    /// permutation) is the value of qubit `s` — the ADR-0004 convention shared
    /// with the SV backends.
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
        amps
    }
}

impl aleph_oracle::HasAmplitudes for MetalMpsState {
    fn amplitudes(&self) -> Vec<Complex<f64>> {
        self.dense_statevector()
    }
}
