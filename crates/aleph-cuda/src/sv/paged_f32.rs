//! Out-of-core (host-memory paged) **FP32** single-GPU state vector (P5.11-01).
//!
//! The FP64 paged executor ([`crate::sv::paged`]) streams an `f64` host state, so
//! n=32 needs `2^32 · 16 B = 64 GiB` of pinned host — past this box's 62 GiB. The
//! FP32 buffer halves that: an FP32 paged n=32 state is `2^32 · 8 B = 32 GiB` of
//! pinned host, comfortably inside 62 GiB. This module is the byte-for-byte FP32
//! mirror of [`crate::sv::paged`] — identical tiling scheme (high/low split,
//! co-resident gather, qubit remap), only the scalar type (`f32`) and the per-tile
//! kernels (routed through [`CudaSvBackendF32::apply_gate`]) differ. It sets a new
//! single-GPU **reach** record (n=32) on the 20 GiB card.
//!
//! See [`crate::sv::paged`] for the full derivation of the tiling scheme; the
//! comments here only flag where FP32 diverges (scalar width, kernel dispatch).

use aleph_backend::BackendError;
use aleph_core::{Complex, GateInstance};
use aleph_ir::{Circuit, Instruction};
use cudarc::driver::PinnedHostSlice;

use crate::sv::backend::to_backend_err;
use crate::sv::fp32::{CudaSvBackendF32, CudaSvStateF32, MAX_CUDA_QUBITS_F32};
use crate::sv::paged::{compose_high, high_count};
use crate::{CudaContext, Error};

/// An out-of-core FP32 state vector: the full `2 · 2^n` interleaved `[re, im]`
/// `f32` amplitudes in pinned host memory. Produced by
/// [`CudaSvBackendF32::run_paged`].
pub struct PagedSvStateF32 {
    num_qubits: u32,
    /// Interleaved `[re0, im0, re1, im1, …]`, length `2 · 2^n`, page-locked f32.
    host: PinnedHostSlice<f32>,
}

impl core::fmt::Debug for PagedSvStateF32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PagedSvStateF32")
            .field("num_qubits", &self.num_qubits)
            .field("amps", &(1u64 << self.num_qubits))
            .finish_non_exhaustive()
    }
}

impl PagedSvStateF32 {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// The amplitudes widened to `Vec<Complex<f64>>` (one per basis state) for
    /// oracle comparison. Allocates a second `2^n`-sized host buffer, so use only
    /// at the small `n` of oracle tests — never at n=32 (another 64 GiB). For
    /// large `n` use [`Self::norm_sqr`], which streams the pinned buffer in place.
    pub fn amplitudes_vec(&self) -> Vec<Complex> {
        let host = self.host.as_slice().expect("pinned host sync");
        host.chunks_exact(2)
            .map(|c| Complex::new(c[0] as f64, c[1] as f64))
            .collect()
    }

    /// `Σ |amp|²` (f64 accumulation), streamed over the pinned buffer with no
    /// extra allocation. A cheap large-`n` sanity check (≈ 1 for a unitary
    /// circuit).
    pub fn norm_sqr(&self) -> f64 {
        let host = self.host.as_slice().expect("pinned host sync");
        host.chunks_exact(2)
            .map(|c| (c[0] as f64) * (c[0] as f64) + (c[1] as f64) * (c[1] as f64))
            .sum()
    }
}

impl CudaSvBackendF32 {
    /// Run `circuit` **out-of-core in FP32**: hold the `2^n` state in pinned host
    /// memory and stream device-sized `2^tile_qubits` FP32 tiles through the GPU,
    /// one gate at a time (P5.11-01). Halving the host footprint vs the FP64 paged
    /// path ([`crate::CudaSvBackend::run_paged`]) reaches **n=32** on the box's
    /// 62 GiB host RAM (32 GiB FP32 pinned), at a bandwidth-bound cost (each gate
    /// reads+writes the whole host state once).
    ///
    /// `tile_qubits` is the low-bit split `m`: a tile is `2^m` amplitudes
    /// (`2^m · 8 B` in FP32). It must satisfy `1 ≤ m < n`, and `m + g ≤
    /// MAX_CUDA_QUBITS_F32` where `g` is the most high-qubits any single gate
    /// touches (so the co-resident tile group fits the device). Forcing a small
    /// `m` at small `n` is how the oracle test exercises the paging path.
    ///
    /// Only [`Instruction::Gate`] and [`Instruction::Barrier`] are supported (the
    /// dense-apply tiling is exact for any unitary gate, but diagonal-phase and
    /// tiled-block instructions read the global index) — pass an **unfused**
    /// circuit, and handle measurement on the returned host state.
    pub fn run_paged(
        &mut self,
        circuit: &Circuit,
        tile_qubits: u32,
    ) -> Result<PagedSvStateF32, BackendError> {
        let n = circuit.num_qubits();
        if n == 0 && circuit.is_empty() {
            return Err(BackendError::EmptyCircuit);
        }
        let m = tile_qubits;
        if m == 0 || m >= n {
            return Err(BackendError::InvalidState {
                reason: "paged tile_qubits must satisfy 1 <= tile_qubits < num_qubits",
            });
        }

        // Scan once: reject unsupported instructions and find the largest
        // high-qubit fan-out `g` (sizes the co-resident device group buffer).
        let mut g_max = 0u32;
        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(gate) => g_max = g_max.max(high_count(gate, m)),
                Instruction::Barrier(_) => {}
                Instruction::Measure { .. } => {
                    return Err(BackendError::UnsupportedInstruction { kind: "measure" })
                }
                Instruction::Reset(_) => {
                    return Err(BackendError::UnsupportedInstruction { kind: "reset" })
                }
                Instruction::DiagonalPhase(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "diagonal-phase",
                    })
                }
                Instruction::TiledBlock(_) => {
                    return Err(BackendError::UnsupportedInstruction {
                        kind: "tiled-block",
                    })
                }
            }
        }
        let local_qubits = m + g_max;
        if local_qubits > MAX_CUDA_QUBITS_F32 {
            return Err(BackendError::TooManyQubits {
                requested: local_qubits,
                limit: MAX_CUDA_QUBITS_F32,
            });
        }

        let ctx = self.ctx();

        // Pinned host state: 2·2^n f32, zeroed, amplitude 0 set to |0…0⟩.
        let host_len = 2usize << n; // 2 · 2^n
                                    // SAFETY: `alloc_pinned` returns an owned page-locked allocation of
                                    // `host_len` f32; we initialise every element below before any copy
                                    // reads it. The slice is freed on `PagedSvStateF32` drop.
        let mut host: PinnedHostSlice<f32> = unsafe { ctx.raw().alloc_pinned(host_len) }
            .map_err(|e| to_backend_err(Error::Driver(e)))?;
        {
            let s = host
                .as_mut_slice()
                .map_err(|e| to_backend_err(Error::Driver(e)))?;
            s.fill(0.0);
            s[0] = 1.0;
        }

        // One reusable device buffer big enough for the largest co-resident group
        // (2^(m + g_max) amplitudes); per gate we use only its 2^(m+hh) prefix.
        let mut group = CudaSvStateF32::allocate(&ctx, local_qubits).map_err(to_backend_err)?;

        // Raw host base pointer (synchronises the pinned event once). All tile
        // copies sub-slice this; the regions a single gate touches are disjoint
        // and every copy is stream-ordered, so no CPU-side aliasing occurs.
        let base: *mut f32 = host
            .as_mut_ptr()
            .map_err(|e| to_backend_err(Error::Driver(e)))?;

        for inst in circuit.instructions() {
            if let Instruction::Gate(gate) = inst {
                self.apply_gate_paged(&ctx, &mut group, base, n, m, gate)?;
            }
        }
        ctx.synchronize().map_err(to_backend_err)?;
        Ok(PagedSvStateF32 {
            num_qubits: n,
            host,
        })
    }

    /// Apply one gate over all tile groups (see [`crate::sv::paged`] module docs).
    /// Reuses the in-core FP32 kernels via [`CudaSvBackendF32::apply_gate`] on the
    /// gathered device buffer.
    fn apply_gate_paged(
        &mut self,
        ctx: &CudaContext,
        group: &mut CudaSvStateF32,
        base: *mut f32,
        n: u32,
        m: u32,
        gate: &GateInstance,
    ) -> Result<(), BackendError> {
        let h = n - m;
        // The gate's high qubits (operands ∪ controls, ascending, deduped).
        let mut high: Vec<u32> = gate
            .qubits
            .iter()
            .chain(gate.controls.iter())
            .copied()
            .filter(|&q| q >= m)
            .collect();
        high.sort_unstable();
        high.dedup();
        let hh = high.len() as u32;

        // Remap: low qubit ↦ itself; high qubit ↦ m + its rank in `high`.
        let remap = |q: u32| -> u32 {
            if q < m {
                q
            } else {
                m + high
                    .iter()
                    .position(|&x| x == q)
                    .expect("high qubit listed") as u32
            }
        };
        let mut rg = gate.clone();
        for q in rg.qubits.iter_mut() {
            *q = remap(*q);
        }
        for q in rg.controls.iter_mut() {
            *q = remap(*q);
        }

        // Use only the 2^(m+hh) prefix of the reusable group buffer.
        group.num_qubits = m + hh;

        let hprime: Vec<u32> = high.iter().map(|&q| q - m).collect();
        let tile_amps = 1usize << m; // amplitudes per tile
        let tile_f32 = tile_amps * 2; // interleaved f32 per tile
        let n_outer = 1u64 << (h - hh);
        let n_sub = 1u64 << hh;

        for outer in 0..n_outer {
            // Gather the 2^hh tiles of this group into device offsets s·2^m.
            for s in 0..n_sub {
                let tile = compose_high(outer, s, &hprime, h);
                let h0 = (tile as usize) << (m + 1); // f32 offset = tile·2^m·2
                let d0 = (s as usize) << (m + 1);
                // SAFETY: `base` is valid for `2·2^n` f32; `h0 + tile_f32 ≤ 2·2^n`
                // since `tile < 2^h`. The device view is the matching prefix slot.
                let src = unsafe { std::slice::from_raw_parts(base.add(h0), tile_f32) };
                let mut dst = group.amps.slice_mut().slice_mut(d0..d0 + tile_f32);
                ctx.stream()
                    .memcpy_htod(src, &mut dst)
                    .map_err(|e| to_backend_err(Error::Driver(e)))?;
            }

            // Apply the remapped gate on the (m+hh)-qubit device sub-state.
            self.apply_gate(group, &rg)?;

            // Scatter the tiles back to their host slots.
            for s in 0..n_sub {
                let tile = compose_high(outer, s, &hprime, h);
                let h0 = (tile as usize) << (m + 1);
                let d0 = (s as usize) << (m + 1);
                let view = group.amps.slice().slice(d0..d0 + tile_f32);
                // SAFETY: same bounds as the gather; disjoint from any concurrent
                // copy (each tile is written by exactly one scatter).
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(h0), tile_f32) };
                ctx.stream()
                    .memcpy_dtoh(&view, dst)
                    .map_err(|e| to_backend_err(Error::Driver(e)))?;
            }
        }
        Ok(())
    }
}
