//! Out-of-core (host-memory paged) single-GPU state vector (P5.10-02).
//!
//! [`crate::sv::state::MAX_CUDA_QUBITS`] = 30 is the largest state that fits the
//! 20 GiB card (16 GiB of FP64 at n=30). Beyond that the `2^n` state must live in
//! host memory and stream through the device a tile at a time. This module holds
//! the full state in **pinned** host memory and processes each gate over
//! device-sized tiles, so n=31+ runs on one GPU at a bandwidth-bound cost.
//!
//! ## Tiling scheme
//!
//! Split the `n` qubits into the `m = tile_qubits` **low** bits (a device tile is
//! `2^m` contiguous amplitudes) and `h = n - m` **high** bits (which tile). The
//! global index is `x = (high << m) | low`; tile `T ∈ [0, 2^h)` is the host slice
//! `state[T·2^m .. (T+1)·2^m)`.
//!
//! A dense gate's result at each amplitude depends only on the *values* of the
//! amplitudes it pairs, never on the absolute index (only diagonal/phase kernels
//! read the index — those are deliberately excluded; the caller passes an
//! unfused gate circuit). So we apply **every** gate densely on a device-local
//! sub-state by *remapping* its qubits into device-local positions:
//!
//! - a gate touching high-qubit set `H` (`hh = |H|`) needs its `2^hh` tiles
//!   (varying the `H` bits, all other high bits fixed) **co-resident** on the
//!   device. We gather them into one contiguous device buffer at offsets
//!   `s·2^m` (`s ∈ [0, 2^hh)`), so device-local bit `m + rank` ↔ global qubit
//!   `H[rank]`; low qubits map to themselves.
//! - the gate is then a normal `(m + hh)`-qubit gate on that buffer — applied by
//!   reusing the existing in-core [`Backend::apply_gate`] kernels with the
//!   qubits/controls remapped — after which the tiles scatter back to host.
//!
//! Iterating over the `2^(h - hh)` "outer" high combinations covers the whole
//! state in exactly one read + one write per gate (bandwidth-optimal, same as
//! in-core). Today the copies are stream-ordered and synchronous (cudarc forces a
//! host sync when a borrowed slice backs an async copy); true H2D/D2H/compute
//! overlap via raw-FFI async + double buffering is a follow-up — see the
//! `tests/paged_bench.rs` report and `docs/perf/p5.10-02-host-paging.md`.

use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, GateInstance};
use aleph_ir::{Circuit, Instruction};
use cudarc::driver::PinnedHostSlice;

use crate::sv::backend::to_backend_err;
use crate::sv::state::{CudaSvState, MAX_CUDA_QUBITS};
use crate::sv::CudaSvBackend;
use crate::{CudaContext, Error};

/// An out-of-core state vector: the full `2 · 2^n` interleaved `[re, im]` FP64
/// amplitudes in pinned host memory. Produced by [`CudaSvBackend::run_paged`].
pub struct PagedSvState {
    num_qubits: u32,
    /// Interleaved `[re0, im0, re1, im1, …]`, length `2 · 2^n`, page-locked.
    host: PinnedHostSlice<f64>,
}

impl core::fmt::Debug for PagedSvState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PagedSvState")
            .field("num_qubits", &self.num_qubits)
            .field("amps", &(1u64 << self.num_qubits))
            .finish_non_exhaustive()
    }
}

impl PagedSvState {
    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// The amplitudes as `Vec<Complex>` (one per basis state). Allocates a second
    /// `2^n`-sized host buffer, so use only at the small `n` of oracle tests —
    /// never at n=31 (that would be another 32 GiB). For large `n` use
    /// [`Self::norm_sqr`], which streams the pinned buffer in place.
    pub fn amplitudes_vec(&self) -> Vec<Complex> {
        let host = self.host.as_slice().expect("pinned host sync");
        host.chunks_exact(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect()
    }

    /// `Σ |amp|²`, streamed over the pinned buffer with no extra allocation. A
    /// cheap large-`n` sanity check (should be ≈ 1 for a unitary circuit).
    pub fn norm_sqr(&self) -> f64 {
        let host = self.host.as_slice().expect("pinned host sync");
        host.chunks_exact(2)
            .map(|c| c[0] * c[0] + c[1] * c[1])
            .sum()
    }
}

/// High-bit index `T` whose bits at the `hprime` positions carry `s` (LSB-first)
/// and whose remaining positions carry `outer` (LSB-first). Reconstructs which
/// tile sub-block `s` of co-resident group `outer` maps to.
fn compose_high(outer: u64, s: u64, hprime: &[u32], h: u32) -> u64 {
    let mut val = 0u64;
    let mut oc = 0u32; // cursor over `outer`'s bits
    for p in 0..h {
        if let Some(rank) = hprime.iter().position(|&x| x == p) {
            if (s >> rank) & 1 == 1 {
                val |= 1u64 << p;
            }
        } else {
            if (outer >> oc) & 1 == 1 {
                val |= 1u64 << p;
            }
            oc += 1;
        }
    }
    val
}

impl CudaSvBackend {
    /// Run `circuit` **out-of-core**: hold the `2^n` state in pinned host memory
    /// and stream device-sized `2^tile_qubits` tiles through the GPU, one gate at
    /// a time (P5.10-02). This lifts the [`MAX_CUDA_QUBITS`] ceiling — n=31 fits a
    /// 20 GiB card — at a bandwidth-bound cost (each gate reads+writes the whole
    /// host state once).
    ///
    /// `tile_qubits` is the low-bit split `m`: a tile is `2^m` amplitudes
    /// (`2^m · 16 B`). It must satisfy `1 ≤ m < n`, and `m + g ≤ MAX_CUDA_QUBITS`
    /// where `g` is the most high-qubits any single gate touches (so the
    /// co-resident tile group fits the device). Forcing a small `m` at small `n`
    /// is how the oracle test exercises the paging path.
    ///
    /// Only [`Instruction::Gate`] and [`Instruction::Barrier`] are supported:
    /// the dense-apply tiling is exact for any unitary gate, but diagonal-phase
    /// and tiled-block instructions read the global index — pass an **unfused**
    /// circuit (no `DiagonalPhase`/`TiledBlock`), and handle measurement on the
    /// returned host state.
    pub fn run_paged(
        &mut self,
        circuit: &Circuit,
        tile_qubits: u32,
    ) -> Result<PagedSvState, BackendError> {
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
        if local_qubits > MAX_CUDA_QUBITS {
            return Err(BackendError::TooManyQubits {
                requested: local_qubits,
                limit: MAX_CUDA_QUBITS,
            });
        }

        let ctx = self.ctx();

        // Pinned host state: 2·2^n f64, zeroed, amplitude 0 set to |0…0⟩.
        let host_len = 2usize << n; // 2 · 2^n
                                    // SAFETY: `alloc_pinned` returns an owned page-locked allocation of
                                    // `host_len` f64; we initialise every element below before any copy
                                    // reads it. The slice is freed on `PagedSvState` drop.
        let mut host: PinnedHostSlice<f64> = unsafe { ctx.stream().alloc_pinned(host_len) }
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
        let mut group = CudaSvState::allocate(&ctx, local_qubits).map_err(to_backend_err)?;

        // Raw host base pointer (synchronises the pinned event once). All tile
        // copies sub-slice this; the regions a single gate touches are disjoint
        // and every copy is stream-ordered, so no CPU-side aliasing occurs.
        let base: *mut f64 = host
            .as_mut_ptr()
            .map_err(|e| to_backend_err(Error::Driver(e)))?;

        for inst in circuit.instructions() {
            if let Instruction::Gate(gate) = inst {
                self.apply_gate_paged(&ctx, &mut group, base, n, m, gate)?;
            }
        }
        ctx.synchronize().map_err(to_backend_err)?;
        Ok(PagedSvState {
            num_qubits: n,
            host,
        })
    }

    /// Apply one gate over all tile groups (see the module docs). Reuses the
    /// in-core kernels via [`Backend::apply_gate`] on the gathered device buffer.
    fn apply_gate_paged(
        &mut self,
        ctx: &CudaContext,
        group: &mut CudaSvState,
        base: *mut f64,
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
        let tile_f64 = tile_amps * 2; // interleaved f64 per tile
        let n_outer = 1u64 << (h - hh);
        let n_sub = 1u64 << hh;

        for outer in 0..n_outer {
            // Gather the 2^hh tiles of this group into device offsets s·2^m.
            for s in 0..n_sub {
                let tile = compose_high(outer, s, &hprime, h);
                let h0 = (tile as usize) << (m + 1); // f64 offset = tile·2^m·2
                let d0 = (s as usize) << (m + 1);
                // SAFETY: `base` is valid for `2·2^n` f64; `h0 + tile_f64 ≤ 2·2^n`
                // since `tile < 2^h`. The device view is the matching prefix slot.
                let src = unsafe { std::slice::from_raw_parts(base.add(h0), tile_f64) };
                let mut dst = group.amps.slice_mut().slice_mut(d0..d0 + tile_f64);
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
                let view = group.amps.slice().slice(d0..d0 + tile_f64);
                // SAFETY: same bounds as the gather; disjoint from any concurrent
                // copy (each tile is written by exactly one scatter).
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(h0), tile_f64) };
                ctx.stream()
                    .memcpy_dtoh(&view, dst)
                    .map_err(|e| to_backend_err(Error::Driver(e)))?;
            }
        }
        Ok(())
    }
}

/// Number of distinct **high** qubits (≥ `m`) a gate touches (operands ∪
/// controls) — the `hh` that sizes its co-resident tile group.
fn high_count(gate: &GateInstance, m: u32) -> u32 {
    let mut high: Vec<u32> = gate
        .qubits
        .iter()
        .chain(gate.controls.iter())
        .copied()
        .filter(|&q| q >= m)
        .collect();
    high.sort_unstable();
    high.dedup();
    high.len() as u32
}

#[cfg(test)]
mod tests {
    use super::compose_high;

    /// `compose_high` must place `s` into the `hprime` bit positions and `outer`
    /// into the rest, LSB-first, so the `2^hh` sub-tiles of a group differ only in
    /// the high qubits the gate touches.
    #[test]
    fn compose_high_scatters_bits() {
        // h=4 high bits, gate touches high positions {1, 3} (hprime).
        let hprime = [1u32, 3u32];
        let h = 4;
        // outer fills positions {0, 2}. outer=0b11 → bit0=1, bit2=1.
        // s fills positions {1, 3}. s=0b10 → rank0(pos1)=0, rank1(pos3)=1 ⇒ bit3=1.
        // expected = bits {0,2,3} set = 0b1101 = 13.
        assert_eq!(compose_high(0b11, 0b10, &hprime, h), 0b1101);
        // s=0 outer=0 ⇒ 0; s=0b11 ⇒ bits 1,3 ⇒ 0b1010=10; outer=0b01 ⇒ bit0 ⇒ +1.
        assert_eq!(compose_high(0, 0, &hprime, h), 0);
        assert_eq!(compose_high(0, 0b11, &hprime, h), 0b1010);
        assert_eq!(compose_high(0b01, 0b11, &hprime, h), 0b1011);
    }
}
