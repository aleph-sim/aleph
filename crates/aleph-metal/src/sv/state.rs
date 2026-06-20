//! `MetalSvState` — device-resident FP32 statevector. AoS
//! `DeviceBuffer<Complex<f32>>` of length 2^n; `amps[i]` is the amplitude of
//! `|i⟩` (MSB qubit convention, ADR 0004), matching every other SV backend.

use std::cell::{Cell, RefCell};

use aleph_core::Complex;
use metal::{CommandBuffer, CommandQueue, ComputeCommandEncoderRef};

use crate::{DeviceBuffer, MetalContext};

/// State vector held by [`crate::MetalSvBackend`]. Unified-memory shared
/// storage: host views are zero-copy windows onto the same bytes the GPU sees.
///
/// Gates are dispatched into a single *batched* command buffer with no per-gate
/// `wait_until_completed` (P5.6-04); the first host read drains it via [`sync`].
/// The batch lives in the state (not the backend) so that *any* driver — the
/// inherent `run` or the generic `aleph_backend::run` — produces a state that
/// becomes current on its first read, with no caller obligation to flush.
/// `RefCell`/`Cell` give that drain through `&self`.
///
/// `kq_pool` is a single reusable matrix-buffer arena: each `apply_kq` dispatch
/// in the batch gets its own `KQ_STRIDE`-sized slot (distinct byte offset), so
/// concurrently-encoded kq gates never race over a shared region — yet the
/// arena is allocated once and reused across batches, avoiding a per-gate buffer
/// allocation. `kq_slot` is the next free slot in the open batch.
///
/// [`sync`]: MetalSvState::sync
pub struct MetalSvState {
    pub(crate) num_qubits: u32,
    pub(crate) amps: DeviceBuffer<Complex<f32>>,
    pending: RefCell<Option<CommandBuffer>>,
    kq_pool: RefCell<Option<DeviceBuffer<Complex<f32>>>>,
    kq_slot: Cell<usize>,
    // Per-dispatch side buffers (e.g. a diagonal phase's condition-mask and
    // term-descriptor buffers) that the GPU reads after `commit`; held alive
    // until `sync`. `Box<dyn Any>` so heterogeneous DeviceBuffer<T> can share one
    // arena without a typed field per kind.
    aux: RefCell<Vec<Box<dyn core::any::Any>>>,
    batch_len: Cell<usize>,
}

/// Dispatches encoded into one batched command buffer before a forced `sync`
/// (P5.6-04). Bounds the open command buffer and the `kq_pool` arena; a deep
/// circuit pays one wait per `BATCH_CAP` gates instead of one per gate.
pub(crate) const BATCH_CAP: usize = 256;

/// Entries reserved per kq matrix slot in `kq_pool`: `4^5 = 1024`, the largest
/// (k=5) dense block. 8 KiB/slot × `BATCH_CAP` ⇒ a 2 MiB arena, allocated lazily
/// on the first kq dispatch (pure-1q circuits never allocate it).
const KQ_STRIDE: usize = 1024;

impl core::fmt::Debug for MetalSvState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `DeviceBuffer` wraps a `metal::Buffer` which is not `Debug`; show the
        // qubit count and amplitude length as a compact summary instead.
        f.debug_struct("MetalSvState")
            .field("num_qubits", &self.num_qubits)
            .field("amps_len", &self.amps.len())
            .finish_non_exhaustive()
    }
}

impl MetalSvState {
    /// Wrap an amplitude buffer with an empty (closed) gate batch.
    pub(crate) fn new(num_qubits: u32, amps: DeviceBuffer<Complex<f32>>) -> Self {
        Self {
            num_qubits,
            amps,
            pending: RefCell::new(None),
            kq_pool: RefCell::new(None),
            kq_slot: Cell::new(0),
            aux: RefCell::new(Vec::new()),
            batch_len: Cell::new(0),
        }
    }

    /// Number of qubits this state represents.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Number of dispatches encoded into the currently-open batch.
    pub(crate) fn batch_len(&self) -> usize {
        self.batch_len.get()
    }

    /// Encode one dispatch into the open batch, opening a command buffer on
    /// `queue` if none is open. `f` sets the pipeline/buffers/bytes and issues
    /// the threadgroup dispatch on the encoder. Does **not** commit — [`sync`]
    /// does, lazily, on the first host read or at a batch-size cap.
    ///
    /// [`sync`]: MetalSvState::sync
    pub(crate) fn encode<F>(&self, queue: &CommandQueue, f: F)
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        let mut pending = self.pending.borrow_mut();
        if pending.is_none() {
            *pending = Some(queue.new_command_buffer().to_owned());
        }
        // Safe: just set to Some above. Encoder borrows the buffer for the scope
        // of this call only; `end_encoding` closes it before `pending` drops.
        let enc = pending
            .as_ref()
            .expect("pending command buffer just set")
            .new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        self.batch_len.set(self.batch_len.get() + 1);
    }

    /// Reserve the next `kq_pool` slot for this batch, uploading `matrix`
    /// (length ≤ `KQ_STRIDE`) into it, and return the slot's **byte** offset for
    /// `set_buffer`. Lazily allocates the arena on first use. The caller is
    /// responsible for keeping the open batch ≤ `BATCH_CAP` dispatches (the
    /// backend's `maybe_flush`), so `kq_slot < BATCH_CAP` always holds here.
    pub(crate) fn kq_upload(&self, ctx: &MetalContext, matrix: &[Complex<f32>]) -> u64 {
        debug_assert!(
            matrix.len() <= KQ_STRIDE,
            "kq matrix wider than a pool slot"
        );
        let slot = self.kq_slot.get();
        let mut pool = self.kq_pool.borrow_mut();
        if pool.is_none() {
            *pool = Some(DeviceBuffer::from_slice(
                ctx,
                &vec![Complex::<f32>::new(0.0, 0.0); BATCH_CAP * KQ_STRIDE],
            ));
        }
        let base = slot * KQ_STRIDE;
        pool.as_mut()
            .expect("kq pool just allocated")
            .as_mut_slice()[base..base + matrix.len()]
            .copy_from_slice(matrix);
        self.kq_slot.set(slot + 1);
        (base * std::mem::size_of::<Complex<f32>>()) as u64
    }

    /// Shared borrow of the kq arena buffer for binding inside an `encode`
    /// closure. Must follow a `kq_upload` in the same batch (the arena exists).
    pub(crate) fn kq_pool_buffer(&self) -> std::cell::Ref<'_, DeviceBuffer<Complex<f32>>> {
        std::cell::Ref::map(self.kq_pool.borrow(), |o| {
            o.as_ref().expect("kq pool allocated by kq_upload")
        })
    }

    /// Keep a per-dispatch side buffer alive until the batch is synced (the GPU
    /// reads it after `commit`). Used for the diagonal-phase condition/descriptor
    /// buffers, which — unlike kq matrices — vary in size and type per dispatch.
    pub(crate) fn retain_aux<T: core::any::Any>(&self, buf: T) {
        self.aux.borrow_mut().push(Box::new(buf));
    }

    /// Commit and wait on the open batch, making `amps` current for a host read.
    /// Resets the batch counters but **keeps** the `kq_pool` arena for reuse by
    /// the next batch (its slots are now safe to overwrite — the GPU has read
    /// them). No-op when no batch is open; idempotent. Every host-read accessor
    /// calls this first, so callers never observe a half-applied state.
    pub(crate) fn sync(&self) {
        if let Some(cmd) = self.pending.borrow_mut().take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
        self.kq_slot.set(0);
        self.aux.borrow_mut().clear();
        self.batch_len.set(0);
    }

    /// Zero-copy read-only view of the single-precision amplitude buffer. Drains
    /// any pending GPU batch first, so the bytes are current.
    pub fn amplitudes_f32(&self) -> &[Complex<f32>] {
        self.sync();
        self.amps.as_slice()
    }

    /// Widen to `Vec<Complex<f64>>` for oracle / interop comparison. The
    /// FP32→FP64 widening is exact; the 1e-5 oracle tolerance accounts for the
    /// single-precision accumulation error already in the buffer.
    pub fn to_aos_f64(&self) -> Vec<Complex<f64>> {
        self.sync();
        self.amps
            .as_slice()
            .iter()
            .map(|a| Complex::<f64>::new(a.re as f64, a.im as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalContext;

    #[test]
    fn widen_and_view_round_trip() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping state test: no Metal device");
                return;
            }
        };
        let data = [
            Complex::<f32>::new(0.5, 0.0),
            Complex::<f32>::new(0.0, -0.25),
        ];
        let amps = DeviceBuffer::from_slice(&ctx, &data);
        let s = MetalSvState::new(1, amps);
        assert_eq!(s.num_qubits(), 1);
        assert_eq!(s.amplitudes_f32().len(), 2);
        let w = s.to_aos_f64();
        assert_eq!(w[0], Complex::<f64>::new(0.5, 0.0));
        assert_eq!(w[1], Complex::<f64>::new(0.0, -0.25));
    }
}
