//! Device-resident CHP stabilizer tableau (P5-07) and its Clifford driver.
//!
//! [`CudaStabState`] holds the qubit-major bit-packed tableau in three device
//! buffers (`x`, `z` of `n·Wr` words, `sign` of `Wr` words, `Wr = ceil((2n+1)/64)`).
//! [`CudaStab`] compiles the kernels once and applies Clifford gates either one
//! at a time ([`CudaStab::apply`]) or a whole disjoint-qubit layer per launch
//! ([`CudaStab::apply_layer`]) — the latter amortises launch overhead, which is
//! the binding cost at the qubit counts where a single tableau still fits in
//! cache.
//!
//! Scope: Clifford *evolution* + bit-exact readout against the CPU tableau.
//! Measurement/sampling readout is a follow-up (see the P5-07 perf note).

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::compile_ptx;

use crate::{CudaContext, DeviceBuffer, Error};

const STAB_SRC: &str = include_str!("stab.cu");
const BLOCK: u32 = 256;

/// Logical generator bits `(x, z, sign)`: `x`/`z` row-major `2n × n`, `sign`
/// length `2n` — the shape [`aleph_stab::Tableau::export_generators`] returns, so
/// the GPU and CPU tableaus compare directly.
pub type Generators = (Vec<bool>, Vec<bool>, Vec<bool>);

/// Clifford opcodes, matching the `switch` in `stab.cu`.
pub mod op {
    pub const H: u32 = 0;
    pub const S: u32 = 1;
    pub const CNOT: u32 = 2;
    pub const X: u32 = 3;
    pub const Y: u32 = 4;
    pub const Z: u32 = 5;
}

/// One Clifford gate for the batched [`CudaStab::apply_layer`] path. `b` is
/// ignored for single-qubit ops. Matches the CUDA `StabOp` struct (3×`u32`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StabOp {
    pub op: u32,
    pub a: u32,
    pub b: u32,
}

// SAFETY: `#[repr(C)]`, three `u32` POD fields, no padding (12 bytes), every bit
// pattern valid — cudarc's `DeviceRepr`/`ValidAsZeroBits` contract.
unsafe impl DeviceRepr for StabOp {}
unsafe impl ValidAsZeroBits for StabOp {}

const _: () = assert!(core::mem::size_of::<StabOp>() == 12);

impl StabOp {
    pub fn h(a: u32) -> Self {
        Self { op: op::H, a, b: 0 }
    }
    pub fn s(a: u32) -> Self {
        Self { op: op::S, a, b: 0 }
    }
    pub fn cnot(a: u32, b: u32) -> Self {
        Self { op: op::CNOT, a, b }
    }
    pub fn x(a: u32) -> Self {
        Self { op: op::X, a, b: 0 }
    }
    pub fn y(a: u32) -> Self {
        Self { op: op::Y, a, b: 0 }
    }
    pub fn z(a: u32) -> Self {
        Self { op: op::Z, a, b: 0 }
    }
}

/// Device-resident stabilizer tableau over `n` qubits (qubit-major, bit-packed).
pub struct CudaStabState {
    n: u32,
    /// Words per qubit column = `ceil((2n+1)/64)`.
    wr: u32,
    /// x-bits, `n·Wr` words: `x[a*Wr + w]` is word `w` of qubit `a`'s column.
    x: DeviceBuffer<u64>,
    /// z-bits, `n·Wr` words.
    z: DeviceBuffer<u64>,
    /// sign bits, `Wr` words (packed over the `2n+1` generator-row axis).
    sign: DeviceBuffer<u64>,
    /// Reusable device buffer for batched layer ops, grown on demand.
    ops_scratch: Option<DeviceBuffer<StabOp>>,
    ctx: CudaContext,
}

impl CudaStabState {
    /// Number of qubits.
    pub fn num_qubits(&self) -> u32 {
        self.n
    }

    /// Download the `2n` generator rows as logical bits — the exact shape of
    /// [`aleph_stab::Tableau::export_generators`] (`x`/`z` row-major `2n × n`,
    /// `sign` length `2n`) so the two can be compared bit-for-bit.
    pub fn export_generators(&self) -> Result<Generators, Error> {
        let n = self.n as usize;
        let wr = self.wr as usize;
        let xw = self.x.to_vec(&self.ctx)?;
        let zw = self.z.to_vec(&self.ctx)?;
        let sw = self.sign.to_vec(&self.ctx)?;
        let rows = 2 * n;
        let mut x = vec![false; rows * n];
        let mut z = vec![false; rows * n];
        let mut sign = vec![false; rows];
        let bit = |words: &[u64], col: usize, row: usize| -> bool {
            (words[col * wr + (row >> 6)] >> (row & 63)) & 1 != 0
        };
        for r in 0..rows {
            sign[r] = (sw[r >> 6] >> (r & 63)) & 1 != 0;
            for c in 0..n {
                x[r * n + c] = bit(&xw, c, r);
                z[r * n + c] = bit(&zw, c, r);
            }
        }
        Ok((x, z, sign))
    }
}

/// Clifford driver: holds the compiled kernels, applies gates to a
/// [`CudaStabState`].
pub struct CudaStab {
    ctx: CudaContext,
    f_init: CudaFunction,
    f_gate: CudaFunction,
    f_layer: CudaFunction,
    _module: Arc<CudaModule>,
}

impl CudaStab {
    /// Construct on device 0. Returns [`Error::NoDevice`] on a GPU-less host.
    pub fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0)?;
        let ptx = compile_ptx(STAB_SRC).map_err(|e| Error::Compile(e.to_string()))?;
        let module = ctx.raw().load_module(ptx)?;
        let f_init = module.load_function("stab_init")?;
        let f_gate = module.load_function("stab_gate")?;
        let f_layer = module.load_function("stab_layer")?;
        Ok(Self {
            ctx,
            f_init,
            f_gate,
            f_layer,
            _module: module,
        })
    }

    /// Allocate `|0…0⟩` on `n` qubits (initialised by the `stab_init` kernel).
    pub fn allocate(&self, n: u32) -> Result<CudaStabState, Error> {
        assert!(n > 0, "zero-qubit stabilizer state");
        let rows = 2 * n as u64 + 1;
        let wr = rows.div_ceil(64) as u32;
        let col_words = n as usize * wr as usize;
        let mut x = DeviceBuffer::<u64>::zeros(&self.ctx, col_words)?;
        let mut z = DeviceBuffer::<u64>::zeros(&self.ctx, col_words)?;
        let sign = DeviceBuffer::<u64>::zeros(&self.ctx, wr as usize)?;
        let cfg = launch_config(n as u64);
        let stream = self.ctx.stream();
        // SAFETY: stab_init(u64* x, u64* z, u32 n, u32 Wr); args match in
        // order/type; `x`/`z` hold n·Wr words; grid covers `n` with a guard.
        unsafe {
            stream
                .launch_builder(&self.f_init)
                .arg(x.slice_mut())
                .arg(z.slice_mut())
                .arg(&n)
                .arg(&wr)
                .launch(cfg)?;
        }
        Ok(CudaStabState {
            n,
            wr,
            x,
            z,
            sign,
            ops_scratch: None,
            ctx: self.ctx.clone(),
        })
    }

    /// Apply one Clifford gate via the per-gate kernel (`Wr` threads).
    pub fn apply(&self, state: &mut CudaStabState, gate: StabOp) -> Result<(), Error> {
        let wr = state.wr;
        let cfg = launch_config(wr as u64);
        let stream = self.ctx.stream();
        // SAFETY: stab_gate(u64* x, u64* z, u64* sign, u32 op, u32 a, u32 b,
        // u32 Wr); args match in order/type; the grid covers `Wr` with a guard
        // and the kernel writes only columns a/b and sign word w.
        unsafe {
            stream
                .launch_builder(&self.f_gate)
                .arg(state.x.slice_mut())
                .arg(state.z.slice_mut())
                .arg(state.sign.slice_mut())
                .arg(&gate.op)
                .arg(&gate.a)
                .arg(&gate.b)
                .arg(&wr)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Apply a whole layer of gates on **disjoint** qubits in one launch
    /// (`n_ops · Wr` threads). The caller guarantees disjointness; the sign
    /// update is an atomicXor so concurrent gates may share sign words safely.
    pub fn apply_layer(&self, state: &mut CudaStabState, ops: &[StabOp]) -> Result<(), Error> {
        if ops.is_empty() {
            return Ok(());
        }
        match state.ops_scratch.as_mut() {
            Some(buf) if buf.len() >= ops.len() => buf.write(&self.ctx, ops)?,
            _ => state.ops_scratch = Some(DeviceBuffer::<StabOp>::from_slice(&self.ctx, ops)?),
        }
        let wr = state.wr;
        let n_ops = ops.len() as u32;
        let total = ops.len() as u64 * wr as u64;
        let cfg = launch_config(total);
        let stream = self.ctx.stream();
        let ops_dev = state
            .ops_scratch
            .as_ref()
            .expect("scratch set above")
            .slice();
        // SAFETY: stab_layer(u64* x, u64* z, u64* sign, const StabOp* ops,
        // u32 n_ops, u32 Wr); args match in order/type; grid covers n_ops·Wr
        // with a guard; disjoint qubits ⇒ x/z writes never alias.
        unsafe {
            stream
                .launch_builder(&self.f_layer)
                .arg(state.x.slice_mut())
                .arg(state.z.slice_mut())
                .arg(state.sign.slice_mut())
                .arg(ops_dev)
                .arg(&n_ops)
                .arg(&wr)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Block until all queued kernels finish (for timing).
    pub fn synchronize(&self) -> Result<(), Error> {
        self.ctx.synchronize()
    }
}

/// `((n + BLOCK - 1) / BLOCK)` blocks of `BLOCK` threads, ≥1 block.
fn launch_config(n_threads: u64) -> LaunchConfig {
    let blocks = n_threads.div_ceil(BLOCK as u64).max(1) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}
