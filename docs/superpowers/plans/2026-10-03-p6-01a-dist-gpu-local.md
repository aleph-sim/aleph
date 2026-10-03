# [P6-01a] Distributed SV on the GPU: `DistSvBackend` + `LocalExchange` (one card): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an `aleph_ir::dist::DistPlan` on CUDA. `R = 2^g` ranks, each a `2^m`-amplitude device slice, all on
**one** GPU. Local steps run through the existing FP64/FP32 kernels with per-rank fusion. Exchanges go through a
`LocalExchange` that swaps contiguous chunks between rank slices in place, through a bounded scratch buffer. Verify it
bit-for-bit against the CPU oracle on the RTX 4000, and measure its overhead against single-GPU `CudaSvBackend`.

**Architecture:**
- **`DeviceSv` trait** (crate-internal impls for `CudaSvBackend` and `CudaSvBackendF32`). It adds what the distributed
  layer needs beyond `Backend`: allocate a rank slice (|0⟩ on rank 0, zeros elsewhere), copy an amplitude range between
  two states, download, and upload (for tests).
  - Rank slices are ordinary `CudaSvState`/`CudaSvStateF32` with `num_qubits = m`, so `Backend::apply_gate` and
    `apply_diagonal_phase` run the existing kernels unchanged. `paged.rs` already does exactly this.
- **`Exchange<B>` trait** with **`LocalExchange<B>`** (one device).
  - The k-bit exchange permutation on chunks is an involution: `(r, c) ↔ (r', c')`.
  - So each non-fixed chunk pair is swapped once: chunk A → scratch, B → A, scratch → B. The work goes in pieces of
    at most `scratch_amps`, with no double buffer.
- **`DistSvBackend<B, X>`** runs a plan:
  - `Local` steps: per rank, `specialize`, then build an m-qubit circuit, then `fuse_for_gpu` (optional), then apply.
  - `Exchange` steps go through `X`.
  - Readout gathers the full state to the host via `final_map` (tests, n ≤ ~30) and computes the norm.
- `NcclExchange` (real multiple GPUs) is the next PR (P6-01b). The `Exchange` trait is shaped so that PR only adds an
  implementation and a `Vec<B>` of per-device backends.

**Tech Stack:** Rust 2021 (MSRV 1.89), `cudarc` 0.19 (existing dep, no new features), `thiserror`. New **dev-dep**:
`aleph-parser` (workspace crate, for the Grover QASM fixture). No new external crates.

**Spec:** `docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md` (§3.3, §3.4, §4, §5 "GPU box", §7 item 3; refs #55).
This branch is stacked on `p6-03-lookahead-router` (P6-03) on top of `p6-02-dist-plan` (#527). Rebase onto `main`
after both merge.

## Global Constraints

- Everything new in `aleph-cuda` is gated `#[cfg(all(target_os = "linux", feature = "cuda"))]`, like the rest of the
  crate.
  - macOS cannot build it. **Build, test and clippy on the GPU box**, `ssh root@openwebgui.splynx.com` (RTX 4000 SFF Ada
    20 GiB, CUDA 13.0).
  - CI (Linux, no GPU) runs `cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings` and
    `cargo build -p aleph-cuda --features cuda`. Both must pass.
- No `unwrap()`/`expect()`/`panic!` in library code. Errors are `DistSvError` (thiserror). Test code may `unwrap`.
- FP64 tolerance is `1e-10` vs `NaiveSvBackend`. FP32 tolerance is `1e-5` per amplitude vs `NaiveSvBackend`, the same
  as `tests/paged_f32_oracle.rs`.
- GPU tests skip cleanly without a device: `match CudaSvBackend::with_seed(0) { Ok(b) => b, Err(e) => { eprintln!("skipping …: {e}"); return; } }`.
- Benchmarks are `#[test] #[ignore]` files under `crates/aleph-cuda/tests/*_bench.rs` (crate convention). Before
  timing, check the box is idle: `uptime` load ≈ 0, and `pgrep -af "cargo bench|cargo test|bencher|Runner.Worker"` is
  empty.
- Interleaved amplitude layout: `amps` holds `[re, im]` pairs. Amplitude `i` is scalars `2i..2i+2`, so amplitude
  ranges ×2 in buffer units.
- Branch `p6-01a-dist-gpu` in the main checkout. No worktrees. To build on the box, rsync the tree (see Task 1 Step 0).
  Never commit from the box.
- PR title `[P6-01a] Distributed SV on one GPU: DistSvBackend + LocalExchange`, body `Refs #55`. It does not close #55;
  2/4/8 real GPUs and NCCL come later.

## Review Focus

1. **Exchange chunk pairing for k ≥ 2 on the GPU** must equal `aleph_sv::dist_ref::exchange_cpu` on arbitrary data,
   not only on states reachable from |0⟩. Pinned in Task 2 (`local_exchange_matches_cpu_reference`, with random
   uploaded slices and bit orders `[m,m+1]`, `[m+1,m]`, `[m+2,m,m+1]`).
2. **Scratch smaller than a chunk** (`scratch_amps` = 4 amps while chunks are thousands of amplitudes). Expected: the
   piece loop covers every amplitude exactly once. Pinned in Task 2 (same test, run with tiny scratch) and in the
   Task 3 oracle (`with_scratch_amps(8)`).
3. **Rank 0 vs other ranks initial state.** Only rank 0 holds amplitude 1 at index 0; the other ranks are zero.
   Expected: the norm is 1 and the GHZ result is correct at every g. Pinned in Task 3.
4. **Per-rank fusion of specialised gates** (a `Unitary1qDiag` with controls, a `Unitary2q` diagonal, an m-qubit
   `DiagonalPhase` with a per-rank scalar term) must survive `fuse_for_gpu` and the kernels. Pinned by the Task 3
   oracle with `fuse = true` on QFT (controlled phases on global qubits) and the all-diagonal-gates case.
5. **FP32 path** goes through the same generic code. Expected: it is oracle-equal at 1e-5. Pinned in Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-cuda/src/dist/mod.rs` (create) | `DeviceSv`, `DistSvError`, `DistSvBackend`, `DistSvState`, re-exports |
| `crates/aleph-cuda/src/dist/device_sv.rs` (create) | `impl DeviceSv for CudaSvBackend` and `for CudaSvBackendF32` |
| `crates/aleph-cuda/src/dist/exchange.rs` (create) | `Exchange<B>` trait, `LocalExchange<B>`, chunk-pair enumeration |
| `crates/aleph-cuda/src/lib.rs` (modify) | `mod dist;` + `pub use dist::{…}` under the cuda cfg |
| `crates/aleph-cuda/Cargo.toml` (modify) | dev-dep `aleph-parser` |
| `crates/aleph-cuda/tests/dist_gpu_oracle.rs` (create) | GPU oracle vs NaiveSv (FP64/FP32, both routers, fuse on/off, tiny scratch) + exchange-vs-CPU-reference |
| `crates/aleph-cuda/tests/dist_local_bench.rs` (create) | `#[ignore]` timing: DistSvBackend(R=1,2,4) vs CudaSvBackend |
| `docs/perf/p6-01a-dist-gpu.md` (create) | correctness + overhead numbers |

---

### Task 1: `DeviceSv` trait + impls (FP64, FP32)

**Files:**
- Create: `crates/aleph-cuda/src/dist/mod.rs` and `crates/aleph-cuda/src/dist/device_sv.rs`
- Modify: `crates/aleph-cuda/src/lib.rs`
- Create: `crates/aleph-cuda/tests/dist_gpu_oracle.rs` (first tests)

**Interfaces:**
- Consumes:
  - `CudaSvState { num_qubits, amps: DeviceBuffer<f64>, ctx, mat_scratch }` (`sv/state.rs:24`, fields `pub(crate)`).
  - `CudaSvStateF32` (`sv/fp32.rs:85`, same shape with `f32`).
  - `DeviceBuffer::{zeros, write, from_slice, to_vec, slice, slice_mut}` (`buffer.rs`).
  - `CudaContext::stream()` (`context.rs`).
  - `CudaSvBackend::ctx()` (`backend.rs:126`, `pub(crate)`), and the FP32 equivalent (check `fp32.rs` for the field
    or accessor name).
  - `to_backend_err` (`sv/backend.rs:610`).
- Produces:

```rust
pub trait DeviceSv: Backend {
    /// A rank slice of `m` qubits: |0…0⟩ if `rank == 0`, all-zero otherwise.
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<Self::State, BackendError>;
    /// Copy `len` amplitudes `src[src_off..]` → `dst[dst_off..]` (device-to-device, same device).
    fn copy_amps(&mut self, src: &Self::State, src_off: usize, dst: &mut Self::State, dst_off: usize, len: usize)
        -> Result<(), BackendError>;
    /// Download all amplitudes as complex f64 (FP32 widens).
    fn download(&mut self, st: &Self::State) -> Result<Vec<Complex<f64>>, BackendError>;
    /// Upload `amps` (len = 2^m) into a fresh `m`-qubit slice (tests / reference diffs).
    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<Self::State, BackendError>;
}
```

- [ ] **Step 0: Prepare the GPU box once.**

```bash
ssh root@openwebgui.splynx.com 'nvidia-smi --query-gpu=name,memory.total --format=csv; uptime; which cargo || ls ~/.cargo/bin'
rsync -a --delete --exclude target --exclude .git ./ root@openwebgui.splynx.com:/root/aleph-p6/
```

Every later "Run (box)" means: rsync first, then
`ssh root@openwebgui.splynx.com 'cd /root/aleph-p6 && <command>'`. If `cargo` is not on the non-interactive PATH,
prefix `source ~/.cargo/env &&`. Record the exact working incantation in the ledger once.

- [ ] **Step 1: Write the failing test** (`crates/aleph-cuda/tests/dist_gpu_oracle.rs`)

```rust
//! P6-01a: distributed SV on one GPU (DistSvBackend + LocalExchange) vs the
//! CPU oracle. Skips without a CUDA device.
#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_core::Complex;
use aleph_cuda::{CudaSvBackend, CudaSvBackendF32, DeviceSv};

fn gpu64() -> Option<CudaSvBackend> {
    match CudaSvBackend::with_seed(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping dist GPU test: {e}");
            None
        }
    }
}

fn gpu32() -> Option<CudaSvBackendF32> {
    match CudaSvBackendF32::with_seed(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping dist GPU test: {e}");
            None
        }
    }
}

fn ramp(len: usize, salt: f64) -> Vec<Complex<f64>> {
    (0..len)
        .map(|i| Complex::new(i as f64 * 0.001 + salt, -(i as f64) * 0.002))
        .collect()
}

#[test]
fn device_sv_alloc_copy_roundtrip_f64() {
    let Some(mut be) = gpu64() else { return };
    let r0 = be.alloc_rank(5, 0).unwrap();
    let r1 = be.alloc_rank(5, 1).unwrap();
    let d0 = be.download(&r0).unwrap();
    let d1 = be.download(&r1).unwrap();
    assert_eq!(d0.len(), 32);
    assert_eq!(d0[0], Complex::new(1.0, 0.0));
    assert!(d0[1..].iter().all(|a| *a == Complex::new(0.0, 0.0)));
    assert!(d1.iter().all(|a| *a == Complex::new(0.0, 0.0)));

    let src = be.upload(5, &ramp(32, 0.5)).unwrap();
    let mut dst = be.alloc_rank(5, 3).unwrap();
    be.copy_amps(&src, 4, &mut dst, 20, 8).unwrap();
    let got = be.download(&dst).unwrap();
    let want = ramp(32, 0.5);
    for i in 0..32 {
        let w = if (20..28).contains(&i) { want[i - 16] } else { Complex::new(0.0, 0.0) };
        assert_eq!(got[i], w, "amp {i}");
    }
}

#[test]
fn device_sv_alloc_copy_roundtrip_f32() {
    let Some(mut be) = gpu32() else { return };
    let src = be.upload(4, &ramp(16, 0.25)).unwrap();
    let mut dst = be.alloc_rank(4, 1).unwrap();
    be.copy_amps(&src, 0, &mut dst, 8, 8).unwrap();
    let got = be.download(&dst).unwrap();
    let want = ramp(16, 0.25);
    for i in 8..16 {
        assert!((got[i] - want[i - 8]).norm() < 1e-6, "amp {i}");
    }
    assert!(got[..8].iter().all(|a| a.norm() == 0.0));
    let r0 = be.alloc_rank(4, 0).unwrap();
    assert!((be.download(&r0).unwrap()[0] - Complex::new(1.0, 0.0)).norm() < 1e-7);
}
```

Add to `crates/aleph-cuda/Cargo.toml` `[dev-dependencies]`:

```toml
aleph-parser  = { path = "../aleph-parser" }
```

- [ ] **Step 2: Run (box) to verify it fails**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle`
Expected: FAIL to compile (`DeviceSv` not found in `aleph_cuda`).

- [ ] **Step 3: Write the implementation**

`crates/aleph-cuda/src/dist/mod.rs`:

```rust
//! Distributed state vector on CUDA (Phase 6, P6-01a).
//!
//! Executes an `aleph_ir::dist::DistPlan`: `2^g` rank slices of `2^m`
//! amplitudes, local steps through the existing kernels (rank-specialised,
//! optionally fused per rank), exchanges through an [`Exchange`]
//! implementation. [`LocalExchange`] keeps every rank on one device (the
//! single-card development path); NCCL across devices is P6-01b.

use aleph_backend::{Backend, BackendError};
use aleph_core::Complex;

mod device_sv;
mod exchange;

pub use exchange::{Exchange, LocalExchange};

/// What the distributed layer needs from a single-device SV backend beyond
/// [`Backend`]: rank-slice allocation, device-to-device amplitude copies, and
/// host transfer for readout / tests.
pub trait DeviceSv: Backend {
    /// A rank slice of `m` qubits: |0…0⟩ if `rank == 0`, all-zero otherwise.
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<Self::State, BackendError>;
    /// Copy `len` amplitudes `src[src_off..]` → `dst[dst_off..]` on the device.
    fn copy_amps(
        &mut self,
        src: &Self::State,
        src_off: usize,
        dst: &mut Self::State,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError>;
    /// All amplitudes as complex f64 (FP32 widens).
    fn download(&mut self, st: &Self::State) -> Result<Vec<Complex<f64>>, BackendError>;
    /// A fresh `m`-qubit slice holding `amps` (`amps.len() == 2^m`).
    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<Self::State, BackendError>;
}

/// Distributed-run failure.
#[derive(Debug, thiserror::Error)]
pub enum DistSvError {
    #[error(transparent)]
    Dist(#[from] aleph_ir::dist::DistError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}
```

`crates/aleph-cuda/src/dist/exchange.rs` gets a stub for now, so the module compiles. Task 2 fills it.

```rust
//! Rank-slice exchange transports (filled in Task 2).

/// Moves amplitudes between rank slices for a `DistStep::Exchange`.
pub trait Exchange<B> {}

/// All ranks on one device (filled in Task 2).
pub struct LocalExchange;
```

`crates/aleph-cuda/src/dist/device_sv.rs`:

```rust
//! `DeviceSv` for the FP64 and FP32 CUDA state-vector backends. Rank slices
//! are ordinary `CudaSvState` / `CudaSvStateF32` with `num_qubits = m`, so the
//! existing kernels apply unchanged (the same trick `run_paged` uses).

use aleph_backend::BackendError;
use aleph_core::Complex;

use super::DeviceSv;
use crate::sv::to_backend_err;
use crate::{CudaSvBackend, CudaSvBackendF32, CudaSvState, CudaSvStateF32, DeviceBuffer};

fn range_err() -> BackendError {
    BackendError::InvalidState { reason: "dist: amplitude copy out of range" }
}

impl DeviceSv for CudaSvBackend {
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<CudaSvState, BackendError> {
        let ctx = self.ctx();
        if rank == 0 {
            return CudaSvState::allocate(&ctx, m).map_err(to_backend_err);
        }
        let amps = DeviceBuffer::<f64>::zeros(&ctx, 2usize << m).map_err(to_backend_err)?;
        Ok(CudaSvState { num_qubits: m, amps, ctx, mat_scratch: None })
    }

    fn copy_amps(
        &mut self,
        src: &CudaSvState,
        src_off: usize,
        dst: &mut CudaSvState,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        let (s0, d0, l) = (2 * src_off, 2 * dst_off, 2 * len);
        if s0 + l > src.amps.len() || d0 + l > dst.amps.len() {
            return Err(range_err());
        }
        let view = src.amps.slice().slice(s0..s0 + l);
        let mut out = dst.amps.slice_mut().slice_mut(d0..d0 + l);
        self.ctx()
            .stream()
            .memcpy_dtod(&view, &mut out)
            .map_err(|e| to_backend_err(e.into()))
    }

    fn download(&mut self, st: &CudaSvState) -> Result<Vec<Complex<f64>>, BackendError> {
        let host = st.amps.to_vec(&st.ctx).map_err(to_backend_err)?;
        Ok(host.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect())
    }

    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<CudaSvState, BackendError> {
        if amps.len() != 1usize << m {
            return Err(range_err());
        }
        let ctx = self.ctx();
        let flat: Vec<f64> = amps.iter().flat_map(|a| [a.re, a.im]).collect();
        let buf = DeviceBuffer::<f64>::from_slice(&ctx, &flat).map_err(to_backend_err)?;
        Ok(CudaSvState { num_qubits: m, amps: buf, ctx, mat_scratch: None })
    }
}

impl DeviceSv for CudaSvBackendF32 {
    fn alloc_rank(&mut self, m: u32, rank: u32) -> Result<CudaSvStateF32, BackendError> {
        let ctx = self.ctx();
        if rank == 0 {
            return CudaSvStateF32::allocate(&ctx, m).map_err(to_backend_err);
        }
        let amps = DeviceBuffer::<f32>::zeros(&ctx, 2usize << m).map_err(to_backend_err)?;
        Ok(CudaSvStateF32 { num_qubits: m, amps, ctx, mat_scratch: None })
    }

    fn copy_amps(
        &mut self,
        src: &CudaSvStateF32,
        src_off: usize,
        dst: &mut CudaSvStateF32,
        dst_off: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        let (s0, d0, l) = (2 * src_off, 2 * dst_off, 2 * len);
        if s0 + l > src.amps.len() || d0 + l > dst.amps.len() {
            return Err(range_err());
        }
        let view = src.amps.slice().slice(s0..s0 + l);
        let mut out = dst.amps.slice_mut().slice_mut(d0..d0 + l);
        self.ctx()
            .stream()
            .memcpy_dtod(&view, &mut out)
            .map_err(|e| to_backend_err(e.into()))
    }

    fn download(&mut self, st: &CudaSvStateF32) -> Result<Vec<Complex<f64>>, BackendError> {
        let host = st.amps.to_vec(&st.ctx).map_err(to_backend_err)?;
        Ok(host
            .chunks_exact(2)
            .map(|p| Complex::new(f64::from(p[0]), f64::from(p[1])))
            .collect())
    }

    fn upload(&mut self, m: u32, amps: &[Complex<f64>]) -> Result<CudaSvStateF32, BackendError> {
        if amps.len() != 1usize << m {
            return Err(range_err());
        }
        let ctx = self.ctx();
        let flat: Vec<f32> = amps.iter().flat_map(|a| [a.re as f32, a.im as f32]).collect();
        let buf = DeviceBuffer::<f32>::from_slice(&ctx, &flat).map_err(to_backend_err)?;
        Ok(CudaSvStateF32 { num_qubits: m, amps: buf, ctx, mat_scratch: None })
    }
}
```

**Verified on the box (2026-10-03):**
- `sv::backend` and `sv::fp32` are **private** modules (`sv/mod.rs:7,12`). Add `pub(crate) use backend::to_backend_err;`
  to `crates/aleph-cuda/src/sv/mod.rs`. `CudaSvBackendF32`/`CudaSvStateF32` are already re-exported publicly.
- `CudaSvBackendF32::ctx()` exists (`fp32.rs:285`, `pub(crate)`).
- cudarc 0.19.8 `CudaStream::memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(self: &Arc<Self>, src: &Src,
  dst: &mut Dst) -> Result<(), DriverError>` (`core.rs:1657`). It **asserts** `dst.len() >= src.len()` (it panics), so
  the range check in `copy_amps` is load-bearing; keep it.
- Box: `ssh root@openwebgui.splynx.com`; `source ~/.cargo/env`; rustc 1.96; CUDA 13.0; driver 580.178.04; tree at
  `/root/aleph-p6`.
  - A resident `text-embeddings-router` holds 1.3 GiB of GPU memory. Check `nvidia-smi` utilization is 0 % before
    timing.
  - NCCL is **not** installed (needed only for P6-01b).

Remaining adaptation notes. Check each against the real code; any divergence goes in the ledger as a `Ruling:`.
- **Module visibility.** If `sv::backend` / `sv::fp32` are private modules, import through what `sv/mod.rs` exposes. If
  `to_backend_err` is not reachable from `crate::dist`, make it `pub(crate)` at its definition. Do not copy it.
- **FP32 context accessor.** If `CudaSvBackendF32` has no `ctx()` accessor, add a `pub(crate) fn ctx(&self) ->
  CudaContext` mirroring `backend.rs:126`.
- **`DeviceBuffer` methods.** If `DeviceBuffer::len`, `from_slice` or `zeros` have other names or need a `T: …` bound,
  follow `buffer.rs`.
- **`memcpy_dtod` in cudarc 0.19.** Check the signature (`CudaStream::memcpy_dtod`) with
  `cargo doc -p cudarc --open`-free grep: `grep -rn "fn memcpy_dtod" ~/.cargo/registry/src/*/cudarc-0.19*/src`.
  - If the name or argument order differs, adapt.
  - If `e.into()` does not convert `DriverError` into `crate::Error`, use `crate::Error::Driver(e)`.
- **Visibility.** `CudaSvState`'s fields are `pub(crate)`, so this file must live inside `aleph-cuda`, which it does.

In `crates/aleph-cuda/src/lib.rs`, next to the other cfg-gated modules:

```rust
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod dist;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use dist::{DeviceSv, DistSvError, Exchange, LocalExchange};
```

- [ ] **Step 4: Run (box) the tests and lints**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings`
Expected: 2 tests PASS, clippy clean. The stub `Exchange<B>`'s unused parameter may need `PhantomData`, or a trait
with no generic use is fine; fix anything clippy flags.

- [ ] **Step 5: Commit** (on the Mac, after the box run is green)

```bash
git add crates/aleph-cuda/src/dist crates/aleph-cuda/src/lib.rs crates/aleph-cuda/Cargo.toml crates/aleph-cuda/tests/dist_gpu_oracle.rs
git commit -m "[P6-01a] aleph-cuda::dist: DeviceSv for the FP64/FP32 CUDA backends"
```

---

### Task 2: `Exchange` trait + `LocalExchange` (in-place chunk swaps via bounded scratch)

**Files:**
- Modify: `crates/aleph-cuda/src/dist/exchange.rs` (replace the stub)
- Modify: `crates/aleph-cuda/tests/dist_gpu_oracle.rs`

**Interfaces:**
- Consumes: `DeviceSv` (Task 1), `aleph_ir::dist::DistLayout`, `aleph_sv::dist_ref::exchange_cpu` (test reference,
  P6-02).
- Produces:

```rust
pub trait Exchange<B: DeviceSv> {
    fn exchange(&mut self, be: &mut B, ranks: &mut [B::State], layout: DistLayout, global_bits: &[u32])
        -> Result<(), BackendError>;
}
pub struct LocalExchange<B: DeviceSv> { scratch_amps: usize, scratch: Option<B::State>, scratch_len: usize, _b: PhantomData<B> }
impl<B: DeviceSv> LocalExchange<B> {
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;          // 256 MiB of FP64 complex
    pub fn new() -> Self;
    pub fn with_scratch_amps(amps: usize) -> Self;            // amps >= 1, rounded up to a power of two
}
pub(crate) fn chunk_pairs(layout: DistLayout, global_bits: &[u32]) -> Vec<((u32, u32), (u32, u32))>;
```

`chunk_pairs` returns each unordered pair `((r, c), (r', c'))` exactly once, with `(r, c) < (r', c')`, and skips the
fixed points. Rule: chunk `c` of rank `r` (value of the top `k` local bits) maps to rank `r'` = `r` with the chosen
global bits set to `c`, at chunk `c'` = `r`'s old values of those bits. This is the same rule as
`aleph_sv::dist_ref::exchange_cpu`.

- [ ] **Step 1: Write the failing tests** (append to `crates/aleph-cuda/tests/dist_gpu_oracle.rs`)

```rust
use aleph_cuda::{Exchange, LocalExchange};
use aleph_ir::dist::DistLayout;

fn rank_data(l: DistLayout) -> Vec<Vec<Complex<f64>>> {
    let size = 1usize << l.m();
    (0..l.ranks() as usize)
        .map(|r| {
            (0..size)
                .map(|i| Complex::new((r * size + i) as f64, 0.5 * i as f64 - r as f64))
                .collect()
        })
        .collect()
}

#[test]
fn local_exchange_matches_cpu_reference() {
    let Some(mut be) = gpu64() else { return };
    for (n, g) in [(9u32, 2u32), (10, 3)] {
        let l = DistLayout::new(n, g).unwrap();
        let m = l.m();
        let orders: Vec<Vec<u32>> = vec![
            vec![m],
            vec![m + 1],
            vec![m, m + 1],
            vec![m + 1, m],
            if g == 3 { vec![m + 2, m, m + 1] } else { vec![m + 1, m] },
        ];
        for bits in orders {
            for scratch in [4usize, 1 << 20] {
                let host = rank_data(l);
                let mut want = host.clone();
                aleph_sv::dist_ref::exchange_cpu(&mut want, l, &bits);
                let mut ranks: Vec<_> = host.iter().map(|v| be.upload(m, v).unwrap()).collect();
                let mut x = LocalExchange::<CudaSvBackend>::with_scratch_amps(scratch);
                x.exchange(&mut be, &mut ranks, l, &bits).unwrap();
                for (r, st) in ranks.iter().enumerate() {
                    let got = be.download(st).unwrap();
                    assert_eq!(got, want[r], "n={n} g={g} bits={bits:?} scratch={scratch} rank {r}");
                }
            }
        }
    }
}
```

Also add unit tests for `chunk_pairs` inside `exchange.rs`. These compile on the box only, because the module is
cfg-gated:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pairs_cover_each_moved_chunk_once() {
        let l = DistLayout::new(8, 2).unwrap(); // m = 6
        for bits in [vec![6u32], vec![7], vec![6, 7], vec![7, 6]] {
            let k = bits.len() as u32;
            let pairs = chunk_pairs(l, &bits);
            let mut seen = std::collections::HashSet::new();
            for (a, b) in &pairs {
                assert!(a < b);
                assert!(seen.insert(*a) && seen.insert(*b), "dup in {bits:?}");
            }
            // moved chunks = all (r,c) minus fixed points; fixed points per rank = 1
            let total = (l.ranks() as usize) << k;
            assert_eq!(seen.len(), total - l.ranks() as usize, "{bits:?}");
        }
    }
}
```

- [ ] **Step 2: Run (box) to verify it fails**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle local_exchange`
Expected: FAIL to compile (the stub has no `exchange`/`with_scratch_amps`).

- [ ] **Step 3: Write the implementation** (replace `exchange.rs`, keep the tests)

```rust
//! Rank-slice exchange transports.
//!
//! A `DistStep::Exchange { global_bits }` swaps global bit `global_bits[j]`
//! with local bit `m-k+j`. On chunks (the top `k` local bits of a rank) that
//! is an involution `(r, c) ↔ (r', c')`, so every moved chunk pair is swapped
//! exactly once, in place, through a bounded scratch buffer — no second copy
//! of the state, so the reach of a rank slice is not halved.

use std::marker::PhantomData;

use aleph_backend::BackendError;
use aleph_ir::dist::DistLayout;

use super::DeviceSv;

/// Moves amplitudes between rank slices for a `DistStep::Exchange`.
pub trait Exchange<B: DeviceSv> {
    fn exchange(
        &mut self,
        be: &mut B,
        ranks: &mut [B::State],
        layout: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError>;
}

/// Every unordered moved chunk pair `((r, c), (r', c'))` once, `(r,c) < (r',c')`.
pub(crate) fn chunk_pairs(l: DistLayout, global_bits: &[u32]) -> Vec<((u32, u32), (u32, u32))> {
    let m = l.m();
    let k = global_bits.len() as u32;
    let mut out = Vec::new();
    for r in 0..l.ranks() {
        for c in 0..(1u32 << k) {
            let mut r2 = r;
            let mut c2 = 0u32;
            for (j, &gb) in global_bits.iter().enumerate() {
                let rb = gb - m;
                c2 |= ((r >> rb) & 1) << j;
                r2 = (r2 & !(1 << rb)) | (((c >> j) & 1) << rb);
            }
            if (r, c) < (r2, c2) {
                out.push(((r, c), (r2, c2)));
            }
        }
    }
    out
}

/// Disjoint `&mut` to two different ranks.
fn two_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// All ranks on one device: chunk swaps are device-to-device copies through a
/// scratch slice of at most `scratch_amps` amplitudes.
pub struct LocalExchange<B: DeviceSv> {
    scratch_amps: usize,
    scratch: Option<B::State>,
    /// Amplitudes the allocated scratch holds (0 = none yet).
    scratch_len: usize,
    _b: PhantomData<B>,
}

impl<B: DeviceSv> LocalExchange<B> {
    /// 2^24 amplitudes = 256 MiB of FP64 complex scratch.
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;

    pub fn new() -> Self {
        Self::with_scratch_amps(Self::DEFAULT_SCRATCH_AMPS)
    }

    /// Scratch of `amps` amplitudes (rounded up to a power of two, min 1).
    pub fn with_scratch_amps(amps: usize) -> Self {
        Self {
            scratch_amps: amps.max(1).next_power_of_two(),
            scratch: None,
            scratch_len: 0,
            _b: PhantomData,
        }
    }
}

impl<B: DeviceSv> Default for LocalExchange<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: DeviceSv> Exchange<B> for LocalExchange<B> {
    fn exchange(
        &mut self,
        be: &mut B,
        ranks: &mut [B::State],
        l: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError> {
        let m = l.m();
        let k = global_bits.len() as u32;
        if k == 0 || k > m || global_bits.iter().any(|&b| !l.is_global(b) || b >= l.n) {
            return Err(BackendError::InvalidState { reason: "dist: bad exchange bits" });
        }
        let chunk = 1usize << (m - k);
        let piece = chunk.min(self.scratch_amps);
        // A later exchange with smaller k has bigger chunks: grow the scratch
        // when the piece no longer fits (never beyond `scratch_amps`).
        if self.scratch_len < piece {
            self.scratch = Some(be.alloc_rank(piece.trailing_zeros(), 1)?);
            self.scratch_len = piece;
        }
        let Some(scr) = self.scratch.as_mut() else {
            return Err(BackendError::InvalidState { reason: "dist: scratch missing" });
        };
        for ((ra, ca), (rb, cb)) in chunk_pairs(l, global_bits) {
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let (sa, sb) = two_mut(ranks, ra as usize, rb as usize);
            let mut off = 0;
            while off < chunk {
                be.copy_amps(sa, a0 + off, scr, 0, piece)?;
                be.copy_amps(sb, b0 + off, sa, a0 + off, piece)?;
                be.copy_amps(scr, 0, sb, b0 + off, piece)?;
                off += piece;
            }
        }
        Ok(())
    }
}
```

The scratch grows on demand. `piece` is a power of two, at most `scratch_amps`, so `alloc_rank(piece.trailing_zeros(), 1)`
allocates exactly `piece` amplitudes.

- [ ] **Step 4: Run (box)**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle && cargo test -p aleph-cuda --features cuda --lib dist:: && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings`
Expected: all PASS. `local_exchange_matches_cpu_reference` is element-exact (`assert_eq!`), since it is pure data
movement.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-cuda/src/dist/exchange.rs crates/aleph-cuda/tests/dist_gpu_oracle.rs
git commit -m "[P6-01a] LocalExchange: in-place chunk-pair swaps through bounded scratch"
```

---

### Task 3: `DistSvBackend` (plan execution, per-rank fusion, readout) + GPU oracle

**Files:**
- Modify: `crates/aleph-cuda/src/dist/mod.rs`
- Modify: `crates/aleph-cuda/src/lib.rs` (also export `DistSvBackend`, `DistSvState`)
- Modify: `crates/aleph-cuda/tests/dist_gpu_oracle.rs`

**Interfaces:**
- Consumes:
  - `aleph_ir::dist::{plan, specialize, DistPlan, DistStep, DistLayout, Router}`;
  - `crate::fuse_for_gpu(&Circuit) -> Circuit` (`fusion.rs:70`);
  - `Backend::{apply_gate, apply_diagonal_phase, apply_tiled_block}`;
  - `DeviceSv`, `Exchange`.
- Produces:

```rust
pub struct DistSvState<B: DeviceSv> { pub layout: DistLayout, pub final_map: Vec<u32>, ranks: Vec<B::State> }
pub struct DistSvBackend<B: DeviceSv, X: Exchange<B>> { be: B, x: X, fuse: bool }
impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    pub fn new(be: B, x: X) -> Self;                  // fuse = true
    pub fn with_fusion(self, fuse: bool) -> Self;
    pub fn run_plan(&mut self, plan: &DistPlan) -> Result<DistSvState<B>, DistSvError>;
    pub fn run(&mut self, c: &Circuit, g: u32, router: Router) -> Result<DistSvState<B>, DistSvError>;
    pub fn amplitudes(&mut self, st: &DistSvState<B>) -> Result<Vec<Complex<f64>>, DistSvError>; // logical order
    pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError>;
}
```

- [ ] **Step 1: Write the failing oracle tests** (append to `dist_gpu_oracle.rs`)

```rust
use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::DistSvBackend;
use aleph_ir::dist::Router;
use aleph_ir::{Circuit, Instruction};
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;

fn reference(c: &Circuit) -> Vec<Complex<f64>> {
    let mut b = NaiveSvBackend::with_seed(0);
    run(&mut b, c).unwrap().amplitudes().to_vec()
}

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / f64::from(1u32 << (j - k));
            c.add_gate(GateInstance::controlled(Gate::Phase(Param::Concrete(th)), vec![j], vec![k]))
                .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64 + 0.05 * f64::from(q), q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

fn grover8() -> Circuit {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm"
    ))
    .unwrap();
    let parsed = aleph_parser::parse(&src).unwrap();
    let mut c = Circuit::new(parsed.num_qubits(), 0);
    for i in parsed.instructions() {
        if let Instruction::Gate(g) = i {
            c.add_gate(g.clone()).unwrap();
        }
    }
    c
}

fn all_diag_on_globals(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let top = n - 1;
    c.rz(0.9, top).unwrap();
    c.add_gate(GateInstance::new(Gate::Cz, vec![top - 1, top])).unwrap();
    c.add_gate(GateInstance::new(Gate::CRz(Param::Concrete(1.1)), vec![0, top])).unwrap();
    c.add_gate(GateInstance::new(Gate::Ccz, vec![top - 1, 1, top])).unwrap();
    c.add_gate(GateInstance::controlled(Gate::T, vec![2u32], vec![top])).unwrap();
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![top, top - 1, 0])).unwrap();
    c
}

fn cases() -> Vec<(&'static str, Circuit)> {
    vec![
        ("ghz10", ghz(10)),
        ("qft10", qft(10)),
        ("brick10", brickwall(10, 6)),
        ("grover8", grover8()),
        ("diag10", all_diag_on_globals(10)),
    ]
}

#[test]
fn dist_f64_matches_oracle_all_layouts_routers_fusion() {
    let Some(be) = gpu64() else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 0..=3u32 {
            for router in [Router::Naive, Router::Lookahead] {
                for fuse in [false, true] {
                    d = d.with_fusion(fuse);
                    let st = d.run(&c, g, router).unwrap();
                    let got = d.amplitudes(&st).unwrap();
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (x - y).norm() < 1e-10,
                            "{name} g={g} {router:?} fuse={fuse} amp {i}: {x} vs {y}"
                        );
                    }
                    assert!((d.norm_sqr(&st).unwrap() - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn dist_f32_matches_oracle() {
    let Some(be) = gpu32() else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in cases() {
        let want = reference(&c);
        for g in [1u32, 2, 3] {
            let st = d.run(&c, g, Router::Lookahead).unwrap();
            let got = d.amplitudes(&st).unwrap();
            for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                assert!((x - y).norm() < 1e-5, "{name} g={g} amp {i}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn dist_matches_single_gpu_at_n20() {
    // Larger slices with the default scratch: equal to single-GPU CudaSvBackend.
    let Some(be) = gpu64() else { return };
    let Some(mut single) = gpu64() else { return };
    let c = brickwall(20, 8);
    let want = run(&mut single, &c).unwrap().amplitudes_vec();
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    for g in [1u32, 2, 3] {
        let st = d.run(&c, g, Router::Lookahead).unwrap();
        let got = d.amplitudes(&st).unwrap();
        let worst = got.iter().zip(&want).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
        assert!(worst < 1e-10, "g={g} worst {worst}");
    }
}
```

`NaiveSvBackend::with_seed` and `HasAmplitudes` follow `tests/paged_oracle.rs`. If `amplitudes()` returns a different
type there, mirror that file exactly.

- [ ] **Step 2: Run (box) to verify it fails**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle dist_`
Expected: FAIL to compile (`DistSvBackend` not found).

- [ ] **Step 3: Write the implementation** (append to `crates/aleph-cuda/src/dist/mod.rs`)

```rust
use aleph_ir::dist::{plan as dist_plan, specialize, DistLayout, DistPlan, DistStep, Router};
use aleph_ir::{Circuit, Instruction};

/// A distributed state: `2^g` rank slices plus the plan's final
/// logical→physical map (needed to read amplitudes in logical order).
pub struct DistSvState<B: DeviceSv> {
    pub layout: DistLayout,
    pub final_map: Vec<u32>,
    ranks: Vec<B::State>,
}

/// Executes `DistPlan`s on `B` (one device) with exchange transport `X`.
pub struct DistSvBackend<B: DeviceSv, X: Exchange<B>> {
    be: B,
    x: X,
    fuse: bool,
}

impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    /// Per-rank fusion on (`fuse_for_gpu`).
    pub fn new(be: B, x: X) -> Self {
        Self { be, x, fuse: true }
    }

    /// Toggle per-rank fusion (off = apply specialised gates one by one).
    pub fn with_fusion(mut self, fuse: bool) -> Self {
        self.fuse = fuse;
        self
    }

    /// Plan `c` over `2^g` ranks with `router`, then execute.
    pub fn run(&mut self, c: &Circuit, g: u32, router: Router) -> Result<DistSvState<B>, DistSvError> {
        let layout = DistLayout::new(c.num_qubits(), g)?;
        let p = dist_plan(c, layout, router)?;
        self.run_plan(&p)
    }

    /// Execute a plan from |0…0⟩.
    pub fn run_plan(&mut self, p: &DistPlan) -> Result<DistSvState<B>, DistSvError> {
        let l = p.layout;
        let m = l.m();
        let mut ranks = Vec::with_capacity(l.ranks() as usize);
        for r in 0..l.ranks() {
            ranks.push(self.be.alloc_rank(m, r)?);
        }
        for step in &p.steps {
            match step {
                DistStep::Local(instrs) => {
                    for (r, st) in ranks.iter_mut().enumerate() {
                        let local = self.rank_program(instrs, l, r as u32)?;
                        for i in local.instructions() {
                            apply_one(&mut self.be, st, i)?;
                        }
                    }
                }
                DistStep::Exchange { global_bits } => {
                    self.x.exchange(&mut self.be, &mut ranks, l, global_bits)?;
                }
            }
        }
        Ok(DistSvState { layout: l, final_map: p.final_map.clone(), ranks })
    }

    /// Rank `r`'s `m`-qubit program for one `Local` step: specialised, then
    /// (optionally) fused — fusion runs *after* `specialize` so it never sees
    /// a global qubit (spec §3.1).
    fn rank_program(&self, instrs: &[Instruction], l: DistLayout, r: u32) -> Result<Circuit, DistSvError> {
        let mut c = Circuit::new(l.m(), 0);
        for i in instrs {
            if let Some(s) = specialize(i, l, r)? {
                c.add_instruction(s).map_err(|_| BackendError::InvalidState {
                    reason: "dist: specialised instruction rejected by Circuit",
                })?;
            }
        }
        Ok(if self.fuse { crate::fuse_for_gpu(&c) } else { c })
    }

    /// Full state in **logical** qubit order (host gather; tests / small n).
    pub fn amplitudes(&mut self, st: &DistSvState<B>) -> Result<Vec<Complex<f64>>, DistSvError> {
        let m = st.layout.m();
        let size = 1usize << m;
        let mut phys: Vec<Complex<f64>> = Vec::with_capacity(size << st.layout.g);
        for r in &st.ranks {
            phys.extend(self.be.download(r)?);
        }
        let mut out = vec![Complex::new(0.0, 0.0); phys.len()];
        for (x, slot) in out.iter_mut().enumerate() {
            let mut p = 0usize;
            for (lq, &pq) in st.final_map.iter().enumerate() {
                p |= ((x >> lq) & 1) << pq;
            }
            *slot = phys[p];
        }
        Ok(out)
    }

    /// Σ|a|² over all ranks (host reduction of per-rank downloads).
    pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError> {
        let mut s = 0.0;
        for r in &st.ranks {
            s += self.be.download(r)?.iter().map(|a| a.norm_sqr()).sum::<f64>();
        }
        Ok(s)
    }
}

fn apply_one<B: DeviceSv>(be: &mut B, st: &mut B::State, i: &Instruction) -> Result<(), DistSvError> {
    match i {
        Instruction::Gate(g) => be.apply_gate(st, g)?,
        Instruction::DiagonalPhase(dp) => be.apply_diagonal_phase(st, dp)?,
        Instruction::TiledBlock(tb) => be.apply_tiled_block(st, tb)?,
        Instruction::Barrier(_) => {}
        Instruction::Measure { .. } | Instruction::Reset(_) => {
            return Err(BackendError::UnsupportedInstruction { kind: "measure/reset in dist" }.into())
        }
    }
    Ok(())
}
```

Export it: in `lib.rs`, make the dist `pub use` read
`pub use dist::{DeviceSv, DistSvBackend, DistSvError, DistSvState, Exchange, LocalExchange};`.

`norm_sqr` downloads the state. This is acceptable in v1 per spec §3.4 ("local reduction then host sum"). Note it in
the PR follow-ups: a device-side per-rank reduction exists for FP32 (`norm_sqr()`), and FP64 can get one later. Do not
add it now.

- [ ] **Step 4: Run (box)**

Run (box): `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle 2>&1 | tail -20 && cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings`
Expected: all PASS, clippy clean. Then the mutation proof. In `LocalExchange::exchange`, temporarily swap the middle
copy's destination offset (`a0 + off` → `b0 + off`), and re-run `dist_f64_matches_oracle_all_layouts_routers_fusion`.
Expected: FAIL. Revert and re-run. Expected: PASS. Record this in the ledger.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-cuda/src/dist/mod.rs crates/aleph-cuda/src/lib.rs crates/aleph-cuda/tests/dist_gpu_oracle.rs
git commit -m "[P6-01a] DistSvBackend: run DistPlans on one GPU with per-rank fusion"
```

---

### Task 4: overhead benchmark, report, PR

**Files:**
- Create: `crates/aleph-cuda/tests/dist_local_bench.rs`
- Create: `docs/perf/p6-01a-dist-gpu.md`

- [ ] **Step 1: Write the bench** (`#[ignore]`, crate convention)

```rust
//! P6-01a overhead: DistSvBackend + LocalExchange (R ranks on ONE GPU) vs the
//! single-GPU CudaSvBackend on the same circuit. On one card an exchange is a
//! device-to-device copy, so this measures the distributed machinery's cost
//! (specialise + per-rank launches + chunk swaps), not interconnect speed.
//! Run: cargo test --release -p aleph-cuda --features cuda --test dist_local_bench -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::{fuse_for_gpu, CudaSvBackend, CudaSvBackendF32, DistSvBackend, LocalExchange};
use aleph_ir::dist::Router;
use aleph_ir::Circuit;

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / 2f64.powi((j - k) as i32);
            c.add_gate(GateInstance::controlled(Gate::Phase(Param::Concrete(th)), vec![j], vec![k]))
                .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64, q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore]
fn dist_local_overhead() {
    let Ok(mut single) = CudaSvBackend::with_seed(0) else { return };
    let Ok(be) = CudaSvBackend::with_seed(0) else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    println!("| precision | circuit | n | single-GPU (s) | R=2 (s) | R=4 (s) | R=2 / single | R=4 / single |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, c) in [("QFT", qft(28)), ("random d=10", brickwall(28, 10))] {
        let fused = fuse_for_gpu(&c);
        let t1 = best_of(3, || {
            let s = run(&mut single, &fused).unwrap();
            let _ = s.amplitudes_vec().len(); // forces completion
        });
        let mut ts = Vec::new();
        for g in [1u32, 2] {
            ts.push(best_of(3, || {
                let st = d.run(&c, g, Router::Lookahead).unwrap();
                let _ = d.norm_sqr(&st).unwrap(); // forces completion
            }));
        }
        println!(
            "| FP64 | {name} | 28 | {t1:.3} | {:.3} | {:.3} | {:.2}× | {:.2}× |",
            ts[0], ts[1], ts[0] / t1, ts[1] / t1
        );
    }
    let Ok(mut single32) = CudaSvBackendF32::with_seed(0) else { return };
    let Ok(be32) = CudaSvBackendF32::with_seed(0) else { return };
    let mut d32 = DistSvBackend::new(be32, LocalExchange::new());
    for (name, c) in [("QFT", qft(29)), ("random d=10", brickwall(29, 10))] {
        let fused = fuse_for_gpu(&c);
        let t1 = best_of(3, || {
            let s = run(&mut single32, &fused).unwrap();
            let _ = s.norm_sqr();
        });
        let mut ts = Vec::new();
        for g in [1u32, 2] {
            ts.push(best_of(3, || {
                let st = d32.run(&c, g, Router::Lookahead).unwrap();
                let _ = d32.norm_sqr(&st).unwrap();
            }));
        }
        println!(
            "| FP32 | {name} | 29 | {t1:.3} | {:.3} | {:.3} | {:.2}× | {:.2}× |",
            ts[0], ts[1], ts[0] / t1, ts[1] / t1
        );
    }
}
```

**Timing caveat for the doc.** Both arms pay a host readback to force completion: `amplitudes_vec` or `norm_sqr` /
`norm_sqr()`. Their sizes differ (a full download for FP64 single, versus the per-rank downloads in `norm_sqr`). Use
the same completion primitive for both arms where one exists. If `CudaSvState` has a synchronising method cheaper than
a full download (e.g. `ctx.synchronize()` via a crate-internal accessor), use it in both arms. Record what was used.
Adjust the code accordingly and ledger the choice. The memory budget at n=28 FP64 is 4 GiB, plus 4 GiB for `single`,
plus scratch, which fits in 20 GiB. At n=29 FP32 it is 4 + 4 GiB, which also fits.

- [ ] **Step 2: Run (box) on an idle box**

```bash
ssh root@openwebgui.splynx.com 'uptime; pgrep -af "cargo bench|cargo test|bencher|Runner.Worker" || echo idle; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
# then (box): cargo test --release -p aleph-cuda --features cuda --test dist_local_bench -- --ignored --nocapture
```

Expected: a markdown table. R=2/4 cost more than single-GPU, because exchanges on one card are extra D2D copies and
there are R× more kernel launches. Expect roughly 1.1–2×. A ratio above 3× means something is wrong: per-step
synchronisation, or re-specialising huge lists. If so, investigate with `superpowers:systematic-debugging` before
writing the doc.

- [ ] **Step 3: Write `docs/perf/p6-01a-dist-gpu.md`**

```markdown
# P6-01a — Distributed SV on one GPU (DistSvBackend + LocalExchange)

Refs #55. Design: `docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md` §3.3. Builds on P6-02 (#56) and P6-03 (#57).

## What it is
<3–5 bullets: DeviceSv over the existing FP64/FP32 kernels (rank slice = CudaSvState with num_qubits=m);
per-rank specialise → fuse_for_gpu → apply; LocalExchange in-place chunk-pair swaps via bounded scratch
(no double buffer); host gather via final_map; NCCL is P6-01b.>

## Correctness (RTX 4000 SFF Ada)
<list exactly what dist_gpu_oracle covers and the mutation that proved it bites>

## Overhead vs single GPU (one card, so exchange = D2D copy)
<table from the bench, box state (uptime/load), what forced completion>

## Reading
<2–4 sentences: where the overhead comes from, why this is NOT the multi-GPU number, what the
AWS 4×L4 session will measure instead.>
```

Fill in every bracket from real output. No placeholders in the committed file.

- [ ] **Step 4: Full verification (box + Mac)**

Box:
```bash
cargo fmt --check
cargo clippy -p aleph-cuda --features cuda --all-targets -- -D warnings
cargo test -p aleph-cuda --features cuda
```

Mac:
```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p aleph-ir -p aleph-sv
```

Expected: all clean and green. On the box, `cargo test -p aleph-cuda --features cuda` runs the whole crate's tests,
including pre-existing ones; report its totals.

- [ ] **Step 5: Commit, push (after #527 and P6-03 have merged and this branch is rebased), PR**

```bash
git add crates/aleph-cuda/tests/dist_local_bench.rs docs/perf/p6-01a-dist-gpu.md
git commit -m "[P6-01a] Single-GPU overhead bench + report"
```

PR body: `Refs #55`, a summary, oracle results with counts, the overhead table with box conditions, notes, and
follow-ups:
- P6-01b `NcclExchange` plus a `Vec<B>` per device;
- a device-side FP64 norm;
- the AWS 4×L4 session;
- the 2-local diagonal emitted as dense `Unitary2q`, deferred from P6-02 review. Check whether `fuse_for_gpu` already
  turns it back into a diagonal; if not, file it.

End with the attribution lines from the session's system reminder.
