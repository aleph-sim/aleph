# P6-01b NcclExchange + multi-device DistSvBackend: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `DistSvBackend` spread its ranks over several CUDA devices and swap chunks between devices with NCCL. Prove
on the single RTX 4000 that everything except the real cross-GPU transfer works, so the AWS session (4×L4) only has to
measure.

**Architecture:** `DistSvBackend` holds a `Vec<B>`, one backend per device. The `R = 2^g` ranks go to devices in
contiguous blocks, so the high rank bits name the device. The `Exchange` trait now takes `&mut [B]`. `LocalExchange`
does any pair whose two devices share a CUDA ordinal, so on one GPU it also covers "several backends, one card".
`NcclExchange` (feature `nccl`) groups the cross-device chunk pieces into rounds that fit a per-device scratch slice.
Each round is a single `ncclGroupStart/End` of send/recv into scratch, followed by local scratch→slice copies. That
keeps the exchange in place with no double buffer, as spec §3.3 asks. Norm and single-qubit probability become
per-rank device reductions with a host sum over `R` scalars, which closes the spec §3.4 deviation.

**Tech Stack:** Rust 2021, MSRV 1.89, cudarc 0.19.8 (`driver`, `nvrtc`, `dynamic-loading`, plus `nccl` → NCCL 2.30
bindings), libnccl2 2.32 (CUDA apt repo) on the box and on AWS.

**Spec:** `docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md` (§3.3, §3.4, §5, §6, §7 item 4, §8).

## Global Constraints

- One process drives all GPUs. There is no MPI (spec §3.3).
- Exchange memory is a fixed per-device scratch with **no double buffer**. The default is `1 << 24` amplitudes
  (256 MiB of FP64 complex) (spec §3.3).
- NCCL comes in through `cudarc` feature `nccl` and is dynamically loaded. No new crate. CI builds without NCCL
  installed (spec §3.3).
- No `panic` on user input. Everything goes through `BackendError` / `DistSvError` (spec §4, CLAUDE.md).
- No `unwrap()`/`expect()` in library code. `unsafe` needs a SAFETY comment.
- FP64 oracle tolerance is 1e-10. FP32 is 1e-5 (CLAUDE.md, spec §5).
- `cargo clippy --workspace --all-targets -- -D warnings`, the same with `-p aleph-cuda --features cuda` and
  `--features nccl`, and `cargo fmt --check` must pass. CI clippy runs **beta**: run `cargo +beta clippy` before
  pushing.
- No git worktrees. Branch `p6-01b-nccl` from `origin/main` (after #530 merges) in `/Users/ex/GitHub/aleph`.
- The box (`ssh root@openwebgui.splynx.com`):
  - run `source ~/.cargo/env` first;
  - the tree is `/root/aleph-p6`, synced with
    `rsync -a --delete --exclude target --exclude .git --exclude .superpowers ./ root@openwebgui.splynx.com:/root/aleph-p6/`;
  - never commit from the box;
  - the embeddings container stays on CPU (user decision 2026-10-04).
- PR title `[P6-01b] NcclExchange + multi-device DistSvBackend`. The body says `Refs #55` and **does not close it**.

## Review Focus

1. **NCCL library absent** (CI runner, a fresh box). `NcclExchange::new` must return `Err`, not panic out of cudarc's
   dlopen. Pinned by the Task 4 test `nccl_new_without_lib_is_err_not_panic`.
2. **Two backends on the same GPU given to NCCL.** NCCL rejects duplicate GPUs with an opaque error. `new` must reject
   up front with a clear reason. Pinned by the Task 4 test `nccl_rejects_duplicate_ordinals`.
3. **More devices than ranks, or a non-power-of-two device count** (`R=2`, `D=4`; or `D=3`). This must be an error,
   not an index panic in the rank→device map. Pinned by the Task 3 test `multi_rejects_bad_device_counts`.
4. **A scratch smaller than one chunk, and odd interleavings of rounds** (many pieces per pair, pairs of different
   devices in one round). This must not corrupt data. Pinned by the Task 4 scheduler property test and by the
   `scratch_amps = 8` GPU oracle.
5. **Probability of a qubit that ends up global** after relabelling (`final_map[q] >= m`). This must sum whole ranks,
   not read a local bit. Pinned by the Task 3 test `prob_one_matches_reference_for_every_qubit` (QFT at g=2, whose final
   map moves qubits across the boundary).

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-cuda/src/sv/backend.rs`, `sv/fp32.rs` | `on_device(ordinal)`, `device_count()`, `raw_branch` (no normalisation check) |
| `crates/aleph-cuda/src/sv/readout.rs`, `sv/readout_f32.rs` | `reduce_branch` → `pub(crate)` |
| `crates/aleph-cuda/src/dist/mod.rs` | `DeviceSv` gains `Scalar`, `ordinal`, `stream`, `amps_view(_mut)`, `branch_norms`; `DistSvBackend` multi-device; device norm / `prob_one`; `final_map` check |
| `crates/aleph-cuda/src/dist/device_sv.rs` | impls of the new `DeviceSv` items for FP64/FP32 |
| `crates/aleph-cuda/src/dist/exchange.rs` | reshaped `Exchange` trait; `rank_device`; per-device-scratch `LocalExchange` |
| `crates/aleph-cuda/src/dist/schedule.rs` (new) | pure piece/round scheduler shared by the NCCL path |
| `crates/aleph-cuda/src/dist/nccl.rs` (new, `cfg(feature = "nccl")`) | `NcclExchange` |
| `crates/aleph-cuda/Cargo.toml` | feature `nccl = ["cuda", "cudarc?/nccl"]` |
| `.github/workflows/ci.yml` | clippy + build with `--features nccl` |
| `crates/aleph-cuda/tests/common/dist.rs` (new) | circuit helpers + CPU reference moved out of `dist_gpu_oracle.rs` |
| `crates/aleph-cuda/tests/dist_gpu_oracle.rs` | uses `common::dist`; multi-backend + readout tests |
| `crates/aleph-cuda/tests/dist_nccl_oracle.rs` (new) | NCCL oracle: D=1 self-routing on the box, D=2/4 real on AWS |
| `crates/aleph-cuda/tests/dist_nccl_bench.rs` (new, ignored) | p2p bandwidth, strong/weak scaling, model prediction |
| `docs/perf/p6-multi-gpu.md` (new) | comm counts, the time model, the **predicted** AWS table (measured columns empty) |
| `scripts/aws/p6-multi-gpu-session.sh` (new) | AWS runbook (not run in this PR) |

---

### Task 1: Backends on any device

**Files:**
- Modify: `crates/aleph-cuda/src/sv/backend.rs` (the `new` / `with_seed` / `build` block, ~lines 75–125)
- Modify: `crates/aleph-cuda/src/sv/fp32.rs` (the same block, ~lines 190–240)
- Test: `crates/aleph-cuda/tests/dist_gpu_oracle.rs`

**Interfaces:**
- Produces:
  - `CudaSvBackend::on_device(ordinal: usize) -> Result<Self, Error>`;
  - `CudaSvBackendF32::on_device(ordinal: usize) -> Result<Self, Error>`;
  - `aleph_cuda::device_count() -> Result<usize, Error>` (in `context.rs`, re-exported from `lib.rs`).

- [ ] **Step 1: Write the failing test** (append to `dist_gpu_oracle.rs`)

```rust
#[test]
fn backends_open_on_explicit_ordinal() {
    let Ok(n) = aleph_cuda::device_count() else { return };
    if n == 0 {
        return;
    }
    assert!(CudaSvBackend::on_device(0).is_ok());
    assert!(CudaSvBackendF32::on_device(0).is_ok());
    // One past the last device: an error, never a panic.
    assert!(CudaSvBackend::on_device(n).is_err());
    assert!(CudaSvBackendF32::on_device(n).is_err());
}
```

- [ ] **Step 2: Run it on the box and confirm it fails to compile**

Run: `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle backends_open -- --nocapture`
Expected: a compile error naming `on_device` / `device_count`.

- [ ] **Step 3: Implement**

`context.rs`:

```rust
/// Number of CUDA devices the driver reports (0 on a GPU-less host).
pub fn device_count() -> Result<usize, Error> {
    match RawContext::device_count() {
        Ok(n) => Ok(usize::try_from(n).unwrap_or(0)),
        Err(e) => Err(classify_init_error(e, 0)),
    }
}
```

In `backend.rs`, thread the ordinal through `build` (do the same in `fp32.rs`):

```rust
pub fn new() -> Result<Self, Error> {
    Self::build(StdRng::from_entropy(), 0)
}

pub fn with_seed(seed: u64) -> Result<Self, Error> {
    Self::build(StdRng::seed_from_u64(seed), 0)
}

/// Construct on device `ordinal` (entropy-seeded). A missing ordinal is
/// [`Error::NoDevice`]. Multi-GPU `DistSvBackend` builds one per device.
pub fn on_device(ordinal: usize) -> Result<Self, Error> {
    Self::build(StdRng::from_entropy(), ordinal)
}

fn build(rng: StdRng, ordinal: usize) -> Result<Self, Error> {
    let ctx = CudaContext::new(ordinal)?;
    // … rest unchanged …
```

Re-export `device_count` next to `CudaContext` in `lib.rs`. If `RawContext::device_count` has a different name in
cudarc 0.19.8, run `grep -n "fn device_count" $CUDARC/src/driver/safe/core.rs` on the box and use that.

- [ ] **Step 4: Run it and confirm it passes**

Run on the box: the same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-cuda/src
git commit -m "[P6-01b] CUDA SV backends constructible on any device ordinal"
```

---

### Task 2: DeviceSv grows device identity, raw views and raw reductions

**Files:**
- Modify: `crates/aleph-cuda/src/dist/mod.rs` (the `DeviceSv` trait)
- Modify: `crates/aleph-cuda/src/dist/device_sv.rs`
- Modify: `crates/aleph-cuda/src/sv/readout.rs:131`, `sv/readout_f32.rs:129` (`reduce_branch` → `pub(crate)`)
- Modify: `crates/aleph-cuda/src/sv/backend.rs`, `sv/fp32.rs` (`raw_branch`)
- Test: `crates/aleph-cuda/tests/dist_gpu_oracle.rs`

**Interfaces:**
- Consumes: `on_device` (Task 1).
- Produces these new `DeviceSv` items:

```rust
/// Device scalar of the interleaved (re, im) amplitude buffer: f64 or f32.
type Scalar: cudarc::driver::DeviceRepr + Copy + 'static;
/// CUDA ordinal this backend launches on.
fn ordinal(&self) -> usize;
/// The stream every kernel and copy of this backend is ordered on (NCCL comms bind to it).
fn stream(&self) -> std::sync::Arc<cudarc::driver::CudaStream>;
/// Interleaved scalars `[2*off, 2*(off+len))` of `st` (bounds-checked).
fn amps_view(st: &Self::State, off: usize, len: usize)
    -> Result<cudarc::driver::CudaView<'_, Self::Scalar>, BackendError>;
fn amps_view_mut(st: &mut Self::State, off: usize, len: usize)
    -> Result<cudarc::driver::CudaViewMut<'_, Self::Scalar>, BackendError>;
/// `(Σ|a|², Σ_{i & qbit ≠ 0} |a|²)` on the device, with no normalisation check
/// (rank slices are not normalised). `qbit = 0` gives the total alone.
fn branch_norms(&mut self, st: &Self::State, qbit: u64) -> Result<(f64, f64), BackendError>;
```

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn device_sv_branch_norms_and_views() {
    let Some(mut be) = gpu64() else { return };
    // amps[i] = (i+1)·0.25 for i < 16
    let amps = ramp(16, 0.25);
    let st = be.upload(4, &amps).unwrap();
    let tot: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    let p1: f64 = amps.iter().enumerate().filter(|(i, _)| i & 4 != 0).map(|(_, a)| a.norm_sqr()).sum();
    let (t, b) = be.branch_norms(&st, 4).unwrap();
    assert!((t - tot).abs() < 1e-9 && (b - p1).abs() < 1e-9);
    assert_eq!(be.ordinal(), 0);
    assert_eq!(CudaSvBackend::amps_view(&st, 8, 8).unwrap().len(), 16);
    assert!(CudaSvBackend::amps_view(&st, 9, 8).is_err());
    let Some(mut be32) = gpu32() else { return };
    let s32 = be32.upload(4, &amps).unwrap();
    let (t32, b32) = be32.branch_norms(&s32, 4).unwrap();
    assert!((t32 - tot).abs() < 1e-3 && (b32 - p1).abs() < 1e-3);
}
```

- [ ] **Step 2: Run it on the box and confirm it fails to compile.**

Run: `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle device_sv_branch`

- [ ] **Step 3: Implement**

`readout.rs` / `readout_f32.rs`: change `fn reduce_branch` to `pub(crate) fn reduce_branch`.

`backend.rs` (mirror it in `fp32.rs` with `CudaSvStateF32`):

```rust
/// Device `(Σ|a|², Σ_{i&qbit≠0}|a|²)` without the normalisation check, for
/// distributed rank slices.
pub(crate) fn raw_branch(&mut self, st: &CudaSvState, qbit: u64) -> Result<(f64, f64), BackendError> {
    self.readout.reduce_branch(st, qbit).map_err(to_backend_err)
}
```

`device_sv.rs`, inside `impl DeviceSv for CudaSvBackend` (FP32 is the same with `f32` / `CudaSvStateF32`):

```rust
type Scalar = f64;

fn ordinal(&self) -> usize {
    self.ctx().raw().ordinal()
}

fn stream(&self) -> Arc<CudaStream> {
    self.ctx().stream().clone()
}

fn amps_view(st: &CudaSvState, off: usize, len: usize) -> Result<CudaView<'_, f64>, BackendError> {
    let (s0, l) = (off.checked_mul(2).ok_or_else(range_err)?, len.checked_mul(2).ok_or_else(range_err)?);
    if s0.checked_add(l).is_none_or(|e| e > st.amps.len()) {
        return Err(range_err());
    }
    Ok(st.amps.slice().slice(s0..s0 + l))
}

fn amps_view_mut(st: &mut CudaSvState, off: usize, len: usize) -> Result<CudaViewMut<'_, f64>, BackendError> {
    let (s0, l) = (off.checked_mul(2).ok_or_else(range_err)?, len.checked_mul(2).ok_or_else(range_err)?);
    if s0.checked_add(l).is_none_or(|e| e > st.amps.len()) {
        return Err(range_err());
    }
    Ok(st.amps.slice_mut().slice_mut(s0..s0 + l))
}

fn branch_norms(&mut self, st: &CudaSvState, qbit: u64) -> Result<(f64, f64), BackendError> {
    self.raw_branch(st, qbit)
}
```

Rewrite `copy_amps` on top of `amps_view` / `amps_view_mut`, so there is one bounds check (overflow-safe through
`checked_*`). `is_none_or` is stable since Rust 1.82.

- [ ] **Step 4: Run the test plus the whole `dist_gpu_oracle` on the box. Expected: all PASS.**

- [ ] **Step 5: Commit** `[P6-01b] DeviceSv: ordinal, stream, raw amplitude views, device branch norms`.

---

### Task 3: Multi-device DistSvBackend, per-device LocalExchange, device readout

**Files:**
- Create: `crates/aleph-cuda/tests/common/dist.rs`. Move `reference`, `ghz`, `qft`, `brickwall`, `grover8`,
  `all_diag_on_globals` and `cases` here verbatim from `dist_gpu_oracle.rs`, all `pub fn`.
- Modify: `crates/aleph-cuda/tests/dist_gpu_oracle.rs` (`mod common; use common::dist::*;`)
- Modify: `crates/aleph-cuda/src/dist/exchange.rs`, `crates/aleph-cuda/src/dist/mod.rs`
- Modify: `crates/aleph-cuda/tests/dist_local_bench.rs` (call sites only, if the signature change touches them)

**Interfaces:**
- Consumes: the Task 2 `DeviceSv` items.
- Produces:

```rust
// exchange.rs
/// Device index of rank `r` when `ranks` ranks sit on `devs` devices in
/// contiguous blocks (`devs` | `ranks`, both powers of two, checked by the caller).
pub(crate) fn rank_device(r: u32, ranks: u32, devs: usize) -> usize;
pub(crate) fn chunk_pairs(..)   // unchanged
pub(crate) fn valid_bits(..)    // now pub(crate), reused by nccl.rs
pub trait Exchange<B: DeviceSv> {
    fn exchange(&mut self, devs: &mut [B], ranks: &mut [B::State],
                layout: DistLayout, global_bits: &[u32]) -> Result<(), BackendError>;
}

// mod.rs
impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    pub fn new(be: B, x: X) -> Self;                                   // one device (unchanged API)
    pub fn multi(devs: Vec<B>, x: X) -> Result<Self, DistSvError>;     // D = devs.len(), power of two, ≥ 1
    pub fn devices(&self) -> usize;
    pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError>;          // now device-side
    pub fn prob_one(&mut self, st: &DistSvState<B>, q: u32) -> Result<f64, DistSvError>; // logical qubit q
}
```

- [ ] **Step 1: Move the helpers to `tests/common/dist.rs` and run `dist_gpu_oracle` on the box. Expected: same 9/9
  PASS (a pure move).**

- [ ] **Step 2: Write the failing tests** (append to `dist_gpu_oracle.rs`)

```rust
/// D backends that all live on GPU 0: exercises rank placement and per-device
/// scratch on one card (LocalExchange pairs by *ordinal*, so cross-"device"
/// pairs on the same GPU are plain D2D copies).
fn same_gpu_devs(d: usize) -> Option<Vec<CudaSvBackend>> {
    (0..d).map(|_| CudaSvBackend::on_device(0).ok()).collect()
}

#[test]
fn multi_backend_same_gpu_matches_oracle() {
    for d in [2usize, 4] {
        let Some(devs) = same_gpu_devs(d) else { return };
        let mut db = DistSvBackend::multi(devs, LocalExchange::with_scratch_amps(8)).unwrap();
        for (name, c) in cases() {
            let want = reference(&c);
            for g in (d.trailing_zeros())..=3u32 {
                for router in [Router::Naive, Router::Lookahead] {
                    let st = db.run(&c, g, router).unwrap();
                    let got = db.amplitudes(&st).unwrap();
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!((x - y).norm() < 1e-10, "{name} D={d} g={g} {router:?} amp {i}");
                    }
                    assert!((db.norm_sqr(&st).unwrap() - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn multi_rejects_bad_device_counts() {
    let Some(devs) = same_gpu_devs(3) else { return };
    assert!(DistSvBackend::multi(devs, LocalExchange::new()).is_err()); // D = 3
    assert!(DistSvBackend::<CudaSvBackend, _>::multi(vec![], LocalExchange::new()).is_err()); // D = 0
    let Some(devs) = same_gpu_devs(4) else { return };
    let mut db = DistSvBackend::multi(devs, LocalExchange::new()).unwrap();
    assert!(db.run(&ghz(8), 1, Router::Lookahead).is_err()); // R = 2 < D = 4
}

#[test]
fn prob_one_matches_reference_for_every_qubit() {
    let Some(be) = gpu64() else { return };
    let mut db = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in [("qft10", qft(10)), ("brick10", brickwall(10, 6))] {
        let want = reference(&c);
        for g in [0u32, 2] {
            let st = db.run(&c, g, Router::Lookahead).unwrap();
            for q in 0..c.num_qubits() {
                let p: f64 = want.iter().enumerate()
                    .filter(|(i, _)| (i >> q) & 1 == 1).map(|(_, a)| a.norm_sqr()).sum();
                let got = db.prob_one(&st, q).unwrap();
                assert!((got - p).abs() < 1e-10, "{name} g={g} q={q}: {got} vs {p}");
            }
            assert!(db.prob_one(&st, c.num_qubits()).is_err());
        }
    }
}

#[test]
fn run_plan_rejects_bad_final_map() {
    let Some(be) = gpu64() else { return };
    let mut db = DistSvBackend::new(be, LocalExchange::new());
    let c = ghz(6);
    let l = aleph_ir::dist::DistLayout::new(6, 1).unwrap();
    let mut p = aleph_ir::dist::plan(&c, l, Router::Naive).unwrap();
    p.final_map[0] = p.final_map[1]; // not a permutation
    assert!(db.run_plan(&p).is_err());
}
```

If `DistPlan`'s fields are not `pub`, write the last test as a `#[cfg(test)]` unit test inside `dist/mod.rs` instead,
building the plan the same way.

- [ ] **Step 3: Run them on the box and confirm they fail to compile** (`multi`, `prob_one` missing).

- [ ] **Step 4: Implement `exchange.rs`**

```rust
pub(crate) fn rank_device(r: u32, ranks: u32, devs: usize) -> usize {
    (r / (ranks / devs as u32)) as usize
}

pub trait Exchange<B: DeviceSv> {
    fn exchange(
        &mut self,
        devs: &mut [B],
        ranks: &mut [B::State],
        layout: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError>;
}

/// Ranks on devices that share one CUDA ordinal (one GPU, possibly several
/// backends): chunk swaps are device-to-device copies through a per-device
/// scratch of at most `scratch_amps` amplitudes. A pair whose two devices are
/// different GPUs is rejected: that needs `NcclExchange`.
pub struct LocalExchange<B: DeviceSv> {
    scratch_amps: usize,
    /// Per device index: (scratch slice, amplitudes it holds).
    scratch: Vec<Option<(B::State, usize)>>,
    _b: PhantomData<B>,
}
```

`new`, `with_scratch_amps` and `Default` keep the same bodies, with `scratch: Vec::new()`. Then:

```rust
impl<B: DeviceSv> Exchange<B> for LocalExchange<B> {
    fn exchange(
        &mut self,
        devs: &mut [B],
        ranks: &mut [B::State],
        l: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError> {
        let m = l.m();
        let k = global_bits.len() as u32;
        if k == 0 || k > m || !valid_bits(l, global_bits) || ranks.len() != l.ranks() as usize || devs.is_empty() {
            return Err(BackendError::InvalidState { reason: "dist: bad exchange bits" });
        }
        let chunk = 1usize << (m - k);
        let piece = chunk.min(self.scratch_amps);
        self.scratch.resize_with(devs.len(), || None);
        for ((ra, ca), (rb, cb)) in chunk_pairs(l, global_bits) {
            let da = rank_device(ra, l.ranks(), devs.len());
            let db = rank_device(rb, l.ranks(), devs.len());
            if devs[da].ordinal() != devs[db].ordinal() {
                return Err(BackendError::InvalidState {
                    reason: "dist: cross-GPU chunk pair needs NcclExchange",
                });
            }
            // Scratch lives on rank a's device; grow it when a later, smaller-k
            // exchange has bigger chunks (never beyond `scratch_amps`).
            let slot = &mut self.scratch[da];
            if slot.as_ref().is_none_or(|(_, len)| *len < piece) {
                *slot = Some((devs[da].alloc_rank(piece.trailing_zeros(), 1)?, piece));
            }
            let Some((scr, _)) = slot.as_mut() else {
                return Err(BackendError::InvalidState { reason: "dist: scratch missing" });
            };
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let Some((sa, sb)) = two_mut(ranks, ra as usize, rb as usize) else {
                return Err(BackendError::InvalidState { reason: "dist: exchange paired a rank with itself" });
            };
            // Same ordinal ⇒ same primary context and legacy default stream,
            // so issuing every copy through devs[da] keeps them stream-ordered.
            let be = &mut devs[da];
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

Update the existing `local_exchange_matches_cpu_reference` and `local_exchange_rejects_duplicate_bits` tests to call
`x.exchange(std::slice::from_mut(&mut be), …)`.

- [ ] **Step 5: Implement `mod.rs`**

```rust
pub struct DistSvBackend<B: DeviceSv, X: Exchange<B>> {
    devs: Vec<B>,
    x: X,
    fuse: bool,
}

impl<B: DeviceSv, X: Exchange<B>> DistSvBackend<B, X> {
    /// One device; per-rank fusion on.
    pub fn new(be: B, x: X) -> Self {
        Self { devs: vec![be], x, fuse: true }
    }

    /// One backend per device. Ranks go to devices in contiguous blocks
    /// (rank r → device r / (R/D)), so the top log2(D) rank bits name the
    /// device and an exchange on lower global bits stays on-device.
    pub fn multi(devs: Vec<B>, x: X) -> Result<Self, DistSvError> {
        if devs.is_empty() || !devs.len().is_power_of_two() {
            return Err(BackendError::InvalidState { reason: "dist: device count must be a power of two ≥ 1" }.into());
        }
        Ok(Self { devs, x, fuse: true })
    }

    pub fn devices(&self) -> usize {
        self.devs.len()
    }
```

In `run_plan`:

```rust
let l = p.layout;
let (m, nr, nd) = (l.m(), l.ranks(), self.devs.len());
if nd as u64 > u64::from(nr) {
    return Err(BackendError::InvalidState { reason: "dist: more devices than ranks" }.into());
}
check_final_map(&p.final_map, l.n)?;
let mut ranks = Vec::with_capacity(nr as usize);
for r in 0..nr {
    ranks.push(self.devs[rank_device(r, nr, nd)].alloc_rank(m, r)?);
}
for step in &p.steps {
    match step {
        DistStep::Local(instrs) => {
            for (r, st) in ranks.iter_mut().enumerate() {
                let local = self.rank_program(instrs, l, r as u32)?;
                let be = &mut self.devs[rank_device(r as u32, nr, nd)];
                for i in local.instructions() {
                    apply_one(be, st, i)?;
                }
            }
        }
        DistStep::Exchange { global_bits } => {
            self.x.exchange(&mut self.devs, &mut ranks, l, global_bits)?;
        }
    }
}
```

The launches are asynchronous on each device's stream, so the serial host loop still overlaps the devices. No host
barrier is needed before an exchange: NCCL and memcpy are ordered on the same streams.

```rust
/// `final_map` must be a permutation of 0..n (a corrupt plan would read the
/// wrong amplitudes or index out of bounds).
fn check_final_map(map: &[u32], n: u32) -> Result<(), DistSvError> {
    let mut seen = vec![false; n as usize];
    if map.len() != n as usize {
        return Err(BackendError::InvalidState { reason: "dist: final_map length != n" }.into());
    }
    for &p in map {
        let Some(s) = seen.get_mut(p as usize) else {
            return Err(BackendError::InvalidState { reason: "dist: final_map out of range" }.into());
        };
        if std::mem::replace(s, true) {
            return Err(BackendError::InvalidState { reason: "dist: final_map not a permutation" }.into());
        }
    }
    Ok(())
}
```

`amplitudes` downloads each rank through `self.devs[rank_device(..)]`. Readout:

```rust
/// Σ|a|²: per-rank device reduction, host sum of R scalars (spec §3.4).
pub fn norm_sqr(&mut self, st: &DistSvState<B>) -> Result<f64, DistSvError> {
    let (nr, nd) = (st.layout.ranks(), self.devs.len());
    let mut s = 0.0;
    for (r, rs) in st.ranks.iter().enumerate() {
        s += self.devs[rank_device(r as u32, nr, nd)].branch_norms(rs, 0)?.0;
    }
    Ok(s)
}

/// P(logical qubit `q` = 1). Physical position `final_map[q]`: a local bit is
/// a per-rank masked reduction; a global bit sums whole ranks with that rank bit set.
pub fn prob_one(&mut self, st: &DistSvState<B>, q: u32) -> Result<f64, DistSvError> {
    let Some(&p) = st.final_map.get(q as usize) else {
        return Err(BackendError::QubitOutOfRange { qubit: q, num_qubits: st.layout.n }.into());
    };
    let (m, nr, nd) = (st.layout.m(), st.layout.ranks(), self.devs.len());
    let mut s = 0.0;
    for (r, rs) in st.ranks.iter().enumerate() {
        let be = &mut self.devs[rank_device(r as u32, nr, nd)];
        if p < m {
            s += be.branch_norms(rs, 1u64 << p)?.1;
        } else if (r as u32 >> (p - m)) & 1 == 1 {
            s += be.branch_norms(rs, 0)?.0;
        }
    }
    Ok(s)
}
```

Update the module doc. "NCCL across devices is P6-01b" becomes a pointer to `NcclExchange`.

- [ ] **Step 6: Run on the box:** `cargo test -p aleph-cuda --features cuda --test dist_gpu_oracle` and
  `--test dist_local_bench --release -- --ignored` (smoke only: confirms the bench still builds and runs). Expected:
  every oracle test PASSes.

- [ ] **Step 7: Mutation check.** Temporarily change `rank_device` to `(r % devs as u32) as usize` (round-robin while
  the exchange assumes blocks). Expected: `multi_backend_same_gpu_matches_oracle` fails. Revert. Then make `prob_one`
  read `1u64 << (p - m)` for the global branch. Expected: `prob_one_matches…` fails. Revert.

- [ ] **Step 8: Commit** `[P6-01b] Multi-device DistSvBackend, per-device LocalExchange, device norm/prob_one`.

---

### Task 4: Piece scheduler + NcclExchange (feature `nccl`)

**Files:**
- Create: `crates/aleph-cuda/src/dist/schedule.rs`, `crates/aleph-cuda/src/dist/nccl.rs`
- Modify: `crates/aleph-cuda/src/dist/mod.rs` (`mod schedule; #[cfg(feature = "nccl")] mod nccl; #[cfg(feature = "nccl")] pub use nccl::NcclExchange;`)
- Modify: `crates/aleph-cuda/src/lib.rs` (re-export `NcclExchange` under `cfg(feature = "nccl")`)
- Modify: `crates/aleph-cuda/Cargo.toml` (`[features]`)
- Create: `crates/aleph-cuda/tests/dist_nccl_oracle.rs`

**Interfaces:**
- Consumes: `chunk_pairs`, `valid_bits`, `rank_device`, `two_mut` (make `pub(crate)`) and the Task 2 `DeviceSv` items.
- Produces:

```rust
// schedule.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub ra: u32, pub a_off: usize, pub da: usize, pub slot_a: usize, // a's piece lands in scratch[da][slot_a..]
    pub rb: u32, pub b_off: usize, pub db: usize, pub slot_b: usize, // b's piece lands in scratch[db][slot_b..]
    pub len: usize,
}
/// Split every chunk pair into pieces of `min(chunk, cap/2)` amplitudes and
/// pack them greedily into rounds in which each device uses at most `cap`
/// scratch amplitudes. `cap` is a power of two ≥ 2.
pub(crate) fn schedule(pairs: &[((u32, u32), (u32, u32))], chunk: usize, cap: usize,
                       dev_of: impl Fn(u32) -> usize, n_dev: usize) -> Vec<Vec<Piece>>;

// nccl.rs
pub struct NcclExchange<B: DeviceSv> { /* comms, scratch, route_all */ }
impl<B: DeviceSv> NcclExchange<B> where B::Scalar: cudarc::nccl::safe::NcclType {
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;
    /// One NCCL rank per backend; ordinals must be distinct.
    pub fn new(devs: &[B]) -> Result<Self, BackendError>;
    pub fn with_scratch_amps(self, amps: usize) -> Self;
    /// Diagnostic: route same-device pairs through NCCL self send/recv too
    /// (lets one GPU exercise the full NCCL piece protocol).
    pub fn route_all_through_nccl(self, on: bool) -> Self;
}
impl<B: DeviceSv> Exchange<B> for NcclExchange<B> where B::Scalar: NcclType { .. }
```

- [ ] **Step 1: Feature flag.** In `Cargo.toml` `[features]`:

```toml
# Multi-GPU exchange through NCCL (P6-01b). cudarc dlopens libnccl at runtime,
# so this builds (and CI clippies) without NCCL installed; the library is only
# needed when an `NcclExchange` is constructed.
nccl = ["cuda", "cudarc?/nccl"]
```

On the box, check: `cargo build -p aleph-cuda --features nccl`. On the Mac (no CUDA), `cargo +beta clippy -p aleph-cuda
--features nccl --all-targets -- -D warnings` must still be clean, because cudarc is target-gated to Linux.

- [ ] **Step 2: Write the failing scheduler test** (in `schedule.rs`, `#[cfg(test)]`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::exchange::{chunk_pairs, rank_device};
    use aleph_ir::dist::DistLayout;

    /// Every amplitude of every moved chunk is in exactly one piece; per
    /// round, each device's scratch slots are disjoint and inside `cap`.
    #[test]
    fn schedule_covers_once_and_respects_scratch() {
        for (n, g, d) in [(8u32, 1u32, 1usize), (8, 2, 2), (10, 3, 4), (10, 3, 8)] {
            let l = DistLayout::new(n, g).unwrap();
            let m = l.m();
            for bits in [vec![m], vec![n - 1], vec![m, n - 1], (m..n).collect::<Vec<_>>()] {
                let k = bits.len() as u32;
                if k > m { continue; }
                let chunk = 1usize << (m - k);
                for cap in [2usize, 8, 1 << 20] {
                    let pairs = chunk_pairs(l, &bits);
                    let rounds = schedule(&pairs, chunk, cap, |r| rank_device(r, l.ranks(), d), d);
                    let mut covered = std::collections::HashSet::new();
                    for round in &rounds {
                        let mut used: Vec<Vec<(usize, usize)>> = vec![vec![]; d];
                        for p in round {
                            for i in 0..p.len {
                                assert!(covered.insert((p.ra, p.a_off + i)), "a dup");
                                assert!(covered.insert((p.rb, p.b_off + i)), "b dup");
                            }
                            used[p.da].push((p.slot_a, p.len));
                            used[p.db].push((p.slot_b, p.len));
                        }
                        for slots in &mut used {
                            slots.sort();
                            let mut end = 0;
                            for &(s, len) in slots.iter() {
                                assert!(s >= end && s + len <= cap, "slot overlap / over cap");
                                end = s + len;
                            }
                        }
                    }
                    let moved = (l.ranks() as usize) * ((1usize << k) - 1) * chunk;
                    assert_eq!(covered.len(), moved, "n={n} g={g} d={d} {bits:?} cap={cap}");
                }
            }
        }
    }
}
```

- [ ] **Step 3: Run on the box and confirm it fails to compile.**

Run: `cargo test -p aleph-cuda --features cuda --lib schedule`

- [ ] **Step 4: Implement `schedule.rs`**

```rust
//! Splits an exchange's chunk pairs into scratch-sized pieces and packs them
//! into rounds. One round = one NCCL group: every piece's two halves are
//! received into scratch slots, then copied back into the rank slices. The
//! scratch is the only extra memory, so a rank slice keeps its full reach.

pub(crate) fn schedule(
    pairs: &[((u32, u32), (u32, u32))],
    chunk: usize,
    cap: usize,
    dev_of: impl Fn(u32) -> usize,
    n_dev: usize,
) -> Vec<Vec<Piece>> {
    // cap/2 so that a same-device pair (both halves on one device) fits an empty round.
    let len = chunk.min((cap / 2).max(1));
    let mut rounds = Vec::new();
    let mut cur: Vec<Piece> = Vec::new();
    let mut used = vec![0usize; n_dev];
    for &((ra, ca), (rb, cb)) in pairs {
        let (da, db) = (dev_of(ra), dev_of(rb));
        let mut off = 0;
        while off < chunk {
            let need_a = len + if da == db { len } else { 0 };
            if used[da] + need_a > cap || used[db] + len > cap {
                rounds.push(std::mem::take(&mut cur));
                used.iter_mut().for_each(|u| *u = 0);
            }
            let slot_a = used[da];
            used[da] += len;
            let slot_b = used[db];
            used[db] += len;
            cur.push(Piece {
                ra, a_off: ca as usize * chunk + off, da, slot_a,
                rb, b_off: cb as usize * chunk + off, db, slot_b,
                len,
            });
            off += len;
        }
    }
    if !cur.is_empty() {
        rounds.push(cur);
    }
    rounds
}
```

- [ ] **Step 5: Run the scheduler test on the box. Expected: PASS.**

- [ ] **Step 6: Write the failing NCCL tests** in `tests/dist_nccl_oracle.rs`

```rust
//! NcclExchange oracle. On a 1-GPU host this runs D=1 with every pair forced
//! through NCCL self send/recv; on a multi-GPU host (AWS 4×L4) it also runs
//! one rank-block per real GPU at D=2 and D=4.
//! Run: cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle -- --nocapture
#![cfg(all(target_os = "linux", feature = "nccl"))]

mod common;
use aleph_cuda::{device_count, CudaSvBackend, CudaSvBackendF32, DistSvBackend, NcclExchange};
use aleph_ir::dist::Router;
use common::dist::*;

fn devs64(d: usize) -> Option<Vec<CudaSvBackend>> {
    (0..d).map(|i| CudaSvBackend::on_device(i).ok()).collect()
}

#[test]
fn nccl_new_without_lib_is_err_not_panic() {
    // Must not unwind whatever the host has; on the box (lib installed) it is Ok.
    let Some(devs) = devs64(1) else { return };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| NcclExchange::new(&devs).is_ok()));
    assert!(r.is_ok(), "NcclExchange::new panicked");
}

#[test]
fn nccl_rejects_duplicate_ordinals() {
    let Some(a) = devs64(1) else { return };
    let Some(b) = devs64(1) else { return };
    let devs: Vec<_> = a.into_iter().chain(b).collect(); // two backends, both GPU 0
    assert!(NcclExchange::new(&devs).is_err());
}

#[test]
fn nccl_self_routed_matches_oracle_f64() {
    let Some(devs) = devs64(1) else { return };
    let Ok(x) = NcclExchange::new(&devs) else { eprintln!("libnccl absent: skip"); return };
    let x = x.with_scratch_amps(8).route_all_through_nccl(true);
    let mut db = DistSvBackend::multi(devs, x).unwrap();
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 1..=3u32 {
            for router in [Router::Naive, Router::Lookahead] {
                let st = db.run(&c, g, router).unwrap();
                let got = db.amplitudes(&st).unwrap();
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!((x - y).norm() < 1e-10, "{name} g={g} {router:?} amp {i}");
                }
            }
        }
    }
}

#[test]
fn nccl_self_routed_matches_oracle_f32() {
    let Some(devs) = (0..1).map(|i| CudaSvBackendF32::on_device(i).ok()).collect::<Option<Vec<_>>>() else { return };
    let Ok(x) = NcclExchange::new(&devs) else { return };
    let mut db = DistSvBackend::multi(devs, x.with_scratch_amps(8).route_all_through_nccl(true)).unwrap();
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 1..=3u32 {
            let st = db.run(&c, g, Router::Lookahead).unwrap();
            for (i, (x, y)) in db.amplitudes(&st).unwrap().iter().zip(&want).enumerate() {
                assert!((x - y).norm() < 1e-5, "{name} g={g} amp {i}");
            }
        }
    }
}

#[test]
fn nccl_real_multi_gpu_matches_oracle() {
    let n_dev = device_count().unwrap_or(0);
    for d in [2usize, 4] {
        if n_dev < d {
            eprintln!("only {n_dev} GPU(s): skip D={d}");
            continue;
        }
        let Some(devs) = devs64(d) else { return };
        let x = NcclExchange::new(&devs).unwrap().with_scratch_amps(8);
        let mut db = DistSvBackend::multi(devs, x).unwrap();
        for (name, c) in cases() {
            let want = reference(&c);
            for g in (d.trailing_zeros())..=3u32 {
                for router in [Router::Naive, Router::Lookahead] {
                    let st = db.run(&c, g, router).unwrap();
                    let got = db.amplitudes(&st).unwrap();
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!((x - y).norm() < 1e-10, "{name} D={d} g={g} {router:?} amp {i}");
                    }
                    assert!((db.norm_sqr(&st).unwrap() - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}
```

- [ ] **Step 7: Run on the box and confirm it fails to compile** (`NcclExchange` missing).

Run: `cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle`

- [ ] **Step 8: Library-presence guard.** On the box, run
  `grep -n "is_culib_present\|fn lib()" $CUDARC/src/nccl/sys/mod.rs` (`CUDARC=$(ls -d ~/.cargo/registry/src/*/cudarc-0.19.8)`).
  - If `is_culib_present()` exists, call it first in `new` and return
    `InvalidState { reason: "nccl: libnccl not found" }` when it is false.
  - Otherwise wrap the first NCCL call (`Comm::from_devices`) in `std::panic::catch_unwind`, mapping an unwind to the
    same error, with a comment saying cudarc's dynamic loader panics on a missing library.

- [ ] **Step 9: Implement `nccl.rs`**

```rust
//! Cross-device chunk exchange through NCCL point-to-point (P6-01b, spec §3.3).
//!
//! One NCCL rank per device, all driven from this process (`ncclCommInitAll`).
//! An exchange is split by `schedule` into rounds that fit a per-device
//! scratch; a round is one `ncclGroupStart/End` in which each piece's halves
//! are sent from the rank slices and received into the peer's scratch,
//! followed by device-local scratch→slice copies. Comms are bound to each
//! backend's own stream, so NCCL is ordered after the preceding kernels and
//! before the following ones without a host barrier.

use std::marker::PhantomData;

use aleph_backend::BackendError;
use aleph_ir::dist::DistLayout;
use cudarc::nccl::safe::{group_end, group_start, Comm, NcclType};

use super::exchange::{chunk_pairs, rank_device, two_mut, valid_bits, Exchange};
use super::schedule::schedule;
use super::DeviceSv;

fn nccl_err(_e: cudarc::nccl::result::NcclError) -> BackendError {
    BackendError::InvalidState { reason: "nccl: communication failure" }
}

pub struct NcclExchange<B: DeviceSv> {
    comms: Vec<Comm>,
    scratch_amps: usize,
    scratch: Vec<Option<(B::State, usize)>>,
    route_all: bool,
    _b: PhantomData<B>,
}

impl<B: DeviceSv> NcclExchange<B>
where
    B::Scalar: NcclType,
{
    pub const DEFAULT_SCRATCH_AMPS: usize = 1 << 24;

    pub fn new(devs: &[B]) -> Result<Self, BackendError> {
        let mut ords: Vec<usize> = devs.iter().map(DeviceSv::ordinal).collect();
        ords.sort_unstable();
        if ords.is_empty() || ords.windows(2).any(|w| w[0] == w[1]) {
            return Err(BackendError::InvalidState {
                reason: "nccl: need ≥1 device and one backend per distinct GPU (NCCL rejects duplicate GPUs)",
            });
        }
        // (Step 8 guard goes here.)
        let comms = Comm::from_devices(devs.iter().map(DeviceSv::stream).collect()).map_err(nccl_err)?;
        Ok(Self {
            comms,
            scratch_amps: Self::DEFAULT_SCRATCH_AMPS,
            scratch: Vec::new(),
            route_all: false,
            _b: PhantomData,
        })
    }

    /// Per-device scratch of `amps` amplitudes (power of two, min 2).
    pub fn with_scratch_amps(mut self, amps: usize) -> Self {
        self.scratch_amps = amps.clamp(2, 1 << 40).next_power_of_two();
        self.scratch.clear();
        self
    }

    pub fn route_all_through_nccl(mut self, on: bool) -> Self {
        self.route_all = on;
        self
    }
}
```

`clamp(2, 1 << 40)` before `next_power_of_two` also fixes the P6-01a deferred debug panic, which happened on a huge
value. Apply the same clamp in `LocalExchange::with_scratch_amps` (min 1).

```rust
impl<B: DeviceSv> Exchange<B> for NcclExchange<B>
where
    B::Scalar: NcclType,
{
    fn exchange(
        &mut self,
        devs: &mut [B],
        ranks: &mut [B::State],
        l: DistLayout,
        global_bits: &[u32],
    ) -> Result<(), BackendError> {
        let m = l.m();
        let k = global_bits.len() as u32;
        if k == 0 || k > m || !valid_bits(l, global_bits) || ranks.len() != l.ranks() as usize
            || devs.len() != self.comms.len()
        {
            return Err(BackendError::InvalidState { reason: "dist: bad exchange bits" });
        }
        let chunk = 1usize << (m - k);
        let nd = devs.len();
        let dev_of = |r: u32| rank_device(r, l.ranks(), nd);
        // Scratch: `scratch_amps` per device (allocated once, reused).
        self.scratch.resize_with(nd, || None);
        for (d, slot) in self.scratch.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some((devs[d].alloc_rank(self.scratch_amps.trailing_zeros(), 1)?, self.scratch_amps));
            }
        }
        let pairs = chunk_pairs(l, global_bits);
        let (nccl_pairs, local_pairs): (Vec<_>, Vec<_>) = pairs
            .into_iter()
            .partition(|&((ra, _), (rb, _))| self.route_all || dev_of(ra) != dev_of(rb));
        // Same-device pairs: three D2D copies through scratch slot 0 (as LocalExchange).
        for ((ra, ca), (rb, cb)) in local_pairs {
            let d = dev_of(ra);
            let Some((scr, cap)) = self.scratch[d].as_mut() else {
                return Err(BackendError::InvalidState { reason: "dist: scratch missing" });
            };
            let piece = chunk.min(*cap);
            let (a0, b0) = (ca as usize * chunk, cb as usize * chunk);
            let Some((sa, sb)) = two_mut(ranks, ra as usize, rb as usize) else {
                return Err(BackendError::InvalidState { reason: "dist: exchange paired a rank with itself" });
            };
            let mut off = 0;
            while off < chunk {
                devs[d].copy_amps(sa, a0 + off, scr, 0, piece)?;
                devs[d].copy_amps(sb, b0 + off, sa, a0 + off, piece)?;
                devs[d].copy_amps(scr, 0, sb, b0 + off, piece)?;
                off += piece;
            }
        }
        // Cross-device (or all, when route_all) pairs: NCCL rounds.
        for round in schedule(&nccl_pairs, chunk, self.scratch_amps, dev_of, nd) {
            group_start().map_err(nccl_err)?;
            for p in &round {
                let (ca, cb) = (&self.comms[p.da], &self.comms[p.db]);
                let peer_a = p.db as i32; // comm rank == device index
                let peer_b = p.da as i32;
                // Recv targets are scratch slots; send sources are the rank
                // slices; nothing is written that is also read in this group.
                let send_a = B::amps_view(&ranks[p.ra as usize], p.a_off, p.len)?;
                let send_b = B::amps_view(&ranks[p.rb as usize], p.b_off, p.len)?;
                ca.send(&send_a, peer_a).map_err(nccl_err)?;
                cb.send(&send_b, peer_b).map_err(nccl_err)?;
                let mut rx_a = scratch_view_mut::<B>(&mut self.scratch, p.da, p.slot_a, p.len)?;
                ca.recv(&mut rx_a, peer_a).map_err(nccl_err)?; // b's piece → a's device
                drop(rx_a);
                let mut rx_b = scratch_view_mut::<B>(&mut self.scratch, p.db, p.slot_b, p.len)?;
                cb.recv(&mut rx_b, peer_b).map_err(nccl_err)?; // a's piece → b's device
            }
            group_end().map_err(nccl_err)?;
            for p in &round {
                copy_back::<B>(devs, &mut self.scratch, ranks, p.da, p.slot_a, p.ra, p.a_off, p.len)?;
                copy_back::<B>(devs, &mut self.scratch, ranks, p.db, p.slot_b, p.rb, p.b_off, p.len)?;
            }
        }
        Ok(())
    }
}
```

The helpers go in the same file:
- `scratch_view_mut::<B>(scratch, d, slot, len) -> Result<CudaViewMut<'_, B::Scalar>, BackendError>` calls
  `B::amps_view_mut` on `scratch[d]`, or returns an error when the scratch is missing.
- `copy_back::<B>(devs, scratch, ranks, d, slot, r, off, len)` runs
  `devs[d].copy_amps(&scratch[d].0, slot, &mut ranks[r], off, len)`.

**A recv must never alias a send in the same group.** A send source is a rank slice and a recv target is a scratch
slot, and the scheduler hands out disjoint slots. If the borrow checker rejects holding `send_*` views across the
`recv` calls (shared borrows of `ranks` alongside mutable borrows of `self.scratch`, which are different fields), issue
each piece's two sends, drop the views, then issue its two recvs. Order inside a group does not matter to NCCL.

- [ ] **Step 10: Run the NCCL tests on the box.**

First install NCCL. Ask the user before installing system packages on the box:
`apt-get install -y libnccl2=2.32.3-1+cuda13.4`. The CUDA repo is already configured: `apt-cache policy libnccl2`
shows that candidate. NCCL 2.32 is a superset of the cudarc `nccl-02030` bindings.

Run: `cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle -- --nocapture`

Expected:
- `nccl_new_without_lib…`, `nccl_rejects_duplicate…` and both `self_routed` tests PASS;
- `nccl_real_multi_gpu…` prints `only 1 GPU(s): skip D=2/4`.

Also run `cargo test -p aleph-cuda --features nccl --test dist_gpu_oracle` (no regression with the feature on).

- [ ] **Step 11: Missing-library check.** On the box, run the oracle with the library hidden:
  `LD_LIBRARY_PATH= cargo test … nccl_new_without_lib` after `mv /usr/lib/x86_64-linux-gnu/libnccl.so.2{,.off}`, then
  `mv` it back. Expected: PASS (an `Err`, not a panic). Only do this on the box; restore immediately. If renaming a
  system lib is unwelcome, skip it and note it in the PR as checked by reading the code instead.

- [ ] **Step 12: Mutation checks.**
  - Set `slot_b = slot_a` in the scheduler. Expected: the scheduler test and `self_routed_f64` fail. Revert.
    (A peer swap cannot be caught at D=1, where both peers are 0. The AWS D=2/4 run covers it.)
  - Drop the `copy_back` of the `b` side. Expected: `self_routed_f64` fails. Revert.

- [ ] **Step 13: Commit** `[P6-01b] NcclExchange: scratch-round NCCL p2p exchange (feature nccl)`.

---

### Task 5: CI coverage for the `nccl` feature

**Files:**
- Modify: `.github/workflows/ci.yml` (next to lines 58–59 and 98–100)

- [ ] **Step 1: Add the steps**

```yaml
      - name: clippy aleph-cuda (nccl feature)
        run: cargo clippy -p aleph-cuda --features nccl --all-targets -- -D warnings
```

and, in the build job:

```yaml
      - name: build aleph-cuda (nccl feature)
        run: cargo build -p aleph-cuda --features nccl
```

- [ ] **Step 2: Locally run** `cargo +beta clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, and on
  the box `cargo +beta clippy -p aleph-cuda --features nccl --all-targets -- -D warnings` (install with `rustup
  toolchain install beta -c clippy` if missing). Expected: clean.

- [ ] **Step 3: Commit** `[P6-01b] CI: clippy + build aleph-cuda with --features nccl`.

---

### Task 6: Scaling bench, time model, AWS runbook (prepared, not run)

**Files:**
- Create: `crates/aleph-cuda/tests/dist_nccl_bench.rs` (`#[ignore]`)
- Create: `docs/perf/p6-multi-gpu.md`
- Create: `scripts/aws/p6-multi-gpu-session.sh`

**Interfaces:**
- Consumes: `NcclExchange`, `DistSvBackend::multi`, `device_count`, the `brickwall` / `qft` / `ghz` helpers from
  `tests/common/dist.rs`, and `run_paged` (existing, for the n=32 FP64 baseline; check its call shape in
  `tests/paged_bench.rs`).

- [ ] **Step 1: Write the bench.** It prints one CSV line per cell. The pieces:

```rust
//! P6 multi-GPU scaling: NCCL p2p bandwidth, strong scaling (fixed n,
//! D = 1→2→4), weak scaling (n = m_max + log2 D) vs single-GPU in-core /
//! `run_paged`. Prints the model's prediction next to each measurement.
//! Run: cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "nccl"))]
```

- `p2p_bandwidth(d)`: one 256 MiB send/recv between GPU 0 and GPU 1 (D≥2) in a group, 5 timed reps after 1 warm-up,
  `stream.synchronize()` on both. It prints `p2p,GBps=…`. At D=1 it measures a self send/recv (a lower bound for the
  protocol overhead).
- `time_dist(c, devs, g)`: one warm-up `run`, then 3 timed `run`s; each ends with `synchronize` on every device stream
  (no host readout, same as P6-01a). It prints the exchange count, `plan.comm_stats()` bytes and the median seconds.
- Strong: QFT-28, brickwall-28 d=10 and GHZ-28. FP64 and FP32. D ∈ {1,2,4} ∩ available, `R = D`.
- Weak:
  - FP64: n = 30 + log2 D (n=32 at D=4, 16 GiB per L4);
  - FP32: n = 31 + log2 D;
  - baseline: single-GPU `run_paged` at the same n when it does not fit in-core.
- Model line per cell:
  `pred = passes · (2^m · AMP_BYTES) / dev_bw + exch_bytes_per_dev / p2p_bw`. Here `dev_bw` is measured with a
  one-pass 1q-gate timing at D=1, and `passes` is the per-rank instruction count after fusion (`rank_program` length).
  Expose a `#[doc(hidden)] pub fn rank_pass_count(&self, p: &DistPlan) -> usize` on `DistSvBackend` for this.

- [ ] **Step 2: Run the bench on the box (D=1 only) to prove it runs.** Before measuring, check the box is idle
  (`uptime`, `nvidia-smi` util ≈ 0; the embed container is on CPU). Expected: `p2p` self line, strong D=1 cells, and
  weak FP64 n=30 / FP32 n=31.

- [ ] **Step 3: Write `docs/perf/p6-multi-gpu.md`.**
  - Comm-count table (Naive vs Lookahead; QFT-32, random-30, GHZ-32, Grover; g ∈ {2,3}). Take it from the existing
    P6-02/P6-03 CPU numbers and re-run the CPU comm-count test if it exists.
  - The time model.
  - A **predicted** strong/weak table for 4×L4. Use L4 numbers stated as assumptions: ~300 GB/s device BW, ~25 GB/s
    PCIe Gen4 x16 p2p, no NVLink. The measured columns stay empty with "AWS session pending".
  - Spec §6 requires the prediction to be stated before the AWS run.

- [ ] **Step 4: Write `scripts/aws/p6-multi-gpu-session.sh`.** It runs step by step with `set -euo pipefail` and
  echoes every action. It is **not run in this PR**.
  1. Region `us-east-1`. Resolve the AMI from SSM `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id`.
  2. `aws ec2 run-instances --instance-type g6.12xlarge --instance-initiated-shutdown-behavior terminate --user-data`
     with a script that runs `shutdown -h +240`, a 200 GB gp3 root, and tag `Name=aleph-p6`.
  3. Wait for SSH. Install rustup, plus `libnccl2` / `libnccl-dev` **2.32** from the CUDA apt repo, so the DLAMI's
     older NCCL does not miss 2.30 symbols. Rsync the tree.
  4. `nvidia-smi topo -m` → results; `cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle --test dist_gpu_oracle`;
     `cargo test --release … dist_nccl_bench -- --ignored --nocapture | tee results/bench.csv`.
  5. scp `results/`, then `aws ec2 terminate-instances`.
  6. All-region scan: `for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do aws ec2 describe-instances --region $r --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text; done`.
     It must print nothing.

  The header comment gives the cost estimate (~$4.6/h × ≤4 h ≤ $19) and the rule: **launch only with the user's
  explicit OK**.

- [ ] **Step 5: Commit** `[P6-01b] Scaling bench, time model + predicted 4×L4 table, AWS runbook`.

---

### Task 7: Full verification, review, PR

- [ ] **Step 1: On the box:**
  - `cargo test -p aleph-cuda --features nccl` (the whole crate). Note the known flaky
    `mem_pool::many_small_circuits_no_leak` if it appears, and re-run it alone.
  - `cargo test --workspace` on the Mac.
- [ ] **Step 2:**
  - `cargo +beta clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` on the Mac;
  - `cargo +beta clippy -p aleph-cuda --features nccl --all-targets -- -D warnings` on the box.
- [ ] **Step 3: Final whole-branch review** (most capable model) against this plan and spec §3.3/§3.4/§4/§8. Fix the
  important findings, reproducing each one with a test first.
- [ ] **Step 4: Push the branch and open the PR** `[P6-01b] NcclExchange + multi-device DistSvBackend`. The body
  covers:
  - `Refs #55`;
  - approach;
  - oracle results (same-GPU multi-backend D=2/4, NCCL self-routed FP64/FP32, scheduler property test);
  - mutation checks;
  - D=1 bench sanity numbers;
  - "real multi-GPU validation = AWS session (runbook in `scripts/aws/`)";
  - deferred items:
    - the router does not yet prefer intra-device global bits;
    - `Comm` drop aborts and panics on failure (cudarc);
    - #529.

## Self-review notes

- **Spec coverage:**
  - §3.3 NcclExchange, scratch with no double buffer, one group per round, dynamic loading: Task 4.
  - §3.4 device norm and prob: Task 3.
  - §4 errors (power of two, device fit through `alloc_rank`, NCCL failure): Tasks 3 and 4.
  - §5 AWS suite: the `nccl_real_multi_gpu…` test.
  - §6 model stated before AWS: Task 6.
  - §7.4 "builds and runs on the box at R=1": Task 4 Step 10.
  - §8 dynamic-loading risk: Task 4 Step 8.
- **Deferred from P6-01a, closed here:**
  - `final_map` check (Task 3);
  - the `with_scratch_amps` huge-value panic (Task 4 Step 9);
  - scratch growth now covered by `multi_backend_same_gpu…` at g=d..3 with `scratch_amps=8` (the chunk size changes
    across exchanges).
- **Not here:**
  - #59 (comm-aware compiler);
  - sampling and mid-circuit measurement;
  - MPI;
  - the AWS run itself, which needs the user's OK.
