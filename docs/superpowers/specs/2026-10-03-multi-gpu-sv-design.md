# Multi-GPU state vector (Phase 6, single node): design

**Status:** approved in conversation 2026-10-03; this file awaits review.
**Issues:** #56 (P6-02), #57 (P6-03), #55 (P6-01). Out of scope: #58 (MPI), #60 (cluster report), #59 (deferred, see §9).

## 1. Intent

The user's goal is to write the whole single-node multi-GPU state-vector stack now, on the one RTX 4000 box. Then run **one**
AWS session (g6.12xlarge, 4×L4, ~$5–18) that shows it (a) is correct on real multiple GPUs and (b) performs. "Performs"
covers **both, with equal weight**:

- **Strong scaling:** the same n=28/30 circuit is faster on 4 GPUs than on 1.
- **Weak scaling / reach:** n=32 FP64 and n=33 FP32 run in-core across 4 cards. On one card they need paging.

Both **FP64 and FP32** are supported from the start, through one generic code path. FP64 gives the 1e-10 oracle.
FP32 gives honest timing on the L4, whose FP64 rate is 1/64 of FP32.

**Success criteria:**

1. Bit-level equivalence vs single device: FP64 to 1e-10, FP32 to 1e-5 vs FP64. Checked on CPU (simulated ranks), on the
   RTX 4000 (several ranks on one GPU) and on AWS (2 and 4 real GPUs).
2. Communication count and bytes reported per circuit (#56). The lookahead router measurably reduces them vs the naive
   router (#57).
3. On AWS: strong- and weak-scaling numbers for FP32 and FP64, shown next to a bytes/bandwidth model's prediction.
   n=32 FP64 runs in-core on 4×L4 and beats single-GPU paged (`run_paged`, ~18.5× in-core cost).

No fixed strong-scaling target is set for 4×L4. That box is PCIe-only with no NVLink, so the number is reported
honestly. The #55 ">70 % at 8 GPUs" criterion needs an NVLink node (p4d) and stays open after this work.

## 2. Algorithm: qubit (index-bit) swap

`R = 2^g` ranks. Rank `r` holds `2^m` amplitudes, `m = n − g`. Physical index = `(r << m) | local`. Physical bits
`m..n` are **global** and bits `0..m` are **local**. A logical→physical qubit map is tracked lazily and never unwound,
the same idea as the IR's permutation handling and Metal MPS lazy-SWAP routing.

- **Gates on local physical qubits** run as-is, through the existing in-core kernels and fusion. No communication.
- **Free gates even on global qubits.** Each rank knows its global bits, so a rank-specialised rewrite removes the
  global qubit:
  - diagonal gates (`Z S Sdg T Tdg Rz Phase Cz Ccz CRz`, controlled `Phase`, `DiagonalPhase`) become a per-rank
    constant phase or a local diagonal;
  - a **control** on a global qubit means "apply the gate without that control" or "skip".
  - `Swap` (any two qubits) is an O(1) map relabel.
- **Non-diagonal targets on a global qubit** (`H X Y Rx Ry U3 Cnot`-target, `Unitary*`, `Iswap`, …) need an
  **exchange**. The exchange swaps `k` chosen global physical bits with the **top `k` local physical bits** `m−k..m`. A
  local `Swap` gate brings the wanted logical qubit to a top-local position first. That costs one local bandwidth pass,
  is a normal local gate, and keeps every exchanged chunk contiguous.

**Exchange data movement.** Rank `r`'s slice splits into `2^k` contiguous chunks `c` (the value of the top `k` local
bits). Chunk `c` goes to rank `r'` = `r` with the chosen `k` global bits replaced by `c`. It lands in chunk slot =
`r`'s old values of those bits. The chunk with `c` equal to `r`'s own bits stays. Each rank sends and receives
`2^k − 1` chunks, which moves `(1 − 2^−k)` of the state.

**DiagonalPhase specialisation.** Each condition `parity(mask & x) == 1` splits into a global part (a known constant per
rank) and a local part. A condition whose local part is empty becomes a constant: the term drops or stays
unconditional. A condition that needs local parity **even** is rewritten by inclusion–exclusion,
`angle·[A ∧ ¬B] = angle·[A] − angle·[A ∧ B]`. That is exponential only in the number of negated conditions, ≤ 2 in
practice. An empty-`conds` term carries the per-rank scalar phase, so no new kernel is needed. The exact
`DiagonalPhase` semantics live in `crates/aleph-ir/src/diagonal_phase.rs`.

References: Häner & Steiger, "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit" (SC'17), §3 (global/local qubit
swaps). The cuStateVec distributed index-bit swap docs, read for the design only, never copied.

## 3. Components

### 3.1 `aleph-ir::dist` (pure, backend-agnostic): #56, #57

```rust
pub struct DistLayout { pub n: u32, pub g: u32 }            // m = n - g
pub enum DistStep {
    Local(Vec<Instruction>),                               // logical qubits already mapped to physical
    Exchange { global_bits: SmallVec<[u32; 4]> },          // swap these global phys bits with top-k local
}
pub struct DistPlan { pub layout: DistLayout, pub steps: Vec<DistStep>,
                      pub final_map: Vec<u32>, pub stats: CommStats }
pub struct CommStats { pub exchanges: u32, pub bytes_per_rank: u64 /* for a given amp size */, pub local_swaps: u32 }

pub fn plan(circuit: &Circuit, layout: DistLayout, router: Router) -> Result<DistPlan, IrError>;
pub fn specialize(instr: &Instruction, layout: DistLayout, rank: u32) -> Option<Instruction>; // None = no-op on this rank
```

- `Router::Naive`: an exchange happens on demand, `k=1`, and evicts the top local qubit. This is the #56 baseline.
- `Router::Lookahead`: chooses which local qubits to evict by the Belady rule (the one whose next use is farthest
  away). It brings in up to `g` needed global qubits in one exchange, picking the set that covers the most upcoming
  non-free gates. This is #57.
- Fusion runs **per rank, after `specialize`**, on the resulting `m`-qubit instruction list, using the existing
  passes. Fusing in physical `n`-qubit space would merge a local gate with a free global-qubit diagonal into a dense
  block on a global qubit, which then could not be specialised. Per-rank fusion lands with the GPU backend (PR 3).
- `Measure`/`Reset` return `IrError::Unsupported` in v1. `Barrier` is dropped.

### 3.2 CPU reference executor (`aleph-sv` test support)

`R` ranks are a `Vec` of CPU states of `m` qubits. `Local` runs `specialize` and then the CPU kernels. `Exchange` is a
chunk copy. The result is gathered with `final_map` and compared to `NaiveSvBackend`. This proves the whole algorithm
on macOS/CI without a GPU.

### 3.3 `aleph-cuda::dist`: #55

```rust
pub trait DeviceSv {                  // implemented by CudaSvBackend (f64) and CudaSvBackendF32
    type State;
    const AMP_BYTES: usize;
    fn alloc_zero(&self, m: u32, rank_is_zero: bool) -> Result<Self::State, BackendError>;
    fn apply(&mut self, st: &mut Self::State, instr: &Instruction) -> Result<(), BackendError>;
    fn view(st: &mut Self::State) -> DeviceView;   // for the exchange only
}
pub struct DeviceView { pub ptr: CUdeviceptr, pub bytes: usize, pub ordinal: usize, pub stream: CUstream }
pub trait Exchange {
    fn exchange(&mut self, ranks: &mut [DeviceView], layout: DistLayout,
                global_bits: &[u32], scratch_bytes: usize) -> Result<(), BackendError>;
}
pub struct LocalExchange;             // all ranks on one device; D2D memcpy
pub struct NcclExchange { comms: Vec<cudarc::nccl::Comm> }  // one per device, ncclCommInitAll, one process
pub struct DistSvBackend<B: DeviceSv, X: Exchange> { ranks: Vec<(B, B::State)>, x: X }
```

- One process drives all GPUs. There is no MPI.
- Each rank runs on its own stream. Between steps there is a barrier: an exchange waits for every rank's `Local` work.
- **Exchange memory:** the exchange goes through a fixed-size scratch `S` (default 256 MiB per peer stream), in pieces
  of send → recv-into-scratch → copy back. There is **no double buffer**, so reach is not halved. All pairs go in one
  `ncclGroupStart/End`.
- NCCL comes in through the existing `cudarc` dep as feature `nccl`. No new crate. The library is dynamically loaded,
  like the rest of cudarc, so the box and CI build without NCCL installed.
- Host RAM on g6.12xlarge (192 GiB) holds the 64 GiB pinned state the n=32 FP64 `run_paged` baseline needs.

### 3.4 Readout (v1, minimal)

- Norm and single-qubit probability: a local reduction, then a host sum over `R` scalars.
- Full gather to host, honouring `final_map`: for tests and n ≤ ~30.
- Sampling and mid-circuit measurement are not in v1.

## 4. Error handling

Everything goes through `BackendError`/`IrError`, with no `panic` on user input. The rejected cases are:

- `R` not a power of two;
- `m` smaller than `k_max` plus the kernels' minimum width;
- the per-rank state does not fit device memory;
- an NCCL or CUDA failure;
- an unsupported instruction.

Gate-parameter `is_finite` checks stay where they already are (ADR 0006).

## 5. Testing

- **CPU, macOS and CI:**
  - `plan` + CPU executor vs `NaiveSvBackend` at 1e-10 on GHZ, QFT, random brickwall and Grover, for R ∈ {2,4,8} and
    several n;
  - proptest on random circuits;
  - unit tests of `specialize` covering every diagonal gate, global controls, `DiagonalPhase` inclusion–exclusion,
    `Swap` relabel, and `final_map`;
  - a mutation check: dropping one exchange or flipping one rank bit must fail the oracle.
- **GPU box (one RTX 4000):** `DistSvBackend<_, LocalExchange>` vs single-GPU `CudaSvBackend`, FP64 at 1e-10 and FP32 at
  1e-5 vs FP64, at R ∈ {2,4,8}. The suite mirrors `tests/paged_oracle.rs`.
- **AWS:** the same suite with `NcclExchange` on 2 and 4 GPUs.

## 6. Measurement

- Communication counts (#56/#57) are CPU-only: Naive vs Lookahead on QFT-32, random-30, GHZ-32 and Grover at g ∈ {2,3}.
  They go in the PR and in `docs/perf/p6-multi-gpu.md`.
- The model's predicted time is `Σ local passes × slice / device BW + Σ exchange bytes / link BW`. It is stated before
  the AWS run and compared after.
- AWS g6.12xlarge (4×L4, PCIe):
  - correctness on 2 and 4 GPUs;
  - strong scaling at n=28/30, FP32 and FP64, 1→2→4 GPUs;
  - weak scaling: n=32 FP64 (16 GiB per GPU) and n=33 FP32 in-core vs single-GPU `run_paged`.

  The bench box must be idle and the AWS hygiene from the F2 sessions applies: a shutdown timer, terminate on
  shutdown, an all-region scan afterwards, and launch only with the user's OK.

## 7. Delivery (one issue, one PR)

1. **[P6-02] #56:** `aleph-ir::dist` (layout, `plan` with `Router::Naive`, `specialize`, `CommStats`), CPU executor,
   CPU oracle + proptests, comm-count table.
2. **[P6-03] #57:** `Router::Lookahead` (multi-bit, Belady eviction) plus a measured byte reduction vs Naive.
3. **[P6-01a] #55 (refs):** `DeviceSv` trait for both backends, `DistSvBackend`, `LocalExchange`, GPU oracle on the
   RTX 4000.
4. **[P6-01b] #55 (refs):** `NcclExchange`. It builds and runs on the box at R=1. Real multi-GPU validation happens in
   the AWS session.
5. **AWS session + `docs/perf/p6-multi-gpu.md`.** #55 keeps its 8-GPU and >70 % items open, for p4d.

## 8. Risks

- **PCIe all-to-all on 4×L4 may cap strong scaling well below linear.** The router (#57) is the lever, and the report
  states the measured number whatever it is.
- **`cudarc`'s `nccl` feature with dynamic loading, untested here.** Fallback: hand-written FFI for the ~6 NCCL calls,
  justified in the PR.
- **Global-index-dependent kernels.** `apply_phase_poly` and `TiledBlock` read the absolute index. Within a rank they
  must see the **local** index only. `specialize` removes every global-bit dependence before an instruction reaches a
  kernel, and the GPU oracle covers QFT for exactly this reason.

## 9. Deferred

- **#59, communication-aware compiler** (commutation-aware reordering, joint fusion and routing). It starts from the AWS
  data, not before.
- Lazy Pauli-X on a global qubit as a rank-bit flip.
- Sampling and mid-circuit measurement on the distributed backend.
- MPI / multi-node (#58).
