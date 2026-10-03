# P6-01a — Distributed SV on one GPU (DistSvBackend + LocalExchange)

Refs #55. Design: `docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md` §3.3. Builds on P6-02 (#56) and P6-03 (#57).

## What it is

- **`DeviceSv`** (`crates/aleph-cuda/src/dist/`) is implemented for `CudaSvBackend` (FP64) and `CudaSvBackendF32`.
  - A rank slice is an ordinary `CudaSvState` / `CudaSvStateF32` with `num_qubits = m`, so the existing kernels apply
    unchanged. `run_paged` uses the same trick.
  - The trait adds rank allocation (|0⟩ on rank 0, zeros elsewhere), D2D amplitude-range copies, and host
    download/upload.
- **`DistSvBackend<B, X>`** executes an `aleph_ir::dist::DistPlan`. On every `Local` step, each rank runs
  `specialize`, then `fuse_for_gpu` (optional, on by default), then the kernels.
  - Fusion runs after specialisation, so it never sees a global qubit.
  - `Exchange` steps go through `X`.
- **`LocalExchange`** keeps all ranks on one device. The k-bit exchange is an involution on chunks, so each moved
  chunk pair is swapped exactly once, in place: A → scratch, B → A, scratch → B.
  - The work goes in pieces of at most `scratch_amps` (default 2^24 amplitudes = 256 MiB FP64).
  - There is no double buffer, so the reach of a rank slice is not halved.
- **Readout** gathers to the host in logical order via `final_map`.
  - **This deviates from spec §3.4.** `norm_sqr` downloads every full rank slice and sums on the host. §3.4 asks for a
    device-side per-rank reduction followed by a host sum of `R` scalars.
  - §3.4's single-qubit probability is not implemented yet.
  - Both are P6-01a follow-ups. They do not affect the correctness results below, and `norm_sqr` is not in the timed
    region of the bench.
- `NcclExchange` (real multiple GPUs, one process) is P6-01b. The `Exchange<B>` trait is shaped so that PR only adds an
  implementation and a per-device `Vec<B>`.

## Correctness (RTX 4000 SFF Ada)

`crates/aleph-cuda/tests/dist_gpu_oracle.rs`, 9 tests, all green:

- `device_sv_alloc_copy_roundtrip_f64` and `…_f32`: rank-0 vs other-rank initial state, offset copies, and the
  upload/download round trip.
- `local_exchange_matches_cpu_reference`: element-exact (`assert_eq!`) against `aleph_sv::dist_ref::exchange_cpu` on
  arbitrary uploaded slice data, not states reachable from |0⟩.
  - Layouts: (n, g) = (9, 2) and (10, 3).
  - Bit orders: `[m]`, `[m+1]`, `[m, m+1]`, `[m+1, m]`, `[m+2, m, m+1]`.
  - Scratch: 4 amplitudes (far smaller than a chunk) and 2^20.
- `dist_f64_matches_oracle_all_layouts_routers_fusion`: against `NaiveSvBackend` at 1e-10 per amplitude, and norm 1
  at 1e-10.
  - Circuits: GHZ-10, QFT-10 (controlled phases on global qubits), brickwall-10 d=6, Grover-8 (QASM fixture), and an
    all-diagonal case on the top qubit (Rz, CZ, CRz, CCZ, controlled-T, Toffoli). The all-diagonal case also has two
    externally controlled CZs touching globals: one specialises to a controlled `Unitary1qDiag`, the other to a
    scalar phase gated on a local control.
  - Swept over g = 0..3 × {Naive, Lookahead} × fuse {off, on}, with an 8-amplitude scratch.
- `dist_f32_matches_oracle`: the same circuits, FP32, g = 1..3, Lookahead, at 1e-5.
- `dist_matches_single_gpu_at_n20`: brickwall-20 d=8 at g = 1..3 with the default scratch, against single-GPU
  `CudaSvBackend` at 1e-10.

- `dist_rejects_rank_slice_over_qubit_cap`, `device_sv_rejects_oversized_slices` and
  `local_exchange_rejects_duplicate_bits` check that bad input returns an error instead of panicking:
  - a rank slice over the backend's qubit cap, or one whose size `1 << m` would overflow (m = 63/64), gives
    `TooManyQubits`;
  - repeated exchange bits are rejected.

Mutation checks prove the tests bite:
- In `LocalExchange`, redirecting the middle copy's destination (`a0 + off` → `b0 + off`) fails
  `dist_f64_matches_oracle_all_layouts_routers_fusion`.
- Dropping the last piece of each chunk (`while off + piece < chunk`) fails `local_exchange_matches_cpu_reference`.
- Dropping the controls of the `Unitary1qDiag` that `specialize` emits fails the all-diagonal case (`diag10`, g=1).
- All three mutations were reverted, and the suite went back to green.

## Overhead vs single GPU (one card, so exchange = D2D copy)

Command: `cargo test --release -p aleph-cuda --features cuda --test dist_local_bench -- --ignored --nocapture`.
- Each cell is the best of 3 runs.
- Box conditions: load ≈ 0.8–1.0 (`uptime`), no competing cargo/CI jobs, GPU at 2 % utilisation with 0 MiB used
  before the run. The embeddings container had been moved to CPU, and Ollama was idle.
- Completion: both arms end each rep with `CudaContext::synchronize()` on the device default stream. There is no host
  readback in the timed region.
- The single-GPU arm runs a circuit pre-fused once with `fuse_for_gpu`, outside the timing. The distributed arm plans,
  specialises and fuses per rank inside the timing.
- The rows reproduce within 2 % across two full runs.

| precision | circuit | n | single-GPU (s) | R=2 (s) | R=4 (s) | R=2 / single | R=4 / single |
|---|---|---|---|---|---|---|---|
| FP64 | QFT | 28 | 2.943 | 2.479 | 2.498 | 0.84× | 0.85× |
| FP64 | random d=10 | 28 | 5.588 | 7.489 | 8.695 | 1.34× | 1.56× |
| FP32 | QFT | 29 | 4.716 | 4.276 | 4.251 | 0.91× | 0.90× |
| FP32 | random d=10 | 29 | 3.237 | 6.079 | 7.335 | 1.88× | 2.27× |

Communication of the plans (Lookahead router):

| circuit | R | exchanges | local steps | amplitudes moved per rank (whole run) |
|---|---|---|---|---|
| QFT-28 | 2 | 2 | 3 | 1.34e8 (1.0 × slice) |
| QFT-28 | 4 | 2 | 3 | 1.01e8 (1.5 × slice) |
| random-28 d=10 | 2 | 10 | 11 | 6.71e8 (5.0 × slice) |
| random-28 d=10 | 4 | 11 | 12 | 5.54e8 (8.25 × slice) |

## Reading

- **QFT is faster distributed than on one card (0.84–0.91×), but the two arms do different work.** Not profiled; the
  most likely cause is the 14 trailing `Swap`s of `qft(28)`.
  - In the single-GPU arm each swap is a full-state pass. The swap pairs are disjoint, so `FuseKq` at width 3 cannot
    merge them.
  - The planner absorbs every user `Swap` as an O(1) relabel.
  - Estimate: ~69 passes in total, so ~43 ms per pass. Removing 14 passes and adding 2 exchanges lands at about the
    measured 2.48 s.
  - `specialize` dropping controlled phases whose global control is 0 may contribute a little.
  - Either way, this is a swap-relabel saving the single-GPU path could also take, not a gain from distribution.
- **The brickwall pays 1.3–2.3×.** The likely sources (not profiled) are 10–11 exchanges, each a full D2D chunk swap (3 copies per moved
  pair), and from fusion being cut at every exchange boundary: 11–12 separately fused local segments instead of one
  fused circuit.
  - FP32 suffers more because its single-GPU baseline is cheaper (3.2 s vs 5.6 s), while the exchange count is the
    same.
- **This is not a multi-GPU number.** All ranks share one card's bandwidth, so the distributed machinery can only add
  work here. On real GPUs each rank gets its own memory bandwidth and the exchanges cross PCIe/NVLink.
- The AWS 4×L4 session (P6-01b with `NcclExchange`) will measure strong scaling (n=28/30, 1→2→4 GPUs) and weak scaling
  (n=32 FP64, n=33 FP32 in-core vs single-GPU `run_paged`), and compare them with the spec §6 time model.
