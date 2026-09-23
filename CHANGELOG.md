# Changelog

All notable changes to this project are documented in this file, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.

## [0.3.0] — 2026-09-22

### Added

- **`aleph.qec` — QEC decoders from Python.** `DetectorErrorModel` +
  `Decoder` bindings over stim Detector Error Models, with GIL-free batch
  decode (dense rows and stim bit-packed rows) via seven named decoders:
  `mwpm`, `union-find`, `union-find-weighted`, `bp`, `bp-osd`, `relay-bp`,
  `relay-bp-osd`. Decoders support at most 64 logical observables; a larger
  model raises `ValueError` rather than silently dropping observables.
- **`aleph.sinter` — picklable sinter adapters.** One adapter class,
  `AlephSinterDecoder`; `aleph.sinter.decoders()` returns one instance per
  decoder name, usable directly in
  `sinter.collect(..., custom_decoders=aleph.sinter.decoders())`.
- **DEM parser: `repeat` / `shift_detectors` / `^`.** `repeat` blocks
  (including nested) are unrolled and `shift_detectors(...)` offsets are
  applied during parsing (an offset overflow is a parse error). In
  `error(p) D0 D1 ^ D2 D3`, the components are preserved: matching decoders
  add one edge per component, and BP-family decoders use the parity-reduced
  merged view (indices flipped an odd number of times).
- **Noise models v1 (Phase 4.6).** A `NoiseModel` / `QuantumError` /
  `ReadoutError` API (`depolarizing_error`, `amplitude_damping_error`,
  `phase_damping_error`, `pauli_error`, `bit_flip_error`, `phase_flip_error`,
  readout error matrices) driving per-shot Monte-Carlo (quantum-jump)
  trajectories on the state-vector backend, Aer-attachment-compatible
  (gate mnemonic + qubits), oracle-matched against Qiskit Aer to 1e-5 at
  100k shots under an identical model
  (`docs/decisions/0014-noise-trajectories.md`, `ROADMAP.md` §7).
- **CUDA SV: Aer-GPU parity (Phase 5.9).** IR gate fusion fed into the CUDA
  state-vector backend, a fused `UnitaryKq`/`FuseKq` apply path, disjoint-1q
  batched dispatch, a CNOT permutation kernel, and a GPU diagonal
  phase-polynomial kernel. Every Tier-1 + Tier-2 workload at n=28 is within
  1.5× of Qiskit Aer-GPU (worst case VQE 1.25×); aleph beats Aer-GPU outright
  on QFT/QPE/QAOA (0.82–0.90×) and beats cuStateVec on every cell (up to
  4.4×) (`docs/perf/p5.9-gpu-fusion.md`).
- **CUDA SV: single-GPU reach & throughput (Phase 5.10).**
  - Warp-cooperative register-tiled fused-block kernel (`apply_kq_tiled`):
    1.07–1.18× over the generic fused kernel at every fusion width, now the
    production default at k≤3 (`docs/perf/p5.10-01-tiled-fused-block.md`).
  - Out-of-core host-memory paging (`run_paged`): the `2^n` state lives in
    pinned host memory and streams device-sized tiles through the GPU,
    reaching n=31 on a 20 GiB card at ~18.5× the in-core cost
    (`docs/perf/p5.10-02-host-paging.md`).
  - FP32 CUDA backend (`CudaSvBackendF32`): 2.03× throughput at 0.50× memory
    and n=31 in-core reach, oracle-equal to FP64 within 1e-5 (worst case
    6.2e-8) (`docs/perf/p5.10-03-fp32.md`).
- **CUDA SV: further reach & throughput (Phase 5.11).**
  - FP32 out-of-core paging sets a new single-GPU reach record of n=32 on the
    same 20 GiB card (`docs/perf/p5.11-01-fp32-paging.md`).
  - Overlapped (double-buffered) paging copies reach ~1.10×, about 89% of the
    card's ~1.24× PCIe full-duplex ceiling (`docs/perf/p5.11-02-overlapped-paging.md`).
  - Multi-gate tile residency (`run_paged_batched`) amortises PCIe traffic
    over a run of consecutive low-qubit-only gates instead of paying it once
    per gate: 25–26× fewer full-state PCIe passes and up to 9.93× wall-time
    at n=31 on a locality-rich circuit
    (`docs/perf/p5.11-03-multi-gate-residency.md`).
  - `CudaSvBackendF32` is now a first-class `Backend` impl with GPU-resident
    readout; its throughput levers stack on the FP32 precision/bandwidth win
    — up to 3.92× (QFT) / 3.58× (random brickwall) over the FP64 per-gate
    baseline (`docs/perf/p5.11-04-fp32-first-class.md`).
  - Tensor-core (TF32) fused-block matvec: 1.23–1.67× over the FP32 ALU path
    at k=4 and 1.51–1.66× at k=5, crossing the k=3 baseline at k=4 (1.02–1.18×)
    (`docs/perf/p5.11-05-tf32-fused-block.md`).

### Changed

- **Python floor lowered from ≥3.12 to ≥3.10** (`abi3-py310`), widening reach
  to the stim/sinter user base that motivated the QEC bindings above.
- **Extension module renamed to `aleph._native`.** `import aleph` is
  unchanged; only the compiled-extension internal module name moved (mixed
  maturin layout).
- **DEM parser rejects invalid probabilities.** `error(p)` with a non-finite
  `p` or `p` outside `[0, 1]` is now a parse error instead of being accepted.

## [0.2.0] — 2026-06-12

CPU parity release; see `docs/perf/parity.md`.
