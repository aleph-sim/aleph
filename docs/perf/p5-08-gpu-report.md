# P5-08 — Phase 5 GPU benchmark report

**Phase 5 exit metric — "GPU backend within 1.5× of cuQuantum standalone" — is
MET.** Across the full Tier-1 + Tier-2 suite at `n = 28`, aleph's hand-written
FP64 state-vector backend is **≤ 1.22× of NVIDIA cuStateVec** on every workload
(worst cell 1.22×; the diagonal-dominated ones land at ~1.00×).

## Hardware & method

- **GPU:** NVIDIA RTX 4000 SFF Ada Generation (sm_89, 20 GiB). The issue lists
  RTX 4090 / A100 / H100 as targets; this single-box track runs on the Ada card
  available to the project. The result is a *ratio* to cuStateVec on the same
  card, which is hardware-portable in spirit (both backends are memory-bandwidth
  bound at this `n`).
- **Precision:** FP64 throughout (1e-10 oracle tolerance), unlike the Metal track's
  FP32 ceiling.
- **Engines:**
  - *aleph SV* — `CudaSvBackend`, the P5-02 NVRTC butterfly kernels **with the
    P5-06 diagonal-gate routing on** (the production default).
  - *cuStateVec* — `CuStateVecBackend` (P5-03), `custatevecApplyMatrix` per gate.
  - *Aer-GPU* — Qiskit Aer `AerSimulator(method='statevector', device='GPU')`,
    **FP64** (`precision='double'`, confirmed), **gate fusion on** (its default),
    and — checked — `cuStateVec_enable=False`, i.e. Aer runs its *own* batched
    GPU state-vector engine, not cuStateVec.
- **Timing:** best of 5 full runs (fresh allocate → all gates → final
  device→host amplitude sync). Same circuit object in every arm.
- Reproduce: `ALEPH_REPORT_N=28 cargo test -p aleph-cuda --features cuquantum
  --release -- --ignored --nocapture gpu_report` and the companion
  `tests/gpu_report_bench.py` for the Aer-GPU column.

## State-vector: Tier-1 + Tier-2 (n = 28)

| workload      | aleph SV (s) | cuStateVec (s) | aleph / cuStateVec | Aer-GPU (s) |
|---------------|-------------:|---------------:|-------------------:|------------:|
| GHZ           | 3.68         | 3.54           | 1.04×              | 2.94        |
| QFT           | 10.59        | 10.60          | **1.00×**          | 6.09        |
| Grover (4 it) | 19.32        | 19.18          | 1.01×              | 3.89        |
| random (d=20) | 50.23        | 41.19          | 1.22×              | 23.42       |
| QPE           | 11.53        | 11.53          | **1.00×**          | 6.29        |
| VQE (8 layers)| 29.40        | 25.92          | 1.13×              | 10.19       |
| QAOA (p=4)    | 22.39        | 19.29          | 1.16×              | 6.54        |

**Exit gate (vs cuStateVec standalone): worst 1.22× ≤ 1.5× → cleared on every cell.**

Two regimes against cuStateVec:

- **Diagonal-dominated** (QFT, QPE, Grover): ~**1.00×**. These are controlled-Phase
  / multi-controlled-Z heavy, and the P5-06 custom diagonal kernel does the same
  single coalesced phase-multiply cuStateVec's best path would — so aleph is
  neck-and-neck.
- **Dense-2q-dominated** (random, VQE, QAOA): **1.13–1.22×**. The generic
  `apply_kq` 4×4 butterfly is a touch behind cuStateVec's hand-tuned 2-qubit
  kernels; the gap is real but comfortably inside the 1.5× gate. Closing it
  (specialised CNOT/2q kernels) is the obvious next SV optimisation.

GHZ's 1.04× is allocation/readout-dominated (28 gates over a 4 GiB state), so the
ratio there is noise around parity.

### The Aer-GPU gap is real — and it points at the next lever

Aer-GPU is **2–5× faster than both aleph and cuStateVec**, and it is *not* a
precision trick (FP64, confirmed) nor cuStateVec under the hood
(`cuStateVec_enable=False`). Two things explain it, both **above the kernel**:

1. **Gate fusion (the bigger half).** Aer fuses runs of gates into larger
   unitaries before simulating, cutting the number of full-state passes — and
   state-vector sim is memory-bandwidth bound, so passes *are* the cost. Turning
   Aer's fusion **off** moves it back toward the per-gate engines: Grover 3.89 →
   5.33 s, QAOA 6.54 → 15.03 s. (QFT barely moves, 6.09 → 6.43 s — its
   controlled-phase chain doesn't fuse into a small dense block.)
2. **Batched dispatch.** Even fusion-off, Aer beats our cuStateVec column (QFT
   6.4 vs 10.6 s): Aer's engine streams many gates per launch with little
   per-gate host work, where our P5-03 path issues a `GetWorkspaceSize` query +
   `ApplyMatrix` per gate.

Neither is a state-vector *kernel* deficiency — aleph already has the IR gate-
fusion passes (`Fuse2q`, the P2-08 diagonal-run fusion) that close item 1 on the
CPU pipeline; they are simply **not yet fed into the GPU backend**. Wiring the
fusion pass ahead of `CudaSvBackend` (and batching gate dispatch to cut per-gate
overhead) is the concrete next optimisation, and the Aer column quantifies the
prize (~2–5×). This is firmly *future work*: the Phase-5 exit metric is defined
against cuQuantum standalone, which aleph meets.

## Stabilizer: the complementary win (Tier-2 surface-code / Clifford)

cuQuantum does **not** simulate stabilizer circuits, so the Tier-2 surface-code /
Clifford showcase is where aleph's GPU stabilizer (P5-07) stands alone. On random
Clifford traffic it beats **Stim's `TableauSimulator` by 3–12×** and the CPU
`aleph-stab` by 2.5–12× across `n = 1000…65000` (`docs/perf/p5-07-gpu-stabilizer.md`).
This is the "massive QEC on GPU" niche the roadmap calls out — complementing
rather than competing with cuQuantum.

## The Phase 5 GPU stack

| ticket | what it added |
|--------|---------------|
| P5-01  | `aleph-cuda` foundation (cudarc, context, typed device buffers) |
| P5-02  | FP64 state-vector butterfly kernels via NVRTC |
| P5-03  | cuStateVec (cuQuantum) backend |
| P5-04  | retaining stream-ordered memory pool |
| P5-05  | GPU-resident readout (measure/sample/expectation/probabilities) |
| P5-06  | custom diagonal-gate kernels (beat cuStateVec 1.7–2.4× in the L2-resident regime) |
| P5-07  | GPU stabilizer tableau (beats Stim 3–12×) |
| P5-08  | this report |

## Verdict

Phase 5 (CUDA / cuQuantum) **exit metric met**: aleph's FP64 GPU state vector is
within 1.5× of cuStateVec on all Tier-1 + Tier-2 workloads (worst 1.22×, median
~1.01×), and the GPU stabilizer opens a regime cuQuantum does not serve. The
strategy stated in ROADMAP §5 — *integrate cuQuantum, do not try to beat it at
dense SV; beat/complement it where it does not optimise* — holds end to end.

**Highest-ROI follow-up (post-Phase-5):** feed the existing IR gate-fusion passes
into `CudaSvBackend` and batch gate dispatch. The Aer-GPU column shows the prize
is ~2–5× — and it is bandwidth-pass reduction the project already knows how to do
on the CPU side, not new kernels.
