# P5-06 — Custom CUDA kernels for niches cuQuantum misses

cuStateVec (cuQuantum) is an excellent general state-vector engine, but its
public gate entry point — `custatevecApplyMatrix` — treats every gate as a dense
`2^k × 2^k` block: it gathers each amplitude's `2^k`-sized neighbourhood and does
a full block matvec. For a **diagonal** gate that is wasted work. A diagonal gate
multiplies each amplitude by a single phase; there is no partner to gather and no
matvec. This issue identifies that regime, implements a custom kernel for it, and
wires it into the per-gate backend selection.

## The niche: diagonal gates

A gate `U` is diagonal when `U|x⟩ = d_x|x⟩` — only the phase of each amplitude
changes. This covers a large, important slice of real circuits:

| circuit family | diagonal gates that dominate it |
|----------------|---------------------------------|
| **QFT** | controlled-`Phase(θ)` (the `O(n²)` body) |
| **QAOA / Trotterised Ising** | `Rz`, `ZZ`-rotations (`Cz`, `CRz`) |
| **phase oracles, Grover diffusion** | multi-controlled `Z` |
| **Clifford+T diagonal layers** | `Z`, `S`, `T` |

For these, the dense path moves and multiplies far more than necessary.

## The custom kernel

`src/sv/diag.cu` adds two FP64 kernels, one thread per amplitude, **in place**:

- `apply_diag_1q` — single-qubit diagonal (`Z/S/T/Rz/Phase`, and any of them
  under external controls: `CZ`, `CPhase`, multi-controlled `Z`). The two
  diagonal entries `d0, d1` ride in the by-value uniform, so there is **no matrix
  upload** at all — just `amps[i] *= (bit ? d1 : d0)` gated by the control mask.
- `apply_diag` — general `k`-qubit diagonal (`Cz`/`CRz` at `k=2`, `Ccz` at
  `k=3`): assemble the local index `l` from the operand bits of `i` and multiply
  by `diag[l]` (a `2^k`-entry table in scratch).

Compared with the dense `apply_kq` (and with `custatevecApplyMatrix`):

- **half the memory traffic** for a 1q gate — one read + one write per amplitude,
  versus the dense kernel's gather/scatter of an amplitude *pair*;
- **~1 complex multiply per amplitude** instead of a `2^k`-wide block matvec
  (`4` cmuls/amp for a 2q dense gate, `8` for 3q);
- coalesced, fully parallel, no shared memory, no workspace query.

Gate detection is **numeric, not gate-type-based** (`common::diagonal_of`): a
gate routes to the diagonal kernel iff its extracted matrix has all off-diagonal
entries within `AMPLITUDE_TOL` of zero. This keeps the dispatch backend-agnostic
(CLAUDE.md: "don't hardcode gate types in kernels") and automatically catches any
diagonal unitary, including user-supplied ones.

## Integration with backend selection

Both GPU backends embed a `DiagKernels` and divert diagonal gates to it per-gate:

- `CudaSvBackend` (hand-written): diagonal → `apply_diag*`, else `apply_1q` /
  `apply_kq`.
- `CuStateVecBackend` (cuQuantum): diagonal → `apply_diag*`, else
  `custatevecApplyMatrix`.

The routing defaults on; `with_custom_kernels(false)` forces the dense /
cuStateVec path. That switch is the A/B arm of the benchmark and is pinned in the
oracle test so both routings are proven equivalent.

## Correctness

`tests/diag_oracle.rs` pins **both** backends, **both** routings, against the CPU
`NaiveSvBackend` at the full FP64 tolerance `1e-10`, over `n = 2..=10`:

- a `diag_mix` circuit that exercises every diagonal path — 1q (`Z/S/T/Rz/Phase`),
  controlled-`Phase`, multi-controlled `Z`, `Cz`/`CRz` (`k=2`), `Ccz` (`k=3`) —
  interleaved with non-diagonal gates (`Ry/Cnot/H`) that must stay on the dense
  path;
- a full QFT.

Both the custom path and the forced dense/cuStateVec path match the CPU exactly,
so the routing is a **pure optimisation**. The pre-existing `sv_oracle` Tier-1
suite (QFT, Grover) now flows through the diagonal kernels by default and still
passes.

## Performance

RTX 4000 SFF Ada (sm_89), FP64, `cargo test --release`, best of 8 warm runs.
Workload: a diagonal-only circuit — `depth = 200` layers of (`Rz` on every qubit)
+ a `Cz` brickwall (`≈ 7000` gates). Same circuit in every arm; the only variable
is which kernel runs the diagonal gates.

| n  | state    | SV custom vs dense | cuStateVec **custom vs cuQuantum** |
|----|----------|-------------------:|-----------------------------------:|
|  4 | 256 B    |             1.47×  |                             1.08×  |
|  6 | 1 KiB    |             1.50×  |                             1.13×  |
|  8 | 4 KiB    |             1.56×  |                             1.08×  |
| 10 | 16 KiB   |             2.43×  |                             1.07×  |
| 12 | 64 KiB   |             2.44×  |                             1.07×  |
| 14 | 256 KiB  |             2.17×  |                             1.27×  |
| 16 | 1 MiB    |             2.70×  |                             1.72×  |
| 18 | 4 MiB    |             3.35×  |                         **2.24×**  |
| 20 | 16 MiB   |             3.31×  |                         **2.38×**  |
| 24 | 256 MiB  |             1.02×  |                             0.98×  |

**The niche is intermediate `n` (≈10–20).** There the working set is small enough
to be largely **L2-resident**, so the gate is *not* DRAM-bandwidth bound — it is
bound by per-gate compute and dispatch, exactly what the diagonal kernel slashes:
no `2^k` block matvec (a `Cz` dense apply does 4 cmuls/amp; diagonal does 1), and
for cuStateVec no per-gate `custatevecApplyMatrixGetWorkspaceSize` query + dense
dispatch. Custom beats cuStateVec **2.2–2.4×** at n=18–20 and the hand-written
dense path **2–3.4×** across n=10–20.

At **large `n` (≥24)** the `2^n` state spills to DRAM and every kernel saturates
memory bandwidth — each reads and writes the full state once regardless of how
many FLOPs it does — so custom ≈ dense ≈ cuStateVec (~1.0×). This is the expected
"memory bandwidth is the bottleneck, not FLOPS" wall (CLAUDE.md). The custom
kernel never *loses* meaningfully; it wins big precisely in the cache-resident
regime cuStateVec's generic dense apply over-serves.

Reproduce (the crossover sweep):

```bash
ALEPH_DIAG_DEPTH=200 ALEPH_DIAG_REPS=8 cargo test -p aleph-cuda \
  --features cuquantum --release -- --ignored --nocapture diag_bench_sweep
```

## Scope / follow-ups

- Other niches cuStateVec leaves on the table (not in this PR): **permutation
  gates** (`X`, `Swap` — pure index remap, zero FLOPs) and **gate fusion of
  diagonal runs** (a string of phases on the same qubits collapses to one
  multiply). The diagonal kernel is the highest-value first cut; these extend the
  same per-gate-dispatch machinery.
- The general `apply_diag` reads its `2^k` table from global scratch; for `k ≤ 3`
  it is tiny and L2-resident, but a constant-memory or by-value table would shave
  the last upload for `CZ`/`CRz`/`Ccz`.
