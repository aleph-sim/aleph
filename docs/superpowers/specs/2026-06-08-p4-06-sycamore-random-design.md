# P4-06 — Random circuit benchmark (Sycamore-style)

**Issue:** #44 · **Milestone:** Phase 4 · **Depends on:** P1-14
**Status:** design approved 2026-06-08

## Goal

Add a `sycamore` benchmark family — a Sycamore-style random circuit, the
worst case for state-vector simulation (maximum entanglement) — and compute a
linear cross-entropy benchmarking (XEB) value that confirms Porter–Thomas
statistics (XEB ≈ 1 for our noiseless simulator). The family plugs into the
existing Phase-4 corpus → bench → report pipeline (P4-01..05) as new rows.

Sizes: **n = 20, 24, 28, 30**, depth **20** cycles.

## Acceptance criteria (from BACKLOG #44)

- [ ] Runs at n = 30 on the state-vector backend.
- [ ] Benchmark report row.
- [ ] Linear XEB value computed (≈ 1, since we are noiseless).

## Locked design decisions

1. **New `sycamore` family** — distinct from the existing Phase-1
   `random_brickwall` fixture (which is deterministic Rz+Rx, *not* Sycamore).
   `random_brickwall` is left untouched.
2. **CZ** as the two-qubit entangling gate — existing AVX-512 kernel, Clifford,
   brick-wall friendly; with {√X, √Y, √W} singles it still produces chaotic
   Porter–Thomas dynamics (Google's earlier supremacy circuits used CZ). No new
   kernel.
3. **Exact XEB** `= 2ⁿ·Σ|aᵢ|⁴ − 1` (the collision-probability form), computed
   analytically from the full final state vector. No sampling / RNG / shots.
   Noiselessly ≈ 1 by construction.
4. **Seeded deterministic gate placement** — the √X/√Y/√W choice per
   `(layer, qubit)` comes from a deterministic seeded hash, *not* numpy RNG, so
   the corpus is byte-reproducible and the choice is independently re-derivable.
   The generator lives in `run.py`; the committed QASM corpus is the single
   source of truth (aleph consumes it, no Rust builder duplication).

## Components

### 1. Corpus generator — `scripts/qiskit-baseline/run.py`

- `build_sycamore(n, depth, seed) -> QuantumCircuit`:
  - Per cycle: a single-qubit layer where each qubit gets one of {√X, √Y, √W}
    selected by a deterministic seeded hash of `(layer, qubit, seed)`, applying
    **Google's rule** that a qubit's gate differs from its gate in the previous
    cycle (ensures variety / faster scrambling).
  - Followed by a CZ brick-wall: alternating-offset adjacent pairs
    (`offset = layer & 1`), the 1-D simplification the spec calls for (not the
    real 2-D grid ABCD coupler pattern).
  - √X/√Y/√W are emitted as explicit `U(θ,φ,λ)` gates, then transpiled to the
    existing `BASIS_GATES` ({h,x,z,rz,rx,ry,cx,cz,ccx,p}) at `optimization_level=0`.
    No change to `BASIS_GATES`, the parser, or the kernels.
- Wire into the family machinery exactly like prior families:
  `FAMILY_SIZES["sycamore"] = [20, 24, 28, 30]`, `FAMILY_BUILDERS`,
  `corpus_stem`/`workload_key` → `sycamore_n{n}_d{depth}`. Constants
  `SYCAMORE_DEPTH = 20`, `SYCAMORE_SEED`.
- `run.py` also computes Aer's XEB from `save_statevector` and records it
  (see Reporting).

### 2. aleph benchmark — `benches/benches/phase4_sycamore.rs`

Mirrors `phase4_qft.rs`: reads the committed corpus QASM, runs `run_optimized`
on `NaiveSvBackend` with `with_qubit_cap(32)`.
- `SMALL_N = [20, 24]` under criterion.
- `[28, 30]` behind the `scaling-bench` feature + the `oneshot` bin (n=30 is
  16 GiB — the QFT-30 pattern: 10-sample criterion is impractical).

### 3. XEB helper — `benches/src/lib.rs`

`linear_xeb(amps: &[Complex]) -> f64 = (1 << n) as f64 * Σ|aᵢ|⁴ − 1`, where
`n = log2(amps.len())`. Pure f64. Used by the bench/oneshot (to print and
emit the value) and by the gated test.

## Data flow

```
run.py ──► corpus QASM  (sycamore_n{20,24,28,30}_d20.qasm)
       └─► Aer single-thread timings  ──► docs/perf/data/phase4-aer.json
       └─► Aer XEB (save_statevector)  ──► docs/perf/data/phase4-xeb.json

phase4_sycamore.rs / oneshot ──► criterion timings ──► docs/perf/data/phase4-aleph.json
                             └─► aleph XEB           ──► docs/perf/data/phase4-xeb.json

report.py (merge) ──► docs/perf/phase4.md  (Sycamore section: timing + XEB columns)
```

## Testing strategy

A random circuit has no structural oracle (no inverse-recovery / all-ones /
amplification property). The correctness gate is a triple in
`benches/tests/sycamore_xeb.rs` — reads the committed corpus; **n = 20 in fast
CI**, n = 24/28 `#[ignore]`-d to nightly:

1. **`run` ≡ `run_optimized` full-amplitude equality (1e-12)** — the strong
   internal oracle. The fusion / diagonal-fusion / FuseKq passes rewrite the
   √X/√Y/√W runs and the CZ brick-wall, so this catches pass corruption on the
   exact path the bench times. (P4-03 lesson: oracle must cover `run_optimized`,
   not just `run`.)
2. **Normalization Σp = 1 (1e-10)** on both paths — unitarity sanity.
3. **XEB band + golden** — `linear_xeb` within a band around 1 (tolerance set
   empirically at n=20, where the Porter–Thomas bias ~2/2²⁰ is negligible), plus
   a pinned golden value for byte-stable regression.

**Transitive Aer correctness:** the corpus is the shared source of truth and
aleph's parser + {rz,rx,ry,cz} kernels are already validated to 1e-10 vs Aer
across QFT/QPE/Grover. We deliberately do **not** add a dedicated
Sycamore amplitude-vs-Aer test (would require a new pyo3 statevector getter =
scope creep). The Aer-XEB vs aleph-XEB agreement shown in the report is the
lightweight cross-simulator sanity check.

Plus a `run.py` self-test that `build_sycamore` is deterministic for a fixed
seed (byte-stable corpus).

## Reporting

- `report.py`: add `"sycamore": "Sycamore random"` to `FAMILY_TITLES`; extend
  the renderer to surface an **XEB column** in the Sycamore section (timing rows
  auto-render; the XEB column is the one genuinely new piece of tooling). Golden
  output test updated. Tests stay stdlib `unittest`.
- XEB values are committed in `docs/perf/data/phase4-xeb.json`
  (`{ "sycamore_n{n}_d20": { "aleph_xeb": .., "aer_xeb": .. } }`).

## EPYC measurement workflow

Single-thread **both** sides (`RAYON_NUM_THREADS=1` for aleph; Aer already
`max_parallel_threads=1`). git-bundle transfer, n=30 via the `oneshot` bin,
commit timings + XEB JSON, **then** open the PR (PR triggers CI + Bench on the
shared runner). Cargo at `~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin`;
pinned qiskit 1.2.4 / aer 0.15.1 venv. Budget hours for the n=30 sweep
(16 GiB state, ~minutes/run per side; Aer single-thread is the slow side).

## Out of scope / non-goals

- No 2-D grid coupler pattern (1-D brick-wall per spec).
- No sampled XEB, no noise model (we are exact).
- No new pyo3 bindings; no change to `BASIS_GATES`, parser, or kernels.
- `random_brickwall` (Phase-1 fixture) is not modified.

## References

- Arute et al. (Google), "Quantum supremacy using a programmable
  superconducting processor", *Nature* 574 (2019). XEB definition §; the
  noiseless linear XEB equals `2ⁿ·Σp(x)² − 1`.
