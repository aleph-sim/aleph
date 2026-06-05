# P4-01 — QFT Benchmark + Phase-4 Benchmark/Report Framework — Design

**Issue:** #39 (P4-01). **Depends on:** P1-14 (Phase-1 perf harness/report). **Milestone:** Phase 4.
**Date:** 2026-06-05. **Status:** approved, pre-implementation.

## Goal

Stand up the **shared Phase-4 benchmark + report framework** and make **QFT** its
first consumer, satisfying the P4-01 acceptance criteria:

- QFT runs to 30 qubits on the state vector.
- Results match Qiskit.
- A benchmark report row (in a reproducible, auto-generated report).
- Testing: QFT(|0…0⟩) ∘ inverse-QFT = |0…0⟩ (and the stronger random-state
  round-trip).

The framework is designed so P4-02..07 (Grover, QPE, VQE, QAOA, random, surface
code) plug in as additional workload rows with no new plumbing.

## Decisions (locked during brainstorming)

1. **Scope = unified Phase-4 framework, QFT first consumer** (not a one-off QFT
   bench, not a from-scratch rebuild — reuse the heavy existing Stage-0 infra).
2. **Backbone = shared QASM corpus + unified results JSON + report generator.**
   One committed Qiskit-generated QASM corpus is the single source of truth for
   each circuit; both aleph (criterion) and Aer (Python) execute the *same*
   QASM; each emits into a unified results JSON; a report-generator script
   merges them into `docs/perf/phase4.md`. Measurement and reporting are
   decoupled, each side stays in its native language.
3. **n=30 via cap-lift, measured on EPYC.** The benchmark/`run_optimized` path
   allows n>28 (the existing soft-cap warning stays); n=28/30 are measured on
   the EPYC box (2³⁰×16 B = 16 GiB fits in 123 GiB). n=10..25 run anywhere.

## What already exists (reused, not rebuilt)

- **Shared circuit corpus:** `scripts/qiskit-baseline/circuits/*.qasm` — already
  consumed by *both* the aleph criterion benches and the Aer harness. QFT exists
  at n=15/20/22/25.
- **Aer baseline harness:** `scripts/qiskit-baseline/run.py` + `results-qiskit.json`
  (schema_version 2: per-workload `n`, `family`, `timing_runs`,
  `gate_count_post_transpile`, `qiskit_aer.{samples_s, median_s, …}`).
- **aleph benches:** `benches/benches/qft_scaling.rs`, `qft.rs`, `tier1_scaling.rs`,
  `qiskit_baseline.rs`; canonical run path `aleph_backend::run_optimized`
  (optimize + simulate) on `NaiveSvBackend`; `benches/src/bin/oneshot.rs` for
  peak-RSS.
- **Report format:** `docs/perf/phase1.md` table style (workload / aleph ms /
  Aer ms / ratio / stdev / gates).
- **Oracle QFT equivalence:** `oracle/circuits/qft_{3,5}.qasm` + fixtures.

**What does NOT exist and this ticket adds:** an automated **report generator**
(phase-1/2 tables were hand-written from raw logs), the **n=10/30** corpus +
measurements, the **inverse-QFT round-trip** correctness test, and the
**unified results schema** that P4-02..07 reuse.

## Architecture / data flow

```
 corpus-gen (Qiskit synthesis)
        │  emits canonical QASM, n = 10/15/20/25/30
        ▼
 scripts/qiskit-baseline/circuits/qft_n{10,15,20,25,30}.qasm   ← single source of truth
        │                                   │
        ▼ (parse + run_optimized)           ▼ (transpile + time)
 aleph criterion bench (NaiveSv)      Aer run.py
        │                                   │
        ▼ extract medians                   ▼
 docs/perf/data/phase4-aleph.json     scripts/qiskit-baseline/results-qiskit.json
        └──────────────┬────────────────────┘
                       ▼
       scripts/bench-report/report.py  (deterministic merge)
                       ▼
              docs/perf/phase4.md  (markdown tables)
```

## Components (one responsibility each)

### 1. Reference corpus generation — extend the existing `run.py` export path

The corpus generator **already exists** inside `scripts/qiskit-baseline/run.py`:
`build_qft(n)` uses Qiskit `QFT(num_qubits=n, do_swaps=False)`, and
`transpile_and_export` transpiles to the project basis and writes committed
QASM3 to `circuits/`. That export path **is** the ticket's "reference
implementation" — no separate `gen_circuits.py` is needed.

Changes:
- **Per-family sizes.** Today `N_QUBITS_LIST = [15, 20, 22, 25]` is global across
  all families (`all_workloads()`). Replace it with a per-family size map so QFT
  uses **{10, 15, 20, 25, 30}** without disturbing ghz/grover/random (changing
  the global list would regenerate every family and make Grover/random at n=30
  intractable). Existing families keep their current sizes.
- The generator stays deterministic (no RNG for QFT); re-running produces
  identical QASM. A test regenerates and asserts byte-identity with the committed
  files (drift guard). If a regenerated QFT file differs from the committed one,
  the committed file is updated in this PR with a noted reason (expected: n=15/20/25
  already came from this same path, so they should match; only n=10/30 are new).
- n=22 may be retained or dropped for QFT; AC needs {10,15,20,25,30}. Keep n=22
  only if cheap; not required.

### 2. aleph-side measurement — corpus-QASM bench (not the builder)

- **Use the committed corpus QASM, not the `qft_circuit` builder.** Today
  `qft_scaling.rs` benches `aleph_benches::qft_circuit(n)` (a Rust builder), which
  is a *different* gate list from the exported fixture (a known divergence — see
  the P2-05 "builder QFT ≠ fixture QFT" note). For the framework's
  single-source-of-truth principle, the aleph QFT measurement must parse and run
  the **same** `scripts/qiskit-baseline/circuits/qft_n{N}.qasm` that Aer runs —
  mirroring `tier1_scaling.rs`'s `fixture_path` approach. Add a QFT corpus bench
  (extend `qft_scaling.rs` or add a small `phase4_qft.rs`) that sweeps
  n ∈ {10,15,20,25,30}, parses the corpus QASM, and runs `run_optimized` on
  `NaiveSvBackend`.
- n≥28 is gated behind the existing `scaling-bench` feature so default
  `cargo bench --workspace` / CI never allocate ≥4 GiB; n=28/30 run explicitly on
  EPYC.
- A thin extractor `scripts/bench-report/extract_criterion.py` reads criterion's
  `target/criterion/<group>/<id>/new/estimates.json` (median in ns + the
  point-estimate spread) and writes `docs/perf/data/phase4-aleph.json` in the
  unified schema (below). The committed JSON is the reproducible snapshot
  (mirrors the phase-1/2 practice of committing `docs/perf/data/*`).

### 3. Aer-side measurement — same `run.py` pass (already produces timings)

- The per-family size change in Component 1 automatically extends QFT's Aer
  timings to n=10/30 (run.py exports the circuit **and** times Aer in one pass,
  writing `results-qiskit.json`, schema v2 — no breaking change). n=30 runs on
  EPYC. (Components 1 and 3 are one code change to `run.py`, listed separately
  only because they feed different downstream consumers.)

### 4. Unified results schema + report generator (NEW, reusable)

- **Schema** (`docs/perf/data/phase4-results.schema.md` documents it): a flat
  list of records `{workload, family, n, gates, aleph_ms_median, aleph_rsd,
  aer_ms_median, aer_rsd}`. The aleph snapshot JSON and the Aer
  `results-qiskit.json` are the two inputs; the report generator joins them on
  `(family, n)`.
- **Report generator** `scripts/bench-report/report.py`:
  - Pure function: reads the two input JSONs, joins, computes `ratio =
    aleph_ms / aer_ms`, renders `docs/perf/phase4.md` (phase1.md table style:
    one table per family, a headline summary, a host/toolchain/pins header).
  - Deterministic: same inputs → byte-identical markdown. No network, no RNG.
  - For P4-02..07: a new family appears automatically once its corpus +
    measurements land in the two input JSONs — no generator changes needed.

### 5. n=30 cap-lift — configurable cap on `NaiveSvBackend`

- **The cap is a HARD reject, not a warning.** `NaiveSvBackend::allocate`
  returns `BackendError::TooManyQubits` for `num_qubits > MAX_NAIVE_QUBITS`
  (=28, `backend.rs:45`). So n=30 cannot run as-is.
- **Make the cap configurable, default preserved.** Add a per-instance qubit cap
  to `NaiveSvBackend` (field defaulting to `MAX_NAIVE_QUBITS` = 28) with a
  builder/setter `with_qubit_cap(u32)` (name TBD to match crate style, e.g.
  `with_max_qubits`). `allocate` checks the instance cap. Default behavior is
  **unchanged** (a laptop still gets a clean `TooManyQubits` at n=29, no silent
  16 GiB allocation); the n=30 EPYC bench/`oneshot` explicitly opts in via
  `with_qubit_cap(30)` (or 32). The user authorized this in brainstorming
  ("lift cap, measure on EPYC").
- This is the minimal change that satisfies the AC without weakening the default
  guardrail for ordinary users. The CLI default run path is untouched.

## Correctness

- **Oracle vs Qiskit (small n):** already covered by `oracle/circuits/qft_{3,5}`
  + fixtures — keep, do not duplicate.
- **Inverse-QFT round-trip (new):** a test in the QFT bench-fixtures / aleph
  test suite: build QFT(n) and its inverse, apply both to a *generic* random
  state (not just |0…0⟩, per the P1-13 lesson that |0…0⟩ oracles miss bugs);
  assert the result equals the input to 1e-10. Also assert the |0…0⟩ case from
  the ticket's testing requirement. n small enough to run in CI (e.g. n≤12).
- **Corpus identity ⇒ "results match Qiskit":** because both sides execute the
  same committed QASM, equal gate lists are guaranteed by construction; the
  oracle test pins numerical equality at small n.

## Testing & reproducibility

- **Report generator unit test:** feed a fixed pair of tiny input JSONs, assert
  the rendered markdown equals a committed golden file (deterministic).
- **Corpus generator test:** regenerate the full set n∈{10,15,20,25,30} and
  assert byte-identity with the committed QASM. Generation is cheap at every n
  (only *execution* of n≥28 is memory-heavy), so all sizes are CI-safe to
  regenerate-and-compare; this guards against silent corpus drift.
- **Inverse-QFT test:** runs in normal `cargo test` (small n).
- **EPYC measurement run:** documented in `docs/perf/phase4.md` header
  (host, toolchain, `RUSTFLAGS`, Aer version, thread pins) exactly as phase1.md;
  the heavy n=25/30 benches stay non-gating (feature-gated), run manually on the
  idle EPYC box (per the idle-check rule).

## Out of scope (YAGNI)

- P4-02..07 algorithms themselves (only the framework + QFT row here).
- A live dashboard / bencher.dev wiring (already exists separately for CI; the
  Phase-4 report is the committed markdown artifact).
- Multi-backend rows beyond `NaiveSvBackend` for QFT (the canonical fast path);
  the schema permits adding columns later without a redesign.
- Automated EPYC orchestration (measurement is a documented manual step).

## Reusability contract for P4-02..07

To add a future benchmark family X:
1. Add X's reference generator output to the corpus (`gen_circuits.py`).
2. Add X to `run.py` (Aer) and the aleph bench sweep.
3. Drop the two measurement snapshots into the unified JSONs.
4. Re-run `report.py` — X's rows appear in `phase4.md`. No framework code change.
