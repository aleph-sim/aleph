# Stage 0 — Qiskit Aer baseline on EPYC (design)

> **Scope.** First task in the Phase 1 completion plan
> (`docs/superpowers/plans/2026-05-26-phase1-completion.md` § Stage 0). Establishes
> a concrete single-thread baseline vs Qiskit Aer on the EPYC bench server so we
> know where we stand vs ROADMAP § 7's "≤ 2× Qiskit at 25 qubits" exit
> criterion **before** investing weeks in further SIMD / IR-opt work.
>
> **Outcome.** Informational only — the numbers do **not** gate Phase 1
> completion. They inform priority for Stage 1-3.
>
> **Ticket type.** `[meta]` PR — no production code changes, only benchmarking
> scaffolding, generated artefacts, and a perf report.

---

## 1. Goal

Produce a reproducible, single-thread, same-circuit comparison between aleph's
fastest x86 backend (`NaiveSvBackend` post-P1-03, i.e. AoS + AVX-512) and
`AerSimulator(method='statevector')` on three canonical workloads at n = 20:

1. **QFT-20** — Nielsen-Chuang § 5.1 textbook QFT.
2. **Grover-20** — 1 marked state, 5 iterations of (oracle + diffusion). (Planned
   as 10 originally; dropped to 5 because 10-iter post-transpile is ~192k gates,
   over the § 8 risk cap. See § 4.3.)
3. **Random-20** — brick-wall random circuit, depth 20 (single-qubit Rz/Rx
   layers interleaved with even/odd CNOT layers).

All three circuits flow through a **single source of truth** (Qiskit), are
exported to OpenQASM 3, and run on both engines from the same `.qasm` file.

## 2. Non-goals

- **Not** an optimisation ticket. No backend changes, no kernel changes.
- **Not** a multi-thread comparison. Qiskit is single-threaded (`OMP_NUM_THREADS=1`,
  `max_parallel_threads=1`); aleph is single-threaded by current implementation.
- **Not** a 25-qubit measurement. n = 20 because (a) `qft/n20` is the existing
  bencher.dev anchor, (b) 25 qubits requires ~512 MiB per state vector and
  pushes Grover wall-clock into minutes per run. ROADMAP § 7's 25-qubit
  criterion gets re-measured at Phase 1 closure (P1-14), not now.
- **Not** a Qiskit Aer multi-method comparison (`density_matrix`, `mps`, etc.).
  Statevector only.
- **Not** the implementation or extension of aleph's existing QASM3 emitter
  (`aleph_parser::emit`, added in P0-08). Circuit construction stays Qiskit-side
  for this ticket.

## 3. Deliverables

A single squash-merge PR titled `[meta] Qiskit Aer baseline on EPYC` adding:

```
scripts/qiskit-baseline/
├── README.md             # how to reproduce on EPYC (and locally)
├── requirements.txt      # pinned qiskit, qiskit-aer, numpy versions
├── run.py                # builds circuits, transpiles, exports QASM, times Aer
└── circuits/             # checked-in QASM3 artefacts (deterministic)
    ├── qft_n20.qasm
    ├── grover_n20_iters5.qasm
    └── random_brickwall_n20_d20.qasm
benches/benches/qiskit_baseline.rs   # criterion bench reading the .qasm files
docs/perf/phase1-vs-qiskit.md        # populated report with the comparison table
```

No changes to `crates/` source. The Rust bench depends on `aleph-parser` (already
a workspace member) and `NaiveSvBackend` (the canonical fast x86 path post-P1-03,
per ADR 0008). A parser regression test (`crates/aleph-parser/tests/qiskit_baseline_fixtures.rs`)
guards against future Qiskit versions emitting gates the parser doesn't accept.

> **Note added 2026-05-26 during implementation:** an earlier draft of this spec
> promised to file a follow-up `[infra]` issue for "QASM3 emitter for aleph-ir".
> That promise is **moot** — `aleph_parser::emit` already exists (added in P0-08
> Task 15, `crates/aleph-parser/src/emit.rs`). No follow-up issue is needed.

## 4. Why this design

### 4.1 Single source of truth (Qiskit → QASM → aleph)

Two alternatives were considered and rejected:

- **Two native builders (Python + Rust):** symmetry breaks silently. Gate-count
  parity is easy to lose when a contributor adds a layer to one but not the
  other. The whole point of a baseline is that the *same circuit* runs on both
  engines.
- **aleph → QASM → Qiskit (reverse direction):** `aleph_parser::emit` already
  exists (P0-08 Task 15), so the emitter side is free. The actual blocker is
  that Qiskit becomes the consumer rather than the producer — we'd need to build
  Grover and QFT in aleph-ir, including a multi-controlled-Z primitive, which
  doesn't exist yet. That's days of work to a half-day ticket and forces aleph
  to be the single source of truth for circuit semantics, which inverts the
  baseline's purpose (we want to measure aleph against Qiskit's idea of these
  circuits, not the other way round).

Qiskit's `QuantumCircuit.qasm()` / `qiskit.qasm3.dumps()` is mature. aleph-parser
already handles OpenQASM 3 (per P0-08). Transpiling to a restricted basis
(`['h', 'x', 'z', 'rz', 'rx', 'ry', 'cx', 'cz', 'ccx', 'p']`) before export
ensures Qiskit doesn't emit anything aleph can't parse.

### 4.2 NaiveSvBackend (AoS + AVX-512) as the headline aleph number

Per ADR 0008, AoS dominates SoA on x86 for QFT-20 after P1-03's AVX-512 kernel
landed. The headline `aleph` column in the comparison table uses `NaiveSvBackend`.
For triangulation, the appendix table also records `SoaSvBackend` numbers so the
reader can see the SoA / AoS gap on the same workload.

### 4.3 Grover at 5 iterations

A single iteration is too small (oracle + diffusion runs in milliseconds;
measurement noise dominates). The full textbook count for n=20 (~804 iters)
balloons wall-clock to minutes per run and adds nothing diagnostic — the kernel
mix is identical.

**Implementation choice (logged 2026-05-26):** the original plan called for 10
iterations, but 10-iter transpiles to ~192k gates on the 10-gate basis we use
— double the § 8 risk-row cap of 100k. Dropped to 5 iters (96,210 gates after
transpile), which still gives Grover roughly the same wall-clock weight as QFT
while staying inside the cap.

### 4.4 Random circuit shape

`random_brickwall_circuit(20, 20)` is the existing aleph builder. We re-create
the *same structure* in Qiskit (same deterministic angle formula
`((layer as f64) + (q as f64) * 0.37).cos()`) inside `run.py`. This is the one
workload where the Python builder mirrors a Rust builder — but only to define
the *circuit*; the *running* of it goes through QASM on both sides.

## 5. Methodology

### 5.1 Python harness (`scripts/qiskit-baseline/run.py`)

For each of the three workloads:

1. **Build the QuantumCircuit** in Qiskit using high-level constructs:
   - QFT: `qiskit.circuit.library.QFT(20, do_swaps=False, inverse=False)`.
     `do_swaps=False` matches `aleph_benches::qft_circuit`'s comment that closing
     SWAPs are omitted.
   - Grover: `qiskit.circuit.library.GroverOperator(oracle, insert_barriers=False)`
     with a 1-marked-state oracle (state |0…01⟩), wrapped 5× in sequence (see § 4.3).
   - Random brick-wall: built explicitly in Python mirroring
     `aleph_benches::random_brickwall_circuit` exactly — for each layer
     `l ∈ 0..20` and qubit `q ∈ 0..n`, emit `rz(cos(l + q*0.37), q)` and
     `rx(cos(l + q*0.37) * 1.13, q)`, then a CNOT layer with offset `l & 1`
     pairing `(offset, offset+1), (offset+2, offset+3), …`. The Python and
     Rust formulas must match bit-for-bit (verified via gate-count + first
     few CNOTs in a smoke test inside `run.py`).
2. **Transpile** to the restricted basis:
   `transpile(qc, basis_gates=['h','x','z','rz','rx','ry','cx','cz','ccx','p'],
              optimization_level=0)`.
   `optimization_level=0` is critical — we want to measure the **engine**, not
   Qiskit's transpiler. Aleph has no equivalent optimiser yet (Stage 2
   addresses this), so transpiling at level 0 keeps the comparison honest.
3. **Export QASM3** to `circuits/<workload>.qasm` via `qiskit.qasm3.dumps`.
4. **Time Aer** by running the loaded QASM through
   `AerSimulator(method='statevector', max_parallel_threads=1,
   max_parallel_experiments=1)`. Use `time.perf_counter` around `simulator.run(qc).result()`. 10 timed iterations (+ 1 untimed warm-up); report median + stdev.
5. **Emit a JSON summary** `results-qiskit.json` so the Rust bench can consume
   the same metadata.

Environment pinning: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. Optionally `taskset -c 0 python run.py` to pin to one
core.

### 5.2 Rust bench (`benches/benches/qiskit_baseline.rs`)

Reads the three `.qasm` files via `aleph_parser::parse_str` (exact entrypoint
verified against P0-08), runs each through `NaiveSvBackend::with_seed(0)`,
times via criterion. Criterion uses its default sampling regime
(`sample_size = 10`, warm-up via its own machinery) — we do **not** roll our
own median loop on the Rust side because criterion's stats are already
calibrated and bencher.dev consumes its output directly.

The `.qasm` files are checked into `scripts/qiskit-baseline/circuits/` so the
bench is hermetic — no Python required to run the Rust side.

Criterion settings: `Throughput::Elements(n · 2^n)` matching existing
`benches/qft.rs` convention, so bencher.dev's elements-per-second metric stays
comparable across phases.

### 5.3 Reproducibility

`scripts/qiskit-baseline/README.md` documents the exact EPYC invocation:

```bash
# On EPYC (root@195.154.249.85), in /tmp/aleph-forensics or similar:
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 taskset -c 0 python run.py        # produces results-qiskit.json
cd ../..
RUSTFLAGS="-C target-cpu=native" cargo bench \
  --bench qiskit_baseline -- --save-baseline phase1-baseline
```

The `requirements.txt` pins exact versions (e.g. `qiskit==1.2.4`,
`qiskit-aer==0.15.1`, `numpy==2.1.3`) so re-runs months later land on the same
Aer.

## 6. Report structure (`docs/perf/phase1-vs-qiskit.md`)

```
# Phase 1 baseline: aleph vs Qiskit Aer (EPYC, single thread)

Date: 2026-05-26
Host: EPYC <model>, <cores>, RAM <X>, kernel <Y>, Rust <Z>, target-cpu=native
Qiskit: <pinned-version>, Aer: <pinned-version>, Python: <X.Y.Z>
Pin: OMP_NUM_THREADS=1, taskset -c 0
Reproducibility: scripts/qiskit-baseline/

## Headline (NaiveSvBackend, AoS + AVX-512 — canonical fast x86 path)

| Workload                     |  aleph (ms) | Qiskit Aer (ms) | aleph / Aer | ROADMAP target |
|------------------------------|------------:|----------------:|------------:|:--------------:|
| qft/n20                      |     <a>     |       <q>       |    <a/q>×   |     ≤ 2×       |
| grover/n20 (5 iters)         |     <a>     |       <q>       |    <a/q>×   |     ≤ 2×       |
| random_brickwall/n20 (d=20)  |     <a>     |       <q>       |    <a/q>×   |     ≤ 2×       |

(Median of 10 runs; stdev in appendix.)

## Appendix — full backend matrix

| Workload | NaiveSvBackend | SoaSvBackend | Qiskit Aer |
| …                                                              … |

## Interpretation

One paragraph: where we stand vs ≤ 2× target. Which workload is weakest. What
Stage 1 (SIMD specialisations) and Stage 2 (IR-opt) are expected to close.
**Does not gate Phase 1 — proceeding to Stage 1 regardless.**

## Reproducing this report

(Link back to scripts/qiskit-baseline/README.md and exact commands.)
```

## 7. Acceptance criteria

- [ ] `scripts/qiskit-baseline/` populated with `requirements.txt`, `run.py`,
      `README.md`, three `.qasm` files.
- [ ] `benches/benches/qiskit_baseline.rs` runs locally via
      `cargo bench --bench qiskit_baseline` and on EPYC under
      `RUSTFLAGS="-C target-cpu=native"`.
- [ ] `docs/perf/phase1-vs-qiskit.md` table fully populated with absolute times,
      ratios, host metadata, version pins, and a one-paragraph interpretation.
- [ ] CI green (clippy + fmt + tests). The new Rust bench file must **compile**
      under `cargo bench --workspace --no-run` (CI already does this); it is
      **not** executed in CI — only on EPYC under `cargo bench`. No new CI job
      added in this PR.
- [ ] Stage 0 ships **regardless of ratio**. The numbers are informational; we
      proceed to Stage 1 even at 5× Aer.
<!-- (Removed 2026-05-26: aleph_parser::emit already exists from P0-08 Task 15;
     no separate issue required.) -->

## 8. Risks & mitigations

| Risk                                                                         | Mitigation                                                                                                                                                                                              |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| aleph-parser doesn't support a gate Qiskit emits after transpile              | Restrict `basis_gates=['h','x','z','rz','rx','ry','cx','cz','ccx','p']` — all of these parse today per P0-08. If a missing gate surfaces, file a follow-up parser ticket and reduce basis further.        |
| Qiskit Aer's `max_parallel_threads=1` doesn't actually pin one thread        | Verify with `taskset -c 0` and `/proc/self/status` Cpus_allowed_list during run. Document the verified pin in the report.                                                                                |
| 10 Grover iterations exceed memory budget (multi-controlled-Z decomposition explodes gate count) | Measure circuit gate count after transpile; if > 100k gates, drop to 5 iters. Document the choice in the report.                                                                                          |
| EPYC bench-server contention with GH Actions runner                           | Use `/tmp/aleph-forensics` per phase1-completion-plan.md breadcrumbs; do not write to runner workdir as root. Run when no CI job is queued. (Optional: `nice -n 19` + `ionice` for fairness.)              |
| Bench results depend on AVX-512 dispatch — local M-series shows nothing useful | Local cargo bench runs the scalar path on M-series. EPYC is the authoritative target. Doc the local-vs-EPYC discrepancy explicitly in README so future contributors don't anchor on local numbers.        |
| Bencher.dev baseline drift if we save a new baseline name                     | Save as `--save-baseline phase1-baseline` and document; do not overwrite `main`.                                                                                                                          |

## 9. Open questions deferred

These are flagged but resolved later, not in this PR:

- **Default backend selection.** AoS now beats SoA on x86 (ADR 0008). Should
  `Backend` default to AoS on x86? Decision belongs in Stage 3 `[meta]` fixup.
- **25-qubit measurement.** ROADMAP § 7 says 25; Stage 0 measures 20. n=25 lands
  in P1-14.
- **QASM emitter.** `aleph_parser::emit` already exists (P0-08); circuit
  construction stays Qiskit-side regardless.

## 10. Workflow

Per the per-ticket workflow established from P0-06 onwards:
brainstorm → spec (**this doc**) → plan → execute → request code review →
fix → squash-merge.

Next step after spec approval: invoke `writing-plans` to author the
implementation plan (`docs/superpowers/plans/2026-05-26-stage0-qiskit-baseline.md`).
