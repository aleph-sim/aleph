# [P1-14] Phase 1 performance report — design

> Status: approved (brainstorm) — pending implementation plan.
> Issue: BACKLOG `[P1-14] Phase 1 performance report` (#26).
> Depends on: P1-01 … P1-13 (all Phase-1 work). This is the Phase-1 EXIT criterion.

## 1. Goal & deliverable

Publish a comprehensive single-thread performance report comparing the
current (post-P1-13) `aleph` backend against Qiskit Aer single-thread,
measured on the EPYC bench host, in `docs/perf/phase1.md`. The report
is the Phase-1 exit artifact: it states, with numbers, whether
ROADMAP § 7's "single-thread within 2× of Qiskit Aer for QFT, Grover,
random circuits at 25 qubits" is met, and files follow-up issues for
any miss.

Acceptance criteria (BACKLOG #26):

- [ ] Report committed (`docs/perf/phase1.md`).
- [ ] All Tier-1 algorithms benchmarked.
- [ ] Targets met, or specific follow-up issues filed for misses.

## 2. Scope

- **Matrix:** {GHZ, QFT, Grover, random-brickwall} × n ∈ {15, 20, 22, 25}.
- **aleph backend:** `NaiveSvBackend` (AoS + AVX-512 — the canonical
  fast x86 path per ADR 0008) across the full matrix. `SoaSvBackend`
  as an appendix at n ≤ 20 only (already known ~2.3× slower; not worth
  n ≥ 22 EPYC time).
- **Reference:** Qiskit Aer `AerSimulator(method='statevector',
  max_parallel_threads=1)`, single-thread-pinned.
- **Per-cell metrics:** total time (criterion/median), time-per-gate
  (total ÷ post-transpile gate count), peak memory.

## 3. Harness (extend Stage-0 infrastructure)

Stage 0 (PR #81) left a working harness; P1-14 extends it rather than
rebuilding:

- `scripts/qiskit-baseline/circuits/` holds **shared OpenQASM circuits**
  (currently `qft_n20`, `grover_n20_iters5`, `random_brickwall_n20_d20`).
  Both `aleph` (via `aleph-parser`) and Aer load the SAME `.qasm`, which
  guarantees identical gate counts. P1-14 generates the full set:
  `ghz_n{15,20,22,25}`, `qft_n{15,20,22,25}`, `grover_n{15,20,22,25}`
  (fixed iteration count, documented), `random_brickwall_n{15,20,22,25}`
  (fixed depth + seed, documented).
- `benches/benches/qiskit_baseline.rs` (the cross-crate `benches/` crate)
  runs the `aleph` side via criterion over those circuits. P1-14 adds
  the new cells (GHZ, n=15/22/25) to the `qiskit_baseline` group.
- `scripts/qiskit-baseline/run.py` runs the Aer side over the same
  `.qasm` files. P1-14 extends it to the full matrix.
- Circuit generation should be deterministic and committed (the `.qasm`
  files in `circuits/`) so the report is reproducible without rerunning
  a generator.

## 4. EPYC execution protocol

- Host: `ssh root@195.154.249.85` (self-hosted EPYC 8124P; see
  `aleph-bench-server` memory). Otherwise-idle during measurement; do
  not push to `benches/**` mid-run (CI Bench races the same runner —
  Stage-0 lesson).
- **Pinning:** Aer under `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 taskset -c 0 python …`; verify Aer
  `max_parallel_threads=1` at runtime. aleph bench built with
  `RUSTFLAGS="-C target-cpu=native"`, AVX-512 emission verified via
  `objdump` (as Stage 0).
- **Peak memory:** report both (a) theoretical state-vector size
  `2^n × 16 B` and (b) measured Maximum RSS via `/usr/bin/time -v` on a
  single-shot run of each side (aleph one-shot binary and the Aer
  script). Measured RSS includes runtime/buffer overhead; theoretical
  is the floor.
- **Sampling policy (physical reality of n=25):** full criterion
  samples where a cell runs in < ~10 s. For multi-minute cells
  (Grover/random at n=22/25 — Grover n=25 ≈ 50 min/iteration) use a
  reduced count (`sample_size = 10` floor, or a single timed run with a
  warm-up), accepting higher variance. **Every cell is measured**; the
  report's RSD/sample table discloses per-cell sample_size and relative
  stdev so noisy cells are visible, never hidden (no-silent-caps).

## 5. Report structure (`docs/perf/phase1.md`)

- **Header:** date, host, toolchain (Rust version, `target-cpu=native`,
  AVX-512 verified), Python/Qiskit/Aer/numpy/scipy versions, pin
  command, link to `scripts/qiskit-baseline/README.md`.
- **Per-algorithm headline tables:** one table per algorithm, rows
  n ∈ {15,20,22,25}, columns: `aleph (ms)`, `Aer (ms)`, `aleph/Aer`,
  `time/gate (ns)`, `peak RSS (MiB)`, `theoretical (MiB)`, `≤2× verdict`.
- **ROADMAP § 7 exit verdict:** explicit met / partially-met summary at
  n=25 for QFT, Grover, random (GHZ is a bonus trend line).
- **Trend section:** how `aleph/Aer` moves n=15→25 per algorithm (does
  the gap grow or hold).
- **Backend appendix:** NaiveSv vs SoA at n ≤ 20.
- **RSD / sample-count table:** per-cell `sample_size` + relative stdev.
- **Interpretation + Known gaps:** what is over 2×, why, with links to
  the follow-up issues filed in § 6.
- **Reproducibility:** exact commands.
- **Banner on `docs/perf/phase1-vs-qiskit.md`:** mark it "superseded by
  `phase1.md` (Stage-0 snapshot, 2026-05-27)"; add reciprocal links.

## 6. Misses → follow-ups

- For each cell **> 2× Aer** (most likely QFT at n=25): add a BACKLOG
  entry (Phase-2 or a "Phase-1 follow-up" section) and file a GitHub
  issue via the standard `scripts/sync-issues.sh` / `CREATE ISSUES.md`
  flow. Link each from the report's "Known gaps".
- If everything is ≤ 2×, state the exit criterion fully met and file no
  issues.

## 7. Verification

- Numbers in the report are reproducible via the documented commands;
  gate counts are identical between aleph and Aer (same `.qasm`).
- Any new harness code: `cargo test`/`clippy`/`fmt` clean; new benches
  compile under the `bench.yml` steps (if feature-gated, add the
  explicit step — P1-09 lesson).
- Bench CI stays green.

## 8. Out of scope (deferred)

- `[meta]` Phase-1 fixup: flip ROADMAP § 7 exit checkboxes, update
  CLAUDE.md "Project Overview" to mark Phase 1 complete — a separate
  follow-on ticket after this report lands.
- Multi-thread / Phase-2 performance (P2-xx).
- Any new optimisation work to close a > 2× gap — that becomes a filed
  follow-up issue, not part of this report.
