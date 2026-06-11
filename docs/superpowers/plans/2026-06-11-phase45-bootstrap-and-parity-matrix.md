# Phase 4.5 Bootstrap + P4.5-01 Competitive Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up Phase 4.5 (ROADMAP § 5 insert, BACKLOG entries, milestone, GitHub issues) and execute P4.5-01: the competitive benchmark matrix — aleph-MT vs Aer-MT (statevector) and aleph-mps vs Aer `matrix_product_state` — producing `docs/perf/parity.md` with per-cell ≤1.2× verdicts.

**Architecture:** Two PRs. PR 1 (`p45-00-phase-spec`, already carries the spec commit `909fbd9`) adds the phase to ROADMAP/BACKLOG and extends the issue-sync tooling to accept `P4.5-NN` IDs. PR 2 (`p45-01-parity-matrix`) extends the existing P1-14 Aer harness with multi-thread + QASM-input modes, adds a new MPS baseline harness whose exported QASM is consumed byte-identically by both sides, adds an `aleph-mps` criterion bench over those fixtures, runs the measurement session on the EPYC box, and writes `docs/perf/parity.md`.

**Tech Stack:** bash (gh CLI, awk), Python 3.12 (qiskit, qiskit-aer, qiskit-qasm3-import), Rust (criterion bench in `aleph-mps`), EPYC 8124P box (`ssh root@195.154.249.85`).

**Spec:** `docs/superpowers/specs/2026-06-11-phase-4.5-cpu-parity-design.md` (approved 2026-06-11).

**Key facts an executor must know (verified 2026-06-11):**

- `scripts/sync-issues.sh` splits BACKLOG.md on `^### \[P[0-9]+-[0-9]+\]` — this does **not** match `P4.5-01`; both the awk pattern and the two `grep -oE` ID/phase extractions must be extended (Task 1).
- Milestone titles use a literal em-dash `—` (U+2014); `gh` matches by exact string.
- Aer harness: `scripts/qiskit-baseline/run.py`. `time_aer()` pins `max_parallel_threads=1`. Tier-1 n=25 fixtures already exist in `scripts/qiskit-baseline/circuits/`: `ghz_n25.qasm`, `qft_n25.qasm`, `grover_n25_iters5.qasm`, `random_brickwall_n25_d20.qasm`.
- aleph MT side: `benches/benches/tier1_scaling.rs` already loads those same four fixtures and has a `tier1_scaling_fused` group (circuit `.optimize()`d outside the timed loop — symmetric with Aer timing `sim.run()` only, transpile excluded). Run: `RAYON_NUM_THREADS=16 cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench`.
- aleph-mps: `aleph-parser` is already a dev-dependency; `parallel` feature is opt-in and stays OFF (default-vs-default fairness — documented in the report).
- EPYC lessons that apply: verify the box idle first (`uptime`, `pgrep -af "cargo bench|bencher run|Runner.Worker"`); cargo lives at `~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/` (not on PATH for non-login shells); transfer the branch via `git bundle` + scp (don't race CI with pushes); `pkill` by comm-name, never `pkill -f`.

---

## PR 1 — Phase 4.5 bootstrap (branch `p45-00-phase-spec`)

### Task 1: Extend issue tooling to accept `P4.5-NN`

**Files:**
- Modify: `scripts/sync-issues.sh` (awk split pattern, ID grep, phase grep, milestone case)
- Modify: `scripts/create-milestones.sh` (new milestone line)

- [ ] **Step 1: Update the awk split pattern in `scripts/sync-issues.sh`**

Replace the awk block's two patterns:

```awk
  /^### \[P[0-9]+(\.[0-9]+)?-[0-9]+\]/ {
    match($0, /\[P[0-9]+(\.[0-9]+)?-[0-9]+\]/);
```

(was `/^### \[P[0-9]+-[0-9]+\]/` and `match($0, /\[P[0-9]+-[0-9]+\]/)`).

- [ ] **Step 2: Update the ID extraction grep**

```bash
  id=$(printf '%s' "$title" | grep -oE 'P[0-9]+(\.[0-9]+)?-[0-9]+' | head -1)
```

- [ ] **Step 3: Update the phase-number extraction and milestone map**

```bash
  phase_num=$(grep -m 1 '^\*\*Milestone:\*\*' "$f" \
    | grep -oE 'Phase [0-9]+(\.[0-9]+)?' | head -1 | awk '{print $2}' || true)
```

and add to `phase_milestone()` between the `4)` and `5)` arms:

```bash
    4.5) echo "Phase 4.5 — CPU Parity" ;;
```

- [ ] **Step 4: Add the milestone to `scripts/create-milestones.sh`**

Insert between the Phase 4 and Phase 5 `create_milestone` calls:

```bash
create_milestone "Phase 4.5 — CPU Parity" \
  "Every competitive-matrix cell (Aer MT statevector, Aer MPS, Stim) within 1.2× of the reference, or a documented structural exception. Gates v0.2 + PyPI."
```

- [ ] **Step 5: Verify the awk split offline (no issue creation)**

Run from repo root:

```bash
TMP=$(mktemp -d) && awk -v outdir="$TMP" '
  /^### \[P[0-9]+(\.[0-9]+)?-[0-9]+\]/ {
    match($0, /\[P[0-9]+(\.[0-9]+)?-[0-9]+\]/);
    id = substr($0, RSTART+1, RLENGTH-2);
    out = outdir "/issue-" id ".md";
    capture = 1; print > out; next;
  }
  capture { print >> out; }
' BACKLOG.md && ls "$TMP" | grep -c issue && ls "$TMP" | tail -5; rm -rf "$TMP"
```

Expected: file count equals the current count produced by the old pattern (run the old pattern the same way and diff the file lists — they must be identical until Task 3 adds P4.5 entries; after Task 3, exactly four `issue-P4.5-0N.md` files are added).

- [ ] **Step 6: Commit**

```bash
git add scripts/sync-issues.sh scripts/create-milestones.sh
git commit -m "[backlog] Teach issue tooling about dotted phase IDs (P4.5-NN)"
```

### Task 2: ROADMAP § 5 row + § 7 exit metric

**Files:**
- Modify: `ROADMAP.md` (~line 112: phases table; ~line 150–158: exit metrics list)

- [ ] **Step 1: Insert the phase row between Phase 4 and Phase 5 in the § 5 table**

```
|4.5  |CPU parity vs Aer/Stim                |2–4 weeks           |Every parity-matrix cell ≤ 1.2× its reference; docs/perf/parity.md|
```

- [ ] **Step 2: Insert the exit metric in § 7 between the Phase 4 and Phase 5 lines**

```
- Phase 4.5: every competitive-matrix cell ≤ 1.2× its reference (Aer MT statevector, Aer MPS, Stim), or a documented structural exception with profiling evidence; published in docs/perf/parity.md. v0.2 + PyPI (P4-09) gate on this.
```

- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md
git commit -m "[backlog] ROADMAP: insert Phase 4.5 — CPU parity (gates v0.2/PyPI before GPU)"
```

### Task 3: BACKLOG.md Phase 4.5 section

**Files:**
- Modify: `BACKLOG.md` (new `## Phase 4.5 — CPU Parity` section after the Phase 4 section, before Phase 5)

- [ ] **Step 1: Add the section header and adoption note**

```markdown
## Phase 4.5 — CPU Parity

Close (or honestly explain) wall-clock gaps vs mainstream simulators on CPU
before GPU work starts. Exit: every competitive-matrix cell ≤ 1.2× its
reference, or a documented structural exception with profiling evidence.
Design: `docs/superpowers/specs/2026-06-11-phase-4.5-cpu-parity-design.md`.

**Adopted tickets:** P3-12 (#148), P3-13 (#149), P3-14 (#150) are part of this
phase (the MPS levers). They keep their IDs and issue numbers; only their
milestone moves to "Phase 4.5 — CPU Parity". Order: P4.5-01 → (P4.5-02 ∥
P3-12..14) → P4.5-06 → P4.5-07.
```

- [ ] **Step 2: Add the four issue entries**

```markdown
### [P4.5-01] Competitive benchmark matrix vs Aer (MT statevector + MPS)

**Labels:** `area:bench`, `type:infra`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** —

**Description** — Measure where aleph actually stands against the references
multi-threaded, before any tuning. Two new rows: (1) aleph SV 16 threads vs
Aer statevector 16 OMP threads, default settings both sides, Tier-1 fixtures
@ n=25; (2) aleph-mps (sequential default) vs Aer `matrix_product_state` on
three MPS workloads consumed byte-identically from the same QASM fixtures.
The stabilizer row is imported from `docs/perf/surface_code.md` (1.64× @ d=11)
without re-measurement.

**Context** — Phase 1 proved aleph ahead of Aer single-thread; Phase 2 only
measured self-scaling. The MT and MPS cells have never been measured, and the
phase's tuning scope (P4.5-06) is defined by this matrix, not guessed.

**Technical Details** — Extend `scripts/qiskit-baseline/run.py` with
`--threads N`, `--from-qasm`, and `--out` (existing fixtures are the source of
truth, including the legacy `grover_n25_iters5.qasm`). aleph MT side =
existing `tier1_scaling_fused` criterion group, `RAYON_NUM_THREADS=16`. New
`scripts/mps-baseline/run.py` builds brickwork-n128-d6, long-range-n12, and
wide-bond-n26 circuits, exports QASM3 fixtures, times Aer MPS with matched
bond caps; new `crates/aleph-mps/benches/parity.rs` times aleph on the same
fixtures. χ chosen so brickwork (χ=64 ≫ max bond 8) and long-range
(χ=64 = exact at n=12) truncate on neither side — equal fidelity by
construction; wide-bond reports both sides' truncation metrics with a caveat.
All measurements on the idle-verified EPYC box.

**Acceptance Criteria**
- [ ] `docs/perf/parity.md` exists with the full matrix, per-cell ratio, and a ≤ 1.2× verdict per cell.
- [ ] Both sides of every cell consumed byte-identical circuits (QASM fixtures), same box, same session; versions and configs pinned in the report.
- [ ] Gap list section explicitly scopes P4.5-06 (or states "no MT gaps").
- [ ] Iteration-capped grover reported as such; Aer default fusion disclosed.

**Testing Requirements** — harness smoke runs at small n locally;
`cargo bench -p aleph-mps --bench parity -- --test` passes in CI; fixture
QASM files parse via aleph-parser (bench panics on parse failure).

**References** — spec § 3; `docs/perf/phase1.md`, `docs/perf/phase2.md`.

### [P4.5-02] Stabilizer: word-parallel transpose + zero_row/copy_row

**Labels:** `area:backend-stab`, `type:optimization`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** —

**Description** — Attack the two levers deferred from P3-11: the
orientation-transpose (~30% of the surface-d11 cycle) and `zero_row`/
`copy_row` (~33%), both still scalar in the dual-orientation tableau.

**Context** — The stabilizer cell is the one *known* parity gap: 1.64× Stim
@ d=11 (`docs/perf/surface_code.md`). These two hot spots are the identified
remainder after the P3-11 word-parallel gate work.

**Technical Details** — Word-parallel (u64 / AVX-512) implementations of the
transpose between X/Z bit-plane orientations and of row clear/copy in the
tableau, mirroring the P3-11 approach (ADR 0013). Bit-exact vs scalar;
Stim oracles d=3..11 unchanged.

**Acceptance Criteria**
- [ ] surface-d11 cycle time improves; target ≤ 1.2× Stim, else documented structural verdict with profile evidence per spec § 5.
- [ ] Bit-exact scalar↔SIMD equivalence tests; Stim oracle d=3..11 green.
- [ ] Before/after criterion numbers (EPYC) in the PR.

**Testing Requirements** — existing stim_oracle suites; new unit tests for
transpose/zero_row/copy_row word-parallel paths on irregular n (not multiples
of 64).

**References** — `docs/perf/surface_code.md` P3-11 addendum; ADR 0013.

### [P4.5-06] Close the MT gaps surfaced by the parity matrix

**Labels:** `area:backend-sv`, `type:optimization`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** M
**Depends on:** P4.5-01

**Description** — Deliberate placeholder: scope is the gap list from
`docs/perf/parity.md` (P4.5-01), not guessed in advance. Re-spec this entry
once the matrix lands; if the matrix shows no cell > 1.2×, close as no-op
with a comment linking the report.

**Context** — Spec § 4. The escalation ladder (profile → algorithm → layout →
SIMD → threads) and the one-PR-cycle-per-lever timebox from spec § 5 apply.

**Acceptance Criteria**
- [ ] Every SV-MT/MPS cell > 1.2× in parity.md either brought ≤ 1.2× or closed with a documented structural verdict.

**Testing Requirements** — standard (unit + property + oracle + before/after
criterion numbers per change).

**References** — spec § 4–5; `docs/perf/parity.md`.

### [P4.5-07] Final parity report, verdicts, and v0.2 gate

**Labels:** `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase 4.5
**Estimate:** S
**Depends on:** P4.5-02, P4.5-06

**Description** — Re-measure changed cells, finalize `docs/perf/parity.md`
with a verdict per cell (≤ 1.2× or structural exception + deferred ticket),
update ROADMAP § 7 (phase met/not-met) and CLAUDE.md project status, then tag
v0.2 and execute PyPI publication (P4-09, #142).

**Acceptance Criteria**
- [ ] parity.md final: every cell has a verdict; exceptions carry profiling evidence and a deferred ticket.
- [ ] ROADMAP § 7 + CLAUDE.md updated; v0.2 tagged; P4-09 unblocked/executed.

**Testing Requirements** — measurement protocol only (idle-verified EPYC);
no code changes expected.

**References** — spec § 2; P4-09 (#142).
```

- [ ] **Step 3: Re-run the Task-1 Step-5 awk verification**

Expected: exactly four new files `issue-P4.5-01.md`, `issue-P4.5-02.md`, `issue-P4.5-06.md`, `issue-P4.5-07.md`, each starting with its `### [P4.5-0N]` line and containing its `**Labels:**` and `**Milestone:** Phase 4.5` lines.

- [ ] **Step 4: Commit**

```bash
git add BACKLOG.md
git commit -m "[backlog] Add Phase 4.5 — CPU parity: P4.5-01/02/06/07, adopt P3-12..14"
```

### Task 4: PR 1, merge, then create milestone + issues

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin p45-00-phase-spec
gh pr create --title "[backlog] Phase 4.5 — CPU parity: spec, ROADMAP §5, P4.5-01/02/06/07" --body "$(cat <<'EOF'
Bootstraps Phase 4.5 (CPU parity before GPU), per the approved design spec
(docs/superpowers/specs/2026-06-11-phase-4.5-cpu-parity-design.md, included).

- ROADMAP § 5: new phase row + § 7 exit metric (≤ 1.2× per matrix cell; gates v0.2/PyPI).
- BACKLOG: Phase 4.5 section — P4.5-01 (competitive matrix), P4.5-02 (Stim levers),
  P4.5-06 (MT gap closure placeholder), P4.5-07 (final report + v0.2 gate);
  adopts P3-12..14 (#148–150) as the MPS levers (milestone move only).
- scripts/sync-issues.sh + create-milestones.sh: dotted phase IDs (P4.5-NN) + new milestone.

No code changes. Issues will be created via sync-issues.sh after merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: CI green, self-review the diff, merge** (squash, repo convention)

- [ ] **Step 3: After merge — create milestone and issues**

```bash
git checkout main && git pull
./scripts/create-milestones.sh        # creates "Phase 4.5 — CPU Parity", skips the rest
./scripts/sync-issues.sh              # creates the four P4.5 issues, skips all existing
```

Expected: `Summary: created=4  skipped=<all prior>  failed=0`.

- [ ] **Step 4: Move adopted issues to the new milestone**

```bash
for i in 148 149 150; do gh issue edit $i --milestone "Phase 4.5 — CPU Parity"; done
```

- [ ] **Step 5: Record the new issue numbers** (`gh issue list --milestone "Phase 4.5 — CPU Parity"`) — the P4.5-01 number is needed for PR 2's `Closes #N`.

---

## PR 2 — P4.5-01 execution (branch `p45-01-parity-matrix` off updated main)

### Task 5: `run.py` — threads, QASM input, output path

**Files:**
- Modify: `scripts/qiskit-baseline/run.py` (`time_aer()`, `main()` argparse)

- [ ] **Step 1: Thread-parameterize `time_aer`**

```python
def time_aer(tqc: QuantumCircuit, runs: int, threads: int = 1) -> dict:
    """Run `tqc` through AerSimulator(method='statevector') `runs` times.
    threads=1 reproduces the historical single-thread pin; threads>1 hands
    Aer that many OMP threads (its default gate fusion stays ON either way —
    we compare default-vs-default and disclose it in the report)."""
    sim = AerSimulator(
        method="statevector",
        max_parallel_threads=threads,
        max_parallel_experiments=1,
    )
```

(rest of the function body unchanged; thread the new parameter through every `time_aer(...)` call site, passing `args.threads`).

- [ ] **Step 2: Add `--threads`, `--from-qasm`, `--out`, `--min-runs` to argparse**

```python
    parser.add_argument("--threads", type=int, default=1,
        help="Aer max_parallel_threads (default 1 = historical pin).")
    parser.add_argument("--from-qasm", nargs="+", default=None, metavar="QASM",
        help="Skip circuit generation; time these QASM3 files verbatim "
             "(workload key = file stem). Requires qiskit-qasm3-import.")
    parser.add_argument("--out", type=str, default="results-qiskit.json",
        help="Output JSON path.")
    parser.add_argument("--min-runs", type=int, default=0,
        help="Lower bound on timed runs per workload (MT runs are cheaper; "
             "the cost-based budget assumes single-thread).")
```

- [ ] **Step 3: Implement the `--from-qasm` branch in `main()`**

Before the existing generation path, add:

```python
    if args.from_qasm:
        from qiskit import qasm3 as _qasm3
        results = {"schema_version": 2, "aer_threads": args.threads, "workloads": {}}
        for path_str in args.from_qasm:
            p = Path(path_str)
            qc = _qasm3.loads(p.read_text())
            gate_count = sum(qc.count_ops().values())
            runs = max(timing_runs_for(qc.num_qubits, gate_count), args.min_runs)
            print(f"timing {p.stem}: n={qc.num_qubits} gates={gate_count} "
                  f"runs={runs} threads={args.threads}", flush=True)
            results["workloads"][p.stem] = {
                "n": qc.num_qubits,
                "gate_count_post_transpile": gate_count,
                "qiskit_aer": time_aer(qc, runs, threads=args.threads),
            }
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
        return
```

Also route the existing path's final write through `args.out` (replace the hardcoded `results-qiskit.json`), and tag `"aer_threads": args.threads` into its results dict too.

- [ ] **Step 4: Local smoke test (small n, 1 thread — behavior must match the historical path)**

```bash
cd scripts/qiskit-baseline
python3 run.py --from-qasm circuits/ghz_n15.qasm --out /tmp/smoke.json --min-runs 3
python3 -c "import json;d=json.load(open('/tmp/smoke.json'));print(d['aer_threads'], list(d['workloads']))"
```

Expected: `1 ['ghz_n15']`. If `qiskit-qasm3-import` is missing locally, `pip install qiskit-qasm3-import` into the venv used for fixture generation (same applies on EPYC later).

- [ ] **Step 5: Commit**

```bash
git add scripts/qiskit-baseline/run.py
git commit -m "[P4.5-01] qiskit harness: --threads / --from-qasm / --out for the MT matrix"
```

### Task 6: MPS baseline harness + QASM fixtures

**Files:**
- Create: `scripts/mps-baseline/run.py`
- Create (generated, checked in): `scripts/mps-baseline/circuits/{brickwork_n128_d6,long_range_n12_dist4,long_range_n12_dist8,long_range_n12_dist11,wide_bond_n26_d12}.qasm`

- [ ] **Step 1: Write `scripts/mps-baseline/run.py`**

```python
#!/usr/bin/env python3
"""P4.5-01 MPS baseline: build the three MPS workload families, export QASM3
fixtures (the single source of truth both sides consume byte-identically),
and time Aer's matrix_product_state method on them.

Families (χ per family chosen in CHI below):
- brickwork_n128_d6 — mirrors crates/aleph-mps/tests/shallow_100q.rs: H wall,
  alternating even/odd CX·RZ(θ)·CX bonds (θ = 0.3 + 0.05·q), RX(φ) mixer wall
  (φ = 0.4 + 0.03·layer). Max bond 8 ⇒ χ=64 is exact on both sides.
- long_range_n12_dist{4,8,11} — H wall + one NN CX·RZ·CX ladder, then a single
  long-range CX(0, dist). χ=64 = 2^(12/2) is exact at n=12 ⇒ no truncation on
  either side (fidelity equal by construction).
- wide_bond_n26_d12 — seeded random-SU(4) brickwall, 12 layers, saturates the
  χ cap. Truncation semantics differ between implementations ⇒ the report
  carries both sides' truncation metrics and a fairness caveat.

Aer config: matrix_product_state_max_bond_dimension=χ,
matrix_product_state_truncation_threshold=1e-16 (bond cap binding, matching
aleph's FixedBond), max_parallel_threads=1 (aleph-mps default is sequential —
default-vs-default).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator

BASIS_GATES = ["h", "x", "z", "rz", "rx", "ry", "cx", "cz", "ccx", "p"]
CIRCUITS_DIR = Path(__file__).parent / "circuits"
CHI = {"brickwork_n128_d6": 64, "long_range_n12_dist4": 64,
       "long_range_n12_dist8": 64, "long_range_n12_dist11": 64,
       "wide_bond_n26_d12": 256}


def brickwork(n: int, layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(layers):
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(0.3 + 0.05 * q, q + 1)
            qc.cx(q, q + 1)
        phi = 0.4 + 0.03 * layer
        for q in range(n):
            qc.rx(phi, q)
    return qc


def long_range(n: int, dist: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for q in range(n - 1):
        qc.cx(q, q + 1)
        qc.rz(0.3 + 0.05 * q, q + 1)
        qc.cx(q, q + 1)
    qc.cx(0, dist)
    return qc


def wide_bond(n: int, layers: int, seed: int = 0x5121A6E0) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    k = 0
    for layer in range(layers):
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n - 1, 2):
            qc.append(UnitaryGate(random_unitary(4, seed=seed + k)), [q, q + 1])
            k += 1
    return qc


def export(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    (CIRCUITS_DIR / f"{name}.qasm").write_text(qasm3.dumps(tqc))
    return tqc


def time_aer_mps(tqc: QuantumCircuit, chi: int, runs: int) -> dict:
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=chi,
        matrix_product_state_truncation_threshold=1e-16,
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    t = tqc.copy()
    t.save_matrix_product_state()
    sim.run(t).result()  # warm-up
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sim.run(t).result()
        samples.append(time.perf_counter() - t0)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P4.5-01 Aer MPS baseline")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only export circuits/*.qasm; do not time Aer.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=str, default="results-aer-mps.json")
    args = parser.parse_args()

    builders = {
        "brickwork_n128_d6": lambda: brickwork(128, 6),
        "long_range_n12_dist4": lambda: long_range(12, 4),
        "long_range_n12_dist8": lambda: long_range(12, 8),
        "long_range_n12_dist11": lambda: long_range(12, 11),
        "wide_bond_n26_d12": lambda: wide_bond(26, 12),
    }
    results = {"schema_version": 1, "workloads": {}}
    for name, build in builders.items():
        tqc = export(build(), name)
        gate_count = sum(tqc.count_ops().values())
        print(f"exported {name}: n={tqc.num_qubits} gates={gate_count}", flush=True)
        if args.gen_only:
            continue
        results["workloads"][name] = {
            "n": tqc.num_qubits,
            "chi": CHI[name],
            "gate_count_post_transpile": gate_count,
            "aer_mps": time_aer_mps(tqc, CHI[name], args.runs),
        }
    if not args.gen_only:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate fixtures locally and sanity-check them through aleph-parser**

```bash
cd scripts/mps-baseline && python3 run.py --gen-only && ls circuits/
cd ../.. && for f in scripts/mps-baseline/circuits/*.qasm; do
  cargo run --bin aleph -- run "$f" --backend mps --max-bond 64 --shots 1 >/dev/null \
    && echo "OK $f" || echo "PARSE/RUN FAIL $f"
done
```

Expected: 5 QASM files, all `OK`. (`wide_bond` may take ~a minute locally at χ=64 — for the smoke test that's fine; the measured run uses χ=256 on EPYC. If the CLI flag spelling differs, check `cargo run --bin aleph -- run --help` and adjust — the bench in Task 7, not the CLI, is the timed path.)

- [ ] **Step 3: Commit (script + generated fixtures checked in, like qiskit-baseline)**

```bash
git add scripts/mps-baseline/
git commit -m "[P4.5-01] MPS baseline harness + QASM fixtures (Aer matrix_product_state side)"
```

### Task 7: aleph-mps `parity` criterion bench

**Files:**
- Create: `crates/aleph-mps/benches/parity.rs`
- Modify: `crates/aleph-mps/Cargo.toml` (new `[[bench]]` block; `criterion` + `aleph-parser` are already dev-deps)

- [ ] **Step 1: Write the bench**

```rust
//! P4.5-01 parity bench: aleph-mps on the byte-identical QASM fixtures that
//! `scripts/mps-baseline/run.py` times through Aer matrix_product_state.
//! Sequential default build (no `parallel` feature) — default-vs-default.
//! χ per family matches the harness's CHI table; brickwork and long_range are
//! exact at their χ (no truncation on either side, fidelity equal by
//! construction); wide_bond saturates the cap (truncation caveat in
//! docs/perf/parity.md).

use aleph_backend::run;
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// (fixture stem, max bond) — keep in sync with scripts/mps-baseline/run.py CHI.
const WORKLOADS: &[(&str, usize)] = &[
    ("brickwork_n128_d6", 64),
    ("long_range_n12_dist4", 64),
    ("long_range_n12_dist8", 64),
    ("long_range_n12_dist11", 64),
    ("wide_bond_n26_d12", 256),
];

fn load(stem: &str) -> aleph_ir::Circuit {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two dirs deep from repo root")
        .join("scripts/mps-baseline/circuits")
        .join(format!("{stem}.qasm"));
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    aleph_parser::parse(&src)
        .unwrap_or_else(|e| panic!("parse {} failed: {e:?}", path.display()))
}

fn bench_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_parity");
    group.sample_size(10);
    for &(stem, chi) in WORKLOADS {
        let circuit = load(stem);
        group.bench_function(stem, |b| {
            b.iter_with_setup(
                || MpsBackend::with_seed(0).with_max_bond(chi),
                |mut backend| {
                    let state = run(&mut backend, &circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parity);
criterion_main!(benches);
```

(If `with_max_bond` takes a different integer type or `aleph_backend`/`aleph_ir` aren't yet dev-deps of aleph-mps, mirror exactly what `benches/long_range.rs` imports and uses — it is the working template.)

- [ ] **Step 2: Register the bench in `crates/aleph-mps/Cargo.toml`** (after the `wide_bond` block)

```toml
[[bench]]
name = "parity"
harness = false
```

No `required-features` — this bench must run in plain `cargo bench --workspace`.

- [ ] **Step 3: Smoke-run (criterion test mode, one iteration per workload)**

```bash
cargo bench -p aleph-mps --bench parity -- --test
```

Expected: all five workloads execute, exit 0. Then `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-mps/benches/parity.rs crates/aleph-mps/Cargo.toml
git commit -m "[P4.5-01] aleph-mps parity bench over the shared QASM fixtures"
```

### Task 8: EPYC measurement session

**Files:** none in-repo until Task 9 (raw JSONs land in the report's data section).

- [ ] **Step 1: Verify the box is idle (mandatory, twice burned)**

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo CLEAN; cat /proc/mdstat | head -3'
```

Expected: load ≈ 0, `CLEAN`, no md resync in progress. Do not push to `main` or `benches/**` while measuring.

- [ ] **Step 2: Ship the branch via git bundle (not a push — CI race lesson)**

```bash
git bundle create /tmp/p45-01.bundle main p45-01-parity-matrix
scp /tmp/p45-01.bundle root@195.154.249.85:/root/
ssh root@195.154.249.85 'cd /root/aleph && git fetch /root/p45-01.bundle p45-01-parity-matrix:p45-01-parity-matrix && git checkout p45-01-parity-matrix && git log --oneline -1'
```

Verify the printed HEAD sha matches local (`git rev-parse --short HEAD`) — the Ryzen stale-bundle lesson.

- [ ] **Step 3: Aer SV-MT row (16 threads, fixtures verbatim)**

```bash
ssh root@195.154.249.85
cd /root/aleph/scripts/qiskit-baseline
source <P1-14 venv>/bin/activate   # locate the uv py3.12 venv from Stage 0; if gone:
                                   # uv venv --python 3.12 /root/parity-venv &&
                                   # /root/parity-venv/bin/pip install qiskit qiskit-aer qiskit-qasm3-import
pip show qiskit-qasm3-import >/dev/null || pip install qiskit-qasm3-import
python3 run.py --threads 16 --min-runs 5 --out results-qiskit-t16.json \
  --from-qasm circuits/ghz_n25.qasm circuits/qft_n25.qasm \
              circuits/grover_n25_iters5.qasm circuits/random_brickwall_n25_d20.qasm
```

Record `qiskit`, `qiskit-aer` versions (`pip show`) for the report.

- [ ] **Step 4: aleph SV-MT row (same fixtures, 16 rayon threads, fused group)**

```bash
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
cd /root/aleph
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=16 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench
```

Medians: `target/criterion/tier1_scaling_fused/*/new/estimates.json` (`median.point_estimate`, ns). The `tier1_scaling_fused` group is the matrix cell (optimize() outside the loop ⇔ Aer times `sim.run` only, transpile outside).

- [ ] **Step 5: Aer MPS row**

```bash
cd /root/aleph/scripts/mps-baseline
python3 run.py --runs 10 --out results-aer-mps.json
```

Capture per-workload `median_s`. Also capture Aer's MPS metadata from one result if available (`result.results[0].metadata`) for the wide_bond truncation note — if the harness doesn't surface it, note "Aer truncation metadata not exposed" in the report rather than guessing.

- [ ] **Step 6: aleph MPS row**

```bash
cd /root/aleph
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --bench parity
```

Medians from `target/criterion/mps_parity/*/new/estimates.json`.

- [ ] **Step 7: Re-verify idleness after the session** (`uptime`, pgrep) — if a CI job started mid-session, discard and re-run the affected rows. Copy all four result sets (`results-qiskit-t16.json`, `results-aer-mps.json`, both criterion estimate trees) back via `scp`.

### Task 9: `docs/perf/parity.md`

**Files:**
- Create: `docs/perf/parity.md`

- [ ] **Step 1: Write the report with measured numbers**

Structure (fill every `<>` from Task 8 data — no placeholders may survive into the commit):

```markdown
# Phase 4.5 competitive parity matrix (P4.5-01 baseline)

**Box:** EPYC 8124P (16c/32t, AVX-512), idle-verified before each row.
**Toolchain:** rustc <ver>, RUSTFLAGS=-C target-cpu=native; qiskit <ver>,
qiskit-aer <ver>, Python 3.12 (uv venv); Stim row imported from
docs/perf/surface_code.md (Stim 1.16.0), not re-measured.
**Protocol:** both sides of every cell consume byte-identical QASM fixtures,
same box, same session. Aer default gate fusion ON (default-vs-default,
disclosed). aleph SV: tier1_scaling_fused (optimize() outside the timed loop);
Aer: sim.run() timed, transpile outside. Bar: ≤ 1.2× (spec § 2).

## Row 1 — state vector, 16 threads both sides
| workload | n | aleph (ms) | Aer (ms) | aleph/Aer | verdict |
| ghz / qft / grover_iters5 / random_brickwall_d20 @ 25 | ... |
grover is iteration-capped (5 iters; full grover-25 ≈ 13 CPU-h single-thread)
— reported as such, no extrapolation.

## Row 2 — MPS, sequential both sides
| workload | n | χ | aleph (ms) | Aer (ms) | aleph/Aer | verdict |
brickwork + long_range: χ exact on both sides (no truncation — fidelity equal
by construction). wide_bond: χ cap saturated; truncation semantics differ
(<both sides' truncation metrics or "not exposed">) — caveat applies.

## Row 3 — stabilizer (imported)
surface-d11: aleph 1.64× Stim (docs/perf/surface_code.md, P3-11). Verdict:
GAP > 1.2× → P4.5-02.

## Gap list (scopes P4.5-06)
<every cell > 1.2×, one line each: cell, ratio, first-suspect lever; or
"no SV-MT/MPS gaps — P4.5-06 closes as no-op">

## Raw data
<medians + RSD tables, run counts, exact commands>
```

- [ ] **Step 2: Commit**

```bash
git add docs/perf/parity.md
git commit -m "[P4.5-01] parity matrix baseline: SV-MT + MPS measured, stab imported"
```

### Task 10: PR 2

- [ ] **Step 1: Final checks** — `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, re-read the diff.

- [ ] **Step 2: Push, open PR**

```bash
git push -u origin p45-01-parity-matrix
gh pr create --title "[P4.5-01] Competitive benchmark matrix vs Aer (MT statevector + MPS)" --body "$(cat <<EOF
Closes #<P4.5-01 issue number from Task 4 Step 5>.

Measure-first baseline for Phase 4.5 (spec § 3): aleph vs Aer, multi-threaded
SV and MPS, byte-identical QASM fixtures both sides, idle-verified EPYC.
Stabilizer row imported from docs/perf/surface_code.md.

- scripts/qiskit-baseline/run.py: --threads / --from-qasm / --out (default
  behavior unchanged: threads=1).
- scripts/mps-baseline/: new Aer matrix_product_state harness + 5 QASM
  fixtures (brickwork n128, long_range n12 ×3, wide_bond n26).
- crates/aleph-mps/benches/parity.rs: aleph side of the MPS row.
- docs/perf/parity.md: full matrix, per-cell ≤1.2× verdicts, gap list
  scoping P4.5-06.

Results: <one-line headline per row after measurement>
Test results: <cargo test summary; parity bench --test smoke>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: CI green → self-review → squash-merge.** Then comment the gap list on the P4.5-06 issue so its re-spec has a paper trail.

---

## Out of scope for this plan

P4.5-02 (Stim levers), the adopted P3-12..14 work, P4.5-06 (needs the matrix first), and P4.5-07 each get their own plan when picked up.
