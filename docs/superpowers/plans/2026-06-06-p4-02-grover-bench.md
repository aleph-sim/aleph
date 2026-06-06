# P4-02 Grover Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Grover (n = 4, 8, 12, 16, optimal √N iterations) as the second consumer of the Phase-4 benchmark/report framework, with a marked-state-convergence acceptance test and a Grover section in `docs/perf/phase4.md`.

**Architecture:** Reuse everything from P4-01. `run.py` generates the shared QASM corpus (single source of truth) and times Aer; a mirror `phase4_grover.rs` criterion bench times aleph over the *same* corpus files; the golden-tested `extract_criterion.py` + `report.py` join the two by workload key and render the report. The only new production logic is the optimal-iteration formula and a cost-based Aer-run budget. A new Rust integration test asserts the acceptance criterion (marked-state probability > 0.9).

**Tech Stack:** Python 3.12 + Qiskit 1.2.4 / Aer 0.15.1 (venv at `scripts/qiskit-baseline/.venv`), Rust 2021 + criterion, `NaiveSvBackend` via `run` / `run_optimized`.

---

## Key design decision: corpus filename vs. join key (READ FIRST)

The Phase-4 report joins the aleph and Aer JSONs **by exact workload key**. `extract_criterion.py` derives the aleph key from the criterion group's per-parameter directory as `f"{family}_n{n}"` — for Grover that is **`grover_n4`** (the criterion `BenchmarkId` parameter is just `n`). Therefore the **Aer side must also key Grover rows as `grover_n{n}`**, NOT `grover_n4_iters3`.

But the on-disk corpus file must carry the iteration count so it is self-describing and so the bench/test can find it (`grover_n16_iters201.qasm`).

**Resolution:** decouple the two concepts in `run.py`:
- `corpus_stem(family, n)` → the QASM filename stem + `QuantumCircuit` name. Grover: `grover_n{n}_iters{opt}`. (Carries iters.)
- `workload_key(family, n)` → the results-JSON join key. Grover: `grover_n{n}`. (No iters — matches `extract_criterion.py`.)

This keeps `extract_criterion.py` and `report.py` **completely untouched** (their golden tests stay green), exactly as the spec intends ("adding a family needs no new tooling test"). The spec's phrase "workload_name for grover → `grover_n{n}_iters{opt}`" refers to the *corpus stem*; this plan implements it as `corpus_stem` and adds the separate `workload_key` to make the join work.

---

## File structure

- **Modify** `scripts/qiskit-baseline/run.py` — add `grover_optimal_iters`, switch the grover family to optimal iters at n ∈ {4,8,12,16}, split `workload_name` into `corpus_stem` + `workload_key`, make `timing_runs_for` cost-based.
- **Create** `scripts/qiskit-baseline/test_run.py` — unit tests for the new pure functions (skips gracefully when Qiskit is absent, e.g. CI).
- **Create** `scripts/qiskit-baseline/circuits/grover_n{4,8,12,16}_iters{opt}.qasm` — generated corpus (committed).
- **Create** `benches/tests/grover_convergence.rs` — the AC test (marked-state p > 0.9; n=16 `#[ignore]`).
- **Create** `benches/benches/phase4_grover.rs` — aleph criterion bench over the corpus (mirror of `phase4_qft.rs`).
- **Modify** `benches/Cargo.toml` — register the `phase4_grover` bench.
- **Modify** `scripts/bench-report/README.md` and `scripts/qiskit-baseline/README.md` — extend the EPYC measurement flow with the grover workloads + the merge-into-existing-JSON step.
- **Modify (EPYC step)** `docs/perf/data/phase4-aleph.json`, `docs/perf/data/phase4-aer.json`, `docs/perf/data/phase4-meta.json`, `docs/perf/phase4.md` — append grover rows, re-render.

---

## Task 1: `run.py` — optimal iters, family sizes, key/stem split, cost-based timing

**Files:**
- Create: `scripts/qiskit-baseline/test_run.py`
- Modify: `scripts/qiskit-baseline/run.py`

The new functions are pure; test them first. The test imports `run`, which imports Qiskit at module load, so it `skipUnless` Qiskit is importable (keeps CI without Qiskit green; run it under the venv locally).

- [ ] **Step 1: Write the failing test**

Create `scripts/qiskit-baseline/test_run.py`:

```python
"""Unit tests for the pure helpers in run.py (optimal iters, run budget, keys).

run.py imports Qiskit at module load, so these tests skip when Qiskit is not
installed (e.g. CI). Run locally under the harness venv:

    scripts/qiskit-baseline/.venv/bin/python -m unittest \
        discover -s scripts/qiskit-baseline -p 'test_run.py'
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    import run  # noqa: E402  (Qiskit-importing module)
    HAVE_RUN = True
except Exception:  # pragma: no cover - environment without Qiskit
    HAVE_RUN = False


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestGroverOptimalIters(unittest.TestCase):
    def test_table(self):
        self.assertEqual(run.grover_optimal_iters(4), 3)
        self.assertEqual(run.grover_optimal_iters(8), 13)
        self.assertEqual(run.grover_optimal_iters(12), 50)
        self.assertEqual(run.grover_optimal_iters(16), 201)


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestTimingRunsForCost(unittest.TestCase):
    def test_qft_budget_preserved(self):
        # QFT-25: 1525 * 2^25 = 5.1e10 -> 3 (unchanged vs the n-only function).
        self.assertEqual(run.timing_runs_for(25, 1525), 3)
        # QFT-30: 2205 * 2^30 = 2.4e12 -> 1 (spec: "1-2 runs, as before").
        self.assertEqual(run.timing_runs_for(30, 2205), 1)

    def test_grover_costs(self):
        self.assertEqual(run.timing_runs_for(4, 268), 10)        # 4.3e3
        self.assertEqual(run.timing_runs_for(8, 17974), 10)      # 4.6e6
        self.assertEqual(run.timing_runs_for(12, 264312), 5)     # 1.08e9
        self.assertEqual(run.timing_runs_for(16, 2258854), 2)    # 1.48e11


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestKeysAndStems(unittest.TestCase):
    def test_grover_stem_carries_iters(self):
        self.assertEqual(run.corpus_stem("grover", 16), "grover_n16_iters201")
        self.assertEqual(run.corpus_stem("grover", 4), "grover_n4_iters3")

    def test_grover_key_has_no_iters(self):
        self.assertEqual(run.workload_key("grover", 16), "grover_n16")

    def test_qft_key_and_stem_align(self):
        self.assertEqual(run.workload_key("qft", 20), "qft_n20")
        self.assertEqual(run.corpus_stem("qft", 20), "qft_n20")

    def test_family_sizes(self):
        self.assertEqual(run.FAMILY_SIZES["grover"], [4, 8, 12, 16])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `scripts/qiskit-baseline/.venv/bin/python -m unittest discover -s scripts/qiskit-baseline -p 'test_run.py' -v`
Expected: FAIL — `AttributeError: module 'run' has no attribute 'grover_optimal_iters'` (and `corpus_stem`/`workload_key`), plus `FAMILY_SIZES["grover"]` is still `[15, 20, 22, 25]`.

- [ ] **Step 3: Add `grover_optimal_iters` and switch the grover family sizes**

In `scripts/qiskit-baseline/run.py`, change `FAMILY_SIZES["grover"]` from `[15, 20, 22, 25]` to `[4, 8, 12, 16]`:

```python
FAMILY_SIZES = {
    "ghz": [15, 20, 22, 25],
    "qft": [10, 15, 20, 25, 30],
    # P4-02: optimal-iteration Grover at small n (tiny cache-resident state,
    # tractable even at 2.26M gates for n=16). The legacy grover_n{15..25}_iters5
    # fixtures stay on disk (frozen Phase-1/2 bench artifacts) but are no longer
    # regenerated here.
    "grover": [4, 8, 12, 16],
    "random_brickwall": [15, 20, 22, 25],
}
```

Immediately after `GROVER_ITERS = 5`, add a comment + the optimal-iters helper. Replace:

```python
GROVER_ITERS = 5
```

with:

```python
# Legacy iteration count for the frozen grover_n{15..25}_iters5 fixtures only.
# The active P4-02 matrix uses grover_optimal_iters(n) instead (see below).
GROVER_ITERS = 5


def grover_optimal_iters(n: int) -> int:
    """Optimal Grover iteration count for a single marked state.

    The success probability peaks at ~round(pi/4 * sqrt(2^n)) Grover operators
    (Nielsen & Chuang sec. 6.1). n in {4,8,12,16} -> {3, 13, 50, 201}.
    """
    return max(1, round(math.pi / 4 * math.sqrt(2**n)))
```

- [ ] **Step 4: Point the grover builder at the optimal iters**

In `FAMILY_BUILDERS`, change the grover entry from `lambda n: build_grover(n, GROVER_ITERS)` to:

```python
FAMILY_BUILDERS = {
    "ghz": lambda n: build_ghz(n),
    "qft": lambda n: build_qft(n),
    "grover": lambda n: build_grover(n, grover_optimal_iters(n)),
    "random_brickwall": lambda n: build_random_brickwall(n, RANDOM_DEPTH),
}
```

- [ ] **Step 5: Split `workload_name` into `corpus_stem` + `workload_key`**

Replace the whole `workload_name` function:

```python
def workload_name(family: str, n: int) -> str:
    if family == "grover":
        return f"grover_n{n}_iters{GROVER_ITERS}"
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"
```

with:

```python
def corpus_stem(family: str, n: int) -> str:
    """QASM filename stem and QuantumCircuit name. Grover/Random embed their
    iteration/depth count so the on-disk corpus is self-describing
    (e.g. grover_n16_iters201.qasm)."""
    if family == "grover":
        return f"grover_n{n}_iters{grover_optimal_iters(n)}"
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"


def workload_key(family: str, n: int) -> str:
    """Join key into the unified results JSON. MUST equal extract_criterion.py's
    `{family}_n{n}` so report.py lines up the aleph (criterion) and Aer rows.
    The grover *file* carries the iter count; the *key* does not — the criterion
    BenchmarkId parameter is just n. (Random keeps its legacy depth-suffixed key;
    it is not a Phase-4 criterion consumer.)"""
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"
```

- [ ] **Step 6: Make `all_workloads` carry both key and stem**

Replace:

```python
def all_workloads() -> list[tuple[str, str, int]]:
    """(name, family, n) for the full matrix, families in stable order."""
    return [
        (workload_name(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in FAMILY_SIZES[fam]
    ]
```

with:

```python
def all_workloads() -> list[tuple[str, str, str, int]]:
    """(key, stem, family, n) for the full matrix, families in stable order."""
    return [
        (workload_key(fam, n), corpus_stem(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in FAMILY_SIZES[fam]
    ]
```

- [ ] **Step 7: Make `timing_runs_for` cost-based**

Replace:

```python
def timing_runs_for(n: int) -> int:
    """Fewer timed Aer runs at large n (each is minutes). Disclosed in the report."""
    if n <= 20:
        return 10
    if n <= 22:
        return 5
    if n <= 25:
        return 3
    return 2  # n >= 28: a single Aer statevector run is many minutes
```

with:

```python
def timing_runs_for(n: int, gate_count: int) -> int:
    """Fewer timed Aer runs for costlier circuits. A single Aer statevector pass
    costs ~ gate_count * 2^n (each gate sweeps the 2^n-amplitude state), so the
    budget keys on that product rather than n alone — Grover-16 is only 16 qubits
    but 2.26M gates. Verified: QFT-25 (5.1e10)->3, QFT-30 (2.4e12)->1,
    Grover-16 (1.5e11)->2. Disclosed in the report."""
    cost = gate_count * (2**n)
    if cost > 1e12:
        return 1
    if cost > 1e11:
        return 2
    if cost > 1e10:
        return 3
    if cost > 1e9:
        return 5
    return 10
```

- [ ] **Step 8: Update `main()` to use key/stem and pass `gate_count`**

In `main()`, change the `selected` line from:

```python
    selected = (
        set(args.workloads.split(",")) if args.workloads else {name for name, _, _ in matrix}
    )
```

to:

```python
    selected = (
        set(args.workloads.split(",")) if args.workloads else {key for key, _, _, _ in matrix}
    )
```

Then replace the whole build/time loop body. Change:

```python
    for name, family, n in matrix:
        print(f"[build] {name} ...", flush=True)
        qc = FAMILY_BUILDERS[family](n)
        tqc = transpile_and_export(qc, name)
        gate_count = len(tqc.data)
        print(f"[build] {name}: {gate_count} gates after transpile", flush=True)
        if args.gen_only or name not in selected:
            continue
        runs = timing_runs_for(n)
        print(f"[time]  {name} (Aer, {runs} runs) ...", flush=True)
        timing = time_aer(tqc, runs)
        print(
            f"[time]  {name}: median={timing['median_s']*1000:.2f} ms "
            f"stdev={timing['stdev_s']*1000:.2f} ms",
            flush=True,
        )
        results["workloads"][name] = {
            "n": n,
            "family": family,
            "timing_runs": runs,
            "gate_count_post_transpile": gate_count,
            "qiskit_aer": timing,
        }
```

to:

```python
    for key, stem, family, n in matrix:
        print(f"[build] {stem} ...", flush=True)
        qc = FAMILY_BUILDERS[family](n)
        tqc = transpile_and_export(qc, stem)
        gate_count = len(tqc.data)
        print(f"[build] {stem}: {gate_count} gates after transpile", flush=True)
        if args.gen_only or key not in selected:
            continue
        runs = timing_runs_for(n, gate_count)
        print(f"[time]  {key} (Aer, {runs} runs) ...", flush=True)
        timing = time_aer(tqc, runs)
        print(
            f"[time]  {key}: median={timing['median_s']*1000:.2f} ms "
            f"stdev={timing['stdev_s']*1000:.2f} ms",
            flush=True,
        )
        results["workloads"][key] = {
            "n": n,
            "family": family,
            "timing_runs": runs,
            "gate_count_post_transpile": gate_count,
            "qiskit_aer": timing,
        }
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `scripts/qiskit-baseline/.venv/bin/python -m unittest discover -s scripts/qiskit-baseline -p 'test_run.py' -v`
Expected: PASS (all classes run, none skipped, since the venv has Qiskit).

- [ ] **Step 10: Smoke-check `--gen-only` builds the grover circuits (no commit yet)**

Run: `cd scripts/qiskit-baseline && .venv/bin/python run.py --gen-only 2>&1 | grep grover`
Expected: lines like `[build] grover_n4_iters3: 268 gates after transpile`, `grover_n8_iters13: 17974`, `grover_n12_iters50: 264312`, `grover_n16_iters201: 2258854` (counts may differ by a few from these reference values across Qiskit patch versions — that is fine; record whatever this run produces). This also confirms `corpus_stem` drives the filename.

- [ ] **Step 11: Commit**

```bash
git add scripts/qiskit-baseline/run.py scripts/qiskit-baseline/test_run.py
git commit -m "[P4-02] run.py: optimal-iter Grover family, key/stem split, cost-based Aer budget

grover_optimal_iters(n)=round(pi/4*sqrt(2^n)); grover sizes -> {4,8,12,16}.
Split workload_name into corpus_stem (iters-suffixed filename) and workload_key
(grover_n{n}, matches extract_criterion's join key). timing_runs_for now budgets
on gate_count*2^n. Python unit tests (skip without Qiskit).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Generate + commit the Grover corpus

> **DECISION (repo owner, 2026-06-06):** commit **n=4/8/12 only** (~4.2 MB total).
> The `grover_n16_iters201.qasm` corpus is **34 MB** (13× the largest existing
> committed circuit; would more than double `.git`) and is consumed only by the
> nightly `#[ignore]`d test and the EPYC measurement — never default CI. It is
> deterministic from `run.py`, so it is **generated on demand** on those paths
> and **`.gitignore`d**, not committed.

**Files:**
- Create: `scripts/qiskit-baseline/circuits/grover_n4_iters3.qasm`
- Create: `scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm`
- Create: `scripts/qiskit-baseline/circuits/grover_n12_iters50.qasm`
- Create: `scripts/qiskit-baseline/circuits/.gitignore` (ignore `grover_n16_iters*.qasm`)
- Generated-not-committed: `scripts/qiskit-baseline/circuits/grover_n16_iters201.qasm`

(Filenames assume the reference iter counts {3,13,50,201}; use whatever `grover_optimal_iters` printed in Task 1 Step 10 if it differs.)

- [ ] **Step 1: Generate the full corpus**

Run: `cd scripts/qiskit-baseline && .venv/bin/python run.py --gen-only`
Expected: exits cleanly; `circuits/grover_n{4,8,12,16}_iters{opt}.qasm` now exist. n=16 is ~2.26M gates so the file is large (tens of MiB) — this is expected and intended (tiny state, large gate list).

- [ ] **Step 2: Verify only the new grover files changed**

Run: `cd /Users/ex/GitHub/aleph && git status --short scripts/qiskit-baseline/circuits/`
Expected: four new `?? .../grover_n{4,8,12,16}_iters*.qasm` entries and **nothing else** (qft/ghz/random regenerate byte-identically under the same Qiskit version, so they show no diff). If any other circuit shows as modified, investigate before continuing — it means a generator changed unexpectedly; `git checkout --` it if the change is spurious.

- [ ] **Step 3: Add the `.gitignore` for the 34 MB n=16 blob**

Create `scripts/qiskit-baseline/circuits/.gitignore`:

```gitignore
# P4-02: the optimal-iteration Grover n=16 corpus is ~34 MB (2.26M gates) and is
# consumed only by the nightly #[ignore]d convergence test and the EPYC
# scaling-bench / oneshot measurement — never default CI. It is deterministic
# from run.py (`run.py --gen-only`), so it is generated on demand on those paths
# rather than committed. n=4/8/12 are small and committed normally.
grover_n16_iters*.qasm
```

- [ ] **Step 4: Sanity-check the n=4 corpus parses and is non-trivial**

Run: `head -c 400 scripts/qiskit-baseline/circuits/grover_n4_iters3.qasm`
Expected: OpenQASM 3.0 header (`OPENQASM 3.0;`) followed by gate statements over a 4-qubit register.

- [ ] **Step 5: Commit (n=4/8/12 + the gitignore; NOT n=16)**

```bash
git add scripts/qiskit-baseline/circuits/.gitignore \
        scripts/qiskit-baseline/circuits/grover_n4_iters3.qasm \
        scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm \
        scripts/qiskit-baseline/circuits/grover_n12_iters50.qasm
# Confirm n=16 is NOT staged (it must be gitignored):
git status --short scripts/qiskit-baseline/circuits/ | grep grover_n16 && echo "ERROR: n16 not ignored" || echo "ok: n16 ignored"
git commit -m "[P4-02] Generate optimal-iteration Grover corpus (n=4,8,12; n=16 gitignored)

Single source of truth shared by Aer (run.py) and aleph (phase4_grover bench +
convergence test). Marks |0..01>; gate counts 268/17974/264312. The 34 MB n=16
corpus (2.26M gates) is generated on demand (run.py --gen-only) for the nightly
ignored test + EPYC measurement, not committed (gitignored).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Grover convergence acceptance test (the AC)

**Files:**
- Create: `benches/tests/grover_convergence.rs`

Cargo auto-discovers `tests/*.rs` as integration tests; `aleph-parser`, `aleph-sv`, `aleph-backend`, `aleph-core` are regular `[dependencies]` of `aleph-benches`, so they are usable here with no Cargo.toml change. Uses the verbatim `run` path (no optimization passes) — the most direct correctness oracle for the AC.

- [ ] **Step 1: Write the test**

Create `benches/tests/grover_convergence.rs`:

```rust
//! P4-02 acceptance test: optimal-iteration Grover converges. For each tested n
//! the marked basis state |0...01> (amplitude index 1 in aleph's MSB-qubit
//! ordering, ADR 0004) must reach probability > 0.9 AND be the most-probable
//! outcome (we amplified the *right* state, not merely *some* state). Reads the
//! committed corpus QASM — the single source of truth shared with Aer — and runs
//! the verbatim `run` path on NaiveSvBackend.

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use std::path::PathBuf;

/// round(pi/4 * sqrt(2^n)); mirrors run.py::grover_optimal_iters so the corpus
/// filename matches. n in {4,8,12,16} -> {3,13,50,201}.
fn optimal_iters(n: u32) -> u32 {
    (std::f64::consts::PI / 4.0 * (2f64.powi(n as i32)).sqrt()).round() as u32
}

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/grover_n{n}_iters{}.qasm",
        optimal_iters(n)
    ))
}

fn assert_converges(n: u32) {
    let path = corpus_path(n);
    // n=16 (34 MB, 2.26M gates) is generated on demand, not committed (see the
    // circuits/.gitignore). If it is absent on this host, skip with a clear hint
    // rather than failing — this only affects the #[ignore]d nightly path.
    if !path.exists() {
        eprintln!(
            "SKIP grover_n{n}: corpus {} not present; generate it with \
             `python scripts/qiskit-baseline/run.py --gen-only`",
            path.display()
        );
        return;
    }
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse grover_n{n}: {e:?}"));
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &circuit).expect("simulate grover");
    let amps = state.amplitudes();

    let p_marked = amps[1].norm_sqr();
    assert!(
        p_marked > 0.9,
        "grover_n{n}: marked-state probability {p_marked:.4} is not > 0.9"
    );

    let (argmax, _) = amps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().total_cmp(&b.1.norm_sqr()))
        .expect("non-empty state");
    assert_eq!(argmax, 1, "grover_n{n}: most-probable index {argmax}, expected 1");
}

#[test]
fn grover_n4_converges() {
    assert_converges(4);
}

#[test]
fn grover_n8_converges() {
    assert_converges(8);
}

#[test]
fn grover_n12_converges() {
    assert_converges(12);
}

/// n=16 is a 2.26M-gate circuit (~seconds single-thread); per CLAUDE.md it is
/// #[ignore]d for normal CI and exercised on the nightly ignored-tests run.
#[test]
#[ignore = "n=16: 2.26M gates, run on the nightly ignored-tests schedule"]
fn grover_n16_converges() {
    assert_converges(16);
}
```

- [ ] **Step 2: Run the non-ignored tests**

Run: `cargo test -p aleph-benches --test grover_convergence`
Expected: `grover_n4_converges`, `grover_n8_converges`, `grover_n12_converges` PASS; `grover_n16_converges` shows as ignored. (n=12 runs ~264k gates; allow a few seconds.)

- [ ] **Step 3: Run the ignored n=16 test once to confirm it passes**

Run: `cargo test -p aleph-benches --test grover_convergence -- --ignored grover_n16_converges`
Expected: PASS (takes a few seconds; ~2.26M gates over a 64k-amplitude state).

- [ ] **Step 4: Commit**

```bash
git add benches/tests/grover_convergence.rs
git commit -m "[P4-02] Grover convergence test: marked-state p>0.9 (n=4,8,12; n=16 ignored)

Asserts |0..01> (index 1) is amplified above 0.9 AND is the argmax, over the
committed corpus via the verbatim run path. n=16 #[ignore]d per CLAUDE.md.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `phase4_grover` aleph criterion bench

**Files:**
- Create: `benches/benches/phase4_grover.rs`
- Modify: `benches/Cargo.toml`

Mirror `phase4_qft.rs` exactly. n=4/8/12 always benched; n=16 behind `scaling-bench` (the report number for n=16 comes from `oneshot`, like QFT n=30).

- [ ] **Step 1: Register the bench in `benches/Cargo.toml`**

At the end of the file, after the `phase4_qft` block, add:

```toml
[[bench]]
name = "phase4_grover"
harness = false
```

- [ ] **Step 2: Write the bench**

Create `benches/benches/phase4_grover.rs`:

```rust
//! P4-02 Grover benchmark over the committed corpus QASM (the SAME files Aer
//! times), run through the canonical optimized state-vector path. This is the
//! aleph half of the Phase-4 Grover report row. Mirrors phase4_qft.rs.
//!
//! n=4/8/12 run anywhere (state <= 4096 amplitudes). n=16 is 2.26M gates; its
//! report number is taken from the `oneshot` single-shot path in the EPYC run
//! (same split as QFT n=30), but it is available here behind `scaling-bench`
//! for spot checks.
//!
//! Corpus: scripts/qiskit-baseline/circuits/grover_n{N}_iters{opt}.qasm.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// Grover sizes always benched (tiny state, fine for any host / CI bench job).
const SMALL_N: &[u32] = &[4, 8, 12];
/// Large size gated behind `scaling-bench` (2.26M gates). Report number comes
/// from `oneshot`; this path stays runnable for spot checks.
#[cfg(feature = "scaling-bench")]
const LARGE_N: &[u32] = &[16];

/// round(pi/4 * sqrt(2^n)); mirrors run.py::grover_optimal_iters so the corpus
/// filename matches.
fn optimal_iters(n: u32) -> u32 {
    (std::f64::consts::PI / 4.0 * (2f64.powi(n as i32)).sqrt()).round() as u32
}

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().expect("workspace root").join(format!(
        "scripts/qiskit-baseline/circuits/grover_n{n}_iters{}.qasm",
        optimal_iters(n)
    ))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read grover_n{n}: {e}"));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse grover_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            // Raised cap matches phase4_qft (only gates allocate(); small n
            // unaffected). All grover n <= 16 are well under the default anyway.
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn grover(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_grover");
    group.sample_size(10);
    for &n in SMALL_N {
        bench_one(&mut group, n);
    }
    #[cfg(feature = "scaling-bench")]
    for &n in LARGE_N {
        bench_one(&mut group, n);
    }
    group.finish();
}

criterion_group!(benches, grover);
criterion_main!(benches);
```

- [ ] **Step 3: Verify the bench compiles and runs (quick, reduced sampling)**

Run: `cargo bench -p aleph-benches --bench phase4_grover -- --quick 2>&1 | tail -20` (if `--quick` is unsupported by the installed criterion, use a plain `cargo bench -p aleph-benches --bench phase4_grover`).
Expected: `phase4_grover/4`, `/8`, `/12` report times with no panic. (This is a local correctness/compile check, not a measurement — real numbers come from the EPYC step.)

- [ ] **Step 4: Confirm the workspace bench build is still clean**

Run: `cargo build -p aleph-benches --benches`
Expected: builds with no warnings.

- [ ] **Step 5: Commit**

```bash
git add benches/benches/phase4_grover.rs benches/Cargo.toml
git commit -m "[P4-02] phase4_grover criterion bench over the Grover corpus

Mirror of phase4_qft: n=4/8/12 via run_optimized on NaiveSvBackend; n=16 behind
scaling-bench (report value from oneshot). BenchmarkId param is n so the
extractor keys rows grover_n{n}.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Extend the EPYC measurement docs

**Files:**
- Modify: `scripts/bench-report/README.md`
- Modify: `scripts/qiskit-baseline/README.md`

Document the grover flow + the merge-into-existing-JSON step (grover is appended to the already-measured QFT JSONs, not regenerated alongside it).

- [ ] **Step 1: Append a Grover subsection to `scripts/bench-report/README.md`**

After the existing "## 4. Render the report" section (the family-agnostic note), add:

````markdown
## Adding Grover (P4-02) — worked example

Grover is *appended* to the already-measured QFT JSONs (re-timing QFT costs
hours). Run from an idle EPYC box (`uptime` ~0, no `cargo bench`/`bencher run`
in `pgrep`), single-thread both sides.

Note: the 34 MB `grover_n16_iters201.qasm` corpus is **not committed**
(gitignored — see `circuits/.gitignore`). Step (a) below regenerates the whole
grover corpus, including n=16, as a side effect (`transpile_and_export` runs for
every matrix entry before timing), so the n=16 oneshot / scaling-bench in step
(b) find the file present. If you skip step (a), run
`python scripts/qiskit-baseline/run.py --gen-only` first.

```
# (a) Aer: time only the grover keys -> results-qiskit.json (grover rows only).
#     Also (re)generates circuits/grover_n*_iters*.qasm including the n=16 blob.
cd scripts/qiskit-baseline
taskset -c 0 .venv/bin/python run.py \
    --workloads grover_n4,grover_n8,grover_n12,grover_n16
cd ../..

# (b) aleph: criterion for n=4,8,12; oneshot for n=16
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench phase4_grover -- --sample-size 10
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo build --release -p aleph-benches --bin oneshot
RAYON_NUM_THREADS=1 ./target/release/oneshot \
  scripts/qiskit-baseline/circuits/grover_n16_iters201.qasm   # prints elapsed_ms

# (c) extract aleph grover medians to a temp file
python3 scripts/bench-report/extract_criterion.py \
    --criterion-root target/criterion --group phase4_grover --family grover \
    --out /tmp/phase4-aleph-grover.json

# (d) merge grover into the existing phase4 JSONs (preserves QFT rows), then
#     paste the n=16 oneshot median where indicated:
python3 - "$ONESHOT_N16_MS" <<'PY'
import json, sys
from pathlib import Path
n16_ms = float(sys.argv[1])
base = Path("docs/perf/data")
aleph = json.loads((base/"phase4-aleph.json").read_text())
aleph["workloads"].update(
    json.loads(Path("/tmp/phase4-aleph-grover.json").read_text())["workloads"])
aleph["workloads"]["grover_n16"] = {
    "n": 16, "family": "grover", "aleph_ms_median": n16_ms, "aleph_rsd": 0.0}
(base/"phase4-aleph.json").write_text(json.dumps(aleph, indent=2) + "\n")
aer = json.loads((base/"phase4-aer.json").read_text())
grover_aer = json.loads(Path("scripts/qiskit-baseline/results-qiskit.json").read_text())
aer["workloads"].update(
    {k: v for k, v in grover_aer["workloads"].items() if v["family"] == "grover"})
(base/"phase4-aer.json").write_text(json.dumps(aer, indent=2))
print("merged grover rows:",
      sorted(k for k in aleph["workloads"] if k.startswith("grover")))
PY

# (e) re-render the report (Grover section appears automatically)
python3 scripts/bench-report/report.py \
    --aleph docs/perf/data/phase4-aleph.json \
    --aer   docs/perf/data/phase4-aer.json \
    --meta  docs/perf/data/phase4-meta.json \
    --out   docs/perf/phase4.md
```
````

- [ ] **Step 2: Note the grover sizes + key/stem split in `scripts/qiskit-baseline/README.md`**

Find the `timing_runs_for` mention (around line 67) and update the surrounding prose to note (a) the grover family is now optimal-iteration n∈{4,8,12,16}, (b) `--workloads` takes *keys* (`grover_n16`, not the iters-suffixed filename), and (c) `timing_runs_for(n, gate_count)` now budgets on `gate_count * 2^n`. Keep it to 2–4 sentences; match the file's existing terse style. Example insertion:

```markdown
The grover family is optimal-iteration (`round(pi/4*sqrt(2^n))`) at n in
{4,8,12,16}; corpus files embed the iter count (`grover_n16_iters201.qasm`) but
the `--workloads` selector and results-JSON keys use `grover_n{n}`. Aer run
counts come from `timing_runs_for(n, gate_count)` (budgets on gate_count*2^n, so
the 2.26M-gate n=16 gets 2 runs).
```

- [ ] **Step 3: Commit**

```bash
git add scripts/bench-report/README.md scripts/qiskit-baseline/README.md
git commit -m "[P4-02] docs: EPYC Grover measurement flow (append-to-existing JSONs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: EPYC measurement + report regeneration

**Files (outputs):**
- Modify: `docs/perf/data/phase4-aleph.json`
- Modify: `docs/perf/data/phase4-aer.json`
- Modify: `docs/perf/data/phase4-meta.json` (notes only)
- Modify: `docs/perf/phase4.md`

This runs on the EPYC bench box (`ssh root@195.154.249.85`). **Verify idle first** (per CLAUDE.md + memory): `uptime` load ≈ 0 and `pgrep -af "cargo bench|bencher run|Runner.Worker"` empty. Transfer the branch via a git bundle (avoid pushing to a CI-watched ref while measuring).

- [ ] **Step 1: Confirm the box is idle**

Run (on EPYC): `uptime && pgrep -af "cargo bench|bencher run|Runner.Worker" || echo IDLE`
Expected: load average near 0 and `IDLE`. If a job is running, wait or use the secondary box — but note the Ryzen box lacks AVX-512 and the NUMA box is 2-socket; the report's single-thread numbers should come from the same EPYC host as the existing QFT rows for consistency.

- [ ] **Step 2: Sync the branch to EPYC via git bundle**

On the Mac:
```bash
git bundle create /tmp/p4-02.bundle origin/main..p4-02-grover-bench
scp /tmp/p4-02.bundle root@195.154.249.85:/tmp/
```
On EPYC (in the existing aleph checkout):
```bash
git fetch /tmp/p4-02.bundle p4-02-grover-bench:p4-02-grover-bench
git checkout p4-02-grover-bench && git log --oneline -1   # confirm HEAD matches
```

- [ ] **Step 3: Time Aer for the grover keys (single-thread)**

Run (on EPYC): follow `scripts/bench-report/README.md` "Adding Grover" step (a) — `taskset -c 0 .venv/bin/python run.py --workloads grover_n4,grover_n8,grover_n12,grover_n16`.
Expected: console prints median ms per grover key; `scripts/qiskit-baseline/results-qiskit.json` now holds 4 grover rows with `timing_runs` = 10/10/5/2 for n = 4/8/12/16.

- [ ] **Step 4: Bench aleph (criterion n=4/8/12 + oneshot n=16)**

Run (on EPYC): step (b) from the README. Record the `elapsed_ms` printed by `oneshot` for `grover_n16_iters201.qasm` — call it `$ONESHOT_N16_MS`.
Expected: criterion writes `target/criterion/phase4_grover/{4,8,12}/new/estimates.json`; oneshot prints `elapsed_ms <ms>`.

- [ ] **Step 5: Extract + merge into the phase4 JSONs**

Run (on EPYC): steps (c) and (d) from the README, passing `$ONESHOT_N16_MS` to the merge snippet.
Expected: the merge snippet prints `merged grover rows: ['grover_n12', 'grover_n16', 'grover_n4', 'grover_n8']`; `phase4-aleph.json` and `phase4-aer.json` now contain both qft and grover workloads.

- [ ] **Step 6: Update the meta notes for the grover addition**

Edit `docs/perf/data/phase4-meta.json`'s `notes` field to mention Grover, e.g. append: ` Grover uses optimal iterations (round(pi/4*sqrt(2^n))); n=16 aleph from oneshot, Aer median of 2.` Leave `date`/`host`/`toolchain`/`qiskit` unless the measurement host or toolchain differs from the committed values — if it does, update them to the actual EPYC values used for this run.

- [ ] **Step 7: Re-render the report and eyeball it**

Run (on EPYC): step (e) from the README, then `sed -n '1,40p' docs/perf/phase4.md`.
Expected: a new `## Grover` section (sorted before `## QFT`) with rows `grover_n4`, `grover_n8`, `grover_n12`, `grover_n16`, each with gate counts 268/17974/264312/2258854 and a finite `aleph / Aer` ratio. The QFT section is unchanged.

- [ ] **Step 8: Confirm `report.py` is deterministic on the merged inputs**

Run (on EPYC): re-run step (e) into a temp file and diff:
```bash
python3 scripts/bench-report/report.py --aleph docs/perf/data/phase4-aleph.json \
  --aer docs/perf/data/phase4-aer.json --meta docs/perf/data/phase4-meta.json \
  --out /tmp/phase4-check.md && diff /tmp/phase4-check.md docs/perf/phase4.md && echo IDENTICAL
```
Expected: `IDENTICAL` (pure function, no drift).

- [ ] **Step 9: Bring the results back to the Mac and commit**

Bundle the EPYC commit back (or copy the 4 changed files via scp into the Mac checkout), then on the Mac:
```bash
git add docs/perf/data/phase4-aleph.json docs/perf/data/phase4-aer.json \
        docs/perf/data/phase4-meta.json docs/perf/phase4.md
git commit -m "[P4-02] EPYC Grover measurements -> phase4.md Grover section

Single-thread both sides on EPYC 8124P. Grover rows appended to the existing
QFT JSONs; report.py re-rendered (deterministic). n=16 aleph via oneshot, Aer
median of 2 (cost-budgeted).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full local verification + PR

**Files:** none (verification + PR).

- [ ] **Step 1: Run the full workspace test suite + lint + fmt**

Run:
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
scripts/qiskit-baseline/.venv/bin/python -m unittest discover -s scripts/qiskit-baseline -p 'test_run.py'
python3 -m unittest discover -s scripts/bench-report -p 'test_*.py'
```
Expected: all green. The grover convergence test (n=4/8/12) runs as part of `cargo test`; the report golden test still passes (tooling untouched).

- [ ] **Step 2: Self-review the diff**

Run: `git diff origin/main...HEAD --stat` then re-read the full diff with fresh eyes. Confirm: no legacy `grover_n{15..25}_iters5.qasm` fixture was modified; `extract_criterion.py`/`report.py` untouched; the AC test asserts `> 0.9` and argmax == 1.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin p4-02-grover-bench
gh pr create --title "[P4-02] Grover benchmark" --body "$(cat <<'EOF'
Closes #40

## Summary
Adds Grover (optimal ~round(pi/4*sqrt(2^n)) iterations, single marked state
|0..01>) as the second consumer of the Phase-4 bench/report framework from P4-01.
n in {4,8,12,16}; tiny cache-resident state makes even the 2.26M-gate n=16 case
tractable with no ancilla and no qubit-cap change.

## Approach
- `run.py`: `grover_optimal_iters(n)`, grover family -> {4,8,12,16}, split
  `workload_name` into `corpus_stem` (iters-suffixed filename) + `workload_key`
  (`grover_n{n}`, matches `extract_criterion.py`'s join key), and a cost-based
  `timing_runs_for(n, gate_count)` (budgets on gate_count*2^n).
- New committed corpus `grover_n{4,8,12,16}_iters{opt}.qasm` (single source of
  truth for Aer + aleph).
- `phase4_grover.rs` criterion bench mirroring `phase4_qft.rs` (n=16 behind
  `scaling-bench`; report number from `oneshot`).
- `report.py`/`extract_criterion.py` untouched — Grover section renders
  automatically.

## Tests
- Convergence (AC): marked-state |0..01> probability > 0.9 AND argmax at n=4,8,12
  (CI) + n=16 (`#[ignore]`d, nightly). `cargo test -p aleph-benches --test
  grover_convergence`.
- Python unit tests for the optimal-iters formula, cost-based run budget, and
  key/stem split (skip without Qiskit).
- `cargo test --workspace`, clippy, fmt all green; report golden test unchanged.

## Benchmark numbers
Single-thread both sides on EPYC 8124P. See the new `## Grover` section in
`docs/perf/phase4.md` (n=4/8/12 criterion, n=16 oneshot vs Aer median-of-2).

## Notes / follow-ups
- Legacy `grover_n{15..25}_iters5.qasm` fixtures left untouched (frozen
  Phase-1/2 bench artifacts).
- Out of scope (own tickets): P4-03..07.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR created referencing issue #40. Per CLAUDE.md PR workflow, let it sit, re-review, then merge once CI is green.

---

## Self-review (plan vs. spec)

- **Spec "Grover converges n=4,8,12,16"** → Task 3 (n=4/8/12 CI + n=16 ignored). ✓
- **Spec "benchmark report row / Grover section in phase4.md"** → Tasks 4 + 6 (bench + EPYC render). ✓
- **Spec "marked-state probability > 0.9"** → Task 3 asserts `amps[1].norm_sqr() > 0.9` + argmax. ✓
- **Spec Component 1 (`grover_optimal_iters`, family sizes, builder, workload_name, cost-based timing)** → Task 1 (with the key/stem split resolving the join wrinkle). ✓
- **Spec Component 2 (`phase4_grover.rs`, SMALL_N={4,8,12}, n=16 oneshot, filename embeds iters)** → Task 4. ✓
- **Spec Component 3 (convergence test, index 1, n=16 ignored)** → Task 3. ✓
- **Spec Component 4 (append grover rows, re-run report.py, single-thread both)** → Tasks 5 + 6. ✓
- **Spec "corpus drift guard"** → covered operationally by Task 2 Step 2 (regeneration shows no diff for existing files; new grover files are the deterministic output). A standalone byte-compare unit test was considered but rejected: it would re-transpile the multi-megabyte n=12/16 circuits on every `cargo test`/CI run for marginal value over the Task-2 Step-2 check. **Decision noted for review.**
- **Spec "GROVER_ITERS becomes unused — keep with comment or remove"** → Task 1 Step 3 keeps it with a legacy-fixtures comment. ✓
- **Type consistency:** `optimal_iters`/`grover_optimal_iters` use the identical `round(pi/4*sqrt(2^n))` formula in run.py (Python), the bench, and the test (Rust); corpus filename `grover_n{n}_iters{opt}` is built the same way in all three. `workload_key` == `extract_criterion`'s `{family}_n{n}`. ✓
- **Placeholder scan:** no TBD/"handle errors"/"similar to"; every code step is complete. ✓
