# [P1-14] Phase 1 Performance Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/perf/phase1.md` — the Phase-1 exit report comparing `aleph` (`NaiveSvBackend`, AoS+AVX-512) against Qiskit Aer single-thread across {GHZ, QFT, Grover, random-brickwall} × n∈{15,20,22,25} on the EPYC bench host, and file follow-up issues for any >2× miss.

**Architecture:** Extend the Stage-0 shared-QASM harness (`scripts/qiskit-baseline/run.py` generates QASM + times Aer; `benches/benches/qiskit_baseline.rs` times aleph on the same QASM). Add GHZ + the full n-matrix, a CI-safe gating so the full matrix only runs on explicit opt-in, and a one-shot aleph binary for RSS measurement. Then run on EPYC, collect numbers, write the report.

**Tech Stack:** Python 3.12 + Qiskit/Aer (baseline), Rust + criterion (aleph), `/usr/bin/time -v` (RSS), EPYC `ssh root@195.154.249.85`.

Design spec: `docs/superpowers/specs/2026-05-30-p1-14-phase1-perf-report-design.md`.

---

## File Structure

- **Modify** `scripts/qiskit-baseline/run.py` — add `build_ghz`, parametrise over the n-matrix, add `--gen-only` / `--workloads` / per-n timing-run reduction.
- **Modify** `benches/benches/qiskit_baseline.rs` — full-matrix workload list, per-n throughput + sample budget, CI-safe gating (full matrix only on `ALEPH_BENCH_FULL_MATRIX=1`).
- **Create** `benches/src/bin/oneshot.rs` — load a QASM, run `NaiveSvBackend` once (for `/usr/bin/time -v` RSS). Register `[[bin]]` in `benches/Cargo.toml`.
- **Create** `scripts/qiskit-baseline/circuits/*.qasm` — the generated full-matrix circuits (committed; deterministic).
- **Modify** `scripts/qiskit-baseline/README.md` — document the matrix + new flags.
- **Create** `docs/perf/phase1.md` — the report (filled from EPYC numbers).
- **Modify** `docs/perf/phase1-vs-qiskit.md` — superseded banner + reciprocal links.
- **Modify** `BACKLOG.md` (+ run `scripts/sync-issues.sh`) — follow-up entries for any >2× miss.

**Naming convention (run.py and the Rust bench MUST agree):** `ghz_n{n}`, `qft_n{n}`, `grover_n{n}_iters5`, `random_brickwall_n{n}_d20` for n∈{15,20,22,25}.

Verified harness facts:
- `run.py` already has `build_qft`, `build_grover`, `build_random_brickwall`, `transpile_and_export` (writes `circuits/{name}.qasm`), `time_aer`, `BASIS_GATES = ["h","x","z","rz","rx","ry","cx","cz","ccx","p"]`.
- `qiskit_baseline.rs` loads `scripts/qiskit-baseline/circuits/{name}.qasm` via `aleph_parser::parse`, runs `NaiveSvBackend`/`SoaSvBackend`, has `select_workloads()` (CI-skips grover) and `sample_budget_for()`.
- `benches/Cargo.toml` declares `[[bench]] name="qiskit_baseline" harness=false` and deps `aleph-{core,ir,backend,sv,parser}`, `criterion` (dev).

---

## Task 1: run.py — GHZ builder + full n-matrix + CLI flags

**Files:**
- Modify: `scripts/qiskit-baseline/run.py`

- [ ] **Step 1: Add the GHZ builder** (after `build_random_brickwall`)

```python
def build_ghz(n: int) -> QuantumCircuit:
    """GHZ state on `n` qubits: H on q0, then a CNOT chain q_i -> q_{i+1}."""
    qc = QuantumCircuit(n, name=f"ghz_n{n}")
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc
```

- [ ] **Step 2: Replace the fixed `N_QUBITS`/`WORKLOADS` with a matrix + naming**

Replace the `N_QUBITS = 20` / `WORKLOADS = {...}` block with:

```python
N_QUBITS_LIST = [15, 20, 22, 25]
GROVER_ITERS = 5
RANDOM_DEPTH = 20

FAMILY_BUILDERS = {
    "ghz": lambda n: build_ghz(n),
    "qft": lambda n: build_qft(n),
    "grover": lambda n: build_grover(n, GROVER_ITERS),
    "random_brickwall": lambda n: build_random_brickwall(n, RANDOM_DEPTH),
}


def workload_name(family: str, n: int) -> str:
    if family == "grover":
        return f"grover_n{n}_iters{GROVER_ITERS}"
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"


def all_workloads() -> list[tuple[str, str, int]]:
    """(name, family, n) for the full matrix, families in stable order."""
    return [
        (workload_name(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in N_QUBITS_LIST
    ]


def timing_runs_for(n: int) -> int:
    """Fewer timed Aer runs at large n (each is minutes). Disclosed in the report."""
    if n <= 20:
        return 10
    if n == 22:
        return 5
    return 3  # n == 25
```

- [ ] **Step 3: Generalise `time_aer` to take a run count, and rewrite `main` with argparse**

Change `time_aer(tqc)` signature to `time_aer(tqc, runs: int)` and replace the hardcoded `for _ in range(TIMING_RUNS)` with `for _ in range(runs)`. Then replace `main()`:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Qiskit Aer Phase-1 baseline harness")
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="Only generate/export circuits/*.qasm; do not time Aer.",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default="",
        help="Comma-separated workload names to time (default: full matrix).",
    )
    args = parser.parse_args()

    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = all_workloads()
    selected = (
        set(args.workloads.split(",")) if args.workloads else {n for n, _, _ in matrix}
    )

    results: dict = {
        "schema_version": 2,
        "n_qubits_list": N_QUBITS_LIST,
        "grover_iters": GROVER_ITERS,
        "random_depth": RANDOM_DEPTH,
        "basis_gates": BASIS_GATES,
        "workloads": {},
    }
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
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[done] results -> {RESULTS_PATH}")
```

Delete the now-unused `TIMING_RUNS` constant (its role moved into `timing_runs_for`). Update the module docstring's "Builds three workloads" line to describe the matrix.

- [ ] **Step 4: Smoke-test generation locally (no Aer timing needed for gen)**

Run (needs the venv from `requirements.txt`; if Qiskit isn't installed locally, this task's verification happens on EPYC in Task 6 instead — note that and skip to Step 5):
```
cd scripts/qiskit-baseline && python run.py --gen-only
```
Expected: writes 16 files `circuits/{ghz,qft,grover,random_brickwall}_n{15,20,22,25}*.qasm`, prints gate counts, no timing.

- [ ] **Step 5: Commit**

```bash
git add scripts/qiskit-baseline/run.py
git commit -m "[P1-14] run.py: GHZ builder + full n-matrix + --gen-only/--workloads"
```

---

## Task 2: Generate and commit the full-matrix QASM circuits

**Files:**
- Create: `scripts/qiskit-baseline/circuits/{ghz,qft,grover,random_brickwall}_n{15,20,22,25}*.qasm` (16 files)

This task needs Qiskit. If unavailable on the dev machine, perform it on EPYC (Task 6 Step 1) and commit from there; otherwise do it locally now.

- [ ] **Step 1: Generate**

Run: `cd scripts/qiskit-baseline && python run.py --gen-only`
Expected: 16 `.qasm` files written (the 3 pre-existing n=20 files are overwritten identically except QASM regeneration; verify the n=20 ones are unchanged or a deliberate diff).

- [ ] **Step 2: Confirm the full set was written**

Run: `ls scripts/qiskit-baseline/circuits/*.qasm | wc -l`
Expected: `16` (4 families × 4 n). Real parse-and-simulate verification of every circuit happens in Task 3 Step 4 (the `oneshot` bin), which is the authoritative check that aleph-parser accepts each generated QASM.

- [ ] **Step 3: Commit**

```bash
git add scripts/qiskit-baseline/circuits/
git commit -m "[P1-14] Generate full-matrix baseline QASM circuits (4 families x n{15,20,22,25})"
```

---

## Task 3: aleph one-shot binary for RSS measurement

**Files:**
- Create: `benches/src/bin/oneshot.rs`
- Modify: `benches/Cargo.toml`

- [ ] **Step 1: Write the one-shot binary**

Create `benches/src/bin/oneshot.rs`:

```rust
//! Single-shot runner for peak-RSS measurement under `/usr/bin/time -v`.
//! Loads a QASM circuit and runs `NaiveSvBackend` exactly once. Not a
//! benchmark — the point is a clean process whose Maximum RSS reflects one
//! state-vector simulation. Usage:
//!   /usr/bin/time -v ./oneshot scripts/qiskit-baseline/circuits/qft_n25.qasm

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use std::hint::black_box;

fn main() {
    let path = std::env::args().nth(1).expect("usage: oneshot <circuit.qasm>");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"));
    let circuit = aleph_parser::parse(&src)
        .unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
    let mut backend = NaiveSvBackend::with_seed(0);
    let state = run(&mut backend, &circuit).expect("simulation failed");
    // Touch the result so the optimiser can't elide the work.
    black_box(state.amplitudes().len());
}
```

- [ ] **Step 2: Register the bin** in `benches/Cargo.toml` (after the `[dependencies]` block, before `[dev-dependencies]`)

```toml
[[bin]]
name = "oneshot"
path = "src/bin/oneshot.rs"
```

- [ ] **Step 3: Build and smoke-test on a small circuit**

Run: `cargo build -p aleph-benches --bin oneshot && ./target/debug/oneshot scripts/qiskit-baseline/circuits/ghz_n15.qasm && echo OK`
Expected: prints nothing but exits 0 (`OK`). This also proves aleph parses the generated GHZ QASM.

- [ ] **Step 4: Verify it parses every generated circuit** (real parse-coverage check)

Run: `for f in scripts/qiskit-baseline/circuits/*.qasm; do ./target/debug/oneshot "$f" >/dev/null || { echo "FAIL $f"; break; }; done && echo ALL_PARSED`
Expected: `ALL_PARSED` (every one of the 16 circuits parses + simulates; n=25 will use ~512 MiB and take a while in debug — acceptable for a one-time check, or restrict to n≤20 here and let EPYC cover n=25).

- [ ] **Step 5: Commit**

```bash
git add benches/src/bin/oneshot.rs benches/Cargo.toml
git commit -m "[P1-14] one-shot QASM runner for peak-RSS measurement"
```

---

## Task 4: qiskit_baseline.rs — full matrix + CI-safe gating

**Files:**
- Modify: `benches/benches/qiskit_baseline.rs`

- [ ] **Step 1: Replace the workload list + selection with the matrix + opt-in gating**

Replace `ALL_WORKLOADS`, `select_workloads`, and `sample_budget_for` with:

```rust
const N_LIST: &[u32] = &[15, 20, 22, 25];
const FAMILIES: &[&str] = &["ghz", "qft", "grover", "random_brickwall"];

fn workload_name(family: &str, n: u32) -> String {
    match family {
        "grover" => format!("grover_n{n}_iters5"),
        "random_brickwall" => format!("random_brickwall_n{n}_d20"),
        _ => format!("{family}_n{n}"),
    }
}

/// (name, n). Full matrix only when `ALEPH_BENCH_FULL_MATRIX=1`; otherwise a
/// cheap CI subset (n<=20, no grover) that stays well under the Bench
/// workflow's 30-minute timeout. The full matrix is a manual EPYC run.
fn selected_workloads() -> Vec<(String, u32)> {
    let full = std::env::var("ALEPH_BENCH_FULL_MATRIX")
        .map(|v| v != "0")
        .unwrap_or(false);
    let mut out = Vec::new();
    for &family in FAMILIES {
        for &n in N_LIST {
            if !full && (n > 20 || family == "grover") {
                continue; // CI subset: fast cells only
            }
            out.push((workload_name(family, n), n));
        }
    }
    out
}

/// Per-cell criterion budget. Large n is minutes/iter, so shrink the sample
/// count (disclosed in the report's RSD table). Grover is the most expensive.
fn sample_budget_for(name: &str, n: u32) -> (usize, Duration) {
    if name.starts_with("grover_") && n >= 22 {
        (10, Duration::from_secs(30))
    } else if n >= 22 {
        (10, Duration::from_secs(20))
    } else if name.starts_with("grover_") {
        (10, Duration::from_secs(20))
    } else {
        (50, Duration::from_secs(10))
    }
}
```

- [ ] **Step 2: Update `bench_qiskit_baseline` to use per-workload n (throughput + budget)**

Replace the body that sets a group-wide `throughput(Throughput::Elements(20 ...))` and iterates `select_workloads()` with per-workload throughput and the `(name, n)` pairs:

```rust
fn bench_qiskit_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("qiskit_baseline");

    for (name, n) in selected_workloads() {
        // n·2^n elements/s axis (matches benches/qft.rs).
        group.throughput(Throughput::Elements(n as u64 * (1u64 << n)));
        let (samples, m_time) = sample_budget_for(&name, n);
        group.sample_size(samples).measurement_time(m_time);

        let path = fixture_path(&name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
        let circuit = aleph_parser::parse(&src)
            .unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e));

        group.bench_with_input(
            BenchmarkId::new("naive_aos_avx512", &name),
            &circuit,
            |b, circuit| {
                b.iter_with_setup(
                    || NaiveSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            },
        );

        // SoA appendix only at n<=20 (known ~2.3x slower; skip the n>=22 time).
        if n <= 20 {
            group.bench_with_input(BenchmarkId::new("soa", &name), &circuit, |b, circuit| {
                b.iter_with_setup(
                    || SoaSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            });
        }
    }
    group.finish();
}
```

Keep `fixture_path` as-is. `BenchmarkId::new` accepts `&String`/`&str`.

- [ ] **Step 3: Compile the bench (CI subset path)**

Run: `cargo bench -p aleph-benches --bench qiskit_baseline --no-run`
Expected: compiles clean.

- [ ] **Step 4: Confirm the CI subset is small and runnable** (default env, no full matrix)

Run: `cargo bench -p aleph-benches --bench qiskit_baseline -- --test 2>&1 | tail -20`
Expected: runs only the cheap subset (ghz/qft/random at n=15,20 — 6 workloads × naive+soa) in `--test` mode (one iteration each), proving every CI-subset QASM file is present and parses. (n=20 simulations in `--test` mode are one-shot, seconds.)

- [ ] **Step 5: Lint + format**

Run: `cargo clippy -p aleph-benches --all-targets -- -D warnings && cargo fmt --check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add benches/benches/qiskit_baseline.rs
git commit -m "[P1-14] qiskit_baseline bench: full matrix + CI-safe gating (ALEPH_BENCH_FULL_MATRIX)"
```

---

## Task 5: README — document the matrix and flags

**Files:**
- Modify: `scripts/qiskit-baseline/README.md`

- [ ] **Step 1: Update the README**

Update the intro and "Circuits" / "Reproducing" sections to describe: the 4-family × n∈{15,20,22,25} matrix; `python run.py --gen-only` (regenerate QASM), `python run.py` (full matrix Aer timing), `python run.py --workloads name1,name2` (subset); the aleph side `ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench qiskit_baseline`; the one-shot RSS command `/usr/bin/time -v ./target/release/oneshot circuits/<name>.qasm`; and that the report is `docs/perf/phase1.md`. Keep the existing pinning/`taskset` guidance.

- [ ] **Step 2: Commit**

```bash
git add scripts/qiskit-baseline/README.md
git commit -m "[P1-14] README: document full-matrix harness + RSS one-shot"
```

---

## Task 6: EPYC measurement run (manual — the heavy phase)

**Files:** produces raw numbers (committed under `scripts/qiskit-baseline/results-qiskit.json` + a criterion estimates capture). No source changes.

> This task requires the EPYC host and takes hours (Grover/random n=25 are the long poles). It cannot be unit-tested; each step lists the exact command and what to capture. Run it in one sitting on an otherwise-idle runner; do NOT push to `benches/**` during the run (CI Bench races the same runner — Stage-0 lesson).

- [ ] **Step 1: Sync the branch on EPYC and build**

```
ssh root@195.154.249.85
cd /tmp/aleph-forensics && rm -rf aleph && git clone https://github.com/ruslan-splynx/aleph.git && cd aleph
git checkout p1-14-perf-report
RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-benches --bin oneshot
```

- [ ] **Step 2: Generate circuits + time Aer (full matrix), pinned**

```
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 python run.py
```
Capture: `results-qiskit.json` (per-workload Aer median/stdev + gate counts).

- [ ] **Step 3: Time aleph (full matrix), AVX-512 verified**

```
cd ../..
ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench qiskit_baseline -- --save-baseline phase1-final
```
Capture: criterion medians per `(naive_aos_avx512|soa, name)` from `target/criterion/**/estimates.json` (or the bencher.dev upload). Verify AVX-512: `objdump -d target/release/deps/qiskit_baseline-* | grep -c 'vmulpd.*zmm'` is > 0.

- [ ] **Step 4: Measure peak RSS at the headline n (n=25 each family; aleph + Aer)**

aleph (per family at n=25):
```
for name in ghz_n25 qft_n25 grover_n25_iters5 random_brickwall_n25_d20; do
  echo "== $name =="
  /usr/bin/time -v ./target/release/oneshot scripts/qiskit-baseline/circuits/$name.qasm 2>&1 | grep 'Maximum resident'
done
```
Aer (per family at n=25) — single-shot via the subset flag with 1 timing run is not exposed; instead wrap a one-circuit Aer run. Use `python run.py --workloads <name>` (it times `timing_runs_for(25)=3` runs) under `/usr/bin/time -v` and read Maximum RSS (peak across the 3 runs is representative):
```
for name in ghz_n25 qft_n25 grover_n25_iters5 random_brickwall_n25_d20; do
  echo "== $name =="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
    /usr/bin/time -v python scripts/qiskit-baseline/run.py --workloads $name 2>&1 | grep 'Maximum resident'
done
```
Capture: the 8 Maximum-RSS values (KB → convert to MiB).

- [ ] **Step 5: Record raw numbers**

Commit `results-qiskit.json` and paste the criterion medians + RSS values into a scratch section at the BOTTOM of `docs/perf/phase1.md` (Task 7 turns them into the tables). Commit from EPYC or copy back:
```bash
git add scripts/qiskit-baseline/results-qiskit.json
git commit -m "[P1-14] EPYC raw Aer baseline results (full matrix)"
```

---

## Task 7: Write `docs/perf/phase1.md`

**Files:**
- Create: `docs/perf/phase1.md`

- [ ] **Step 1: Write the report from the Task-6 numbers**, following the spec §5 structure exactly:
  - Header (date, EPYC 8124P host, Rust version + `target-cpu=native` + AVX-512 verified, Python/Qiskit/Aer/numpy/scipy versions, pin command, link to `scripts/qiskit-baseline/README.md`).
  - One headline table per algorithm (rows n=15/20/22/25; columns `aleph (ms)`, `Aer (ms)`, `aleph/Aer`, `time/gate (ns)` = aleph_ms·1e6 ÷ gate_count, `peak RSS (MiB)`, `theoretical (MiB)` = `(1<<n)*16/2^20`, `≤2× verdict`).
  - ROADMAP §7 exit verdict at n=25 for QFT/Grover/random (+ GHZ trend).
  - Trend section (aleph/Aer across n).
  - Backend appendix: NaiveSv vs SoA at n≤20.
  - RSD / sample-count table (per-cell `sample_size` from `sample_budget_for`/`timing_runs_for` + relative stdev).
  - Interpretation + Known gaps (link issues from Task 8).
  - Reproducibility (the exact Task-6 commands).

- [ ] **Step 2: Sanity-check arithmetic**

Confirm each `aleph/Aer` ratio = aleph_ms ÷ Aer_ms; each `theoretical (MiB)` = `2^n × 16 / 1048576` (n=15→0.5, n=20→16, n=22→64, n=25→512); `time/gate` uses the post-transpile gate count from `results-qiskit.json`.

- [ ] **Step 3: Commit**

```bash
git add docs/perf/phase1.md
git commit -m "[P1-14] Phase 1 performance report (docs/perf/phase1.md)"
```

---

## Task 8: Misses → BACKLOG + issues; supersede Stage-0 report

**Files:**
- Modify: `BACKLOG.md`, `docs/perf/phase1-vs-qiskit.md`
- Run: `scripts/sync-issues.sh`

- [ ] **Step 1: For each cell >2× Aer**, add a BACKLOG entry

For every workload/n where `aleph/Aer > 2`, add a follow-up entry under a new `### [P1-FU-NN] <workload> exceeds 2× Aer at n=<n>` block in `BACKLOG.md` (Phase-1 follow-up / Phase-2 area), describing the gap, the measured ratio, and the suspected kernel cause (e.g. QFT controlled-Phase routing). If NO cell exceeds 2×, skip this step and state "exit criterion fully met; no follow-ups filed" in the report's Known-gaps section.

- [ ] **Step 2: Sync issues** (only if entries were added)

Run: `bash scripts/sync-issues.sh`
Expected: creates the GitHub issue(s); note the issue number(s) and link them from `docs/perf/phase1.md` Known-gaps.

- [ ] **Step 3: Add the superseded banner to the Stage-0 report**

At the top of `docs/perf/phase1-vs-qiskit.md`, add:
```markdown
> **Superseded by [`phase1.md`](./phase1.md).** This is the Stage-0 snapshot
> (2026-05-27, pre-P1-06/07) kept for historical reference. The canonical
> Phase-1 closure numbers — full {GHZ,QFT,Grover,random} × n{15,20,22,25}
> matrix on the post-P1-13 backend — live in `phase1.md`.
```
And add a reciprocal "supersedes the Stage-0 snapshot in `phase1-vs-qiskit.md`" line near the top of `phase1.md`.

- [ ] **Step 4: Commit**

```bash
git add BACKLOG.md docs/perf/phase1-vs-qiskit.md docs/perf/phase1.md
git commit -m "[P1-14] File >2x follow-ups + supersede Stage-0 report"
```

---

## Task 9: Verification

**Files:** none (verification only)

- [ ] **Step 1: Harness builds + lints**

Run: `cargo build -p aleph-benches --bin oneshot && cargo bench -p aleph-benches --bench qiskit_baseline --no-run && cargo clippy -p aleph-benches --all-targets -- -D warnings && cargo fmt --check`
Expected: all clean.

- [ ] **Step 2: CI subset stays cheap**

Run: `cargo bench -p aleph-benches --bench qiskit_baseline -- --test 2>&1 | tail -5`
Expected: runs only the cheap subset (no n>20, no grover) — confirms CI won't blow the 30-min Bench timeout. (Full matrix only with `ALEPH_BENCH_FULL_MATRIX=1`.)

- [ ] **Step 3: Report internal consistency**

Re-read `docs/perf/phase1.md`: every cell has all columns filled (no blanks), ratios and theoretical-memory arithmetic check out, the exit verdict matches the n=25 numbers, and every Known-gap links a filed issue (or states none needed).

- [ ] **Step 4: Confirm `git status` clean and the branch is coherent**

Run: `git status --porcelain && git log --oneline main..HEAD`
Expected: clean tree; a clean sequence of `[P1-14]` commits.

---

## Notes / follow-ups

- **`[meta]` Phase-1 fixup is a SEPARATE follow-on ticket** (not this plan): flip ROADMAP §7 exit checkboxes, update CLAUDE.md "Project Overview" (currently says "Phase 0"). Do it after this report lands.
- **Execution reality:** Tasks 1–5 + 9 are local/code (verifiable without EPYC). Tasks 6–8 require the EPYC host and are the multi-hour measurement phase; drive them in one sitting on an idle runner.
- **Do not push to `benches/**` while an EPYC measurement is in flight** — CI Bench shares the runner (Stage-0 lesson).
- **`/code-review`** the harness code (Tasks 1–5) before merging; the report prose (Task 7) benefits from a read-through but isn't code.
