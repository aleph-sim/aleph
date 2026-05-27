# Stage 0 — Qiskit Aer baseline on EPYC: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `[meta]` PR that produces a reproducible single-thread baseline comparison between aleph's `NaiveSvBackend` (AoS + AVX-512, post-P1-03) and Qiskit Aer on EPYC across QFT-20, Grover-20 (10 iters), and random-brickwall-20.

**Architecture:** Qiskit builds all three circuits, transpiles to a restricted basis aleph-parser supports, exports QASM3 files into `scripts/qiskit-baseline/circuits/`. A Python harness times Aer; a hermetic Rust criterion bench (`benches/benches/qiskit_baseline.rs`) reads the same `.qasm` files and times aleph. Results land in `docs/perf/phase1-vs-qiskit.md`.

**Tech Stack:** Python 3.11+, Qiskit 1.x, Qiskit Aer 0.15.x, Rust 1.89, criterion 0.5, existing aleph workspace (`aleph-parser`, `aleph-sv`, `aleph-backend`, `aleph-benches`).

**Spec:** `docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md`

**Pre-flight note for the executing agent:**
- The spec mentions filing a follow-up issue for "QASM3 emitter for aleph-ir". This is **moot** — `aleph_parser::emit` already exists (see `crates/aleph-parser/src/emit.rs`, 197 LOC, added during P0-08 Task 15). No follow-up issue should be filed. Mention this in the PR body and update the spec inline as part of Task 11.

---

## File Structure

**New files:**
- `scripts/qiskit-baseline/README.md` — reproducibility instructions
- `scripts/qiskit-baseline/requirements.txt` — pinned Python dependencies
- `scripts/qiskit-baseline/run.py` — builds circuits, transpiles, exports QASM, times Aer
- `scripts/qiskit-baseline/.gitignore` — ignores `.venv/`, `results-qiskit.json`
- `scripts/qiskit-baseline/circuits/qft_n20.qasm` (generated, checked in)
- `scripts/qiskit-baseline/circuits/grover_n20_iters10.qasm` (generated, checked in)
- `scripts/qiskit-baseline/circuits/random_brickwall_n20_d20.qasm` (generated, checked in)
- `benches/benches/qiskit_baseline.rs` — criterion bench reading the `.qasm` files
- `docs/perf/phase1-vs-qiskit.md` — populated comparison report

**Modified files:**
- `benches/Cargo.toml` — add `[[bench]]` entry for `qiskit_baseline`, add `aleph-parser` dep
- `docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md` — strike the moot "QASM emitter follow-up issue" line (Task 11)

**Untouched:** No changes under `crates/`. No new CI job.

---

## Task 1: Set up `scripts/qiskit-baseline/` skeleton

**Files:**
- Create: `scripts/qiskit-baseline/README.md`
- Create: `scripts/qiskit-baseline/requirements.txt`
- Create: `scripts/qiskit-baseline/.gitignore`
- Create: `scripts/qiskit-baseline/circuits/.gitkeep`

- [ ] **Step 1: Create the directory and `.gitignore`**

```bash
mkdir -p scripts/qiskit-baseline/circuits
```

Write `scripts/qiskit-baseline/.gitignore`:

```
.venv/
__pycache__/
*.pyc
results-qiskit.json
```

- [ ] **Step 2: Write `requirements.txt` with pinned versions**

Write `scripts/qiskit-baseline/requirements.txt`:

```
qiskit==1.2.4
qiskit-aer==0.15.1
numpy==2.1.3
```

(These are the latest stable as of 2026-05; the executing agent may bump them after `pip install` if a newer minor exists, but pins must be exact, not floating.)

- [ ] **Step 3: Write `README.md`**

Write `scripts/qiskit-baseline/README.md`:

````markdown
# Qiskit Aer baseline (Phase 1, Stage 0)

Reproducibility harness for `docs/perf/phase1-vs-qiskit.md`. Produces a single-thread, same-circuit comparison between aleph and Qiskit Aer across QFT-20, Grover-20 (10 iters), and random-brickwall-20.

## Reproducing on EPYC

```bash
# 1. Time Qiskit Aer
ssh root@195.154.249.85
cd /tmp/aleph-forensics                 # NOT the GH Actions runner workdir
git clone https://github.com/<you>/aleph.git && cd aleph
git checkout <branch>
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python run.py
# Produces results-qiskit.json and writes circuits/*.qasm if missing.

# 2. Time aleph against the same QASM files
cd ../..
RUSTFLAGS="-C target-cpu=native" cargo bench \
  --bench qiskit_baseline -- --save-baseline phase1-baseline
```

## Reproducing locally (M-series, Linux, etc.)

Same commands minus `taskset` and `OMP_*` pinning, but **note**: local numbers are not authoritative. EPYC + AVX-512 is the comparison target. The Rust bench runs scalar code paths on non-x86-AVX-512 hosts.

## Circuits

The three workloads:

- `circuits/qft_n20.qasm` — Nielsen-Chuang § 5.1 QFT, no closing SWAPs.
- `circuits/grover_n20_iters10.qasm` — Grover-20, 1 marked state (|0…01⟩), 10 iterations.
- `circuits/random_brickwall_n20_d20.qasm` — brick-wall random circuit (Rz/Rx 1q layers + alternating CNOT pairs), depth 20, deterministic angles `cos(layer + qubit*0.37)`.

All three are transpiled by Qiskit to the basis `[h, x, z, rz, rx, ry, cx, cz, ccx, p]` at `optimization_level=0` so we measure engines, not the transpiler.

## Updating the QASM files

`run.py` regenerates `circuits/*.qasm` deterministically on every run. To refresh after a Qiskit version bump, simply re-run; commit the diff.
````

- [ ] **Step 4: Create `circuits/.gitkeep` so the directory ships even before first QASM**

```bash
touch scripts/qiskit-baseline/circuits/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add scripts/qiskit-baseline/
git commit -m "[meta] Stage 0: scaffold scripts/qiskit-baseline/"
```

---

## Task 2: Write `run.py` — circuit construction

**Files:**
- Create: `scripts/qiskit-baseline/run.py`

This task implements the circuit builders only — no Aer timing yet (that lands in Task 4). Splitting keeps tasks bite-sized and the QASM-generation half independently testable.

- [ ] **Step 1: Write the file skeleton**

Write `scripts/qiskit-baseline/run.py`:

```python
"""Qiskit Aer baseline harness for aleph Phase 1, Stage 0.

Builds three workloads (QFT-20, Grover-20 × 10 iters, random-brickwall-20),
transpiles each to the basis aleph-parser supports, exports QASM3, and times
AerSimulator(method='statevector') under single-thread pinning.

Spec: docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import QFT, GroverOperator
from qiskit_aer import AerSimulator

N_QUBITS = 20
GROVER_ITERS = 10
RANDOM_DEPTH = 20
TIMING_RUNS = 10
BASIS_GATES = ["h", "x", "z", "rz", "rx", "ry", "cx", "cz", "ccx", "p"]

CIRCUITS_DIR = Path(__file__).parent / "circuits"
RESULTS_PATH = Path(__file__).parent / "results-qiskit.json"


def build_qft(n: int) -> QuantumCircuit:
    """Textbook QFT on `n` qubits, no closing SWAPs (matches aleph_benches::qft_circuit)."""
    qc = QuantumCircuit(n, name=f"qft_n{n}")
    qc.compose(QFT(num_qubits=n, do_swaps=False, inverse=False), inplace=True)
    return qc


def build_grover(n: int, iters: int) -> QuantumCircuit:
    """Grover on `n` qubits with 1 marked state |0…01⟩, applied `iters` times."""
    # Oracle: flip the phase of |0…01⟩ — encoded as Z on qubit 0 surrounded by
    # X's on every other qubit (so only |0…01⟩ has all-ones after the X layer).
    oracle = QuantumCircuit(n, name="oracle")
    for q in range(1, n):
        oracle.x(q)
    # Multi-controlled Z on the all-ones subspace: H + ccx-chain + H, or simply
    # an n-qubit MCZ via GroverOperator's internal construction. We give the
    # GroverOperator the oracle and let it build the diffusion.
    oracle.h(0)
    oracle.mcx(list(range(1, n)), 0)
    oracle.h(0)
    for q in range(1, n):
        oracle.x(q)

    grover_op = GroverOperator(oracle, insert_barriers=False)

    qc = QuantumCircuit(n, name=f"grover_n{n}_iters{iters}")
    # Initial superposition.
    qc.h(range(n))
    # Iters × Grover operator.
    for _ in range(iters):
        qc.compose(grover_op, inplace=True)
    return qc


def build_random_brickwall(n: int, depth: int) -> QuantumCircuit:
    """Brick-wall random circuit mirroring `aleph_benches::random_brickwall_circuit`
    bit-for-bit.  Angles are deterministic: cos(layer + qubit*0.37) and the same * 1.13.
    """
    qc = QuantumCircuit(n, name=f"random_brickwall_n{n}_d{depth}")
    for layer in range(depth):
        for q in range(n):
            theta = math.cos(layer + q * 0.37)
            qc.rz(theta, q)
            qc.rx(theta * 1.13, q)
        offset = layer & 1
        q = offset
        while q + 1 < n:
            qc.cx(q, q + 1)
            q += 2
    return qc


WORKLOADS = {
    "qft_n20": lambda: build_qft(N_QUBITS),
    "grover_n20_iters10": lambda: build_grover(N_QUBITS, GROVER_ITERS),
    "random_brickwall_n20_d20": lambda: build_random_brickwall(N_QUBITS, RANDOM_DEPTH),
}


def transpile_and_export(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    """Transpile to aleph's basis (level 0, no optimisation) and write QASM3."""
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    qasm = qasm3.dumps(tqc)
    out = CIRCUITS_DIR / f"{name}.qasm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(qasm)
    return tqc


def main() -> None:
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    transpiled = {}
    for name, builder in WORKLOADS.items():
        print(f"[build] {name} …", flush=True)
        qc = builder()
        tqc = transpile_and_export(qc, name)
        print(
            f"[build] {name}: {len(tqc.data)} gates after transpile "
            f"(was {len(qc.data)} pre-transpile)",
            flush=True,
        )
        transpiled[name] = tqc
    # (Aer timing lands in Task 4.)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test locally**

```bash
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
ls -la circuits/
```

Expected: three `.qasm` files written. Print lines for each workload show gate counts. `circuits/qft_n20.qasm` should be ~5-10 KB; the others larger.

- [ ] **Step 3: Sanity-check that aleph-parser can read each file**

```bash
cd ../..
cat > /tmp/parse_check.rs <<'EOF'
fn main() {
    for path in [
        "scripts/qiskit-baseline/circuits/qft_n20.qasm",
        "scripts/qiskit-baseline/circuits/grover_n20_iters10.qasm",
        "scripts/qiskit-baseline/circuits/random_brickwall_n20_d20.qasm",
    ] {
        let src = std::fs::read_to_string(path).unwrap();
        match aleph_parser::parse(&src) {
            Ok(c) => println!("OK {} → {} gates, {} qubits", path, c.gates().len(), c.num_qubits()),
            Err(e) => panic!("FAIL {}: {:?}", path, e),
        }
    }
}
EOF
# Quick ad-hoc check: drop this into a temporary scratch file. If executing inline
# this can be skipped — Task 5's bench compile will hit any parser breakage.
```

Don't commit this scratch — it's just a verification. The authoritative parse-then-run check lives in Task 5.

- [ ] **Step 4: Commit**

```bash
git add scripts/qiskit-baseline/run.py scripts/qiskit-baseline/circuits/*.qasm
git rm -f scripts/qiskit-baseline/circuits/.gitkeep   # superseded by real files
git commit -m "[meta] Stage 0: generate QASM circuits via Qiskit"
```

---

## Task 3: Verify aleph-parser accepts every transpiled gate

**Files:**
- Test: `crates/aleph-parser/tests/qiskit_baseline_fixtures.rs` (new integration test)

This task pins a regression — if a future Qiskit version emits a gate aleph doesn't parse, CI fails loudly. Belt-and-braces alongside Task 2 Step 3's smoke check.

- [ ] **Step 1: Write the integration test**

Create `crates/aleph-parser/tests/qiskit_baseline_fixtures.rs`:

```rust
//! Regression test: every Stage 0 QASM circuit parses cleanly.
//!
//! Files generated by `scripts/qiskit-baseline/run.py` and checked into the
//! repo.  If a future Qiskit version emits a gate aleph-parser doesn't
//! support, this test fails — we then either widen the parser or restrict
//! Qiskit's basis_gates list further.

use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two dirs deep from repo root")
        .to_path_buf();
    repo_root
        .join("scripts/qiskit-baseline/circuits")
        .join(name)
}

#[test]
fn parses_qft_n20() {
    let src = std::fs::read_to_string(fixture_path("qft_n20.qasm")).unwrap();
    let circuit = aleph_parser::parse(&src).expect("qft_n20.qasm must parse");
    assert_eq!(circuit.num_qubits(), 20);
    assert!(!circuit.gates().is_empty());
}

#[test]
fn parses_grover_n20_iters10() {
    let src =
        std::fs::read_to_string(fixture_path("grover_n20_iters10.qasm")).unwrap();
    let circuit = aleph_parser::parse(&src).expect("grover_n20_iters10.qasm must parse");
    assert_eq!(circuit.num_qubits(), 20);
}

#[test]
fn parses_random_brickwall_n20_d20() {
    let src = std::fs::read_to_string(fixture_path("random_brickwall_n20_d20.qasm"))
        .unwrap();
    let circuit =
        aleph_parser::parse(&src).expect("random_brickwall_n20_d20.qasm must parse");
    assert_eq!(circuit.num_qubits(), 20);
}
```

- [ ] **Step 2: Run the test, expect PASS**

```bash
cargo test -p aleph-parser --test qiskit_baseline_fixtures
```

Expected: 3 passed.

If any test fails with a parser error citing a specific gate, narrow `BASIS_GATES` in `run.py` to exclude the offending mnemonic and regenerate (loop back to Task 2 Step 2).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-parser/tests/qiskit_baseline_fixtures.rs
git commit -m "[meta] Stage 0: regression test for QASM fixture parseability"
```

---

## Task 4: `run.py` — Aer timing + JSON output

**Files:**
- Modify: `scripts/qiskit-baseline/run.py`

- [ ] **Step 1: Append the timing logic**

Modify `scripts/qiskit-baseline/run.py` — replace the `main()` function with:

```python
def time_aer(tqc: QuantumCircuit) -> dict:
    """Run `tqc` through AerSimulator(method='statevector') TIMING_RUNS times
    under single-thread pinning. Returns dict with median, mean, stdev (seconds)."""
    sim = AerSimulator(
        method="statevector",
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    # Aer needs a save-statevector to actually compute the state.
    tqc_with_save = tqc.copy()
    tqc_with_save.save_statevector()
    # Warm-up: one run not timed.
    sim.run(tqc_with_save).result()
    samples = []
    for _ in range(TIMING_RUNS):
        t0 = time.perf_counter()
        sim.run(tqc_with_save).result()
        samples.append(time.perf_counter() - t0)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "schema_version": 1,
        "n_qubits": N_QUBITS,
        "timing_runs": TIMING_RUNS,
        "basis_gates": BASIS_GATES,
        "workloads": {},
    }
    for name, builder in WORKLOADS.items():
        print(f"[build] {name} …", flush=True)
        qc = builder()
        tqc = transpile_and_export(qc, name)
        gate_count = len(tqc.data)
        print(f"[build] {name}: {gate_count} gates after transpile", flush=True)
        print(f"[time]  {name} (Aer) …", flush=True)
        timing = time_aer(tqc)
        print(
            f"[time]  {name}: median={timing['median_s']*1000:.2f} ms "
            f"stdev={timing['stdev_s']*1000:.2f} ms",
            flush=True,
        )
        results["workloads"][name] = {
            "gate_count_post_transpile": gate_count,
            "qiskit_aer": timing,
        }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[done] results → {RESULTS_PATH}")
```

- [ ] **Step 2: Smoke-run locally (expect minutes on M-series for Grover; that's fine)**

```bash
cd scripts/qiskit-baseline
source .venv/bin/activate
OMP_NUM_THREADS=1 python run.py
```

Expected: prints `[build]` + `[time]` lines for all three workloads; emits `results-qiskit.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/qiskit-baseline/run.py
git commit -m "[meta] Stage 0: time Aer and emit results-qiskit.json"
```

---

## Task 5: Wire the Rust criterion bench (`benches/benches/qiskit_baseline.rs`)

**Files:**
- Modify: `benches/Cargo.toml`
- Create: `benches/benches/qiskit_baseline.rs`

- [ ] **Step 1: Add aleph-parser dep + bench entry to `benches/Cargo.toml`**

Modify `benches/Cargo.toml`:

```toml
[dependencies]
aleph-core    = { path = "../crates/aleph-core" }
aleph-ir      = { path = "../crates/aleph-ir" }
aleph-backend = { path = "../crates/aleph-backend" }
aleph-sv      = { path = "../crates/aleph-sv" }
aleph-parser  = { path = "../crates/aleph-parser" }
smallvec      = { workspace = true }
```

And add to the bottom of the file:

```toml
[[bench]]
name = "qiskit_baseline"
harness = false
```

- [ ] **Step 2: Write the bench**

Create `benches/benches/qiskit_baseline.rs`:

```rust
//! Phase 1, Stage 0: time NaiveSvBackend on the same QASM circuits Qiskit Aer
//! runs (`scripts/qiskit-baseline/circuits/`).  Report numbers feed
//! `docs/perf/phase1-vs-qiskit.md`.
//!
//! `NaiveSvBackend` is the AoS + AVX-512 path post-P1-03 (see ADR 0008) — the
//! canonical fast x86 backend.  Runs scalar on non-AVX-512 hosts; that's fine
//! locally but EPYC is the authoritative measurement target.

use aleph_backend::run;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;

const WORKLOADS: &[&str] = &[
    "qft_n20",
    "grover_n20_iters10",
    "random_brickwall_n20_d20",
];

fn fixture_path(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits")
        .join(format!("{name}.qasm"))
}

fn bench_qiskit_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("qiskit_baseline");
    // Throughput is per-workload; we report wall-time directly. n·2^n keeps
    // bencher.dev's elements/s axis aligned with existing benches/qft.rs.
    group.throughput(Throughput::Elements(20u64 * (1u64 << 20)));

    for &name in WORKLOADS {
        let path = fixture_path(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
        let circuit = aleph_parser::parse(&src)
            .unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e));

        // Headline: NaiveSvBackend (AoS + AVX-512).
        group.bench_with_input(
            BenchmarkId::new("naive_aos_avx512", name),
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

        // Appendix triangulation: SoaSvBackend.
        group.bench_with_input(
            BenchmarkId::new("soa", name),
            &circuit,
            |b, circuit| {
                b.iter_with_setup(
                    || SoaSvBackend::with_seed(0),
                    |mut backend| {
                        let state = run(&mut backend, circuit).unwrap();
                        black_box(state);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_qiskit_baseline);
criterion_main!(benches);
```

- [ ] **Step 3: Verify it compiles (no run)**

```bash
cargo bench --workspace --no-run
```

Expected: clean compile, including a `qiskit_baseline-<hash>` binary in `target/release/deps/`.

- [ ] **Step 4: Quick local smoke-run (M-series will be slow on Grover — that's fine, we just want non-zero output)**

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench qiskit_baseline -- \
  --sample-size 10 --measurement-time 5
```

Expected: criterion reports times for all 6 (workload × backend) combinations. The Grover row may take ~60s on M-series; OK.

- [ ] **Step 5: Lint + fmt**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add benches/Cargo.toml benches/benches/qiskit_baseline.rs
git commit -m "[meta] Stage 0: hermetic Rust bench over QASM fixtures"
```

---

## Task 6: Verify single-thread pinning works on EPYC

**Files:** none (verification only)

This is a manual verification that the agent must perform before producing the headline numbers in Task 8. It's a single SSH session with two checks.

- [ ] **Step 1: SSH and clone**

```bash
ssh root@195.154.249.85
mkdir -p /tmp/aleph-forensics && cd /tmp/aleph-forensics
git clone https://github.com/<owner>/aleph.git || (cd aleph && git fetch)
cd aleph && git checkout <branch>
```

(Adapt `<owner>` and `<branch>` to whatever the user is working on. **Do not** clone into `/home/runner/actions-runner/_work/aleph/aleph/` — that wrecks the runner per memory `[p1-03-merged.md]`.)

- [ ] **Step 2: Verify Qiskit threading pin**

```bash
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  python -c "
import os, threading
from qiskit_aer import AerSimulator
sim = AerSimulator(method='statevector', max_parallel_threads=1)
print('OMP_NUM_THREADS=', os.environ.get('OMP_NUM_THREADS'))
print('threads_at_import=', threading.active_count())
print('aer_options=', sim.options)
"
cat /proc/self/status | grep -E 'Cpus_allowed_list|Threads'
```

Expected: `OMP_NUM_THREADS=1`, `max_parallel_threads=1` in aer options, `Cpus_allowed_list: 0`.

- [ ] **Step 3: Verify Rust toolchain on EPYC**

```bash
cd /tmp/aleph-forensics/aleph
rustc --version    # must be >= 1.89 per CLAUDE.md
RUSTFLAGS="-C target-cpu=native" cargo build --release --workspace
# Sanity: is AVX-512 actually being emitted?
objdump --disassemble target/release/libaleph_sv.rlib 2>/dev/null | \
  grep -E 'vmulpd.*zmm' | head -5
```

Expected: `vmulpd …zmm…` lines present (P1-03's AVX-512 kernel). If absent, the perf numbers later won't reflect the AoS+AVX-512 win.

- [ ] **Step 4: Record findings in a scratch note**

Save observed Python/Qiskit/Aer/numpy versions, CPU model (`lscpu | head -10`), kernel (`uname -a`), and Rust version. These get pasted into the report header in Task 9.

No commit for this task — it's pure verification.

---

## Task 7: Run Qiskit baseline on EPYC

**Files:**
- Modify: `scripts/qiskit-baseline/circuits/*.qasm` (regenerated; may not actually change if Qiskit version pins match)
- Create (locally on EPYC, transferred back): `scripts/qiskit-baseline/results-qiskit.json` (NOT committed; it's in `.gitignore`)

- [ ] **Step 1: Run the harness on EPYC**

```bash
cd /tmp/aleph-forensics/aleph/scripts/qiskit-baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python run.py 2>&1 | tee /tmp/aleph-forensics/aer-run.log
```

Expected runtime: ~5-15 minutes total. QFT is fast; Grover is the bottleneck (~minutes for 10 iters of multi-controlled-Z); random is moderate.

- [ ] **Step 2: Sanity-check `results-qiskit.json`**

```bash
cat results-qiskit.json | python -m json.tool | head -50
```

Verify median values are sane (QFT < Grover < ~Random in wall-clock; stdev < ~10% of median).

- [ ] **Step 3: Pull the JSON + log to your workstation**

```bash
# Locally (not on EPYC):
scp root@195.154.249.85:/tmp/aleph-forensics/aleph/scripts/qiskit-baseline/results-qiskit.json \
  /tmp/results-qiskit-epyc.json
scp root@195.154.249.85:/tmp/aleph-forensics/aer-run.log /tmp/aer-run-epyc.log
```

- [ ] **Step 4: If QASM diffs are non-trivial, commit the regenerated fixtures**

```bash
git -C /Users/ex/GitHub/aleph diff scripts/qiskit-baseline/circuits/
```

If they changed (e.g. transpile output drift), commit:

```bash
git add scripts/qiskit-baseline/circuits/*.qasm
git commit -m "[meta] Stage 0: regenerate QASM fixtures on EPYC"
```

If unchanged, no commit. `results-qiskit.json` itself is `.gitignore`'d — we don't commit raw timing JSON; numbers land in the report.

---

## Task 8: Run aleph criterion bench on EPYC

**Files:** none (results consumed in Task 9)

- [ ] **Step 1: Run criterion against the same QASM fixtures**

```bash
ssh root@195.154.249.85
cd /tmp/aleph-forensics/aleph
RUSTFLAGS="-C target-cpu=native" taskset -c 0 cargo bench \
  --bench qiskit_baseline -- --save-baseline phase1-baseline 2>&1 \
  | tee /tmp/aleph-forensics/aleph-bench.log
```

Expected: criterion prints `qiskit_baseline/naive_aos_avx512/qft_n20`, `…/soa/qft_n20`, etc. Six rows total.

- [ ] **Step 2: Pull the criterion `estimates.json` files**

```bash
# On EPYC:
find target/criterion/qiskit_baseline -name estimates.json -path '*new*' \
  -exec ls -la {} \;
# Locally:
scp -r root@195.154.249.85:/tmp/aleph-forensics/aleph/target/criterion/qiskit_baseline \
  /tmp/criterion-epyc/
scp root@195.154.249.85:/tmp/aleph-forensics/aleph-bench.log /tmp/aleph-bench-epyc.log
```

- [ ] **Step 3: Extract median ns values per (backend, workload)**

The relevant field is `mean.point_estimate` in `estimates.json` (nanoseconds). For each combination collect:
- `naive_aos_avx512/qft_n20`
- `naive_aos_avx512/grover_n20_iters10`
- `naive_aos_avx512/random_brickwall_n20_d20`
- `soa/qft_n20`
- `soa/grover_n20_iters10`
- `soa/random_brickwall_n20_d20`

These six numbers feed the report tables in Task 9.

No commit for this task.

---

## Task 9: Write `docs/perf/phase1-vs-qiskit.md`

**Files:**
- Create: `docs/perf/phase1-vs-qiskit.md`

- [ ] **Step 1: Populate the report**

Create `docs/perf/phase1-vs-qiskit.md` (replace `<…>` placeholders with the actual numbers collected in Tasks 6, 7, 8):

```markdown
# Phase 1 baseline: aleph vs Qiskit Aer (EPYC, single thread)

**Date:** 2026-05-26
**Host:** <EPYC model from `lscpu`>, <cores>, RAM <X> GB, kernel <`uname -r`>
**Toolchain:** Rust <`rustc --version`>, `RUSTFLAGS="-C target-cpu=native"`, `taskset -c 0`
**Qiskit:** <pin from requirements.txt>, **Aer:** <pin>, **Python:** <X.Y.Z>, **numpy:** <pin>
**Pin:** `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `max_parallel_threads=1` (Aer)
**Reproducibility:** `scripts/qiskit-baseline/README.md`

## Headline — `NaiveSvBackend` (AoS + AVX-512, canonical fast x86 path post-P1-03)

| Workload                              |  aleph (ms) |  Aer (ms) | aleph / Aer | ROADMAP § 7 target |
|---------------------------------------|------------:|----------:|------------:|:------------------:|
| `qft_n20`                             |     <a>     |    <q>    |    <a/q>×   |       ≤ 2×         |
| `grover_n20_iters10`                  |     <a>     |    <q>    |    <a/q>×   |       ≤ 2×         |
| `random_brickwall_n20_d20`            |     <a>     |    <q>    |    <a/q>×   |       ≤ 2×         |

Times are medians; stdev in appendix. Lower is faster. Ratios > 1 mean aleph is slower than Aer.

## Appendix — full backend matrix

| Workload                              | `NaiveSvBackend` (ms) | `SoaSvBackend` (ms) | Aer (ms) | gates (post-transpile) |
|---------------------------------------|---------------------:|--------------------:|---------:|-----------------------:|
| `qft_n20`                             |          <a>         |         <s>         |   <q>    |          <g>           |
| `grover_n20_iters10`                  |          <a>         |         <s>         |   <q>    |          <g>           |
| `random_brickwall_n20_d20`            |          <a>         |         <s>         |   <q>    |          <g>           |

stdev (% of median): aleph naive <p>%, soa <p>%, Aer <p>%.

## Interpretation

<One paragraph: where we stand vs ≤ 2× target on each workload. Which is the
weakest. What Stage 1 (SIMD specialisations for Pauli-X, diagonal, 2q kernels)
is expected to close. What Stage 2 (IR-opt gate fusion + cancellation) is
expected to close. Statement that **Phase 1 proceeds to Stage 1 regardless of
the ratio** — these numbers are informational, not a gate.>

## Reproducing this report

See `scripts/qiskit-baseline/README.md`. Exact commands:

```bash
ssh root@195.154.249.85
cd /tmp/aleph-forensics && git clone <repo> && cd aleph && git checkout <branch>
# Aer side
cd scripts/qiskit-baseline && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 taskset -c 0 python run.py
# aleph side
cd ../..
RUSTFLAGS="-C target-cpu=native" taskset -c 0 cargo bench --bench qiskit_baseline -- \
  --save-baseline phase1-baseline
```

## Related work

- P1-03 AVX-512 kernel: PR #80 (`f596e9a`).
- ADR 0008: AoS + AVX-512 beats SoA on Zen 4 (`docs/decisions/0008-aos-avx512-beats-soa-simd.md`).
- Phase 1 plan: `docs/superpowers/plans/2026-05-26-phase1-completion.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/perf/phase1-vs-qiskit.md
git commit -m "[meta] Stage 0: Qiskit Aer baseline report on EPYC"
```

---

## Task 10: BACKLOG / ROADMAP touch-up

**Files:**
- Modify: `BACKLOG.md` (if it has a Stage-0 line item to tick)
- Modify: `docs/superpowers/plans/2026-05-26-phase1-completion.md` (mark Stage 0 done)

- [ ] **Step 1: Check whether `BACKLOG.md` has a Stage 0 / `[meta]` Phase-1-baseline entry**

```bash
grep -nE 'qiskit|baseline|stage 0' BACKLOG.md docs/superpowers/plans/2026-05-26-phase1-completion.md
```

If `BACKLOG.md` doesn't list this `[meta]` task, that's expected — Stage 0 was born in the phase-1 plan, not the backlog. No edit needed.

- [ ] **Step 2: Mark Stage 0 complete in the phase-1 plan**

Edit `docs/superpowers/plans/2026-05-26-phase1-completion.md`, at the top of the `## Stage 0` section, add (above the `**Goal**` line):

```markdown
**Status:** Done as of 2026-05-26. Report at `docs/perf/phase1-vs-qiskit.md`. Proceeding to Stage 1.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-26-phase1-completion.md
git commit -m "[meta] Stage 0: mark complete in phase 1 plan"
```

---

## Task 11: Fix the spec's moot "QASM emitter follow-up" line

**Files:**
- Modify: `docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md`

The spec promised to file a follow-up issue for "QASM3 emitter for aleph-ir". This is moot — `aleph_parser::emit` already exists. Strike the promise so the spec is accurate.

- [ ] **Step 1: Find and remove the two lines that mention filing the follow-up**

Locations to fix (search the spec):

In section 3 ("Deliverables"), remove:

> A separate GitHub issue is filed (not implemented): **"QASM3 emitter for aleph-ir"** — round-trip QASM round-trip support, deferred to Phase 2 prep.

In section 7 ("Acceptance criteria"), remove:

> - [ ] Separate GitHub issue filed: "QASM3 emitter for aleph-ir" (Phase 2 prep).

In section 4.1, replace:

> Genuinely useful long-term (round-trip serialisation, interop with Cirq / Stim / IBM hardware), but doesn't pay off in Stage 0 and adds days of work to a half-day ticket.

with:

> Genuinely useful long-term (round-trip serialisation, interop with Cirq / Stim / IBM hardware). `aleph_parser::emit` already provides this (P0-08 Task 15), but it's still not the right direction for Stage 0: it forces aleph to be the single source of truth, including building Grover in aleph-ir, which adds days to a half-day ticket.

In section 2 ("Non-goals"), replace:

> **Not** the implementation of a QASM3 **emitter** for aleph-ir. That stays Qiskit-side. A separate `[infra]` issue is filed for the round-trip emitter.

with:

> **Not** the implementation or extension of aleph's existing QASM3 emitter (`aleph_parser::emit`, added in P0-08). Circuit construction stays Qiskit-side for this ticket.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md
git commit -m "[meta] Stage 0: spec fixup — aleph_parser::emit already exists"
```

---

## Task 12: Open the PR

**Files:** none (PR creation)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin <branch>
```

- [ ] **Step 2: Open the PR via `gh`**

```bash
gh pr create --title "[meta] Qiskit Aer baseline on EPYC (Phase 1 Stage 0)" \
  --body "$(cat <<'EOF'
## Summary

- First task of the Phase 1 completion plan — `docs/superpowers/plans/2026-05-26-phase1-completion.md` § Stage 0.
- Single-thread, same-circuit baseline comparison between `NaiveSvBackend` (AoS + AVX-512 post-P1-03) and Qiskit Aer on EPYC across QFT-20, Grover-20 (10 iters), and random-brickwall-20.
- **Informational only — does not gate Phase 1.** Proceeding to Stage 1 regardless.

## What's in the PR

- `scripts/qiskit-baseline/` — Python harness (`run.py`), `requirements.txt`, `README.md`, three QASM3 fixture files.
- `benches/benches/qiskit_baseline.rs` — hermetic criterion bench reading the same `.qasm` files; runs `NaiveSvBackend` (headline) + `SoaSvBackend` (appendix).
- `crates/aleph-parser/tests/qiskit_baseline_fixtures.rs` — regression test that every fixture parses.
- `docs/perf/phase1-vs-qiskit.md` — populated report with EPYC numbers, host metadata, version pins, interpretation paragraph.
- Spec fixup: `aleph_parser::emit` already exists (P0-08 Task 15), so the spec's promise to file a follow-up "QASM emitter" issue was struck.

## Headline numbers

(see `docs/perf/phase1-vs-qiskit.md` for full table — paste the headline 3-row table here when filing the PR.)

## Test plan

- [x] `cargo test --workspace` green (incl. the new parser integration test)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [x] `cargo fmt --check` clean
- [x] `cargo bench --workspace --no-run` compiles
- [x] EPYC: `python run.py` produces `results-qiskit.json`
- [x] EPYC: `cargo bench --bench qiskit_baseline` produces criterion estimates
- [x] EPYC: `objdump` confirms AVX-512 emission in `libaleph_sv.rlib`

Closes <issue-number-once-filed>.   <!-- If this `[meta]` task has no GH issue yet, drop the line. CLAUDE.md says to use #issue-number not #PR-number. -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Self-review the diff once more, then mark Stage 0 closed**

Open the PR in browser, read the diff end-to-end, fix anything obviously wrong. Tag for code review per the established per-ticket workflow.

---

## Self-review notes (executing agent: read before starting Task 1)

1. **Spec → plan coverage** — each spec section is covered:
   - § 1 Goal → Tasks 2, 4, 8 produce the three workload measurements.
   - § 3 Deliverables → Tasks 1-2, 4-5, 7, 9 produce every listed file.
   - § 5 Methodology → Tasks 2, 4 (Python side), Task 5 (Rust side).
   - § 6 Report structure → Task 9.
   - § 7 Acceptance → Tasks 3 (parser fixture test), 7-8 (EPYC runs), 9 (report).
   - § 8 Risks → Task 6 verifies pinning; Task 3 catches parser gaps.
   - § 10 Workflow → Task 12 opens the PR for code review.

2. **No CI bench job, no `crates/` changes** — confirmed across all 12 tasks.

3. **The Grover construction detail** — Task 2 uses `qc.mcx(controls, target)` + H sandwich for the multi-controlled Z. Qiskit's transpile at level 0 decomposes this into the basis (h/cx/ccx primarily). The resulting gate count is logged so if it explodes (> 100k gates) the agent drops to 5 iters per spec § 8 risk row.

4. **No placeholders** — every code block is complete and runnable. The only `<…>` placeholders are intentional report-template fields filled in at Task 9 Step 1 from the EPYC numbers collected in Tasks 7-8.

5. **Branch name convention** — `pNN-NN-short-description` doesn't fit a meta ticket. Suggest `meta-phase1-stage0-qiskit` for the branch.
