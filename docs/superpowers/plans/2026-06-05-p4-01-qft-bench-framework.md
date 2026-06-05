# P4-01 QFT Benchmark + Phase-4 Bench/Report Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the shared Phase-4 benchmark+report framework (committed QASM corpus → aleph criterion + Aer timings → unified JSON → auto-generated `docs/perf/phase4.md`) with QFT as the first consumer, running to n=30 on the state vector.

**Architecture:** Reuse the existing `scripts/qiskit-baseline/run.py` (already generates+exports+times) by making circuit sizes per-family so QFT gets {10,15,20,25,30}. The aleph side runs the *same* committed corpus QASM (not the builder) via `run_optimized`. Two new Python tools — a criterion extractor and a deterministic report generator — merge both result sources into the report. A per-instance configurable qubit cap on `NaiveSvBackend` unblocks n=30 without weakening the default 28-qubit guardrail.

**Tech Stack:** Rust 2021 (criterion, aleph-sv/backend/parser), Python 3.12 (Qiskit Aer harness, report tooling), markdown report.

**Spec:** `docs/superpowers/specs/2026-06-05-p4-01-qft-bench-framework-design.md`

## File structure

- Modify `crates/aleph-sv/src/backend.rs` — add `qubit_cap` field + `with_qubit_cap`; `allocate` honors it.
- Modify `scripts/qiskit-baseline/run.py` — per-family `FAMILY_SIZES`; QFT→{10,15,20,25,30}; `timing_runs_for` covers n=10/30.
- Create `scripts/qiskit-baseline/circuits/qft_n10.qasm`, `qft_n30.qasm` — generated, committed.
- Create `benches/benches/phase4_qft.rs` — corpus-QASM QFT bench, feature-gated for n≥28.
- Modify `benches/src/lib.rs` — add `qft_inverse_circuit(n)` + inverse-QFT round-trip test.
- Create `scripts/bench-report/extract_criterion.py` — criterion estimates.json → unified aleph JSON.
- Create `scripts/bench-report/report.py` — merge → `docs/perf/phase4.md`.
- Create `scripts/bench-report/test_report.py` — golden-file unit test.
- Create `scripts/bench-report/README.md` — end-to-end pipeline instructions.
- Create (after EPYC run) `docs/perf/data/phase4-aleph.json`, `docs/perf/data/phase4-meta.json`, `docs/perf/phase4.md`.
- Modify `BACKLOG.md` — tick P4-01 ACs.

---

## Task 1: Configurable qubit cap on `NaiveSvBackend`

**Files:**
- Modify: `crates/aleph-sv/src/backend.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` in `crates/aleph-sv/src/backend.rs`:

```rust
#[test]
fn default_cap_still_rejects_above_28() {
    let mut b = NaiveSvBackend::new();
    let err = b.allocate(MAX_NAIVE_QUBITS + 1).unwrap_err();
    assert!(matches!(err, BackendError::TooManyQubits { .. }));
}

#[test]
fn raised_cap_allows_more_qubits() {
    // Don't actually allocate 2^30 in a unit test; use a small raised cap and a
    // small n to prove the cap field gates allocate(), not memory.
    let mut b = NaiveSvBackend::new().with_qubit_cap(4);
    assert!(b.allocate(4).is_ok());
    let err = b.allocate(5).unwrap_err();
    assert!(matches!(err, BackendError::TooManyQubits { requested: 5, limit: 4 }));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-sv backend::tests::raised_cap_allows_more_qubits`
Expected: FAIL — `with_qubit_cap` not found.

- [ ] **Step 3: Add the field, constructor wiring, and builder**

In `crates/aleph-sv/src/backend.rs`:

1. Add a field to the struct (after `rng`):

```rust
pub struct NaiveSvBackend {
    pub(crate) rng: StdRng,
    /// Per-instance max qubit count for `allocate`. Defaults to
    /// [`MAX_NAIVE_QUBITS`]; raised only for large-memory benchmarks on hosts
    /// that can hold the state vector (e.g. n=30 ⇒ 16 GiB on the EPYC box).
    qubit_cap: u32,
}
```

2. Set the default in BOTH `new()` and `with_seed()` (add `qubit_cap: MAX_NAIVE_QUBITS,` to each struct literal).

3. Add the builder after `with_seed`:

```rust
/// Override the qubit cap (default [`MAX_NAIVE_QUBITS`] = 28). Use only on a
/// host with enough RAM for `2^cap * 16` bytes — n=30 ⇒ 16 GiB.
pub fn with_qubit_cap(mut self, cap: u32) -> Self {
    self.qubit_cap = cap;
    self
}
```

4. Change `allocate` to use the instance cap:

```rust
    fn allocate(&mut self, num_qubits: u32) -> Result<Self::State, BackendError> {
        if num_qubits > self.qubit_cap {
            return Err(BackendError::TooManyQubits {
                requested: num_qubits,
                limit: self.qubit_cap,
            });
        }
```

(Leave the rest of `allocate` unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aleph-sv backend::tests`
Expected: PASS (including the existing `allocate_rejects_too_many_qubits`).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/backend.rs
git commit -m "[P4-01] NaiveSvBackend: configurable qubit cap (default 28)"
```

---

## Task 2: Per-family sizes in run.py + generate QFT n=10/30 corpus

**Files:**
- Modify: `scripts/qiskit-baseline/run.py`
- Create (generated): `scripts/qiskit-baseline/circuits/qft_n10.qasm`, `qft_n30.qasm`

> Requires the qiskit-baseline venv. Per `scripts/qiskit-baseline/README.md`, it
> is at `scripts/qiskit-baseline/.venv` (Python 3.12 + qiskit/aer). Activate it
> for the generate step.

- [ ] **Step 1: Replace the global size list with a per-family map**

In `scripts/qiskit-baseline/run.py`, replace:

```python
N_QUBITS_LIST = [15, 20, 22, 25]
```

with:

```python
# Per-family qubit sizes. QFT extends to the P4-01 matrix {10,15,20,25,30}
# (n=30 is the AC ceiling, measured on the EPYC box); other families keep the
# Stage-0 sizes. A global list would regenerate every family and make
# Grover/random at n=30 intractable.
FAMILY_SIZES = {
    "ghz": [15, 20, 22, 25],
    "qft": [10, 15, 20, 25, 30],
    "grover": [15, 20, 22, 25],
    "random_brickwall": [15, 20, 22, 25],
}
# Union for the results header (sorted, de-duplicated).
N_QUBITS_LIST = sorted({n for sizes in FAMILY_SIZES.values() for n in sizes})
```

- [ ] **Step 2: Make `all_workloads()` use per-family sizes**

Replace the body of `all_workloads()`:

```python
def all_workloads() -> list[tuple[str, str, int]]:
    """(name, family, n) for the full matrix, families in stable order."""
    return [
        (workload_name(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in FAMILY_SIZES[fam]
    ]
```

- [ ] **Step 3: Extend `timing_runs_for` for n=10 and n=30**

Replace `timing_runs_for`:

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

- [ ] **Step 4: Generate and commit the new QFT corpus files**

Run (from repo root, with the venv active):

```bash
cd scripts/qiskit-baseline
. .venv/bin/activate 2>/dev/null || true
python run.py --gen-only --workloads qft_n10,qft_n30
```

Expected stdout includes `[build] qft_n10: <k> gates after transpile` and
`[build] qft_n30: ...`. This writes `circuits/qft_n10.qasm` and
`circuits/qft_n30.qasm` (and re-exports any other QFT sizes touched — verify the
already-committed `qft_n{15,20,25}.qasm` are byte-identical with `git diff`; if
they changed, that is corpus drift to investigate, not auto-accept).

Verify the new files parse with aleph:

```bash
cd ../..
cargo run -q --bin aleph -- run scripts/qiskit-baseline/circuits/qft_n10.qasm --shots 1 --backend statevector >/dev/null && echo "qft_n10 parses+runs"
```

Expected: `qft_n10 parses+runs`.

- [ ] **Step 5: Commit**

```bash
git add scripts/qiskit-baseline/run.py scripts/qiskit-baseline/circuits/qft_n10.qasm scripts/qiskit-baseline/circuits/qft_n30.qasm
git commit -m "[P4-01] run.py: per-family sizes; generate QFT n=10/30 corpus"
```

---

## Task 3: aleph corpus-QASM QFT benchmark

**Files:**
- Create: `benches/benches/phase4_qft.rs`
- Modify: `benches/Cargo.toml` (register the bench)

> Pattern reference: `benches/benches/tier1_scaling.rs` (reads fixture QASM via
> `fixture_path`, runs `aleph_backend::run` / `run_optimized`, feature-gated).

- [ ] **Step 1: Write the bench**

Create `benches/benches/phase4_qft.rs`:

```rust
//! P4-01 QFT benchmark over the committed corpus QASM (the SAME files Aer
//! times), run through the canonical optimized state-vector path. This is the
//! aleph half of the Phase-4 QFT report row.
//!
//! n=10/15/20/25 run anywhere; n=28/30 allocate ≥4/16 GiB and are gated behind
//! the `scaling-bench` feature so default `cargo bench --workspace` / CI skip
//! them. Measure on the EPYC box:
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
//!       --bench phase4_qft --features scaling-bench
//!
//! The corpus QASM lives at `scripts/qiskit-baseline/circuits/qft_n{N}.qasm`.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

/// QFT sizes always benched (small enough for any host / CI bench job).
const SMALL_N: &[u32] = &[10, 15, 20, 25];
/// Large sizes gated behind `scaling-bench` (≥4 GiB state vector).
#[cfg(feature = "scaling-bench")]
const LARGE_N: &[u32] = &[28, 30];

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/qft_n{n}.qasm"))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read qft_n{n}.qasm: {e}"));
    let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse qft_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            // Raised cap so n=30 is permitted on a host with enough RAM; small
            // n are unaffected (cap only gates allocate()).
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_qft");
    // n=25 already allocates 512 MiB; keep sample counts modest.
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

criterion_group!(benches, qft);
criterion_main!(benches);
```

- [ ] **Step 2: Register the bench in `benches/Cargo.toml`**

Find the existing `[[bench]]` blocks (e.g. `tier1_scaling`) and add, matching
their style (they use `harness = false` and the `scaling-bench` feature):

```toml
[[bench]]
name = "phase4_qft"
harness = false
```

Confirm the `scaling-bench` feature already exists in `[features]` (it gates
`tier1_scaling`/`qft_scaling`); reuse it, do not add a new one.

- [ ] **Step 3: Verify it compiles and runs at small n locally**

Run: `cargo bench -p aleph-benches --bench phase4_qft -- --quick qft 10`
Expected: builds and runs n=10/15/20/25 (no `scaling-bench` ⇒ no n=28/30). If
`--quick` is unsupported by the installed criterion, use
`cargo bench -p aleph-benches --bench phase4_qft -- --sample-size 10 --measurement-time 1`.

Also confirm the feature-gated path compiles:
Run: `cargo build -p aleph-benches --bench phase4_qft --features scaling-bench`
Expected: clean build (LARGE_N referenced only under the feature).

- [ ] **Step 4: Commit**

```bash
git add benches/benches/phase4_qft.rs benches/Cargo.toml
git commit -m "[P4-01] aleph QFT corpus bench (phase4_qft), feature-gated n>=28"
```

---

## Task 4: inverse-QFT round-trip correctness test

**Files:**
- Modify: `benches/src/lib.rs` (add `qft_inverse_circuit` + tests)

> **Verified facts (no guessing needed):** `qft_circuit(n)` (benches/src/lib.rs:43)
> emits `c.h(j)` and `GateInstance::controlled(Gate::Phase(θ), smallvec![j], smallvec![k])`
> (controlled-Phase, target `j`, external control `k`, θ = π/2^(k−j)).
> `aleph_core::Gate::inverse()` EXISTS and covers exactly these: `H → H`
> (self-inverse) and `Phase(p) → Phase(negate(p))`. So the inverse of any
> instruction is: clone it (preserving `qubits` AND `controls`), replace its
> `.gate` with `.gate.inverse()`, and emit the instructions in reverse order.
> `NaiveSvBackend`'s state exposes `.amplitudes() -> &[Complex]` (as in
> `benches/src/bin/oneshot.rs`).

- [ ] **Step 1: Write the failing tests**

Add to `benches/src/lib.rs` (inside or alongside its `#[cfg(test)] mod tests`;
create the module if absent):

```rust
#[cfg(test)]
mod qft_roundtrip_tests {
    use super::*;
    use aleph_backend::run;
    use aleph_core::Complex;
    use aleph_sv::NaiveSvBackend;

    /// Apply `circuit` to a freshly allocated |0…0⟩ and return amplitudes.
    fn run_amps(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
        let mut b = NaiveSvBackend::with_seed(7);
        let state = run(&mut b, circuit).expect("run");
        // `HasAmplitudes`/state amplitudes accessor — match the crate's API.
        state.amplitudes().to_vec()
    }

    #[test]
    fn qft_then_inverse_is_identity_on_zero_state() {
        let n = 6;
        let mut c = qft_circuit(n);
        // Append the inverse so the combined circuit should be the identity.
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let amps = run_amps(&c);
        assert!((amps[0].re - 1.0).abs() < 1e-10, "amp[0] should be 1");
        assert!(amps[0].im.abs() < 1e-10);
        for (k, a) in amps.iter().enumerate().skip(1) {
            assert!(a.norm() < 1e-10, "amp[{k}] should be ~0");
        }
    }

    #[test]
    fn qft_then_inverse_is_identity_on_generic_state() {
        // Per the P1-13 lesson, a |0…0⟩-only check misses bugs. Prep a generic
        // state with a layer of H + T-like rotations, snapshot it, then apply
        // QFT∘QFT⁻¹ and assert the state is unchanged.
        let n = 5;
        let prep = generic_prep_circuit(n); // defined below
        let before = run_amps(&prep);

        let mut c = prep.clone();
        for inst in qft_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let after = run_amps(&c);

        assert_eq!(before.len(), after.len());
        for (k, (x, y)) in before.iter().zip(after.iter()).enumerate() {
            assert!((x - y).norm() < 1e-10, "amp[{k}] changed: {x:?} vs {y:?}");
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-benches qft_roundtrip_tests`
Expected: FAIL — `qft_inverse_circuit` and `generic_prep_circuit` not found.

- [ ] **Step 3: Implement `qft_inverse_circuit` and the prep helper**

Add to `benches/src/lib.rs` (public, next to `qft_circuit`). The inverse emits
`qft_circuit`'s instructions in reverse order, each with its `.gate` replaced by
`.gate.inverse()` (which is `H→H` and `Phase(θ)→Phase(−θ)`), preserving qubits
and controls:

```rust
/// Inverse QFT on `n` qubits: `qft_circuit(n)`'s instructions in reverse order,
/// each gate replaced by its inverse (`H→H`, `Phase(θ)→Phase(−θ)`), preserving
/// the control/target qubits. `qft_circuit(n)` followed by `qft_inverse_circuit(n)`
/// is the identity.
pub fn qft_inverse_circuit(n: u32) -> aleph_ir::Circuit {
    let fwd = qft_circuit(n);
    let mut inv = aleph_ir::Circuit::new(n, 0);
    for inst in fwd.instructions().iter().rev() {
        match inst {
            aleph_ir::Instruction::Gate(g) => {
                let mut g2 = g.clone(); // preserves qubits AND controls
                g2.gate = g.gate.inverse();
                inv.add_instruction(aleph_ir::Instruction::Gate(g2)).unwrap();
            }
            other => inv.add_instruction(other.clone()).unwrap(),
        }
    }
    inv
}
```

Add `generic_prep_circuit`:

```rust
/// A small non-trivial state prep: H on every qubit, then a phase rotation on
/// each, producing a generic (non-basis) state for round-trip testing.
fn generic_prep_circuit(n: u32) -> aleph_ir::Circuit {
    use aleph_core::{Gate, GateInstance, Param};
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(GateInstance::new(Gate::H, vec![q])).unwrap();
    }
    for q in 0..n {
        c.add_gate(GateInstance::new(Gate::Rz(Param::Concrete(0.1 * (q as f64 + 1.0))), vec![q]))
            .unwrap();
    }
    c
}
```

> If `Gate::inverse()` does not exist or does not cover QFT's phase gate, write
> `invert_gate` to match `qft_circuit`'s exact gate variant by hand (negate the
> `Param::Concrete` angle; `H` maps to `H`). Do not guess the variant — use what
> Step 1 showed.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aleph-benches qft_roundtrip_tests`
Expected: PASS (both tests). If the generic-state test fails while the zero-state
passes, the inverse gate order is wrong (reverse-order is load-bearing) — fix the
ordering, do not loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add benches/src/lib.rs
git commit -m "[P4-01] qft_inverse_circuit + QFT∘QFT⁻¹ round-trip tests (zero + generic state)"
```

---

## Task 5: criterion → unified-JSON extractor

**Files:**
- Create: `scripts/bench-report/extract_criterion.py`
- Create: `scripts/bench-report/test_extract.py`

> Criterion writes `target/criterion/<group>/<param>/new/estimates.json` with
> top-level keys `mean`, `median`, `std_dev`, each `{"point_estimate": <ns>, …}`.
> The extractor reads `median.point_estimate` (ns) and `std_dev.point_estimate`.

- [ ] **Step 1: Write the failing test with a sample estimates.json**

Create `scripts/bench-report/test_extract.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _write_estimates(root: Path, group: str, param: str, median_ns: float, std_ns: float):
    d = root / "criterion" / group / param / "new"
    d.mkdir(parents=True, exist_ok=True)
    (d / "estimates.json").write_text(json.dumps({
        "median": {"point_estimate": median_ns},
        "std_dev": {"point_estimate": std_ns},
    }))


def test_extract_qft(tmp_path):
    _write_estimates(tmp_path, "phase4_qft", "10", 1_000_000.0, 5_000.0)   # 1.0 ms
    _write_estimates(tmp_path, "phase4_qft", "25", 500_000_000.0, 250_000.0)  # 500 ms
    out = tmp_path / "phase4-aleph.json"
    subprocess.run(
        [sys.executable, str(HERE / "extract_criterion.py"),
         "--criterion-root", str(tmp_path / "criterion"),
         "--group", "phase4_qft", "--family", "qft",
         "--out", str(out)],
        check=True,
    )
    data = json.loads(out.read_text())
    w = data["workloads"]
    assert w["qft_n10"]["aleph_ms_median"] == 1.0
    assert w["qft_n10"]["n"] == 10
    assert w["qft_n10"]["family"] == "qft"
    assert abs(w["qft_n25"]["aleph_ms_median"] - 500.0) < 1e-9
    assert abs(w["qft_n25"]["aleph_rsd"] - (250_000.0 / 500_000_000.0)) < 1e-12


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts/bench-report/test_extract.py -v`
Expected: FAIL — `extract_criterion.py` does not exist.

- [ ] **Step 3: Implement the extractor**

Create `scripts/bench-report/extract_criterion.py`:

```python
#!/usr/bin/env python3
"""Extract criterion medians into the unified Phase-4 aleph results JSON.

Reads target/criterion/<group>/<param>/new/estimates.json for each parameter
(the qubit count) and writes {workloads: {<family>_n<n>: {n, family,
aleph_ms_median, aleph_rsd}}}. Deterministic; no network.
"""
import argparse
import json
from pathlib import Path


def extract(criterion_root: Path, group: str, family: str) -> dict:
    workloads = {}
    group_dir = criterion_root / group
    for param_dir in sorted(group_dir.iterdir()):
        if not param_dir.is_dir() or param_dir.name == "report":
            continue
        est = param_dir / "new" / "estimates.json"
        if not est.exists():
            continue
        e = json.loads(est.read_text())
        median_ns = float(e["median"]["point_estimate"])
        std_ns = float(e["std_dev"]["point_estimate"])
        n = int(param_dir.name)
        workloads[f"{family}_n{n}"] = {
            "n": n,
            "family": family,
            "aleph_ms_median": median_ns / 1e6,
            "aleph_rsd": (std_ns / median_ns) if median_ns else 0.0,
        }
    return {"schema_version": 1, "workloads": workloads}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion-root", required=True, type=Path)
    ap.add_argument("--group", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = extract(args.criterion_root, args.group, args.family)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[extract] {len(data['workloads'])} workloads -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest scripts/bench-report/test_extract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench-report/extract_criterion.py scripts/bench-report/test_extract.py
git commit -m "[P4-01] criterion->unified-JSON extractor + test"
```

---

## Task 6: deterministic report generator

**Files:**
- Create: `scripts/bench-report/report.py`
- Create: `scripts/bench-report/test_report.py`
- Create: `scripts/bench-report/testdata/` (golden inputs/output)

- [ ] **Step 1: Write the golden-file test**

Create `scripts/bench-report/testdata/aleph.json`:

```json
{"schema_version": 1, "workloads": {
  "qft_n10": {"n": 10, "family": "qft", "aleph_ms_median": 1.0, "aleph_rsd": 0.01},
  "qft_n25": {"n": 25, "family": "qft", "aleph_ms_median": 500.0, "aleph_rsd": 0.005}
}}
```

Create `scripts/bench-report/testdata/aer.json` (a minimal results-qiskit.json):

```json
{"schema_version": 2, "workloads": {
  "qft_n10": {"n": 10, "family": "qft", "gate_count_post_transpile": 55,
              "qiskit_aer": {"median_s": 0.0008, "stdev_s": 0.00001}},
  "qft_n25": {"n": 25, "family": "qft", "gate_count_post_transpile": 350,
              "qiskit_aer": {"median_s": 0.46, "stdev_s": 0.001}}
}}
```

Create `scripts/bench-report/testdata/meta.json`:

```json
{"date": "2026-06-05", "host": "EPYC 8124P (test)", "toolchain": "rust 1.95",
 "qiskit": "1.2.4 / Aer 0.15.1", "notes": "golden fixture"}
```

Create `scripts/bench-report/test_report.py`:

```python
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TD = HERE / "testdata"


def test_report_matches_golden(tmp_path):
    out = tmp_path / "phase4.md"
    subprocess.run(
        [sys.executable, str(HERE / "report.py"),
         "--aleph", str(TD / "aleph.json"),
         "--aer", str(TD / "aer.json"),
         "--meta", str(TD / "meta.json"),
         "--out", str(out)],
        check=True,
    )
    got = out.read_text()
    golden = (TD / "phase4.golden.md").read_text()
    assert got == golden, f"report drift:\n{got}"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts/bench-report/test_report.py -v`
Expected: FAIL — `report.py` missing.

- [ ] **Step 3: Implement the report generator**

Create `scripts/bench-report/report.py`:

```python
#!/usr/bin/env python3
"""Merge the unified aleph results JSON and the Aer results-qiskit.json into a
deterministic Phase-4 markdown report. Pure: same inputs -> identical output.
"""
import argparse
import json
from pathlib import Path

FAMILY_TITLES = {
    "qft": "QFT",
    "grover": "Grover",
    "ghz": "GHZ",
    "random_brickwall": "Random brickwall",
    "qpe": "QPE",
    "vqe": "VQE",
    "qaoa": "QAOA",
    "surface_code": "Surface code",
}


def _rows(aleph: dict, aer: dict):
    rows = []
    for name, a in aleph["workloads"].items():
        b = aer["workloads"].get(name)
        if b is None:
            continue
        aleph_ms = a["aleph_ms_median"]
        aer_ms = b["qiskit_aer"]["median_s"] * 1000.0
        aer_rsd = b["qiskit_aer"]["stdev_s"] / b["qiskit_aer"]["median_s"] if b["qiskit_aer"]["median_s"] else 0.0
        rows.append({
            "name": name,
            "family": a["family"],
            "n": a["n"],
            "gates": b.get("gate_count_post_transpile"),
            "aleph_ms": aleph_ms,
            "aer_ms": aer_ms,
            "ratio": aleph_ms / aer_ms if aer_ms else float("nan"),
            "aleph_rsd": a["aleph_rsd"],
            "aer_rsd": aer_rsd,
        })
    rows.sort(key=lambda r: (r["family"], r["n"]))
    return rows


def render(aleph: dict, aer: dict, meta: dict) -> str:
    rows = _rows(aleph, aer)
    out = []
    out.append("# Phase 4 — algorithm benchmarks vs Qiskit Aer\n")
    out.append("> Auto-generated by `scripts/bench-report/report.py`. Do not edit by hand.\n")
    out.append(f"**Date:** {meta['date']}  ")
    out.append(f"**Host:** {meta['host']}  ")
    out.append(f"**Toolchain:** {meta['toolchain']}  ")
    out.append(f"**Qiskit:** {meta['qiskit']}\n")
    if meta.get("notes"):
        out.append(f"_{meta['notes']}_\n")
    families = sorted({r["family"] for r in rows})
    for fam in families:
        frows = [r for r in rows if r["family"] == fam]
        out.append(f"## {FAMILY_TITLES.get(fam, fam)}\n")
        out.append("| workload | n | gates | aleph (ms) | Aer (ms) | aleph / Aer | aleph RSD | Aer RSD |")
        out.append("|----------|--:|------:|-----------:|---------:|------------:|----------:|--------:|")
        for r in frows:
            out.append(
                f"| `{r['name']}` | {r['n']} | {r['gates']} | "
                f"{r['aleph_ms']:.2f} | {r['aer_ms']:.2f} | {r['ratio']:.2f}× | "
                f"{r['aleph_rsd']*100:.2f}% | {r['aer_rsd']*100:.2f}% |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aleph", required=True, type=Path)
    ap.add_argument("--aer", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    aleph = json.loads(args.aleph.read_text())
    aer = json.loads(args.aer.read_text())
    meta = json.loads(args.meta.read_text())
    args.out.write_text(render(aleph, aer, meta))
    print(f"[report] -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the golden file, then verify the test passes**

Generate the golden once (this becomes the expected output — inspect it for
sanity before committing):

```bash
python scripts/bench-report/report.py \
  --aleph scripts/bench-report/testdata/aleph.json \
  --aer scripts/bench-report/testdata/aer.json \
  --meta scripts/bench-report/testdata/meta.json \
  --out scripts/bench-report/testdata/phase4.golden.md
```

Inspect `phase4.golden.md`: it should have a QFT table with two rows (n=10, 25),
ratios `1.25×` (1.0/0.8) and `1.09×` (500/460). Then:

Run: `python -m pytest scripts/bench-report/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench-report/report.py scripts/bench-report/test_report.py scripts/bench-report/testdata/
git commit -m "[P4-01] deterministic Phase-4 report generator + golden test"
```

---

## Task 7: pipeline README + EPYC measurement → committed report

**Files:**
- Create: `scripts/bench-report/README.md`
- Create: `docs/perf/data/phase4-meta.json`, `docs/perf/data/phase4-aleph.json`
- Create: `docs/perf/phase4.md`

> This task produces the real numbers. The aleph n=28/30 benches and the Aer
> n=25/30 timings must run on the **EPYC box** (`ssh root@195.154.249.85`).
> **Verify the box is idle first** (`uptime` load ≈ 0; `pgrep -af "cargo bench|bencher"`)
> per the CLAUDE.md idle rule — and do not push to `main`/`benches/**` during the
> measurement (the self-hosted Bench job races on the same runner).

- [ ] **Step 1: Write the pipeline README**

Create `scripts/bench-report/README.md` documenting the four steps end-to-end:

```markdown
# Phase-4 benchmark pipeline

Produces `docs/perf/phase4.md` from two measurement sources over the shared
corpus `scripts/qiskit-baseline/circuits/*.qasm`.

## 1. Aer baseline (Python, EPYC)
    cd scripts/qiskit-baseline && . .venv/bin/activate
    python run.py --workloads qft_n10,qft_n15,qft_n20,qft_n25,qft_n30
    # writes results-qiskit.json (QFT rows)

## 2. aleph criterion (Rust, EPYC, idle box)
    RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches \
        --bench phase4_qft --features scaling-bench
    # writes target/criterion/phase4_qft/<n>/new/estimates.json

## 3. Extract aleph medians
    python scripts/bench-report/extract_criterion.py \
        --criterion-root target/criterion --group phase4_qft --family qft \
        --out docs/perf/data/phase4-aleph.json

## 4. Render the report
    python scripts/bench-report/report.py \
        --aleph docs/perf/data/phase4-aleph.json \
        --aer   scripts/qiskit-baseline/results-qiskit.json \
        --meta  docs/perf/data/phase4-meta.json \
        --out   docs/perf/phase4.md

Adding a family (P4-02..07): generate its corpus + run.py timing + an aleph
bench, then re-run steps 1–4. No tooling changes.
```

- [ ] **Step 2: Run steps 1–3 on EPYC; capture the two JSONs**

Follow the README on the EPYC box (idle-checked). Copy back
`scripts/qiskit-baseline/results-qiskit.json` (QFT rows) and produce
`docs/perf/data/phase4-aleph.json`. Record the environment in
`docs/perf/data/phase4-meta.json`:

```json
{
  "date": "2026-06-05",
  "host": "AMD EPYC 8124P (16c/32t, Zen 4c), 123 GiB, Ubuntu",
  "toolchain": "rust <fill from `rustc --version`>, RUSTFLAGS=-C target-cpu=native",
  "qiskit": "<fill from pip show qiskit / qiskit-aer>",
  "notes": "single-thread Aer pin (OMP/MKL/OPENBLAS=1, taskset -c 0); criterion sample_size=10; n=30 SV = 16 GiB"
}
```

- [ ] **Step 3: Render and sanity-check the report**

Run step 4 from the README. Open `docs/perf/phase4.md`: confirm a QFT table with
rows n=10/15/20/25/30, plausible ratios, and that n=30 actually has a number
(closing the AC). If n=30 aleph is missing, the `scaling-bench` feature or the
raised cap was not used — fix and re-measure.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench-report/README.md docs/perf/data/phase4-meta.json docs/perf/data/phase4-aleph.json docs/perf/phase4.md
git commit -m "[P4-01] Phase-4 QFT report: EPYC numbers to n=30 + pipeline README"
```

---

## Task 8: tick BACKLOG ACs + full local gate

**Files:**
- Modify: `BACKLOG.md`

- [ ] **Step 1: Tick the P4-01 acceptance criteria**

In `BACKLOG.md` under `### [P4-01]`, change the three AC checkboxes to `[x]`:

```
- [x] QFT runs to 30 qubits on state vector
- [x] Results match Qiskit
- [x] Benchmark report row
```

- [ ] **Step 2: Run the full gate**

Run:
```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python -m pytest scripts/bench-report -v
```
Expected: all green. `cargo fmt --all` (NOT `-p`) if formatting drifts.

- [ ] **Step 3: Commit**

```bash
git add BACKLOG.md
git commit -m "[P4-01] docs: tick acceptance criteria"
```

---

## Task 9: Open the PR

- [ ] **Step 1: Push and open**

```bash
git push -u origin p4-01-qft-bench-framework
gh pr create --title "[P4-01] QFT benchmark + Phase-4 bench/report framework" --body "$(cat <<'EOF'
Closes #39

## Summary
Stands up the shared Phase-4 benchmark+report framework with QFT as the first
consumer; QFT measured to n=30 on the state vector vs Qiskit Aer.

- Shared corpus: `run.py` per-family sizes ⇒ QFT {10,15,20,25,30}; same QASM
  drives both aleph and Aer (single source of truth).
- aleph side: `phase4_qft` corpus bench (NOT the builder), feature-gated n≥28.
- n=30: configurable per-instance qubit cap on `NaiveSvBackend` (default 28
  preserved; bench opts in to 32). Measured on EPYC (16 GiB SV).
- New reusable tooling: `extract_criterion.py` + deterministic `report.py`
  (golden-tested) ⇒ `docs/perf/phase4.md`. P4-02..07 plug in as rows.
- Correctness: existing QFT oracle (n=3,5) + new QFT∘QFT⁻¹ round-trip on both
  |0…0⟩ and a generic random state.

## Test results
- `cargo test --workspace`, `clippy -D warnings`, `cargo fmt --all --check` green.
- `pytest scripts/bench-report` green (extractor + report golden).
- `docs/perf/phase4.md` QFT row to n=30 (EPYC numbers).

## Notes
- n=30 measurement is a documented manual EPYC step (heavy); tooling + small-n
  paths are CI-tested.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch gating CI**

`gh pr checks <PR#>` — gating rustfmt/clippy/test linux+macos. Self-hosted
`bench` is non-gating and may queue behind a Bench run (see
`aleph-bench-server` notes). Merge on gating-green.

---

## Self-Review notes

- **Spec coverage:** corpus/reference impl → Task 2; aleph corpus bench → Task 3;
  Aer timings → Task 2 (run.py); unified schema + report → Tasks 5/6; n=30 cap →
  Task 1; inverse-QFT correctness → Task 4; report deliverable → Task 7; AC ticks
  → Task 8. Reusability contract exercised by the family-agnostic report.py.
- **Type/name consistency:** group name `phase4_qft` used in Task 3 bench, Task 5
  extractor `--group`, and Task 7 README. `with_qubit_cap` used in Task 1 + Task 3.
  Unified aleph JSON keys (`aleph_ms_median`, `aleph_rsd`, `family`, `n`) match
  across extractor (Task 5), report (Task 6), golden fixtures (Task 6).
- **Open verification (flagged in-task, not guessed):** Task 4 depends on the
  exact gate variant `qft_circuit` uses and whether `Gate::inverse()` covers it —
  Step 1 inspects before implementing; the `state.amplitudes()` accessor name is
  matched to the crate API at implementation time.
- **No placeholders:** every code step is concrete; the only manual step (Task 7
  EPYC measurement) has exact commands.
