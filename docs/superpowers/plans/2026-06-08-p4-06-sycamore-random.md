# P4-06 Sycamore-style Random Circuit Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `sycamore` Phase-4 benchmark family (n=20/24/28/30, depth 20) — Sycamore-style {√X,√Y,√W} single-qubit layers + a CZ brick-wall — with an exact linear-XEB value (≈1, noiseless) computed and reported.

**Architecture:** Reuse the Phase-4 corpus→bench→report pipeline. `run.py` deterministically generates the corpus QASM (seeded splitmix64 gate choice, no numpy RNG) and computes Aer's XEB. aleph consumes the committed QASM via `run_optimized` (criterion for n=20/24, the `oneshot` bin for n=28/30 and for all XEB values). `report.py` gains a `sycamore` family title + an optional XEB sub-table. Correctness is gated by a Rust test: `run`≡`run_optimized` amplitude equality + normalization + an XEB sanity band.

**Tech Stack:** Rust (criterion, `NaiveSvBackend`, `aleph-parser`), Python 3.12 (Qiskit 1.2.4 / Aer 0.15.1, stdlib `unittest`).

---

## File Structure

- **Create** `benches/benches/phase4_sycamore.rs` — aleph criterion bench (n=20,24), mirrors `phase4_qpe.rs`.
- **Create** `benches/tests/sycamore_xeb.rs` — correctness oracle (run≡run_optimized, normalization, XEB band).
- **Create** `docs/perf/data/phase4-xeb.json` — committed XEB values (aleph + Aer), produced on EPYC.
- **Create** `scripts/qiskit-baseline/circuits/sycamore_n{20,24,28,30}_d20.qasm` — committed corpus.
- **Create** `scripts/bench-report/testdata/xeb.json` — golden-test input for the XEB sub-table.
- **Modify** `benches/src/lib.rs` — add `linear_xeb`.
- **Modify** `benches/src/bin/oneshot.rs` — also print `xeb <value>`.
- **Modify** `benches/Cargo.toml` — register the `phase4_sycamore` bench.
- **Modify** `scripts/qiskit-baseline/run.py` — `build_sycamore`, family wiring, `aer_xeb`.
- **Modify** `scripts/qiskit-baseline/test_run.py` — determinism + key + XEB unit tests.
- **Modify** `scripts/bench-report/report.py` — `sycamore` title + optional `--xeb` sub-table.
- **Modify** `scripts/bench-report/test_report.py` + `testdata/{aleph,aer}.json` + `testdata/phase4.golden.md` — cover the new path.
- **Modify** `docs/perf/data/{phase4-aer.json,phase4-aleph.json,phase4-meta.json}` + `docs/perf/phase4.md` — EPYC results (Task 8).

---

## Task 1: `linear_xeb` helper

**Files:**
- Modify: `benches/src/lib.rs`
- Test: `benches/src/lib.rs` (`#[cfg(test)]` module at end of file)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests { ... }` at the bottom of `benches/src/lib.rs` (or add the module if absent). Use `aleph_core::Complex`:

```rust
#[test]
fn linear_xeb_uniform_is_zero() {
    // Uniform distribution: every p(x) = 1/D, so 2^n * sum(1/D^2) - 1
    // = D * (D * 1/D^2) - 1 = 0. (A fully depolarized / unscrambled output.)
    let n = 4u32;
    let dim = 1usize << n;
    let amp = Complex::new((1.0 / dim as f64).sqrt(), 0.0);
    let amps = vec![amp; dim];
    assert!(linear_xeb(&amps).abs() < 1e-12);
}

#[test]
fn linear_xeb_peaked_is_dim_minus_one() {
    // A single basis state carries all probability: XEB = D*1 - 1 = D - 1.
    let n = 4u32;
    let dim = 1usize << n;
    let mut amps = vec![Complex::new(0.0, 0.0); dim];
    amps[0] = Complex::new(1.0, 0.0);
    assert!((linear_xeb(&amps) - (dim as f64 - 1.0)).abs() < 1e-12);
}
```

If the `tests` module does not yet `use` `Complex`, add `use aleph_core::Complex;` inside it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-benches --lib linear_xeb`
Expected: FAIL — `cannot find function linear_xeb`.

- [ ] **Step 3: Write the implementation**

Add near the top-level functions of `benches/src/lib.rs` (after the `use` lines). Note `lib.rs` does not currently import `Complex`; add it to the existing top `use aleph_core::{...}` line:

```rust
/// Linear cross-entropy benchmarking (XEB) value of a final state vector,
/// in the exact noiseless (collision-probability) form
/// `XEB = 2^n · Σ_x p(x)² − 1`, where `p(x) = |amp_x|²`.
///
/// For a Porter–Thomas (well-scrambled) circuit this is ≈ 1; for the uniform
/// distribution it is 0. Equivalent to the experimental
/// `2^n·⟨p(x_i)⟩ − 1` when the samples `x_i` are drawn from the ideal
/// distribution itself (the noiseless case). See Arute et al., Nature 574 (2019).
///
/// # Panics
/// Panics if `amps` is empty or its length is not a power of two.
#[must_use]
pub fn linear_xeb(amps: &[Complex]) -> f64 {
    let dim = amps.len();
    assert!(dim.is_power_of_two() && dim > 0, "state length must be a non-zero power of two");
    let sum_p_sq: f64 = amps
        .iter()
        .map(|a| {
            let p = a.norm_sqr();
            p * p
        })
        .sum();
    dim as f64 * sum_p_sq - 1.0
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p aleph-benches --lib linear_xeb`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add benches/src/lib.rs
git commit -m "[P4-06] Add linear_xeb helper (2^n·Σp² − 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Emit XEB from the `oneshot` bin

**Files:**
- Modify: `benches/src/bin/oneshot.rs`

- [ ] **Step 1: Add the XEB print**

In `benches/src/bin/oneshot.rs`, after the `elapsed_ms` print, compute and print XEB from the final amplitudes. Replace the tail of `main` (`black_box(...)` + `println!`) with:

```rust
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let amps = state.amplitudes();
    let xeb = aleph_benches::linear_xeb(amps);
    // Touch the result so the optimiser can't elide the work.
    black_box(amps.len());
    println!("elapsed_ms {elapsed_ms:.3}");
    println!("xeb {xeb:.6}");
```

Update the module doc comment's first paragraph to mention it also prints `xeb <value>` (the noiseless linear XEB of the final state).

- [ ] **Step 2: Build and smoke-test on an existing tiny corpus**

The `oneshot` bin reads any QASM. Use the committed `qft_n10.qasm` to confirm it prints both lines (the value need not be ≈1 for QFT — this only verifies wiring):

Run:
```bash
cargo build -p aleph-benches --bin oneshot && \
./target/debug/oneshot scripts/qiskit-baseline/circuits/qft_n10.qasm
```
Expected: two lines, `elapsed_ms <number>` then `xeb <number>`.

- [ ] **Step 3: Commit**

```bash
git add benches/src/bin/oneshot.rs
git commit -m "[P4-06] oneshot: also print linear XEB of the final state

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `build_sycamore` + family wiring + `aer_xeb` in run.py

**Files:**
- Modify: `scripts/qiskit-baseline/run.py`
- Test: `scripts/qiskit-baseline/test_run.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts/qiskit-baseline/test_run.py` (all gated on `HAVE_RUN`, matching the file's existing pattern):

```python
@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestSycamore(unittest.TestCase):
    def test_keys_and_stem(self):
        # workload_key joins with extract_criterion's "{family}_n{n}" (NO depth
        # suffix, unlike random_brickwall); the file stem is self-describing.
        self.assertEqual(run.workload_key("sycamore", 20), "sycamore_n20")
        self.assertEqual(run.corpus_stem("sycamore", 30), "sycamore_n30_d20")
        self.assertIn("sycamore", run.FAMILY_BUILDERS)
        self.assertEqual(run.FAMILY_SIZES["sycamore"], [20, 24, 28, 30])

    def test_deterministic(self):
        # Same seed -> byte-identical circuit (QASM), twice.
        a = run.build_sycamore(6, 4, run.SYCAMORE_SEED)
        b = run.build_sycamore(6, 4, run.SYCAMORE_SEED)
        from qiskit import qasm3
        self.assertEqual(qasm3.dumps(a), qasm3.dumps(b))

    def test_no_repeat_gate_per_qubit(self):
        # Google's rule: a qubit's single-qubit gate differs from its previous
        # cycle's gate. Check the index sequence per qubit never repeats.
        prev = [None] * 6
        for layer in range(8):
            for q in range(6):
                gi = run._syc_gate_index(run.SYCAMORE_SEED, layer, q, prev[q])
                self.assertIn(gi, (0, 1, 2))
                if prev[q] is not None:
                    self.assertNotEqual(gi, prev[q])
                prev[q] = gi

    def test_aer_xeb_bell_is_one(self):
        # Bell state |00>+|11> /sqrt2: p = [.5,0,0,.5], sum p^2 = .5,
        # D=4 -> XEB = 4*.5 - 1 = 1.0 exactly.
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        self.assertAlmostEqual(run.aer_xeb(qc), 1.0, places=9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
scripts/qiskit-baseline/.venv/bin/python -m unittest discover \
  -s scripts/qiskit-baseline -p 'test_run.py'
```
Expected: FAIL — `module 'run' has no attribute 'build_sycamore'` / `_syc_gate_index` / `aer_xeb`, and `KeyError: 'sycamore'`.

- [ ] **Step 3: Implement in run.py**

Add `import numpy as np` to the imports (numpy is a Qiskit dependency, present in the venv).

Add the Sycamore constants and builders near `RANDOM_DEPTH` (after line 60):

```python
SYCAMORE_DEPTH = 20
SYCAMORE_SEED = 0x5121A6E0  # fixed seed -> byte-reproducible corpus

# R(theta, phi) = exp(-i*theta/2*(cos phi*X + sin phi*Y)). The Sycamore
# single-qubit set is sqrt(X)=R(pi/2,0), sqrt(Y)=R(pi/2,pi/2),
# sqrt(W)=R(pi/2,pi/4) with W=(X+Y)/sqrt(2) (Arute et al., Nature 574, 2019).
_SYC_GATES = [
    (math.pi / 2, 0.0),
    (math.pi / 2, math.pi / 2),
    (math.pi / 2, math.pi / 4),
]


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def _syc_gate_index(seed: int, layer: int, q: int, prev):
    """Deterministic single-qubit gate choice in {0,1,2} for (layer, q),
    honoring Google's rule that a qubit's gate differs from its previous cycle.
    Pure splitmix64 (no numpy RNG / Python hash) so the corpus is byte-stable
    across machines and Python versions."""
    h = _splitmix64(seed ^ _splitmix64((layer << 32) | q))
    if prev is None:
        return h % 3
    others = [g for g in (0, 1, 2) if g != prev]
    return others[h % 2]


def build_sycamore(n: int, depth: int, seed: int) -> QuantumCircuit:
    """Sycamore-style random circuit: alternating single-qubit {sqrt X, sqrt Y,
    sqrt W} layers and a CZ brick-wall (1-D simplification of the 2-D coupler
    grid). Worst case for state-vector simulation (maximum entanglement)."""
    qc = QuantumCircuit(n, name=f"sycamore_n{n}_d{depth}")
    prev = [None] * n
    for layer in range(depth):
        for q in range(n):
            gi = _syc_gate_index(seed, layer, q, prev[q])
            theta, phi = _SYC_GATES[gi]
            qc.r(theta, phi, q)
            prev[q] = gi
        q = layer & 1
        while q + 1 < n:
            qc.cz(q, q + 1)
            q += 2
    return qc


def aer_xeb(tqc: QuantumCircuit) -> float:
    """Exact noiseless linear XEB = 2^n*sum_x p(x)^2 - 1 of the final state
    (collision-probability form). ~1 for a Porter-Thomas circuit, 0 for the
    uniform distribution. Computed from Aer's exact statevector."""
    sim = AerSimulator(method="statevector", max_parallel_threads=1)
    t = tqc.copy()
    t.save_statevector()
    sv = np.asarray(sim.run(t).result().get_statevector())
    p = np.abs(sv) ** 2
    return float(len(p) * np.sum(p**2) - 1.0)
```

Wire the family into the three registries:

```python
# in FAMILY_SIZES (add after random_brickwall):
    "sycamore": [20, 24, 28, 30],
```
```python
# in FAMILY_BUILDERS (add after random_brickwall):
    "sycamore": lambda n: build_sycamore(n, SYCAMORE_DEPTH, SYCAMORE_SEED),
```
In `corpus_stem`, add before the final `return`:
```python
    if family == "sycamore":
        return f"sycamore_n{n}_d{SYCAMORE_DEPTH}"
```
`workload_key` needs **no** change — sycamore falls through to the default `f"{family}_n{n}"` = `sycamore_n{n}`, which is exactly what `extract_criterion.py` emits (do NOT add it to the `random_brickwall` depth-suffixed branch).

Finally, record Aer's XEB for sycamore in the timing loop of `main()`. After the `results["workloads"][key] = {...}` assignment, add:
```python
        if family == "sycamore":
            results["workloads"][key]["aer_xeb"] = aer_xeb(tqc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
scripts/qiskit-baseline/.venv/bin/python -m unittest discover \
  -s scripts/qiskit-baseline -p 'test_run.py'
```
Expected: PASS (existing tests + the 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add scripts/qiskit-baseline/run.py scripts/qiskit-baseline/test_run.py
git commit -m "[P4-06] run.py: build_sycamore family + deterministic gate seq + aer_xeb

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Generate and commit the corpus QASM

**Files:**
- Create: `scripts/qiskit-baseline/circuits/sycamore_n{20,24,28,30}_d20.qasm`

- [ ] **Step 1: Generate (corpus only, no timing)**

`--gen-only` regenerates **all** families' QASM (a side effect of `transpile_and_export` running for every workload). Generate, then keep only the new sycamore files and restore the rest:

```bash
scripts/qiskit-baseline/.venv/bin/python scripts/qiskit-baseline/run.py --gen-only
git checkout -- scripts/qiskit-baseline/circuits  # discard rewrites of existing corpus
git status --porcelain scripts/qiskit-baseline/circuits  # should now show only sycamore_n* as untracked
```
Expected: four new files `sycamore_n20_d20.qasm`, `sycamore_n24_d20.qasm`, `sycamore_n28_d20.qasm`, `sycamore_n30_d20.qasm`.

- [ ] **Step 2: Sanity-check the corpus parses in aleph and is non-trivial**

```bash
./target/debug/oneshot scripts/qiskit-baseline/circuits/sycamore_n20_d20.qasm
```
Expected: prints `elapsed_ms` and an `xeb` near 1.0 (e.g. within ±0.2). If `xeb` is ≈0, the circuit failed to entangle — investigate before committing.

- [ ] **Step 3: Commit**

```bash
git add scripts/qiskit-baseline/circuits/sycamore_n20_d20.qasm \
        scripts/qiskit-baseline/circuits/sycamore_n24_d20.qasm \
        scripts/qiskit-baseline/circuits/sycamore_n28_d20.qasm \
        scripts/qiskit-baseline/circuits/sycamore_n30_d20.qasm
git commit -m "[P4-06] Commit Sycamore-style random circuit corpus (n=20/24/28/30, d=20)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: aleph criterion bench `phase4_sycamore.rs`

**Files:**
- Create: `benches/benches/phase4_sycamore.rs`
- Modify: `benches/Cargo.toml`

- [ ] **Step 1: Write the bench**

Create `benches/benches/phase4_sycamore.rs` (mirrors `phase4_qpe.rs`; criterion sizes 20 & 24, the heavy 28/30 are measured via `oneshot`):

```rust
//! P4-06 Sycamore-style random-circuit benchmark over the committed corpus
//! QASM (the SAME files Aer times), run through the optimized state-vector
//! path. The aleph half of the Phase-4 Sycamore report row. Mirrors
//! phase4_qpe.rs.
//!
//! Criterion sizes n in {20,24} (n=24 => 256 MiB state). The heavy n=28
//! (4 GiB) and n=30 (16 GiB) are measured single-shot via the `oneshot`
//! bin instead, exactly like QFT-30.
//!
//! Corpus: scripts/qiskit-baseline/circuits/sycamore_n{N}_d20.qasm.

use aleph_backend::run_optimized;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::path::PathBuf;

const SMALL_N: &[u32] = &[20, 24];
const DEPTH: u32 = 20;

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/sycamore_n{n}_d{DEPTH}.qasm"))
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, n: u32) {
    let src = std::fs::read_to_string(corpus_path(n))
        .unwrap_or_else(|e| panic!("read sycamore_n{n}_d{DEPTH}.qasm: {e}"));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse sycamore_n{n}: {e:?}"));
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        b.iter(|| {
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
            let state = run_optimized(&mut backend, black_box(&circuit)).expect("simulate");
            black_box(state.amplitudes().len())
        })
    });
}

fn sycamore(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase4_sycamore");
    // n=24 allocates 256 MiB; keep sample counts modest.
    group.sample_size(10);
    for &n in SMALL_N {
        bench_one(&mut group, n);
    }
    group.finish();
}

criterion_group!(benches, sycamore);
criterion_main!(benches);
```

Register it in `benches/Cargo.toml` after the `phase4_qpe` block:
```toml
[[bench]]
name = "phase4_sycamore"
harness = false
```

- [ ] **Step 2: Smoke-test the bench compiles and runs n=20**

Run (short criterion budget so it returns fast):
```bash
cargo bench -p aleph-benches --bench phase4_sycamore -- \
  --warm-up-time 1 --measurement-time 2 --sample-size 10 'phase4_sycamore/20'
```
Expected: compiles, completes, prints a `phase4_sycamore/20` timing.

- [ ] **Step 3: Commit**

```bash
git add benches/benches/phase4_sycamore.rs benches/Cargo.toml
git commit -m "[P4-06] aleph criterion bench phase4_sycamore (n=20,24)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Correctness oracle `sycamore_xeb.rs`

**Files:**
- Create: `benches/tests/sycamore_xeb.rs`

- [ ] **Step 1: Write the test**

A random circuit has no structural oracle; the correctness gate is (1) `run`≡`run_optimized` amplitude equality — proving the fusion passes don't corrupt the timed path, (2) normalization, (3) an XEB sanity band. Create `benches/tests/sycamore_xeb.rs`:

```rust
//! P4-06 acceptance test for the Sycamore-style random circuit. No structural
//! oracle exists for a random circuit, so we gate on three properties over the
//! committed corpus (the SAME QASM Aer runs):
//!
//!  1. `run` ≡ `run_optimized` to 1e-12 — the strong internal oracle. The
//!     fusion / diagonal-fusion / FuseKq passes rewrite the √X/√Y/√W runs and
//!     the CZ brick-wall, so this proves they preserve the exact state on the
//!     path the bench actually times (the P4-03 lesson: oracle must cover
//!     `run_optimized`, not just `run`).
//!  2. Normalization Σp = 1 (1e-10) — unitarity sanity.
//!  3. Linear XEB in a sanity band around 1 (noiseless Porter–Thomas), and
//!     well above 0 (the uniform/depolarized value) — the AC's "XEB ≈ 1".
//!
//! n=20 runs in fast CI; n=24 (256 MiB) is #[ignore]d to the nightly schedule.

use aleph_backend::{run, run_optimized};
use aleph_benches::linear_xeb;
use aleph_sv::NaiveSvBackend;
use std::path::PathBuf;

const DEPTH: u32 = 20;

fn corpus_path(n: u32) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("workspace root")
        .join(format!("scripts/qiskit-baseline/circuits/sycamore_n{n}_d{DEPTH}.qasm"))
}

fn check(n: u32) {
    let path = corpus_path(n);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let circuit =
        aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse sycamore_n{n}: {e:?}"));

    let mut raw = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
    let s_raw = run(&mut raw, &circuit).expect("simulate (run)");
    let mut opt = NaiveSvBackend::with_seed(0).with_qubit_cap(32);
    let s_opt = run_optimized(&mut opt, &circuit).expect("simulate (run_optimized)");

    let a = s_raw.amplitudes();
    let b = s_opt.amplitudes();
    assert_eq!(a.len(), b.len(), "sycamore_n{n}: length mismatch");

    // (1) run ≡ run_optimized.
    let mut max_diff = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).norm());
    }
    assert!(
        max_diff < 1e-12,
        "sycamore_n{n}: run vs run_optimized max amplitude diff {max_diff:.3e} exceeds 1e-12"
    );

    // (2) Normalization on both paths.
    let norm_raw: f64 = a.iter().map(|c| c.norm_sqr()).sum();
    let norm_opt: f64 = b.iter().map(|c| c.norm_sqr()).sum();
    assert!((norm_raw - 1.0).abs() < 1e-10, "sycamore_n{n}: run Σp = {norm_raw}");
    assert!((norm_opt - 1.0).abs() < 1e-10, "sycamore_n{n}: run_optimized Σp = {norm_opt}");

    // (3) XEB sanity band: noiseless Porter–Thomas ⇒ ≈ 1, never the uniform 0.
    // The band is deliberately generous (catches gross corruption / failure to
    // entangle); the precise value is what the benchmark report records.
    let xeb = linear_xeb(b);
    assert!(
        (0.5..=1.5).contains(&xeb),
        "sycamore_n{n}: linear XEB {xeb:.4} outside the sane Porter–Thomas band [0.5, 1.5]"
    );
}

#[test]
fn sycamore_n20_is_correct_and_porter_thomas() {
    check(20);
}

/// n=24 allocates a 256 MiB state vector; over the CLAUDE.md fast-CI budget, so
/// it joins the nightly ignored-tests schedule. n=20 keeps coverage in fast CI.
#[test]
#[ignore = "n=24: 256 MiB state; nightly ignored-tests schedule"]
fn sycamore_n24_is_correct_and_porter_thomas() {
    check(24);
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p aleph-benches --test sycamore_xeb`
Expected: PASS (`sycamore_n20...`); the n24 test shows as ignored.

If the XEB band assertion fails low (≈0.5 or below) at n=20, the 1-D depth-20 circuit under-scrambled — widen `DEPTH` is NOT an option (corpus is fixed); instead confirm via the printed value whether it is a real physics result before loosening the band. Record the observed value in the PR.

- [ ] **Step 3: Commit**

```bash
git add benches/tests/sycamore_xeb.rs
git commit -m "[P4-06] Sycamore oracle: run≡run_optimized + normalization + XEB band

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `report.py` — Sycamore title + optional XEB sub-table

**Files:**
- Modify: `scripts/bench-report/report.py`
- Modify: `scripts/bench-report/test_report.py`
- Modify: `scripts/bench-report/testdata/{aleph.json,aer.json}`
- Create: `scripts/bench-report/testdata/xeb.json`
- Modify: `scripts/bench-report/testdata/phase4.golden.md`

- [ ] **Step 1: Extend the testdata to include a sycamore workload**

Edit `scripts/bench-report/testdata/aleph.json` — add a sycamore row to `workloads`:
```json
  "sycamore_n20": {"n": 20, "family": "sycamore", "aleph_ms_median": 42.0, "aleph_rsd": 0.02}
```
Edit `scripts/bench-report/testdata/aer.json` — add the matching Aer row (mirror the shape of the existing entries there, with `"family": "sycamore"`, `"n": 20`, a `gate_count_post_transpile`, and a `qiskit_aer` block with `median_s`/`stdev_s`). Read the file first and copy an existing entry's structure exactly, changing only name/n/family/numbers, e.g. `median_s: 0.05`, `stdev_s: 0.001`, `gate_count_post_transpile: 1400`.

Create `scripts/bench-report/testdata/xeb.json`:
```json
{"schema_version": 1, "workloads": {
  "sycamore_n20": {"aleph_xeb": 0.9871, "aer_xeb": 0.9872}
}}
```

- [ ] **Step 2: Add the failing test (xeb path)**

Replace the body of `test_report.py`'s test to also pass `--xeb` and assert against the regenerated golden:

```python
    def test_report_matches_golden(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "phase4.md"
            subprocess.run([sys.executable, str(HERE / "report.py"),
                "--aleph", str(TD / "aleph.json"), "--aer", str(TD / "aer.json"),
                "--meta", str(TD / "meta.json"), "--xeb", str(TD / "xeb.json"),
                "--out", str(out)], check=True)
            self.assertEqual(out.read_text(), (TD / "phase4.golden.md").read_text())
```

Run: `python3 -m unittest discover -s scripts/bench-report -p 'test_report.py'`
Expected: FAIL — `report.py: error: unrecognized arguments: --xeb`.

- [ ] **Step 3: Implement the report.py changes**

Add to `FAMILY_TITLES`:
```python
    "sycamore": "Sycamore random",
```
Change `render` to accept an optional `xeb` dict and append a sub-table per family that has XEB data. Replace the `for fam in families:` loop body with:
```python
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
        xw = (xeb or {}).get("workloads", {})
        xrows = [r for r in frows if r["name"] in xw]
        if xrows:
            out.append("### Linear XEB (noiseless)\n")
            out.append("| workload | n | aleph XEB | Aer XEB |")
            out.append("|----------|--:|----------:|--------:|")
            for r in xrows:
                x = xw[r["name"]]
                out.append(f"| `{r['name']}` | {r['n']} | {x['aleph_xeb']:.4f} | {x['aer_xeb']:.4f} |")
            out.append("")
```
Change `render`'s signature to `def render(aleph, aer, meta, xeb=None):`. In `main`, add the optional arg and pass it through:
```python
    ap.add_argument("--xeb", type=Path, default=None)
    ...
    xeb = json.loads(args.xeb.read_text()) if args.xeb else None
    args.out.write_text(render(aleph, aer, meta, xeb))
```

- [ ] **Step 4: Regenerate the golden and verify**

Regenerate the golden from the (now deterministic) tool output, then re-run the test:
```bash
python3 scripts/bench-report/report.py \
  --aleph scripts/bench-report/testdata/aleph.json \
  --aer scripts/bench-report/testdata/aer.json \
  --meta scripts/bench-report/testdata/meta.json \
  --xeb scripts/bench-report/testdata/xeb.json \
  --out scripts/bench-report/testdata/phase4.golden.md
git diff scripts/bench-report/testdata/phase4.golden.md
```
Verify the diff ONLY adds a `## Sycamore random` section (timing row) and a `### Linear XEB (noiseless)` sub-table — nothing in the QFT section changed. Then:
```bash
python3 -m unittest discover -s scripts/bench-report -p 'test_report.py'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench-report/report.py scripts/bench-report/test_report.py \
        scripts/bench-report/testdata/aleph.json scripts/bench-report/testdata/aer.json \
        scripts/bench-report/testdata/xeb.json scripts/bench-report/testdata/phase4.golden.md
git commit -m "[P4-06] report.py: Sycamore title + optional linear-XEB sub-table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: EPYC measurement + commit data + regenerate `phase4.md`

> This is an operational task following the established Phase-4 workflow ([[phase4-status]]). Single-thread BOTH sides. Budget hours for the n=30 sweep (16 GiB state). Do this BEFORE opening the PR (a PR triggers CI + Bench on the shared self-hosted runner).

**Files:**
- Modify: `docs/perf/data/phase4-aer.json`, `docs/perf/data/phase4-aleph.json`, `docs/perf/data/phase4-meta.json`
- Create: `docs/perf/data/phase4-xeb.json`
- Modify: `docs/perf/phase4.md`

- [ ] **Step 1: Transfer to EPYC via git bundle**

```bash
git bundle create /tmp/p4-06.bundle origin/main..p4-06-sycamore-random
# scp /tmp/p4-06.bundle root@195.154.249.85:/tmp/   (see [[aleph_bench_server]])
```
On EPYC, fetch into the reusable checkout `/tmp/aleph-p114/aleph` (keeps the qiskit 1.2.4 / aer 0.15.1 venv + cargo registry): `git fetch /tmp/p4-06.bundle p4-06-sycamore-random && git reset --hard FETCH_HEAD`. `export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH`. Verify the box is idle (`uptime` load ≈ 0, `pgrep -a cargo`) per [[feedback-check-server-clean]].

- [ ] **Step 2: Build release, drive the whole sweep under one nohup script**

Single nohup bash driver (poll once via `until DONE`):
- Aer (single-thread): `.venv/bin/python scripts/qiskit-baseline/run.py --workloads sycamore_n20,sycamore_n24,sycamore_n28,sycamore_n30` → writes `results-qiskit.json` with timings + `aer_xeb`. (n=30 Aer is the slow side — budget ~minutes/run × `timing_runs_for`.)
- aleph criterion (n=20,24): `RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 cargo bench -p aleph-benches --bench phase4_sycamore`.
- aleph oneshot (n=28,30 timing + XEB for all four): `RAYON_NUM_THREADS=1 ./target/release/oneshot scripts/qiskit-baseline/circuits/sycamore_n{N}_d20.qasm` for N in 20,24,28,30 — capture `elapsed_ms` and `xeb`.
Build release first: `RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-benches --bench phase4_sycamore --bin oneshot`.

- [ ] **Step 3: Assemble the committed data files (back on the workstation)**

- `phase4-aer.json`: append the four `sycamore_n{n}` workload entries from `results-qiskit.json` to the existing `workloads` map. Add `"sycamore"` n's to `n_qubits_list` if not present (24, 28).
- `phase4-aleph.json`: run `extract_criterion.py --group phase4_sycamore --family sycamore` to get n=20,24 medians; append; then add n=28,30 entries by hand from the oneshot `elapsed_ms` (single-shot → set `aleph_rsd: 0.0`), shape `{"n":N,"family":"sycamore","aleph_ms_median":<ms>,"aleph_rsd":0.0}`.
- `phase4-xeb.json` (new): `{"schema_version":1,"workloads":{"sycamore_n20":{"aleph_xeb":<oneshot>,"aer_xeb":<run.py>}, ... for 24,28,30}}`.
- `phase4-meta.json`: append a Sycamore note to `notes` (gate set √X/√Y/√W = R(π/2,{0,π/2,π/4}), CZ brick-wall depth 20, seed; n=20/24 criterion, n=28/30 oneshot single-shot; XEB = 2ⁿΣp²−1, ≈1 noiseless). Update `date`/`toolchain` to the measured rustc if it changed.

- [ ] **Step 4: Regenerate phase4.md and verify**

```bash
python3 scripts/bench-report/report.py \
  --aleph docs/perf/data/phase4-aleph.json --aer docs/perf/data/phase4-aer.json \
  --meta docs/perf/data/phase4-meta.json --xeb docs/perf/data/phase4-xeb.json \
  --out docs/perf/phase4.md
git diff docs/perf/phase4.md
```
Verify a `## Sycamore random` section with 4 timing rows + a `### Linear XEB (noiseless)` sub-table whose values are all ≈1. Confirm the AC: n=30 row present (runs on SV), XEB computed.

- [ ] **Step 5: Commit**

```bash
git add docs/perf/data/phase4-aer.json docs/perf/data/phase4-aleph.json \
        docs/perf/data/phase4-meta.json docs/perf/data/phase4-xeb.json docs/perf/phase4.md
git commit -m "[P4-06] EPYC: Sycamore timings + linear XEB; regenerate phase4.md

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Final verification + PR

- [ ] **Step 1: Full local gate**

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p aleph-benches --test sycamore_xeb
python3 -m unittest discover -s scripts/bench-report -p 'test_*.py'
```
Expected: all green. (run.py tests need the venv python and are skipped under bare `python3`; that's expected — they ran in Task 3.)

- [ ] **Step 2: Open the PR**

```bash
git push -u origin p4-06-sycamore-random
gh pr create --title "[P4-06] Random circuit benchmark (Sycamore-style)" --body "$(cat <<'EOF'
Closes #44

## Approach
New `sycamore` Phase-4 benchmark family (n=20/24/28/30, depth 20): deterministic
seeded {√X,√Y,√W} single-qubit layers (R(π/2,{0,π/2,π/4})) + a CZ brick-wall.
Worst case for state-vector simulation (max entanglement). Linear XEB computed
exactly as 2ⁿ·Σp²−1 (≈1, noiseless). Plugs into the existing corpus/bench/report
pipeline — corpus generated by run.py, consumed by aleph via run_optimized.

## Tests
- `linear_xeb` unit tests (uniform→0, peaked→D−1).
- `sycamore_xeb.rs`: run≡run_optimized (1e-12), normalization (1e-10), XEB band
  [0.5,1.5] @ n=20 (n=24 nightly).
- run.py determinism + no-repeat-gate + Bell aer_xeb=1.0 unit tests.
- report.py golden updated for the Sycamore section + XEB sub-table.

## Benchmark
EPYC single-thread both sides; see `docs/perf/phase4.md` Sycamore section
(timing rows n=20/24/28/30 + linear-XEB sub-table). n=30 runs on SV (AC met).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Watch CI green, then merge (squash) per the one-issue-one-PR workflow.**

---

## Self-Review notes

- **Spec coverage:** family (T3) ✓; CZ entangler (T3) ✓; exact XEB (T1, used T2/T6/T8) ✓; seeded determinism (T3) ✓; bench n=20/24 + oneshot 28/30 (T5) ✓; XEB oracle/test (T6) ✓; report row + XEB column (T7) ✓; n=30 on SV + report row + XEB computed AC (T8) ✓; EPYC single-thread workflow (T8) ✓.
- **Refinement vs spec:** dropped the `scaling-bench` feature for sycamore — n=28/30 use the `oneshot` bin directly (the QFT-30 precedent), no criterion entry, so no feature gate is needed. Within the spec's intent.
- **Type consistency:** `linear_xeb(&[Complex]) -> f64` defined T1, called identically in T2 (`aleph_benches::linear_xeb`) and T6. `workload_key` → `sycamore_n{n}` (default branch) matches `extract_criterion` `{family}_n{n}` and the bench group param `n`. `corpus_stem` → `sycamore_n{n}_d20` matches the filenames in T4/T5/T6.
