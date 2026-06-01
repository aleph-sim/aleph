# QFT Parity — Enable Optimization Pipeline in Run Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `run_optimized` driver that runs the existing IR optimization pipeline before simulating, prove it semantics-preserving end-to-end, switch the benches to it, and re-measure QFT (and the full matrix) on EPYC — closing the lone QFT-vs-Aer gap.

**Architecture:** A new `run_optimized` / `run_optimized_with_outcomes` pair in `aleph-backend` clones the circuit, calls the already-existing `Circuit::optimize()` (which runs `PassPipeline::default_pipeline()` — Cancel, DCE, Fuse1q, Fuse2q from P1-09…P1-13), then delegates to the existing raw `run_with_outcomes`. `run()` stays raw (the oracle reference). No new passes, kernels, or `Gate` variants — profiling proved the gap was simply that optimization was never wired into execution (2.45× on qft_n20, 2.89× on qft_n25, EPYC).

**Tech Stack:** Rust (workspace), `aleph-ir::passes` (existing), `thiserror`, `proptest` + `aleph-test` strategies, criterion benches, EPYC (`ssh root@195.154.249.85`) for authoritative perf.

**Spec:** `docs/superpowers/specs/2026-05-31-qft-parity-enable-pipeline-design.md`
**Branch:** `qft-parity-pipeline` (spec already committed at `b40bede`).

### Verified API facts (do not re-derive)
- `aleph-backend/src/lib.rs`: `run` (114), `run_with_outcomes` (143), `BackendError` enum (12–56), `MeasurementRecord` (120). Deps: `aleph-core`, `aleph-ir`, `thiserror`. **No `[dev-dependencies]` yet.**
- `Circuit::optimize(&mut self) -> Result<PassStats, PassError>` **already exists** (`aleph-ir/src/circuit.rs:165`); it runs `PassPipeline::default_pipeline()`. `Circuit` derives `Clone`.
- `aleph_ir::passes::PassError` derives `Debug, Error, PartialEq, Eq` (`passes/mod.rs:49`).
- `NaiveSvBackend::with_seed(u64)` (`aleph-sv/src/backend.rs:28`); `state.amplitudes() -> &[Complex]` (`aleph-sv/src/state.rs:25`).
- `aleph-test` strategies: `arb_circuit_full(nq, nc, n_ops)` and `arb_circuit_emittable(nq, nc, n_ops)` (`aleph-test/src/circuit.rs:329,229`).
- Existing oracle idiom (`aleph-sv/tests/fuse_2q_oracle.rs`): `use aleph_backend::run; use aleph_sv::NaiveSvBackend;` then `run(&mut backend, c).unwrap().amplitudes().to_vec()`.
- Bench: `benches/benches/qiskit_baseline.rs` (`use aleph_backend::run;`, two `run(&mut backend, circuit)` call sites — naive + soa). One-shot: `benches/src/bin/oneshot.rs` (`use aleph_backend::run;`, one call site).

### EPYC operational facts (for Tasks 8–10)
- `ssh root@195.154.249.85`; checkout at `/tmp/aleph-p114/aleph`; cargo at `/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin` (export to PATH). uv-bootstrapped py3.12 venv already at `scripts/qiskit-baseline/.venv` (qiskit 1.2.4 / aer 0.15.1).
- Run long jobs detached (`nohup … > log 2>&1; echo EXIT=$? >> log`) + poll the log. criterion `sample_size` is a **floor**; grover_n25 is the multi-hour long pole. Don't push to `benches/**` while a Bench CI run shares the runner.
- Commits/pushes need the sandbox **off**. Local dev box is arm64 (scalar) — perf numbers are EPYC-only.

---

## Task 1: `BackendError::Optimization` variant

**Files:**
- Modify: `crates/aleph-backend/src/lib.rs` (enum at 12–56; import at 59)

- [ ] **Step 1: Add the import**

At `crates/aleph-backend/src/lib.rs:59`, the line is `use aleph_ir::Circuit;`. Change it to also bring in `PassError`:

```rust
use aleph_ir::passes::PassError;
use aleph_ir::Circuit;
```

- [ ] **Step 2: Add the enum variant**

Inside `pub enum BackendError { … }`, after the `InvalidState { reason }` variant (lib.rs:54–55) and before the closing `}`:

```rust
    #[error("optimization pipeline failed: {0}")]
    Optimization(#[from] PassError),
```

`PassError: PartialEq + Eq` so it composes with `BackendError`'s `#[derive(Debug, thiserror::Error, PartialEq)]`. `#[from]` provides the `?` conversion used in Task 2.

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p aleph-backend`
Expected: builds clean (a `dead_code`/`unused` warning for the new variant is fine until Task 2 uses it).

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-backend/src/lib.rs
git commit -m "[qft-parity] BackendError::Optimization(PassError) variant"
```
(Run commits with the sandbox disabled.)

---

## Task 2: `run_optimized` + `run_optimized_with_outcomes` drivers

**Files:**
- Modify: `crates/aleph-backend/src/lib.rs` (insert after `run_with_outcomes`, which ends at line 171)

- [ ] **Step 1: Add the two driver functions**

Insert immediately after the closing `}` of `run_with_outcomes` (lib.rs:171), before `#[cfg(test)] mod tests` (173):

```rust
/// Optimize `circuit` with the default IR pipeline, then simulate.
///
/// Unlike [`run`], which executes the circuit verbatim (the raw reference
/// path used by oracle tests), this first runs `Circuit::optimize`
/// (`PassPipeline::default_pipeline`: cancellation, DCE, and 1q/2q fusion
/// from P1-09…P1-13). Semantics are preserved — see the end-to-end oracle in
/// `tests/run_optimized_oracle.rs` — and the win is far fewer state-vector
/// passes (QFT collapses ~5×: 970→190 gates at n=20).
///
/// The optimization runs on a clone, so the caller's `circuit` is untouched.
pub fn run_optimized<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<B::State, BackendError> {
    let (state, _outcomes) = run_optimized_with_outcomes(backend, circuit)?;
    Ok(state)
}

/// [`run_optimized`] preserving measurement outcomes.
///
/// Same ordering contract as [`run_with_outcomes`]. The optimization
/// pipeline must not reorder gates across `Measure`/`Barrier`; that
/// invariant is pinned by `tests/run_optimized_oracle.rs`.
pub fn run_optimized_with_outcomes<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<(B::State, Vec<MeasurementRecord>), BackendError> {
    let mut optimized = circuit.clone();
    optimized.optimize()?; // PassError -> BackendError via #[from]
    run_with_outcomes(backend, &optimized)
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p aleph-backend`
Expected: builds clean, no warnings (the `Optimization` variant is now constructed via `?`).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-backend/src/lib.rs
git commit -m "[qft-parity] run_optimized + run_optimized_with_outcomes drivers"
```

---

## Task 3: Dev-dependencies for the end-to-end oracle

**Files:**
- Modify: `crates/aleph-backend/Cargo.toml` (no `[dev-dependencies]` section exists yet)

- [ ] **Step 1: Append the dev-dependencies block**

At the end of `crates/aleph-backend/Cargo.toml` (after the `thiserror` line in `[dependencies]`):

```toml

[dev-dependencies]
aleph-sv     = { path = "../aleph-sv" }
aleph-parser = { path = "../aleph-parser" }
aleph-test   = { path = "../aleph-test" }
proptest     = { workspace = true }
```

These mirror `aleph-sv`'s dev-deps (verified: `aleph-parser`, `aleph-test`, `proptest` are all workspace-keyed there). `aleph-sv` depends on `aleph-backend`, so using it as a *dev*-dependency of `aleph-backend` is a dev-only cycle, which Cargo allows.

- [ ] **Step 2: Verify the dependency graph resolves**

Run: `cargo build -p aleph-backend --tests`
Expected: builds clean (no test files yet, just confirms the dev-deps resolve without a real cycle error).

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-backend/Cargo.toml
git commit -m "[qft-parity] aleph-backend dev-deps for end-to-end oracle"
```

---

## Task 4: End-to-end oracle — amplitude equivalence (fused ≡ raw)

**Files:**
- Create: `crates/aleph-backend/tests/run_optimized_oracle.rs`

- [ ] **Step 1: Write the failing test (amplitude equivalence on fixtures + property)**

Create `crates/aleph-backend/tests/run_optimized_oracle.rs`:

```rust
//! End-to-end oracle: `run_optimized` ≡ `run` on `NaiveSvBackend`, amplitudes
//! within 1e-12. Per-pass oracles already exist in aleph-sv; this pins the
//! whole pipeline+sim path — the exact gap that let P1-14 measure raw-vs-fused.

use aleph_backend::{run, run_optimized};
use aleph_core::Complex;
use aleph_ir::Circuit;
use aleph_sv::NaiveSvBackend;

const TOL: f64 = 1e-12;

fn raw_state(c: &Circuit) -> Vec<Complex> {
    run(&mut NaiveSvBackend::with_seed(0), c)
        .expect("raw run")
        .amplitudes()
        .to_vec()
}

fn opt_state(c: &Circuit) -> Vec<Complex> {
    run_optimized(&mut NaiveSvBackend::with_seed(0), c)
        .expect("optimized run")
        .amplitudes()
        .to_vec()
}

fn assert_states_match(c: &Circuit, label: &str) {
    let a = raw_state(c);
    let b = opt_state(c);
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (*x - *y).norm() < TOL,
            "{label}: amp[{i}] raw={x:?} opt={y:?}"
        );
    }
}

#[test]
fn fixtures_optimized_equals_raw() {
    // Committed shared QASM circuits at small n (cheap to simulate here).
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/"
    );
    let names = [
        "ghz_n15",
        "qft_n15",
        "random_brickwall_n15_d20",
        "grover_n15_iters5",
    ];
    for name in names {
        let path = format!("{dir}{name}.qasm");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let circuit = aleph_parser::parse(&src).unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
        assert_states_match(&circuit, name);
    }
}
```

- [ ] **Step 2: Run it to verify it fails for the right reason first, then passes**

Run: `cargo test -p aleph-backend --test run_optimized_oracle fixtures_optimized_equals_raw`
Expected: **PASS** (drivers from Tasks 1–2 are implemented; this test exercises them). If it FAILS on an amplitude mismatch, that is a real pipeline correctness bug — stop and investigate (do not weaken `TOL`). grover_n15 has 47k gates but n=15 (32k amplitudes) so it runs in seconds.

- [ ] **Step 3: Add the property test (random circuits)**

Append to the same file:

```rust
use aleph_test::circuit::arb_circuit_full;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_optimized_equals_raw(c in arb_circuit_full(5, 2, 30)) {
        // arb_circuit_full may emit measurements; amplitude check ignores
        // outcomes (measurement equivalence is covered in Task 5).
        let a = raw_state(&c);
        let b = opt_state(&c);
        prop_assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            prop_assert!((*x - *y).norm() < TOL);
        }
    }
}
```

(If `arb_circuit_full` produces circuits whose measurements collapse the state nondeterministically and break a pure-amplitude compare, fall back to `arb_circuit_emittable(5, 2, 30)` which the existing fuse oracles use, or filter to gate-only circuits. Decide at implementation time by running Step 4; prefer the strategy the existing `fuse_*_oracle.rs` tests use.)

- [ ] **Step 4: Run the full test file**

Run: `cargo test -p aleph-backend --test run_optimized_oracle`
Expected: PASS (both tests). Run by exit code, not grep.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/tests/run_optimized_oracle.rs
git commit -m "[qft-parity] end-to-end oracle: optimized amplitudes == raw (1e-12)"
```

---

## Task 5: Oracle — measurement outcomes + barrier non-crossing

**Files:**
- Modify: `crates/aleph-backend/tests/run_optimized_oracle.rs`

- [ ] **Step 1: Add a measurement-outcome equivalence test**

Append to the file (uses the public `run_with_outcomes` / `run_optimized_with_outcomes`):

```rust
use aleph_backend::{run_optimized_with_outcomes, run_with_outcomes};

/// Build a small deterministic circuit with measurements: prepare a basis
/// state via X gates, add fusible 1q runs, then measure. Both drivers, same
/// seed, must yield identical (qubit, clbit, outcome) records — fusion must
/// not reorder across measurements.
#[test]
fn measurement_outcomes_optimized_equals_raw() {
    let mut c = Circuit::new(3, 3);
    // Fusible 1q run on q0 that collapses to identity-ish but stays before measure.
    c.x(0).unwrap();
    c.z(0).unwrap();
    c.x(0).unwrap(); // X·Z·X run -> fused 1q block
    c.cnot(0, 1).unwrap();
    c.h(2).unwrap();
    c.add_instruction(aleph_ir::Instruction::Measure { qubit: 0, clbit: 0 }).unwrap();
    c.add_instruction(aleph_ir::Instruction::Measure { qubit: 1, clbit: 1 }).unwrap();
    c.add_instruction(aleph_ir::Instruction::Measure { qubit: 2, clbit: 2 }).unwrap();

    let (_s_raw, raw) = run_with_outcomes(&mut NaiveSvBackend::with_seed(7), &c).unwrap();
    let (_s_opt, opt) = run_optimized_with_outcomes(&mut NaiveSvBackend::with_seed(7), &c).unwrap();

    let key = |r: &aleph_backend::MeasurementRecord| (r.qubit, r.clbit, r.outcome);
    let raw_keys: Vec<_> = raw.iter().map(key).collect();
    let opt_keys: Vec<_> = opt.iter().map(key).collect();
    assert_eq!(raw_keys, opt_keys, "outcomes diverged after optimization");
}
```

Note: `instruction_index` may legitimately differ between raw and optimized (fewer gates shift indices), so compare on `(qubit, clbit, outcome)`, **not** the full record.

- [ ] **Step 2: Add a barrier non-crossing test**

Append:

```rust
/// A Barrier between two fusible 1q gates on q0 must block fusion across it,
/// and the optimized result must still equal the raw result either way.
#[test]
fn barrier_respected_optimized_equals_raw() {
    let mut c = Circuit::new(1, 0);
    c.t(0).unwrap();
    c.add_instruction(aleph_ir::Instruction::Barrier(smallvec::smallvec![0])).unwrap();
    c.t(0).unwrap();
    assert_states_match(&c, "barrier_between_t_gates");
}
```

Add `use smallvec;` if not already imported, or construct the `SmallVec` via `aleph_ir`'s re-export — check how `instruction.rs` tests build `Barrier` (`smallvec::smallvec![...]`) and match it; add `smallvec = { workspace = true }` to `aleph-backend` `[dev-dependencies]` if needed (verify with the build in Step 3).

- [ ] **Step 3: Run the file**

Run: `cargo test -p aleph-backend --test run_optimized_oracle`
Expected: PASS (all four tests). If `smallvec` is unresolved, add it to dev-deps and re-run.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-backend/tests/run_optimized_oracle.rs crates/aleph-backend/Cargo.toml
git commit -m "[qft-parity] oracle: measurement outcomes + barrier non-crossing preserved"
```

---

## Task 6: Switch the criterion bench to `run_optimized`

**Files:**
- Modify: `benches/benches/qiskit_baseline.rs`

- [ ] **Step 1: Update the import**

Change `use aleph_backend::run;` to:

```rust
use aleph_backend::run_optimized;
```

- [ ] **Step 2: Update both call sites**

In `bench_qiskit_baseline`, the two `iter_with_setup` closures (the `naive_aos_avx512` arm and the `soa` arm) each contain `let state = run(&mut backend, circuit).unwrap();`. Change both to:

```rust
let state = run_optimized(&mut backend, circuit).unwrap();
```

- [ ] **Step 3: Update the module doc comment**

Append a sentence to the top-of-file doc comment explaining the timed body now runs `run_optimized` (optimize + simulate) — the honest comparison, since Aer fuses by default. The raw `run` path remains the oracle reference. The optimize cost is intentionally inside the timed region (Aer's fusion is inside its timing too).

- [ ] **Step 4: Verify it compiles**

Run: `cargo bench -p aleph-benches --bench qiskit_baseline --no-run`
Expected: compiles clean.

- [ ] **Step 5: Commit**

```bash
git add benches/benches/qiskit_baseline.rs
git commit -m "[qft-parity] qiskit_baseline bench: time run_optimized (honest vs Aer fusion)"
```

---

## Task 7: Switch the one-shot RSS binary to `run_optimized`

**Files:**
- Modify: `benches/src/bin/oneshot.rs`

- [ ] **Step 1: Update import + call site**

Change `use aleph_backend::run;` to `use aleph_backend::run_optimized;`, and the call `let state = run(&mut backend, &circuit)…` to `let state = run_optimized(&mut backend, &circuit)…`.

- [ ] **Step 2: Update the doc comment**

Change the file's `//!` header to say it runs the optimized pipeline (RSS path matches the bench).

- [ ] **Step 3: Verify it builds**

Run: `cargo build -p aleph-benches --bin oneshot`
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add benches/src/bin/oneshot.rs
git commit -m "[qft-parity] oneshot: run_optimized so RSS path matches the bench"
```

---

## Task 8: Local verification gates

**Files:** none (verification only)

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: PASS, including the new `run_optimized_oracle` and all existing pass/oracle tests.

- [ ] **Step 2: Clippy (separate gate)**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3: Format (separate gate)**

Run: `cargo fmt --check`
Expected: clean. (If it reports diffs, run `cargo fmt` and amend.)

- [ ] **Step 4: CI subset stays cheap and now exercises `run_optimized`**

Run: `cargo bench -p aleph-benches --bench qiskit_baseline -- --test 2>&1 | tail -5`
Expected: runs only the cheap subset (no n>20, no grover), one iteration each — proving the optimized path works end-to-end on the CI cells and stays well under the 30-min Bench timeout.

- [ ] **Step 5: No commit** (verification only). If any gate failed, fix in the relevant task's file and re-run before proceeding.

---

## Task 9: EPYC build + AVX-512 verification

**Files:** none (remote build). Push the branch first (sandbox off), only when `gh run list --branch main` shows no active Bench run.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin qft-parity-pipeline
```

- [ ] **Step 2: Sync + build on EPYC**

```
ssh root@195.154.249.85
cd /tmp/aleph-p114/aleph
git fetch origin && git checkout qft-parity-pipeline && git reset --hard origin/qft-parity-pipeline
export PATH=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-benches --bin oneshot
```
Expected: build EXIT=0.

- [ ] **Step 3: Verify AVX-512 still emitted**

After the bench compiles (Step 1 of Task 10 builds it), run:
```
objdump -d target/release/deps/qiskit_baseline-* | grep -c 'vmulpd.*zmm'
```
Expected: > 0 (the AoS+AVX-512 kernel is still on the hot path; optimization changed the IR, not the kernels).

---

## Task 10: EPYC re-measure full matrix (aleph optimized) + RSS

**Files:** produces criterion medians + RSS values (no source changes). Reuse P1-14 Aer numbers (`results-qiskit.json`) — the QASM and Aer version are unchanged, so only the aleph side is re-timed.

- [ ] **Step 1: Time the full matrix with `run_optimized`, detached + pinned**

```
cd /tmp/aleph-p114/aleph
export PATH=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
nohup bash -c 'ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" taskset -c 0 \
  cargo bench -p aleph-benches --bench qiskit_baseline -- --save-baseline qft-parity \
  > /tmp/aleph-p114/aleph_opt_bench.log 2>&1; echo EXIT=$? >> /tmp/aleph-p114/aleph_opt_bench.log' >/dev/null 2>&1 &
```
Poll `aleph_opt_bench.log` for `EXIT=` with a local background monitor. grover_n25 is again the multi-hour long pole (sample_size is a floor; expect ~hours for its 10 samples, faster than P1-14 since optimized). Capture criterion medians per `(naive_aos_avx512|soa, name)` from the log (`time:` lines) or `target/criterion/**/estimates.json`.

- [ ] **Step 2: Peak RSS for the optimized path at n=25 (aleph)**

```
for name in ghz_n25 qft_n25 random_brickwall_n25_d20; do
  echo "== $name =="
  /usr/bin/time -v taskset -c 0 ./target/release/oneshot scripts/qiskit-baseline/circuits/$name.qasm 2>&1 \
    | grep 'Maximum resident'
done
```
Expected: ~515 MiB (fusion shouldn't materially change RSS — confirm; Aer RSS reused from P1-14).

- [ ] **Step 3: Extract + record the numbers**

Pull the 16 aleph optimized medians (+ 8 soa) and the 3 RSS values back to the workstation (scp the log, or paste). Compute `aleph_opt / Aer` per cell using the existing `results-qiskit.json` Aer medians. Confirm QFT is now ≤ 1× at all n.

---

## Task 11: Add "Optimized pipeline" section to `docs/perf/phase1.md`

**Files:**
- Modify: `docs/perf/phase1.md` (append a new section; **do not overwrite** the raw tables — per user decision)

- [ ] **Step 1: Write the new section**

After the existing report body (before or after the appendix — pick a clear top-level position), add:

```markdown
## Optimized pipeline (run_optimized) — the honest comparison

The headline tables above are **raw single-gate kernel throughput** — the run
path executed each parsed gate verbatim, with the P1-09…P1-13 IR optimization
passes (cancellation, DCE, 1q/2q fusion) NOT applied. Qiskit Aer fuses gates by
default, so those tables compare un-fused aleph against fused Aer. This section
re-measures with `run_optimized` (aleph's `default_pipeline` + simulate), the
apples-to-apples comparison. EPYC 8124P, same pinning, same committed QASM, same
Aer numbers (`results-qiskit.json`).

Gate-count reduction from `default_pipeline` (per family, representative):
qft_n20 970→190, qft_n25 1525→300, random_n20 990→192, grover_n15 47805→12083,
ghz_n20 20→19.
```

Then one headline table per algorithm with columns: `n`, `aleph_opt (ms)`, `Aer (ms)`, `aleph/Aer`, `≤2× verdict`, plus a `peak RSS` row note at n=25. Fill from Task 10 numbers. Add a re-stated **ROADMAP §7 verdict**: QFT now ≤ 1× at n=25 (parity/ahead), all families improved.

- [ ] **Step 2: Sanity-check arithmetic**

Verify each `aleph/Aer = aleph_opt_ms ÷ Aer_ms`; verify the QFT n=25 cell matches the probe direction (~0.6× ahead). Re-read for internal consistency; every cell filled, no placeholders.

- [ ] **Step 3: Commit**

```bash
git add docs/perf/phase1.md
git commit -m "[qft-parity] phase1.md: Optimized pipeline section (QFT now <=1x Aer)"
```

---

## Task 12: Code review, finish branch, PR

**Files:** none (process)

- [ ] **Step 1: `/code-review high`** on the code diff (driver + oracle + bench/oneshot wiring). Address any correctness findings; re-run Task 8 gates after fixes.

- [ ] **Step 2: Final verification before PR**

Run: `git status --porcelain && git log --oneline main..HEAD`
Expected: clean tree; a coherent sequence of `[qft-parity]` commits.

- [ ] **Step 3: Open the PR** (use superpowers:finishing-a-development-branch)

Title: `[qft-parity] Enable optimization pipeline in run path`. Body: root cause (P1-14 measured raw aleph vs fused Aer), the EPYC QFT numbers (qft_n20 2.45×, qft_n25 2.89× speedup; QFT flips from 1.73× behind to ~0.61× ahead at n=25), the **no-new-kernels** framing, test results (end-to-end oracle), the new report section. **No `Closes #`** — this is a Phase-1 follow-up, not a numbered backlog item (state that in the body). Poll CI; squash-merge on green **with user approval**; delete branch.

---

## Self-Audit

- **Spec coverage:** §1 root cause → report section (T11); §3 architecture (`run_optimized` pair, `run()` stays raw, error variant) → T1–T2; §4 correctness (per-pass oracles exist + new end-to-end amplitude/outcome/barrier) → T4–T5; §5 bench + oneshot + report-as-new-section → T6, T7, T10, T11; §6 verification gates → T8; §7 risks (reorder across measure/barrier → T5; clone cost → documented T2/T6; long EPYC run → T10 nohup+monitor, reuse Aer numbers) ✓. §8 out-of-scope correctly excluded.
- **Placeholder scan:** every code step shows real code; the one judgment call (arb strategy choice in T4 Step 3) has an explicit fallback and a decision procedure. ✓
- **Type/name consistency:** `run_optimized` / `run_optimized_with_outcomes`, `BackendError::Optimization`, `MeasurementRecord` `(qubit, clbit, outcome)`, `Circuit::optimize()`, `state.amplitudes()`, `NaiveSvBackend::with_seed` — used identically across tasks. ✓
- **Ordering:** error variant → drivers → dev-deps → oracle (amplitude → outcomes/barrier) → bench → oneshot → local gates → EPYC build → measure → report → review/PR. Correctness gate (T4–T5) precedes any perf claim (T10–T11). ✓
- **No new kernels/passes/Gate variants** anywhere — only wiring, tests, bench, docs. ✓
