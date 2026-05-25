# P0-10 — Oracle comparison harness vs. Qiskit Aer

**Issue:** P0-10 (see `BACKLOG.md`)
**Depends on:** P0-08 (`aleph-parser`), P0-09 (`Backend` trait + `NaiveSvBackend`)
**Date:** 2026-05-24

---

## 1. Goal

Build a test harness that runs a corpus of circuits through `NaiveSvBackend`
and compares the resulting state vector against Qiskit Aer, asserting
equivalence to within tolerance. Qiskit Aer is treated as ground truth;
any disagreement is a bug in `aleph` (until proven otherwise).

This is the project's primary regression-detection mechanism for backend
correctness from P0-10 onward. Every future backend, IR pass, kernel
rewrite, and optimization rides on top of it.

---

## 2. Scope

### In scope

- A new test-only crate `aleph-oracle` containing the fixture loader, a
  generic `run_state_oracle<B: Backend>` harness, assertion helpers, and
  one integration test (`tests/naive_sv.rs`) that pins
  `NaiveSvBackend`.
- A Python fixture generator at `oracle/` packaged with `uv` +
  `pyproject.toml` + `uv.lock`, pinning exact Qiskit and Aer versions.
- A regeneration entry point `scripts/regen-fixtures.sh` that invokes
  `uv run --project oracle gen.py`.
- A corpus of **26 hand-curated OpenQASM 3.0 circuits** under
  `oracle/circuits/` covering: identity, every supported single- and
  multi-qubit gate kernel, Bell pair, GHZ-{3,5,10}, QFT-{3,5}, random
  Clifford and non-Clifford circuits at n=4, Grover-2, and an
  awkward-angle rotation chain.
- A JSON fixture format (`schema_version: 1`) that stores both the
  state vector and 100k-shot counts per circuit, so P0-11 (sampling)
  can re-use the same fixtures without regenerating anything.
- A `docs/testing.md` "Oracle harness" section covering: what is
  checked, how to add a new circuit, how to interpret a failure, and
  the policy for bumping Qiskit/Aer versions.

### Out of scope

- Sampling-side assertions (`run_distribution_oracle`). The `counts`
  field is captured in fixtures now but only **consumed** in P0-11.
  No dead `#[cfg]`-gated entry point ships in P0-10.
- Live Python-subprocess comparison. Pre-generated, committed fixtures
  are the contract.
- Mid-circuit measurement, conditional gates, and reset (`aleph-ir`
  supports them, P0-09 backend does not, fixtures will not exercise
  them yet).
- Backends other than `NaiveSvBackend`. The harness is generic over
  `Backend`, but only one impl exists today, so only one integration
  test is written. Future backends bring their own test file in their
  own crate and re-use `aleph-oracle` as a library dep.
- A CI workflow that regenerates fixtures automatically. Recorded as a
  follow-up for a later phase (see §10).

---

## 3. Architecture

```
┌──────────────────────────────────┐      committed to git
│ oracle/                          │      ────────────────
│  ├── pyproject.toml  (Qiskit/Aer)│
│  ├── uv.lock                     │
│  ├── gen.py          (generator) │
│  ├── circuits/*.qasm  (26 files) │  ── inputs
│  └── fixtures/*.json  (26 files) │  ── outputs
└──────────────────────────────────┘
                 │
       scripts/regen-fixtures.sh
                 │
                 ▼
┌──────────────────────────────────┐
│ crates/aleph-oracle/             │  test-only library + integration test
│  ├── src/lib.rs                  │   re-exports
│  ├── src/fixture.rs              │   Fixture + load_fixture()
│  ├── src/harness.rs              │   run_state_oracle, assertion helpers
│  ├── build.rs                    │   enumerates fixtures → one #[test] each
│  └── tests/naive_sv.rs           │   includes! the generated tests
└──────────────────────────────────┘
                 │
                 ▼
        Rust CI: cargo test --workspace  (no Python in CI lane)
```

The dataflow is one-directional: contributor edits a `.qasm` or adds a
new one, runs `scripts/regen-fixtures.sh` locally, commits both the
QASM and the regenerated JSON. CI consumes the JSON exclusively.

---

## 4. Fixture format

One JSON file per circuit, schema-versioned. Floats serialized at full
17-digit precision so Python → Rust round-trip is bit-exact through
`serde_json`.

```json
{
  "schema_version": 1,
  "name": "ghz_10",
  "num_qubits": 10,
  "qasm_path": "circuits/ghz_10.qasm",
  "qiskit_version": "1.2.4",
  "aer_version": "0.15.1",
  "generated_at": "2026-05-24T00:00:00Z",
  "shots": 100000,
  "rng_seed": 0,
  "statevector": {
    "endianness": "little",
    "amplitudes": [[0.7071067811865475, 0.0], [0.0, 0.0], "…"]
  },
  "counts": {
    "0000000000": 50012,
    "1111111111": 49988
  }
}
```

Notes:

- **JSON, not binary.** GHZ-10 fixture is ≈ 50 KB. Full corpus
  ≲ 1 MB committed. Trivial repo impact. Human-diffable in PR review.
- **Float precision.** Python's stdlib `json.dumps` writes each float
  via `float.__repr__`, which is the shortest decimal that round-trips
  back to the same IEEE 754 `f64`. `serde_json` reads JSON numbers
  into `f64` via a correct-rounding parser. The two together are
  bit-exact for finite `f64`. No custom encoder is needed.
  `aleph-oracle` confirms this once with a round-trip property test
  (`generate random f64 → serde_json round-trip → assert
  ==`).
- **Endianness pinned.** Qiskit uses little-endian qubit ordering
  (qubit 0 is the LSB of the basis-state index `i`).
  `aleph_sv::CpuState::amplitudes()` already follows the same
  convention. Both sides assert `endianness == "little"`. A future
  backend with a different layout will translate on read.
- **Tool versions recorded** so triage can tell at a glance whether a
  Qiskit/Aer upgrade explains a delta.
- **`counts` always present.** Generated with `shots = 100_000`,
  `seed_simulator = 0`. Untouched by the P0-10 Rust code; P0-11 turns
  it into an assertion.

---

## 5. Rust crate: `aleph-oracle`

### 5.1 Layout

```
crates/aleph-oracle/
├── Cargo.toml          # publish = false
├── build.rs            # generates tests/_generated.rs at compile time
├── src/
│   ├── lib.rs
│   ├── fixture.rs
│   └── harness.rs
└── tests/
    └── naive_sv.rs     # include!s the generated #[test] list
```

`Cargo.toml` declares dev/lib dependencies on `aleph-backend`,
`aleph-sv`, `aleph-ir`, `aleph-parser`, `aleph-core`, `num-complex`,
`serde`, `serde_json`, `thiserror`. Marked `publish = false`.

### 5.2 `fixture.rs`

```rust
#[derive(Debug, serde::Deserialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub name: String,
    pub num_qubits: u32,
    pub qasm_path: String,
    pub qiskit_version: String,
    pub aer_version: String,
    pub generated_at: String,
    pub shots: u64,
    pub rng_seed: u64,
    pub statevector: StateVectorFixture,
    pub counts: std::collections::BTreeMap<String, u64>,
}

#[derive(Debug, serde::Deserialize)]
pub struct StateVectorFixture {
    pub endianness: String,           // must be "little"
    pub amplitudes: Vec<(f64, f64)>,  // (re, im)
}

pub fn load_fixture(path: &std::path::Path) -> Result<Fixture, OracleError> { … }
pub fn load_qasm(path: &std::path::Path) -> Result<String, OracleError> { … }
```

Paths into `oracle/fixtures/` and `oracle/circuits/` are resolved
relative to `CARGO_MANIFEST_DIR` (`../../oracle/...`). This is brittle
to crate-tree moves but matches the repo's existing fixture pattern
(`crates/aleph-parser/tests/...`).

### 5.3 `harness.rs`

```rust
pub const STATE_TOLERANCE: f64 = 1e-10;

pub fn run_state_oracle<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
) -> Result<(), OracleError>
where
    B: Backend,
    B::State: HasAmplitudes,   // see note
{
    let circuit = aleph_parser::parse(qasm_source)?;
    let state = aleph_backend::run(backend, &circuit)?;
    assert_state_close(&fixture.name, state.amplitudes(), &fixture.statevector.amplitudes)?;
    Ok(())
}
```

Note on `HasAmplitudes`: `Backend::State` has no amplitude accessor in
the trait today (P0-09 deliberately kept it opaque). For P0-10 we
declare a small **internal-to-`aleph-oracle`** trait
`HasAmplitudes` and implement it for `aleph_sv::CpuState` only. When
P0-11+ exposes a public amplitudes method (or its equivalent), the
internal trait collapses into a re-export. This avoids a premature
public API change to `aleph-backend`.

`assert_state_close` walks amplitude-by-amplitude and panics on the
first index where `(ours - theirs).norm() > STATE_TOLERANCE`. The
panic message contains: fixture name, index, basis-state label
(`format!("|{:0width$b}>", i, width = n)`), our value, expected,
`|Δ|`, tolerance. Tests fail loudly with full triage information.

### 5.4 `build.rs` test enumeration

The build script walks `oracle/fixtures/*.json` at compile time and
emits `OUT_DIR/_generated.rs` containing one `#[test]` per fixture:

```rust
#[test]
fn ghz_10() {
    let fx = aleph_oracle::load_fixture(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/fixtures/ghz_10.json"
        ))).expect("load fixture");
    let qasm = aleph_oracle::load_qasm(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../oracle/", "circuits/ghz_10.qasm"
        ))).expect("load qasm");
    let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
    aleph_oracle::run_state_oracle(&mut backend, &fx, &qasm).unwrap();
}
```

`tests/naive_sv.rs` is one line: `include!(concat!(env!("OUT_DIR"), "/_generated.rs"));`.

Test names match fixture names, so a failure surfaces as
`naive_sv::ghz_10 FAILED`. Adding a fixture requires no Rust edits.

`build.rs` declares `cargo:rerun-if-changed=../../oracle/fixtures` so
adding/removing JSON re-runs codegen.

### 5.5 `OracleError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("fixture I/O at {path}: {source}")]
    Io { path: std::path::PathBuf, source: std::io::Error },

    #[error("fixture {name} has unsupported schema_version {found} (expected {expected})")]
    SchemaVersion { name: String, found: u32, expected: u32 },

    #[error("fixture {name} qubit-count mismatch: fixture says {fixture}, parsed circuit has {circuit}")]
    QubitMismatch { name: String, fixture: u32, circuit: u32 },

    #[error("fixture {name} dimension mismatch: fixture has {fixture} amplitudes, state has {state}")]
    DimensionMismatch { name: String, fixture: usize, state: usize },

    #[error("fixture {name} declares endianness {endianness:?}, only \"little\" is supported")]
    UnsupportedEndianness { name: String, endianness: String },

    #[error("backend error: {0}")]
    Backend(#[from] aleph_backend::BackendError),

    #[error("parse error: {0}")]
    Parse(#[from] aleph_parser::ParseError),
}
```

Harness-level problems (missing files, schema drift, parse failures)
surface as `OracleError`. **Correctness failures** — an amplitude
outside tolerance — surface as panics inside `#[test]`. This split
keeps the test output readable: an `Err(...)` indicates the harness
is broken; a panic indicates the simulator is broken.

---

## 6. Python generator: `oracle/`

### 6.1 `pyproject.toml`

```toml
[project]
name = "aleph-oracle-gen"
version = "0"
requires-python = ">=3.11"
dependencies = [
  "qiskit==1.2.4",
  "qiskit-aer==0.15.1",
]
```

`oracle/uv.lock` is committed. Bumping either version is a deliberate
PR that also regenerates and commits all 26 fixtures.

### 6.2 `gen.py`

Single file, ≲ 200 LoC. Responsibilities:

1. Enumerate `circuits/*.qasm`.
2. For each: parse via `qiskit.qasm3.loads`, run through
   `AerSimulator(method="statevector")` with `save_statevector()` to
   obtain the exact final amplitudes.
3. Separately, append `measure_all()` to a copy, run with
   `shots=100_000, seed_simulator=0`, and capture the counts dict.
4. Write `fixtures/<name>.json` with the schema in §4 via plain
   `json.dumps(..., indent=2, ensure_ascii=False, sort_keys=True)`.
   No custom float formatting: Python's default already round-trips
   `f64` losslessly through `serde_json` (see §4).
5. Print a one-line summary per fixture (`✓ ghz_10 (1024 amps, 2 counts)`).

The generator is idempotent: re-running over an unchanged corpus
produces byte-identical fixture files (same float repr, sorted
counts keys, no timestamps in volatile fields).

Caveat on `generated_at`: it's stamped per regeneration, so two
regen runs *do* produce different bytes there. That field is
documentation, not part of the contract. The Rust deserializer
treats it as an opaque string.

### 6.3 `scripts/regen-fixtures.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../oracle"
uv run --project . gen.py
```

Marked executable, follows existing `scripts/*.sh` pattern.

The BACKLOG mentions `make regen-fixtures`. The repo has no Makefile
today and the existing helper scripts all live under `scripts/*.sh`.
**Spec amendment §10.1**: substitute `scripts/regen-fixtures.sh` for
`make regen-fixtures`; `BACKLOG.md` will be amended in the
implementation PR to match.

---

## 7. Corpus

26 circuits, all written as plain OpenQASM 3.0 text (no Qiskit
library calls embedded in the QASM itself — the QASM is the
contract). The two `random_*` circuits are produced **once** by a
one-shot Python helper (`oracle/_one_shot_random.py`, committed but
not part of the regen flow) that uses `qiskit.quantum_info.random_clifford`
and a hand-rolled non-Clifford generator with `random.Random(seed=0)`,
then writes the result via `qiskit.qasm3.dumps` into
`oracle/circuits/random_*.qasm`. After they are committed, the QASM
is the input of record; the helper does not run again. Only the
corresponding `fixtures/*.json` re-computes on `regen-fixtures.sh`.

| # | File                              | Qubits | Notes                                        |
|---|-----------------------------------|--------|----------------------------------------------|
| 1 | `identity_1q.qasm`                | 1      | `qubit[1] q;` and nothing else               |
| 2 | `kernel_h.qasm`                   | 1      | Single H                                     |
| 3 | `kernel_x.qasm`                   | 1      | Single X                                     |
| 4 | `kernel_y.qasm`                   | 1      | Single Y                                     |
| 5 | `kernel_z.qasm`                   | 1      | Single Z                                     |
| 6 | `kernel_s.qasm`                   | 1      | Single S                                     |
| 7 | `kernel_sdg.qasm`                 | 1      | Single Sdg                                   |
| 8 | `kernel_t.qasm`                   | 1      | Single T                                     |
| 9 | `kernel_tdg.qasm`                 | 1      | Single Tdg                                   |
| 10| `kernel_rx.qasm`                  | 1      | `rx(pi/3)`                                   |
| 11| `kernel_ry.qasm`                  | 1      | `ry(pi/3)`                                   |
| 12| `kernel_rz.qasm`                  | 1      | `rz(pi/3)`                                   |
| 13| `kernel_cx.qasm`                  | 2      | CX with input `|10>` (target flip)           |
| 14| `kernel_cz.qasm`                  | 2      | CZ with input `|11>` (phase check)           |
| 15| `kernel_swap.qasm`                | 2      | SWAP with input `|10>`                       |
| 16| `kernel_ccx.qasm`                 | 3      | CCX with input `|110>`                       |
| 17| `bell_phi_plus.qasm`              | 2      | H + CX, the canonical Bell pair              |
| 18| `ghz_3.qasm`                      | 3      | H + chained CX                               |
| 19| `ghz_5.qasm`                      | 5      |                                              |
| 20| `ghz_10.qasm`                     | 10     | Largest fixture (≈ 50 KB JSON)               |
| 21| `qft_3.qasm`                      | 3      | Hand-written QFT, no library calls           |
| 22| `qft_5.qasm`                      | 5      |                                              |
| 23| `random_clifford_n4_d20.qasm`     | 4      | Pre-generated with a seeded Clifford sampler |
| 24| `random_nonclifford_n4_d20.qasm`  | 4      | Same idea, includes RX/RY/RZ                 |
| 25| `grover_2q_mark11.qasm`           | 2      | One Grover iteration, marked state `|11>`    |
| 26| `awkward_angles.qasm`             | 1      | `rx(pi/7); ry(pi/3); rz(pi/5);`              |

Total committed JSON: ≈ 150 KB. Total committed QASM: ≈ 4 KB.

Inputs that produce non-trivial output states (kernels 13–16 in
particular) need explicit state preparation in the QASM file
(e.g. `x q[0]; x q[1]; ccx q[0], q[1], q[2];` for fixture #16). The
generator does not "set up" inputs — it runs whatever the QASM says
from `|0…0>`.

Adding a new circuit later: drop a `.qasm` into `oracle/circuits/`,
run `scripts/regen-fixtures.sh`, commit both files. No Rust edit
needed — `build.rs` discovers it automatically.

---

## 8. Tolerances and floating-point semantics

- **Per-amplitude tolerance:** `|(a_ours - a_qiskit)| < 1e-10`,
  where `|·|` is the norm of the complex difference (not max of
  re/im separately, which would yield a slightly larger ε-ball).
- **No relative tolerance.** Amplitudes are bounded by 1 in magnitude
  for normalized states, so an absolute bound is appropriate.
- **No normalization re-check.** The naive backend already maintains
  normalization to machine precision; if it didn't, the
  per-amplitude check would catch it anyway.
- **Sampling assertions** (P0-11) will use `1e-5` on per-basis-state
  probabilities at 100k shots, as specified in BACKLOG. Out of scope
  here.

If a fixture starts failing in the future at a `|Δ| ≈ 1e-12` level,
the suspected causes (in order) are: (a) Qiskit/Aer upgrade changed
internal gate ordering, (b) `aleph` kernel change introduced
slightly different summation order, (c) genuine bug. The triage
playbook in `docs/testing.md` walks through this.

---

## 9. Testing the harness itself

`aleph-oracle` ships its own unit tests:

- **Float round-trip:** generate 10k random `f64` values, serialize
  via the encoder format used by `gen.py` (replicated in a Rust
  helper for the test), deserialize, assert bit-equal.
- **Fixture loader rejects bad schema:** `schema_version: 999` →
  `OracleError::SchemaVersion`.
- **Fixture loader rejects qubit-count mismatch.**
- **Endianness check:** fixture with `endianness: "big"` →
  `OracleError::UnsupportedEndianness`.
- **At least one positive smoke test in-tree** that does not depend
  on `oracle/fixtures/` (a synthetic `Fixture` built in the test),
  so harness logic can be tested even when fixtures are stale.

The corpus-driven tests in `tests/naive_sv.rs` are the integration
tier.

---

## 10. Out-of-scope follow-ups

10.1 **`BACKLOG.md` amendment.** Change "`make regen-fixtures`
     regenerates from Python" to "`scripts/regen-fixtures.sh`
     regenerates from Python" in the P0-10 acceptance criteria.
     Performed in the implementation PR alongside the new files.

10.2 **CI workflow for fixture drift detection.** A
     `workflow_dispatch`-triggered Action that runs
     `scripts/regen-fixtures.sh` in a Python-equipped runner and
     opens a PR if anything changed. Useful when a contributor
     forgets to regenerate. **Not P0-10.** Estimate: S, target
     Phase 1.

10.3 **Distribution-side assertions** (P0-11). The `counts` field is
     already in fixtures; `run_distribution_oracle` will be added by
     P0-11 as a new public function in `aleph-oracle`.

10.4 **Backend coverage expansion.** When the second backend lands
     (MPS, Stab, or GPU), its test file (`tests/<backend>.rs`) in its
     own crate depends on `aleph-oracle` as a dev-dependency and
     re-runs the same fixtures.

---

## 11. Acceptance-criteria mapping

| BACKLOG AC                                          | Where satisfied                  |
|-----------------------------------------------------|----------------------------------|
| At least 10 circuits in the test corpus             | 26, listed in §7                 |
| All tests pass against the naive backend            | `tests/naive_sv.rs` via `build.rs` codegen |
| `make regen-fixtures` regenerates from Python       | `scripts/regen-fixtures.sh` (§6.3); BACKLOG amended per §10.1 |
| Documented in `docs/testing.md`                     | "Oracle harness" section added   |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qiskit/Aer upgrade changes amplitudes at `1e-15` (within our `1e-10` tolerance but visible in JSON diffs). | Acceptable; tolerance is loose enough to absorb. Fixture diffs in PR review will look noisy on Qiskit upgrades — call this out in `docs/testing.md`. |
| Generated JSON not byte-stable across regen runs (`generated_at`). | Documented explicitly. Reviewers know to ignore the timestamp. |
| Brittle path resolution (`CARGO_MANIFEST_DIR/../../oracle/...`). | Centralized in two helper functions in `fixture.rs`; if the workspace layout moves, two lines change. |
| Adding a new gate to `Gate` enum without a corresponding kernel fixture leaves a blind spot. | `aleph-oracle` ships a unit test with an explicit list of expected `Gate::name()` strings (mirroring the supported gate set) and asserts each name appears as a token in at least one `oracle/circuits/*.qasm` file. The list itself is updated by hand when `Gate` grows. This is fragile by design — a missing gate triggers a missing-coverage error, which is the desired failure mode. |
| `build.rs` codegen breaks IDE integration (rust-analyzer occasionally struggles with `include!`). | Acceptable; the generated file is plain Rust and `cargo expand` works. If it becomes a real pain point we can move to `rstest`. |
