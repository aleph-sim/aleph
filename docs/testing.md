# Testing

## Oracle harness (P0-10)

`aleph-oracle` is the project's primary regression-detection net. It
runs each fixture circuit through `NaiveSvBackend` and asserts the
final state vector matches Qiskit Aer's `save_statevector` output to
within `1e-10` per amplitude (complex magnitude of the difference).

### Layout

- `oracle/circuits/*.qasm` — input circuits (26 today).
- `oracle/fixtures/*.json` — generated Qiskit ground truth, committed.
- `oracle/gen.py` — generator; `oracle/pyproject.toml` pins
  `qiskit==1.2.4` and `qiskit-aer==0.15.1` via `uv`.
- `scripts/regen-fixtures.sh` — one-line wrapper around `gen.py`.
- `crates/aleph-oracle/` — Rust loader + harness + `build.rs` test
  codegen.

### Adding a new circuit

1. Drop a `.qasm` file into `oracle/circuits/`.
2. Run `./scripts/regen-fixtures.sh`.
3. Commit both the QASM file and the new `oracle/fixtures/<name>.json`.

No Rust changes needed — `build.rs` discovers the new fixture and
generates a `#[test]` named after the file stem on the next
`cargo test -p aleph-oracle`.

### Interpreting a failure

Failures take one of two shapes:

- **`OracleError` returned from the harness** — a wiring problem:
  missing file, JSON schema drift, qubit-count or dimension
  mismatch, parse error. Read the message; the failure is in the
  fixture or the QASM, not the simulator.
- **`panic!` with `oracle: <name> amplitude mismatch …`** — a
  correctness deviation. The message names the basis state, our
  amplitude, Qiskit's amplitude, and the magnitude of the
  difference. This is almost always a bug in `aleph` (P0-09 backend
  or a future kernel). The Qiskit/Aer version that generated the
  fixture is captured in the JSON for triage.

### Bumping Qiskit / Aer

Bumping `qiskit` or `qiskit-aer` in `oracle/pyproject.toml` is a
deliberate PR that:

1. Updates the version pins.
2. Re-locks via `cd oracle && uv lock`.
3. Runs `./scripts/regen-fixtures.sh` to regenerate every fixture.
4. Reviews the resulting fixture diff. Amplitude-level changes at
   `~1e-15` are expected (Qiskit's internal gate-application order
   may shift slightly between versions). Anything at `1e-10` or
   larger is suspicious — investigate before merging.

### Determinism notes

- Generator output is byte-stable across runs except for the
  `generated_at` timestamp in each fixture.
- Counts use `seed_simulator=0`; identical Qiskit/Aer versions
  produce identical count distributions.
- The two `random_*.qasm` files are produced once via
  `oracle/_one_shot_random.py` with hard-coded seeds (0 and 1).
  The QASM itself is the input of record — re-running the
  one-shot generator after a Qiskit upgrade may change the
  text, which would then change the fixtures. Treat
  `_one_shot_random.py` as documentation, not part of regen.

### f64 round-trip note

`serde_json`'s **default** parser can drift up to 2 ulps from
`f64::from_str` on some inputs. `aleph-oracle` enables the
`float_roundtrip` feature on `serde_json` to switch in a high-precision
parser; with it, every finite `f64` round-trips bit-exactly through
the fixture format. The property test
`f64_pair_round_trips_through_serde_json` locks this in.

## Distribution oracle (P0-11)

The state-vector oracle compares amplitudes (an `O(2^n)` exact
check). The distribution oracle complements it: for each fixture
it samples `DISTRIBUTION_SHOTS = 100_000` shots through
`NaiveSvBackend` and asserts the empirical histogram lies within
a `5σ + 1e-6` band of `|ψ_qiskit|²` per outcome.

This catches bugs the state oracle would miss:

- A regression in `Backend::sample` that ships valid amplitudes
  but wrong indices.
- A drift in the alias-table build that shifts probability mass
  between outcomes while leaving amplitudes intact.
- A future RNG / seed-handling change that produces correlated
  shots.

Layout:

- `crates/aleph-oracle/src/harness.rs::run_distribution_oracle`
  is the entry point.
- `crates/aleph-oracle/build.rs` emits a `mod <stem> { #[test]
  fn state(); #[test] fn distribution(); }` per fixture, so a
  failure shows up as `<stem>::state` or `<stem>::distribution`
  — distinct enough that triage knows which check fired.

Tolerance derivation (spec §6.2): per-outcome flake probability
is ≤ 5.7e-7 at 5σ; per-fixture flake is ≤ 5.8e-4; per-CI-run
flake across 28 fixtures is ≤ 1.6%. With `seed = 0` pinned the
result is deterministic per machine, so the empirical flake rate
is in practice zero.
