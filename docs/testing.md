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
  fn naive_state(); #[test] fn naive_distribution(); #[test]
  fn soa_state(); #[test] fn soa_distribution(); }` per fixture
  (P1-01 split this from the original `state` / `distribution`
  pair so both backends are exercised against every fixture).
  A failure surfaces as e.g. `<stem>::soa_distribution` — the
  prefix names the backend, the suffix names the check.

Tolerance derivation (spec §6.2): per-outcome flake probability
is ≤ 5.7e-7 at 5σ; per-fixture flake is ≤ 5.8e-4; per-CI-run
flake across 28 fixtures is ≤ 1.6%. With `seed = 0` pinned the
result is deterministic per machine, so the empirical flake rate
is in practice zero.

## Property-based testing (P0-05)

The workspace uses [proptest] for invariant testing. Shared
strategies live in the `aleph-test` crate (`crates/aleph-test/`),
consumed as a `[dev-dependencies]` entry by every crate that
needs them. No production code depends on `proptest`.

### Generators

| Strategy | Module | What it produces |
|---|---|---|
| `arb_state_vector(n)` | `aleph_test::state` | Normalised `Vec<Complex>` of length `2^n` |
| `arb_1q_gate()` / `arb_2q_gate()` / `arb_gate()` | `aleph_test::gate` | Random `Gate` |
| `arb_diagonal_1q_gate()` | `aleph_test::gate` | Z / S / Sdg / T / Tdg / Rz only |
| `arb_circuit_emittable(nq, nc, n_ops)` | `aleph_test::circuit` | Emitter-supported `Circuit` (parser tests) |
| `arb_circuit_full(nq, nc, n_ops)` | `aleph_test::circuit` | Broader-vocabulary `Circuit` (IR layer tests) |
| `arb_op_emittable(nq, nc)` / `arb_op_full(nq, nc)` | `aleph_test::circuit` | Single random `OpKind` |
| `arb_pauli_string(n, mix_xy)` | `aleph_test::pauli` | `PauliString` |
| `distinct_pair(nq)` / `distinct_triple(nq)` | `aleph_test::circuit` | Raw qubit-tuple helpers |

### Invariants exercised

| Invariant | Where |
|---|---|
| Norm preservation after any gate | `aleph-sv/src/backend.rs::tests::normalisation_invariant` |
| Reversibility (`G†·G·ψ = ψ`) | 10+ proptests in `aleph-sv/src/backend.rs` (`*_then_*_negative_returns_identity`, `*_squared_is_identity`) |
| Diagonal gates leave \|aᵢ\| invariant | `aleph-sv/src/backend.rs::tests::diagonal_gate_preserves_magnitudes` |
| Σ P(outcome) = 1 over full basis | `aleph-sv/src/measure.rs::tests::probabilities_full_basis_sums_to_one` |
| Z fast path ≡ slow path (Z-only Pauli) | `aleph-sv/src/measure.rs::tests::z_fast_path_matches_slow_path` |
| Parser ↔ emitter round-trip | `aleph-parser/tests/round_trip_property.rs::parse_emit_roundtrip` |
| IR layer partitioning correctness | `aleph-ir/tests/layers_properties.rs` (4 proptests) |
| f64 round-trip through serde_json | `aleph-oracle/src/fixture.rs::tests::f64_pair_round_trips_through_serde_json` |
| Pauli arg parser ↔ Display | `aleph-cli/src/pauli.rs::tests::z_only_round_trip` |

### Failure persistence

proptest writes shrunk failure seeds to
`<crate>/proptest-regressions/*.txt`. **Commit these files** —
they replay historical failure cases on every future run,
preventing regression of bugs the suite previously caught.

### Adding a property test

1. Pick or compose a strategy from `aleph_test::*`.
2. Inside a `proptest! { #[test] fn ... { ... } }` block, assert
   the invariant with `prop_assert!` (not plain `assert!` — the
   former shrinks).
3. Default `ProptestConfig::default()` (256 cases) is fine for
   most tests; bump `cases: N` for expensive setups.

[proptest]: https://github.com/proptest-rs/proptest

## SoA backend (P1-01)

`aleph-sv` ships two state-vector backends:

* `NaiveSvBackend` — reference, array-of-structs (`Vec<Complex<f64>>`).
  Stays as the correctness yardstick.
* `SoaSvBackend` — Phase-1 perf backend, struct-of-arrays
  (`Vec<f64>` × 2). Same algorithms, layout chosen for SIMD-friendly
  sequential memory access (P1-03 / P1-04 add the explicit AVX2 /
  AVX-512 vectorisation).

Equivalence is pinned three ways:

1. `crates/aleph-oracle/tests/soa_vs_naive.rs::all_fixtures_match_naive`
   — every committed oracle circuit produces the same state vector
   on both backends within 1e-12 per amplitude.
2. Proptest equivalence in `crates/aleph-sv/src/kernels/soa.rs`
   over `aleph_test::gate::arb_1q_gate` / `arb_2q_gate` against
   `aleph_test::state::arb_state_vector`.
3. Both backends pass the full oracle suite vs Qiskit Aer. The
   `aleph-oracle/build.rs` codegen emits `naive_state`,
   `naive_distribution`, `soa_state`, `soa_distribution` per
   fixture (28 × 4 = 112 generated tests).

When introducing a new state-vector backend (e.g. SIMD-specialised
variants in P1-03), add it to the workhorse equivalence test
+ `build.rs` codegen rather than relying on the oracle alone.
