# P4.6-05 — Noise models: Python/CLI surface + docs

**Issue:** #168 · **Milestone:** Phase 4.6 · **Depends on:** P4.6-04 (#167, `aleph_sv::noise`, merged `740c926`)
**Date:** 2026-06-14

## Goal

Expose the merged `aleph_sv::noise` Monte-Carlo quantum-jump engine through an
Aer-compatible Python API and a single-parameter CLI preset, per the P4.6-03
design spec §4. No new simulation logic: this ticket is a binding/translation
layer plus docs and tests.

## Background / constraints (from P4.6-04 handoff)

1. **Gate-name vocabulary.** `NoiseModel` keys on aleph's internal
   `Gate::name()` — `"H"`, `"X"`, `"Cnot"`, `"Phase"`, `"Toffoli"`, … — NOT the
   Aer/QASM mnemonics `"h"`, `"cx"`, `"p"`, `"ccx"`. The Python layer translates
   Aer mnemonic → aleph name at attach time so the Rust core's keys stay
   internal.
2. **No `id` gate** in aleph (the parser has no `id`, there is no `Gate`
   variant). Aer's idle-decoherence idiom `add_*_quantum_error(err, ["id"])` has
   no carrier here. **Decision: reject unknown gate names with a `ValueError`**
   (after Aer→aleph mapping) rather than silently no-op. This makes the v1
   "no idle-noise" limitation explicit at attach time.
3. **Constructors panic** on bad params (`p ∉ [0,1]`, `num_qubits ∉ {1,2}`,
   zero-sum `pauli_error`). The Python factories must validate ranges *before*
   calling the Rust constructor and raise `ValueError`, so a debug wheel never
   surfaces a `PanicException`.
4. **Run on the UN-optimized circuit.** `run_noisy` rejects optimizer artifacts
   (`TiledBlock`/`UnitaryKq`). `PyCircuit.inner` is already raw IR; the noise
   path must NOT call `run_optimized` (the noiseless SV path's optimizing
   driver).
5. **`Counts`** is a dense `Vec<u64>` of length `2ⁿ`, indexed by basis state
   (little-endian; qubit 0 is the LSB). Map to a bitstring→count dict using the
   same convention as the noiseless path (qubit 0 = rightmost character,
   ADR 0004).

## Architecture

| File | Change |
|------|--------|
| `crates/aleph-py/src/noise.rs` | **new** — `PyNoiseModel`, `PyQuantumError`, error factories, gate-name mapping |
| `crates/aleph-py/src/run.rs` | extend `run()` with a `noise=None` keyword; dispatch to `run_noisy` when set |
| `crates/aleph-py/src/lib.rs` | register `NoiseModel`, `QuantumError`, and the factory functions (flat, top-level namespace) |
| `crates/aleph-cli/src/cli.rs` | add `--noise <preset>:<p>` (repeatable) to the `Run` subcommand |
| `crates/aleph-cli/src/exec.rs` | build a `NoiseModel` from presets; dispatch to `run_noisy` |
| `scripts/python/test_aleph.py` | behavioural noise tests against the built wheel |
| `crates/aleph-cli/tests/cli.rs` | assert_cmd tests for `--noise` |
| `README.md`, `crates/aleph-py/README.md` | noise usage examples |

Each unit is independently testable: the gate-name map is a pure function; the
factories validate-then-wrap; `PyNoiseModel` is a thin attach/translate shell;
the `run()` dispatch and the histogram formatter are separable from both.

## Python API (flat namespace — `aleph.NoiseModel`, `aleph.depolarizing_error`)

```python
import aleph

nm = aleph.NoiseModel()
nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["h", "x"])
nm.add_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"], [0, 1])
nm.add_readout_error([[0.98, 0.02], [0.03, 0.97]], 0)

res = aleph.run(circ, shots=100_000, noise=nm, seed=7)
print(res.counts())   # {"00": ..., "11": ...}
```

### Error factories (return opaque `QuantumError`)

| Factory | Aer name | Validation (→ `ValueError`) |
|---------|----------|------------------------------|
| `depolarizing_error(p, num_qubits=1)` | `depolarizing_error` | `p ∈ [0,1]`, `num_qubits ∈ {1,2}` |
| `amplitude_damping_error(gamma)` | `amplitude_damping_error` | `gamma ∈ [0,1]` |
| `phase_damping_error(lam)` | `phase_damping_error` | `lam ∈ [0,1]` |
| `pauli_error(terms)` — `terms = list[(str, float)]` | `pauli_error` | labels ∈ {I,X,Y,Z}; Σweights > 0 |
| `bit_flip_error(p)` | (convenience) | `p ∈ [0,1]` |
| `phase_flip_error(p)` | (convenience) | `p ∈ [0,1]` |

Validation mirrors the Rust `assert!`s and runs in Python before the FFI call.

### `NoiseModel` methods

- `add_quantum_error(err, gates, qubits)` — `gates` is `str` or `list[str]`;
  `qubits` is `list[int]`. Each gate name is mapped Aer→aleph; unknown → `ValueError`.
- `add_all_qubit_quantum_error(err, gates)` — `gates` is `str` or `list[str]`.
- `add_readout_error(probs, qubit)` — `probs` is `[[P(0|0),P(1|0)],[P(0|1),P(1|1)]]`
  (Aer's per-qubit confusion matrix; `m[t][o] = P(o|t)`, rows sum to 1).
  `qubit` is an `int` (or `list[int]`, applied to each).

`PyQuantumError` is opaque (no introspection methods in v1) beyond an `arity`
read-only property for friendly error messages.

### Gate-name mapping (Aer mnemonic → aleph `Gate::name()`)

Pure function mirroring `aleph-parser/src/lower.rs` + `Gate::name()`:

```
h→H  x→X  y→Y  z→Z  s→S  sdg→Sdg  t→T  tdg→Tdg
rx→Rx  ry→Ry  rz→Rz  p→Phase  u3→U3  u→U3
cx→Cnot  cnot→Cnot  cz→Cz  swap→Swap  iswap→Iswap
crx→CRx  cry→CRy  crz→CRz  ccx→Toffoli  ccz→Ccz
```

The lookup is case-insensitive on the Aer side; an exact aleph internal name
(e.g. `"Cnot"`) is also accepted as a pass-through. Anything else — including
`"id"`/`"i"` — raises `ValueError` listing the supported names.

## Execution path: `aleph.run(circuit, *, shots, backend, seed, noise=None)`

- `noise=None` → existing noiseless path, unchanged.
- `noise=nm`:
  - **Backend guard:** `run_noisy` is SV-only. `backend="sv"` (the default) runs;
    an explicit `backend="mps"`/`"stab"` with `noise=` raises `ValueError`
    ("noise is only supported on the sv backend").
  - Call `aleph_sv::noise::run_noisy(&circuit.inner, &nm.inner, shots, seed)` on
    the **un-optimized** IR (no `run_optimized`).
  - `seed=None` → draw a random `u64` (run_noisy requires a concrete seed).
  - GIL released via `allow_threads` during the (potentially minutes-long) run.
  - Histogram (`Vec<u64>`, index = basis state) → bitstring dict: for each
    non-zero `hist[i]`, key `format!("{i:0width$b}")`, matching the noiseless
    formatter's qubit-0-rightmost convention.
  - `RunResult.amps = None`; `RunResult.statevector()` raises `ValueError`
    (trajectories produce a mixed state, not one vector).
  - `NoiseError::MidCircuit` (explicit `measure()`/`reset()` in the circuit) and
    `NoiseError::Unsupported` surface as `ValueError` with the core's message.

**Semantics note (documented):** unlike the noiseless `run()` (which samples one
collapsed final state `shots` times), the noisy path is per-shot Monte-Carlo —
each shot is an independent trajectory, matching Aer's execution model.

## CLI: `--noise <preset>:<p>` (repeatable) on `aleph run`

- `#[arg(long)] noise: Vec<String>` — repeatable, e.g.
  `aleph run c.qasm --noise depol:0.01 --noise readout:0.02`.
- Presets:
  - `depol:<p>` — scan the circuit's distinct 1q/2q gate names; attach
    `depolarizing_error(p, 1)` to each 1q name and `depolarizing_error(p, 2)` to
    each 2q name via `add_all_qubit_quantum_error`.
  - `readout:<p>` — attach symmetric `[[1-p, p], [p, 1-p]]` to every qubit.
- Parse errors (unknown preset, `p` not a float, `p ∉ [0,1]`) → non-zero exit
  with a clear message.
- **Backend guard:** `--noise` forces the state-vector backend. An explicit
  `--backend stabilizer`/`mps` with `--noise` is an error; `auto` resolves to
  state vector.
- **View guard (v1):** `--noise` is shots-only. Combining it with
  `--statevector`/`--force-statevector`/`--expectation` is an error (no single
  state vector exists under noise). `--seed` feeds `run_noisy`; absent → entropy.

## Testing

**Python (`scripts/python/test_aleph.py`, built wheel, seeded for determinism):**
- empty `NoiseModel` ≡ noiseless distribution (same seed).
- depolarizing on `h` over a 1-qubit `h` circuit pushes counts toward 50/50 →
  away from the noiseless near-deterministic outcome (qualitative, generous band).
- readout error visibly shifts the distribution.
- bad params (`depolarizing_error(1.5, 1)`, `num_qubits=3`, empty `pauli_error`)
  → `ValueError`.
- unknown gate name (`add_all_qubit_quantum_error(err, ["id"])`) → `ValueError`.
- `backend="mps"` with `noise=` → `ValueError`.
- `statevector()` on a noisy result → `ValueError`.

**CLI (`crates/aleph-cli/tests/cli.rs`, assert_cmd):**
- `--noise depol:0.05` changes counts vs. the same circuit without noise.
- `--noise readout:0.1` runs and prints counts.
- `--noise bad:x` / `--noise depol:2.0` → non-zero exit.
- `--backend stabilizer --noise depol:0.01` → non-zero exit.
- `--noise depol:0.01 --statevector` → non-zero exit.

These are behavioural/qualitative. The quantitative 1e-5 @ 100k Aer oracle is
already covered by P4.6-04's Rust-side `noise_oracle.rs`; this ticket does not
re-run it.

## Docs

- `README.md`: a short "Noise" subsection with the Python example above and the
  CLI preset one-liner.
- `crates/aleph-py/README.md`: the Python noise example.
- Release notes: there is no committed CHANGELOG (releases use GitHub release
  bodies — see P4-08). Add the noise feature to the next tag's GitHub release
  notes; no in-repo changelog file is created.

## Out of scope (v1)

- Mid-circuit measurement/reset under noise (`NoiseError::MidCircuit`).
- Idle/`id`-gate decoherence (no carrier gate — rejected at attach time).
- Frame-sampler routing for Clifford+Pauli noise (representation ready,
  separately scoped).
- Density-matrix backend; thermal-relaxation/Kraus beyond the v1 channel set.
- `QuantumError` introspection/serialization beyond `arity`.
```
