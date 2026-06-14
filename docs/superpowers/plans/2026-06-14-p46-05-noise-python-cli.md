# P4.6-05 Noise Python/CLI Surface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the merged `aleph_sv::noise` engine through an Aer-compatible Python API (`aleph.NoiseModel` + error factories + `run(noise=)`) and an `aleph run --noise <preset>:<p>` CLI preset, with tests and docs.

**Architecture:** Pure binding/translation layer — no new simulation logic. A new `aleph-py` module translates Aer gate mnemonics to aleph's internal `Gate::name()`, validates factory params before the panicking Rust constructors, and routes `run(noise=)` to `run_noisy` on the un-optimized circuit. The CLI builds a `NoiseModel` from one-parameter presets by scanning the circuit's gate names.

**Tech Stack:** Rust, PyO3 0.22 (`extension-module`, `abi3-py312`), maturin, `rand` (workspace), clap, `assert_cmd`, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-06-14-p46-05-noise-python-cli-design.md`

---

## Important context for the implementer

- **Cannot `cargo test` aleph-py.** The `extension-module` PyO3 feature does not link libpython, so building a test binary fails. The Rust binding code is verified with `cargo build -p aleph-py --features python` + clippy; behaviour is verified by building the wheel (`maturin develop`) and running `scripts/python/test_aleph.py`. There is no Rust-side unit test in aleph-py beyond the existing trivial `crate_loads`.
- **PyO3 0.22 idioms:** use `Bound<'_, PyAny>` for polymorphic args, `wrap_pyfunction!`, `m.add_class::<T>()`. The `#![allow(clippy::useless_conversion)]` at the top of binding modules suppresses pyo3-macro false positives — include it in the new module.
- **Gate-name truth:** `aleph_core::Gate::name()` returns `"H"`, `"Cnot"`, `"Phase"`, `"Toffoli"`, etc. (`crates/aleph-core/src/gate/kinds.rs:148`). The Aer mnemonics are in `crates/aleph-parser/src/lower.rs:257`.
- **`run_noisy` signature:** `aleph_sv::noise::run_noisy(&Circuit, &NoiseModel, shots: u32, seed: u64) -> Result<Counts, NoiseError>`, where `Counts = Vec<u64>` is a dense histogram indexed by basis state (qubit 0 = LSB). `NoiseError` is `Display` (thiserror).
- **No worktrees** (CLAUDE.md). Work on branch `p46-05-noise-python-cli` (already created and checked out).
- After each task, run `cargo fmt` before committing.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/aleph-py/Cargo.toml` | add optional `rand` dep, gated behind `python` feature |
| `crates/aleph-py/src/noise.rs` | **new** — gate-name map, `PyQuantumError`, factories, `PyNoiseModel` |
| `crates/aleph-py/src/run.rs` | `run(noise=)` dispatch + `hist_to_counts` |
| `crates/aleph-py/src/lib.rs` | register the new module's classes/functions |
| `crates/aleph-cli/Cargo.toml` | add `rand` dep |
| `crates/aleph-cli/src/cli.rs` | `--noise` arg on `Run` |
| `crates/aleph-cli/src/exec.rs` | preset → `NoiseModel`, dispatch to `run_noisy` |
| `crates/aleph-cli/src/output.rs` | `format_counts_hist` |
| `crates/aleph-cli/src/main.rs` | thread `noise` through the `Run` match |
| `scripts/python/test_aleph.py` | `TestNoise` behavioural tests |
| `crates/aleph-cli/tests/cli.rs` | `--noise` assert_cmd tests |
| `README.md`, `crates/aleph-py/README.md` | noise examples |

---

## Task 1: aleph-py noise module — factories + gate-name map

**Files:**
- Modify: `crates/aleph-py/Cargo.toml`
- Create: `crates/aleph-py/src/noise.rs`
- Modify: `crates/aleph-py/src/lib.rs`

- [ ] **Step 1: Add the `rand` dependency, gated behind the `python` feature**

In `crates/aleph-py/Cargo.toml`, change the `[features]` block's `python` line and add `rand` to `[dependencies]`:

```toml
[features]
# Enables the PyO3 extension module. OFF by default so `cargo build --workspace`
# never needs a Python interpreter; `maturin develop --features python` turns it on.
python = ["dep:pyo3", "dep:rand"]
```

Add to `[dependencies]` (after the `pyo3` line):

```toml
# Only used by the noise run path (run.rs) to draw a u64 seed when the caller
# passes seed=None; pulled in only with the `python` feature.
rand = { workspace = true, optional = true }
```

- [ ] **Step 2: Create `crates/aleph-py/src/noise.rs` with the gate-name map, `PyQuantumError`, and factories**

```rust
//! Aer-compatible noise surface for Python.
//!
//! Wraps `aleph_sv::noise`: error factories validate their params *before*
//! the panicking Rust constructors (raising `ValueError`, never a
//! `PanicException`), and `NoiseModel` translates Aer gate mnemonics
//! (`"h"`, `"cx"`) to aleph's internal `Gate::name()` (`"H"`, `"Cnot"`) at
//! attach time so the Rust core's keys stay internal. Unknown gate names —
//! including Aer's idle `"id"`, which aleph has no carrier for — are rejected.
// pyo3 0.22 proc-macro expansion emits trivial PyErr→PyErr `.into()` calls.
#![allow(clippy::useless_conversion)]

use aleph_sv::noise::{self as sv_noise, NoiseModel, QuantumError, ReadoutError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Map an Aer/QASM gate mnemonic (case-insensitive) — or an exact aleph
/// internal name — to aleph's `Gate::name()`. Mirrors
/// `aleph-parser/src/lower.rs` + `Gate::name()`. Returns `None` for unknown
/// names (notably `"id"`/`"i"`: aleph has no idle gate).
fn aer_to_aleph(name: &str) -> Option<&'static str> {
    // Lowercasing makes this accept both the Aer mnemonic ("cx") and the
    // lowercased aleph name ("Cnot" -> "cnot"); every aleph 1q/2q/3q gate
    // name lowercases onto a key below.
    Some(match name.to_ascii_lowercase().as_str() {
        "h" => "H",
        "x" => "X",
        "y" => "Y",
        "z" => "Z",
        "s" => "S",
        "sdg" => "Sdg",
        "t" => "T",
        "tdg" => "Tdg",
        "rx" => "Rx",
        "ry" => "Ry",
        "rz" => "Rz",
        "p" | "phase" => "Phase",
        "u3" | "u" => "U3",
        "cx" | "cnot" => "Cnot",
        "cz" => "Cz",
        "swap" => "Swap",
        "iswap" => "Iswap",
        "crx" => "CRx",
        "cry" => "CRy",
        "crz" => "CRz",
        "ccx" | "toffoli" => "Toffoli",
        "ccz" => "Ccz",
        _ => return None,
    })
}

/// Supported names, for error messages.
const SUPPORTED_GATES: &str =
    "h,x,y,z,s,sdg,t,tdg,rx,ry,rz,p,u3,cx,cz,swap,iswap,crx,cry,crz,ccx,ccz";

/// Translate a list of Aer mnemonics to aleph internal names, erroring on the
/// first unknown one.
fn map_names(names: &[String]) -> PyResult<Vec<&'static str>> {
    names
        .iter()
        .map(|n| {
            aer_to_aleph(n).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown gate name {n:?}; supported: {SUPPORTED_GATES} \
                     (aleph has no 'id'/idle gate)"
                ))
            })
        })
        .collect()
}

/// `gates` accepts either a single `str` or a `list[str]`.
fn extract_gates(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(vec![s]);
    }
    obj.extract::<Vec<String>>()
        .map_err(|_| PyValueError::new_err("gates must be a str or list[str]"))
}

/// `qubit` accepts either a single `int` or a `list[int]`.
fn extract_qubits(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(q) = obj.extract::<u32>() {
        return Ok(vec![q]);
    }
    obj.extract::<Vec<u32>>()
        .map_err(|_| PyValueError::new_err("qubit must be an int or list[int]"))
}

/// Opaque CPTP error channel (the result of an error factory).
#[pyclass(name = "QuantumError")]
#[derive(Clone)]
pub(crate) struct PyQuantumError {
    pub(crate) inner: QuantumError,
}

#[pymethods]
impl PyQuantumError {
    /// Number of qubits this channel acts on (1 or 2).
    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }
}

/// Depolarizing channel on `num_qubits` (1 or 2) with total error prob `p`.
#[pyfunction]
#[pyo3(signature = (p, num_qubits = 1))]
pub(crate) fn depolarizing_error(p: f64, num_qubits: u8) -> PyResult<PyQuantumError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "depolarizing p must be in [0,1], got {p}"
        )));
    }
    if num_qubits != 1 && num_qubits != 2 {
        return Err(PyValueError::new_err(format!(
            "depolarizing num_qubits must be 1 or 2, got {num_qubits}"
        )));
    }
    Ok(PyQuantumError {
        inner: sv_noise::depolarizing_error(p, num_qubits),
    })
}

/// Amplitude damping channel with decay `gamma` (0..=1).
#[pyfunction]
pub(crate) fn amplitude_damping_error(gamma: f64) -> PyResult<PyQuantumError> {
    if !(0.0..=1.0).contains(&gamma) {
        return Err(PyValueError::new_err(format!(
            "gamma must be in [0,1], got {gamma}"
        )));
    }
    Ok(PyQuantumError {
        inner: sv_noise::amplitude_damping_error(gamma),
    })
}

/// Phase damping (dephasing) channel with `lam` (0..=1).
#[pyfunction]
pub(crate) fn phase_damping_error(lam: f64) -> PyResult<PyQuantumError> {
    if !(0.0..=1.0).contains(&lam) {
        return Err(PyValueError::new_err(format!(
            "lambda must be in [0,1], got {lam}"
        )));
    }
    Ok(PyQuantumError {
        inner: sv_noise::phase_damping_error(lam),
    })
}

/// Bit-flip channel: X with probability `p`, identity otherwise.
#[pyfunction]
pub(crate) fn bit_flip_error(p: f64) -> PyResult<PyQuantumError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "flip p must be in [0,1], got {p}"
        )));
    }
    Ok(PyQuantumError {
        inner: sv_noise::bit_flip_error(p),
    })
}

/// Phase-flip channel: Z with probability `p`, identity otherwise.
#[pyfunction]
pub(crate) fn phase_flip_error(p: f64) -> PyResult<PyQuantumError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "flip p must be in [0,1], got {p}"
        )));
    }
    Ok(PyQuantumError {
        inner: sv_noise::phase_flip_error(p),
    })
}

/// Single-qubit Pauli channel from `[(label, prob), ...]`; labels are
/// "I"/"X"/"Y"/"Z". Weights are renormalized. 1q only — for 2-qubit Pauli
/// noise use `depolarizing_error(p, 2)`.
#[pyfunction]
pub(crate) fn pauli_error(terms: Vec<(String, f64)>) -> PyResult<PyQuantumError> {
    if terms.is_empty() {
        return Err(PyValueError::new_err("pauli_error needs at least one term"));
    }
    let mut total = 0.0;
    for (label, prob) in &terms {
        if !matches!(label.as_str(), "I" | "X" | "Y" | "Z") {
            return Err(PyValueError::new_err(format!(
                "pauli_error label {label:?} must be one of I,X,Y,Z (1q only); \
                 use depolarizing_error(p, 2) for 2-qubit noise"
            )));
        }
        if !prob.is_finite() || *prob < 0.0 {
            return Err(PyValueError::new_err(format!(
                "pauli_error weight for {label:?} must be finite and >= 0, got {prob}"
            )));
        }
        total += *prob;
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "pauli_error weights must sum to > 0",
        ));
    }
    let refs: Vec<(&str, f64)> = terms.iter().map(|(s, p)| (s.as_str(), *p)).collect();
    Ok(PyQuantumError {
        inner: sv_noise::pauli_error(&refs),
    })
}

/// Aer-style attachment of channels to gates/qubits, consumed by `run(noise=)`.
#[pyclass(name = "NoiseModel")]
pub(crate) struct PyNoiseModel {
    pub(crate) inner: NoiseModel,
}

#[pymethods]
impl PyNoiseModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: NoiseModel::new(),
        }
    }

    /// Attach `err` to `gates` (str or list[str]) on the specific `qubits`
    /// tuple. Gate names are Aer mnemonics, translated to aleph names.
    fn add_quantum_error(
        &mut self,
        err: &PyQuantumError,
        gates: &Bound<'_, PyAny>,
        qubits: Vec<u32>,
    ) -> PyResult<()> {
        let names = map_names(&extract_gates(gates)?)?;
        self.inner
            .add_quantum_error(err.inner.clone(), &names, &qubits);
        Ok(())
    }

    /// Attach `err` to `gates` (str or list[str]) on whichever qubits each
    /// gate acts on.
    fn add_all_qubit_quantum_error(
        &mut self,
        err: &PyQuantumError,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let names = map_names(&extract_gates(gates)?)?;
        self.inner
            .add_all_qubit_quantum_error(err.inner.clone(), &names);
        Ok(())
    }

    /// Attach a per-qubit readout error. `probs` is the 2×2 confusion matrix
    /// `[[P(0|0), P(1|0)], [P(0|1), P(1|1)]]`; `qubit` is an int or list[int].
    fn add_readout_error(
        &mut self,
        probs: Vec<Vec<f64>>,
        qubit: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if probs.len() != 2 || probs[0].len() != 2 || probs[1].len() != 2 {
            return Err(PyValueError::new_err(
                "readout probs must be a 2x2 matrix [[P(0|0),P(1|0)],[P(0|1),P(1|1)]]",
            ));
        }
        for (r, row) in probs.iter().enumerate() {
            for v in row {
                if !(0.0..=1.0).contains(v) {
                    return Err(PyValueError::new_err(format!(
                        "readout probability {v} (row {r}) must be in [0,1]"
                    )));
                }
            }
            let sum = row[0] + row[1];
            if (sum - 1.0).abs() > 1e-9 {
                return Err(PyValueError::new_err(format!(
                    "readout row {r} must sum to 1, got {sum}"
                )));
            }
        }
        let re = ReadoutError::new([[probs[0][0], probs[0][1]], [probs[1][0], probs[1][1]]]);
        for q in extract_qubits(qubit)? {
            self.inner.add_readout_error(re, q);
        }
        Ok(())
    }
}
```

- [ ] **Step 3: Register the module in `crates/aleph-py/src/lib.rs`**

Add the module declaration after the `run` module (around line 15):

```rust
#[cfg(feature = "python")]
mod noise;
```

Inside `fn aleph(...)`, after the `run_circuit` registration (line 38), add:

```rust
        m.add_class::<crate::noise::PyNoiseModel>()?;
        m.add_class::<crate::noise::PyQuantumError>()?;
        m.add_function(wrap_pyfunction!(crate::noise::depolarizing_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::amplitude_damping_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::phase_damping_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::bit_flip_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::phase_flip_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::pauli_error, m)?)?;
```

- [ ] **Step 4: Verify it compiles and lints**

Run: `cargo build -p aleph-py --features python`
Expected: builds clean.

Run: `cargo clippy -p aleph-py --features python --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add crates/aleph-py/Cargo.toml crates/aleph-py/src/noise.rs crates/aleph-py/src/lib.rs
git commit -m "[P4.6-05] aleph-py noise module: factories + NoiseModel + gate-name map"
```

---

## Task 2: `run(noise=)` dispatch in aleph-py

**Files:**
- Modify: `crates/aleph-py/src/run.rs`

- [ ] **Step 1: Add the `hist_to_counts` helper**

In `crates/aleph-py/src/run.rs`, add after the existing `counts_map` function (around line 71):

```rust
/// Format a dense `run_noisy` histogram (index = basis state, qubit 0 = LSB)
/// into the same bitstring→count dict as `counts_map`, skipping zero bins.
fn hist_to_counts(hist: &[u64], num_qubits: u32) -> BTreeMap<String, u64> {
    let width = num_qubits as usize;
    hist.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (format!("{i:0width$b}"), c))
        .collect()
}
```

- [ ] **Step 2: Add the `noise` keyword to `run_circuit` and dispatch**

Change the import line 13 to also bring in the noise model:

```rust
use crate::circuit::PyCircuit;
use crate::noise::PyNoiseModel;
```

Replace the `#[pyfunction]` attribute + signature (lines 83-91) with:

```rust
#[pyfunction]
#[pyo3(name = "run", signature = (circuit, *, shots = 1024, backend = "sv", seed = None, noise = None))]
pub(crate) fn run_circuit(
    py: Python<'_>,
    circuit: &PyCircuit,
    shots: u32,
    backend: &str,
    seed: Option<u64>,
    noise: Option<&PyNoiseModel>,
) -> PyResult<RunResult> {
    let c = &circuit.inner;
    let n = c.num_qubits();

    // Noisy path: Monte-Carlo trajectories via run_noisy on the UN-optimized
    // circuit (the optimizer emits TiledBlock/UnitaryKq that run_noisy rejects).
    // SV-only; unlike the noiseless path, each shot is an independent trajectory.
    if let Some(nm) = noise {
        if backend != "sv" {
            return Err(PyValueError::new_err(format!(
                "noise is only supported on the \"sv\" backend, got {backend:?}"
            )));
        }
        let model = &nm.inner;
        let seed = seed.unwrap_or_else(rand::random::<u64>);
        let counts = py.allow_threads(|| -> PyResult<BTreeMap<String, u64>> {
            let hist = aleph_sv::noise::run_noisy(c, model, shots, seed)
                .map_err(|e| err("run noisy", e))?;
            Ok(hist_to_counts(&hist, n))
        })?;
        return Ok(RunResult { counts, amps: None });
    }
```

Leave the rest of the function (the `match backend { "sv" => ... }` block) unchanged — it now follows this early-return block. Remove the now-duplicate `let c = &circuit.inner;` / `let n = c.num_qubits();` lines that previously opened the function body (they are moved above the noise block).

- [ ] **Step 3: Verify build + clippy**

Run: `cargo build -p aleph-py --features python`
Expected: builds clean.

Run: `cargo clippy -p aleph-py --features python --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 4: Commit**

```bash
cargo fmt
git add crates/aleph-py/src/run.rs
git commit -m "[P4.6-05] aleph-py: run(noise=) dispatch to run_noisy"
```

---

## Task 3: Python behaviour tests (wheel)

**Files:**
- Modify: `scripts/python/test_aleph.py`

- [ ] **Step 1: Append a `TestNoise` class**

Add at the end of `scripts/python/test_aleph.py`:

```python
@unittest.skipUnless(HAVE_ALEPH, "aleph extension module not installed")
class TestNoise(unittest.TestCase):
    def _bell(self):
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        return c

    def test_empty_model_matches_noiseless(self):
        # An empty NoiseModel must reproduce the noiseless distribution
        # (same seed). run() with noise= is per-shot Monte-Carlo, but with
        # no channels every trajectory equals the clean run.
        nm = aleph.NoiseModel()
        noisy = aleph.run(self._bell(), shots=4096, noise=nm, seed=11).counts()
        self.assertEqual(set(noisy), {"00", "11"})
        self.assertEqual(sum(noisy.values()), 4096)

    def test_depolarizing_on_h_spreads_distribution(self):
        # Strong depolarizing on H injects X/Y/Z errors, so a circuit that is
        # otherwise |0> on qubit 0 picks up "1" outcomes.
        c = aleph.Circuit(1)
        c.x(0)
        c.x(0)  # net identity -> noiseless is all "0"
        nm = aleph.NoiseModel()
        nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.5, 1), ["x"])
        counts = aleph.run(c, shots=8000, noise=nm, seed=3).counts()
        # With p=0.5 depolarizing applied twice, a non-trivial fraction flips.
        self.assertIn("1", counts)
        self.assertGreater(counts.get("1", 0), 200)

    def test_readout_error_flips_outcomes(self):
        # A near-certain |00> state with heavy readout error produces "1"s.
        c = aleph.Circuit(2)  # noiseless -> all "00"
        nm = aleph.NoiseModel()
        nm.add_readout_error([[0.7, 0.3], [0.3, 0.7]], 0)
        counts = aleph.run(c, shots=8000, noise=nm, seed=5).counts()
        flipped = sum(v for k, v in counts.items() if k.endswith("1"))
        self.assertGreater(flipped, 1500)  # ~0.3 of 8000, generous slack

    def test_bad_params_raise(self):
        with self.assertRaises(ValueError):
            aleph.depolarizing_error(1.5, 1)
        with self.assertRaises(ValueError):
            aleph.depolarizing_error(0.1, 3)
        with self.assertRaises(ValueError):
            aleph.amplitude_damping_error(-0.1)
        with self.assertRaises(ValueError):
            aleph.pauli_error([])

    def test_unknown_gate_name_raises(self):
        # aleph has no idle "id" gate; attaching to it is an explicit error.
        nm = aleph.NoiseModel()
        with self.assertRaises(ValueError):
            nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["id"])

    def test_aer_mnemonic_maps_to_internal_name(self):
        # "cx" must reach the engine as "Cnot": attach 2q depol to cx and run
        # a Bell circuit without error.
        nm = aleph.NoiseModel()
        nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"])
        counts = aleph.run(self._bell(), shots=2048, noise=nm, seed=9).counts()
        self.assertEqual(sum(counts.values()), 2048)

    def test_noise_rejects_non_sv_backend(self):
        nm = aleph.NoiseModel()
        with self.assertRaises(ValueError):
            aleph.run(self._bell(), shots=64, backend="mps", noise=nm, seed=1)

    def test_noisy_result_has_no_statevector(self):
        nm = aleph.NoiseModel()
        res = aleph.run(self._bell(), shots=64, noise=nm, seed=1)
        with self.assertRaises(ValueError):
            res.statevector()
```

- [ ] **Step 2: Build the wheel into a venv**

Run from repo root:

```bash
cd crates/aleph-py && maturin develop --release --features python && cd ../..
```

Expected: `🛠 Installed aleph-...`. (Requires an active venv with maturin; the release gate runs this manually — see the test file's docstring.)

- [ ] **Step 3: Run the noise tests, expect PASS**

Run: `python -m unittest scripts.python.test_aleph -v -k Noise` (or `python -m unittest discover -s scripts/python -v`)
Expected: all `TestNoise` tests PASS; the rest of the suite still passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/python/test_aleph.py
git commit -m "[P4.6-05] aleph-py: noise behaviour tests"
```

---

## Task 4: CLI `--noise <preset>:<p>`

**Files:**
- Modify: `crates/aleph-cli/Cargo.toml`
- Modify: `crates/aleph-cli/src/cli.rs:84-137` (the `Run` variant)
- Modify: `crates/aleph-cli/src/output.rs`
- Modify: `crates/aleph-cli/src/exec.rs`
- Modify: `crates/aleph-cli/src/main.rs:18-40`

- [ ] **Step 1: Add `rand` to aleph-cli deps**

In `crates/aleph-cli/Cargo.toml`, add to `[dependencies]` (after `thiserror`):

```toml
rand          = { workspace = true }
```

- [ ] **Step 2: Add the `--noise` arg to the `Run` subcommand**

In `crates/aleph-cli/src/cli.rs`, inside the `Run { ... }` variant, add after the `max_error` field (line 136):

```rust
        /// Apply a built-in noise preset, repeatable. Format `<preset>:<p>`
        /// with `p` in [0,1]. Presets: `depol:<p>` (depolarizing on every 1q
        /// and 2q gate in the circuit) and `readout:<p>` (symmetric readout
        /// flip on every qubit). Forces the state-vector backend; cannot be
        /// combined with --statevector or --expectation. Full NoiseModel
        /// construction is available in the Python API.
        #[arg(long)]
        noise: Vec<String>,
```

- [ ] **Step 3: Add `format_counts_hist` to output.rs**

In `crates/aleph-cli/src/output.rs`, add after `format_counts` (after its closing brace, around line 42):

```rust
/// Like [`format_counts`] but for a dense histogram (index = basis state,
/// qubit 0 = LSB) as returned by `aleph_sv::noise::run_noisy`. Skips empty
/// bins; identical line format to `format_counts`.
pub fn format_counts_hist<W: Write>(
    out: &mut W,
    hist: &[u64],
    total: u32,
    num_qubits: u32,
    seed_label: &str,
) -> io::Result<()> {
    let width = num_qubits as usize;
    writeln!(out, "counts ({total} shots, {seed_label}):")?;
    let total_f = total as f64;
    for (idx, count) in hist.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        let prob = *count as f64 / total_f;
        writeln!(out, "  |{idx:0width$b}⟩  {count}  ({prob:.4})", width = width)?;
    }
    Ok(())
}
```

- [ ] **Step 4: Build the NoiseModel and dispatch in exec.rs**

In `crates/aleph-cli/src/exec.rs`, add `noise: &[String]` to the `run_circuit` signature (after `max_error: Option<f64>,`, before `out`):

```rust
    max_error: Option<f64>,
    noise: &[String],
    out: &mut W,
```

Immediately after `let n = circuit.num_qubits();` (line 70), insert the noise branch:

```rust
    // Noise preset path: build a NoiseModel from --noise presets and run the
    // Monte-Carlo trajectory engine. SV-only, shots-only (no statevector /
    // expectation view under a mixed state). Returns before the normal
    // backend dispatch.
    if !noise.is_empty() {
        if matches!(backend, BackendChoice::Stabilizer | BackendChoice::Mps) {
            return Err(anyhow!(
                "--noise is only supported on the state-vector backend; \
                 remove --backend {backend:?}"
            ));
        }
        if print_statevector || force_statevector || !expectations.is_empty() {
            return Err(anyhow!(
                "--noise is shots-only in v1; it cannot be combined with \
                 --statevector or --expectation"
            ));
        }
        let model = build_noise_model(noise, &circuit, n)?;
        let shots = shots_opt.unwrap_or(DEFAULT_SHOTS);
        let run_seed = seed.unwrap_or_else(rand::random::<u64>);
        let seed_label = match seed {
            Some(s) => format!("seed={s}"),
            None => "seed=entropy".to_string(),
        };
        let hist = aleph_sv::noise::run_noisy(&circuit, &model, shots, run_seed)
            .context("running noisy circuit")?;
        output::format_counts_hist(out, &hist, shots, n, &seed_label)?;
        return Ok(());
    }
```

Add the `build_noise_model` helper near the other private helpers (e.g. after `run_with_backend`, or at the end of the file before `#[cfg(test)]`):

```rust
/// Build a `NoiseModel` from `--noise <preset>:<p>` strings. `depol:<p>`
/// attaches depolarizing error to every distinct 1q and 2q gate name present
/// in `circuit`; `readout:<p>` attaches a symmetric readout flip to every
/// qubit. 3q+ gates get no preset noise (depolarizing_error supports 1q/2q).
fn build_noise_model(
    presets: &[String],
    circuit: &aleph_ir::Circuit,
    n: u32,
) -> Result<aleph_sv::noise::NoiseModel> {
    use aleph_sv::noise::{depolarizing_error, NoiseModel, ReadoutError};

    let mut nm = NoiseModel::new();
    for raw in presets {
        let (kind, val) = raw
            .split_once(':')
            .ok_or_else(|| anyhow!("--noise expects <preset>:<p>, got {raw:?}"))?;
        let p: f64 = val
            .parse()
            .map_err(|_| anyhow!("--noise {raw:?}: {val:?} is not a number"))?;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(anyhow!("--noise {raw:?}: p must be in [0,1], got {p}"));
        }
        match kind {
            "depol" => {
                // Distinct gate name -> arity for the 1q/2q gates present.
                // Gate::name() is &'static, so this borrows nothing from the
                // circuit.
                let mut seen: std::collections::BTreeMap<&'static str, u8> =
                    std::collections::BTreeMap::new();
                for inst in circuit.instructions() {
                    if let aleph_ir::Instruction::Gate(gi) = inst {
                        let arity = gi.qubits.len();
                        if arity == 1 || arity == 2 {
                            seen.insert(gi.gate.name(), arity as u8);
                        }
                    }
                }
                for (name, arity) in seen {
                    nm.add_all_qubit_quantum_error(depolarizing_error(p, arity), &[name]);
                }
            }
            "readout" => {
                let re = ReadoutError::new([[1.0 - p, p], [p, 1.0 - p]]);
                for q in 0..n {
                    nm.add_readout_error(re, q);
                }
            }
            other => {
                return Err(anyhow!(
                    "unknown --noise preset {other:?} (expected depol or readout)"
                ))
            }
        }
    }
    Ok(nm)
}
```

- [ ] **Step 5: Thread `noise` through main.rs**

In `crates/aleph-cli/src/main.rs`, add `noise,` to the `Cmd::Run { ... }` destructure (after `max_error,`) and pass `&noise` to `run_circuit` (after `max_error,`, before `&mut out`):

```rust
        Cmd::Run {
            qasm,
            shots,
            statevector,
            force_statevector,
            expectation,
            seed,
            precision,
            backend,
            max_bond,
            max_error,
            noise,
        } => run_circuit(
            &qasm,
            shots,
            statevector,
            force_statevector,
            &expectation,
            seed,
            precision,
            backend,
            max_bond,
            max_error,
            &noise,
            &mut out,
        )?,
```

- [ ] **Step 6: Verify build + clippy**

Run: `cargo build -p aleph-cli`
Expected: builds clean.

Run: `cargo clippy -p aleph-cli --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
cargo fmt
git add crates/aleph-cli/Cargo.toml crates/aleph-cli/src/cli.rs crates/aleph-cli/src/output.rs crates/aleph-cli/src/exec.rs crates/aleph-cli/src/main.rs
git commit -m "[P4.6-05] CLI: --noise <preset>:<p> depolarizing/readout presets"
```

---

## Task 5: CLI assert_cmd tests

**Files:**
- Modify: `crates/aleph-cli/tests/cli.rs`

- [ ] **Step 1: Write the failing tests**

Append to `crates/aleph-cli/tests/cli.rs`:

```rust
#[test]
fn noise_depol_runs_and_prints_counts() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--shots", "512", "--seed", "0", "--noise", "depol:0.05"])
        .assert()
        .success()
        .stdout(contains("counts (512 shots, seed=0):"));
}

#[test]
fn noise_readout_runs() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--shots", "256", "--seed", "1", "--noise", "readout:0.1"])
        .assert()
        .success()
        .stdout(contains("counts (256 shots, seed=1):"));
}

#[test]
fn noise_bad_value_fails() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--noise", "depol:x"])
        .assert()
        .failure()
        .stderr(contains("is not a number"));
}

#[test]
fn noise_out_of_range_fails() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--noise", "depol:2.0"])
        .assert()
        .failure()
        .stderr(contains("p must be in [0,1]"));
}

#[test]
fn noise_unknown_preset_fails() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--noise", "bogus:0.1"])
        .assert()
        .failure()
        .stderr(contains("unknown --noise preset"));
}

#[test]
fn noise_rejects_stabilizer_backend() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--backend", "stabilizer", "--noise", "depol:0.01"])
        .assert()
        .failure()
        .stderr(contains("only supported on the state-vector backend"));
}

#[test]
fn noise_rejects_statevector_view() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--noise", "depol:0.01", "--statevector"])
        .assert()
        .failure()
        .stderr(contains("shots-only"));
}
```

- [ ] **Step 2: Run, expect PASS**

Run: `cargo test -p aleph-cli --test cli noise`
Expected: all seven `noise_*` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-cli/tests/cli.rs
git commit -m "[P4.6-05] CLI: assert_cmd tests for --noise"
```

---

## Task 6: Docs + BACKLOG acceptance boxes

**Files:**
- Modify: `README.md`
- Modify: `crates/aleph-py/README.md`
- Modify: `BACKLOG.md:2818-2820` (tick the P4.6-05 AC boxes)

- [ ] **Step 1: Add a Noise section to the root README**

Find the Python usage / CLI section in `README.md` and add a "Noise simulation" subsection. Use this content (adjust heading level to match surrounding sections):

````markdown
### Noise simulation

Aer-compatible noise via a `NoiseModel` (Python) or a one-parameter CLI preset.

```python
import aleph

c = aleph.Circuit(2)
c.h(0); c.cx(0, 1)

nm = aleph.NoiseModel()
nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["h"])
nm.add_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"], [0, 1])
nm.add_readout_error([[0.98, 0.02], [0.03, 0.97]], 0)

print(aleph.run(c, shots=100_000, noise=nm, seed=7).counts())
```

Error factories mirror Qiskit Aer names: `depolarizing_error(p, num_qubits)`,
`amplitude_damping_error(gamma)`, `phase_damping_error(lam)`,
`pauli_error([("X", 0.1), ("I", 0.9)])`, plus `bit_flip_error`/`phase_flip_error`.
Noise runs on the state-vector backend as per-shot Monte-Carlo trajectories.

CLI presets (depolarizing on every gate, symmetric readout flip on every qubit):

```bash
aleph run circuit.qasm --shots 4096 --noise depol:0.01 --noise readout:0.02
```
````

- [ ] **Step 2: Add the same Python example to `crates/aleph-py/README.md`**

Append a "Noise" section to `crates/aleph-py/README.md` with the Python code block from Step 1 (the `import aleph ... .counts()` block) and the one-line factory description.

- [ ] **Step 3: Tick the P4.6-05 acceptance boxes in BACKLOG.md**

Change `crates/.../BACKLOG.md` lines 2819-2820 from `- [ ]` to `- [x]`:

```markdown
- [x] Python `NoiseModel` + error factories (Aer names) + `aleph.run(..., noise=)` per spec, with tests in `scripts/python/test_aleph.py`; CLI exposure for at least a depolarizing preset.
- [x] README + crate-README examples; docs updated; release-notes entry.
```

- [ ] **Step 4: Commit**

```bash
git add README.md crates/aleph-py/README.md BACKLOG.md
git commit -m "[P4.6-05] docs: noise examples in READMEs; tick BACKLOG AC"
```

---

## Task 7: Final verification + PR

- [ ] **Step 1: Workspace lint + fmt**

Run: `cargo fmt --check`
Expected: clean.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings. (This does NOT build aleph-py with the `python` feature; run the Task 1/2 clippy commands separately to cover the binding code.)

Run: `cargo clippy -p aleph-py --features python --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 2: Full Rust test sweep**

Run: `cargo test -p aleph-cli`
Expected: all CLI tests (including the new `noise_*`) pass.

Run: `cargo test --workspace`
Expected: green (no regressions; aleph-py is not cargo-tested by design).

- [ ] **Step 3: Wheel behaviour sweep**

Run (in the maturin venv): `cd crates/aleph-py && maturin develop --release --features python && cd ../.. && python -m unittest discover -s scripts/python -v`
Expected: full suite passes, including `TestNoise`.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin p46-05-noise-python-cli
gh pr create --title "[P4.6-05] Noise models: Python/CLI surface + docs" --body "$(cat <<'EOF'
Closes #168

## Summary
Aer-compatible noise surface over the merged `aleph_sv::noise` engine:
- Python: `aleph.NoiseModel` (`add_quantum_error`/`add_all_qubit_quantum_error`/`add_readout_error`), error factories (`depolarizing_error`, `amplitude_damping_error`, `phase_damping_error`, `pauli_error`, `bit_flip_error`, `phase_flip_error`), and `aleph.run(circuit, shots, noise=, seed=)`.
- Aer gate mnemonics (`h`/`cx`/`p`/`ccx`) are translated to aleph internal `Gate::name()` at attach time; unknown names (incl. Aer's idle `id`, which aleph has no carrier for) raise `ValueError`.
- Factories validate params before the panicking Rust constructors → `ValueError`, never `PanicException`.
- CLI: `aleph run --noise depol:<p>` / `--noise readout:<p>` (repeatable), SV-only, shots-only.

## Approach
Pure binding/translation layer — no new simulation logic. `run(noise=)` and the CLI dispatch to `run_noisy` on the **un-optimized** circuit (the optimizer emits TiledBlock/UnitaryKq that `run_noisy` rejects). Noise is per-shot Monte-Carlo, matching Aer's execution model.

## Tests
- Python (`scripts/python/test_aleph.py::TestNoise`): empty-model ≡ noiseless, depolarizing spreads the distribution, readout flips outcomes, bad params → ValueError, unknown gate name → ValueError, Aer-mnemonic→internal mapping, mps+noise → ValueError, `statevector()` on noisy → ValueError. Run via the built wheel.
- CLI (`crates/aleph-cli/tests/cli.rs`): `--noise depol/readout` run, bad value / out-of-range / unknown preset / stabilizer-backend / statevector-view all rejected.
- The quantitative 1e-5 @ 100k Aer oracle is covered by P4.6-04's Rust-side `noise_oracle.rs`; not re-run here.

## Notes / follow-ups
- v1 limits (documented): no mid-circuit measure/reset under noise, no idle/`id`-gate decoherence, no statevector view under noise.
- Release notes: added to the READMEs; the next GitHub release body should mention noise (no committed CHANGELOG exists).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (completed by plan author)

- **Spec coverage:** Python NoiseModel + factories + `run(noise=)` (Tasks 1-2), gate-name map incl. `id`→error (Task 1), param validation (Task 1), un-optimized circuit + SV-only guard (Task 2), histogram→dict (Task 2), CLI depol+readout presets + guards (Task 4), Python tests (Task 3), CLI tests (Task 5), README/crate-README/BACKLOG (Task 6). All spec sections map to a task.
- **Type consistency:** `PyQuantumError.inner` / `PyNoiseModel.inner` (pub(crate)) used in run.rs; `aer_to_aleph` returns `&'static str` consumed by `&[&str]` core API; `build_noise_model` returns `aleph_sv::noise::NoiseModel`; `format_counts_hist` / `hist_to_counts` match the dense `Vec<u64>` shape of `Counts`.
- **No placeholders:** every code step shows complete code.
