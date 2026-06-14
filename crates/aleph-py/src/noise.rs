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
        return Err(PyValueError::new_err("pauli_error weights must sum to > 0"));
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
