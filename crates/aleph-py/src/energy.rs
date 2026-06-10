//! Energy-evaluation bindings (VQE/QAOA) — pre-P4-08 API, unchanged.

// pyo3 0.22's proc-macro expansion emits trivial PyErr->PyErr `.into()` calls
// that clippy flags as useless_conversion; suppress for the whole binding module.
#![allow(clippy::useless_conversion)]

use aleph_backend::{expectation_pauli_sum, run};
use aleph_core::PauliSum as CorePauliSum;
use aleph_ir::build_hea;
use aleph_ir::{build_qaoa, maxcut_pauli_sum};
use aleph_mps::MpsBackend;
use aleph_sv::NaiveSvBackend;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Pauli-sum observable loaded from the committed text format.
#[pyclass]
pub(crate) struct PauliSum {
    inner: CorePauliSum,
}

#[pymethods]
impl PauliSum {
    /// Load from a `<coeff> <pauli>`-per-line file on `n_qubits` qubits.
    #[staticmethod]
    fn load(path: &str, n_qubits: u32) -> PyResult<Self> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("read {path}: {e}")))?;
        let inner = CorePauliSum::parse(&src, n_qubits)
            .map_err(|e| PyValueError::new_err(format!("parse {path}: {e}")))?;
        Ok(Self { inner })
    }

    /// Number of Pauli terms.
    fn num_terms(&self) -> usize {
        self.inner.terms.len()
    }
}

/// Energy ⟨ψ(θ)|H|ψ(θ)⟩ of the Ry+CNOT HEA at `thetas` under `ham`.
/// One call = one VQE energy evaluation (build ansatz → simulate → ⟨H⟩).
#[pyfunction]
pub(crate) fn hea_energy(
    n_qubits: u32,
    depth: u32,
    thetas: Vec<f64>,
    ham: &PauliSum,
) -> PyResult<f64> {
    let circuit = build_hea(n_qubits, depth, &thetas)
        .map_err(|e| PyValueError::new_err(format!("build_hea: {e}")))?;
    let mut backend = NaiveSvBackend::with_seed(0);
    let state =
        run(&mut backend, &circuit).map_err(|e| PyValueError::new_err(format!("run: {e:?}")))?;
    expectation_pauli_sum(&mut backend, &state, &ham.inner)
        .map_err(|e| PyValueError::new_err(format!("energy: {e:?}")))
}

/// QAOA Max-Cut energy ⟨H_C⟩ for `edges` at angles `gammas`/`betas`, on the
/// chosen backend ("sv" = state vector, "mps" = matrix product state).
/// One call = build QAOA circuit + cost Hamiltonian + simulate + ⟨H_C⟩.
#[pyfunction]
pub(crate) fn qaoa_energy(
    n_qubits: u32,
    edges: Vec<(u32, u32)>,
    gammas: Vec<f64>,
    betas: Vec<f64>,
    backend: &str,
) -> PyResult<f64> {
    let circuit = build_qaoa(n_qubits, &edges, &gammas, &betas)
        .map_err(|e| PyValueError::new_err(format!("build_qaoa: {e}")))?;
    let ham = maxcut_pauli_sum(n_qubits, &edges)
        .map_err(|e| PyValueError::new_err(format!("maxcut: {e}")))?;
    match backend {
        "sv" => {
            let mut be = NaiveSvBackend::with_seed(0);
            let st = run(&mut be, &circuit)
                .map_err(|e| PyValueError::new_err(format!("run sv: {e:?}")))?;
            expectation_pauli_sum(&mut be, &st, &ham)
                .map_err(|e| PyValueError::new_err(format!("energy sv: {e:?}")))
        }
        "mps" => {
            let mut be = MpsBackend::with_seed(0).with_max_bond(128);
            let st = run(&mut be, &circuit)
                .map_err(|e| PyValueError::new_err(format!("run mps: {e:?}")))?;
            expectation_pauli_sum(&mut be, &st, &ham)
                .map_err(|e| PyValueError::new_err(format!("energy mps: {e:?}")))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (expected \"sv\" or \"mps\")"
        ))),
    }
}
