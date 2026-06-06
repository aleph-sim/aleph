//! `aleph-py`: minimal PyO3 bindings for VQE (P4-04).
//!
//! Exposes a `PauliSum` observable loader and a single `hea_energy` energy
//! evaluation, behind the `python` feature so the default workspace build needs
//! no Python interpreter.

#[cfg(feature = "python")]
// pyo3 0.22's proc-macro expansion emits trivial PyErr->PyErr `.into()` calls
// that clippy flags as useless_conversion; suppress for the whole binding module.
#[allow(clippy::useless_conversion)]
mod py {
    use aleph_backend::{expectation_pauli_sum, run};
    use aleph_core::PauliSum as CorePauliSum;
    use aleph_ir::build_hea;
    use aleph_sv::NaiveSvBackend;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// A Pauli-sum observable loaded from the committed text format.
    #[pyclass]
    pub struct PauliSum {
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
    fn hea_energy(n_qubits: u32, depth: u32, thetas: Vec<f64>, ham: &PauliSum) -> PyResult<f64> {
        let circuit = build_hea(n_qubits, depth, &thetas)
            .map_err(|e| PyValueError::new_err(format!("build_hea: {e}")))?;
        let mut backend = NaiveSvBackend::with_seed(0);
        let state = run(&mut backend, &circuit)
            .map_err(|e| PyValueError::new_err(format!("run: {e:?}")))?;
        expectation_pauli_sum(&mut backend, &state, &ham.inner)
            .map_err(|e| PyValueError::new_err(format!("energy: {e:?}")))
    }

    #[pymodule]
    fn aleph(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PauliSum>()?;
        m.add_function(wrap_pyfunction!(hea_energy, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_loads() {}
}
