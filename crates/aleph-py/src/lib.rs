//! `aleph-py`: PyO3 bindings for the aleph quantum circuit simulator.
//!
//! Exposes a `Circuit` builder, a `run()` entry point over the SV/MPS/
//! stabilizer backends, and the VQE/QAOA energy helpers, behind the
//! `python` feature so the default workspace build needs no Python
//! interpreter.

/// Batch-decode core over `aleph-qec`: builds named decoders from a DEM and decodes shot
/// batches in parallel with rayon. Kept free of the `python` cfg so `cargo test -p aleph-py`
/// exercises it without a Python interpreter; Task 5 wraps it in pyo3.
pub mod qec_core;

#[cfg(feature = "python")]
mod circuit;

#[cfg(feature = "python")]
mod energy;

#[cfg(feature = "python")]
mod run;

#[cfg(feature = "python")]
mod noise;

#[cfg(feature = "python")]
mod qec;

#[cfg(feature = "python")]
mod module {
    use pyo3::prelude::*;

    /// Crate version as a `&str` (`aleph.version()`); the same string is
    /// exported as `aleph.__version__`. Single source: `CARGO_PKG_VERSION`,
    /// which maturin also uses for the wheel version.
    #[pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[pymodule]
    #[pyo3(name = "_native")]
    fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_class::<crate::circuit::PyCircuit>()?;
        m.add_class::<crate::energy::PauliSum>()?;
        m.add_function(wrap_pyfunction!(crate::energy::hea_energy, m)?)?;
        m.add_function(wrap_pyfunction!(crate::energy::qaoa_energy, m)?)?;
        m.add_class::<crate::run::RunResult>()?;
        m.add_function(wrap_pyfunction!(crate::run::run_circuit, m)?)?;
        m.add_class::<crate::noise::PyNoiseModel>()?;
        m.add_class::<crate::noise::PyQuantumError>()?;
        m.add_function(wrap_pyfunction!(crate::noise::depolarizing_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::amplitude_damping_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::phase_damping_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::bit_flip_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::phase_flip_error, m)?)?;
        m.add_function(wrap_pyfunction!(crate::noise::pauli_error, m)?)?;
        crate::qec::register(m)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_loads() {}
}
