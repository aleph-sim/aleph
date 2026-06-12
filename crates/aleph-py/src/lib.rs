//! `aleph-py`: PyO3 bindings for the aleph quantum circuit simulator.
//!
//! Exposes a `Circuit` builder, a `run()` entry point over the SV/MPS/
//! stabilizer backends, and the VQE/QAOA energy helpers, behind the
//! `python` feature so the default workspace build needs no Python
//! interpreter.

#[cfg(feature = "python")]
mod circuit;

#[cfg(feature = "python")]
mod energy;

#[cfg(feature = "python")]
mod run;

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
    fn aleph(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_class::<crate::circuit::PyCircuit>()?;
        m.add_class::<crate::energy::PauliSum>()?;
        m.add_function(wrap_pyfunction!(crate::energy::hea_energy, m)?)?;
        m.add_function(wrap_pyfunction!(crate::energy::qaoa_energy, m)?)?;
        m.add_class::<crate::run::RunResult>()?;
        m.add_function(wrap_pyfunction!(crate::run::run_circuit, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_loads() {}
}
