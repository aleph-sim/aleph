//! `aleph-py`: PyO3 bindings for the aleph quantum circuit simulator.
//!
//! Exposes a `Circuit` builder, a `run()` entry point over the SV/MPS/
//! stabilizer backends, and the VQE/QAOA energy helpers, behind the
//! `python` feature so the default workspace build needs no Python
//! interpreter.

#[cfg(feature = "python")]
mod energy;

#[cfg(feature = "python")]
mod module {
    use pyo3::prelude::*;

    #[pymodule]
    fn aleph(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<crate::energy::PauliSum>()?;
        m.add_function(wrap_pyfunction!(crate::energy::hea_energy, m)?)?;
        m.add_function(wrap_pyfunction!(crate::energy::qaoa_energy, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_loads() {}
}
