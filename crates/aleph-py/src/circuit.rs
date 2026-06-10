//! Python `Circuit` builder over `aleph_ir::Circuit`.
//!
//! Every method validates its qubit operands up front: `GateInstance::new`
//! / `::controlled` debug-assert uniqueness (panic, not `Err`), so without
//! the explicit duplicate check a debug wheel would surface a
//! `PanicException` instead of `ValueError` on e.g. `c.cx(0, 0)`.
#![allow(clippy::useless_conversion)]

use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::Circuit as IrCircuit;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use smallvec::smallvec;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn check_distinct(qs: &[u32]) -> PyResult<()> {
    for (i, a) in qs.iter().enumerate() {
        if qs[i + 1..].contains(a) {
            return Err(PyValueError::new_err(format!(
                "duplicate qubit operand {a}"
            )));
        }
    }
    Ok(())
}

/// A quantum circuit under construction.
#[pyclass(name = "Circuit")]
pub(crate) struct PyCircuit {
    pub(crate) inner: IrCircuit,
}

#[pymethods]
impl PyCircuit {
    /// `Circuit(num_qubits, num_clbits=None)` — clbits default to qubits.
    #[new]
    #[pyo3(signature = (num_qubits, num_clbits = None))]
    fn py_new(num_qubits: u32, num_clbits: Option<u32>) -> PyResult<Self> {
        let clbits = num_clbits.unwrap_or(num_qubits);
        Ok(Self {
            inner: IrCircuit::try_new(num_qubits, clbits).map_err(err)?,
        })
    }

    /// Parse an OpenQASM 3.0 source string.
    #[staticmethod]
    fn from_qasm(source: &str) -> PyResult<Self> {
        Ok(Self {
            inner: aleph_parser::parse(source).map_err(err)?,
        })
    }

    /// Parse an OpenQASM 3.0 file.
    #[staticmethod]
    fn from_qasm_file(path: &str) -> PyResult<Self> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("read {path}: {e}")))?;
        Self::from_qasm(&src)
    }

    #[getter]
    fn num_qubits(&self) -> u32 {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> u32 {
        self.inner.num_clbits()
    }

    /// Number of gate instructions (excludes measure/reset/barrier).
    #[getter]
    fn num_gates(&self) -> usize {
        self.inner
            .instructions()
            .iter()
            .filter(|i| matches!(i, aleph_ir::Instruction::Gate(_)))
            .count()
    }

    // --- 1q standard ---
    fn h(&mut self, q: u32) -> PyResult<()> {
        self.inner.h(q).map_err(err).map(drop)
    }
    fn x(&mut self, q: u32) -> PyResult<()> {
        self.inner.x(q).map_err(err).map(drop)
    }
    fn y(&mut self, q: u32) -> PyResult<()> {
        self.inner.y(q).map_err(err).map(drop)
    }
    fn z(&mut self, q: u32) -> PyResult<()> {
        self.inner.z(q).map_err(err).map(drop)
    }
    fn s(&mut self, q: u32) -> PyResult<()> {
        self.inner.s(q).map_err(err).map(drop)
    }
    fn sdg(&mut self, q: u32) -> PyResult<()> {
        self.inner.sdg(q).map_err(err).map(drop)
    }
    fn t(&mut self, q: u32) -> PyResult<()> {
        self.inner.t(q).map_err(err).map(drop)
    }
    fn tdg(&mut self, q: u32) -> PyResult<()> {
        self.inner.tdg(q).map_err(err).map(drop)
    }

    // --- 1q parametric ---
    fn rx(&mut self, theta: f64, q: u32) -> PyResult<()> {
        self.inner.rx(theta, q).map_err(err).map(drop)
    }
    fn ry(&mut self, theta: f64, q: u32) -> PyResult<()> {
        self.inner.ry(theta, q).map_err(err).map(drop)
    }
    fn rz(&mut self, theta: f64, q: u32) -> PyResult<()> {
        self.inner.rz(theta, q).map_err(err).map(drop)
    }
    /// `p(θ)` = `diag(1, e^{iθ})` (OpenQASM `p`, Qiskit `PhaseGate`).
    fn p(&mut self, theta: f64, q: u32) -> PyResult<()> {
        self.inner.phase(theta, q).map_err(err).map(drop)
    }
    fn u3(&mut self, theta: f64, phi: f64, lam: f64, q: u32) -> PyResult<()> {
        self.inner.u3(theta, phi, lam, q).map_err(err).map(drop)
    }

    // --- 2q ---
    fn cx(&mut self, control: u32, target: u32) -> PyResult<()> {
        check_distinct(&[control, target])?;
        self.inner.cnot(control, target).map_err(err).map(drop)
    }
    fn cz(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        check_distinct(&[q0, q1])?;
        self.inner.cz(q0, q1).map_err(err).map(drop)
    }
    fn swap(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        check_distinct(&[q0, q1])?;
        self.inner.swap(q0, q1).map_err(err).map(drop)
    }
    fn iswap(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        check_distinct(&[q0, q1])?;
        self.inner
            .add_gate(GateInstance::new(Gate::Iswap, smallvec![q0, q1]))
            .map_err(err)
            .map(drop)
    }
    fn crx(&mut self, theta: f64, control: u32, target: u32) -> PyResult<()> {
        check_distinct(&[control, target])?;
        self.inner
            .add_gate(GateInstance::new(
                Gate::CRx(Param::Concrete(theta)),
                smallvec![control, target],
            ))
            .map_err(err)
            .map(drop)
    }
    fn cry(&mut self, theta: f64, control: u32, target: u32) -> PyResult<()> {
        check_distinct(&[control, target])?;
        self.inner
            .add_gate(GateInstance::new(
                Gate::CRy(Param::Concrete(theta)),
                smallvec![control, target],
            ))
            .map_err(err)
            .map(drop)
    }
    fn crz(&mut self, theta: f64, control: u32, target: u32) -> PyResult<()> {
        check_distinct(&[control, target])?;
        self.inner
            .add_gate(GateInstance::new(
                Gate::CRz(Param::Concrete(theta)),
                smallvec![control, target],
            ))
            .map_err(err)
            .map(drop)
    }
    /// Controlled-phase: `Phase(θ)` on `target` with one external control —
    /// the `qft_circuit` construction (benches/src/lib.rs), since the
    /// parser/IR have no first-class `cp` gate.
    fn cp(&mut self, theta: f64, control: u32, target: u32) -> PyResult<()> {
        check_distinct(&[control, target])?;
        self.inner
            .add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                smallvec![target],
                smallvec![control],
            ))
            .map_err(err)
            .map(drop)
    }

    // --- 3q ---
    fn ccx(&mut self, c0: u32, c1: u32, target: u32) -> PyResult<()> {
        check_distinct(&[c0, c1, target])?;
        self.inner.ccx(c0, c1, target).map_err(err).map(drop)
    }
    fn ccz(&mut self, q0: u32, q1: u32, q2: u32) -> PyResult<()> {
        check_distinct(&[q0, q1, q2])?;
        self.inner
            .add_gate(GateInstance::new(Gate::Ccz, smallvec![q0, q1, q2]))
            .map_err(err)
            .map(drop)
    }

    // --- non-gate ---
    fn measure(&mut self, qubit: u32, clbit: u32) -> PyResult<()> {
        self.inner.measure(qubit, clbit).map_err(err).map(drop)
    }
    fn reset(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.reset(qubit).map_err(err).map(drop)
    }
    fn barrier(&mut self, qubits: Vec<u32>) -> PyResult<()> {
        self.inner.barrier(qubits).map_err(err).map(drop)
    }
}
