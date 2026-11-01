//! Exact FP64 execution of observable-specific compact zero-state circuits.
//! Dense state/workspace is capped before any dense-state allocation. Plans, input IR,
//! allocator metadata and process/runtime overhead are outside that buffer cap.
use crate::{CpuState, NaiveSvBackend};
use aleph_backend::{Backend, BackendError};
use aleph_core::{Complex, Gate, PauliSum};
use aleph_ir::{
    expectation_query::{ExpectationQueryPlan, QueryPlanError},
    Circuit, Instruction,
};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum ExactQueryError {
    #[error(transparent)]
    Plan(#[from] QueryPlanError),
    #[error("exact expectation cone has {active_qubits} active qubits; dense workspace needs {required_bytes:?} bytes (None = overflow), budget {budget_bytes} bytes")]
    Memory {
        active_qubits: u32,
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("expectation sum overflows or is non-finite")]
    NonFiniteResult,
}

/// Measured execution includes allocation, gates, expectations and state release.
#[derive(Debug, Clone)]
pub struct ConeExecutionStats {
    pub active_qubits: u32,
    pub retained_operations: usize,
    pub dense_workspace_bytes: usize,
    pub execution_time: Duration,
}
#[derive(Debug, Clone)]
pub struct ExactQueryResult {
    pub value: f64,
    /// Input term order, including coefficients and analytic identity terms.
    pub term_values: Vec<f64>,
    pub planning_time: Duration,
    pub execution_time: Duration,
    pub peak_dense_workspace_bytes: usize,
    pub cones: Vec<ConeExecutionStats>,
}
#[derive(Debug, Clone)]
pub struct ExactExpectationQuery {
    plan: ExpectationQueryPlan,
    planning_time: Duration,
    workspace: Vec<usize>,
}
impl ExactExpectationQuery {
    /// Input is a unitary |0…0>-initialized circuit only. Dense workspace bound
    /// conservatively includes state (16N), probability validation (8N), the X/Y
    /// expectation clone (16N), 4 KiB fixed scratch, and SIMD matrix broadcasts
    /// for uncontrolled Kq gates (128 * 4^k bytes, conservatively on every CPU).
    /// Runtime/worker stacks and initialization are excluded. Cones run sequentially;
    /// the cap is not multiplied by the number of terms. Identity needs no state.
    pub fn compile(
        circuit: &Circuit,
        observable: &PauliSum,
        budget_bytes: usize,
    ) -> Result<Self, ExactQueryError> {
        let start = Instant::now();
        let plan = ExpectationQueryPlan::compile(circuit, observable)?;
        let mut workspace = Vec::new();
        for cone in plan.cones() {
            let active = cone.circuit().num_qubits();
            let broadcasts = cone
                .circuit()
                .instructions()
                .iter()
                .filter_map(|inst| match inst {
                    Instruction::Gate(g) if g.controls.is_empty() => match g.gate {
                        Gate::UnitaryKq { k, .. } => Some(128usize * (1usize << (2 * k))),
                        _ => None,
                    },
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            let bytes = 1usize
                .checked_shl(active)
                .and_then(|n| n.checked_mul(40))
                .and_then(|n| n.checked_add(4096))
                .and_then(|n| n.checked_add(broadcasts));
            match bytes {
                Some(bytes) if bytes <= budget_bytes && bytes <= isize::MAX as usize => {
                    workspace.push(bytes)
                }
                _ => {
                    return Err(ExactQueryError::Memory {
                        active_qubits: active,
                        required_bytes: bytes,
                        budget_bytes,
                    })
                }
            }
        }
        Ok(Self {
            plan,
            planning_time: start.elapsed(),
            workspace,
        })
    }
    pub fn plan(&self) -> &ExpectationQueryPlan {
        &self.plan
    }
    pub fn peak_dense_workspace_bytes(&self) -> usize {
        self.workspace.iter().copied().max().unwrap_or(0)
    }
    pub fn planning_time(&self) -> Duration {
        self.planning_time
    }

    /// Reusable deterministic query, returning values/diagnostics only. No compact
    /// state is exposed as if it were the full circuit's final state.
    pub fn execute(&self) -> Result<ExactQueryResult, ExactQueryError> {
        let start = Instant::now();
        let mut values = vec![0.; self.plan.term_count()];
        for &(index, value) in self.plan.analytic_terms() {
            values[index] = value;
        }
        let mut stats = Vec::new();
        for (cone, &bytes) in self.plan.cones().iter().zip(&self.workspace) {
            let started = Instant::now();
            let active = cone.circuit().num_qubits();
            let mut backend = NaiveSvBackend::with_seed(0).with_qubit_cap(active);
            let mut state = backend.allocate(active)?;
            for inst in cone.circuit().instructions() {
                match inst {
                    Instruction::Gate(g) => {
                        if let Gate::UnitaryKq { k, data } = &g.gate {
                            if g.controls.is_empty() {
                                // Private immutable plan validated width, matrix and
                                // indices before allocation. Reuse the SIMD kernel
                                // without a second public-backend validation snapshot.
                                crate::kernels::unitary_kq::apply_kq_aos(
                                    &mut state.amps,
                                    &g.qubits,
                                    *k,
                                    data,
                                );
                            } else {
                                apply_query_kq(&mut state, &g.qubits, &g.controls, *k, data);
                            }
                        } else {
                            backend.apply_gate(&mut state, g)?;
                        }
                    }
                    Instruction::DiagonalPhase(dp) => {
                        backend.apply_diagonal_phase(&mut state, dp)?
                    }
                    Instruction::Barrier(_) => {}
                    _ => unreachable!("immutable validated query plan"),
                }
            }
            for (index, observable) in cone.observables() {
                values[*index] = backend.expectation_value(&state, observable)?;
            }
            drop(state);
            stats.push(ConeExecutionStats {
                active_qubits: active,
                retained_operations: cone.circuit().len(),
                dense_workspace_bytes: bytes,
                execution_time: started.elapsed(),
            });
        }
        let value = values.iter().sum::<f64>();
        if !value.is_finite() || values.iter().any(|x| !x.is_finite()) {
            return Err(ExactQueryError::NonFiniteResult);
        }
        Ok(ExactQueryResult {
            value,
            term_values: values,
            planning_time: self.planning_time,
            execution_time: start.elapsed(),
            peak_dense_workspace_bytes: self.peak_dense_workspace_bytes(),
            cones: stats,
        })
    }
}

// Query-local bounded controlled matvec: public CPU Kq controls are unsupported.
// Preserves MSB-first operand order with no unsafe code or state clone.
fn apply_query_kq(state: &mut CpuState, qubits: &[u32], controls: &[u32], k: u8, data: &[Complex]) {
    let dim = 1usize << k;
    let target_mask = qubits.iter().fold(0usize, |m, q| m | 1usize << q);
    let control_mask = controls.iter().fold(0usize, |m, q| m | 1usize << q);
    let mut offsets = [0usize; 32];
    for (index, offset) in offsets.iter_mut().enumerate().take(dim) {
        for (bit, q) in qubits.iter().rev().enumerate() {
            if index & (1 << bit) != 0 {
                *offset |= 1usize << q;
            }
        }
    }
    let mut input = [Complex::new(0., 0.); 32];
    for base in 0..state.amps.len() {
        if base & target_mask != 0 || base & control_mask != control_mask {
            continue;
        }
        for i in 0..dim {
            input[i] = state.amps[base | offsets[i]];
        }
        for row in 0..dim {
            let value: Complex = (0..dim).map(|col| data[row * dim + col] * input[col]).sum();
            state.amps[base | offsets[row]] = value;
        }
    }
}
