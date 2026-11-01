//! Exact observable queries for unitary circuits starting in |0…0>.
//!
//! Reverse support propagation promotes the reference in
//! `aleph-mps/tests/shallow_100q.rs` (P3-10). A discarded unitary acts outside
//! the current observable support, so U† O U = O there. Barriers impose order
//! but are not entangling operations. This does NOT produce the full final state.
use std::collections::BTreeMap;

use crate::{Circuit, DiagonalPhase, Instruction};
use aleph_core::{Complex, Gate, GateInstance, GateMatrix, PauliString, PauliSum};

#[derive(Debug, thiserror::Error)]
pub enum QueryPlanError {
    #[error("unsupported or invalid expectation query: {0}")]
    Invalid(String),
}
fn invalid(message: impl Into<String>) -> QueryPlanError {
    QueryPlanError::Invalid(message.into())
}

/// One compact zero-state simulation, shared only by identical retained circuits
/// and active supports. Mapping indices exposed here always use original LOGICAL qubits.
#[derive(Debug, Clone)]
pub struct ExpectationCone {
    circuit: Circuit,
    original_to_compact: Vec<Option<u32>>,
    compact_to_original: Vec<u32>,
    retained_source_indices: Vec<usize>,
    observables: Vec<(usize, PauliString)>,
}
impl ExpectationCone {
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    pub fn original_to_compact(&self) -> &[Option<u32>] {
        &self.original_to_compact
    }
    pub fn compact_to_original(&self) -> &[u32] {
        &self.compact_to_original
    }
    /// Original instruction indices; a flattened tile may contribute several entries.
    pub fn retained_source_indices(&self) -> &[usize] {
        &self.retained_source_indices
    }
    /// Input term index and its compact Pauli, including its original real coefficient.
    pub fn observables(&self) -> &[(usize, PauliString)] {
        &self.observables
    }
}

#[derive(Debug, Clone)]
pub struct ExpectationQueryPlan {
    original_qubits: u32,
    original_instructions: usize,
    term_count: usize,
    cones: Vec<ExpectationCone>,
    analytic_terms: Vec<(usize, f64)>,
}
impl ExpectationQueryPlan {
    pub fn original_qubits(&self) -> u32 {
        self.original_qubits
    }
    pub fn original_instructions(&self) -> usize {
        self.original_instructions
    }
    pub fn term_count(&self) -> usize {
        self.term_count
    }
    pub fn cones(&self) -> &[ExpectationCone] {
        &self.cones
    }
    pub fn analytic_terms(&self) -> &[(usize, f64)] {
        &self.analytic_terms
    }

    /// Validate ALL instructions before pruning, including for identity/zero
    /// observables. Nonunitary disjoint operations would invalidate the proof.
    /// Tiles are expanded in order; diagonal polynomials conservatively retain
    /// their entire support. No DCE, sampling or initial-state API is involved.
    pub fn compile(circuit: &Circuit, sum: &PauliSum) -> Result<Self, QueryPlanError> {
        let n = circuit.num_qubits() as usize;
        let physical: Vec<u32> = circuit
            .qubit_permutation()
            .map_or_else(|| (0..n as u32).collect(), |p| p.to_vec());
        let mut logical = vec![u32::MAX; n];
        if physical.len() != n {
            return Err(invalid("invalid permutation length"));
        }
        for (q, &p) in physical.iter().enumerate() {
            if p as usize >= n || logical[p as usize] != u32::MAX {
                return Err(invalid("invalid permutation"));
            }
            logical[p as usize] = q as u32;
        }
        let mut operations = Vec::new();
        for (source, inst) in circuit.instructions().iter().enumerate() {
            match inst {
                Instruction::Measure { .. } | Instruction::Reset(_) => return Err(invalid(
                    format!("instruction {source}: measure/reset requires a separate boundary-state/noise treatment"))),
                Instruction::Gate(g) => { validate_gate(g, n)?; operations.push((source, inst.clone())); }
                Instruction::TiledBlock(tile) => {
                    for g in &tile.gates {
                        validate_gate(g, n)?;
                        operations.push((source, Instruction::Gate(g.clone())));
                    }
                }
                Instruction::DiagonalPhase(dp) => {
                    if dp.n_qubits > 64 || dp.n_qubits as usize > n { return Err(invalid("invalid diagonal scope")); }
                    let mask = if dp.n_qubits == 64 { u64::MAX } else { (1u64 << dp.n_qubits)-1 };
                    if dp.support_mask() & !mask != 0 { return Err(invalid("diagonal mask outside scope")); }
                    let mut absolute_sum = 0.;
                    for term in &dp.terms { absolute_sum += term.angle.abs(); }
                    if !absolute_sum.is_finite() { return Err(invalid("non-finite/overflowing diagonal angles")); }
                    operations.push((source, inst.clone()));
                }
                Instruction::Barrier(qs) => {
                    if qs.is_empty() || qs.iter().any(|q| *q as usize >= n) { return Err(invalid("invalid barrier")); }
                    operations.push((source, inst.clone()));
                }
            }
        }
        let mut plan = Self {
            original_qubits: n as u32,
            original_instructions: circuit.len(),
            term_count: sum.terms.len(),
            cones: Vec::new(),
            analytic_terms: Vec::new(),
        };
        let mut seen = BTreeMap::new();
        for (term_index, term) in sum.terms.iter().enumerate() {
            // Public Pauli fields can bypass constructor invariants.
            if term.terms.iter().any(|(q, _)| *q as usize >= n) {
                return Err(invalid("observable qubit out of range"));
            }
            let term = PauliString::new(term.coefficient, term.terms.clone())
                .map_err(|e| invalid(e.to_string()))?;
            if term.terms.is_empty() || term.coefficient == 0. {
                plan.analytic_terms.push((
                    term_index,
                    if term.terms.is_empty() {
                        term.coefficient
                    } else {
                        0.
                    },
                ));
                continue;
            }
            let mut live = vec![false; n];
            for &(q, _) in &term.terms {
                live[physical[q as usize] as usize] = true;
            }
            let mut retained = Vec::new();
            for (index, (_, inst)) in operations.iter().enumerate().rev() {
                let used = inst.used_qubits();
                if used.iter().any(|q| live[*q as usize]) {
                    retained.push(index);
                    if !matches!(inst, Instruction::Barrier(_)) {
                        for q in used {
                            live[q as usize] = true;
                        }
                    }
                }
            }
            retained.reverse();
            let active: Vec<u32> = (0..n as u32).filter(|q| live[*q as usize]).collect();
            let key = (active.clone(), retained.clone());
            let group = if let Some(&group) = seen.get(&key) {
                group
            } else {
                let mut map = vec![None; n];
                for (compact, &q) in active.iter().enumerate() {
                    map[q as usize] = Some(compact as u32);
                }
                let mut compact = Circuit::new(active.len() as u32, 0);
                for &index in &retained {
                    let inst = remap(&operations[index].1, &map, active.len() as u32);
                    compact
                        .add_instruction(inst)
                        .map_err(|e| invalid(e.to_string()))?;
                }
                let mut original_to_compact = vec![None; n];
                for q in 0..n {
                    original_to_compact[q] = map[physical[q] as usize];
                }
                let compact_to_original = active.iter().map(|q| logical[*q as usize]).collect();
                let group = plan.cones.len();
                plan.cones.push(ExpectationCone {
                    circuit: compact,
                    original_to_compact,
                    compact_to_original,
                    retained_source_indices: retained.iter().map(|i| operations[*i].0).collect(),
                    observables: Vec::new(),
                });
                seen.insert(key, group);
                group
            };
            let cone = &mut plan.cones[group];
            let remapped = PauliString::new(
                term.coefficient,
                term.terms
                    .iter()
                    .map(|&(q, p)| {
                        (
                            cone.original_to_compact[q as usize].expect("live observable support"),
                            p,
                        )
                    })
                    .collect(),
            )
            .map_err(|e| invalid(e.to_string()))?;
            cone.observables.push((term_index, remapped));
        }
        Ok(plan)
    }
}

fn remap(inst: &Instruction, map: &[Option<u32>], n: u32) -> Instruction {
    let mapped = |q: &u32| map[*q as usize].expect("kept operation support");
    match inst {
        Instruction::Gate(g) => {
            let mut g = g.clone();
            g.qubits = g.qubits.iter().map(mapped).collect();
            g.controls = g.controls.iter().map(mapped).collect();
            Instruction::Gate(g)
        }
        Instruction::Barrier(qs) => {
            Instruction::Barrier(qs.iter().filter_map(|q| map[*q as usize]).collect())
        }
        Instruction::DiagonalPhase(dp) => {
            let mut dp: DiagonalPhase = (**dp).clone();
            dp.n_qubits = n.min(64);
            for term in &mut dp.terms {
                for mask in &mut term.conds {
                    let old = *mask;
                    *mask = 0;
                    for q in 0..64 {
                        if (old >> q) & 1 != 0 {
                            *mask |= 1u64 << mapped(&(q as u32));
                        }
                    }
                }
            }
            Instruction::DiagonalPhase(Box::new(dp))
        }
        _ => unreachable!("validated flattened unitary operations"),
    }
}

/// Bounded unitary validation also protects manual fused IR before state allocation.
fn validate_gate(g: &GateInstance, n: usize) -> Result<(), QueryPlanError> {
    if g.qubits.len() != g.gate.arity() || g.controls.len() > crate::MAX_GATE_CONTROLS {
        return Err(invalid("gate arity/control count"));
    }
    let mut seen = Vec::new();
    for &q in g.qubits.iter().chain(&g.controls) {
        if q as usize >= n || seen.contains(&q) {
            return Err(invalid("gate operand range/duplicates"));
        }
        seen.push(q);
    }
    let (dim, data): (usize, Vec<Complex>) = if let Gate::UnitaryKq { k, data } = &g.gate {
        if !(2..=5).contains(k) || data.len() != 1usize << (2 * usize::from(*k)) {
            return Err(invalid(
                "UnitaryKq requires k=2..5 and a complete row-major matrix",
            ));
        }
        (1 << k, data.to_vec())
    } else {
        match g
            .gate
            .matrix()
            .map_err(|e| invalid(format!("{}: {e}", g.gate.name())))?
        {
            GateMatrix::M2x2(m) => (2, m.into_iter().flatten().collect()),
            GateMatrix::M4x4(m) => (4, m.into_iter().flatten().collect()),
            GateMatrix::M8x8(m) => (8, m.into_iter().flatten().collect()),
        }
    };
    if data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(invalid("non-finite gate matrix"));
    }
    for i in 0..dim {
        for j in 0..dim {
            let inner: Complex = (0..dim)
                .map(|k| data[i * dim + k] * data[j * dim + k].conj())
                .sum();
            let deviation = (inner - Complex::new(if i == j { 1. } else { 0. }, 0.)).norm();
            if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
                return Err(invalid("non-unitary gate matrix"));
            }
        }
    }
    Ok(())
}
