//! Untruncated f64 Heisenberg propagation for zero-state Pauli-sum queries.
//! No state vector, sampling, coefficient threshold or automatic routing.
use aleph_core::{Gate, GateInstance, Param, Pauli, PauliString, PauliSum};
use aleph_ir::{expectation_query::ExpectationQueryPlan, Circuit, Instruction};
use std::time::{Duration, Instant};

/// Explicit limits for the sparse workspace (not input IR, plans or allocator metadata).
#[derive(Clone, Copy, Debug)]
pub struct SparsePauliOptions {
    /// Maximum candidate terms BEFORE merging. A conservative bound may reject canceling work.
    pub max_terms: usize,
    /// Packed keys plus both term-vector capacities; checked before expansion allocation.
    pub max_workspace_bytes: usize,
    /// Reuse P7-03's validated, observable-specific causal cones.
    pub use_cones: bool,
}
impl Default for SparsePauliOptions {
    fn default() -> Self {
        Self {
            max_terms: 65_536,
            max_workspace_bytes: 64 << 20,
            use_cones: true,
        }
    }
}
/// Failure never returns an incomplete expectation.
#[derive(Debug, thiserror::Error)]
pub enum SparsePauliError {
    /// Unsupported gate, invalid observable, parameter or operand.
    #[error("unsupported/invalid sparse Pauli query: {0}")]
    Invalid(String),
    /// Conservative pre-merge candidate count exceeds the limit.
    #[error("sparse Pauli candidate terms {required} exceed limit {limit}")]
    Terms { required: usize, limit: usize },
    /// Workspace bound or checked size arithmetic exceeded the configured budget.
    #[error("sparse Pauli workspace {required_bytes:?} exceeds budget {budget_bytes}")]
    Memory {
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    /// Floating-point coefficient arithmetic overflowed; no clipping is used.
    #[error("non-finite sparse Pauli coefficient/result")]
    NonFinite,
}
type Result<T> = std::result::Result<T, SparsePauliError>;
fn invalid(s: impl Into<String>) -> SparsePauliError {
    SparsePauliError::Invalid(s.into())
}
/// Resource and algebra diagnostics; counts sum across sequential cone groups.
#[derive(Clone, Debug, Default)]
pub struct SparsePauliStats {
    /// Largest compact support, or original width without cones.
    pub max_active_qubits: u32,
    /// Gates visited across groups (barriers excluded).
    pub retained_gates: usize,
    /// Maximum resident nonzero terms after merging.
    pub peak_terms: usize,
    /// Maximum pre-merge candidate population.
    pub peak_candidates: usize,
    /// Conservative simultaneous vector/key payload bound, excluding plan/runtime metadata.
    pub peak_workspace_bytes: usize,
    /// Rotation applications that actually split at least one anticommuting term.
    pub branching_rotations: usize,
    /// Extra candidate terms produced by rotation splits.
    pub splits: usize,
    /// Duplicate keys folded by deterministic sorting (including exact cancellation).
    pub merges: usize,
    /// Sum of final group coefficient L1 norms (not an error bound).
    pub final_coefficient_l1: f64,
    /// Euclidean norm of final group coefficient vectors, concatenated across cones.
    pub final_coefficient_l2: f64,
}
/// Expectation only; no amplitudes or samples are implied by this result.
#[derive(Clone, Debug)]
pub struct SparsePauliResult {
    /// Weighted Pauli sum on the specified circuit's |0…0> input.
    pub value: f64,
    /// Includes validation and optional cone planning.
    pub planning_time: Duration,
    /// Includes allocation, propagation, merging, expectation and workspace release.
    pub execution_time: Duration,
    /// Sparse resource/algebra counters.
    pub stats: SparsePauliStats,
}
#[derive(Clone, Copy, Debug)]
enum Op {
    Single(u8, u32),
    Cx(u32, u32),
    Cz(u32, u32),
    Cy(u32, u32),
    Swap(u32, u32),
    Iswap(u32, u32, bool),
    Rotation(u8, u32, f64, f64),
}
struct Group {
    n: u32,
    ops: Vec<Op>,
    terms: Vec<PauliString>,
}
/// Reusable opt-in query. Supports H/X/Y/Z/S/Sdg, CNOT/CZ/SWAP/iSWAP/adjoint,
/// one external control on X/Y/Z, and finite Rx/Ry/Rz/Phase/T/Tdg rotations.
/// Generic matrices, fused diagonal polynomials, other controls, measure/reset and
/// nonzero/noisy initial states are not silently approximated or routed elsewhere.
pub struct SparsePauliQuery {
    groups: Vec<Group>,
    analytic: Vec<f64>,
    options: SparsePauliOptions,
    planning_time: Duration,
}
impl SparsePauliQuery {
    /// Validate the ENTIRE circuit before cone pruning, including disjoint unsupported gates.
    /// Observable indices are logical, following P7-03's permutation contract. Memory budget
    /// covers propagation payload only; caller-owned inputs and compiled plans are excluded.
    pub fn compile(circuit: &Circuit, sum: &PauliSum, options: SparsePauliOptions) -> Result<Self> {
        let start = Instant::now();
        if options.max_terms == 0 {
            return Err(invalid("max_terms must be positive"));
        }
        limit_terms(sum.terms.len(), options)?;
        let original_operations = operations(circuit)?;
        let n = circuit.num_qubits();
        let mapping: Vec<u32> = circuit
            .qubit_permutation()
            .map_or_else(|| (0..n).collect(), |p| p.to_vec());
        let mut sorted = mapping.clone();
        sorted.sort_unstable();
        if sorted.len() != n as usize || sorted.iter().copied().ne(0..n) {
            return Err(invalid("invalid qubit permutation"));
        }
        let mut terms = Vec::new();
        for p in &sum.terms {
            if p.terms.iter().any(|(q, _)| *q >= n) {
                return Err(invalid("observable qubit out of range"));
            }
            terms.push(
                PauliString::new(p.coefficient, p.terms.clone())
                    .map_err(|e| invalid(e.to_string()))?,
            );
        }
        let mut groups = Vec::new();
        let mut analytic = Vec::new();
        if options.use_cones {
            let plan = ExpectationQueryPlan::compile(circuit, &PauliSum { terms })
                .map_err(|e| invalid(e.to_string()))?;
            analytic.extend(plan.analytic_terms().iter().map(|(_, v)| *v));
            for cone in plan.cones() {
                groups.push(Group {
                    n: cone.circuit().num_qubits(),
                    ops: operations(cone.circuit())?,
                    terms: cone.observables().iter().map(|(_, p)| p.clone()).collect(),
                });
            }
        } else {
            for p in &mut terms {
                for (q, _) in &mut p.terms {
                    *q = mapping[*q as usize];
                }
            }
            groups.push(Group {
                n,
                ops: original_operations,
                terms,
            });
        }
        for g in &groups {
            workspace(0, 0, g.terms.len(), words(g.n), options)?;
        }
        Ok(Self {
            groups,
            analytic,
            options,
            planning_time: start.elapsed(),
        })
    }
    /// Execute without intentional truncation. Only exactly zero f64 coefficients disappear;
    /// ordinary roundoff/underflow remains possible. Exhaustion returns a structured error.
    ///
    /// ```
    /// use aleph_core::{Pauli, PauliString, PauliSum};
    /// use aleph_ir::Circuit;
    /// use aleph_stab::{SparsePauliQuery, SparsePauliOptions};
    /// let mut c = Circuit::new(128, 0);
    /// c.h(0).unwrap();
    /// let p = PauliSum { terms: vec![PauliString::new(1., vec![(0, Pauli::X)]).unwrap()] };
    /// let r = SparsePauliQuery::compile(&c, &p, SparsePauliOptions::default()).unwrap().execute().unwrap();
    /// assert!((r.value - 1.).abs() < 1e-12);
    /// ```
    pub fn execute(&self) -> Result<SparsePauliResult> {
        let start = Instant::now();
        let mut stats = SparsePauliStats::default();
        let mut value = 0.;
        for &v in &self.analytic {
            value = finite(value + v)?;
            stats.final_coefficient_l1 = finite(stats.final_coefficient_l1 + v.abs())?;
            stats.final_coefficient_l2 = finite(stats.final_coefficient_l2.hypot(v))?;
        }
        for group in &self.groups {
            let w = words(group.n);
            let bytes = workspace(0, 0, group.terms.len(), w, self.options)?;
            stats.peak_workspace_bytes = stats.peak_workspace_bytes.max(bytes);
            stats.max_active_qubits = stats.max_active_qubits.max(group.n);
            stats.retained_gates += group.ops.len();
            let mut terms = Vec::with_capacity(group.terms.len());
            for p in &group.terms {
                let mut key = vec![0; 2 * w];
                for &(q, p) in &p.terms {
                    set(&mut key, w, q, pauli(p));
                }
                terms.push(Term {
                    key,
                    coefficient: p.coefficient,
                });
            }
            stats.peak_candidates = stats.peak_candidates.max(terms.len());
            stats.merges += merge(&mut terms)?;
            stats.peak_terms = stats.peak_terms.max(terms.len());
            for &op in group.ops.iter().rev() {
                if let Op::Rotation(axis, q, cos, sin) = op {
                    let anticommuting = terms
                        .iter()
                        .filter(|t| anticommutes(get(&t.key, w, q), axis))
                        .count();
                    // Commuting conjugation is exactly identity; do not copy an unchanged sum.
                    if anticommuting == 0 || (cos == 1. && sin == 0.) {
                        continue;
                    }
                    let splits = if cos != 0. && sin != 0. {
                        anticommuting
                    } else {
                        0
                    };
                    let candidates =
                        terms
                            .len()
                            .checked_add(splits)
                            .ok_or(SparsePauliError::Terms {
                                required: usize::MAX,
                                limit: self.options.max_terms,
                            })?;
                    let bytes =
                        workspace(terms.capacity(), terms.len(), candidates, w, self.options)?;
                    stats.peak_workspace_bytes = stats.peak_workspace_bytes.max(bytes);
                    stats.peak_candidates = stats.peak_candidates.max(candidates);
                    if splits > 0 {
                        stats.branching_rotations += 1;
                        stats.splits += splits;
                    }
                    let mut next = Vec::with_capacity(candidates);
                    for t in &terms {
                        let p = get(&t.key, w, q);
                        if !anticommutes(p, axis) {
                            next.push(t.clone());
                            continue;
                        }
                        if cos != 0. {
                            next.push(Term {
                                key: t.key.clone(),
                                coefficient: finite(t.coefficient * cos)?,
                            });
                        }
                        if sin != 0. {
                            let mut key = t.key.clone();
                            set(&mut key, w, q, p ^ axis);
                            // U=exp(-i theta A/2): U† P U=cos(theta)P+i sin(theta)AP.
                            // AP local phases: XY=iZ,YZ=iX,ZX=iY; reverse products=-i*.
                            // Derivation is the untruncated Heisenberg expansion, cf. Begusic
                            // et al. arXiv:2306.04797 §II; no perturbative-order pruning.
                            let sign = if matches!((axis, p), (1, 3) | (3, 2) | (2, 1)) {
                                -1.
                            } else {
                                1.
                            };
                            next.push(Term {
                                key,
                                coefficient: finite(t.coefficient * sin * sign)?,
                            });
                        }
                    }
                    terms = next;
                    stats.merges += merge(&mut terms)?;
                } else {
                    for t in &mut terms {
                        clifford(t, w, op);
                    }
                }
                stats.peak_terms = stats.peak_terms.max(terms.len());
            }
            for t in &terms {
                if t.key[..w].iter().all(|x| *x == 0) {
                    value = finite(value + t.coefficient)?;
                }
                stats.final_coefficient_l1 =
                    finite(stats.final_coefficient_l1 + t.coefficient.abs())?;
                stats.final_coefficient_l2 =
                    finite(stats.final_coefficient_l2.hypot(t.coefficient))?;
            }
        }
        Ok(SparsePauliResult {
            value,
            planning_time: self.planning_time,
            execution_time: start.elapsed(),
            stats,
        })
    }
}
fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(SparsePauliError::NonFinite)
    }
}
fn words(n: u32) -> usize {
    (n as usize).div_ceil(64)
}
fn limit_terms(n: usize, o: SparsePauliOptions) -> Result<()> {
    if n > o.max_terms {
        Err(SparsePauliError::Terms {
            required: n,
            limit: o.max_terms,
        })
    } else {
        Ok(())
    }
}
#[derive(Clone)]
struct Term {
    key: Vec<u64>,
    coefficient: f64,
}
fn workspace(
    old_capacity: usize,
    old_len: usize,
    next: usize,
    w: usize,
    o: SparsePauliOptions,
) -> Result<usize> {
    limit_terms(next, o)?;
    let bytes = old_capacity
        .checked_add(next)
        .and_then(|n| n.checked_mul(std::mem::size_of::<Term>()))
        .and_then(|headers| {
            old_len
                .checked_add(next)
                .and_then(|n| n.checked_mul(w))
                .and_then(|n| n.checked_mul(16))
                .and_then(|keys| headers.checked_add(keys))
        });
    match bytes {
        Some(b) if b <= o.max_workspace_bytes && b <= isize::MAX as usize => Ok(b),
        _ => Err(SparsePauliError::Memory {
            required_bytes: bytes,
            budget_bytes: o.max_workspace_bytes,
        }),
    }
}
fn pauli(p: Pauli) -> u8 {
    match p {
        Pauli::I => 0,
        Pauli::X => 1,
        Pauli::Z => 2,
        Pauli::Y => 3,
    }
}
fn get(k: &[u64], w: usize, q: u32) -> u8 {
    let i = q as usize / 64;
    let bit = q % 64;
    (((k[i] >> bit) & 1) | (((k[w + i] >> bit) & 1) << 1)) as u8
}
fn set(k: &mut [u64], w: usize, q: u32, p: u8) {
    let i = q as usize / 64;
    let mask = 1u64 << (q % 64);
    k[i] = (k[i] & !mask) | if p & 1 != 0 { mask } else { 0 };
    k[w + i] = (k[w + i] & !mask) | if p & 2 != 0 { mask } else { 0 };
}
fn anticommutes(a: u8, b: u8) -> bool {
    a != 0 && b != 0 && a != b
}
fn merge(v: &mut Vec<Term>) -> Result<usize> {
    v.sort_unstable_by(|a, b| a.key.cmp(&b.key));
    let mut kept = 0;
    let mut merges = 0;
    for i in 0..v.len() {
        if kept > 0 && v[kept - 1].key == v[i].key {
            v[kept - 1].coefficient = finite(v[kept - 1].coefficient + v[i].coefficient)?;
            merges += 1;
        } else {
            v.swap(kept, i);
            kept += 1;
        }
    }
    v.truncate(kept);
    v.retain(|t| t.coefficient != 0.);
    Ok(merges)
}
fn single(t: &mut Term, w: usize, kind: u8, q: u32) {
    let p = get(&t.key, w, q);
    let (out, negative) = match kind {
        0 => (
            match p {
                1 => 2,
                2 => 1,
                x => x,
            },
            p == 3,
        ), // H
        1 => (p, p & 2 != 0),       // X
        2 => (p, p == 1 || p == 2), // Y
        3 => (p, p & 1 != 0),       // Z
        4 => (
            match p {
                1 => 3,
                3 => 1,
                x => x,
            },
            p == 1,
        ), // S† P S
        _ => (
            match p {
                1 => 3,
                3 => 1,
                x => x,
            },
            p == 3,
        ), // S P S†
    };
    set(&mut t.key, w, q, out);
    if negative {
        t.coefficient = -t.coefficient;
    }
}
fn cx(t: &mut Term, w: usize, c: u32, q: u32) {
    let a = get(&t.key, w, c);
    let b = get(&t.key, w, q);
    // Aaronson-Gottesman signed Hermitian-Pauli CNOT rule (quant-ph/0406196).
    if a & 1 != 0 && b & 2 != 0 && ((b & 1) ^ ((a >> 1) & 1) ^ 1) != 0 {
        t.coefficient = -t.coefficient;
    }
    set(&mut t.key, w, c, a ^ (b & 2));
    set(&mut t.key, w, q, b ^ (a & 1));
}
fn clifford(t: &mut Term, w: usize, op: Op) {
    match op {
        Op::Single(k, q) => single(t, w, k, q),
        Op::Cx(c, q) => cx(t, w, c, q),
        Op::Cz(c, q) => {
            single(t, w, 0, q);
            cx(t, w, c, q);
            single(t, w, 0, q);
        }
        Op::Cy(c, q) => {
            single(t, w, 4, q);
            cx(t, w, c, q);
            single(t, w, 5, q);
        }
        Op::Swap(a, b) => {
            let x = get(&t.key, w, a);
            let y = get(&t.key, w, b);
            set(&mut t.key, w, a, y);
            set(&mut t.key, w, b, x);
        }
        Op::Iswap(a, b, dag) => {
            single(t, w, if dag { 5 } else { 4 }, a);
            single(t, w, if dag { 5 } else { 4 }, b);
            clifford(t, w, Op::Cz(a, b));
            clifford(t, w, Op::Swap(a, b));
        }
        Op::Rotation(..) => {}
    }
}
fn angle(p: Param) -> Result<f64> {
    match p {
        Param::Concrete(x) if x.is_finite() => Ok(x),
        _ => Err(invalid("rotation requires finite concrete angle")),
    }
}
fn rotation(axis: u8, q: u32, theta: f64) -> Op {
    // Only EXACT f64 multiples of pi/2 are classified. No tolerance/snapping of nearby angles.
    let quarter = std::f64::consts::FRAC_PI_2;
    let k = (theta / quarter).round();
    let (sin, cos) = if k.abs() <= 1_000_000. && theta == k * quarter {
        match (k as i64).rem_euclid(4) {
            0 => (0., 1.),
            1 => (1., 0.),
            2 => (0., -1.),
            _ => (-1., 0.),
        }
    } else {
        theta.sin_cos()
    };
    Op::Rotation(axis, q, cos, sin)
}
fn gate(g: &GateInstance, n: u32) -> Result<Op> {
    if g.qubits.is_empty() || g.qubits.len() != g.gate.arity() {
        return Err(invalid("gate arity"));
    }
    let qs: Vec<u32> = g.qubits.iter().chain(&g.controls).copied().collect();
    if qs.iter().any(|q| *q >= n) || qs.iter().enumerate().any(|(i, q)| qs[..i].contains(q)) {
        return Err(invalid("gate qubit range/duplicates"));
    }
    if !g.controls.is_empty() {
        if g.controls.len() != 1 {
            return Err(invalid("only single external Pauli controls supported"));
        }
        return match g.gate {
            Gate::X => Ok(Op::Cx(g.controls[0], g.qubits[0])),
            Gate::Y => Ok(Op::Cy(g.controls[0], g.qubits[0])),
            Gate::Z => Ok(Op::Cz(g.controls[0], g.qubits[0])),
            _ => Err(invalid("unsupported external control")),
        };
    }
    let q = g.qubits[0];
    Ok(match g.gate {
        Gate::H => Op::Single(0, q),
        Gate::X => Op::Single(1, q),
        Gate::Y => Op::Single(2, q),
        Gate::Z => Op::Single(3, q),
        Gate::S => Op::Single(4, q),
        Gate::Sdg => Op::Single(5, q),
        Gate::Cnot => Op::Cx(q, g.qubits[1]),
        Gate::Cz => Op::Cz(q, g.qubits[1]),
        Gate::Swap => Op::Swap(q, g.qubits[1]),
        Gate::Iswap => Op::Iswap(q, g.qubits[1], false),
        Gate::IswapDg => Op::Iswap(q, g.qubits[1], true),
        Gate::Rx(p) => rotation(1, q, angle(p)?),
        Gate::Ry(p) => rotation(3, q, angle(p)?),
        Gate::Rz(p) | Gate::Phase(p) => rotation(2, q, angle(p)?),
        Gate::T => rotation(2, q, std::f64::consts::FRAC_PI_4),
        Gate::Tdg => rotation(2, q, -std::f64::consts::FRAC_PI_4),
        _ => return Err(invalid(format!("unsupported gate {}", g.gate.name()))),
    })
}
fn operations(c: &Circuit) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for inst in c.instructions() {
        match inst {
            Instruction::Gate(g) => ops.push(gate(g, c.num_qubits())?),
            Instruction::TiledBlock(t) => {
                for g in &t.gates {
                    ops.push(gate(g, c.num_qubits())?);
                }
            }
            Instruction::Barrier(qs)
                if !qs.is_empty() && qs.iter().all(|q| *q < c.num_qubits()) => {}
            _ => {
                return Err(invalid(
                    "unsupported instruction (measure/reset/fused diagonal or invalid barrier)",
                ))
            }
        }
    }
    Ok(ops)
}
