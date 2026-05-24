//! `Circuit` — the IR's top-level container.
//!
//! `instructions`, `num_qubits`, and `num_clbits` are private so a
//! future DAG refactor and invariant changes stay non-breaking. Access
//! is via `instructions()`, `num_qubits()`, `num_clbits()`, `len()`,
//! `is_empty()`, and the `layers()` helper (see `layers.rs`).

use aleph_core::{Gate, GateInstance, Param};

use crate::CircuitError;
use crate::Instruction;

/// Maximum number of qubits a single `Circuit` may declare. Bounds the
/// O(num_qubits) bookkeeping in `extract_layers` and prevents
/// pathological allocations from inputs like `qreg q[u32::MAX]`.
///
/// 65,535 is comfortably above anything a Phase-0 backend can simulate
/// while keeping `layers()`'s `Vec<Option<(usize,usize)>>` allocation
/// at ~1.5 MB worst case. A fallible `Circuit::try_new` that returns
/// `CircuitError` instead of panicking will land alongside the parser
/// in P0-08 (where untrusted input actually enters the system).
pub const MAX_QUBITS: u32 = 65_535;

/// Maximum number of classical bits a single `Circuit` may declare.
/// Same rationale as [`MAX_QUBITS`].
pub const MAX_CLBITS: u32 = 65_535;

/// Maximum number of external `controls` a single `GateInstance` may
/// carry. Validating uniqueness across `qubits ∪ controls` is O(N²)
/// over the chained iterator (linear-scan `SmallVec::contains`); an
/// adversarially-large `controls` list would otherwise hang the IR.
/// Phase-0 gates use 0–2 controls (`controlled` constructions in
/// `aleph-core` cap inline at 2). The bound is generous and prevents
/// unbounded work from `pub`-field literal construction.
pub const MAX_GATE_CONTROLS: usize = 8;

/// Backend-agnostic circuit representation.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub(crate) num_qubits: u32,
    pub(crate) num_clbits: u32,
    pub(crate) instructions: Vec<Instruction>,
    metadata: CircuitMetadata,
}

/// Optional metadata attached to a `Circuit` — kept tiny on purpose.
#[derive(Debug, Clone, Default)]
pub struct CircuitMetadata {
    pub name: Option<String>,
    pub generated_from: Option<String>,
}

impl Circuit {
    /// Construct an empty circuit with the given qubit/clbit capacity.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > MAX_QUBITS` or `num_clbits > MAX_CLBITS`.
    /// This is a programmer-error check at construction; untrusted
    /// callers (parser, RPC) should sanity-check inputs before calling.
    /// A fallible `try_new` returning [`CircuitError`] lands in P0-08
    /// alongside the OpenQASM parser, where untrusted input actually
    /// enters the system.
    pub fn new(num_qubits: u32, num_clbits: u32) -> Self {
        assert!(
            num_qubits <= MAX_QUBITS,
            "Circuit::new: num_qubits={num_qubits} exceeds MAX_QUBITS={MAX_QUBITS}",
        );
        assert!(
            num_clbits <= MAX_CLBITS,
            "Circuit::new: num_clbits={num_clbits} exceeds MAX_CLBITS={MAX_CLBITS}",
        );
        Self {
            num_qubits,
            num_clbits,
            instructions: Vec::new(),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Fallible constructor — same as [`Circuit::new`] but returns a
    /// recoverable [`CircuitError`] instead of panicking. Intended for
    /// untrusted-input boundaries (parser, RPC). See spec § 12.4 of
    /// `docs/superpowers/specs/2026-05-24-p0-07-circuit-ir-design.md`.
    pub fn try_new(num_qubits: u32, num_clbits: u32) -> Result<Self, CircuitError> {
        if num_qubits > MAX_QUBITS {
            return Err(CircuitError::TooManyQubits {
                requested: num_qubits,
                max: MAX_QUBITS,
            });
        }
        if num_clbits > MAX_CLBITS {
            return Err(CircuitError::TooManyClbits {
                requested: num_clbits,
                max: MAX_CLBITS,
            });
        }
        Ok(Self {
            num_qubits,
            num_clbits,
            instructions: Vec::new(),
            metadata: CircuitMetadata::default(),
        })
    }

    /// Number of qubits this circuit was constructed with. Immutable
    /// for the lifetime of the `Circuit`.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Number of classical bits this circuit was constructed with.
    /// Immutable for the lifetime of the `Circuit`.
    pub fn num_clbits(&self) -> u32 {
        self.num_clbits
    }

    /// Set the circuit's display name. Consuming — intended for the
    /// init chain (`Circuit::new(2, 2).with_name("bell")`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.name = Some(name.into());
        self
    }

    /// Record where the circuit came from (e.g. `"openqasm:3.0"`).
    /// Consuming — intended for the init chain.
    pub fn with_generated_from(mut self, source: impl Into<String>) -> Self {
        self.metadata.generated_from = Some(source.into());
        self
    }

    /// Access the optional metadata.
    pub fn metadata(&self) -> &CircuitMetadata {
        &self.metadata
    }

    /// Slice over all instructions in execution order.
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Number of instructions in the circuit.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Whether the circuit contains no instructions.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Append a `GateInstance` after validating qubit ranges and arity.
    pub fn add_gate(&mut self, gate: GateInstance) -> Result<&mut Self, CircuitError> {
        self.validate_gate(&gate)?;
        self.instructions.push(Instruction::Gate(gate));
        Ok(self)
    }

    /// Append any `Instruction` after validating its qubits/clbits.
    pub fn add_instruction(&mut self, inst: Instruction) -> Result<&mut Self, CircuitError> {
        self.validate_instruction(&inst)?;
        self.instructions.push(inst);
        Ok(self)
    }

    // --- validation helpers ---

    fn check_qubit(&self, q: u32) -> Result<(), CircuitError> {
        if q >= self.num_qubits {
            Err(CircuitError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.num_qubits,
            })
        } else {
            Ok(())
        }
    }

    fn check_clbit(&self, c: u32) -> Result<(), CircuitError> {
        if c >= self.num_clbits {
            Err(CircuitError::ClbitOutOfRange {
                clbit: c,
                num_clbits: self.num_clbits,
            })
        } else {
            Ok(())
        }
    }

    /// Validate a `GateInstance` against the circuit's qubit count.
    ///
    /// Checks (in this order): arity, per-qubit range, and uniqueness
    /// across `qubits ∪ controls`. The uniqueness check is the IR's
    /// release-build safety net — `GateInstance::new`'s own check is
    /// `debug_assert`-gated and inert in release builds.
    ///
    /// Extra external `controls` beyond what the base gate semantically
    /// expects are admitted: the IR treats `GateInstance::controlled`
    /// as a generic mechanism, leaving backend-specific "is this a
    /// sensible control set for `Gate::X`?" decisions to the backend.
    fn validate_gate(&self, gate: &GateInstance) -> Result<(), CircuitError> {
        let expected = gate.gate.arity();
        let got = gate.qubits.len();
        if expected != got {
            return Err(CircuitError::ArityMismatch {
                gate: gate_variant_name(&gate.gate),
                expected,
                got,
            });
        }
        if gate.controls.len() > MAX_GATE_CONTROLS {
            return Err(CircuitError::TooManyControls {
                gate: gate_variant_name(&gate.gate),
                controls: gate.controls.len(),
                max: MAX_GATE_CONTROLS,
            });
        }
        let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            self.check_qubit(q)?;
            if seen.contains(&q) {
                return Err(CircuitError::DuplicateQubit { qubit: q });
            }
            seen.push(q);
        }
        Ok(())
    }

    fn validate_instruction(&self, inst: &Instruction) -> Result<(), CircuitError> {
        match inst {
            Instruction::Gate(g) => self.validate_gate(g),
            Instruction::Measure { qubit, clbit } => {
                self.check_qubit(*qubit)?;
                self.check_clbit(*clbit)
            }
            Instruction::Reset(q) => self.check_qubit(*q),
            Instruction::Barrier(qs) => {
                if qs.is_empty() {
                    return Err(CircuitError::EmptyBarrier);
                }
                let mut seen: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
                for &q in qs {
                    self.check_qubit(q)?;
                    if seen.contains(&q) {
                        return Err(CircuitError::DuplicateQubit { qubit: q });
                    }
                    seen.push(q);
                }
                Ok(())
            }
        }
    }

    // --- 1q standard convenience ---

    /// Append `H` to qubit `q`.
    pub fn h(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::H, smallvec::smallvec![q]))
    }

    /// Append `X` to qubit `q`.
    pub fn x(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::X, smallvec::smallvec![q]))
    }

    /// Append `Y` to qubit `q`.
    pub fn y(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Y, smallvec::smallvec![q]))
    }

    /// Append `Z` to qubit `q`.
    pub fn z(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Z, smallvec::smallvec![q]))
    }

    /// Append `S` to qubit `q`.
    pub fn s(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::S, smallvec::smallvec![q]))
    }

    /// Append `Sdg` to qubit `q`.
    pub fn sdg(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Sdg, smallvec::smallvec![q]))
    }

    /// Append `T` to qubit `q`.
    pub fn t(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::T, smallvec::smallvec![q]))
    }

    /// Append `Tdg` to qubit `q`.
    pub fn tdg(&mut self, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Tdg, smallvec::smallvec![q]))
    }

    // --- 1q parametric convenience ---

    /// Append `Rx(θ)` to qubit `q`.
    pub fn rx(&mut self, theta: f64, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Rx(Param::Concrete(theta)),
            smallvec::smallvec![q],
        ))
    }

    /// Append `Ry(θ)` to qubit `q`.
    pub fn ry(&mut self, theta: f64, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Ry(Param::Concrete(theta)),
            smallvec::smallvec![q],
        ))
    }

    /// Append `Rz(θ)` to qubit `q`.
    pub fn rz(&mut self, theta: f64, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Rz(Param::Concrete(theta)),
            smallvec::smallvec![q],
        ))
    }

    /// Append `Phase(θ)` (= `diag(1, e^{iθ})`) to qubit `q`.
    pub fn phase(&mut self, theta: f64, q: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Phase(Param::Concrete(theta)),
            smallvec::smallvec![q],
        ))
    }

    /// Append `U3(θ, φ, λ)` to qubit `q` (Qiskit convention).
    pub fn u3(
        &mut self,
        theta: f64,
        phi: f64,
        lambda: f64,
        q: u32,
    ) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::U3(
                Param::Concrete(theta),
                Param::Concrete(phi),
                Param::Concrete(lambda),
            ),
            smallvec::smallvec![q],
        ))
    }

    // --- 2q convenience ---

    /// Append `Cnot` with `qubits = [control, target]`.
    pub fn cnot(&mut self, control: u32, target: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Cnot,
            smallvec::smallvec![control, target],
        ))
    }

    /// Append `Cz` on `(q0, q1)`. Symmetric — qubit order does not matter.
    pub fn cz(&mut self, q0: u32, q1: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Cz, smallvec::smallvec![q0, q1]))
    }

    /// Append `Swap` on `(q0, q1)`.
    pub fn swap(&mut self, q0: u32, q1: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(Gate::Swap, smallvec::smallvec![q0, q1]))
    }

    // --- 3q convenience ---

    /// Append `Toffoli` (CCX) with `qubits = [c0, c1, target]`.
    pub fn ccx(&mut self, c0: u32, c1: u32, target: u32) -> Result<&mut Self, CircuitError> {
        self.add_gate(GateInstance::new(
            Gate::Toffoli,
            smallvec::smallvec![c0, c1, target],
        ))
    }

    // --- Non-gate convenience ---

    /// Measure `qubit` into `clbit`.
    pub fn measure(&mut self, qubit: u32, clbit: u32) -> Result<&mut Self, CircuitError> {
        self.add_instruction(Instruction::Measure { qubit, clbit })
    }

    /// Reset `qubit` to `|0⟩`.
    pub fn reset(&mut self, qubit: u32) -> Result<&mut Self, CircuitError> {
        self.add_instruction(Instruction::Reset(qubit))
    }

    /// Insert a barrier covering `qubits`. Accepts any iterator.
    ///
    /// Returns:
    /// - [`CircuitError::EmptyBarrier`] if `qubits` yields no items
    ///   (a barrier with nothing to synchronize is rejected).
    /// - [`CircuitError::DuplicateQubit`] if the same qubit appears twice.
    /// - [`CircuitError::QubitOutOfRange`] if any index ≥ `num_qubits()`.
    ///
    /// Callers building the qubit list via a filter (`.iter().filter(...)`)
    /// should check for emptiness up front if a zero-match result is
    /// semantically a no-op rather than an error.
    pub fn barrier(
        &mut self,
        qubits: impl IntoIterator<Item = u32>,
    ) -> Result<&mut Self, CircuitError> {
        let qs: smallvec::SmallVec<[u32; 8]> = qubits.into_iter().collect();
        self.add_instruction(Instruction::Barrier(qs))
    }

    /// Group instruction indices into layers of (logically) parallel
    /// instructions. See `crate::layers` for the algorithm.
    pub fn layers(&self) -> Vec<Vec<usize>> {
        crate::layers::extract_layers(self)
    }
}

/// Stable string name for each `Gate` variant — used in
/// `CircuitError::ArityMismatch`.
fn gate_variant_name(g: &Gate) -> &'static str {
    match g {
        Gate::H => "H",
        Gate::X => "X",
        Gate::Y => "Y",
        Gate::Z => "Z",
        Gate::S => "S",
        Gate::Sdg => "Sdg",
        Gate::T => "T",
        Gate::Tdg => "Tdg",
        Gate::Rx(_) => "Rx",
        Gate::Ry(_) => "Ry",
        Gate::Rz(_) => "Rz",
        Gate::Phase(_) => "Phase",
        Gate::U3(_, _, _) => "U3",
        Gate::Cnot => "Cnot",
        Gate::Cz => "Cz",
        Gate::Swap => "Swap",
        Gate::Iswap => "Iswap",
        Gate::IswapDg => "IswapDg",
        Gate::CRx(_) => "CRx",
        Gate::CRy(_) => "CRy",
        Gate::CRz(_) => "CRz",
        Gate::Toffoli => "Toffoli",
        Gate::Ccz => "Ccz",
        Gate::Unitary1q(_) => "Unitary1q",
        Gate::Unitary2q(_) => "Unitary2q",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CircuitError;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn h_gate(q: u32) -> GateInstance {
        GateInstance::new(Gate::H, smallvec![q])
    }

    #[test]
    fn add_gate_accepts_in_range() {
        let mut c = Circuit::new(2, 0);
        let r = c.add_gate(h_gate(1));
        assert!(r.is_ok());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn add_gate_rejects_oob_qubit() {
        let mut c = Circuit::new(2, 0);
        let err = c.add_gate(h_gate(5)).unwrap_err();
        assert_eq!(
            err,
            CircuitError::QubitOutOfRange {
                qubit: 5,
                num_qubits: 2
            }
        );
        assert!(c.is_empty(), "circuit must not be mutated on error");
    }

    #[test]
    fn add_gate_rejects_arity_mismatch() {
        let bad = GateInstance {
            gate: Gate::Cnot,
            qubits: smallvec![0u32],
            controls: smallvec![],
        };
        let mut c = Circuit::new(2, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(
            err,
            CircuitError::ArityMismatch {
                gate: "Cnot",
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn add_instruction_barrier_rejects_oob() {
        let mut c = Circuit::new(2, 0);
        let err = c
            .add_instruction(Instruction::Barrier(smallvec![0u32, 7u32]))
            .unwrap_err();
        assert_eq!(
            err,
            CircuitError::QubitOutOfRange {
                qubit: 7,
                num_qubits: 2
            }
        );
    }

    #[test]
    fn add_instruction_barrier_rejects_duplicate() {
        let mut c = Circuit::new(3, 0);
        let err = c
            .add_instruction(Instruction::Barrier(smallvec![1u32, 1u32]))
            .unwrap_err();
        assert_eq!(err, CircuitError::DuplicateQubit { qubit: 1 });
    }

    #[test]
    fn add_instruction_measure_validates_both_ranges() {
        let mut c = Circuit::new(2, 1);
        assert!(matches!(
            c.add_instruction(Instruction::Measure { qubit: 9, clbit: 0 }),
            Err(CircuitError::QubitOutOfRange { qubit: 9, .. })
        ));
        assert!(matches!(
            c.add_instruction(Instruction::Measure { qubit: 0, clbit: 9 }),
            Err(CircuitError::ClbitOutOfRange { clbit: 9, .. })
        ));
        assert!(c.is_empty());
    }

    #[test]
    fn new_is_empty() {
        let c = Circuit::new(3, 2);
        assert_eq!(c.num_qubits(), 3);
        assert_eq!(c.num_clbits(), 2);
        assert!(c.instructions.is_empty());
        assert!(c.metadata.name.is_none());
        assert!(c.metadata.generated_from.is_none());
    }

    #[test]
    fn with_name_sets_metadata() {
        let c = Circuit::new(2, 0).with_name("bell");
        assert_eq!(c.metadata().name.as_deref(), Some("bell"));
    }

    #[test]
    fn with_generated_from_sets_metadata() {
        let c = Circuit::new(0, 0).with_generated_from("openqasm:3.0");
        assert_eq!(c.metadata().generated_from.as_deref(), Some("openqasm:3.0"));
    }

    #[test]
    fn h_appends_h_gate() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.gate, Gate::H);
                assert_eq!(g.qubits.as_slice(), &[0]);
            }
            other => panic!("expected Gate, got {other:?}"),
        }
    }

    #[test]
    fn h_rejects_oob() {
        let mut c = Circuit::new(2, 0);
        assert!(matches!(
            c.h(5),
            Err(CircuitError::QubitOutOfRange { qubit: 5, .. })
        ));
    }

    #[test]
    fn one_qubit_standard_set_all_work() {
        let mut c = Circuit::new(1, 0);
        c.h(0).unwrap();
        c.x(0).unwrap();
        c.y(0).unwrap();
        c.z(0).unwrap();
        c.s(0).unwrap();
        c.sdg(0).unwrap();
        c.t(0).unwrap();
        c.tdg(0).unwrap();
        let variants: Vec<&Gate> = c
            .instructions()
            .iter()
            .map(|i| match i {
                Instruction::Gate(g) => &g.gate,
                _ => panic!("expected Gate"),
            })
            .collect();
        assert_eq!(
            variants,
            vec![
                &Gate::H,
                &Gate::X,
                &Gate::Y,
                &Gate::Z,
                &Gate::S,
                &Gate::Sdg,
                &Gate::T,
                &Gate::Tdg,
            ]
        );
    }

    #[test]
    fn h_chains_with_question_mark() -> Result<(), CircuitError> {
        let mut c = Circuit::new(2, 0);
        c.h(0)?.x(1)?;
        assert_eq!(c.len(), 2);
        Ok(())
    }

    #[test]
    fn rx_records_angle() {
        let mut c = Circuit::new(1, 0);
        c.rx(0.5, 0).unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.gate, Gate::Rx(Param::Concrete(0.5)));
                assert_eq!(g.qubits.as_slice(), &[0]);
            }
            other => panic!("expected Gate, got {other:?}"),
        }
    }

    #[test]
    fn u3_records_all_three_angles() {
        let mut c = Circuit::new(1, 0);
        c.u3(0.1, 0.2, 0.3, 0).unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(
                    g.gate,
                    Gate::U3(
                        Param::Concrete(0.1),
                        Param::Concrete(0.2),
                        Param::Concrete(0.3),
                    )
                );
            }
            other => panic!("expected Gate, got {other:?}"),
        }
    }

    #[test]
    fn parametric_one_qubit_set_all_work() {
        let mut c = Circuit::new(1, 0);
        c.rx(0.1, 0).unwrap();
        c.ry(0.2, 0).unwrap();
        c.rz(0.3, 0).unwrap();
        c.phase(0.4, 0).unwrap();
        c.u3(0.5, 0.6, 0.7, 0).unwrap();
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn rx_rejects_oob() {
        let mut c = Circuit::new(2, 0);
        assert!(matches!(
            c.rx(0.5, 9),
            Err(CircuitError::QubitOutOfRange { qubit: 9, .. })
        ));
    }

    #[test]
    fn cnot_records_qubit_order_control_target() {
        let mut c = Circuit::new(2, 0);
        c.cnot(0, 1).unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.gate, Gate::Cnot);
                assert_eq!(g.qubits.as_slice(), &[0, 1]);
            }
            other => panic!("expected Gate, got {other:?}"),
        }
    }

    #[test]
    fn cz_and_swap_record_qubits() {
        let mut c = Circuit::new(3, 0);
        c.cz(0, 2).unwrap();
        c.swap(1, 2).unwrap();
        let g0 = match &c.instructions()[0] {
            Instruction::Gate(g) => g,
            _ => panic!(),
        };
        assert_eq!(g0.gate, Gate::Cz);
        assert_eq!(g0.qubits.as_slice(), &[0, 2]);
        let g1 = match &c.instructions()[1] {
            Instruction::Gate(g) => g,
            _ => panic!(),
        };
        assert_eq!(g1.gate, Gate::Swap);
        assert_eq!(g1.qubits.as_slice(), &[1, 2]);
    }

    #[test]
    fn cnot_rejects_oob_target() {
        let mut c = Circuit::new(2, 0);
        assert!(matches!(
            c.cnot(0, 9),
            Err(CircuitError::QubitOutOfRange { qubit: 9, .. })
        ));
    }

    #[test]
    fn ccx_records_toffoli_with_correct_qubit_order() {
        let mut c = Circuit::new(3, 0);
        c.ccx(0, 1, 2).unwrap();
        match &c.instructions()[0] {
            Instruction::Gate(g) => {
                assert_eq!(g.gate, Gate::Toffoli);
                assert_eq!(g.qubits.as_slice(), &[0, 1, 2]);
            }
            other => panic!("expected Gate, got {other:?}"),
        }
    }

    #[test]
    fn ccx_rejects_oob_target() {
        let mut c = Circuit::new(3, 0);
        assert!(matches!(
            c.ccx(0, 1, 9),
            Err(CircuitError::QubitOutOfRange { qubit: 9, .. })
        ));
    }

    #[test]
    fn measure_records_qubit_and_clbit() {
        let mut c = Circuit::new(2, 2);
        c.measure(1, 0).unwrap();
        match c.instructions()[0] {
            Instruction::Measure { qubit, clbit } => {
                assert_eq!(qubit, 1);
                assert_eq!(clbit, 0);
            }
            ref other => panic!("expected Measure, got {other:?}"),
        }
    }

    #[test]
    fn reset_records_qubit() {
        let mut c = Circuit::new(2, 0);
        c.reset(1).unwrap();
        assert!(matches!(c.instructions()[0], Instruction::Reset(1)));
    }

    #[test]
    fn barrier_accepts_iterator_input() {
        let mut c = Circuit::new(3, 0);
        c.barrier([0u32, 2u32]).unwrap();
        match &c.instructions()[0] {
            Instruction::Barrier(qs) => assert_eq!(qs.as_slice(), &[0, 2]),
            other => panic!("expected Barrier, got {other:?}"),
        }
    }

    #[test]
    fn barrier_accepts_vec_input() {
        let mut c = Circuit::new(3, 0);
        c.barrier(vec![0u32, 1u32, 2u32]).unwrap();
        match &c.instructions()[0] {
            Instruction::Barrier(qs) => assert_eq!(qs.as_slice(), &[0, 1, 2]),
            other => panic!("expected Barrier, got {other:?}"),
        }
    }

    #[test]
    fn measure_rejects_oob_qubit() {
        let mut c = Circuit::new(2, 2);
        assert!(matches!(
            c.measure(9, 0),
            Err(CircuitError::QubitOutOfRange { qubit: 9, .. })
        ));
    }

    #[test]
    fn measure_rejects_oob_clbit() {
        let mut c = Circuit::new(2, 2);
        assert!(matches!(
            c.measure(0, 9),
            Err(CircuitError::ClbitOutOfRange { clbit: 9, .. })
        ));
    }

    #[test]
    fn barrier_rejects_duplicate() {
        let mut c = Circuit::new(3, 0);
        assert!(matches!(
            c.barrier([1u32, 1u32]),
            Err(CircuitError::DuplicateQubit { qubit: 1 })
        ));
    }

    #[test]
    fn barrier_rejects_empty() {
        let mut c = Circuit::new(2, 0);
        assert!(matches!(
            c.barrier(std::iter::empty::<u32>()),
            Err(CircuitError::EmptyBarrier)
        ));
        assert!(c.is_empty(), "circuit must not be mutated on error");
    }

    #[test]
    fn add_gate_rejects_duplicate_qubit_cnot() {
        // Cnot(0, 0): control == target — ill-defined. GateInstance's
        // own check is debug-only; the IR must catch this in release.
        let bad = GateInstance {
            gate: Gate::Cnot,
            qubits: smallvec![0u32, 0u32],
            controls: smallvec![],
        };
        let mut c = Circuit::new(2, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(err, CircuitError::DuplicateQubit { qubit: 0 });
        assert!(c.is_empty());
    }

    #[test]
    fn add_gate_rejects_duplicate_qubit_toffoli() {
        let bad = GateInstance {
            gate: Gate::Toffoli,
            qubits: smallvec![1u32, 0u32, 1u32],
            controls: smallvec![],
        };
        let mut c = Circuit::new(3, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(err, CircuitError::DuplicateQubit { qubit: 1 });
    }

    #[test]
    fn add_gate_rejects_qubit_control_overlap() {
        let bad = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: smallvec![0u32],
        };
        let mut c = Circuit::new(2, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(err, CircuitError::DuplicateQubit { qubit: 0 });
    }

    #[test]
    fn add_gate_rejects_oob_control() {
        let bad = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: smallvec![9u32],
        };
        let mut c = Circuit::new(2, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(
            err,
            CircuitError::QubitOutOfRange {
                qubit: 9,
                num_qubits: 2
            }
        );
    }

    #[test]
    fn add_gate_rejects_duplicate_within_controls() {
        // Duplicate appears only inside `controls`; qubits is fine
        // and qubits ∩ controls is empty. Exercises the within-controls
        // branch of the unified uniqueness loop.
        let bad = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: smallvec![1u32, 1u32],
        };
        let mut c = Circuit::new(3, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(err, CircuitError::DuplicateQubit { qubit: 1 });
    }

    #[test]
    fn add_gate_rejects_too_many_controls() {
        let bad = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: (1u32..1 + (crate::circuit::MAX_GATE_CONTROLS as u32) + 1).collect(),
        };
        let mut c = Circuit::new(64, 0);
        let err = c.add_gate(bad).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TooManyControls {
                gate: "X",
                controls: crate::circuit::MAX_GATE_CONTROLS + 1,
                max: crate::circuit::MAX_GATE_CONTROLS,
            }
        );
    }

    #[test]
    fn add_gate_accepts_max_gate_controls() {
        // Boundary: exactly MAX_GATE_CONTROLS must succeed.
        let ok = GateInstance {
            gate: Gate::X,
            qubits: smallvec![0u32],
            controls: (1u32..1 + (crate::circuit::MAX_GATE_CONTROLS as u32)).collect(),
        };
        let mut c = Circuit::new(64, 0);
        assert!(c.add_gate(ok).is_ok());
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_QUBITS")]
    fn new_panics_on_too_many_qubits() {
        let _ = Circuit::new(crate::circuit::MAX_QUBITS + 1, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_CLBITS")]
    fn new_panics_on_too_many_clbits() {
        let _ = Circuit::new(0, crate::circuit::MAX_CLBITS + 1);
    }

    #[test]
    fn new_accepts_max_qubits() {
        // Boundary: exactly MAX_QUBITS must succeed.
        let c = Circuit::new(crate::circuit::MAX_QUBITS, 0);
        assert_eq!(c.num_qubits(), crate::circuit::MAX_QUBITS);
    }

    #[test]
    fn new_accepts_zero_qubits_zero_clbits() {
        // Empty circuit must build and produce empty layers.
        let c = Circuit::new(0, 0);
        assert_eq!(c.num_qubits(), 0);
        assert_eq!(c.num_clbits(), 0);
        assert!(c.is_empty());
        assert_eq!(c.layers(), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn try_new_accepts_max_bounds() {
        let c = Circuit::try_new(MAX_QUBITS, MAX_CLBITS).unwrap();
        assert_eq!(c.num_qubits(), MAX_QUBITS);
        assert_eq!(c.num_clbits(), MAX_CLBITS);
    }

    #[test]
    fn try_new_rejects_too_many_qubits() {
        let err = Circuit::try_new(MAX_QUBITS + 1, 0).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TooManyQubits {
                requested: MAX_QUBITS + 1,
                max: MAX_QUBITS,
            }
        );
    }

    #[test]
    fn try_new_rejects_too_many_clbits() {
        let err = Circuit::try_new(0, MAX_CLBITS + 1).unwrap_err();
        assert_eq!(
            err,
            CircuitError::TooManyClbits {
                requested: MAX_CLBITS + 1,
                max: MAX_CLBITS,
            }
        );
    }

    #[test]
    fn try_new_zero_zero_works() {
        let c = Circuit::try_new(0, 0).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn layers_wrapper_matches_extract_layers() {
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        c.cnot(0, 1).unwrap();
        let via_method = c.layers();
        let via_helper = crate::layers::extract_layers(&c);
        assert_eq!(via_method, via_helper);
    }

    #[test]
    fn instructions_empty_on_new() {
        let c = Circuit::new(2, 0);
        assert_eq!(c.instructions().len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn instructions_reports_pushed() {
        use crate::Instruction;
        let mut c = Circuit::new(2, 0);
        c.instructions.push(Instruction::Reset(0));
        c.instructions.push(Instruction::Reset(1));
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
        assert!(matches!(c.instructions()[0], Instruction::Reset(0)));
        assert!(matches!(c.instructions()[1], Instruction::Reset(1)));
    }

    #[test]
    fn with_chain_is_idempotent() {
        let c = Circuit::new(1, 1)
            .with_name("a")
            .with_generated_from("b")
            .with_name("c");
        assert_eq!(c.metadata().name.as_deref(), Some("c"));
        assert_eq!(c.metadata().generated_from.as_deref(), Some("b"));
    }
}
