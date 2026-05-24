//! `Circuit` — the IR's top-level container.
//!
//! `instructions` is private so a future DAG refactor stays
//! non-breaking. Access is via `instructions()`, `len()`,
//! `is_empty()`, and the `layers()` helper (see `layers.rs`).

use aleph_core::{Gate, GateInstance};

use crate::CircuitError;
use crate::Instruction;

/// Backend-agnostic circuit representation.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub num_qubits: u32,
    pub num_clbits: u32,
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
    pub fn new(num_qubits: u32, num_clbits: u32) -> Self {
        Self {
            num_qubits,
            num_clbits,
            instructions: Vec::new(),
            metadata: CircuitMetadata::default(),
        }
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
    pub fn add_instruction(
        &mut self,
        inst: Instruction,
    ) -> Result<&mut Self, CircuitError> {
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
        for &q in gate.qubits.iter().chain(gate.controls.iter()) {
            self.check_qubit(q)?;
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
        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.num_clbits, 2);
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
