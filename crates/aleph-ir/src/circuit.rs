//! `Circuit` — the IR's top-level container.
//!
//! `instructions` is private so a future DAG refactor stays
//! non-breaking. Access is via `instructions()`, `len()`,
//! `is_empty()`, and the `layers()` helper (see `layers.rs`).

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
