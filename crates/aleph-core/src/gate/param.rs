//! Gate parameter type. Phase 0 only constructs `Param::Concrete`;
//! `Param::Symbolic` has no public constructor and is reserved for
//! Phase 4 (VQE / parametrized circuits).

/// A gate parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param {
    /// Concrete real-valued parameter (angle in radians for rotations).
    Concrete(f64),
    /// Placeholder for a symbolic parameter resolved at execution time.
    /// No public constructor in Phase 0 — encountering this variant in
    /// `Gate::matrix()` returns `GateError::SymbolicParam`.
    Symbolic(SymbolId),
}

/// Opaque identifier for a symbolic parameter. Constructor is
/// crate-private so external code cannot synthesize a `Symbolic` param
/// in Phase 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub(crate) u32);

impl From<f64> for Param {
    fn from(v: f64) -> Self {
        Param::Concrete(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_yields_concrete() {
        let p: Param = 1.5.into();
        assert_eq!(p, Param::Concrete(1.5));
    }

    #[test]
    fn equality() {
        assert_eq!(Param::Concrete(0.0), Param::Concrete(0.0));
        assert_ne!(Param::Concrete(0.0), Param::Concrete(1.0));
        assert_eq!(Param::Symbolic(SymbolId(7)), Param::Symbolic(SymbolId(7)));
        assert_ne!(Param::Symbolic(SymbolId(7)), Param::Symbolic(SymbolId(8)));
        assert_ne!(Param::Concrete(0.0), Param::Symbolic(SymbolId(0)));
    }
}
