//! Error type for gate operations.

/// Errors that can arise when querying a gate's matrix.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GateError {
    /// `Gate::matrix()` was called on a gate containing a
    /// `Param::Symbolic` parameter. Unreachable through the public API
    /// in Phase 0; reserved for Phase 4 (VQE).
    #[error("symbolic parameter cannot produce a concrete matrix")]
    SymbolicParam,
    /// `Gate::matrix()` was called on a gate containing a NaN or
    /// infinite parameter. Without this guard, downstream cos/sin
    /// silently produce all-NaN matrices that propagate through the
    /// state vector with no diagnostic.
    #[error("parameter must be finite (was NaN or infinite)")]
    NonFiniteParam,
}
