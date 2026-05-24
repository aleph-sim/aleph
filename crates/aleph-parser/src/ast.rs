//! Private AST for the OpenQASM 3 subset.
//!
//! Every node carries a `Position` (line, col) so lowering can emit
//! `ParseError`s with accurate source locations. Expression results
//! are evaluated to `f64` at parse time (per spec § 9), so the AST
//! stores already-evaluated parameters — not raw expression strings.

/// 1-based source position captured at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub header_version: Option<String>,
    pub includes: Vec<Include>,
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Include {
    pub pos: Position,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Qreg {
        pos: Position,
        name: String,
        size: u32,
    },
    Creg {
        pos: Position,
        name: String,
        size: u32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Gate(GateStmt),
    Barrier(BarrierStmt),
    Measure(MeasureStmt),
    Reset(ResetStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateStmt {
    pub pos: Position,
    pub name: String,
    pub params: Vec<f64>,
    pub args: Vec<IndexedRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BarrierStmt {
    pub pos: Position,
    pub args: Vec<RegOrIdx>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasureStmt {
    pub pos: Position,
    pub source: RegOrIdx,
    pub target: RegOrIdx,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetStmt {
    pub pos: Position,
    pub target: IndexedRef,
}

/// A `name[index]` reference. Used by gate args (which require an index)
/// and as one variant of [`RegOrIdx`].
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedRef {
    pub pos: Position,
    pub name: String,
    pub index: u32,
}

/// Either a whole register (`name`) or an indexed slot (`name[index]`).
/// Used by `barrier` and `measure` where both forms are legal.
#[derive(Debug, Clone, PartialEq)]
pub enum RegOrIdx {
    Whole { pos: Position, name: String },
    Indexed(IndexedRef),
}
