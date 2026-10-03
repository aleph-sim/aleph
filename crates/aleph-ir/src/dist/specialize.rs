//! Per-rank rewriting of physical instructions (filled in Task 2).
use super::{DistError, DistLayout};
use crate::Instruction;

/// Rewrite physical `instr` for `rank`; `None` = no-op on this rank.
pub fn specialize(
    _instr: &Instruction,
    _layout: DistLayout,
    _rank: u32,
) -> Result<Option<Instruction>, DistError> {
    Ok(None)
}
