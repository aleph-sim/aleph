//! Distributed state-vector partitioning (Phase 6, P6-02).
//!
//! A `2^n` state is split across `R = 2^g` ranks. Physical index =
//! `(rank << m) | local`, `m = n - g`: physical qubits `< m` are *local*,
//! `>= m` are *global* (they select the rank). The planner tracks a lazy
//! logical→physical map and inserts index-bit-swap `Exchange`s only where a
//! non-diagonal gate needs a global qubit; `specialize` rewrites everything
//! else per rank so no instruction reaching a kernel depends on global bits.
//!
//! Backend-agnostic: nothing here knows how ranks are stored or how an
//! exchange moves bytes. Häner & Steiger, "0.5 Petabyte Simulation of a
//! 45-Qubit Quantum Circuit" (SC'17), §3 — global/local qubit swaps.

use smallvec::SmallVec;

use crate::Instruction;

pub mod dag;
mod next_use;
mod plan;
mod specialize;

pub use dag::{Act, Dag};
pub use plan::{plan, plan_from, required_local};
pub use specialize::specialize;

/// Largest supported `g`: rank indices are `u32`, so `2^g` ranks must fit.
pub const MAX_GLOBAL_QUBITS: u32 = 31;

/// Global/local split of an `n`-qubit state over `2^g` ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistLayout {
    pub n: u32,
    pub g: u32,
}

impl DistLayout {
    /// Validate `1 <= n <= 64`, `g < n` and `g <= 31` (ranks are `u32`).
    pub fn new(n: u32, g: u32) -> Result<Self, DistError> {
        if n == 0 || n > 64 || g >= n || g > MAX_GLOBAL_QUBITS {
            return Err(DistError::BadLayout { n, g });
        }
        Ok(Self { n, g })
    }

    /// Local qubits per rank.
    pub fn m(&self) -> u32 {
        self.n - self.g
    }

    /// Number of ranks, `2^g`.
    pub fn ranks(&self) -> u32 {
        1u32 << self.g
    }

    /// Whether physical qubit `p` is global.
    pub fn is_global(&self, p: u32) -> bool {
        p >= self.m()
    }

    /// Value of global physical qubit `p` on `rank`.
    pub fn rank_bit(&self, rank: u32, p: u32) -> u32 {
        (rank >> (p - self.m())) & 1
    }
}

/// One step of a distributed execution. `Local` instructions use **physical**
/// qubit indices in `0..n`; run them through [`specialize`] per rank.
#[derive(Debug, Clone)]
pub enum DistStep {
    Local(Vec<Instruction>),
    /// Swap each listed global physical bit `global_bits[j]` with local
    /// physical bit `m - k + j` (`k = global_bits.len()`): the top `k` local
    /// bits, so every exchanged chunk is contiguous.
    Exchange {
        global_bits: SmallVec<[u32; 4]>,
    },
}

/// Communication accounting for a plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommStats {
    pub exchanges: u32,
    /// Amplitudes each rank sends (= receives) over the whole plan.
    pub amps_moved_per_rank: u64,
    /// Local `Swap`s inserted to free the top local slot.
    pub local_swaps: u32,
    /// User `Swap`s absorbed as O(1) relabels.
    pub relabels: u32,
}

/// A complete distributed execution plan.
///
/// Every plan assumes the |0…0⟩ initial state. It is invariant under qubit
/// permutations, so a plan may start from a non-identity map (see
/// [`plan_from`]) at no cost.
#[derive(Debug, Clone)]
pub struct DistPlan {
    pub layout: DistLayout,
    pub steps: Vec<DistStep>,
    /// `final_map[logical] = physical` after the last step.
    pub final_map: Vec<u32>,
    pub stats: CommStats,
}

/// Exchange-placement strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Router {
    /// One global qubit per exchange, on demand, evicting the top local slot.
    Naive,
    /// Up to `g` global qubits per exchange: everything the current gate
    /// needs plus prefetched qubits needed before their victim's next use;
    /// victims chosen by farthest next use (Belady). P6-03.
    Lookahead,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DistError {
    #[error("bad layout: n={n}, g={g} (need 1 <= n <= 64, g < n, g <= 31)")]
    BadLayout { n: u32, g: u32 },
    #[error("circuit has {circuit} qubits but layout has n={layout}")]
    QubitCountMismatch { circuit: u32, layout: u32 },
    #[error("instruction `{kind}` is not supported by the distributed planner")]
    Unsupported { kind: &'static str },
    #[error("gate needs {need} local qubits but only m={m} are local")]
    TooFewLocalQubits { need: usize, m: u32 },
    #[error("initial placement is not a permutation of 0..n")]
    BadPlacement,
    #[error("non-diagonal target on global qubit {qubit} reached specialize (planner bug)")]
    GlobalTarget { qubit: u32 },
    #[error(transparent)]
    Gate(#[from] aleph_core::GateError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_basic() {
        let l = DistLayout::new(10, 2).unwrap();
        assert_eq!(l.m(), 8);
        assert_eq!(l.ranks(), 4);
        assert!(!l.is_global(7));
        assert!(l.is_global(8));
        // rank 2 = 0b10: global phys 8 -> bit 0 = 0, phys 9 -> bit 1 = 1
        assert_eq!(l.rank_bit(2, 8), 0);
        assert_eq!(l.rank_bit(2, 9), 1);
    }

    #[test]
    fn test_layout_single_rank() {
        let l = DistLayout::new(5, 0).unwrap();
        assert_eq!(l.m(), 5);
        assert_eq!(l.ranks(), 1);
    }

    #[test]
    fn test_layout_rejects_bad() {
        assert!(matches!(
            DistLayout::new(4, 4),
            Err(DistError::BadLayout { .. })
        ));
        assert!(matches!(
            DistLayout::new(65, 1),
            Err(DistError::BadLayout { .. })
        ));
        assert!(matches!(
            DistLayout::new(0, 0),
            Err(DistError::BadLayout { .. })
        ));
        // rank is u32: more than 31 global qubits would overflow rank arithmetic
        assert!(matches!(
            DistLayout::new(64, 40),
            Err(DistError::BadLayout { .. })
        ));
        assert!(DistLayout::new(64, 31).is_ok());
    }
}
