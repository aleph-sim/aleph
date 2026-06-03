//! `FuseDiagonalRuns` — fuses runs of {diagonal gates ∪ Cnot} into one
//! `DiagonalPhase`, absorbing interleaved `cx` via monomial tracking.
//! See docs/superpowers/specs/2026-06-02-p2-06-diagonal-fusion-design.md.

/// GF(2) bit-permutation tracker. `row[i]` is the mask such that the
/// i-th output bit equals `parity(row[i] & x)`. Starts as identity.
/// A `cx(c, t)` on the LEFT of the accumulated product does
/// `row[t] ^= row[c]` (design §1.2).
// Used by FuseDiagonalRuns (Task 4/5).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Perm {
    row: Vec<u64>,
}

#[allow(dead_code)]
impl Perm {
    pub(crate) fn identity(n: u32) -> Self {
        Perm {
            row: (0..n).map(|i| 1u64 << i).collect(),
        }
    }
    pub(crate) fn cx(&mut self, control: u32, target: u32) {
        let c = self.row[control as usize];
        self.row[target as usize] ^= c;
    }
    /// Mask for the image of input bit `b` under the current permutation.
    pub(crate) fn image(&self, b: u32) -> u64 {
        self.row[b as usize]
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.row.iter().enumerate().all(|(i, &m)| m == 1u64 << i)
    }
}

#[cfg(test)]
mod perm_tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        assert!(Perm::identity(4).is_identity());
        assert_eq!(Perm::identity(3).image(2), 0b100);
    }

    #[test]
    fn single_cx_xors_control_into_target() {
        let mut p = Perm::identity(3);
        p.cx(0, 1); // row[1] ^= row[0]
        assert_eq!(p.image(1), 0b011);
        assert_eq!(p.image(0), 0b001);
        assert!(!p.is_identity());
    }

    #[test]
    fn cx_pair_cancels_to_identity() {
        // The QFT invariant: cx(c,t) applied twice nets to identity.
        let mut p = Perm::identity(4);
        p.cx(3, 1);
        p.cx(3, 1);
        assert!(p.is_identity());
    }

    #[test]
    fn distinct_target_cx_pairs_all_cancel() {
        let mut p = Perm::identity(5);
        for t in [3u32, 2, 1, 0] {
            p.cx(4, t);
        }
        for t in [3u32, 2, 1, 0] {
            p.cx(4, t);
        }
        assert!(p.is_identity());
    }
}
