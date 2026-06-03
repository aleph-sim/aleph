//! `FuseDiagonalRuns` — fuses runs of {diagonal gates ∪ Cnot} into one
//! `DiagonalPhase`, absorbing interleaved `cx` via monomial tracking.
//! See docs/superpowers/specs/2026-06-02-p2-06-diagonal-fusion-design.md.

use crate::PhaseTerm;
use aleph_core::{GateInstance, GateMatrix};
use smallvec::SmallVec;

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

/// Drop terms whose angle is within this of a 2π multiple.
// Used by FuseDiagonalRuns (Task 5).
#[allow(dead_code)]
pub(crate) const PHASE_EPS: f64 = 1e-12;

/// Expand a diagonal `GateInstance` into additive phase terms in the
/// current permuted basis. Returns `None` if the gate is not diagonal,
/// or has a symbolic/non-finite parameter (non-extractable matrix).
///
/// A diagonal gate's phase function `f(b_0..b_{k-1})` over its `k`
/// targets expands uniquely as a multilinear polynomial
/// `f = Σ_{S⊆targets} α_S · Π_{i∈S} b_i`, where the monomial `Π b_i`
/// is the AND-of-ones condition. Möbius inversion gives the coefficient
/// `α_S = Σ_{T⊆S} (-1)^{|S\T|} f(T)`. Each nonzero `α_S` becomes one
/// [`PhaseTerm`] whose `conds` are the permuted images of the qubits in
/// `S`. External controls gate the whole operator, so every term picks
/// up the control images (the `S=∅` global term thereby becomes
/// conditioned on the controls).
// Used by FuseDiagonalRuns (Task 5).
#[allow(dead_code)]
pub(crate) fn diagonal_to_terms(g: &GateInstance, perm: &Perm) -> Option<Vec<PhaseTerm>> {
    if !g.gate.is_diagonal() {
        return None;
    }
    let targets = &g.qubits;
    let k = targets.len();
    debug_assert!(k <= 3, "diagonal gates have ≤3 targets");

    let diag = diagonal_entries(g)?; // length 2^k, MSB-first by targets

    // f(subset_pattern): bit p of pattern set => targets[p] == 1.
    // The matrix index is MSB-first in the gate's own qubit list, so
    // targets[0] is the most significant bit of the diagonal index.
    let f = |subset_pattern: usize| -> f64 {
        let mut r = 0usize;
        for p in 0..k {
            if (subset_pattern >> p) & 1 == 1 {
                r |= 1 << (k - 1 - p); // MSB-first row index
            }
        }
        diag[r].arg()
    };

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut terms: Vec<PhaseTerm> = Vec::new();
    for s in 0..(1usize << k) {
        // α_S = Σ_{T⊆S} (-1)^{|S\T|} f(T)  (iterate subsets T of S)
        let mut alpha = 0.0;
        let mut t = s;
        loop {
            let sign = if ((s ^ t).count_ones()) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            alpha += sign * f(t);
            if t == 0 {
                break;
            }
            t = (t - 1) & s;
        }
        // Negligibility uses the mod-2π reduced value, but the stored
        // angle stays RAW so sums of raw angles round-trip exactly for
        // the downstream 1e-12 oracle.
        let a = alpha.rem_euclid(two_pi);
        if a < PHASE_EPS || (two_pi - a) < PHASE_EPS {
            continue; // negligible mod 2π
        }
        let mut conds: SmallVec<[u64; 2]> = SmallVec::new();
        for (p, &target) in targets.iter().enumerate() {
            if (s >> p) & 1 == 1 {
                conds.push(perm.image(target));
            }
        }
        for &c in g.controls.iter() {
            conds.push(perm.image(c));
        }
        terms.push(PhaseTerm {
            conds,
            angle: alpha,
        });
    }
    Some(terms)
}

/// Diagonal entries (length 2^arity), MSB-first. `None` for symbolic/
/// non-finite params (`Gate::matrix` returns `Err`).
// Used by FuseDiagonalRuns (Task 5).
#[allow(dead_code)]
fn diagonal_entries(g: &GateInstance) -> Option<Vec<aleph_core::Complex>> {
    match g.gate.matrix().ok()? {
        GateMatrix::M2x2(m) => Some(vec![m[0][0], m[1][1]]),
        GateMatrix::M4x4(m) => Some(vec![m[0][0], m[1][1], m[2][2], m[3][3]]),
        GateMatrix::M8x8(m) => Some((0..8).map(|i| m[i][i]).collect()),
    }
}

#[cfg(test)]
mod terms_tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;
    use std::f64::consts::PI;

    fn dp_phase(terms: &[crate::PhaseTerm], x: u64) -> f64 {
        let mut phi = 0.0;
        for t in terms {
            if t.conds.iter().all(|&m| (m & x).count_ones() & 1 == 1) {
                phi += t.angle;
            }
        }
        phi
    }

    #[test]
    fn plain_phase_is_one_single_bit_term() {
        let g = GateInstance::new(Gate::Phase(0.5.into()), smallvec![1u32]);
        let perm = Perm::identity(3);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        assert!((dp_phase(&terms, 0b000) - 0.0).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b010) - 0.5).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b110) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn controlled_phase_fires_only_on_both_set() {
        let g = GateInstance::controlled(
            Gate::Phase(PI.into()),
            smallvec![0u32], // target
            smallvec![1u32], // control
        );
        let perm = Perm::identity(2);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        assert!(dp_phase(&terms, 0b00).abs() < 1e-15);
        assert!(dp_phase(&terms, 0b01).abs() < 1e-15);
        assert!(dp_phase(&terms, 0b10).abs() < 1e-15);
        assert!((dp_phase(&terms, 0b11) - PI).abs() < 1e-12);
    }

    #[test]
    fn cz_phase_pi_on_eleven() {
        let g = GateInstance::new(Gate::Cz, smallvec![0u32, 1u32]);
        let perm = Perm::identity(2);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        for x in [0b00u64, 0b01, 0b10] {
            assert!(dp_phase(&terms, x).abs() < 1e-15, "x={x:b}");
        }
        let p = dp_phase(&terms, 0b11);
        assert!(((p - PI).rem_euclid(2.0 * PI)).abs() < 1e-12 || (p + PI).abs() < 1e-12);
    }

    #[test]
    fn ccz_phase_pi_on_one_one_one() {
        // Ccz = diag(1,1,1,1,1,1,1,-1) -> single term conds={q0,q1,q2},
        // angle = π. Exercises the M8x8 extraction path.
        let g = GateInstance::new(Gate::Ccz, smallvec![0u32, 1u32, 2u32]);
        let perm = Perm::identity(3);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        for x in 0b000u64..0b111 {
            assert!(dp_phase(&terms, x).abs() < 1e-15, "x={x:b}");
        }
        let p = dp_phase(&terms, 0b111);
        assert!(((p - PI).rem_euclid(2.0 * PI)).abs() < 1e-12 || (p + PI).abs() < 1e-12);
    }

    #[test]
    fn conjugated_phase_picks_up_control_bit() {
        // p(θ) on bit 1, recorded while P has cx(0,1) applied
        // (perm.image(1) = bits {0,1}). Then it fires on parity(b0^b1).
        let g = GateInstance::new(Gate::Phase(0.5.into()), smallvec![1u32]);
        let mut perm = Perm::identity(2);
        perm.cx(0, 1);
        let terms = diagonal_to_terms(&g, &perm).unwrap();
        assert!(dp_phase(&terms, 0b00).abs() < 1e-15); // b0^b1 = 0
        assert!((dp_phase(&terms, 0b01) - 0.5).abs() < 1e-15); // 1^0 = 1
        assert!((dp_phase(&terms, 0b10) - 0.5).abs() < 1e-15); // 0^1 = 1
        assert!(dp_phase(&terms, 0b11).abs() < 1e-15); // 1^1 = 0
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
