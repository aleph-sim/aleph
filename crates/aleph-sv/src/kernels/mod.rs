//! Indexed gate application kernels.
//!
//! Two layouts share the same MSB qubit-ordering convention (ADR 0004
//! / P0-06 spec §6): `qubits[0]` is the MSB of the matrix index. They
//! diverge only in storage:
//!
//! * `aos` — `Vec<Complex<f64>>` (the naive `Vec<num_complex::Complex>`
//!   layout used by `NaiveSvBackend`).
//! * `soa` — paired `Vec<f64>` (real, imaginary) used by `SoaSvBackend`
//!   (P1-01). Same algorithms, layout chosen for SIMD-friendly
//!   sequential reads — explicit vectorisation lands in P1-03 / P1-04.

pub(crate) mod aos;
pub(crate) mod soa;

/// Bitwise-OR of `1 << q` over `controls`. Layout-agnostic — used by
/// both AoS and SoA kernels to compute the control gate-mask.
///
/// Returns `usize` so the result composes directly with index
/// arithmetic in the kernel loops; `q` is bounded by `state.num_qubits`
/// at the apply_gate boundary, which itself is capped at `MAX_*_QUBITS
/// ≤ 28`, so `1 << q` never overflows on any supported platform.
pub(crate) fn control_mask(controls: &[u32]) -> usize {
    let mut mask: usize = 0;
    for &c in controls {
        mask |= 1usize << c;
    }
    mask
}

/// For a 1-qubit gate on `target` with external `controls`, returns
/// the "base" amplitude index `i0` (the one with `target` bit = 0
/// and every control bit = 1) for the `k`-th iteration of the
/// free-bit outer loop.
///
/// `controls` must contain no element equal to `target` (callers
/// enforce this via the `DuplicateQubit` check before reaching the
/// kernel).  `target` and every control must be `< usize::BITS`
/// (caller enforces via `MAX_SOA_QUBITS = 28`).
///
/// `k` ranges over `0..(1usize << free_bits)` where
/// `free_bits = n_qubits - 1 - controls.len()`.  The pair partner is
/// `i0 | (1usize << target)`.
///
/// The implementation walks the sorted set of fixed bit positions
/// (target + controls) and splices chunks of `k`'s low bits into the
/// free slots between them, leaving the fixed slots to be filled in
/// the same pass (target → 0, controls → 1).  Algorithm is canonical
/// — cf. QuEST `statevec_unitary` for the reference shape.
pub(crate) fn base_index_1q(k: usize, target: u32, controls: &[u32]) -> usize {
    // Stack-only for the realistic `controls.len() ≤ 7` range (the
    // `SmallVec<[u32; 6]>` in `apply_gate`'s `seen` set tolerates up
    // to ~6 unique qubit indices; this cap of 8 leaves headroom and
    // avoids any heap allocation in the hot path).
    let mut fixed: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed.push((target, false));
    for &c in controls {
        fixed.push((c, true));
    }
    fixed.sort_unstable_by_key(|(pos, _)| *pos);

    let mut i = 0usize;
    let mut k_rem = k;
    let mut prev = 0u32;
    for (pos, val) in &fixed {
        let span = (*pos - prev) as usize;
        let chunk = k_rem & ((1usize << span) - 1);
        i |= chunk << (prev as usize);
        k_rem >>= span;
        if *val {
            i |= 1usize << pos;
        }
        prev = *pos + 1;
    }
    // Remaining free bits above the highest fixed position.
    i |= k_rem << (prev as usize);
    i
}

#[cfg(test)]
mod tests {
    use super::{base_index_1q, control_mask};

    #[test]
    fn control_mask_empty_is_zero() {
        assert_eq!(control_mask(&[]), 0);
    }

    #[test]
    fn control_mask_combines_bits() {
        // Controls on qubits 0, 2, 5 → bit positions 0, 2, 5 → 0b100101 = 37.
        assert_eq!(control_mask(&[0, 2, 5]), 0b100101);
    }

    #[test]
    fn control_mask_is_order_independent() {
        assert_eq!(control_mask(&[5, 0, 2]), control_mask(&[0, 2, 5]));
    }

    #[test]
    fn base_index_uncontrolled_target_zero_is_2k() {
        // target = 0, no controls → k-th iteration's i0 is 2*k.
        for k in 0..16 {
            assert_eq!(base_index_1q(k, 0, &[]), 2 * k);
        }
    }

    #[test]
    fn base_index_uncontrolled_target_two_keeps_bit_zero() {
        // target = 2 → bit 2 of i0 is always 0.
        for k in 0..16 {
            let i = base_index_1q(k, 2, &[]);
            assert_eq!((i >> 2) & 1, 0, "k = {k}, i = {i:#06b}");
        }
    }

    #[test]
    fn base_index_controlled_sets_control_bits() {
        // target = 0, control = 2 → bit 0 of i0 is 0, bit 2 of i0 is 1.
        for k in 0..8 {
            let i = base_index_1q(k, 0, &[2]);
            assert_eq!(i & 1, 0, "target bit must be 0");
            assert_eq!((i >> 2) & 1, 1, "control bit must be 1");
        }
    }

    #[test]
    fn base_index_two_controls_target_between() {
        // target = 2, controls = [0, 5] → bit 0 = 1, bit 2 = 0, bit 5 = 1.
        for k in 0..16 {
            let i = base_index_1q(k, 2, &[0, 5]);
            assert_eq!(i & 1, 1, "control bit 0 must be 1");
            assert_eq!((i >> 2) & 1, 0, "target bit must be 0");
            assert_eq!((i >> 5) & 1, 1, "control bit 5 must be 1");
        }
    }

    #[test]
    fn base_index_enumerates_all_pairs_exactly_once() {
        // For n = 5 qubits, target = 3, controls = [1], the valid pair
        // count is 2^(5 - 1 - 1) = 8.  Collect (i0, i1) across all k;
        // every value must be distinct, and the union must equal exactly
        // the 16 amplitudes whose bit-1 is set (2^(5-1) = 16).
        use std::collections::HashSet;
        let mut seen: HashSet<usize> = HashSet::new();
        for k in 0..8 {
            let i0 = base_index_1q(k, 3, &[1]);
            let i1 = i0 | (1usize << 3);
            assert!(seen.insert(i0), "duplicate i0 = {i0:#07b} at k = {k}");
            assert!(seen.insert(i1), "duplicate i1 = {i1:#07b} at k = {k}");
            assert_eq!(
                (i0 >> 1) & 1,
                1,
                "control bit 1 unset at i0 = {i0:#07b}"
            );
        }
        assert_eq!(seen.len(), 16);
    }
}
