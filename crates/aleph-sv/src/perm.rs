//! Final physical→logical state reorder for the P2-09 relabelling pass
//! (`passes::RelabelQubits`). The pass permutes qubit indices for cache
//! locality, leaving the simulated state in PHYSICAL-bit order; this helper
//! produces the LOGICAL-order amplitude vector with a single gather.
//!
//! `perm[logical] = physical`: logical qubit `lq` lives at physical bit
//! `perm[lq]`. For a logical basis index `i`, the corresponding physical
//! index `j` has, for each logical bit `lq` set in `i`, physical bit
//! `perm[lq]` set. The logical amplitude `out[i]` is the physical
//! amplitude at `j`.

use aleph_core::{AlignedBuf, Complex};

/// Reorder `phys` (physical-bit order) into logical order per `perm`.
/// `perm.len() == num_qubits` and `perm` is a permutation of
/// `0..num_qubits`. `phys.len() == 2^num_qubits`.
// Not yet called at the call site: will be used by the P2-09 driver (Task 9).
#[allow(dead_code)]
pub(crate) fn bit_permute_state(phys: &[Complex], perm: &[u32]) -> AlignedBuf<Complex> {
    let n = perm.len();
    debug_assert_eq!(phys.len(), 1usize << n, "state length must be 2^num_qubits");
    let mut out = AlignedBuf::<Complex>::zeroed(phys.len());
    for (i, slot) in out.iter_mut().enumerate() {
        // i = logical basis index; build the physical index j.
        let mut j = 0usize;
        for (lq, &pq) in perm.iter().enumerate() {
            if (i >> lq) & 1 == 1 {
                j |= 1usize << pq;
            }
        }
        *slot = phys[j];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn identity_perm_unchanged() {
        let phys = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let out = bit_permute_state(&phys, &[0, 1]);
        assert_eq!(&out[..], &phys[..]);
    }

    #[test]
    fn swap_bits_0_and_2_n3() {
        // perm: logical 0 -> physical 2, logical 1 -> physical 1, logical 2 -> physical 0.
        // i.e. logical index bit0<->bit2 swapped relative to physical.
        let perm = [2u32, 1, 0];
        let n = 3;
        let phys: Vec<Complex> = (0..(1usize << n)).map(|k| c(k as f64, 0.0)).collect();
        let out = bit_permute_state(&phys, &perm);
        // out[i] = phys[j] where j has bit perm[lq] set for each logical bit lq of i.
        // logical i=1 (0b001) -> physical bit perm[0]=2 -> j=0b100=4.
        assert_eq!(out[1], c(4.0, 0.0));
        // logical i=4 (0b100) -> physical bit perm[2]=0 -> j=0b001=1.
        assert_eq!(out[4], c(1.0, 0.0));
        // logical i=2 (0b010) -> physical bit perm[1]=1 -> j=0b010=2 (fixed).
        assert_eq!(out[2], c(2.0, 0.0));
    }

    #[test]
    fn round_trip_with_inverse() {
        // Applying perm then its inverse returns the original.
        let perm = [2u32, 0, 1]; // logical->physical
        let n = 3;
        let phys: Vec<Complex> = (0..(1usize << n))
            .map(|k| c(k as f64, (k as f64).cos()))
            .collect();
        let once = bit_permute_state(&phys, &perm);
        // inverse: inv[physical] = logical, expressed as a logical->physical map
        // that undoes perm. inv such that inv[perm[l]] = l.
        let mut inv = [0u32; 3];
        for (l, &p) in perm.iter().enumerate() {
            inv[p as usize] = l as u32;
        }
        let back = bit_permute_state(&once, &inv);
        for (a, b) in back.iter().zip(phys.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
