//! Physical→logical bit-permutation gather, shared across backends.
//!
//! Several backends store an amplitude vector whose qubit/bit ordering has
//! been permuted for performance (the P2-09 `RelabelQubits` cache-locality
//! pass in `aleph-sv`; the P3-09 lazy-SWAP router in `aleph-mps`) and must
//! reorder it back to logical-qubit order before returning it. The index
//! arithmetic is identical for every backend and every element type, so it
//! lives here once with its tests rather than being hand-maintained per crate
//! (any ADR-0004 bit-order fix then applies in exactly one place).

use crate::AlignedBuf;

/// Gather `phys` (physical-bit order) into logical-qubit order per `perm`.
///
/// `perm[logical] = physical`: logical qubit `lq` is stored at physical bit
/// `perm[lq]`. For a logical basis index `i`, the source physical index `j`
/// sets physical bit `perm[lq]` for each logical bit `lq` set in `i`, and
/// `out[i] = phys[j]`.
///
/// `perm` must be a permutation of `0..num_qubits` and `phys.len()` must be
/// `2^perm.len()`. Generic over the element type so AoS `Complex` (f64/f32)
/// and bare SoA `f64`/`f32` planes share the one body; `T: Copy` plus the
/// `AlignedBuf::zeroed` const-asserts (`!needs_drop`, non-ZST) hold for all of
/// them.
pub fn bit_permute_buf<T: Copy>(phys: &[T], perm: &[u32]) -> AlignedBuf<T> {
    let n = perm.len();
    debug_assert_eq!(phys.len(), 1usize << n, "state length must be 2^num_qubits");
    let mut out = AlignedBuf::<T>::zeroed(phys.len());
    for (i, slot) in out.iter_mut().enumerate() {
        // i = logical basis index; build the physical source index j.
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
    use crate::Complex;

    fn c(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn identity_perm_unchanged() {
        let phys = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let out = bit_permute_buf(&phys, &[0, 1]);
        assert_eq!(&out[..], &phys[..]);
    }

    #[test]
    fn swap_bits_0_and_2_n3() {
        // perm: logical 0 -> physical 2, logical 1 -> physical 1, logical 2 -> physical 0.
        // i.e. logical index bit0<->bit2 swapped relative to physical.
        let perm = [2u32, 1, 0];
        let n = 3;
        let phys: Vec<Complex> = (0..(1usize << n)).map(|k| c(k as f64, 0.0)).collect();
        let out = bit_permute_buf(&phys, &perm);
        // out[i] = phys[j] where j has bit perm[lq] set for each logical bit lq of i.
        // logical i=1 (0b001) -> physical bit perm[0]=2 -> j=0b100=4.
        assert_eq!(out[1], c(4.0, 0.0));
        // logical i=4 (0b100) -> physical bit perm[2]=0 -> j=0b001=1.
        assert_eq!(out[4], c(1.0, 0.0));
        // logical i=2 (0b010) -> physical bit perm[1]=1 -> j=0b010=2 (fixed).
        assert_eq!(out[2], c(2.0, 0.0));
    }

    #[test]
    fn f32_swap_bits_0_and_2_n3() {
        // Same asymmetric perm as `swap_bits_0_and_2_n3`, over Complex<f32>.
        let perm = [2u32, 1, 0];
        let n = 3;
        let phys: Vec<Complex<f32>> = (0..(1usize << n))
            .map(|k| Complex::<f32>::new(k as f32, 0.0))
            .collect();
        let out = bit_permute_buf(&phys, &perm);
        // logical i=1 -> physical bit perm[0]=2 -> j=4.
        assert_eq!(out[1], Complex::<f32>::new(4.0, 0.0));
        // logical i=4 -> physical bit perm[2]=0 -> j=1.
        assert_eq!(out[4], Complex::<f32>::new(1.0, 0.0));
        // logical i=2 -> physical bit perm[1]=1 -> j=2 (fixed).
        assert_eq!(out[2], Complex::<f32>::new(2.0, 0.0));
    }

    #[test]
    fn plane_swap_bits_0_and_2_n3() {
        // Same asymmetric perm as `swap_bits_0_and_2_n3`, over a bare f64 plane.
        let perm = [2u32, 1, 0];
        let n = 3;
        let plane: Vec<f64> = (0..(1usize << n)).map(|k| k as f64).collect();
        let out = bit_permute_buf(&plane, &perm);
        assert_eq!(out[1], 4.0); // i=1 -> j=4
        assert_eq!(out[4], 1.0); // i=4 -> j=1
        assert_eq!(out[2], 2.0); // i=2 -> j=2 (fixed)
    }

    #[test]
    fn round_trip_with_inverse() {
        // Applying perm then its inverse returns the original.
        let perm = [2u32, 0, 1]; // logical->physical
        let n = 3;
        let phys: Vec<Complex> = (0..(1usize << n))
            .map(|k| c(k as f64, (k as f64).cos()))
            .collect();
        let once = bit_permute_buf(&phys, &perm);
        // inverse: inv such that inv[perm[l]] = l.
        let mut inv = [0u32; 3];
        for (l, &p) in perm.iter().enumerate() {
            inv[p as usize] = l as u32;
        }
        let back = bit_permute_buf(&once[..], &inv);
        for (a, b) in back.iter().zip(phys.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn asymmetric_three_cycle_matches_scatter() {
        // The aleph-mps `dense_statevector` usage: a 3-cycle permutation, gather
        // form (perm = site_of_qubit) must equal the inverse-map scatter
        // (out[scatter_idx] = phys[raw], scatter_idx bit = qubit_of_site[s]).
        // qubit_of_site = [1, 2, 0]  =>  site_of_qubit = [2, 0, 1].
        let qubit_of_site = [1u32, 2, 0];
        let site_of_qubit = [2u32, 0, 1];
        let n = 3;
        let phys: Vec<Complex> = (0..(1usize << n)).map(|k| c(k as f64, 0.0)).collect();

        let gather = bit_permute_buf(&phys, &site_of_qubit);

        // Reference scatter, exactly as MPS phase-2 builds it.
        let mut scatter = vec![c(0.0, 0.0); 1usize << n];
        #[allow(clippy::needless_range_loop)]
        for raw_idx in 0..(1usize << n) {
            let mut logical_idx = 0usize;
            for s in 0..n {
                let bit = (raw_idx >> s) & 1;
                logical_idx |= bit << qubit_of_site[s] as usize;
            }
            scatter[logical_idx] = phys[raw_idx];
        }
        assert_eq!(&gather[..], &scatter[..]);
    }
}
