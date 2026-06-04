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

/// Generic physical→logical bit-permutation gather, parameterised over the
/// buffer element type. The index-permutation arithmetic is identical for
/// every backend's amplitude representation — only the element differs
/// (`Complex` for AoS f64, `f64` for an SoA plane, `Complex<f32>` for FP32) —
/// so the three backend overrides share this one body.
///
/// `T: Copy` and the `AlignedBuf::zeroed` const-asserts (`!needs_drop`,
/// non-ZST) are satisfied by all three element types.
///
/// `out[i_logical] = phys[j]` where `j` sets physical bit `perm[lq]` for each
/// logical bit `lq` set in `i`.
fn bit_permute_buf<T: Copy>(phys: &[T], perm: &[u32]) -> AlignedBuf<T> {
    let n = perm.len();
    debug_assert_eq!(phys.len(), 1usize << n, "state length must be 2^num_qubits");
    let mut out = AlignedBuf::<T>::zeroed(phys.len());
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

/// Reorder `phys` (physical-bit order) into logical order per `perm`.
/// `perm.len() == num_qubits` and `perm` is a permutation of
/// `0..num_qubits`. `phys.len() == 2^num_qubits`.
// Called by `NaiveSvBackend::unpermute_state` (the P2-09 driver tail).
pub(crate) fn bit_permute_state(phys: &[Complex], perm: &[u32]) -> AlignedBuf<Complex> {
    bit_permute_buf(phys, perm)
}

/// f32 AoS analogue of [`bit_permute_state`] for the FP32 backend.
pub(crate) fn bit_permute_state_f32(
    phys: &[aleph_core::Complex<f32>],
    perm: &[u32],
) -> AlignedBuf<aleph_core::Complex<f32>> {
    bit_permute_buf(phys, perm)
}

/// Split-buffer (SoA) analogue: permutes a single `f64` plane (re or im).
pub(crate) fn bit_permute_plane(plane: &[f64], perm: &[u32]) -> AlignedBuf<f64> {
    bit_permute_buf(plane, perm)
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
    fn f32_swap_bits_0_and_2_n3() {
        // Same asymmetric perm as `swap_bits_0_and_2_n3`, over Complex<f32>.
        let perm = [2u32, 1, 0];
        let n = 3;
        let phys: Vec<Complex<f32>> = (0..(1usize << n))
            .map(|k| Complex::<f32>::new(k as f32, 0.0))
            .collect();
        let out = bit_permute_state_f32(&phys, &perm);
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
        let out = bit_permute_plane(&plane, &perm);
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
