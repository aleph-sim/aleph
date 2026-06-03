//! Dense k-qubit gate kernel: one pass, a 2^k×2^k matvec per 2^k-block.
//! Real bodies land in P2-07 Task 7; this module is wired by Task 6.
use aleph_core::Complex;

/// Apply a dense `2^k × 2^k` unitary (`data`, row-major, MSB-first operand
/// order `qubits[0]`=MSB) to an AoS state in one pass.
pub(crate) fn apply_kq_aos(amps: &mut [Complex], qubits: &[u32], k: u8, data: &[Complex]) {
    apply_kq_scalar_aos(amps, qubits, k, data);
}

/// SoA variant (split real/imag arrays).
pub(crate) fn apply_kq_soa(
    re: &mut [f64],
    im: &mut [f64],
    qubits: &[u32],
    k: u8,
    data: &[Complex],
) {
    apply_kq_scalar_soa(re, im, qubits, k, data);
}

// ---------------------------------------------------------------------------
// Shared helper: sort targets, build offsets + fixed list.
//
// MSB-first operand convention (qubits[0] = matrix MSB, ADR 0004):
//   matrix bit p (0=LSB … k-1=MSB) ↔ qubit Q[k-1-p]
// so offset for matrix index m:
//   offsets[m] = Σ_{p: bit p of m set} (1 << Q[k-1-p])
// ---------------------------------------------------------------------------
fn targets_offsets_fixed(qubits: &[u32], k: u8) -> (Vec<usize>, Vec<(u32, bool)>) {
    let k = k as usize;
    let mut q = qubits.to_vec();
    q.sort_unstable();

    let offsets: Vec<usize> = (0..(1usize << k))
        .map(|m| {
            let mut off = 0usize;
            for p in 0..k {
                if (m >> p) & 1 == 1 {
                    off |= 1usize << q[k - 1 - p];
                }
            }
            off
        })
        .collect();

    // fixed: each target qubit is cleared; expand_with_fixed requires ascending order.
    let fixed: Vec<(u32, bool)> = q.iter().map(|&x| (x, false)).collect();

    (offsets, fixed)
}

/// Scalar AoS implementation.
///
/// Iterates over `outer = 2^(n-k)` outer counters. For each counter,
/// `expand_with_fixed` gives a base index with all k target bits cleared.
/// The 2^k amplitudes in this block are at `base | offsets[m]` for each
/// matrix index `m`. The matvec reads them all into a local buffer, then
/// writes back the contracted result.
///
/// # Safety (parallel-write contract)
/// `par_blocks` hands each task a distinct `counter` value. The bit-positions
/// used by `expand_with_fixed` are exactly the FREE bits (not in `fixed`).
/// Two distinct counters produce distinct `base` values that differ in at
/// least one FREE bit position, so `base_a | offsets[m] ≠ base_b | offsets[n]`
/// for any m, n — disjoint writes, no aliasing. The indexing-coverage test
/// (`index_coverage_disjoint_and_complete`) verifies this exhaustively for a
/// concrete (n, k, targets) triple.
pub(crate) fn apply_kq_scalar_aos(amps: &mut [Complex], qubits: &[u32], k: u8, data: &[Complex]) {
    let dim = 1usize << k;
    let (offsets, fixed) = targets_offsets_fixed(qubits, k);
    let len = amps.len();
    let outer = len >> k; // 2^(n-k) outer blocks

    let p = crate::kernels::ComplexPtr(amps.as_mut_ptr());

    crate::kernels::par_blocks(
        crate::kernels::tuning::DEFAULT_POLICY,
        outer,
        len,
        |c| c,
        move |counter| {
            let base = crate::kernels::expand_with_fixed(counter, &fixed);

            // Read the block of 2^k amplitudes into a local buffer.
            let mut inb = vec![Complex::new(0.0, 0.0); dim];
            for (m, inb_m) in inb.iter_mut().enumerate() {
                // SAFETY: base|offsets[m] is within [0, len), distinct across
                // m (coverage test), and disjoint from other counters'
                // blocks — no two parallel tasks share an index. The pointer
                // lives for the duration of apply_kq_scalar_aos.
                *inb_m = unsafe { *p.ptr().add(base | offsets[m]) };
            }

            // Matvec: out[r] = Σ_c data[r*dim + c] * in[c].
            for r in 0..dim {
                let mut acc = Complex::new(0.0, 0.0);
                for cc in 0..dim {
                    acc += data[r * dim + cc] * inb[cc];
                }
                // SAFETY: same disjointness guarantee as the read above.
                unsafe { *p.ptr().add(base | offsets[r]) = acc };
            }
        },
    );
}

/// Scalar SoA implementation — mirrors `apply_kq_scalar_aos` with split
/// real/imaginary arrays. Complex multiplication is expanded in-line:
/// `(ar + i·ai) = Σ_c (dr + i·di) * (inr + i·ini)
///              = Σ_c (dr·inr − di·ini) + i·(dr·ini + di·inr)`.
pub(crate) fn apply_kq_scalar_soa(
    re: &mut [f64],
    im: &mut [f64],
    qubits: &[u32],
    k: u8,
    data: &[Complex],
) {
    let dim = 1usize << k;
    let (offsets, fixed) = targets_offsets_fixed(qubits, k);
    let len = re.len();
    debug_assert_eq!(len, im.len());
    let outer = len >> k;

    let rp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let ip = crate::kernels::BlockPtr(im.as_mut_ptr());

    crate::kernels::par_blocks(
        crate::kernels::tuning::DEFAULT_POLICY,
        outer,
        len,
        |c| c,
        move |counter| {
            let base = crate::kernels::expand_with_fixed(counter, &fixed);

            // Read the block of 2^k SoA amplitudes into local buffers.
            let mut inr = vec![0.0f64; dim];
            let mut ini = vec![0.0f64; dim];
            for m in 0..dim {
                let idx = base | offsets[m];
                // SAFETY: idx is within [0, len), distinct across m, and
                // disjoint from other counters — identical contract as AoS.
                inr[m] = unsafe { *rp.ptr().add(idx) };
                ini[m] = unsafe { *ip.ptr().add(idx) };
            }

            // Matvec with in-line complex multiply.
            for r in 0..dim {
                let mut ar = 0.0f64;
                let mut ai = 0.0f64;
                for cc in 0..dim {
                    let d = data[r * dim + cc];
                    ar += d.re * inr[cc] - d.im * ini[cc];
                    ai += d.re * ini[cc] + d.im * inr[cc];
                }
                let idx = base | offsets[r];
                // SAFETY: same disjointness guarantee as the read above.
                unsafe {
                    *rp.ptr().add(idx) = ar;
                    *ip.ptr().add(idx) = ai;
                }
            }
        },
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Reference offset computation matching the spec's MSB-first convention.
    fn offsets_ref(q_sorted: &[u32], k: u8) -> Vec<usize> {
        let k = k as usize;
        (0..(1usize << k))
            .map(|m| {
                let mut off = 0usize;
                for p in 0..k {
                    if (m >> p) & 1 == 1 {
                        off |= 1usize << q_sorted[k - 1 - p];
                    }
                }
                off
            })
            .collect()
    }

    /// n=5, k=3, targets {0,2,4}: 2^(n-k)=4 bases × 2^k=8 offsets cover all 32 once.
    #[test]
    fn index_coverage_disjoint_and_complete() {
        let n = 5u32;
        let k = 3u8;
        let q = [0u32, 2, 4];
        let fixed: Vec<(u32, bool)> = q.iter().map(|&x| (x, false)).collect();
        let offs = offsets_ref(&q, k);
        let mut seen = vec![false; 1usize << n];
        for counter in 0..(1usize << (n as usize - k as usize)) {
            let base = crate::kernels::expand_with_fixed(counter, &fixed);
            for &o in &offs {
                let idx = base | o;
                assert!(!seen[idx], "dup idx {idx} (base={base} off={o})");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "all 32 indices must be covered");
    }

    /// Targets touching the TOP qubit and non-adjacent: n=6, k=3, {1,3,5}.
    #[test]
    fn index_coverage_top_qubits_and_nonadjacent() {
        let n = 6u32;
        let k = 3u8;
        let q = [1u32, 3, 5];
        let fixed: Vec<(u32, bool)> = q.iter().map(|&x| (x, false)).collect();
        let offs = offsets_ref(&q, k);
        let mut seen = vec![false; 1usize << n];
        for counter in 0..(1usize << (n as usize - k as usize)) {
            let base = crate::kernels::expand_with_fixed(counter, &fixed);
            for &o in &offs {
                let idx = base | o;
                assert!(!seen[idx], "dup idx {idx}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Identity matrix on 3 qubits of a 4-qubit state must be a no-op.
    #[test]
    fn apply_kq_3q_identity_is_noop() {
        let n = 4u32;
        let k = 3u8;
        let dim = 8;
        let mut data = vec![Complex::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            data[i * dim + i] = Complex::new(1.0, 0.0);
        }
        let mut amps: Vec<Complex> = (0..(1 << n))
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        let orig = amps.clone();
        apply_kq_scalar_aos(&mut amps, &[0, 1, 2], k, &data);
        for i in 0..amps.len() {
            assert!(
                (amps[i] - orig[i]).norm() < 1e-12,
                "mismatch at i={i}: got {:?} want {:?}",
                amps[i],
                orig[i]
            );
        }
    }

    /// k=2 SWAP matrix on qubits [0,1] of n=2: swaps |01⟩ and |10⟩.
    ///
    /// SWAP as 4×4 MSB-first (qubits[0]=MSB):
    ///   basis |00⟩=0 → |00⟩, |01⟩=1 → |10⟩=2, |10⟩=2 → |01⟩=1, |11⟩=3 → |11⟩
    /// Row-major: row r has a 1 in column π(r).
    ///   row 0 → col 0 (|00⟩→|00⟩)
    ///   row 1 → col 2 (|01⟩→|10⟩)
    ///   row 2 → col 1 (|10⟩→|01⟩)
    ///   row 3 → col 3 (|11⟩→|11⟩)
    #[test]
    fn apply_kq_swap_matches_manual() {
        let mut data = vec![Complex::new(0.0, 0.0); 16];
        data[0] = Complex::new(1.0, 0.0); // row 0 col 0: |00⟩→|00⟩
        data[6] = Complex::new(1.0, 0.0); // row 1 col 2: |01⟩→|10⟩
        data[9] = Complex::new(1.0, 0.0); // row 2 col 1: |10⟩→|01⟩
        data[15] = Complex::new(1.0, 0.0); // row 3 col 3: |11⟩→|11⟩
        let mut amps = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        apply_kq_scalar_aos(&mut amps, &[0, 1], 2, &data);
        // amps[1] = old |10⟩ = old amps[2] = 3.0
        assert!(
            (amps[1] - Complex::new(3.0, 0.0)).norm() < 1e-12,
            "amps[1]={:?}",
            amps[1]
        );
        // amps[2] = old |01⟩ = old amps[1] = 2.0
        assert!(
            (amps[2] - Complex::new(2.0, 0.0)).norm() < 1e-12,
            "amps[2]={:?}",
            amps[2]
        );
        // amps[0] and amps[3] unchanged
        assert!((amps[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
        assert!((amps[3] - Complex::new(4.0, 0.0)).norm() < 1e-12);
    }

    /// AoS and SoA must produce bit-identical results (within 1e-12).
    #[test]
    fn aos_soa_agree() {
        let n = 4u32;
        let k = 3u8;
        let dim = 8;
        let data: Vec<Complex> = (0..dim * dim)
            .map(|i| Complex::new((i % 5) as f64 * 0.1, (i % 3) as f64 * 0.2))
            .collect();
        let aos: Vec<Complex> = (0..(1 << n))
            .map(|i| Complex::new(0.3 * i as f64 + 1.0, 0.1 - 0.05 * i as f64))
            .collect();
        let mut a = aos.clone();
        apply_kq_scalar_aos(&mut a, &[1, 2, 3], k, &data);

        let mut re: Vec<f64> = aos.iter().map(|c| c.re).collect();
        let mut im: Vec<f64> = aos.iter().map(|c| c.im).collect();
        apply_kq_scalar_soa(&mut re, &mut im, &[1, 2, 3], k, &data);

        for i in 0..a.len() {
            assert!(
                (a[i].re - re[i]).abs() < 1e-12 && (a[i].im - im[i]).abs() < 1e-12,
                "mismatch at i={i}: aos=({}, {}), soa=({}, {})",
                a[i].re,
                a[i].im,
                re[i],
                im[i]
            );
        }
    }
}
