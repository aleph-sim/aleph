//! Dense k-qubit gate kernel: one pass, a 2^k×2^k matvec per 2^k-block.
//!
//! Two implementations of the same map live here:
//!
//! * **Scalar** (`*_scalar_aos` / `*_scalar_soa`) — the reference, runs on
//!   every target. Iterates over `outer = 2^(n-k)` blocks; per block it gathers
//!   the `2^k` amplitudes via `base | offsets[m]`, does the dense `2^k × 2^k`
//!   matvec, and scatters the result back.
//! * **AVX-512** (`*_avx512_aos` / `*_avx512_soa`, x86_64-only) — vectorizes the
//!   **outer walk**: 8 independent outer blocks at once, one `__m512d` lane per
//!   block. The dense matvec is small (`2^k × 2^k`) and shared across the 8
//!   blocks, so each matrix cell `data[r*dim+c]` is a scalar broadcast and the
//!   per-lane input `in[block][c]` drives a packed complex FMA across the 8
//!   lanes. The 8 lanes' c-th amplitude is fetched with `_mm512_i64gather_pd`
//!   (the 8 block bases are not contiguous — `expand_with_fixed` scatters free
//!   bits) and written back with `_mm512_i64scatter_pd`.
//!
//! The public `apply_kq_aos` / `_soa` dispatchers pick the SIMD path at runtime
//! when `avx512f` is present AND there are at least 8 outer blocks (`outer >= 8`,
//! i.e. `n - k >= 3`); they fall back to the scalar kernel otherwise. The two
//! paths are bit-for-bit equivalent modulo FP rounding (gated by
//! `dispatcher_matches_scalar_dense`, run for real on an AVX-512 box).
use aleph_core::Complex;

/// Apply a dense `2^k × 2^k` unitary (`data`, row-major, MSB-first operand
/// order `qubits[0]`=MSB) to an AoS state in one pass.
pub(crate) fn apply_kq_aos(amps: &mut [Complex], qubits: &[u32], k: u8, data: &[Complex]) {
    #[cfg(target_arch = "x86_64")]
    {
        let outer = amps.len() >> k;
        if outer >= AVX512_LANES && std::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f detected immediately above, and `outer >= 8`
            // guarantees at least one full 8-block group; `outer = 2^(n-k)` is
            // a power of two ≥ 8, hence a multiple of 8, so the SIMD kernel
            // processes every block with no scalar tail.
            unsafe { apply_kq_avx512_aos(amps, qubits, k, data) };
            return;
        }
    }
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
    #[cfg(target_arch = "x86_64")]
    {
        let outer = re.len() >> k;
        if outer >= AVX512_LANES && re.len() == im.len() && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: avx512f detected above; `re.len() == im.len()` and
            // `outer = 2^(n-k) >= 8` is a multiple of 8, so the SIMD kernel has
            // no scalar tail. `re`/`im` are separate, non-aliasing buffers.
            unsafe { apply_kq_avx512_soa(re, im, qubits, k, data) };
            return;
        }
    }
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
pub(crate) fn targets_offsets_fixed(qubits: &[u32], k: u8) -> (Vec<usize>, Vec<(u32, bool)>) {
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
// AVX-512 kernels (x86_64 only).
//
// Strategy: vectorize the OUTER walk. We process `AVX512_LANES` (= 8)
// independent outer blocks per step — one `__m512d` lane per block. The dense
// matvec (`2^k × 2^k`) is identical across the 8 blocks, so each matrix cell
// `data[r*dim+c]` is broadcast to all lanes and the per-lane input
// `in[block][c]` drives a packed complex FMA. We keep separate real/imag
// `__m512d` accumulators (mirrors the scalar SoA in-line complex multiply):
//   acc_re += d_re*in_re[c] - d_im*in_im[c]
//   acc_im += d_re*in_im[c] + d_im*in_re[c]
// The 8 lanes' c-th amplitude lives at `base_b | offsets[c]` for the 8 block
// bases `base_b`; these are gathered with `_mm512_i64gather_pd` and the result
// scattered back with `_mm512_i64scatter_pd`.
// ---------------------------------------------------------------------------

/// SIMD lane width in `f64`s: one `__m512d` holds 8 lanes → 8 outer blocks.
#[cfg(target_arch = "x86_64")]
const AVX512_LANES: usize = 8;

/// AVX-512 AoS dense `apply_kq`. Vectorizes the outer walk: 8 outer blocks per
/// step, gather/scatter the per-lane block amplitudes from the interleaved
/// `[re, im, re, im, …]` storage (amplitude index `i` → f64 offset `2*i` for
/// re, `2*i + 1` for im).
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports `avx512f`.
/// * `outer = amps.len() >> k >= AVX512_LANES` (= 8); since `outer` is a power
///   of two this makes it a multiple of 8, so there is no scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_kq_avx512_aos(amps: &mut [Complex], qubits: &[u32], k: u8, data: &[Complex]) {
    use std::arch::x86_64::*;

    let dim = 1usize << k;
    let (offsets, fixed) = targets_offsets_fixed(qubits, k);
    let len = amps.len();
    let outer = len >> k;
    debug_assert_eq!(outer % AVX512_LANES, 0);

    // Pre-split each matrix cell into broadcast real/imag vectors — constant
    // across all outer blocks.
    let mut d_re = vec![_mm512_setzero_pd(); dim * dim];
    let mut d_im = vec![_mm512_setzero_pd(); dim * dim];
    for (cell, d) in data.iter().enumerate().take(dim * dim) {
        d_re[cell] = _mm512_set1_pd(d.re);
        d_im[cell] = _mm512_set1_pd(d.im);
    }

    // Flat `*mut f64` view over the interleaved Complex storage.
    let bp = crate::kernels::BlockPtr(amps.as_mut_ptr() as *mut f64);
    let groups = outer / AVX512_LANES;

    crate::kernels::par_blocks(
        crate::kernels::tuning::DEFAULT_POLICY,
        groups,
        len,
        |g| g,
        move |g| {
            // SAFETY: par_blocks hands each task a distinct group index `g`.
            // The 8 counters g*8..g*8+8 map (via expand_with_fixed) to 8
            // distinct base indices; combined with offsets[m] they cover 8
            // disjoint 2^k-blocks, themselves disjoint from every other group's
            // blocks (the scalar kernel's coverage invariant, indexed in groups
            // of 8). So gather/scatter across tasks never alias. BlockPtr is
            // Send+Sync; avx512f is guaranteed by this fn's #[target_feature].
            let ptr = bp.ptr();
            let counter0 = g * AVX512_LANES;

            // Base index (target bits cleared) for each of the 8 lanes.
            let mut bases = [0i64; AVX512_LANES];
            for (lane, b) in bases.iter_mut().enumerate() {
                *b = crate::kernels::expand_with_fixed(counter0 + lane, &fixed) as i64;
            }
            let base_v = _mm512_loadu_si512(bases.as_ptr() as *const _);

            // Gather the 2^k inputs (one __m512d per matrix column), as
            // separate real/imag vectors. AoS: re at 2*idx, im at 2*idx+1.
            let mut in_re = vec![_mm512_setzero_pd(); dim];
            let mut in_im = vec![_mm512_setzero_pd(); dim];
            for c in 0..dim {
                // f64 index of amplitude (base | offsets[c]) is 2*(base|off).
                let amp_idx = _mm512_or_si512(base_v, _mm512_set1_epi64(offsets[c] as i64));
                let re_idx = _mm512_slli_epi64(amp_idx, 1); // *2
                let im_idx = _mm512_add_epi64(re_idx, _mm512_set1_epi64(1));
                // scale = 8 bytes per f64; indices are in f64 units.
                in_re[c] = _mm512_i64gather_pd(re_idx, ptr as *const f64, 8);
                in_im[c] = _mm512_i64gather_pd(im_idx, ptr as *const f64, 8);
            }

            // Matvec: out[r] = Σ_c data[r*dim+c] * in[c], complex, per lane.
            for r in 0..dim {
                let mut acc_re = _mm512_setzero_pd();
                let mut acc_im = _mm512_setzero_pd();
                for c in 0..dim {
                    let dr = d_re[r * dim + c];
                    di_fma(
                        &mut acc_re,
                        &mut acc_im,
                        dr,
                        d_im[r * dim + c],
                        in_re[c],
                        in_im[c],
                    );
                }
                // Scatter back to amplitude (base | offsets[r]).
                let amp_idx = _mm512_or_si512(base_v, _mm512_set1_epi64(offsets[r] as i64));
                let re_idx = _mm512_slli_epi64(amp_idx, 1);
                let im_idx = _mm512_add_epi64(re_idx, _mm512_set1_epi64(1));
                _mm512_i64scatter_pd(ptr, re_idx, acc_re, 8);
                _mm512_i64scatter_pd(ptr, im_idx, acc_im, 8);
            }
        },
    );
}

/// AVX-512 SoA dense `apply_kq`. Same outer-walk vectorization; gathers/scatters
/// directly from the split `re` / `im` arrays (no de-interleave).
///
/// # Safety
///
/// Caller MUST ensure all of:
/// * Host CPU supports `avx512f`.
/// * `re.len() == im.len()`, and `outer = re.len() >> k >= AVX512_LANES` (= 8),
///   a power of two and hence a multiple of 8 (no scalar tail).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_kq_avx512_soa(
    re: &mut [f64],
    im: &mut [f64],
    qubits: &[u32],
    k: u8,
    data: &[Complex],
) {
    use std::arch::x86_64::*;

    let dim = 1usize << k;
    let (offsets, fixed) = targets_offsets_fixed(qubits, k);
    let len = re.len();
    debug_assert_eq!(len, im.len());
    let outer = len >> k;
    debug_assert_eq!(outer % AVX512_LANES, 0);

    let mut d_re = vec![_mm512_setzero_pd(); dim * dim];
    let mut d_im = vec![_mm512_setzero_pd(); dim * dim];
    for (cell, d) in data.iter().enumerate().take(dim * dim) {
        d_re[cell] = _mm512_set1_pd(d.re);
        d_im[cell] = _mm512_set1_pd(d.im);
    }

    let rp = crate::kernels::BlockPtr(re.as_mut_ptr());
    let ip = crate::kernels::BlockPtr(im.as_mut_ptr());
    let groups = outer / AVX512_LANES;

    crate::kernels::par_blocks(
        crate::kernels::tuning::DEFAULT_POLICY,
        groups,
        len,
        |g| g,
        move |g| {
            // SAFETY: distinct group index `g` per task ⇒ 8 disjoint 2^k-blocks
            // per group, disjoint across groups (same coverage invariant as the
            // scalar kernel, in groups of 8). rp/ip index two separate,
            // non-aliasing buffers; both BlockPtrs are Send+Sync. avx512f
            // guaranteed by this fn's #[target_feature].
            let rptr = rp.ptr();
            let iptr = ip.ptr();
            let counter0 = g * AVX512_LANES;

            let mut bases = [0i64; AVX512_LANES];
            for (lane, b) in bases.iter_mut().enumerate() {
                *b = crate::kernels::expand_with_fixed(counter0 + lane, &fixed) as i64;
            }
            let base_v = _mm512_loadu_si512(bases.as_ptr() as *const _);

            let mut in_re = vec![_mm512_setzero_pd(); dim];
            let mut in_im = vec![_mm512_setzero_pd(); dim];
            for c in 0..dim {
                let amp_idx = _mm512_or_si512(base_v, _mm512_set1_epi64(offsets[c] as i64));
                // scale = 8 bytes per f64; indices are amplitude indices.
                in_re[c] = _mm512_i64gather_pd(amp_idx, rptr as *const f64, 8);
                in_im[c] = _mm512_i64gather_pd(amp_idx, iptr as *const f64, 8);
            }

            for r in 0..dim {
                let mut acc_re = _mm512_setzero_pd();
                let mut acc_im = _mm512_setzero_pd();
                for c in 0..dim {
                    di_fma(
                        &mut acc_re,
                        &mut acc_im,
                        d_re[r * dim + c],
                        d_im[r * dim + c],
                        in_re[c],
                        in_im[c],
                    );
                }
                let amp_idx = _mm512_or_si512(base_v, _mm512_set1_epi64(offsets[r] as i64));
                _mm512_i64scatter_pd(rptr, amp_idx, acc_re, 8);
                _mm512_i64scatter_pd(iptr, amp_idx, acc_im, 8);
            }
        },
    );
}

/// Packed complex multiply-accumulate across 8 lanes:
///   acc_re += d_re*in_re - d_im*in_im
///   acc_im += d_re*in_im + d_im*in_re
/// where `d_re`/`d_im` are per-cell broadcasts and `in_re`/`in_im` vary per lane.
/// Mirrors the scalar SoA in-line complex multiply exactly.
///
/// # Safety
///
/// Caller MUST ensure the host supports `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn di_fma(
    acc_re: &mut std::arch::x86_64::__m512d,
    acc_im: &mut std::arch::x86_64::__m512d,
    d_re: std::arch::x86_64::__m512d,
    d_im: std::arch::x86_64::__m512d,
    in_re: std::arch::x86_64::__m512d,
    in_im: std::arch::x86_64::__m512d,
) {
    use std::arch::x86_64::*;
    // acc_re += d_re*in_re - d_im*in_im
    *acc_re = _mm512_fmadd_pd(d_re, in_re, *acc_re);
    *acc_re = _mm512_fnmadd_pd(d_im, in_im, *acc_re);
    // acc_im += d_re*in_im + d_im*in_re
    *acc_im = _mm512_fmadd_pd(d_re, in_im, *acc_im);
    *acc_im = _mm512_fmadd_pd(d_im, in_re, *acc_im);
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

    /// Scalar kernel vs the runtime dispatcher (SIMD on AVX-512 hosts, scalar
    /// elsewhere) must agree to 1e-13, for both AoS and SoA. On aarch64 this is
    /// scalar-vs-scalar (smoke); on EPYC it gates the AVX-512 kernel against the
    /// reference. Triples chosen so `outer = 2^(n-k) >= 8` (SIMD path engages):
    /// n=6,k=3 → outer=8; n=7,k=4 → outer=8.
    #[test]
    fn dispatcher_matches_scalar_dense() {
        use aleph_core::Complex;
        for (n, k, q) in [
            (6u32, 3u8, vec![0u32, 2, 4]),
            (6, 3, vec![1, 3, 5]),
            (7, 4, vec![0u32, 1, 2, 3]),
        ] {
            let dim = 1usize << k;
            // arbitrary dense matrix (not unitary — kernel is linear, fine for equivalence)
            let data: Vec<Complex> = (0..dim * dim)
                .map(|i| Complex::new((i % 7) as f64 * 0.13 - 0.4, (i % 5) as f64 * 0.21 - 0.3))
                .collect();
            let st: Vec<Complex> = (0..(1usize << n))
                .map(|i| Complex::new(0.3 * i as f64 + 1.0, 0.1 - 0.05 * i as f64))
                .collect();
            // AoS: scalar vs dispatcher
            let mut a_sc = st.clone();
            apply_kq_scalar_aos(&mut a_sc, &q, k, &data);
            let mut a_di = st.clone();
            apply_kq_aos(&mut a_di, &q, k, &data);
            for i in 0..a_sc.len() {
                assert!((a_sc[i] - a_di[i]).norm() < 1e-13, "aos n{n} k{k} i{i}");
            }
            // SoA
            let mut r_sc: Vec<f64> = st.iter().map(|c| c.re).collect();
            let mut i_sc: Vec<f64> = st.iter().map(|c| c.im).collect();
            apply_kq_scalar_soa(&mut r_sc, &mut i_sc, &q, k, &data);
            let mut r_di: Vec<f64> = st.iter().map(|c| c.re).collect();
            let mut i_di: Vec<f64> = st.iter().map(|c| c.im).collect();
            apply_kq_soa(&mut r_di, &mut i_di, &q, k, &data);
            for i in 0..r_sc.len() {
                assert!(
                    (r_sc[i] - r_di[i]).abs() < 1e-13 && (i_sc[i] - i_di[i]).abs() < 1e-13,
                    "soa n{n} k{k} i{i}"
                );
            }
        }
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
