//! Word-parallel `rowsum` kernels for the CHP tableau. `rowsum(h, i)`
//! left-multiplies Pauli row `i` onto row `h`: it XORs the x/z bit-vectors
//! and returns the Aaronson–Gottesman phase exponent Σ_j g(row_i[j], row_h[j])
//! (the caller adds the `2·sign` terms and reduces mod 4). See AG (2004) §2.
//!
//! Rows are contiguous `u64` words (see `BitGrid::row_pair_mut`). Out-of-range
//! high bits in the last word are always zero, so no tail masking is needed.
//! (`BitGrid` guarantees this; these kernels never mask.)

/// Scalar word-parallel kernel. Reads the original bits of both rows to
/// compute the phase, then XORs row `i` into row `h` in place. Returns the
/// phase exponent contribution (Σ g, may be negative).
pub(crate) fn rowsum_words(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    debug_assert_eq!(xh.len(), xi.len());
    debug_assert_eq!(zh.len(), zi.len());
    debug_assert_eq!(xh.len(), zh.len());
    let mut acc: i64 = 0;
    // Phase pass: must read the original bits of both rows before any
    // mutation. A merged single pass would corrupt the phase for bits
    // already overwritten by the XOR.
    for w in 0..xh.len() {
        let (xiw, ziw, xhw, zhw) = (xi[w], zi[w], xh[w], zh[w]);
        let plus = (xiw & !ziw & zhw & xhw) | (!xiw & ziw & xhw & !zhw) | (xiw & ziw & zhw & !xhw);
        let minus = (xiw & !ziw & zhw & !xhw) | (!xiw & ziw & xhw & zhw) | (xiw & ziw & xhw & !zhw);
        acc += plus.count_ones() as i64 - minus.count_ones() as i64;
    }
    // XOR pass (write row h).
    for w in 0..xh.len() {
        xh[w] ^= xi[w];
        zh[w] ^= zi[w];
    }
    acc
}

/// Dispatch to the AVX-512 kernel when the CPU supports it, else the scalar
/// word kernel. Same contract as [`rowsum_words`].
#[inline]
pub(crate) fn rowsum_dispatch(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            // SAFETY: both features verified present immediately above; slices
            // are equal-length and the kernel only does aligned-agnostic
            // (`loadu`/`storeu`) accesses within bounds. Each chunk accesses
            // offsets `c*8 .. c*8+8` with `c < len/8`, so all loads/stores
            // stay in bounds; the `len % 8` remainder is handled by the scalar
            // tail.
            return unsafe { rowsum_avx512(xh, xi, zh, zi) };
        }
    }
    rowsum_words(xh, xi, zh, zi)
}

/// AVX-512 + VPOPCNTQ implementation of [`rowsum_words`]. Processes 8 `u64`
/// (512 bits) per step; a scalar tail handles the remaining `len % 8` words.
///
/// # Safety
/// Caller must ensure `avx512f` and `avx512vpopcntdq` are available (checked by
/// [`rowsum_dispatch`]). All four slices (`xh`, `xi`, `zh`, `zi`) must have the
/// same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn rowsum_avx512(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
    debug_assert_eq!(xh.len(), xi.len());
    debug_assert_eq!(zh.len(), zi.len());
    debug_assert_eq!(xh.len(), zh.len());
    use core::arch::x86_64::*;
    let len = xh.len();
    let chunks = len / 8;
    let mut acc_v = _mm512_setzero_si512(); // 8 lanes of signed i64: Σ(plus-minus)
    for c in 0..chunks {
        let off = c * 8;
        let xiw = _mm512_loadu_si512(xi.as_ptr().add(off) as *const __m512i);
        let ziw = _mm512_loadu_si512(zi.as_ptr().add(off) as *const __m512i);
        let xhw = _mm512_loadu_si512(xh.as_ptr().add(off) as *const __m512i);
        let zhw = _mm512_loadu_si512(zh.as_ptr().add(off) as *const __m512i);

        // helpers
        let nzi = _mm512_andnot_si512(ziw, _mm512_set1_epi64(-1)); // !zi
        let nzh = _mm512_andnot_si512(zhw, _mm512_set1_epi64(-1)); // !zh
        let nxi = _mm512_andnot_si512(xiw, _mm512_set1_epi64(-1)); // !xi
        let nxh = _mm512_andnot_si512(xhw, _mm512_set1_epi64(-1)); // !xh

        let and4 = |a, b, cc, d| _mm512_and_si512(_mm512_and_si512(a, b), _mm512_and_si512(cc, d));

        let plus = _mm512_or_si512(
            _mm512_or_si512(and4(xiw, nzi, zhw, xhw), and4(nxi, ziw, xhw, nzh)),
            and4(xiw, ziw, zhw, nxh),
        );
        let minus = _mm512_or_si512(
            _mm512_or_si512(and4(xiw, nzi, zhw, nxh), and4(nxi, ziw, xhw, zhw)),
            and4(xiw, ziw, xhw, nzh),
        );
        // per-lane popcount (VPOPCNTQ), accumulate plus - minus
        let pc_plus = _mm512_popcnt_epi64(plus);
        let pc_minus = _mm512_popcnt_epi64(minus);
        acc_v = _mm512_add_epi64(acc_v, _mm512_sub_epi64(pc_plus, pc_minus));

        // XOR pass for this chunk
        let nx = _mm512_xor_si512(xhw, xiw);
        let nz = _mm512_xor_si512(zhw, ziw);
        _mm512_storeu_si512(xh.as_mut_ptr().add(off) as *mut __m512i, nx);
        _mm512_storeu_si512(zh.as_mut_ptr().add(off) as *mut __m512i, nz);
    }
    let mut acc = _mm512_reduce_add_epi64(acc_v);
    // scalar tail
    for w in (chunks * 8)..len {
        let (xiw, ziw, xhw, zhw) = (xi[w], zi[w], xh[w], zh[w]);
        let plus = (xiw & !ziw & zhw & xhw) | (!xiw & ziw & xhw & !zhw) | (xiw & ziw & zhw & !xhw);
        let minus = (xiw & !ziw & zhw & !xhw) | (!xiw & ziw & xhw & zhw) | (xiw & ziw & xhw & !zhw);
        acc += plus.count_ones() as i64 - minus.count_ones() as i64;
        xh[w] ^= xi[w];
        zh[w] ^= zi[w];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-bit reference (the pre-P3-08 algorithm), used only to validate the
    /// word kernels. Mirrors the original `Tableau::g` + per-bit loops.
    // Verbatim copy of the canonical g (tableau.rs); kept private there, so duplicated here as the independent oracle.
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        let x2 = x2 as i32;
        let z2 = z2 as i32;
        match (x1, z1) {
            (false, false) => 0,
            (true, false) => z2 * (2 * x2 - 1),
            (false, true) => x2 * (1 - 2 * z2),
            (true, true) => z2 - x2,
        }
    }

    fn rowsum_ref(xh: &mut [u64], xi: &[u64], zh: &mut [u64], zi: &[u64]) -> i64 {
        let bits = xh.len() * 64;
        let get = |w: &[u64], j: usize| (w[j >> 6] >> (j & 63)) & 1 == 1;
        let mut acc: i64 = 0;
        for j in 0..bits {
            acc += g(get(xi, j), get(zi, j), get(xh, j), get(zh, j)) as i64;
        }
        for w in 0..xh.len() {
            xh[w] ^= xi[w];
            zh[w] ^= zi[w];
        }
        acc
    }

    // Deterministic xorshift RNG (no proptest dep; reproducible).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn avx512_matches_reference_when_available() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512vpopcntdq"))
            {
                return; // no AVX-512 here (e.g. GitHub macos / non-AVX512 linux) — skip
            }
            let mut rng = Rng(0xC2B2AE3D27D4EB4F);
            for stride in [1usize, 2, 7, 8, 9, 16, 17] {
                for _ in 0..1000 {
                    let xi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let zi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let xh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let zh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                    let (mut xa, mut za) = (xh0.clone(), zh0.clone());
                    let pa = unsafe { rowsum_avx512(&mut xa, &xi, &mut za, &zi) };
                    let (mut xb, mut zb) = (xh0.clone(), zh0.clone());
                    let pb = rowsum_ref(&mut xb, &xi, &mut zb, &zi);
                    assert_eq!(pa, pb, "avx512 phase mismatch stride {stride}");
                    assert_eq!(xa, xb, "avx512 x mismatch stride {stride}");
                    assert_eq!(za, zb, "avx512 z mismatch stride {stride}");
                }
            }
        }
    }

    #[test]
    fn words_match_per_bit_reference() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for stride in [1usize, 2, 4, 8, 13] {
            for _ in 0..2000 {
                let xi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let zi: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let xh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();
                let zh0: Vec<u64> = (0..stride).map(|_| rng.next()).collect();

                let (mut xa, mut za) = (xh0.clone(), zh0.clone());
                let pa = rowsum_words(&mut xa, &xi, &mut za, &zi);

                let (mut xb, mut zb) = (xh0.clone(), zh0.clone());
                let pb = rowsum_ref(&mut xb, &xi, &mut zb, &zi);

                assert_eq!(pa, pb, "phase mismatch at stride {stride}");
                assert_eq!(xa, xb, "x XOR mismatch at stride {stride}");
                assert_eq!(za, zb, "z XOR mismatch at stride {stride}");
            }
        }
    }
}
