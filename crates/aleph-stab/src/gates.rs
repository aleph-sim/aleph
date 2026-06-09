//! Word-parallel Clifford gate kernels over column-major spans.
//!
//! In ColMajor orientation a qubit column is a contiguous `W = ceil((2n+1)/64)`
//! word-span (one `BitGrid` row); `sign` is a matching `W`-word bit-vector over
//! the same `2n+1` generator-row axis. Each kernel updates the whole column
//! word-parallel. Unused high bits in the final word are zero (BitGrid/BitVec
//! guarantee), so no tail masking is needed. See Aaronson–Gottesman (2004) §2.
//!
//! Scratch-row note: the span covers all `2n+1` rows, so these kernels also
//! update the scratch row `2n` — unlike the pre-P3-11 row-major path, which
//! looped `0..2n`. This is harmless: every scratch-row consumer (`measure`'s
//! deterministic branch, `pauli_eigenvalue`) calls `zero_row(2n)` before
//! reading it, so a gate-dirtied scratch row is always overwritten first.

/// H(a): `sign ^= x_a & z_a`; swap `x_a, z_a`.
pub(crate) fn h_words(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    debug_assert_eq!(xa.len(), za.len());
    debug_assert_eq!(xa.len(), sign.len());
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & za[w];
        core::mem::swap(&mut xa[w], &mut za[w]);
    }
}

/// S(a): `sign ^= x_a & z_a`; `z_a ^= x_a`.
pub(crate) fn s_words(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & za[w];
        za[w] ^= xa[w];
    }
}

/// CNOT(a,b): `sign ^= x_a & z_b & ~(x_b ^ z_a)`; `x_b ^= x_a`; `z_a ^= z_b`.
/// `xb`/`za` are read (for the sign) before being written, in one pass.
pub(crate) fn cnot_words(xa: &[u64], xb: &mut [u64], za: &mut [u64], zb: &[u64], sign: &mut [u64]) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] & zb[w] & !(xb[w] ^ za[w]);
        xb[w] ^= xa[w];
        za[w] ^= zb[w];
    }
}

/// `sign ^= col`. X uses the z-column; Z uses the x-column.
pub(crate) fn sign_xor_words(col: &[u64], sign: &mut [u64]) {
    for w in 0..col.len() {
        sign[w] ^= col[w];
    }
}

/// Y(a): `sign ^= x_a ^ z_a`.
pub(crate) fn y_sign_words(xa: &[u64], za: &[u64], sign: &mut [u64]) {
    for w in 0..xa.len() {
        sign[w] ^= xa[w] ^ za[w];
    }
}

/// Dispatch to the AVX-512 kernel when the CPU supports it, else scalar words.
#[inline]
pub(crate) fn h_dispatch(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f verified present; all slices equal length; the
            // kernel uses unaligned loads/stores within bounds and a scalar
            // tail for the `len % 8` remainder.
            return unsafe { h_avx512(xa, za, sign) };
        }
    }
    h_words(xa, za, sign);
}

#[inline]
pub(crate) fn s_dispatch(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: see `h_dispatch`.
            return unsafe { s_avx512(xa, za, sign) };
        }
    }
    s_words(xa, za, sign);
}

#[inline]
pub(crate) fn cnot_dispatch(
    xa: &[u64],
    xb: &mut [u64],
    za: &mut [u64],
    zb: &[u64],
    sign: &mut [u64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // SAFETY: see `h_dispatch`.
            return unsafe { cnot_avx512(xa, xb, za, zb, sign) };
        }
    }
    cnot_words(xa, xb, za, zb, sign);
}

/// AVX-512 form of [`h_words`]: 8×`u64` (512 bits) per step, scalar tail.
///
/// # Safety
/// Caller must ensure `avx512f` is available (checked by [`h_dispatch`]).
/// All three slices must have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn h_avx512(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let xw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let zw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        let ns = _mm512_xor_si512(sw, _mm512_and_si512(xw, zw));
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        // swap x and z
        _mm512_storeu_si512(xa.as_mut_ptr().add(off) as *mut __m512i, zw);
        _mm512_storeu_si512(za.as_mut_ptr().add(off) as *mut __m512i, xw);
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & za[w];
        core::mem::swap(&mut xa[w], &mut za[w]);
    }
}

/// AVX-512 form of [`s_words`].
///
/// # Safety
/// See [`h_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn s_avx512(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let xw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let zw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        let ns = _mm512_xor_si512(sw, _mm512_and_si512(xw, zw));
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        _mm512_storeu_si512(
            za.as_mut_ptr().add(off) as *mut __m512i,
            _mm512_xor_si512(zw, xw),
        );
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & za[w];
        za[w] ^= xa[w];
    }
}

/// AVX-512 form of [`cnot_words`]. Reads `x_b`/`z_a` for the sign before
/// overwriting them (all four operands are loaded up front each step).
///
/// # Safety
/// See [`h_avx512`]. All five slices must have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cnot_avx512(xa: &[u64], xb: &mut [u64], za: &mut [u64], zb: &[u64], sign: &mut [u64]) {
    use core::arch::x86_64::*;
    let len = xa.len();
    let chunks = len / 8;
    let ones = _mm512_set1_epi64(-1);
    for c in 0..chunks {
        let off = c * 8;
        let xaw = _mm512_loadu_si512(xa.as_ptr().add(off) as *const __m512i);
        let xbw = _mm512_loadu_si512(xb.as_ptr().add(off) as *const __m512i);
        let zaw = _mm512_loadu_si512(za.as_ptr().add(off) as *const __m512i);
        let zbw = _mm512_loadu_si512(zb.as_ptr().add(off) as *const __m512i);
        let sw = _mm512_loadu_si512(sign.as_ptr().add(off) as *const __m512i);
        // ~(x_b ^ z_a)
        let nxnz = _mm512_andnot_si512(_mm512_xor_si512(xbw, zaw), ones);
        // x_a & z_b & ~(x_b ^ z_a)
        let term = _mm512_and_si512(_mm512_and_si512(xaw, zbw), nxnz);
        let ns = _mm512_xor_si512(sw, term);
        _mm512_storeu_si512(sign.as_mut_ptr().add(off) as *mut __m512i, ns);
        _mm512_storeu_si512(
            xb.as_mut_ptr().add(off) as *mut __m512i,
            _mm512_xor_si512(xbw, xaw),
        );
        _mm512_storeu_si512(
            za.as_mut_ptr().add(off) as *mut __m512i,
            _mm512_xor_si512(zaw, zbw),
        );
    }
    for w in (chunks * 8)..len {
        sign[w] ^= xa[w] & zb[w] & !(xb[w] ^ za[w]);
        xb[w] ^= xa[w];
        za[w] ^= zb[w];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(w: &[u64], j: usize) -> bool {
        w[j >> 6] & (1u64 << (j & 63)) != 0
    }
    fn put(w: &mut [u64], j: usize, v: bool) {
        let (i, m) = (j >> 6, 1u64 << (j & 63));
        if v {
            w[i] |= m;
        } else {
            w[i] &= !m;
        }
    }

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
    fn h_matches_per_bit() {
        let mut rng = Rng(0xDEADBEEF1234);
        for w in [1usize, 2, 5, 8, 9] {
            let bits = w * 64;
            let mut xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (x0, z0, s0) = (xa.clone(), za.clone(), sg.clone());
            h_words(&mut xa, &mut za, &mut sg);
            let (mut xr, mut zr, mut sr) = (x0.clone(), z0.clone(), s0.clone());
            for j in 0..bits {
                let (xj, zj) = (get(&x0, j), get(&z0, j));
                put(&mut sr, j, get(&s0, j) ^ (xj & zj));
                put(&mut xr, j, zj); // swap
                put(&mut zr, j, xj);
            }
            assert_eq!(xa, xr, "h x w={w}");
            assert_eq!(za, zr, "h z w={w}");
            assert_eq!(sg, sr, "h sign w={w}");
        }
    }

    #[test]
    fn s_matches_per_bit() {
        let mut rng = Rng(0xABCDEF01);
        for w in [1usize, 3, 8] {
            let bits = w * 64;
            let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (z0, s0) = (za.clone(), sg.clone());
            s_words(&xa, &mut za, &mut sg);
            let (mut zr, mut sr) = (z0.clone(), s0.clone());
            for j in 0..bits {
                let (xj, zj) = (get(&xa, j), get(&z0, j));
                put(&mut sr, j, get(&s0, j) ^ (xj & zj));
                put(&mut zr, j, zj ^ xj);
            }
            assert_eq!(za, zr, "s z w={w}");
            assert_eq!(sg, sr, "s sign w={w}");
        }
    }

    #[test]
    fn cnot_matches_per_bit() {
        let mut rng = Rng(0x55AA55AA);
        for w in [1usize, 2, 8, 9] {
            let bits = w * 64;
            let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut xb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let zb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let mut sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();
            let (xb0, za0, s0) = (xb.clone(), za.clone(), sg.clone());
            cnot_words(&xa, &mut xb, &mut za, &zb, &mut sg);
            let (mut xbr, mut zar, mut sr) = (xb0.clone(), za0.clone(), s0.clone());
            for j in 0..bits {
                let (xaj, xbj, zaj, zbj) = (get(&xa, j), get(&xb0, j), get(&za0, j), get(&zb, j));
                put(&mut sr, j, get(&s0, j) ^ (xaj & zbj & !(xbj ^ zaj)));
                put(&mut xbr, j, xbj ^ xaj);
                put(&mut zar, j, zaj ^ zbj);
            }
            assert_eq!(xb, xbr, "cnot xb w={w}");
            assert_eq!(za, zar, "cnot za w={w}");
            assert_eq!(sg, sr, "cnot sign w={w}");
        }
    }

    #[test]
    fn avx512_gates_match_scalar_when_available() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                return; // skip on non-AVX512 hosts (local aarch64, GH macos, Ryzen)
            }
            let mut rng = Rng(0x0F1E2D3C4B5A6978);
            for w in [1usize, 2, 7, 8, 9, 16, 17] {
                for _ in 0..500 {
                    let xa: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                    let za: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                    let zb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                    let xb: Vec<u64> = (0..w).map(|_| rng.next()).collect();
                    let sg: Vec<u64> = (0..w).map(|_| rng.next()).collect();

                    // H
                    let (mut xa1, mut za1, mut s1) = (xa.clone(), za.clone(), sg.clone());
                    let (mut xa2, mut za2, mut s2) = (xa.clone(), za.clone(), sg.clone());
                    unsafe { h_avx512(&mut xa1, &mut za1, &mut s1) };
                    h_words(&mut xa2, &mut za2, &mut s2);
                    assert_eq!((&xa1, &za1, &s1), (&xa2, &za2, &s2), "H w={w}");

                    // S
                    let (mut za1, mut s1) = (za.clone(), sg.clone());
                    let (mut za2, mut s2) = (za.clone(), sg.clone());
                    unsafe { s_avx512(&xa, &mut za1, &mut s1) };
                    s_words(&xa, &mut za2, &mut s2);
                    assert_eq!((&za1, &s1), (&za2, &s2), "S w={w}");

                    // CNOT
                    let (mut xb1, mut za1, mut s1) = (xb.clone(), za.clone(), sg.clone());
                    let (mut xb2, mut za2, mut s2) = (xb.clone(), za.clone(), sg.clone());
                    unsafe { cnot_avx512(&xa, &mut xb1, &mut za1, &zb, &mut s1) };
                    cnot_words(&xa, &mut xb2, &mut za2, &zb, &mut s2);
                    assert_eq!((&xb1, &za1, &s1), (&xb2, &za2, &s2), "CNOT w={w}");
                }
            }
        }
    }
}
