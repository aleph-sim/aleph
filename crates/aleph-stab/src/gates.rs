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

/// Dispatch wrappers (AVX-512 lands in Task 4; for now they just call scalar).
#[inline]
pub(crate) fn h_dispatch(xa: &mut [u64], za: &mut [u64], sign: &mut [u64]) {
    h_words(xa, za, sign);
}
#[inline]
pub(crate) fn s_dispatch(xa: &[u64], za: &mut [u64], sign: &mut [u64]) {
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
    cnot_words(xa, xb, za, zb, sign);
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
}
