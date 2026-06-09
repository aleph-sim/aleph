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
