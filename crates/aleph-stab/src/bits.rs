//! Packed-bit grid: `rows × cols` bits in a flat `Vec<u64>`.
//!
//! Row-major: row `r` lives in words `[r*stride, (r+1)*stride)`,
//! `stride = ceil(cols/64)`. All accessors are O(1). No bounds checks in
//! release (callers — the tableau — pass in-range indices derived from
//! `n`); `debug_assert` guards catch logic bugs in tests.

#[derive(Clone)]
pub(crate) struct BitGrid {
    words: Vec<u64>,
    stride: usize, // u64 words per row
    cols: usize,
}

impl BitGrid {
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        let stride = cols.div_ceil(64);
        BitGrid {
            words: vec![0u64; rows * stride],
            stride,
            cols,
        }
    }

    #[inline]
    fn word_index(&self, row: usize, col: usize) -> (usize, u64) {
        debug_assert!(col < self.cols, "col {col} out of range {}", self.cols);
        (row * self.stride + (col >> 6), 1u64 << (col & 63))
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, col: usize) -> bool {
        let (w, mask) = self.word_index(row, col);
        self.words[w] & mask != 0
    }

    #[inline]
    pub(crate) fn set(&mut self, row: usize, col: usize, val: bool) {
        let (w, mask) = self.word_index(row, col);
        if val {
            self.words[w] |= mask;
        } else {
            self.words[w] &= !mask;
        }
    }

    // --- word-level accessors for hoisted hot-loop indexing ---

    /// Number of `u64` words per row.
    #[inline]
    pub(crate) fn row_stride(&self) -> usize {
        self.stride
    }

    /// Read the word at a precomputed flat index `row * stride + word_col`.
    /// Callers must ensure `idx < self.words.len()`.
    #[inline]
    pub(crate) fn word(&self, idx: usize) -> u64 {
        self.words[idx]
    }

    /// Mutable reference to the word at a precomputed flat index.
    /// Callers must ensure `idx < self.words.len()`.
    #[inline]
    pub(crate) fn word_mut(&mut self, idx: usize) -> &mut u64 {
        &mut self.words[idx]
    }

    /// The `stride` words of row `r` (contiguous).
    // Part of the row word-slice accessor API (exercised by tests); the rowsum
    // hot path uses row_pair_mut. Kept for completeness / future column ops.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn row_words(&self, r: usize) -> &[u64] {
        let s = self.stride;
        debug_assert!((r + 1) * s <= self.words.len(), "row {r} out of range");
        &self.words[r * s..(r + 1) * s]
    }

    /// Mutable contiguous words of row `r`.
    // Part of the row word-slice accessor API (exercised by tests); the rowsum
    // hot path uses row_pair_mut. Kept for completeness / future column ops.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        let s = self.stride;
        debug_assert!((r + 1) * s <= self.words.len(), "row {r} out of range");
        &mut self.words[r * s..(r + 1) * s]
    }

    /// Number of rows in the grid (`words.len() / stride`). `stride ≥ 1` always.
    // Consumed by P3-11 Task 2+; allow until then.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.words.len() / self.stride
    }

    /// Bit-transpose: returns a `cols × rows` grid with output bit `(c, r)` =
    /// `self` bit `(r, c)`. Scalar reference implementation — a blocked kernel
    /// replaces the body in P3-11 Task 5, validated against this via a diff test.
    // Consumed by P3-11 Task 2+; allow until then.
    #[allow(dead_code)]
    pub(crate) fn transpose(&self) -> BitGrid {
        let rows = self.rows();
        let mut out = BitGrid::zeros(self.cols, rows);
        for r in 0..rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    out.set(c, r, true);
                }
            }
        }
        out
    }

    /// Mutable words of row `dst` and shared words of row `src`, borrowed
    /// simultaneously. Requires `dst != src` (rows live in one backing `Vec`,
    /// split via `split_at_mut`).
    #[inline]
    pub(crate) fn row_pair_mut(&mut self, dst: usize, src: usize) -> (&mut [u64], &[u64]) {
        debug_assert_ne!(dst, src, "row_pair_mut needs distinct rows");
        let s = self.stride;
        debug_assert!(
            (dst.max(src) + 1) * s <= self.words.len(),
            "row out of range"
        );
        if dst < src {
            let (lo, hi) = self.words.split_at_mut(src * s);
            (&mut lo[dst * s..(dst + 1) * s], &hi[..s])
        } else {
            let (lo, hi) = self.words.split_at_mut(dst * s);
            (&mut hi[..s], &lo[src * s..(src + 1) * s])
        }
    }
}

/// Packed bit-vector of `len` bits in `ceil(len/64)` u64 words. Unused high
/// bits in the final word are always zero (`set` only touches valid indices),
/// so word-parallel `&`/`^` consumers need no tail masking.
// Consumed by P3-11 Task 2+; allow until then.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct BitVec {
    words: Vec<u64>,
    len: usize,
}

#[allow(dead_code)]
impl BitVec {
    /// Allocate a zero-initialised bit-vector of `len` bits.
    pub(crate) fn zeros(len: usize) -> Self {
        BitVec {
            words: vec![0u64; len.div_ceil(64).max(1)],
            len,
        }
    }

    /// Return `true` if bit `i` is set.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len, "bit {i} out of range {}", self.len);
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }

    /// Set or clear bit `i`.
    #[inline]
    pub(crate) fn set(&mut self, i: usize, val: bool) {
        debug_assert!(i < self.len, "bit {i} out of range {}", self.len);
        let (w, m) = (i >> 6, 1u64 << (i & 63));
        if val {
            self.words[w] |= m;
        } else {
            self.words[w] &= !m;
        }
    }

    /// Shared slice of backing words for word-parallel operations.
    #[inline]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    /// Mutable slice of backing words for word-parallel operations.
    #[inline]
    pub(crate) fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

#[cfg(test)]
mod tests {
    use super::{BitGrid, BitVec};

    #[test]
    fn set_get_roundtrip() {
        let mut g = BitGrid::zeros(3, 130); // 3 rows, 130 cols (3 words/row)
        assert!(!g.get(2, 129));
        g.set(2, 129, true);
        assert!(g.get(2, 129));
        g.set(2, 129, false); // clear
        assert!(!g.get(2, 129));
        g.set(1, 0, true);
        assert!(g.get(1, 0));
        // independence: untouched cell stays false
        assert!(!g.get(0, 64));
    }

    #[test]
    fn word_accessors_consistent_with_get_set() {
        let mut g = BitGrid::zeros(4, 70); // stride = 2 words/row
        g.set(1, 5, true);
        g.set(1, 66, true); // second word of row 1
        let stride = g.row_stride();
        assert_eq!(stride, 2);
        // Row 1 word 0: bit 5 set
        assert_eq!(g.word(stride), 1u64 << 5);
        // Row 1 word 1: bit (66-64)=2 set
        assert_eq!(g.word(stride + 1), 1u64 << 2);
        // Mutate via word_mut and confirm via get
        *g.word_mut(2 * stride) |= 1u64 << 63;
        assert!(g.get(2, 63));
    }

    #[test]
    fn row_words_roundtrip_and_pair() {
        let mut g = BitGrid::zeros(4, 130); // stride 3 words/row
        g.set(1, 0, true);
        g.set(1, 129, true);
        g.set(2, 64, true);
        // row_words: row 1 has bit 0 (word0) and bit 129 (word2)
        let r1 = g.row_words(1);
        assert_eq!(r1.len(), 3);
        assert_eq!(r1[0], 1u64);
        assert_eq!(r1[2], 1u64 << (129 - 128));
        // row_pair_mut: borrow row 1 (mut) and row 2 (shared) at once, dst<src
        {
            let (dst, src) = g.row_pair_mut(1, 2);
            assert_eq!(dst.len(), 3);
            assert_eq!(src.len(), 3);
            assert_eq!(src[1], 1u64); // row 2 bit 64 -> word1 bit0
            dst[1] ^= src[1]; // row1 word1 gets bit 64
        }
        assert!(g.get(1, 64));
        // dst>src ordering also works
        {
            let (dst, src) = g.row_pair_mut(2, 1);
            dst[0] ^= src[0]; // row2 word0 ^= row1 word0 (bit0)
        }
        assert!(g.get(2, 0));
        // row_words_mut: mutate row 0 through the mutable slice
        {
            let w = g.row_words_mut(0);
            w[0] |= 1u64 << 5;
        }
        assert!(g.get(0, 5));
    }

    #[test]
    fn bitvec_set_get_and_words() {
        let mut v = BitVec::zeros(130); // 3 words
        assert_eq!(v.words().len(), 3);
        assert!(!v.get(129));
        v.set(129, true);
        assert!(v.get(129));
        v.set(129, false);
        assert!(!v.get(129));
        v.set(0, true);
        v.set(64, true);
        assert_eq!(v.words()[0], 1u64);
        assert_eq!(v.words()[1], 1u64);
        // word-level mutation visible through get
        v.words_mut()[2] ^= 1u64 << 1;
        assert!(v.get(129));
    }

    #[test]
    fn grid_rows_accessor() {
        let g = BitGrid::zeros(5, 70); // 5 rows, stride 2
        assert_eq!(g.rows(), 5);
        assert_eq!(BitGrid::zeros(1, 1).rows(), 1);
    }

    #[test]
    fn transpose_roundtrip_and_values() {
        // Deterministic fill, transpose, check (c,r)==(r,c), and T∘T == id.
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for &(rows, cols) in &[
            (1usize, 1usize),
            (3, 5),
            (7, 64),
            (65, 9),
            (128, 130),
            (483, 241),
        ] {
            let mut g = BitGrid::zeros(rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    if next() & 1 == 1 {
                        g.set(r, c, true);
                    }
                }
            }
            let t = g.transpose();
            assert_eq!(t.rows(), cols, "transpose row count ({rows}x{cols})");
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(t.get(c, r), g.get(r, c), "({r},{c}) {rows}x{cols}");
                }
            }
            // round trip
            let tt = t.transpose();
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(tt.get(r, c), g.get(r, c), "roundtrip ({r},{c})");
                }
            }
        }
    }
}
