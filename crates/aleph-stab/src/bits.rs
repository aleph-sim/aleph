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
    // used by the word-parallel rowsum kernel (wired in a later P3-08 commit)
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn row_words(&self, r: usize) -> &[u64] {
        let s = self.stride;
        debug_assert!((r + 1) * s <= self.words.len(), "row {r} out of range");
        &self.words[r * s..(r + 1) * s]
    }

    /// Mutable contiguous words of row `r`.
    // used by the word-parallel rowsum kernel (wired in a later P3-08 commit)
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn row_words_mut(&mut self, r: usize) -> &mut [u64] {
        let s = self.stride;
        debug_assert!((r + 1) * s <= self.words.len(), "row {r} out of range");
        &mut self.words[r * s..(r + 1) * s]
    }

    /// Mutable words of row `dst` and shared words of row `src`, borrowed
    /// simultaneously. Requires `dst != src` (rows live in one backing `Vec`,
    /// split via `split_at_mut`).
    // used by the word-parallel rowsum kernel (wired in a later P3-08 commit)
    #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::BitGrid;

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
}
