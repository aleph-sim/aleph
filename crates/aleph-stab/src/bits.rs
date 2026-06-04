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

#[allow(dead_code)] // methods used in later tasks (tableau gate implementations)
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

    #[inline]
    pub(crate) fn toggle(&mut self, row: usize, col: usize) {
        let (w, mask) = self.word_index(row, col);
        self.words[w] ^= mask;
    }
}

#[cfg(test)]
mod tests {
    use super::BitGrid;

    #[test]
    fn set_get_toggle_roundtrip() {
        let mut g = BitGrid::zeros(3, 130); // 3 rows, 130 cols (3 words/row)
        assert!(!g.get(2, 129));
        g.set(2, 129, true);
        assert!(g.get(2, 129));
        g.toggle(2, 129); // -> false
        assert!(!g.get(2, 129));
        g.toggle(1, 0); // -> true
        assert!(g.get(1, 0));
        // independence: untouched cell stays false
        assert!(!g.get(0, 64));
    }
}
