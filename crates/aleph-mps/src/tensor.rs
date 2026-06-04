//! Rank-3 MPS site tensor `(left, 2, right)` and its reshape ↔ nalgebra views.

use aleph_core::Complex;
use nalgebra::DMatrix;

/// A single MPS site tensor of shape `(left, 2, right)`.
/// Row-major flat storage: `data[(l*2 + p)*right + r]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub left: usize,
    pub right: usize,
    pub data: Vec<Complex>,
}

impl Site {
    /// Returns a `(left, 2, right)` tensor filled with zeros.
    pub fn zeros(left: usize, right: usize) -> Self {
        Site {
            left,
            right,
            data: vec![Complex::new(0.0, 0.0); left * 2 * right],
        }
    }

    /// Returns the `(1, 2, 1)` tensor `[1, 0]` — a qubit in state |0⟩.
    pub fn ket0() -> Self {
        let mut s = Site::zeros(1, 1);
        s.data[0] = Complex::new(1.0, 0.0);
        s
    }

    /// Flat row-major index for `(l, p, r)`.
    #[inline]
    pub fn idx(&self, l: usize, p: usize, r: usize) -> usize {
        (l * 2 + p) * self.right + r
    }

    /// Read element `(l, p, r)`.
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> Complex {
        self.data[self.idx(l, p, r)]
    }

    /// Mutable reference to element `(l, p, r)`.
    #[inline]
    pub fn get_mut(&mut self, l: usize, p: usize, r: usize) -> &mut Complex {
        let i = self.idx(l, p, r);
        &mut self.data[i]
    }

    /// Reshape to a `(left*2) × right` matrix, grouping the left bond and
    /// physical index into rows — the form needed to move the orthogonality
    /// center rightward via QR/SVD.
    pub fn to_group_left(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left * 2, self.right, |row, r| {
            let l = row / 2;
            let p = row % 2;
            self.get(l, p, r)
        })
    }

    /// Reconstruct a `Site` from a `(left*2) × right` matrix produced by
    /// [`to_group_left`].
    pub fn from_group_left(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (row → (l, p)) split.
        #[allow(clippy::needless_range_loop)]
        for row in 0..left * 2 {
            for r in 0..right {
                let l = row / 2;
                let p = row % 2;
                *s.get_mut(l, p, r) = m[(row, r)];
            }
        }
        s
    }

    /// Reshape to a `left × (2*right)` matrix, grouping the physical index and
    /// right bond into columns — the form needed to move the orthogonality
    /// center leftward via QR/SVD.
    pub fn to_group_right(&self) -> DMatrix<Complex> {
        DMatrix::from_fn(self.left, 2 * self.right, |l, col| {
            let p = col / self.right;
            let r = col % self.right;
            self.get(l, p, r)
        })
    }

    /// Reconstruct a `Site` from a `left × (2*right)` matrix produced by
    /// [`to_group_right`].
    pub fn from_group_right(m: &DMatrix<Complex>, left: usize, right: usize) -> Site {
        let mut s = Site::zeros(left, right);
        // Allow explicit index arithmetic — clearer than iterator gymnastics
        // for the (col → (p, r)) split.
        #[allow(clippy::needless_range_loop)]
        for l in 0..left {
            for col in 0..2 * right {
                let p = col / right;
                let r = col % right;
                *s.get_mut(l, p, r) = m[(l, col)];
            }
        }
        s
    }
}

/// Thin QR decomposition: returns `(Q, R)` with `Q` of shape `rows × k` and
/// `R` of shape `k × cols`, where `k = min(rows, cols)`.
///
/// Used to move the orthogonality center one site at a time without introducing
/// truncation (no singular-value threshold applied here; that comes in SVD-based
/// truncation in Task 5+).
pub fn thin_qr(m: &DMatrix<Complex>) -> (DMatrix<Complex>, DMatrix<Complex>) {
    let qr = m.clone().qr();
    let q_full = qr.q(); // rows × rows unitary
    let r_full = qr.r(); // rows × cols upper-triangular
    let k = m.nrows().min(m.ncols());
    let q = q_full.columns(0, k).into_owned(); // rows × k
    let r = r_full.rows(0, k).into_owned(); // k × cols
    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Complex;

    fn c(re: f64) -> Complex {
        Complex::new(re, 0.0)
    }

    #[test]
    fn ket0_site_shape() {
        let s = Site::ket0();
        assert_eq!((s.left, s.right), (1, 1));
        assert_eq!(s.get(0, 0, 0), c(1.0));
        assert_eq!(s.get(0, 1, 0), c(0.0));
    }

    #[test]
    fn group_left_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    *s.get_mut(l, p, r) = c((l * 100 + p * 10 + r) as f64);
                }
            }
        }
        let m = s.to_group_left(); // (left*2) rows × right cols
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);
        let back = Site::from_group_left(&m, 2, 3);
        assert_eq!(back, s);
    }

    #[test]
    fn group_right_roundtrip() {
        let mut s = Site::zeros(2, 3);
        for l in 0..2 {
            for p in 0..2 {
                for r in 0..3 {
                    *s.get_mut(l, p, r) = c((l * 100 + p * 10 + r) as f64);
                }
            }
        }
        let m = s.to_group_right(); // left rows × (2*right) cols
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 6);
        let back = Site::from_group_right(&m, 2, 3);
        assert_eq!(back, s);
    }
}
