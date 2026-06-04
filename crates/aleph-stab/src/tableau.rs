//! Aaronson–Gottesman (CHP) stabilizer tableau.
//!
//! Rows `0..n` are destabilizer generators, `n..2n` are stabilizer
//! generators, and row `2n` is a scratch row reserved for measurement
//! (P3-02). Each row carries `n` x-bits, `n` z-bits, and a sign bit
//! (`true` = leading `-`). Gates update all `2n` non-scratch rows in
//! O(1) each → O(n) per gate. See AG (2004) §2.

use crate::bits::BitGrid;

/// A stabilizer state over `n` qubits in CHP tableau form.
#[derive(Clone)]
pub struct Tableau {
    n: usize,
    /// x-bits: `2n+1` rows × `n` cols.
    x: BitGrid,
    /// z-bits: `2n+1` rows × `n` cols.
    z: BitGrid,
    /// sign bit per row (`true` = `-`); length `2n+1`.
    sign: Vec<bool>,
}

#[allow(dead_code)] // fields and accessor methods used in later tasks
impl Tableau {
    /// Allocate the `|0…0⟩` stabilizer state on `n` qubits.
    ///
    /// Destabilizer `i` = `X_i`, stabilizer `i` = `Z_i`, all signs `+`.
    pub fn new(n: usize) -> Self {
        let rows = 2 * n + 1;
        let mut x = BitGrid::zeros(rows, n.max(1));
        let mut z = BitGrid::zeros(rows, n.max(1));
        for i in 0..n {
            x.set(i, i, true); // destabilizer i = X_i
            z.set(n + i, i, true); // stabilizer i = Z_i
        }
        Tableau {
            n,
            x,
            z,
            sign: vec![false; rows],
        }
    }

    /// Number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    // --- read accessors (used by tests + readout) ---
    #[inline]
    pub(crate) fn x(&self, row: usize, col: usize) -> bool {
        self.x.get(row, col)
    }
    #[inline]
    pub(crate) fn z(&self, row: usize, col: usize) -> bool {
        self.z.get(row, col)
    }
    #[inline]
    pub(crate) fn sign(&self, row: usize) -> bool {
        self.sign[row]
    }
}

#[cfg(test)]
mod tests {
    use super::Tableau;

    #[test]
    fn identity_tableau_is_zero_state() {
        let t = Tableau::new(3);
        assert_eq!(t.num_qubits(), 3);
        // destabilizer i = X_i  -> x[i][i]=1, all z=0, sign=+
        // stabilizer  i = Z_i  -> z[n+i][i]=1, all x=0, sign=+
        for i in 0..3 {
            assert!(t.x(i, i), "destab {i} should have X on qubit {i}");
            assert!(t.z(3 + i, i), "stab {i} should have Z on qubit {i}");
            assert!(!t.sign(i) && !t.sign(3 + i), "all signs +");
            for j in 0..3 {
                if j != i {
                    assert!(!t.x(i, j));
                    assert!(!t.z(3 + i, j));
                }
            }
        }
    }
}
