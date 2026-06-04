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

    #[inline]
    fn check_qubit(&self, q: usize) -> Result<(), crate::StabError> {
        if q >= self.n {
            return Err(crate::StabError::QubitOutOfRange {
                qubit: q as u32,
                num_qubits: self.n as u32,
            });
        }
        Ok(())
    }

    /// Hadamard on qubit `a`. AG §2: `r ^= x_a·z_a`; swap `x_a, z_a`.
    pub fn h(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            let xa = self.x.get(i, a);
            let za = self.z.get(i, a);
            if xa && za {
                self.sign[i] ^= true;
            }
            self.x.set(i, a, za);
            self.z.set(i, a, xa);
        }
        Ok(())
    }

    /// Phase gate S on qubit `a`. AG §2: `r ^= x_a·z_a`; `z_a ^= x_a`.
    pub fn s(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        for i in 0..2 * self.n {
            if self.x.get(i, a) && self.z.get(i, a) {
                self.sign[i] ^= true;
            }
            if self.x.get(i, a) {
                self.z.toggle(i, a);
            }
        }
        Ok(())
    }

    /// CNOT control `a`, target `b`. AG §2:
    /// `r ^= x_a·z_b·(x_b ⊕ z_a ⊕ 1)`; `x_b ^= x_a`; `z_a ^= z_b`.
    pub fn cnot(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        for i in 0..2 * self.n {
            let xa = self.x.get(i, a);
            let xb = self.x.get(i, b);
            let za = self.z.get(i, a);
            let zb = self.z.get(i, b);
            if xa && zb && (xb ^ za ^ true) {
                self.sign[i] ^= true;
            }
            if xa {
                self.x.toggle(i, b);
            }
            if zb {
                self.z.toggle(i, a);
            }
        }
        Ok(())
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

    #[test]
    fn bell_state_stabilizers() {
        // H(0); CNOT(0,1) on |00> -> stabilized by +XX and +ZZ.
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        // Stabilizer rows are 2 and 3 (n=2). Check the *group* by its
        // canonical generators is awkward here; instead check the two
        // raw stabilizer rows are {XX:+, ZZ:+} in some order.
        let stabs: Vec<(bool, [bool; 2], [bool; 2])> = (2..4)
            .map(|r| {
                (
                    t.sign(r),
                    [t.x(r, 0), t.x(r, 1)],
                    [t.z(r, 0), t.z(r, 1)],
                )
            })
            .collect();
        // XX row: x=[1,1], z=[0,0], sign=+
        assert!(stabs.contains(&(false, [true, true], [false, false])), "missing +XX: {stabs:?}");
        // ZZ row: x=[0,0], z=[1,1], sign=+
        assert!(stabs.contains(&(false, [false, false], [true, true])), "missing +ZZ: {stabs:?}");
    }

    #[test]
    fn out_of_range_qubit_rejected() {
        let mut t = Tableau::new(2);
        assert!(t.h(2).is_err());
        assert!(t.cnot(0, 2).is_err());
    }
}
