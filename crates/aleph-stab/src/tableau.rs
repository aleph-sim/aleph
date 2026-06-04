//! Aaronson–Gottesman (CHP) stabilizer tableau.
//!
//! Rows `0..n` are destabilizer generators, `n..2n` are stabilizer
//! generators, and row `2n` is a scratch row reserved for measurement
//! (P3-02). Each row carries `n` x-bits, `n` z-bits, and a sign bit
//! (`true` = leading `-`). Gates update all `2n` non-scratch rows in
//! O(1) each → O(n) per gate. See AG (2004) §2.

use aleph_core::{Pauli, PauliString};

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
    // `pub(crate)` so tests and dispatch can read raw tableau bits; P3-02
    // will use these for measurement. `allow(dead_code)` because they're
    // only referenced inside `#[cfg(test)]` until P3-02.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn x(&self, row: usize, col: usize) -> bool {
        self.x.get(row, col)
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn z(&self, row: usize, col: usize) -> bool {
        self.z.get(row, col)
    }
    #[allow(dead_code)]
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
        // Hoist column word-offset and mask; both are loop-invariant.
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            let xw = self.x.word(base + wa);
            let zw = self.z.word(base + wa);
            let xa = (xw & ma) != 0;
            let za = (zw & ma) != 0;
            // Branchless sign update: sign ^= x_a & z_a.
            self.sign[i] ^= xa & za;
            // Swap x_a and z_a in-place: write the other grid's bit into
            // each. Clear the bit then OR in the swapped value (branchless:
            // `ma & (0 - bit)` is `ma` when bit=1, else 0).
            *self.x.word_mut(base + wa) = (xw & !ma) | (ma & 0u64.wrapping_sub(za as u64));
            *self.z.word_mut(base + wa) = (zw & !ma) | (ma & 0u64.wrapping_sub(xa as u64));
        }
        Ok(())
    }

    /// Phase gate S on qubit `a`. AG §2: `r ^= x_a·z_a`; `z_a ^= x_a`.
    pub fn s(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            let xa = (self.x.word(base + wa) & ma) != 0;
            let za = (self.z.word(base + wa) & ma) != 0;
            // sign ^= x_a & z_a  (branchless bool-and)
            self.sign[i] ^= xa & za;
            // z_a ^= x_a  (branchless: XOR mask when x_a is set)
            // 0u64.wrapping_sub(xa as u64) is all-ones if xa, else 0.
            *self.z.word_mut(base + wa) ^= ma & (0u64.wrapping_sub(xa as u64));
        }
        Ok(())
    }

    /// CNOT control `a`, target `b`. AG §2:
    /// `r ^= x_a·z_b·(x_b ⊕ z_a ⊕ 1)`; `x_b ^= x_a`; `z_a ^= z_b`.
    pub fn cnot(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        // Hoist both column word-offsets and masks; both are loop-invariant.
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        let wb = b >> 6;
        let mb = 1u64 << (b & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            // Read all four bits from the ORIGINAL row before any mutation.
            let xa = (self.x.word(base + wa) & ma) != 0;
            let xb = (self.x.word(base + wb) & mb) != 0;
            let za = (self.z.word(base + wa) & ma) != 0;
            let zb = (self.z.word(base + wb) & mb) != 0;
            // sign ^= x_a & z_b & (x_b ^ z_a ^ 1)  — fully branchless.
            // (x_b ^ z_a ^ true) is !(x_b ^ z_a), i.e. x_b XNOR z_a.
            self.sign[i] ^= xa & zb & !(xb ^ za);
            // x_b ^= x_a  (branchless)
            *self.x.word_mut(base + wb) ^= mb & (0u64.wrapping_sub(xa as u64));
            // z_a ^= z_b  (branchless)
            *self.z.word_mut(base + wa) ^= ma & (0u64.wrapping_sub(zb as u64));
        }
        Ok(())
    }

    /// Pauli-X on `a`. Sign rule: `r ^= z_a` (X anticommutes with Z).
    pub fn x_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        let stride = self.z.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let za = (self.z.word(i * stride + wa) & ma) != 0;
            self.sign[i] ^= za;
        }
        Ok(())
    }

    /// Pauli-Z on `a`. Sign rule: `r ^= x_a`.
    pub fn z_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let xa = (self.x.word(i * stride + wa) & ma) != 0;
            self.sign[i] ^= xa;
        }
        Ok(())
    }

    /// Pauli-Y on `a`. Sign rule: `r ^= x_a ⊕ z_a`.
    pub fn y_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            let xa = (self.x.word(base + wa) & ma) != 0;
            let za = (self.z.word(base + wa) & ma) != 0;
            self.sign[i] ^= xa ^ za;
        }
        Ok(())
    }

    /// S† on `a`. `S† = S³` (since `S⁴ = I`).
    pub fn sdg(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.s(a)?;
        self.s(a)?;
        self.s(a)
    }

    /// Controlled-Z on `(a,b)`. `CZ = H_b · CNOT_{a,b} · H_b`. Symmetric.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.h(b)?;
        self.cnot(a, b)?;
        self.h(b)
    }

    /// SWAP `(a,b)`. `SWAP = CNOT_{a,b} · CNOT_{b,a} · CNOT_{a,b}`.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.cnot(a, b)?;
        self.cnot(b, a)?;
        self.cnot(a, b)
    }

    /// iSWAP `(a,b)`: `|01⟩ ↔ i|10⟩`. Clifford.
    /// Decomposition: `S_a S_b H_a CNOT_{a,b} CNOT_{b,a} H_b`.
    /// Correctness pinned by the SV-equivalence test (P3-01 §6.1).
    pub fn iswap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.s(a)?;
        self.s(b)?;
        self.h(a)?;
        self.cnot(a, b)?;
        self.cnot(b, a)?;
        self.h(b)
    }

    /// iSWAP† `(a,b)`: reverse circuit of `iswap` with each primitive
    /// inverted (`H†=H`, `CNOT†=CNOT`, `S†=Sdg`).
    pub fn iswap_dg(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.h(b)?;
        self.cnot(b, a)?;
        self.cnot(a, b)?;
        self.h(a)?;
        self.sdg(b)?;
        self.sdg(a)
    }

    /// Read a single row as a signed Pauli string. Identity terms are
    /// omitted (sparse), matching `aleph_core::PauliString`.
    fn row_to_pauli(&self, row: usize) -> PauliString {
        let mut terms = Vec::new();
        for c in 0..self.n {
            let p = match (self.x.get(row, c), self.z.get(row, c)) {
                (false, false) => continue, // I
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
            terms.push((c as u32, p));
        }
        let coeff = if self.sign[row] { -1.0 } else { 1.0 };
        // PauliString::new sorts/validates; terms here are already unique
        // and ascending, so this cannot error.
        PauliString::new(coeff, terms).unwrap_or_else(|_| PauliString::identity(coeff))
    }

    /// The `n` stabilizer generators (rows `n..2n`).
    pub fn stabilizers(&self) -> Vec<PauliString> {
        (self.n..2 * self.n).map(|r| self.row_to_pauli(r)).collect()
    }

    /// The `n` destabilizer generators (rows `0..n`).
    pub fn destabilizers(&self) -> Vec<PauliString> {
        (0..self.n).map(|r| self.row_to_pauli(r)).collect()
    }

    /// Symplectic inner product of rows `i` and `j`:
    /// `⊕_a (x_{i,a}·z_{j,a} ⊕ z_{i,a}·x_{j,a})`. `true` ⇒ the two Pauli
    /// strings anticommute.
    pub fn rows_anticommute(&self, i: usize, j: usize) -> bool {
        let mut acc = false;
        for a in 0..self.n {
            acc ^= (self.x.get(i, a) && self.z.get(j, a)) ^ (self.z.get(i, a) && self.x.get(j, a));
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::Tableau;
    use aleph_core::{Pauli, PauliString};

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
            .map(|r| (t.sign(r), [t.x(r, 0), t.x(r, 1)], [t.z(r, 0), t.z(r, 1)]))
            .collect();
        // XX row: x=[1,1], z=[0,0], sign=+
        assert!(
            stabs.contains(&(false, [true, true], [false, false])),
            "missing +XX: {stabs:?}"
        );
        // ZZ row: x=[0,0], z=[1,1], sign=+
        assert!(
            stabs.contains(&(false, [false, false], [true, true])),
            "missing +ZZ: {stabs:?}"
        );
    }

    #[test]
    fn out_of_range_qubit_rejected() {
        let mut t = Tableau::new(2);
        assert!(t.h(2).is_err());
        assert!(t.cnot(0, 2).is_err());
    }

    // Apply gate `g` and its primitive decomposition to two fresh
    // tableaux prepared identically, and assert the full tableaux match.
    fn assert_tableaux_eq(a: &Tableau, b: &Tableau) {
        assert_eq!(a.num_qubits(), b.num_qubits());
        let n = a.num_qubits();
        for r in 0..2 * n {
            assert_eq!(a.sign(r), b.sign(r), "sign row {r}");
            for c in 0..n {
                assert_eq!(a.x(r, c), b.x(r, c), "x[{r}][{c}]");
                assert_eq!(a.z(r, c), b.z(r, c), "z[{r}][{c}]");
            }
        }
    }

    // Prepare a generic (non-|0>) 3-qubit Clifford state to exercise
    // sign rules on populated rows (P1-13 lesson: don't test on |0...0>).
    fn generic_state() -> Tableau {
        let mut t = Tableau::new(3);
        t.h(0).unwrap();
        t.s(0).unwrap();
        t.cnot(0, 1).unwrap();
        t.h(2).unwrap();
        t.cnot(2, 1).unwrap();
        t
    }

    #[test]
    fn z_equals_ss() {
        let mut direct = generic_state();
        direct.z_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.s(1).unwrap();
        decomp.s(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }

    #[test]
    fn x_equals_hssh() {
        let mut direct = generic_state();
        direct.x_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.h(1).unwrap();
        decomp.s(1).unwrap();
        decomp.s(1).unwrap();
        decomp.h(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }

    #[test]
    fn y_equals_xz_up_to_phase() {
        // Y = i·X·Z, and the i global phase is unobservable in the
        // stabilizer group, so Y and X∘Z must produce identical tableaux.
        let mut direct = generic_state();
        direct.y_gate(1).unwrap();
        let mut decomp = generic_state();
        decomp.z_gate(1).unwrap();
        decomp.x_gate(1).unwrap();
        assert_tableaux_eq(&direct, &decomp);
    }

    #[test]
    fn sdg_inverts_s() {
        // S then Sdg must restore the original tableau.
        let before = generic_state();
        let mut t = before.clone();
        t.s(1).unwrap();
        t.sdg(1).unwrap();
        assert_tableaux_eq(&t, &before);
    }

    #[test]
    fn cz_is_symmetric_and_hch() {
        let mut a = generic_state();
        a.cz(0, 2).unwrap();
        let mut b = generic_state();
        b.cz(2, 0).unwrap(); // CZ symmetric
        assert_tableaux_eq(&a, &b);
    }

    #[test]
    fn swap_twice_is_identity() {
        let before = generic_state();
        let mut t = before.clone();
        t.swap(0, 2).unwrap();
        t.swap(0, 2).unwrap();
        assert_tableaux_eq(&t, &before);
    }

    #[test]
    fn iswap_then_iswapdg_is_identity() {
        let before = generic_state();
        let mut t = before.clone();
        t.iswap(0, 2).unwrap();
        t.iswap_dg(0, 2).unwrap();
        assert_tableaux_eq(&t, &before);
    }

    #[test]
    fn bell_readout_is_xx_and_zz() {
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        let stabs = t.stabilizers();
        assert_eq!(stabs.len(), 2);
        let xx = PauliString::new(1.0, vec![(0, Pauli::X), (1, Pauli::X)]).unwrap();
        let zz = PauliString::new(1.0, vec![(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        // order-independent membership
        assert!(stabs.iter().any(|p| same_pauli(p, &xx)));
        assert!(stabs.iter().any(|p| same_pauli(p, &zz)));
    }

    fn same_pauli(a: &PauliString, b: &PauliString) -> bool {
        if (a.coefficient - b.coefficient).abs() > 1e-12 {
            return false;
        }
        // PauliString::new already sorts terms by qubit index, so direct
        // comparison is sufficient (both inputs were created via ::new).
        a.terms == b.terms
    }

    #[test]
    fn identity_state_symplectic() {
        let t = Tableau::new(4);
        let n = 4;
        for i in 0..n {
            // destab i anticommutes with stab i, commutes with others
            assert!(t.rows_anticommute(i, n + i));
            for j in 0..n {
                if j != i {
                    assert!(!t.rows_anticommute(i, n + j));
                }
                assert!(!t.rows_anticommute(n + i, n + j)); // stabs commute
            }
        }
    }
}
