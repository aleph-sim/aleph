//! Aaronson–Gottesman (CHP) stabilizer tableau.
//!
//! Rows `0..n` are destabilizer generators, `n..2n` are stabilizer
//! generators, and row `2n` is a scratch row reserved for measurement
//! (P3-02). Each row carries `n` x-bits, `n` z-bits, and a sign bit
//! (`true` = leading `-`). Gates update all `2n` non-scratch rows in
//! O(1) each → O(n) per gate. See AG (2004) §2.

use aleph_core::{Pauli, PauliString};

use crate::bits::{BitGrid, BitVec};

/// Physical layout of the `x`/`z` grids. Gates need ColMajor (a column is a
/// contiguous word-span); `rowsum`/`measure` need RowMajor (a generator row is
/// contiguous, per P3-08). `sign` is orientation-invariant (the generator-row
/// axis is preserved by the transpose).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Orientation {
    RowMajor,
    ColMajor,
}

/// A stabilizer state over `n` qubits in CHP tableau form.
#[derive(Clone)]
pub struct Tableau {
    n: usize,
    /// x-bits. Dims depend on `orientation`: RowMajor `(2n+1) × n`,
    /// ColMajor `n × (2n+1)`.
    x: BitGrid,
    /// z-bits. Dims depend on `orientation`: RowMajor `(2n+1) × n`,
    /// ColMajor `n × (2n+1)`.
    z: BitGrid,
    /// sign bit per row (`true` = `-`); length `2n+1`.
    sign: BitVec,
    /// Current physical layout of `x`/`z`.
    orientation: Orientation,
}

/// Aaronson-Gottesman §2 phase exponent: the power of `i` introduced when
/// the single-qubit Pauli `(x1,z1)` is left-multiplied onto `(x2,z2)`.
/// Returns a value in `{-1, 0, 1}`. Used by [`Tableau::rowsum_scalar`] (test-only).
#[cfg(test)]
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
            sign: BitVec::zeros(rows),
            orientation: Orientation::RowMajor,
        }
    }

    /// Number of qubits.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    // --- read accessors (used by tests + readout) ---
    // `pub(crate)` so tests and dispatch can read raw tableau bits.
    // `allow(dead_code)`: only referenced in `#[cfg(test)]` until P3-03
    // wires up a readout API in non-test code.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn x(&self, row: usize, col: usize) -> bool {
        self.get_x(row, col)
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn z(&self, row: usize, col: usize) -> bool {
        self.get_z(row, col)
    }
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn sign(&self, row: usize) -> bool {
        self.sign.get(row)
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

    /// Logical x-bit of generator `row`, qubit `col`, regardless of orientation.
    #[inline]
    fn get_x(&self, row: usize, col: usize) -> bool {
        match self.orientation {
            Orientation::RowMajor => self.x.get(row, col),
            Orientation::ColMajor => self.x.get(col, row),
        }
    }
    /// Logical z-bit of generator `row`, qubit `col`, regardless of orientation.
    #[inline]
    fn get_z(&self, row: usize, col: usize) -> bool {
        match self.orientation {
            Orientation::RowMajor => self.z.get(row, col),
            Orientation::ColMajor => self.z.get(col, row),
        }
    }

    /// Ensure RowMajor (generator rows contiguous) for `rowsum`/`measure`/readout.
    fn ensure_row_major(&mut self) {
        if self.orientation == Orientation::ColMajor {
            self.x = self.x.transpose();
            self.z = self.z.transpose();
            self.orientation = Orientation::RowMajor;
        }
    }
    /// Ensure ColMajor (qubit columns contiguous) for word-parallel gates.
    fn ensure_col_major(&mut self) {
        if self.orientation == Orientation::RowMajor {
            self.x = self.x.transpose();
            self.z = self.z.transpose();
            self.orientation = Orientation::ColMajor;
        }
    }

    /// Hadamard on qubit `a`. AG §2: `r ^= x_a·z_a`; swap `x_a, z_a`.
    ///
    /// ColMajor word-parallel: qubit column `a` is a contiguous word-span
    /// (`row_words(a)`); the kernel updates the whole column + matching sign
    /// words in one pass. `x`, `z`, `sign` are three distinct fields, so the
    /// three simultaneous borrows are allowed.
    pub fn h(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_col_major();
        let xa = self.x.row_words_mut(a);
        let za = self.z.row_words_mut(a);
        let sign = self.sign.words_mut();
        crate::gates::h_dispatch(xa, za, sign);
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::h`], kept as the
    /// equivalence oracle for the word-parallel kernel (test-only).
    #[cfg(test)]
    fn h_scalar(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
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
            let v = self.sign.get(i) ^ (xa & za);
            self.sign.set(i, v);
            // Swap x_a and z_a in-place: write the other grid's bit into
            // each. Clear the bit then OR in the swapped value (branchless:
            // `ma & (0 - bit)` is `ma` when bit=1, else 0).
            *self.x.word_mut(base + wa) = (xw & !ma) | (ma & 0u64.wrapping_sub(za as u64));
            *self.z.word_mut(base + wa) = (zw & !ma) | (ma & 0u64.wrapping_sub(xa as u64));
        }
        Ok(())
    }

    /// Phase gate S on qubit `a`. AG §2: `r ^= x_a·z_a`; `z_a ^= x_a`.
    /// ColMajor word-parallel: `x_a` shared, `z_a` mutable, `sign` mutable.
    pub fn s(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_col_major();
        let xa = self.x.row_words(a);
        let za = self.z.row_words_mut(a);
        let sign = self.sign.words_mut();
        crate::gates::s_dispatch(xa, za, sign);
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::s`] (test-only oracle).
    #[cfg(test)]
    fn s_scalar(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            let xa = (self.x.word(base + wa) & ma) != 0;
            let za = (self.z.word(base + wa) & ma) != 0;
            // sign ^= x_a & z_a  (branchless bool-and)
            let v = self.sign.get(i) ^ (xa & za);
            self.sign.set(i, v);
            // z_a ^= x_a  (branchless: XOR mask when x_a is set)
            // 0u64.wrapping_sub(xa as u64) is all-ones if xa, else 0.
            *self.z.word_mut(base + wa) ^= ma & (0u64.wrapping_sub(xa as u64));
        }
        Ok(())
    }

    /// CNOT control `a`, target `b`. AG §2:
    /// `r ^= x_a·z_b·(x_b ⊕ z_a ⊕ 1)`; `x_b ^= x_a`; `z_a ^= z_b`.
    ///
    /// ColMajor word-parallel. Needs two distinct columns from each grid via
    /// `row_pair_mut`, which requires `a != b` — so `a == b` is rejected
    /// (otherwise `row_pair_mut` would panic).
    pub fn cnot(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
        }
        self.ensure_col_major();
        // x grid: row b mutable (x_b), row a shared (x_a).
        let (xb, xa) = self.x.row_pair_mut(b, a);
        // z grid: row a mutable (z_a), row b shared (z_b).
        let (za, zb) = self.z.row_pair_mut(a, b);
        crate::gates::cnot_dispatch(xa, xb, za, zb, self.sign.words_mut());
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::cnot`] (test-only oracle).
    #[cfg(test)]
    fn cnot_scalar(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        self.ensure_row_major();
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
            let v = self.sign.get(i) ^ (xa & zb & !(xb ^ za));
            self.sign.set(i, v);
            // x_b ^= x_a  (branchless)
            *self.x.word_mut(base + wb) ^= mb & (0u64.wrapping_sub(xa as u64));
            // z_a ^= z_b  (branchless)
            *self.z.word_mut(base + wa) ^= ma & (0u64.wrapping_sub(zb as u64));
        }
        Ok(())
    }

    /// Pauli-X on `a`. Sign rule: `r ^= z_a` (X anticommutes with Z).
    /// ColMajor word-parallel: `sign ^= z`-column.
    pub fn x_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_col_major();
        crate::gates::sign_xor_words(self.z.row_words(a), self.sign.words_mut());
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::x_gate`] (test-only oracle).
    #[cfg(test)]
    fn x_gate_scalar(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
        let stride = self.z.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let za = (self.z.word(i * stride + wa) & ma) != 0;
            let v = self.sign.get(i) ^ za;
            self.sign.set(i, v);
        }
        Ok(())
    }

    /// Pauli-Z on `a`. Sign rule: `r ^= x_a`.
    /// ColMajor word-parallel: `sign ^= x`-column.
    pub fn z_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_col_major();
        crate::gates::sign_xor_words(self.x.row_words(a), self.sign.words_mut());
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::z_gate`] (test-only oracle).
    #[cfg(test)]
    fn z_gate_scalar(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let xa = (self.x.word(i * stride + wa) & ma) != 0;
            let v = self.sign.get(i) ^ xa;
            self.sign.set(i, v);
        }
        Ok(())
    }

    /// Pauli-Y on `a`. Sign rule: `r ^= x_a ⊕ z_a`.
    /// ColMajor word-parallel: `sign ^= x`-column `^` `z`-column.
    pub fn y_gate(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_col_major();
        let xa = self.x.row_words(a);
        let za = self.z.row_words(a);
        crate::gates::y_sign_words(xa, za, self.sign.words_mut());
        Ok(())
    }

    /// Pre-ColMajor row-major reference for [`Tableau::y_gate`] (test-only oracle).
    #[cfg(test)]
    fn y_gate_scalar(&mut self, a: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
        let stride = self.x.row_stride();
        let wa = a >> 6;
        let ma = 1u64 << (a & 63);
        for i in 0..2 * self.n {
            let base = i * stride;
            let xa = (self.x.word(base + wa) & ma) != 0;
            let za = (self.z.word(base + wa) & ma) != 0;
            let v = self.sign.get(i) ^ (xa ^ za);
            self.sign.set(i, v);
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
    ///
    /// Rejects `a == b` up front so the gate never mutates then fails
    /// mid-decomposition (the inner `cnot` would reject `a == b` anyway).
    pub fn cz(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
        }
        self.h(b)?;
        self.cnot(a, b)?;
        self.h(b)
    }

    /// SWAP `(a,b)`. `SWAP = CNOT_{a,b} · CNOT_{b,a} · CNOT_{a,b}`.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
        }
        self.cnot(a, b)?;
        self.cnot(b, a)?;
        self.cnot(a, b)
    }

    /// iSWAP `(a,b)`: `|01⟩ ↔ i|10⟩`. Clifford.
    /// Decomposition: `S_a S_b H_a CNOT_{a,b} CNOT_{b,a} H_b`.
    /// Correctness pinned by the SV-equivalence test (P3-01 §6.1).
    pub fn iswap(&mut self, a: usize, b: usize) -> Result<(), crate::StabError> {
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
        }
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
        self.check_qubit(a)?;
        self.check_qubit(b)?;
        if a == b {
            return Err(crate::StabError::DuplicateQubit { qubit: a as u32 });
        }
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
            let p = match (self.get_x(row, c), self.get_z(row, c)) {
                (false, false) => continue, // I
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
            terms.push((c as u32, p));
        }
        let coeff = if self.sign.get(row) { -1.0 } else { 1.0 };
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
            acc ^= (self.get_x(i, a) && self.get_z(j, a)) ^ (self.get_z(i, a) && self.get_x(j, a));
        }
        acc
    }

    /// Left-multiply stabilizer/destabilizer row `i` onto row `h`, tracking the
    /// sign. Dispatches to the word-parallel kernel.
    ///
    /// Precondition: RowMajor — `row_pair_mut` needs contiguous generator rows.
    fn rowsum(&mut self, h: usize, i: usize) {
        debug_assert!(
            self.orientation == Orientation::RowMajor,
            "rowsum needs RowMajor"
        );
        let base = 2 * self.sign.get(h) as i64 + 2 * self.sign.get(i) as i64;
        let (xh, xi) = self.x.row_pair_mut(h, i);
        let (zh, zi) = self.z.row_pair_mut(h, i);
        let phase = crate::rowsum::rowsum_dispatch(xh, xi, zh, zi);
        let m = (base + phase).rem_euclid(4);
        debug_assert!(m == 0 || m == 2, "rowsum phase {m} not in {{0, 2}}");
        self.sign.set(h, m == 2);
    }

    /// Pre-P3-08 per-bit reference, kept for the equivalence test in this file.
    #[cfg(test)]
    fn rowsum_scalar(&mut self, h: usize, i: usize) {
        debug_assert!(
            self.orientation == Orientation::RowMajor,
            "rowsum_scalar needs RowMajor"
        );
        let mut acc: i32 = 2 * self.sign.get(h) as i32 + 2 * self.sign.get(i) as i32;
        for j in 0..self.n {
            acc += g(
                self.x.get(i, j),
                self.z.get(i, j),
                self.x.get(h, j),
                self.z.get(h, j),
            );
        }
        let m = acc.rem_euclid(4);
        debug_assert!(m == 0 || m == 2, "rowsum phase {m} not in {{0, 2}}");
        self.sign.set(h, m == 2);
        for j in 0..self.n {
            let xh = self.x.get(h, j) ^ self.x.get(i, j);
            let zh = self.z.get(h, j) ^ self.z.get(i, j);
            self.x.set(h, j, xh);
            self.z.set(h, j, zh);
        }
    }

    /// Copy a full generator row (x bits, z bits, sign) from `src` to `dst`.
    ///
    /// Precondition: RowMajor (direct `(row, col)` grid access).
    fn copy_row(&mut self, dst: usize, src: usize) {
        debug_assert!(
            self.orientation == Orientation::RowMajor,
            "copy_row needs RowMajor"
        );
        for j in 0..self.n {
            self.x.set(dst, j, self.x.get(src, j));
            self.z.set(dst, j, self.z.get(src, j));
        }
        let s = self.sign.get(src);
        self.sign.set(dst, s);
    }

    /// Reset a row to the identity Pauli with `+` sign.
    ///
    /// Precondition: RowMajor (direct `(row, col)` grid access).
    fn zero_row(&mut self, r: usize) {
        debug_assert!(
            self.orientation == Orientation::RowMajor,
            "zero_row needs RowMajor"
        );
        for j in 0..self.n {
            self.x.set(r, j, false);
            self.z.set(r, j, false);
        }
        self.sign.set(r, false);
    }

    /// Projective Z-basis measurement of qubit `a` with state collapse
    /// (Aaronson-Gottesman §3). Returns the outcome bit (`true` = `|1>`).
    ///
    /// If `Z_a` anticommutes with some stabilizer the outcome is random
    /// (drawn from `rng`) and the tableau collapses accordingly; otherwise
    /// the outcome is determined by the current state. `rng` is consumed
    /// only in the random case.
    pub fn measure<R: rand::Rng>(
        &mut self,
        a: usize,
        rng: &mut R,
    ) -> Result<bool, crate::StabError> {
        self.check_qubit(a)?;
        self.ensure_row_major();
        // A stabilizer row anticommuting with Z_a (i.e. with an X/Y on a)
        // ⇒ random outcome.
        let p = (self.n..2 * self.n).find(|&row| self.x.get(row, a));
        match p {
            Some(p) => {
                // Random outcome: eliminate column `a`'s X from every other
                // row, promote p to a destabilizer, install Z_a as the new
                // stabilizer with a random sign.
                //
                // Skip row `p - n` (the destabilizer paired with stab_p):
                // it anticommutes with stab_p, so their product has phase ±i
                // (unrepresentable as ±Pauli in the sign bit), and
                // `copy_row(p - n, p)` below overwrites it correctly anyway.
                let paired_destab = p - self.n;
                for i in 0..2 * self.n {
                    if i != p && i != paired_destab && self.x.get(i, a) {
                        self.rowsum(i, p);
                    }
                }
                self.copy_row(p - self.n, p);
                self.zero_row(p);
                self.z.set(p, a, true);
                let outcome = rng.gen::<bool>();
                self.sign.set(p, outcome);
                Ok(outcome)
            }
            None => {
                // Deterministic outcome: accumulate the relevant stabilizers
                // into the scratch row; its resulting sign is the outcome.
                let scratch = 2 * self.n;
                self.zero_row(scratch);
                for i in 0..self.n {
                    if self.x.get(i, a) {
                        self.rowsum(scratch, i + self.n);
                    }
                }
                Ok(self.sign.get(scratch))
            }
        }
    }
    /// `⟨ψ|P|ψ⟩` for the unsigned Pauli `P` given by `(x_p, z_p)` per
    /// qubit (`x` bit = X-component, `z` bit = Z-component; both set = Y).
    /// Returns `+1`/`-1` if `P` (up to sign) is in the stabilizer group,
    /// `0` if `P` anticommutes with some stabilizer generator.
    ///
    /// `x_p` and `z_p` must each have length `self.num_qubits()`.
    pub(crate) fn pauli_eigenvalue(&self, x_p: &[bool], z_p: &[bool]) -> i8 {
        debug_assert_eq!(x_p.len(), self.n);
        debug_assert_eq!(z_p.len(), self.n);
        // Symplectic product of P with row `r`: odd ⇒ anticommute.
        let anti_with = |t: &Tableau, r: usize| -> bool {
            let mut acc = false;
            for j in 0..t.n {
                acc ^= (x_p[j] & t.get_z(r, j)) ^ (z_p[j] & t.get_x(r, j));
            }
            acc
        };
        // 1. Anticommutes with any stabilizer generator ⇒ expectation 0.
        for k in self.n..2 * self.n {
            if anti_with(self, k) {
                return 0;
            }
        }
        // 2. P commutes with all stabilizers ⇒ (pure stabilizer state,
        //    maximal abelian group) P ∈ ⟨generators⟩ up to sign. The
        //    stabilizers whose product equals P are those whose paired
        //    destabilizer anticommutes with P; accumulate them into a
        //    scratch row (on a clone) and read the resulting sign.
        let mut t = self.clone();
        t.ensure_row_major();
        let scratch = 2 * t.n;
        t.zero_row(scratch);
        for k in 0..t.n {
            if anti_with(&t, k) {
                t.rowsum(scratch, k + t.n);
            }
        }
        debug_assert!(
            (0..t.n).all(|j| t.get_x(scratch, j) == x_p[j] && t.get_z(scratch, j) == z_p[j]),
            "accumulated stabilizer product does not equal P"
        );
        if t.sign.get(scratch) {
            -1
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tableau;
    use aleph_core::{Pauli, PauliString};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
    fn g_phase_exponent_table() {
        use super::g;
        // (x1,z1)=(0,0) → always 0
        for &(x2, z2) in &[(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(g(false, false, x2, z2), 0);
        }
        // (x1,z1)=(1,0): z2*(2*x2-1)
        assert_eq!(g(true, false, false, false), 0); // z2=0
        assert_eq!(g(true, false, true, false), 0); // z2=0
        assert_eq!(g(true, false, false, true), -1); // z2=1,x2=0 → 1*(−1)
        assert_eq!(g(true, false, true, true), 1); // z2=1,x2=1 → 1*(1)
                                                   // (x1,z1)=(0,1): x2*(1-2*z2)
        assert_eq!(g(false, true, false, false), 0); // x2=0
        assert_eq!(g(false, true, true, false), 1); // x2=1,z2=0 → 1*(1)
        assert_eq!(g(false, true, true, true), -1); // x2=1,z2=1 → 1*(−1)
                                                    // (x1,z1)=(1,1): z2 - x2
        assert_eq!(g(true, true, false, false), 0);
        assert_eq!(g(true, true, true, false), -1); // 0-1
        assert_eq!(g(true, true, false, true), 1); // 1-0
        assert_eq!(g(true, true, true, true), 0); // 1-1
    }

    // rowsum(h,i) does x_h ^= x_i, z_h ^= z_i (XOR involution): applying
    // it twice restores row h's bits. Sign tracking is exercised more
    // thoroughly by the measurement + Stim oracle; here we pin the bit
    // involution and that the sign stays in {false,true} (no panic on the
    // mod-4 debug_assert) over a generic state.
    //
    // Rows must commute for rowsum to produce a real phase (m ∈ {0,2}):
    // the CHP invariant guarantees destab_i ⊥ stab_i but commutes with
    // stab_j (j ≠ i). We use row 0 (destab 0) and row 4 (stab 1), which
    // commute.
    #[test]
    fn rowsum_bit_involution() {
        let mut t = generic_state(); // 3-qubit entangled Clifford state
                                     // Gates now leave the tableau ColMajor (P3-11); `rowsum` is a
                                     // RowMajor-only internal helper (its real callers — `measure` — flip
                                     // first), so flip here before poking it directly.
        t.ensure_row_major();
        // snapshot row 0 (destab 0) bits; we rowsum with row 4 (stab 1)
        // which commutes with destab 0 (j ≠ i in the CHP invariant).
        let snap = |t: &Tableau, r: usize| -> Vec<(bool, bool)> {
            (0..t.num_qubits())
                .map(|j| (t.x(r, j), t.z(r, j)))
                .collect()
        };
        let before = snap(&t, 0);
        t.rowsum(0, 4);
        t.rowsum(0, 4); // second application cancels the bit XORs
        assert_eq!(snap(&t, 0), before, "rowsum bit XOR is not involutive");
    }

    #[test]
    fn rowsum_matches_scalar_reference() {
        // Drive both implementations from identical entangled states and assert
        // the full tableau (x, z, sign) agrees after the same rowsum.
        // rowsum is only called on commuting row pairs (CHP invariant), so we
        // skip anticommuting pairs to stay in the valid domain.
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: usize) -> usize {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                (x as usize) % n
            }
        }
        // n=1 has only two rows (destab_0=X, stab_0=Z) which always anticommute,
        // so there are no valid rowsum pairs to test. Start from n=2.
        for n in [2usize, 3, 7, 8, 9, 65] {
            let mut rng = Rng(0xD1B54A32D192ED03 ^ n as u64);
            // Build a shared random Clifford state.
            let mut base = Tableau::new(n);
            for _ in 0..(6 * n + 10) {
                match rng.below(3) {
                    0 => {
                        let _ = base.h(rng.below(n));
                    }
                    1 => {
                        let _ = base.s(rng.below(n));
                    }
                    _ => {
                        let a = rng.below(n);
                        let b = (a + 1 + rng.below(n.max(2) - 1)) % n.max(1);
                        if n > 1 && a != b {
                            let _ = base.cnot(a, b);
                        }
                    }
                }
            }
            // Gates now leave `base` ColMajor (P3-11); `rowsum`/`rowsum_scalar`
            // are RowMajor-only, so flip before cloning + poking them directly.
            base.ensure_row_major();
            let rows = 2 * n + 1;
            let mut tested = 0usize;
            for attempt in 0..2000 {
                let h = rng.below(rows);
                let mut i = rng.below(rows);
                if i == h {
                    i = (i + 1) % rows;
                }
                // rowsum is only valid on commuting row pairs (phase ∈ {0, 2}).
                // Skip anticommuting pairs (they arise in the CHP algorithm
                // only transiently; callers always guarantee commutativity).
                if base.rows_anticommute(h, i) {
                    let _ = attempt; // silence unused warning
                    continue;
                }
                let mut a = base.clone();
                let mut b = base.clone();
                a.rowsum(h, i);
                b.rowsum_scalar(h, i);
                for r in 0..rows {
                    for c in 0..n {
                        assert_eq!(a.x(r, c), b.x(r, c), "x[{r},{c}] n={n}");
                        assert_eq!(a.z(r, c), b.z(r, c), "z[{r},{c}] n={n}");
                    }
                    assert_eq!(a.sign(r), b.sign(r), "sign[{r}] n={n}");
                }
                tested += 1;
                if tested >= 200 {
                    break;
                }
            }
            assert!(
                tested >= 10,
                "too few commuting pairs found for n={n}: {tested}"
            );
        }
    }

    // copy_row duplicates a full row; zero_row clears it.
    #[test]
    fn copy_and_zero_row() {
        let mut t = generic_state();
        // `copy_row`/`zero_row` are RowMajor-only internal helpers; gates now
        // leave the tableau ColMajor (P3-11), so flip before poking them.
        t.ensure_row_major();
        t.copy_row(0, 4); // row 0 (destab) ← row 4 (stab 1)
        for j in 0..t.num_qubits() {
            assert_eq!(t.x(0, j), t.x(4, j));
            assert_eq!(t.z(0, j), t.z(4, j));
        }
        assert_eq!(t.sign(0), t.sign(4));
        t.zero_row(0);
        for j in 0..t.num_qubits() {
            assert!(!t.x(0, j) && !t.z(0, j));
        }
        assert!(!t.sign(0));
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

    #[test]
    fn measure_zero_state_is_deterministic_zero() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut t = Tableau::new(3);
        for a in 0..3 {
            assert!(!t.measure(a, &mut rng).unwrap(), "|0> qubit {a}");
        }
        // Out-of-range rejected.
        assert!(t.measure(3, &mut rng).is_err());
    }

    #[test]
    fn measure_bell_forces_correlation() {
        // |Φ+> = (|00>+|11>)/√2: measuring q0 is random; q1 must equal q0.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..32 {
            let mut t = Tableau::new(2);
            t.h(0).unwrap();
            t.cnot(0, 1).unwrap();
            let b0 = t.measure(0, &mut rng).unwrap();
            let b1 = t.measure(1, &mut rng).unwrap();
            assert_eq!(b0, b1, "Bell correlation broken");
            // Re-measuring q0 after collapse returns the same value.
            let b0_again = t.measure(0, &mut rng).unwrap();
            assert_eq!(b0, b0_again, "post-collapse determinism broken");
        }
    }

    #[test]
    fn measure_plus_state_is_random() {
        // H|0> = |+>: measuring in Z is random; over many seeds we should
        // see both outcomes.
        let mut saw_false = false;
        let mut saw_true = false;
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(1);
            t.h(0).unwrap();
            match t.measure(0, &mut rng).unwrap() {
                false => saw_false = true,
                true => saw_true = true,
            }
        }
        assert!(
            saw_false && saw_true,
            "|+> measurement never produced both outcomes"
        );
    }

    #[test]
    fn pauli_eigenvalue_bell() {
        // Bell |Φ+>: stabilized by +XX and +ZZ; anticommutes with Z⊗I.
        let mut t = Tableau::new(2);
        t.h(0).unwrap();
        t.cnot(0, 1).unwrap();
        assert_eq!(t.pauli_eigenvalue(&[true, true], &[false, false]), 1); // XX
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, true]), 1); // ZZ
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, false]), 0); // ZI anticommutes
                                                                            // Prepare |Φ-> = Z_0 |Φ+>: stabilized by -XX and +ZZ.
                                                                            // ZZ eigenvalue is unchanged (+1); XX flips to -1.
        t.z_gate(0).unwrap();
        assert_eq!(t.pauli_eigenvalue(&[false, false], &[true, true]), 1); // ZZ still +1
        assert_eq!(t.pauli_eigenvalue(&[true, true], &[false, false]), -1); // XX now -1
    }

    #[test]
    fn pauli_eigenvalue_zero_state() {
        let t = Tableau::new(1);
        assert_eq!(t.pauli_eigenvalue(&[false], &[true]), 1); // Z on |0> = +1
        assert_eq!(t.pauli_eigenvalue(&[true], &[false]), 0); // X on |0> = 0
    }

    #[test]
    fn orientation_flip_preserves_logical_bits() {
        // Build a generic state (row-major), force a col-major flip and back,
        // and confirm every logical (x,z,sign) bit is unchanged.
        let t = generic_state();
        let snap: Vec<(bool, bool)> = (0..2 * t.num_qubits())
            .flat_map(|r| (0..t.num_qubits()).map(move |c| (r, c)))
            .map(|(r, c)| (t.x(r, c), t.z(r, c)))
            .collect();
        let signs: Vec<bool> = (0..2 * t.num_qubits() + 1).map(|r| t.sign(r)).collect();

        let mut t2 = t.clone();
        t2.ensure_col_major();
        // reads are orientation-agnostic: identical logical bits in col-major
        let snap2: Vec<(bool, bool)> = (0..2 * t2.num_qubits())
            .flat_map(|r| (0..t2.num_qubits()).map(move |c| (r, c)))
            .map(|(r, c)| (t2.x(r, c), t2.z(r, c)))
            .collect();
        assert_eq!(snap, snap2, "col-major reads diverged");
        t2.ensure_row_major();
        for (r, &want) in signs.iter().enumerate() {
            assert_eq!(t2.sign(r), want, "sign row {r} after flip-back");
        }
        for (i, &(r, c)) in (0..2 * t2.num_qubits())
            .flat_map(|r| (0..t2.num_qubits()).map(move |c| (r, c)))
            .collect::<Vec<_>>()
            .iter()
            .enumerate()
        {
            assert_eq!(
                (t2.x(r, c), t2.z(r, c)),
                snap[i],
                "bit ({r},{c}) after flip-back"
            );
        }
    }

    #[test]
    fn measure_ghz_is_balanced_and_consistent() {
        // GHZ_n = (|0…0> + |1…1>)/√2. Each trial: all qubits agree; over
        // many trials q0 is ~50/50.
        const N: usize = 5;
        const TRIALS: u32 = 4000;
        let mut ones = 0u32;
        for seed in 0..TRIALS as u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(N);
            t.h(0).unwrap();
            for i in 0..N - 1 {
                t.cnot(i, i + 1).unwrap();
            }
            let b0 = t.measure(0, &mut rng).unwrap();
            for q in 1..N {
                assert_eq!(
                    t.measure(q, &mut rng).unwrap(),
                    b0,
                    "GHZ qubit {q} disagreed"
                );
            }
            if b0 {
                ones += 1;
            }
        }
        // Binomial(4000, 0.5): mean 2000, sd ≈ 31.6. ±5 sd ≈ ±158 → [1842,2158].
        assert!(
            (1842..=2158).contains(&ones),
            "GHZ q0 balance out of range: {ones}/4000 ones"
        );
    }

    #[test]
    fn colmajor_gates_match_scalar_reference() {
        // Drive identical random Clifford circuits through the public ColMajor
        // kernels and the preserved row-major *_scalar references; assert the
        // full logical tableau (x, z, sign) agrees. a != b enforced for 2q gates.
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: usize) -> usize {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                (x as usize) % n
            }
        }
        for n in [1usize, 2, 3, 8, 9, 64, 65, 130] {
            let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (n as u64).wrapping_mul(0x9E37));
            let mut a = Tableau::new(n);
            let mut b = Tableau::new(n);
            for _ in 0..(20 * n + 50) {
                let pick = rng.below(7);
                let q = rng.below(n);
                match pick {
                    0 => {
                        a.h(q).unwrap();
                        b.h_scalar(q).unwrap();
                    }
                    1 => {
                        a.s(q).unwrap();
                        b.s_scalar(q).unwrap();
                    }
                    2 => {
                        a.x_gate(q).unwrap();
                        b.x_gate_scalar(q).unwrap();
                    }
                    3 => {
                        a.y_gate(q).unwrap();
                        b.y_gate_scalar(q).unwrap();
                    }
                    4 => {
                        a.z_gate(q).unwrap();
                        b.z_gate_scalar(q).unwrap();
                    }
                    _ => {
                        if n >= 2 {
                            let mut q2 = rng.below(n);
                            if q2 == q {
                                q2 = (q2 + 1) % n;
                            }
                            a.cnot(q, q2).unwrap();
                            b.cnot_scalar(q, q2).unwrap();
                        }
                    }
                }
            }
            // Compare only the 2n generator rows, NOT the scratch row 2n: the
            // ColMajor word kernels update the whole 2n+1-bit column span (so
            // they dirty the scratch row), whereas the `*_scalar` refs loop
            // 0..2n. This divergence is harmless because every scratch-row
            // consumer (`measure`, `pauli_eigenvalue`) calls `zero_row(2n)`
            // before reading it — see the scratch-row note in `gates.rs`.
            for r in 0..2 * n {
                assert_eq!(a.sign(r), b.sign(r), "sign[{r}] n={n}");
                for c in 0..n {
                    assert_eq!(a.x(r, c), b.x(r, c), "x[{r},{c}] n={n}");
                    assert_eq!(a.z(r, c), b.z(r, c), "z[{r},{c}] n={n}");
                }
            }
        }
    }

    #[test]
    fn cnot_duplicate_qubit_rejected() {
        let mut t = Tableau::new(2);
        assert!(matches!(
            t.cnot(1, 1),
            Err(crate::StabError::DuplicateQubit { qubit: 1 })
        ));
        assert!(matches!(
            t.swap(0, 0),
            Err(crate::StabError::DuplicateQubit { qubit: 0 })
        ));
    }
}
