//! Bivariate-bicycle (BB) / **gross** codes (Q5-01) — the qLDPC frontier (Bravyi et al.,
//! [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)).
//!
//! A BB code is a CSS code built from two commuting polynomials over the group algebra of
//! `Z_ℓ × Z_m`. Let `x` and `y` be the cyclic shifts on the two factors (`xᵉ = I`, `yᵐ = I`,
//! `xy = yx`), each acting on `ℓm` cells. Pick `A = Σ xᵃyᵇ` and `B = Σ xᶜyᵈ` (three monomials each).
//! With `n = 2ℓm` qubits split into a left and a right block, the parity checks are
//!
//! ```text
//! H_X = [ A | B ]        H_Z = [ Bᵀ | Aᵀ ]
//! ```
//!
//! The CSS condition `H_X H_Zᵀ = A B + B A = 0 (mod 2)` holds automatically because `A` and `B`
//! commute. Every check has weight 6 (three monomials per polynomial), every qubit sits in 6 checks,
//! and crucially **each qubit's error lights 3 checks of one type** — the syndrome graph is a
//! *hypergraph*, not a matching graph, which is exactly why these codes need belief propagation
//! (Q3-02) and BP+OSD (Q5-02) rather than MWPM.
//!
//! The **gross code** `[[144, 12, 12]]` is `ℓ = 12, m = 6`, `A = x³ + y + y²`, `B = y³ + x + x²`.
//! [`BBCode::gross`] builds it; [`BBCode::n`]/[`BBCode::k`] verify `n = 144`, `k = 12` from the GF(2)
//! ranks of the checks (`d = 12` is from the paper — exact minimum distance of a `[[144,12]]` code is
//! intractable to recompute here).
//!
//! [`BBCode::code_capacity_dem`] emits a [`DetectorErrorModel`] for the standard code-capacity
//! benchmark (independent `Z` noise, the `X`-checks as detectors, the dual logical-`X` operators as
//! observables). Feed it to [`TannerGraph::new`](crate::TannerGraph) / `BpDecoder` for decoding.

use crate::dem::{DemError, DetectorErrorModel};

/// A bivariate-bicycle CSS code over `Z_ℓ × Z_m`.
#[derive(Clone, Debug)]
pub struct BBCode {
    l: usize,
    m: usize,
    /// `X`-check rows of `H_X = [A | B]`: for each of the `ℓm` checks, the qubit indices it touches
    /// (`0..ℓm` left block, `ℓm..2ℓm` right block). Weight 6.
    hx_rows: Vec<Vec<usize>>,
    /// `Z`-check rows of `H_Z = [Bᵀ | Aᵀ]`. Weight 6.
    hz_rows: Vec<Vec<usize>>,
    /// Logical `Z` operators (a basis of `ker H_X / rowspace H_Z`), as `n`-bit qubit supports.
    lz: Vec<BitVec>,
    /// Logical `X` operators, the **symplectic dual** of `lz` (`lx[i]·lz[j] = δ_ij`).
    lx: Vec<BitVec>,
}

impl BBCode {
    /// Build a BB code from `ℓ`, `m`, and the monomial exponent pairs `(a, b)` (meaning `xᵃ yᵇ`) of
    /// `A` and `B`. Computes the checks, verifies the CSS condition, and extracts a dual logical
    /// basis.
    ///
    /// # Panics
    /// If `ℓ == 0` or `m == 0`, or if the CSS condition `H_X H_Zᵀ = 0` fails (a malformed code).
    pub fn new(l: usize, m: usize, a_monos: &[(usize, usize)], b_monos: &[(usize, usize)]) -> Self {
        assert!(l > 0 && m > 0, "ℓ and m must be positive");
        let lm = l * m;
        let n = 2 * lm;

        // Cell index k = i*m + j with i in 0..ℓ, j in 0..m. A monomial xᵖyᵠ maps cell (i,j) to
        // ((i+p) mod ℓ, (j+q) mod m). Column support of M = forward map; row support = inverse map.
        let cell = |i: usize, j: usize| i * m + j;
        let coords = |k: usize| (k / m, k % m);
        let fwd = |monos: &[(usize, usize)], c: usize| -> Vec<usize> {
            let (ci, cj) = coords(c);
            monos
                .iter()
                .map(|&(p, q)| cell((ci + p) % l, (cj + q) % m))
                .collect()
        };
        let inv = |monos: &[(usize, usize)], r: usize| -> Vec<usize> {
            let (ri, rj) = coords(r);
            monos
                .iter()
                .map(|&(p, q)| cell((ri + l - p % l) % l, (rj + m - q % m) % m))
                .collect()
        };

        // H_X check c (a row of [A|B]): A-row(c) in the left block, B-row(c) in the right block.
        let hx_rows: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut row = inv(a_monos, c);
                row.extend(inv(b_monos, c).into_iter().map(|q| q + lm));
                row.sort_unstable();
                row
            })
            .collect();
        // H_Z check c (a row of [Bᵀ|Aᵀ]): B-col(c) left, A-col(c) right.
        let hz_rows: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut row = fwd(b_monos, c);
                row.extend(fwd(a_monos, c).into_iter().map(|q| q + lm));
                row.sort_unstable();
                row
            })
            .collect();

        // Bit-vector parity checks over n qubits.
        let hx: Vec<BitVec> = hx_rows.iter().map(|r| BitVec::from_iter(n, r)).collect();
        let hz: Vec<BitVec> = hz_rows.iter().map(|r| BitVec::from_iter(n, r)).collect();

        // CSS condition: every X-check commutes with every Z-check (even overlap).
        for x in &hx {
            for z in &hz {
                assert!(x.dot(z) == 0, "CSS condition H_X H_Zᵀ = 0 violated");
            }
        }

        // Logical Z = ker(H_X) mod rowspace(H_Z); logical X = ker(H_Z) mod rowspace(H_X).
        let lz = quotient_basis(&gf2_kernel(&hx, n), &hz, n);
        let lx_raw = quotient_basis(&gf2_kernel(&hz, n), &hx, n);
        let lx = symplectic_dualize(&lx_raw, &lz, n);

        Self {
            l,
            m,
            hx_rows,
            hz_rows,
            lz,
            lx,
        }
    }

    /// The `[[144, 12, 12]]` **gross** code: `ℓ = 12, m = 6, A = x³ + y + y², B = y³ + x + x²`
    /// (Bravyi et al. Table 3).
    pub fn gross() -> Self {
        Self::new(12, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
    }

    /// Number of physical qubits `n = 2ℓm`.
    pub fn n(&self) -> usize {
        2 * self.l * self.m
    }

    /// Number of checks of each type (`ℓm`).
    pub fn num_checks(&self) -> usize {
        self.l * self.m
    }

    /// Number of logical qubits `k = n − rank(H_X) − rank(H_Z)` (= the size of the dual logical
    /// basis).
    pub fn k(&self) -> usize {
        self.lz.len()
    }

    /// `(ℓ, m)`.
    pub fn params(&self) -> (usize, usize) {
        (self.l, self.m)
    }

    /// `X`-check rows (`H_X = [A|B]`), each a sorted list of qubit indices.
    pub fn hx_rows(&self) -> &[Vec<usize>] {
        &self.hx_rows
    }

    /// `Z`-check rows (`H_Z = [Bᵀ|Aᵀ]`).
    pub fn hz_rows(&self) -> &[Vec<usize>] {
        &self.hz_rows
    }

    /// Code-capacity [`DetectorErrorModel`] for independent `Z` noise at physical rate `p`: one
    /// mechanism per qubit (a `Z` error), its detectors the `X`-checks that contain the qubit
    /// (3 of them — a hyperedge), its observables the dual logical-`X` operators it anticommutes
    /// with. Detectors are the `ℓm` `X`-checks; observables are the `k` logicals. This is the DEM
    /// `BpDecoder`/BP+OSD (Q5-02) decode; the `Z`-noise direction is decoded by `X`-checks, and the
    /// `X`-noise direction is the mirror image under `A ↔ B`.
    pub fn code_capacity_dem(&self, p: f64) -> DetectorErrorModel {
        let n = self.n();
        // For each qubit q: which X-checks contain it, and which logical-X operators cover it.
        let mut check_of_qubit: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (c, row) in self.hx_rows.iter().enumerate() {
            for &q in row {
                check_of_qubit[q].push(c as u32);
            }
        }
        let errors = (0..n)
            .map(|q| {
                let dets = check_of_qubit[q].clone();
                let obs: Vec<u32> = (0..self.lx.len())
                    .filter(|&o| self.lx[o].get(q))
                    .map(|o| o as u32)
                    .collect();
                DemError::new(p, dets, obs)
            })
            .collect();
        DetectorErrorModel {
            detectors: self.num_checks(),
            observables: self.lz.len(),
            errors,
        }
    }
}

/// A fixed-width GF(2) bit vector over `Vec<u64>` words.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    fn zeros(nbits: usize) -> Self {
        Self {
            words: vec![0u64; nbits.div_ceil(64)],
        }
    }
    fn from_iter(nbits: usize, set: &[usize]) -> Self {
        let mut v = Self::zeros(nbits);
        for &b in set {
            v.set(b);
        }
        v
    }
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }
    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }
    #[inline]
    fn xor_assign(&mut self, other: &Self) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= b;
        }
    }
    /// GF(2) inner product (parity of the overlap).
    #[inline]
    fn dot(&self, other: &Self) -> u32 {
        self.words
            .iter()
            .zip(&other.words)
            .fold(0u32, |acc, (a, b)| acc ^ (a & b).count_ones())
            & 1
    }
    /// Highest set bit index, or `None` if zero.
    fn leading(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate().rev() {
            if w != 0 {
                return Some(wi * 64 + (63 - w.leading_zeros() as usize));
            }
        }
        None
    }
}

/// Basis of the null space `{v : M v = 0}` over GF(2), `M` given by its rows, `ncols` wide.
fn gf2_kernel(mat: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let mut rows: Vec<BitVec> = mat.to_vec();
    // `pivot_row_of_col[c]` = the row whose pivot column is `c`, after reduction (or MAX).
    let mut pivot_row_of_col = vec![usize::MAX; ncols];
    let mut r = 0usize;
    // `c` is a bit index into each `BitVec` row (not a slice index), so enumerate() does not apply.
    #[allow(clippy::needless_range_loop)]
    for c in 0..ncols {
        if r >= rows.len() {
            break;
        }
        if let Some(pr) = (r..rows.len()).find(|&i| rows[i].get(c)) {
            rows.swap(r, pr);
            for i in 0..rows.len() {
                if i != r && rows[i].get(c) {
                    let pivot = rows[r].clone();
                    rows[i].xor_assign(&pivot);
                }
            }
            pivot_row_of_col[c] = r;
            r += 1;
        }
    }
    // Each free column → one kernel basis vector.
    let mut ker = Vec::new();
    for f in 0..ncols {
        if pivot_row_of_col[f] != usize::MAX {
            continue;
        }
        let mut v = BitVec::zeros(ncols);
        v.set(f);
        for (c, &pr) in pivot_row_of_col.iter().enumerate() {
            if pr != usize::MAX && rows[pr].get(f) {
                v.set(c);
            }
        }
        ker.push(v);
    }
    ker
}

/// Reduce `vectors` modulo `rowspace(span)` and return a basis of the quotient (the new independent
/// directions) — used to peel logical operators out of `ker(H) / rowspace(H')`.
fn quotient_basis(vectors: &[BitVec], span: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let mut echelon = Echelon::new(ncols);
    for s in span {
        echelon.insert(s.clone());
    }
    let mut logicals = Vec::new();
    for v in vectors {
        if let Some(reduced) = echelon.reduce_nonzero(v.clone()) {
            echelon.insert(reduced.clone());
            logicals.push(reduced);
        }
    }
    logicals
}

/// GF(2) row-echelon span keyed by leading set-bit, for membership / reduction queries.
struct Echelon {
    /// `by_leading[i]` = a basis vector whose leading bit is `i` (or `None`).
    by_leading: Vec<Option<BitVec>>,
}

impl Echelon {
    fn new(ncols: usize) -> Self {
        Self {
            by_leading: vec![None; ncols],
        }
    }
    /// Reduce `v` against the span; return the residue if it is independent (nonzero), else `None`.
    fn reduce_nonzero(&self, mut v: BitVec) -> Option<BitVec> {
        while let Some(lead) = v.leading() {
            match &self.by_leading[lead] {
                Some(b) => v.xor_assign(b),
                None => return Some(v),
            }
        }
        None
    }
    /// Insert `v` (already reduced is fine; this re-reduces) into the span.
    fn insert(&mut self, mut v: BitVec) {
        while let Some(lead) = v.leading() {
            match &self.by_leading[lead] {
                Some(b) => v.xor_assign(b),
                None => {
                    self.by_leading[lead] = Some(v);
                    return;
                }
            }
        }
    }
}

/// Replace `lx_raw` by a symplectic **dual** basis of `lz`: returns `lx` with `lx[i]·lz[j] = δ_ij`.
/// Solves `G · lx_new = lx_raw` where `G[i][j] = lx_raw[i]·lz[j]` (invertible for nondegenerate
/// logicals), i.e. `lx_new = G⁻¹ lx_raw`.
fn symplectic_dualize(lx_raw: &[BitVec], lz: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let k = lz.len();
    assert_eq!(lx_raw.len(), k, "logical X/Z counts must match");
    if k == 0 {
        return Vec::new();
    }
    // G[i][j] = lx_raw[i] · lz[j].
    let mut g: Vec<BitVec> = (0..k)
        .map(|i| {
            let mut row = BitVec::zeros(k);
            for (j, lzj) in lz.iter().enumerate() {
                if lx_raw[i].dot(lzj) == 1 {
                    row.set(j);
                }
            }
            row
        })
        .collect();
    // Invert G over GF(2) via Gauss-Jordan on [G | I].
    let mut inv: Vec<BitVec> = (0..k)
        .map(|i| BitVec::from_iter(k, &[i]))
        .collect::<Vec<_>>();
    for col in 0..k {
        let pivot = (col..k)
            .find(|&r| g[r].get(col))
            .expect("logicals nondegenerate ⇒ G invertible");
        g.swap(col, pivot);
        inv.swap(col, pivot);
        for r in 0..k {
            if r != col && g[r].get(col) {
                let (gp, ip) = (g[col].clone(), inv[col].clone());
                g[r].xor_assign(&gp);
                inv[r].xor_assign(&ip);
            }
        }
    }
    // lx_new[i] = Σ_j inv[i][j] · lx_raw[j].
    (0..k)
        .map(|i| {
            let mut v = BitVec::zeros(ncols);
            for (j, lxj) in lx_raw.iter().enumerate() {
                if inv[i].get(j) {
                    v.xor_assign(lxj);
                }
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gross_code_has_expected_parameters() {
        let code = BBCode::gross();
        assert_eq!(code.params(), (12, 6));
        assert_eq!(code.n(), 144, "n = 2ℓm");
        assert_eq!(code.num_checks(), 72, "ℓm X-checks and ℓm Z-checks");
        assert_eq!(code.k(), 12, "gross code encodes 12 logical qubits");
    }

    #[test]
    fn checks_are_weight_six() {
        let code = BBCode::gross();
        for row in code.hx_rows() {
            assert_eq!(row.len(), 6, "every X-check has weight 6");
        }
        for row in code.hz_rows() {
            assert_eq!(row.len(), 6, "every Z-check has weight 6");
        }
        // Each qubit sits in exactly 3 X-checks (column weight 3 ⇒ hyperedges, not matching).
        let mut col_weight = vec![0usize; code.n()];
        for row in code.hx_rows() {
            for &q in row {
                col_weight[q] += 1;
            }
        }
        assert!(col_weight.iter().all(|&w| w == 3));
    }

    #[test]
    fn logicals_are_dual_and_nontrivial() {
        let code = BBCode::gross();
        // Dual basis: lx[i]·lz[j] = δ_ij.
        for i in 0..code.k() {
            for j in 0..code.k() {
                let expect = u32::from(i == j);
                assert_eq!(
                    code.lx[i].dot(&code.lz[j]),
                    expect,
                    "lx[{i}]·lz[{j}] must be δ"
                );
            }
        }
        // Logicals commute with the stabilizers: lz ∈ ker(H_X), lx ∈ ker(H_Z).
        let n = code.n();
        let hx: Vec<BitVec> = code
            .hx_rows()
            .iter()
            .map(|r| BitVec::from_iter(n, r))
            .collect();
        let hz: Vec<BitVec> = code
            .hz_rows()
            .iter()
            .map(|r| BitVec::from_iter(n, r))
            .collect();
        for lz in &code.lz {
            assert!(
                hx.iter().all(|c| c.dot(lz) == 0),
                "logical Z must commute with X-checks"
            );
        }
        for lx in &code.lx {
            assert!(
                hz.iter().all(|c| c.dot(lx) == 0),
                "logical X must commute with Z-checks"
            );
        }
    }

    #[test]
    fn code_capacity_dem_is_well_formed() {
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.01);
        assert_eq!(dem.detectors, 72);
        assert_eq!(dem.observables, 12);
        assert_eq!(dem.errors.len(), 144, "one Z-error mechanism per qubit");
        // Each mechanism is a 3-detector hyperedge.
        assert!(dem.errors.iter().all(|e| e.dets.len() == 3));
        // A single qubit's error must flip at least one observable for some qubit (logicals cover
        // the block) and the DEM is non-degenerate overall.
        let total_obs: usize = dem.errors.iter().map(|e| e.obs.len()).sum();
        assert!(
            total_obs > 0,
            "logical observables must be reachable by single-qubit errors"
        );
    }

    /// A smaller BB code `[[72,12,6]]` (ℓ=m=6, same polynomials) also verifies — guards the
    /// construction against hard-coding the gross parameters.
    #[test]
    fn small_bb_code_parameters() {
        let code = BBCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
        assert_eq!(code.n(), 72);
        assert_eq!(code.k(), 12);
    }
}
