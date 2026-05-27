//! Indexed gate application kernels.
//!
//! Two layouts share the same MSB qubit-ordering convention (ADR 0004
//! / P0-06 spec §6): `qubits[0]` is the MSB of the matrix index. They
//! diverge only in storage:
//!
//! * `aos` — `Vec<Complex<f64>>` (the naive `Vec<num_complex::Complex>`
//!   layout used by `NaiveSvBackend`).
//! * `soa` — paired `Vec<f64>` (real, imaginary) used by `SoaSvBackend`
//!   (P1-01). Same algorithms, layout chosen for SIMD-friendly
//!   sequential reads — explicit vectorisation lands in P1-03 / P1-04.

pub(crate) mod aos;
pub(crate) mod soa;

/// Bitwise-OR of `1 << q` over `controls`. Layout-agnostic — used by
/// both AoS and SoA kernels to compute the control gate-mask.
///
/// Returns `usize` so the result composes directly with index
/// arithmetic in the kernel loops; `q` is bounded by `state.num_qubits`
/// at the apply_gate boundary, which itself is capped at `MAX_*_QUBITS
/// ≤ 28`, so `1 << q` never overflows on any supported platform.
pub(crate) fn control_mask(controls: &[u32]) -> usize {
    let mut mask: usize = 0;
    for &c in controls {
        mask |= 1usize << c;
    }
    mask
}

/// Expand a "free-bit counter" `k` into a full bit index by
/// interleaving `k`'s bits into the **free** positions, with the
/// **fixed** bit positions set to their prescribed value. `fixed`
/// MUST be sorted by ascending position (caller's responsibility —
/// the SIMD kernels hoist this sort once outside their outer loops).
///
/// Used by the controlled AVX-512 kernel (P1-03,
/// `aos::apply_1q_avx512`): the outer loop counts `k` over
/// `2^(n_qubits − target − 1 − controls.len())` free-bit values; for
/// each `k`, `expand_with_fixed(k, &sorted_controls_renormalised)`
/// is the base index of the next outer block where every control is
/// set and the target + below-target bits are clear (the inner SIMD
/// walk fills those).
///
/// Bit positions in `fixed.0` are `u32` to match `Gate` qubit
/// indices; the caller guarantees they are < 64 (in practice < 28
/// since `MAX_*_QUBITS ≤ 28`), so the `1usize << pos` shifts never
/// overflow.
// Allow dead_code: the only caller (avx512 path in aos.rs) is
// `#[cfg(target_arch = "x86_64")]`, so on ARM / WASM / RISC-V the
// helper is unreferenced. Unit tests below run on all targets.
#[allow(dead_code)]
pub(crate) fn expand_with_fixed(k: usize, fixed: &[(u32, bool)]) -> usize {
    let mut result: usize = 0;
    let mut k_bit: u32 = 0;
    let mut fixed_iter = fixed.iter().peekable();
    let mut pos: u32 = 0;
    let k_bits_needed = usize::BITS - k.leading_zeros();
    while k_bit < k_bits_needed || fixed_iter.peek().is_some() {
        match fixed_iter.peek() {
            Some(&&(fpos, fval)) if fpos == pos => {
                if fval {
                    result |= 1usize << pos;
                }
                fixed_iter.next();
            }
            _ => {
                if (k >> k_bit) & 1 == 1 {
                    result |= 1usize << pos;
                }
                k_bit += 1;
            }
        }
        pos += 1;
    }
    result
}

/// Tolerance (squared magnitude) for the diagonal-2x2 detection
/// heuristic.  `EPS_SQ = 1e-30` ⇒ `|m_off| < ~3.16e-16`, just above
/// FP64 machine epsilon (~2.22e-16), so an off-diagonal entry the
/// caller produced as a "true" zero (e.g. `Phase::matrix()` literal
/// `0.0`) detects as diagonal while any caller-supplied off-diagonal
/// of magnitude ≥ machine eps falls through.
const DIAGONAL_EPS_SQ: f64 = 1e-30;

/// Returns true iff both off-diagonal entries of a 2×2 matrix have
/// squared magnitude below `DIAGONAL_EPS_SQ`.
///
/// Used as the dispatch heuristic for the 1q diagonal fast path
/// (P1-06). Cost is dominated by 2 complex `norm_sqr` calls plus the
/// NaN-reject; invoked once per gate, not per amplitude, so the
/// overhead is amortised against the inner kernel.
///
/// ADR 0006: explicit `is_finite` reject precedes the magnitude test.
/// A NaN-poisoned off-diagonal compares `false` for every `<`, which
/// would silently classify the matrix as diagonal and route the NaN
/// to the fast path (which only consults `m[i][i]`). Rejecting
/// non-finite off-diagonals forces the generic kernel to see and
/// propagate the NaN.
#[inline]
pub(crate) fn is_diagonal_2x2(m: &[[aleph_core::Complex; 2]; 2]) -> bool {
    let off = [&m[0][1], &m[1][0]];
    for entry in off {
        if !entry.re.is_finite() || !entry.im.is_finite() {
            return false;
        }
        if entry.norm_sqr() >= DIAGONAL_EPS_SQ {
            return false;
        }
    }
    true
}

/// Tolerance for permutation-matrix detection in `classify_2q_permutation`.
/// `PERM_TOL = 1e-14` requires `(|m[r][c]|² - 1).abs() < 1e-14` AND
/// `(re - 1).abs() < 1e-14` AND `im.abs() < 1e-14`. Any "almost-permutation"
/// whose off-diagonals exceed `~1e-15` magnitude already fails the
/// diagonal pre-test (`DIAGONAL_EPS_SQ`), so this looser tolerance only
/// guards against unitarity-normalisation drift in user-built matrices.
// Allow dead_code: this constant is wired into the 2q kernel dispatch
// by subsequent P1-07 tasks; Task 1 only ships the helpers + tests.
#[allow(dead_code)]
const PERM_TOL: f64 = 1e-14;

/// Canonical 4×4 permutation matrices recognised by the 2q dispatch.
/// Other 6 valid 4-element permutations (e.g. `X⊗I = [1,0,3,2]`) fall
/// through to the generic kernel.
// Allow dead_code: variants are pattern-matched by the 2q kernel
// dispatch landing in subsequent P1-07 tasks; Task 1 only ships the
// classifier + tests.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Perm2qKind {
    /// `π = [0, 1, 2, 3]` — identity.
    Identity,
    /// `π = [0, 1, 3, 2]` — control = `targets[0]` (MSB), as `Gate::Cnot`.
    CnotHi,
    /// `π = [0, 3, 2, 1]` — control = `targets[1]` (LSB).
    CnotLo,
    /// `π = [0, 2, 1, 3]` — symmetric swap, as `Gate::Swap`.
    Swap,
}

/// Returns true iff every off-diagonal entry of a 4×4 matrix has
/// squared magnitude below `DIAGONAL_EPS_SQ`. Used by the 2q diagonal
/// fast path (P1-07). Invoked once per gate (not per amplitude), so
/// the 12 `norm_sqr` + 12 NaN checks + 12 compares are amortised
/// against the inner kernel. Reuses the same `DIAGONAL_EPS_SQ`
/// tolerance as the 1q diagonal heuristic (P1-06) — semantics
/// identical.
///
/// ADR 0006: explicit `is_finite` reject precedes the magnitude
/// comparison. A NaN-poisoned off-diagonal compares `false` for every
/// `<`, which would silently classify the matrix as diagonal and
/// route the NaN to the fast path (which only consults `m[i][i]`).
/// Rejecting non-finite off-diagonals forces the generic kernel to
/// see and propagate the NaN.
// Allow dead_code: wired into the 2q kernel dispatch by subsequent
// P1-07 tasks; Task 1 only ships the helper + tests.
#[allow(dead_code)]
#[inline]
pub(crate) fn is_diagonal_4x4(m: &[[aleph_core::Complex; 4]; 4]) -> bool {
    for (r, row) in m.iter().enumerate() {
        for (c, entry) in row.iter().enumerate() {
            if r == c {
                continue;
            }
            if !entry.re.is_finite() || !entry.im.is_finite() {
                return false;
            }
            if entry.norm_sqr() >= DIAGONAL_EPS_SQ {
                return false;
            }
        }
    }
    true
}

/// Classifies a 4×4 matrix as one of the four canonical 2q permutations
/// recognised by the dispatch. Returns `None` for any matrix that is
/// not a `+1`-entry permutation matrix in the canonical set.
///
/// Algorithm: for each row, find the unique column with `(re ≈ 1, im ≈ 0)`
/// within `PERM_TOL`; reject if multiple non-zero entries or any non-zero
/// off-canonical phase. Check the column-permutation is injective.
/// Match against the four canonical patterns.
// Allow dead_code: wired into the 2q kernel dispatch by subsequent
// P1-07 tasks; Task 1 only ships the classifier + tests.
#[allow(dead_code)]
pub(crate) fn classify_2q_permutation(m: &[[aleph_core::Complex; 4]; 4]) -> Option<Perm2qKind> {
    let mut perm = [0u8; 4];
    for (r, row) in m.iter().enumerate() {
        let mut hit: Option<u8> = None;
        for (c, entry) in row.iter().enumerate() {
            let nsq = entry.norm_sqr();
            // ADR 0006 / NaN-handling: a NaN `nsq` produces `false` for
            // both the "absent" (`nsq < DIAGONAL_EPS_SQ`) and "canonical"
            // (`(nsq - 1.0).abs() < PERM_TOL`) branches, so it falls
            // through to the `else { return None }` arm. The function
            // therefore naturally rejects NaN entries as "not a
            // permutation" — no explicit `is_finite` check needed.
            if nsq < DIAGONAL_EPS_SQ {
                continue;
            }
            // Require exact +1+0i within PERM_TOL.
            if (nsq - 1.0).abs() < PERM_TOL
                && (entry.re - 1.0).abs() < PERM_TOL
                && entry.im.abs() < PERM_TOL
            {
                if hit.is_some() {
                    return None; // two non-zero entries in row
                }
                hit = Some(c as u8);
            } else {
                return None; // non-canonical magnitude or phase
            }
        }
        perm[r] = hit?;
    }
    // Reject duplicate columns (not a permutation).
    let mut seen = [false; 4];
    for &c in &perm {
        if seen[c as usize] {
            return None;
        }
        seen[c as usize] = true;
    }
    match perm {
        [0, 1, 2, 3] => Some(Perm2qKind::Identity),
        [0, 1, 3, 2] => Some(Perm2qKind::CnotHi),
        [0, 3, 2, 1] => Some(Perm2qKind::CnotLo),
        [0, 2, 1, 3] => Some(Perm2qKind::Swap),
        _ => None,
    }
}

/// Returns true iff the four diagonal entries match the CZ phase
/// pattern `(1, 1, 1, -1)` within `PERM_TOL`. Detected as a shortcut
/// to swap the generic 2q-diagonal multiply for `vxorpd` sign-flip.
// Allow dead_code: wired into the 2q kernel dispatch by subsequent
// P1-07 tasks; Task 1 only ships the helper + tests.
#[allow(dead_code)]
#[inline]
pub(crate) fn is_cz_signature(d: [aleph_core::Complex; 4]) -> bool {
    // Component-wise comparison matches the contract used by
    // `classify_2q_permutation` — both predicates agree on what counts
    // as "close to canonical". An earlier `(z - target).norm_sqr() <
    // PERM_TOL` form was effectively `|z - target| < sqrt(PERM_TOL) ≈
    // 1e-7`, seven orders looser than the documented `PERM_TOL = 1e-14`.
    let close = |z: aleph_core::Complex, target_re: f64, target_im: f64| {
        (z.re - target_re).abs() < PERM_TOL && (z.im - target_im).abs() < PERM_TOL
    };
    close(d[0], 1.0, 0.0)
        && close(d[1], 1.0, 0.0)
        && close(d[2], 1.0, 0.0)
        && close(d[3], -1.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::control_mask;

    #[test]
    fn control_mask_empty_is_zero() {
        assert_eq!(control_mask(&[]), 0);
    }

    #[test]
    fn control_mask_combines_bits() {
        // Controls on qubits 0, 2, 5 → bit positions 0, 2, 5 → 0b100101 = 37.
        assert_eq!(control_mask(&[0, 2, 5]), 0b100101);
    }

    #[test]
    fn control_mask_is_order_independent() {
        assert_eq!(control_mask(&[5, 0, 2]), control_mask(&[0, 2, 5]));
    }

    #[test]
    fn expand_with_fixed_target_only_passthroughs_k() {
        // fixed = [(target=2, false)] → bit 2 cleared, other bits from k.
        // Free positions: 0, 1, 3, 4, ...  k = 0b011 → set positions 0 and 1.
        // Expected: 0b0011.
        assert_eq!(super::expand_with_fixed(0b011, &[(2, false)]), 0b0011);
    }

    #[test]
    fn expand_with_fixed_control_set_high() {
        // fixed sorted: (1, false), (3, true). Free positions: 0, 2, 4, ...
        // k = 0b010 → free bit at position 2; plus control bit at position 3.
        // Expected: bit 2 + bit 3 = 0b1100.
        assert_eq!(
            super::expand_with_fixed(0b010, &[(1, false), (3, true)]),
            0b1100
        );
    }

    #[test]
    fn expand_with_fixed_empty_fixed_is_identity() {
        assert_eq!(super::expand_with_fixed(0xDEAD, &[]), 0xDEAD);
    }

    use super::is_diagonal_2x2;
    use aleph_core::Complex;

    fn z(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn is_diagonal_2x2_pauli_z() {
        // diag(1, -1) — both off-diagonals exactly zero
        let m = [[z(1.0, 0.0), z(0.0, 0.0)], [z(0.0, 0.0), z(-1.0, 0.0)]];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rz_random_theta() {
        // diag(e^{-iθ/2}, e^{+iθ/2}) for θ = 1.234
        let theta = 1.234_f64;
        let m = [
            [z((theta / 2.0).cos(), -(theta / 2.0).sin()), z(0.0, 0.0)],
            [z(0.0, 0.0), z((theta / 2.0).cos(), (theta / 2.0).sin())],
        ];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_hadamard() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let m = [[z(s, 0.0), z(s, 0.0)], [z(s, 0.0), z(-s, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_pauli_x() {
        let m = [[z(0.0, 0.0), z(1.0, 0.0)], [z(1.0, 0.0), z(0.0, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_accepts_subepsilon_off_diagonal() {
        // |m_off| = 1e-17, well below FP64 eps — counts as zero
        let m = [[z(1.0, 0.0), z(1e-17, 0.0)], [z(0.0, 1e-17), z(-1.0, 0.0)]];
        assert!(is_diagonal_2x2(&m));
    }

    #[test]
    fn is_diagonal_2x2_rejects_superepsilon_off_diagonal() {
        // |m_off| = 1e-8, well above FP64 eps — counts as non-zero
        let m = [[z(1.0, 0.0), z(1e-8, 0.0)], [z(0.0, 0.0), z(-1.0, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn expand_with_fixed_two_controls_around_target() {
        // fixed sorted: [(0, true), (2, false), (4, true)].
        // Free positions: 1, 3, 5, 6, 7, ...
        // k = 0b011 → free bits at positions 1 and 3.
        // Plus fixed: bit 0 set, bit 2 clear, bit 4 set.
        // Expected: 1 + 2 + 8 + 16 = 0b11011 = 27.
        assert_eq!(
            super::expand_with_fixed(0b011, &[(0, true), (2, false), (4, true)]),
            0b11011,
        );
    }

    use super::{classify_2q_permutation, is_cz_signature, is_diagonal_4x4, Perm2qKind};

    fn id_4x4() -> [[Complex; 4]; 4] {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = z(1.0, 0.0);
        }
        m
    }

    fn cnot_hi_matrix() -> [[Complex; 4]; 4] {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        m[0][0] = z(1.0, 0.0);
        m[1][1] = z(1.0, 0.0);
        m[2][3] = z(1.0, 0.0);
        m[3][2] = z(1.0, 0.0);
        m
    }

    fn cnot_lo_matrix() -> [[Complex; 4]; 4] {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        m[0][0] = z(1.0, 0.0);
        m[1][3] = z(1.0, 0.0);
        m[2][2] = z(1.0, 0.0);
        m[3][1] = z(1.0, 0.0);
        m
    }

    fn swap_matrix() -> [[Complex; 4]; 4] {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        m[0][0] = z(1.0, 0.0);
        m[1][2] = z(1.0, 0.0);
        m[2][1] = z(1.0, 0.0);
        m[3][3] = z(1.0, 0.0);
        m
    }

    fn cz_matrix() -> [[Complex; 4]; 4] {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        m[0][0] = z(1.0, 0.0);
        m[1][1] = z(1.0, 0.0);
        m[2][2] = z(1.0, 0.0);
        m[3][3] = z(-1.0, 0.0);
        m
    }

    #[test]
    fn is_diagonal_4x4_accepts_identity() {
        assert!(is_diagonal_4x4(&id_4x4()));
    }

    #[test]
    fn is_diagonal_4x4_accepts_cz() {
        assert!(is_diagonal_4x4(&cz_matrix()));
    }

    #[test]
    fn is_diagonal_4x4_rejects_cnot() {
        assert!(!is_diagonal_4x4(&cnot_hi_matrix()));
    }

    #[test]
    fn is_diagonal_4x4_rejects_swap() {
        assert!(!is_diagonal_4x4(&swap_matrix()));
    }

    #[test]
    fn is_diagonal_4x4_accepts_subepsilon_off_diagonal() {
        let mut m = cz_matrix();
        m[0][2] = z(1e-17, 0.0); // below DIAGONAL_EPS_SQ
        assert!(is_diagonal_4x4(&m));
    }

    #[test]
    fn is_diagonal_4x4_rejects_superepsilon_off_diagonal() {
        let mut m = cz_matrix();
        m[0][2] = z(1e-8, 0.0); // above DIAGONAL_EPS_SQ
        assert!(!is_diagonal_4x4(&m));
    }

    #[test]
    fn classify_perm_identity() {
        assert_eq!(
            classify_2q_permutation(&id_4x4()),
            Some(Perm2qKind::Identity)
        );
    }

    #[test]
    fn classify_perm_cnot_hi() {
        assert_eq!(
            classify_2q_permutation(&cnot_hi_matrix()),
            Some(Perm2qKind::CnotHi)
        );
    }

    #[test]
    fn classify_perm_cnot_lo() {
        assert_eq!(
            classify_2q_permutation(&cnot_lo_matrix()),
            Some(Perm2qKind::CnotLo)
        );
    }

    #[test]
    fn classify_perm_swap() {
        assert_eq!(
            classify_2q_permutation(&swap_matrix()),
            Some(Perm2qKind::Swap)
        );
    }

    #[test]
    fn classify_perm_rejects_x_kron_i() {
        // X⊗I = π[1, 0, 3, 2] — valid permutation but not in canonical set.
        let mut m = [[z(0.0, 0.0); 4]; 4];
        m[0][1] = z(1.0, 0.0);
        m[1][0] = z(1.0, 0.0);
        m[2][3] = z(1.0, 0.0);
        m[3][2] = z(1.0, 0.0);
        assert_eq!(classify_2q_permutation(&m), None);
    }

    #[test]
    fn classify_perm_rejects_cz() {
        // CZ is diagonal with a -1 entry — not a permutation in the canonical sense.
        assert_eq!(classify_2q_permutation(&cz_matrix()), None);
    }

    #[test]
    fn classify_perm_rejects_phased_cnot() {
        // CNOT with global phase e^{iπ/4} on row 2 — not a "pure" permutation.
        let mut m = cnot_hi_matrix();
        m[2][3] = z(
            (std::f64::consts::PI / 4.0).cos(),
            (std::f64::consts::PI / 4.0).sin(),
        );
        assert_eq!(classify_2q_permutation(&m), None);
    }

    #[test]
    fn classify_perm_rejects_almost_permutation_with_off_diag() {
        // Mostly CNOT but with a tiny extra off-diagonal exceeding DIAGONAL_EPS_SQ.
        let mut m = cnot_hi_matrix();
        m[0][1] = z(1e-7, 0.0);
        assert_eq!(classify_2q_permutation(&m), None);
    }

    #[test]
    fn classify_perm_rejects_hadamard_tensor_hadamard() {
        let mut m = [[z(0.0, 0.0); 4]; 4];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                let sign = if (r as u32 & c as u32).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                *entry = z(0.5 * sign, 0.0);
            }
        }
        assert_eq!(classify_2q_permutation(&m), None);
    }

    #[test]
    fn cz_signature_accepts_canonical() {
        assert!(is_cz_signature([
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(-1.0, 0.0)
        ]));
    }

    #[test]
    fn cz_signature_rejects_identity_diagonals() {
        assert!(!is_cz_signature([
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0)
        ]));
    }

    #[test]
    fn cz_signature_rejects_phase_pi_over_two() {
        // Controlled-Phase(π/2): d3 = e^{iπ/2} = i, not -1.
        assert!(!is_cz_signature([
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 1.0)
        ]));
    }

    // ---- ADR 0006 NaN-reject contract -------------------------------

    #[test]
    fn is_diagonal_4x4_rejects_nan_off_diagonal() {
        let mut m = id_4x4();
        m[0][2] = z(f64::NAN, 0.0);
        assert!(!is_diagonal_4x4(&m), "NaN off-diagonal must reject");
    }

    #[test]
    fn is_diagonal_4x4_rejects_inf_off_diagonal() {
        let mut m = id_4x4();
        m[1][3] = z(0.0, f64::INFINITY);
        assert!(!is_diagonal_4x4(&m), "Inf off-diagonal must reject");
    }

    #[test]
    fn is_diagonal_2x2_rejects_nan_off_diagonal() {
        let m = [[z(1.0, 0.0), z(f64::NAN, 0.0)], [z(0.0, 0.0), z(-1.0, 0.0)]];
        assert!(!is_diagonal_2x2(&m));
    }

    #[test]
    fn classify_perm_rejects_nan_entry() {
        let mut m = cnot_hi_matrix();
        m[2][3] = z(f64::NAN, 0.0);
        assert_eq!(classify_2q_permutation(&m), None);
    }

    // ---- is_cz_signature: tightened-tolerance boundary --------------

    #[test]
    fn cz_signature_rejects_phase_one_microradian() {
        // Phase of ~1 microradian on d[3] gives Im(d[3]) ≈ 1e-6 — well
        // above PERM_TOL = 1e-14 and clearly not "actually CZ".  Old
        // sqrt(PERM_TOL)≈1e-7 tolerance would have accepted; new
        // component-wise PERM_TOL=1e-14 rejects.
        assert!(!is_cz_signature([
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(-(1e-6_f64).cos(), -(1e-6_f64).sin())
        ]));
    }

    // ---- classify_2q_permutation: per-leg isolation -----------------

    #[test]
    fn classify_perm_rejects_cnot_with_im_perturbation_only() {
        // re-leg passes (1.0 exact), im-leg fails (1e-7 >> PERM_TOL).
        let mut m = cnot_hi_matrix();
        m[2][3] = z(1.0, 1e-7);
        assert_eq!(classify_2q_permutation(&m), None);
    }

    #[test]
    fn classify_perm_rejects_cnot_with_re_perturbation_only() {
        // im-leg passes (0.0 exact), re-leg fails (re = 1 + 1e-7).
        let mut m = cnot_hi_matrix();
        m[2][3] = z(1.0 + 1e-7, 0.0);
        assert_eq!(classify_2q_permutation(&m), None);
    }

    // ---- classify_2q_permutation: acceptance-region boundary --------

    #[test]
    fn classify_perm_accepts_cnot_with_fp_noise_within_perm_tol() {
        let mut m = cnot_hi_matrix();
        m[2][3] = z(1.0 + 1e-15, 0.0); // within PERM_TOL = 1e-14
        assert_eq!(classify_2q_permutation(&m), Some(Perm2qKind::CnotHi));
    }
}
