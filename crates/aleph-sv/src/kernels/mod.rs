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

// In normal builds `aos` and `soa` are crate-private. When the
// `internal-bench` feature is active (criterion benches) they are
// exposed publicly so the bench binary — which compiles as an external
// crate — can reach `aleph_sv::kernels::aos::apply_1q`.
#[cfg(not(feature = "internal-bench"))]
pub(crate) mod aos;
#[cfg(feature = "internal-bench")]
pub mod aos;

#[cfg(not(feature = "internal-bench"))]
pub(crate) mod soa;
#[cfg(feature = "internal-bench")]
pub mod soa;

#[cfg(not(feature = "internal-bench"))]
pub(crate) mod tuning;
#[cfg(feature = "internal-bench")]
pub mod tuning;

use crate::kernels::tuning::ChunkPolicy;

/// Raw write pointer shareable across rayon worker threads.
///
/// The kernels drive their outer walk over pairwise-disjoint amplitude
/// blocks (the SIMD-kernel invariant: `block | offsets | j` occupy
/// disjoint bit-fields). `par_blocks` hands each parallel task a
/// distinct block, so no two threads ever write the same byte — the
/// pointer behaves as a partition into disjoint `&mut` slices.
///
/// SAFETY: callers MUST only use this with `par_blocks`, whose
/// `block_of` produces disjoint block bases. Aliased writes would be
/// undefined behaviour; the disjointness is what makes the `Send`/`Sync`
/// impls sound.
// Allow dead_code: every constructor is in a `#[cfg(target_arch =
// "x86_64")]` SIMD kernel, so on ARM / WASM / RISC-V the type is
// unreferenced (same situation as `expand_with_fixed` below).
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct BlockPtr(pub(crate) *mut f64);

#[allow(dead_code)]
impl BlockPtr {
    /// Read the raw pointer back out.
    ///
    /// Kernels MUST extract the pointer through this `&self` accessor
    /// (not a direct `bp.0` field read) so the enclosing closure
    /// captures the whole `BlockPtr` — which is `Sync` — rather than the
    /// bare `*mut f64` field, which is not. Rust 2021's disjoint capture
    /// would otherwise capture `bp.0` precisely and reject the closure
    /// from rayon's `Sync` bound.
    #[inline(always)]
    pub(crate) fn ptr(&self) -> *mut f64 {
        self.0
    }
}
// SAFETY: see the type-level note — concurrent use only ever touches
// disjoint regions, so sharing the pointer across threads is sound.
unsafe impl Send for BlockPtr {}
unsafe impl Sync for BlockPtr {}

/// `*mut Complex` analogue of [`BlockPtr`] for the scalar fallback
/// kernels, which operate on whole `Complex` amplitudes rather than the
/// paired-`f64` view the AVX-512 kernels use.
///
/// Same disjointness contract: each parallel task writes a distinct set
/// of amplitudes (selected by a per-index guard), so concurrent use is
/// sound. Read the pointer back through [`ComplexPtr::ptr`] (not `.0`)
/// so the enclosing closure captures the whole `Copy` wrapper — which is
/// `Sync` — rather than the bare `!Sync` `*mut Complex` field (Rust 2021
/// disjoint capture; see [`BlockPtr::ptr`]).
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) struct ComplexPtr(pub(crate) *mut aleph_core::Complex);

#[allow(dead_code)]
impl ComplexPtr {
    #[inline(always)]
    pub(crate) fn ptr(&self) -> *mut aleph_core::Complex {
        self.0
    }
}
// SAFETY: as BlockPtr — guarded per-index writes never alias across tasks.
unsafe impl Send for ComplexPtr {}
unsafe impl Sync for ComplexPtr {}

/// Run `body(block_of(k))` for every `k` in `0..count`.
///
/// Sequential when the state vector (`len` amplitudes) is below
/// `policy.min_amps`, otherwise rayon-parallel over the block index. The
/// `block_of(k)` bases MUST be pairwise-disjoint blocks (the
/// SIMD-kernel invariant) so parallel `body` calls never race.
///
/// The result is bit-identical regardless of thread count: each block
/// writes disjoint memory and there is no cross-thread floating-point
/// reduction, so no operation is ever reordered. Oracle equivalence
/// (1e-12) therefore holds for any `RAYON_NUM_THREADS`.
// Allow dead_code: callers are x86_64-only SIMD kernels (see
// `BlockPtr`); the `par_tests` module exercises it on every target.
#[allow(dead_code)]
pub(crate) fn par_blocks(
    policy: ChunkPolicy,
    count: usize,
    len: usize,
    block_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
) {
    if len < policy.min_amps {
        for k in 0..count {
            body(block_of(k));
        }
    } else {
        use rayon::prelude::*;
        // `with_min_len` keeps fine-grained (low-target) kernels from
        // drowning in per-element task overhead: each task runs a
        // contiguous batch of blocks sequentially.
        (0..count)
            .into_par_iter()
            .with_min_len(policy.grain.max(1)) // grain==0 is a misconfiguration; clamp to 1
            .for_each(|k| body(block_of(k)));
    }
}

/// Flatten an outer-block × inner-SIMD-unit iteration into a single
/// parallel dimension of size `outer_count * units_per_block`, so the
/// available parallelism is **independent of which qubit the gate
/// targets**.
///
/// The block-walk kernels nest two loops: an outer walk over
/// `outer_count` blocks and an inner SIMD walk over `units_per_block`
/// LANES-wide units within each block. `par_blocks` alone parallelizes
/// only the outer dimension, which collapses to 1 for a gate on the top
/// qubit (`outer_count == 1`) — that gate then runs fully sequentially
/// despite the inner walk covering the whole state. Flattening exposes
/// the product as the parallel dimension instead.
///
/// `base_of(block_k)` returns the block's base amplitude index; unit
/// `unit_k` within the block lives at `base_of(block_k) + unit_k *
/// stride`. `body(i0)` processes the one SIMD unit at amplitude `i0`.
/// `units_per_block` MUST be a power of two (it is `target_bit / LANES`,
/// a ratio of powers of two), so the block/unit split is a shift+mask.
/// When `units_per_block == 1` this degenerates exactly to
/// `par_blocks(outer_count, …)`.
#[allow(dead_code)]
pub(crate) fn par_units(
    policy: ChunkPolicy,
    outer_count: usize,
    units_per_block: usize,
    stride: usize,
    len: usize,
    base_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
) {
    debug_assert!(units_per_block.is_power_of_two());
    let total = outer_count * units_per_block;
    let unit_bits = units_per_block.trailing_zeros();
    let unit_mask = units_per_block - 1;
    par_blocks(
        policy,
        total,
        len,
        move |u| base_of(u >> unit_bits) + (u & unit_mask) * stride,
        body,
    );
}

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
const PERM_TOL: f64 = 1e-14;

/// Canonical 4×4 permutation matrices recognised by the 2q dispatch.
/// Other 6 valid 4-element permutations (e.g. `X⊗I = [1,0,3,2]`) fall
/// through to the generic kernel.
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

/// Returns true iff both diagonal entries of a 2×2 matrix have
/// squared magnitude below `DIAGONAL_EPS_SQ`. Mirror of
/// `is_diagonal_2x2`. Used as the dispatch heuristic for the 1q
/// anti-diagonal fast path (P1-05).
///
/// ADR 0006: explicit `is_finite` reject precedes the magnitude
/// test. A NaN-poisoned diagonal entry compares `false` for every
/// `<`, which would silently classify the matrix as anti-diagonal
/// and route the NaN to a swap-only path (which only consults
/// `m[0][1]`, `m[1][0]`). Rejecting non-finite diagonals forces the
/// generic kernel to see and propagate the NaN. Three Phase-0 review
/// rounds regressed on the equivalent guard for `is_diagonal_2x2`.
#[inline]
pub(crate) fn is_antidiagonal_2x2(m: &[[aleph_core::Complex; 2]; 2]) -> bool {
    let diag = [&m[0][0], &m[1][1]];
    for entry in diag {
        if !entry.re.is_finite() || !entry.im.is_finite() {
            return false;
        }
        if entry.norm_sqr() >= DIAGONAL_EPS_SQ {
            return false;
        }
    }
    true
}

/// Canonical anti-diagonal 1q matrices recognised by the dispatch.
/// Anti-diagonals not in this set (e.g. arbitrary phased swaps) fall
/// through to `apply_1q_antidiag_*` which does the full complex
/// multiply on `m[0][1]` and `m[1][0]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Perm1qKind {
    /// `X = [[0, 1], [1, 0]]` — pure swap, zero arithmetic.
    X,
    /// `Y = [[0, -i], [i, 0]]` — canonical Pauli-Y.
    YPos,
    /// `Y' = [[0, +i], [-i, 0]]` — anti-Pauli-Y. Rare but trivial extension.
    YNeg,
}

/// Classifies a 4-entry anti-diagonal matrix as one of `{X, YPos, YNeg}`
/// or `None` (= caller dispatches to generic anti-diagonal kernel).
///
/// Caller MUST have already established `is_antidiagonal_2x2(m)`.
/// Component-wise comparison within `PERM_TOL = 1e-14`. Component-wise
/// (not `(z - target).norm_sqr() < PERM_TOL`) because `norm_sqr` is
/// effectively `|z - target|² < PERM_TOL` ⇒ `|z - target| <
/// sqrt(PERM_TOL) ≈ 1e-7`, seven orders looser than the documented
/// tolerance. (Same mistake caught in `is_cz_signature` during P1-07
/// review.)
///
/// NaN handling: if `m[0][1]` or `m[1][0]` is non-finite, every
/// `close()` predicate yields `false` (NaN comparisons), so the
/// function returns `None` and the caller routes to the generic
/// anti-diagonal kernel, which propagates NaN through its complex
/// multiply. No explicit `is_finite` guard needed here.
#[inline]
pub(crate) fn classify_1q_antidiag(m: &[[aleph_core::Complex; 2]; 2]) -> Option<Perm1qKind> {
    let a = m[0][1]; // upper-right
    let b = m[1][0]; // lower-left
    let close = |z: aleph_core::Complex, re: f64, im: f64| {
        (z.re - re).abs() < PERM_TOL && (z.im - im).abs() < PERM_TOL
    };

    if close(a, 1.0, 0.0) && close(b, 1.0, 0.0) {
        return Some(Perm1qKind::X);
    }
    if close(a, 0.0, -1.0) && close(b, 0.0, 1.0) {
        return Some(Perm1qKind::YPos);
    }
    if close(a, 0.0, 1.0) && close(b, 0.0, -1.0) {
        return Some(Perm1qKind::YNeg);
    }
    None
}

/// Tolerance for 8×8 shape-detector functions (`is_identity_8x8`,
/// `is_toffoli`, `is_ccz`). `SHAPE_8X8_TOL = 1e-12` admits FP64
/// rounding drift (~1e-15) with five orders of margin, while still
/// rejecting any experimentally visible mis-calibration (typ. ≥ 1e-6).
/// Coarser than `DIAGONAL_EPS_SQ`/`PERM_TOL` (which guard inner-loop
/// hot paths); these detectors are called once per gate dispatch, not
/// per amplitude.
const SHAPE_8X8_TOL: f64 = 1e-12;

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the 8×8 identity.
///
/// Used as the dispatch pre-check in `apply_3q` to skip the generic
/// 3-qubit kernel entirely when the gate compiles to an identity (e.g.
/// a global-phase-stripped no-op). ADR 0006: the component-wise
/// magnitude test (`abs() > TOL`) already rejects NaN because
/// `NaN > x` is `false`, which would incorrectly classify a NaN entry
/// as "within tolerance". We therefore pair each real/imaginary check
/// with an explicit `is_finite` reject so NaN-poisoned matrices fall
/// through to the generic kernel.
#[inline]
pub(crate) fn is_identity_8x8(m: &[[aleph_core::Complex; 8]; 8]) -> bool {
    for (r, row) in m.iter().enumerate() {
        for (c, entry) in row.iter().enumerate() {
            let expected = if r == c { 1.0 } else { 0.0 };
            if !entry.re.is_finite() || !entry.im.is_finite() {
                return false;
            }
            if (entry.re - expected).abs() > SHAPE_8X8_TOL {
                return false;
            }
            if entry.im.abs() > SHAPE_8X8_TOL {
                return false;
            }
        }
    }
    true
}

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the canonical
/// Toffoli (CCX) matrix: identity on rows 0..=5, then rows 6 ↔ 7
/// swapped. MSB qubit ordering (ADR 0004): for `[q0, q1, q2]`, the
/// basis state index is `(q0<<2)|(q1<<1)|q2`, so row 6 = `|110⟩` and
/// row 7 = `|111⟩`. The CCX gate flips q2 when q0=q1=1, i.e. swaps
/// `|110⟩ ↔ |111⟩`.
///
/// ADR 0006: explicit `is_finite` reject before every magnitude test —
/// same rationale as `is_identity_8x8`.
#[inline]
pub(crate) fn is_toffoli(m: &[[aleph_core::Complex; 8]; 8]) -> bool {
    // Rows 0..=5: identity rows.
    for (r, row) in m.iter().enumerate().take(6) {
        for (c, entry) in row.iter().enumerate() {
            let expected = if r == c { 1.0 } else { 0.0 };
            if !entry.re.is_finite() || !entry.im.is_finite() {
                return false;
            }
            if (entry.re - expected).abs() > SHAPE_8X8_TOL {
                return false;
            }
            if entry.im.abs() > SHAPE_8X8_TOL {
                return false;
            }
        }
    }
    // Row 6: e6 -> e7  ⇒  m[6][7] = 1, all other entries 0.
    for (c, entry) in m[6].iter().enumerate() {
        let expected = if c == 7 { 1.0 } else { 0.0 };
        if !entry.re.is_finite() || !entry.im.is_finite() {
            return false;
        }
        if (entry.re - expected).abs() > SHAPE_8X8_TOL {
            return false;
        }
        if entry.im.abs() > SHAPE_8X8_TOL {
            return false;
        }
    }
    // Row 7: e7 -> e6  ⇒  m[7][6] = 1, all other entries 0.
    for (c, entry) in m[7].iter().enumerate() {
        let expected = if c == 6 { 1.0 } else { 0.0 };
        if !entry.re.is_finite() || !entry.im.is_finite() {
            return false;
        }
        if (entry.re - expected).abs() > SHAPE_8X8_TOL {
            return false;
        }
        if entry.im.abs() > SHAPE_8X8_TOL {
            return false;
        }
    }
    true
}

/// Returns true if `m` is within `SHAPE_8X8_TOL` of the canonical
/// CCZ matrix: diagonal with d[0..6] = +1 and d[7] = -1. MSB qubit
/// ordering (ADR 0004): d[7] corresponds to basis state `|111⟩`, so
/// the CCZ gate sign-flips the state with all three qubits set.
///
/// ADR 0006: explicit `is_finite` reject before every magnitude test —
/// same rationale as `is_identity_8x8`.
#[inline]
pub(crate) fn is_ccz(m: &[[aleph_core::Complex; 8]; 8]) -> bool {
    for (r, row) in m.iter().enumerate() {
        for (c, entry) in row.iter().enumerate() {
            let expected_re = match (r, c) {
                (i, j) if i == j && i < 7 => 1.0,
                (7, 7) => -1.0,
                _ => 0.0,
            };
            if !entry.re.is_finite() || !entry.im.is_finite() {
                return false;
            }
            if (entry.re - expected_re).abs() > SHAPE_8X8_TOL {
                return false;
            }
            if entry.im.abs() > SHAPE_8X8_TOL {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod shape_8x8_tests {
    use super::*;
    use aleph_core::Complex;

    // Diagonal assignment `m[i][i]` requires the same index for row and column,
    // which cannot be expressed as a single iterator; suppress the lint here.
    #[allow(clippy::needless_range_loop)]
    fn identity_8x8() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..8 {
            m[i][i] = o;
        }
        m
    }

    #[allow(clippy::needless_range_loop)]
    fn toffoli_8x8() -> [[Complex; 8]; 8] {
        // Identity rows 0..=5; swap rows 6 ↔ 7.
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..6 {
            m[i][i] = o;
        }
        m[6][7] = o;
        m[7][6] = o;
        m
    }

    #[allow(clippy::needless_range_loop)]
    fn ccz_8x8() -> [[Complex; 8]; 8] {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let mut m = [[z; 8]; 8];
        for i in 0..7 {
            m[i][i] = o;
        }
        m[7][7] = Complex::new(-1.0, 0.0);
        m
    }

    #[test]
    fn identity_detected() {
        assert!(is_identity_8x8(&identity_8x8()));
        assert!(!is_identity_8x8(&toffoli_8x8()));
        assert!(!is_identity_8x8(&ccz_8x8()));
    }

    #[test]
    fn toffoli_detected() {
        assert!(is_toffoli(&toffoli_8x8()));
        assert!(!is_toffoli(&identity_8x8()));
        assert!(!is_toffoli(&ccz_8x8()));
    }

    #[test]
    fn ccz_detected() {
        assert!(is_ccz(&ccz_8x8()));
        assert!(!is_ccz(&identity_8x8()));
        assert!(!is_ccz(&toffoli_8x8()));
    }

    #[test]
    fn toffoli_tolerates_tiny_noise() {
        let mut m = toffoli_8x8();
        m[0][1] = Complex::new(1e-14, 0.0);
        assert!(is_toffoli(&m));
    }

    #[test]
    fn toffoli_rejects_visible_noise() {
        let mut m = toffoli_8x8();
        m[0][1] = Complex::new(1e-6, 0.0);
        assert!(!is_toffoli(&m));
    }

    #[test]
    fn identity_tolerates_tiny_noise() {
        let mut m = identity_8x8();
        m[0][1] = Complex::new(1e-14, 0.0);
        assert!(is_identity_8x8(&m));
    }

    #[test]
    fn ccz_tolerates_tiny_noise() {
        let mut m = ccz_8x8();
        m[0][1] = Complex::new(1e-14, 0.0);
        assert!(is_ccz(&m));
    }
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

    // ---- is_antidiagonal_2x2 ----------------------------------------

    #[test]
    fn is_antidiagonal_2x2_pauli_x() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let o = aleph_core::Complex::new(1.0, 0.0);
        let m = [[zero, o], [o, zero]];
        assert!(super::is_antidiagonal_2x2(&m));
    }

    #[test]
    fn is_antidiagonal_2x2_pauli_y() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let pi = aleph_core::Complex::new(0.0, 1.0);
        let ni = aleph_core::Complex::new(0.0, -1.0);
        let m = [[zero, ni], [pi, zero]];
        assert!(super::is_antidiagonal_2x2(&m));
    }

    #[test]
    fn is_antidiagonal_2x2_rejects_hadamard() {
        let s = aleph_core::Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let m = [[s, s], [s, -s]];
        assert!(!super::is_antidiagonal_2x2(&m));
    }

    #[test]
    fn is_antidiagonal_2x2_rejects_pauli_z() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let o = aleph_core::Complex::new(1.0, 0.0);
        let no = aleph_core::Complex::new(-1.0, 0.0);
        let m = [[o, zero], [zero, no]];
        assert!(!super::is_antidiagonal_2x2(&m));
    }

    #[test]
    fn is_antidiagonal_2x2_rejects_nan_diagonal() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let o = aleph_core::Complex::new(1.0, 0.0);
        let nan = aleph_core::Complex::new(f64::NAN, 0.0);
        let m = [[nan, o], [o, zero]];
        assert!(!super::is_antidiagonal_2x2(&m));
    }

    // ---- classify_1q_antidiag ----------------------------------------

    #[test]
    fn classify_1q_antidiag_pauli_x() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let o = aleph_core::Complex::new(1.0, 0.0);
        let m = [[zero, o], [o, zero]];
        assert_eq!(super::classify_1q_antidiag(&m), Some(super::Perm1qKind::X));
    }

    #[test]
    fn classify_1q_antidiag_pauli_y() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let pi = aleph_core::Complex::new(0.0, 1.0);
        let ni = aleph_core::Complex::new(0.0, -1.0);
        assert_eq!(
            super::classify_1q_antidiag(&[[zero, ni], [pi, zero]]),
            Some(super::Perm1qKind::YPos)
        );
        assert_eq!(
            super::classify_1q_antidiag(&[[zero, pi], [ni, zero]]),
            Some(super::Perm1qKind::YNeg)
        );
    }

    #[test]
    fn classify_1q_antidiag_generic_anti() {
        // [[0, e^{iπ/3}], [e^{-iπ/3}, 0]] — anti-diag but not Pauli.
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let a = aleph_core::Complex::new(0.5, 0.8660254037844386); // e^{iπ/3}: |a|²=0.25+0.75=1
        let b = aleph_core::Complex::new(0.5, -0.8660254037844386); // e^{-iπ/3}
        let m = [[zero, a], [b, zero]];
        assert!(super::classify_1q_antidiag(&m).is_none());
    }

    #[test]
    fn classify_1q_antidiag_nan_off_diagonal_returns_none() {
        let zero = aleph_core::Complex::new(0.0, 0.0);
        let nan = aleph_core::Complex::new(f64::NAN, 0.0);
        let o = aleph_core::Complex::new(1.0, 0.0);
        assert!(super::classify_1q_antidiag(&[[zero, nan], [o, zero]]).is_none());
    }

    // ---- P1-05 T11: portable indexing-coverage test (integer-only, no FP, no SIMD) ----

    mod indexing_coverage {
        use super::super::{control_mask, expand_with_fixed};
        use std::collections::HashSet;

        /// Reproduce the anti-diagonal kernel's pair enumeration as
        /// integer-only operations. Returns the set of (i0, i1) pairs the
        /// kernel would touch.
        fn enumerate_pairs(n_qubits: u32, target: u32, controls: &[u32]) -> Vec<(usize, usize)> {
            let len = 1usize << n_qubits;
            let t_bit = 1usize << target;
            let ctrl_mask = control_mask(controls);
            let mut out = Vec::new();
            for i in 0..len {
                if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
                    out.push((i, i | t_bit));
                }
            }
            out
        }

        /// Reproduce the SIMD outer-walk's pair enumeration using
        /// `expand_with_fixed`. MUST equal `enumerate_pairs` element-wise
        /// (after sorting) for every (target, controls, n) triple.
        fn enumerate_simd_outer_walk(
            n_qubits: u32,
            target: u32,
            controls: &[u32],
        ) -> Vec<(usize, usize)> {
            let t_bit = 1usize << target;
            // Tier-A SIMD outer-walk (mirror of apply_1q_x_avx512's controlled path):
            if controls.is_empty() {
                let mut pairs = Vec::new();
                let outer_step = t_bit << 1;
                let mut block = 0usize;
                while block < (1usize << n_qubits) {
                    for j in 0..t_bit {
                        pairs.push((block | j, block | t_bit | j));
                    }
                    block += outer_step;
                }
                return pairs;
            }
            let mut fixed_above: Vec<(u32, bool)> =
                controls.iter().map(|&c| (c - target - 1, true)).collect();
            fixed_above.sort_unstable_by_key(|&(p, _)| p);
            let outer_count = 1usize << (n_qubits - target - 1 - controls.len() as u32);
            let mut pairs = Vec::new();
            for k in 0..outer_count {
                let block = expand_with_fixed(k, &fixed_above) << (target + 1);
                for j in 0..t_bit {
                    pairs.push((block | j, block | t_bit | j));
                }
            }
            pairs
        }

        #[test]
        fn coverage_matches_naive_no_controls() {
            for n in 2..=8u32 {
                for target in 0..n {
                    let mut naive = enumerate_pairs(n, target, &[]);
                    let mut simd = enumerate_simd_outer_walk(n, target, &[]);
                    naive.sort();
                    simd.sort();
                    assert_eq!(simd, naive, "n={} target={}: pair sets differ", n, target);
                }
            }
        }

        #[test]
        fn coverage_matches_naive_with_one_control_above_target() {
            for n in 3..=8u32 {
                for target in 0..(n - 1) {
                    for c in (target + 1)..n {
                        let mut naive = enumerate_pairs(n, target, &[c]);
                        let mut simd = enumerate_simd_outer_walk(n, target, &[c]);
                        naive.sort();
                        simd.sort();
                        assert_eq!(
                            simd, naive,
                            "n={} target={} c={}: pair sets differ",
                            n, target, c
                        );
                    }
                }
            }
        }

        #[test]
        fn coverage_pairs_are_disjoint_and_in_range() {
            for n in 2..=8u32 {
                let len = 1usize << n;
                for target in 0..n {
                    let pairs = enumerate_pairs(n, target, &[]);
                    let mut seen: HashSet<usize> = HashSet::new();
                    for (i, j) in &pairs {
                        assert!(
                            *i < len && *j < len,
                            "out of range: ({}, {}) for n={}",
                            i,
                            j,
                            n
                        );
                        assert!(seen.insert(*i), "duplicate i={}", i);
                        assert!(seen.insert(*j), "duplicate j={}", j);
                        assert_eq!(i & (1 << target), 0);
                        assert_eq!(j & (1 << target), 1usize << target);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod par_tests {
    use super::par_blocks;
    use crate::kernels::tuning::ChunkPolicy;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn coverage(count: usize, policy: ChunkPolicy, len: usize) -> Vec<usize> {
        let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
        par_blocks(
            policy,
            count,
            len,
            |k| k,
            |slot| {
                hits[slot].fetch_add(1, Ordering::Relaxed);
            },
        );
        hits.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }

    #[test]
    fn par_blocks_visits_each_block_once_sequential() {
        let p = ChunkPolicy {
            min_amps: usize::MAX,
            grain: 64,
        };
        assert!(coverage(1000, p, 0).iter().all(|&h| h == 1));
    }

    #[test]
    fn par_blocks_visits_each_block_once_parallel_across_grains() {
        for grain in [1usize, 16, 64, 1024] {
            let p = ChunkPolicy { min_amps: 0, grain };
            assert!(
                coverage(1000, p, usize::MAX).iter().all(|&h| h == 1),
                "grain={grain} dropped/duplicated a block"
            );
        }
    }
}
