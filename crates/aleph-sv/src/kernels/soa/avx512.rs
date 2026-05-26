//! AVX-512 `apply_1q` kernel. Lands in Task 7. Currently a stub that
//! delegates to the AVX2 path (which itself forwards to scalar until
//! Task 4 ships).

use aleph_core::Complex;

/// # Safety
///
/// Caller MUST ensure the host CPU supports AVX-512F. The
/// `kernels::soa::apply_1q` dispatcher checks this before invoking.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    // Task 7 replaces this with __m512d intrinsics. Until then,
    // forward to AVX2.
    // SAFETY: AVX-512F implies AVX2 + FMA on every shipping CPU; the
    // caller's avx512f feature gate is sufficient for the AVX2 path.
    super::avx2::apply_1q(re, im, target, controls, m);
}
