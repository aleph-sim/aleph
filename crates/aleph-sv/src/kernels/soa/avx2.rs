//! AVX2 + FMA `apply_1q` kernel. Lands in Task 4 (uncontrolled fast
//! path) and Task 5 (controlled path). This file is currently a stub
//! that delegates to the scalar fallback so the dispatcher in
//! `kernels/soa.rs` can compile and be wired up first.

use aleph_core::Complex;

/// # Safety
///
/// Caller MUST ensure the host CPU supports AVX2 + FMA. The
/// `kernels::soa::apply_1q` dispatcher checks this before invoking.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    // Task 4 replaces this with __m256d intrinsics. Until then,
    // forward to scalar so the runtime dispatcher is safe to enable.
    super::scalar::apply_1q(re, im, target, controls, m);
}
