//! Scalar SoA `apply_1q` fallback.
//!
//! Compiles to ~16 vmulsd + 8 vp*q (masked-loop vectorisation) on
//! x86-64 with `-C target-cpu=native`; ARM NEON auto-vec yields a
//! similar packed-double shape. Per ADR 0007 do NOT replace this with
//! a bit-manip restructure — that pessimises LLVM's masked-loop
//! transformation. The hand-written SIMD paths (`avx2.rs`,
//! `avx512.rs`) override on x86 hosts that support those features;
//! everywhere else this is the only kernel.

use aleph_core::Complex;

pub(crate) fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let ctrl_mask = crate::kernels::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let a0_re = re[i];
            let a0_im = im[i];
            let a1_re = re[j];
            let a1_im = im[j];
            // row 0
            re[i] =
                m[0][0].re * a0_re - m[0][0].im * a0_im + m[0][1].re * a1_re - m[0][1].im * a1_im;
            im[i] =
                m[0][0].re * a0_im + m[0][0].im * a0_re + m[0][1].re * a1_im + m[0][1].im * a1_re;
            // row 1
            re[j] =
                m[1][0].re * a0_re - m[1][0].im * a0_im + m[1][1].re * a1_re - m[1][1].im * a1_im;
            im[j] =
                m[1][0].re * a0_im + m[1][0].im * a0_re + m[1][1].re * a1_im + m[1][1].im * a1_re;
        }
        i += 1;
    }
}
