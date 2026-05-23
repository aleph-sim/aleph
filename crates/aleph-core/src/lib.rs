//! `aleph-core`: Core primitives: Complex, StateVector, Gate, Circuit.
//!
//! Phase 0 — currently exposes the project-wide [`Complex`] type alias.
//! State-vector / gate / circuit types land in later issues.

/// Project-wide complex-number type.
///
/// Defaults to `f64` precision. See [ADR 0001](https://github.com/ruslan-splynx/aleph/blob/main/docs/decisions/0001-complex-type.md)
/// for why we alias [`num_complex::Complex`] rather than rolling our own.
///
/// The generic parameter exists so a future GPU backend can use
/// `Complex<f32>` (or `Complex<f16>` via a different crate) without
/// renaming the alias.
pub type Complex<T = f64> = num_complex::Complex<T>;

#[cfg(test)]
mod tests {
    use super::Complex;

    // f64 comparison helper. Use this instead of `assert_eq!` on floats
    // (which compares bit-for-bit and breaks on the slightest rounding).
    fn approx_eq(a: f64, b: f64) {
        let tol = 1e-12;
        assert!(
            (a - b).abs() < tol,
            "assertion failed: |{a} - {b}| = {} >= {tol}",
            (a - b).abs()
        );
    }

    #[test]
    fn arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);

        let sum = a + b;
        assert_eq!(sum, Complex::new(4.0, 1.0));

        let diff = a - b;
        assert_eq!(diff, Complex::new(-2.0, 3.0));

        // (1 + 2i)(3 − i) = 3 − i + 6i − 2i² = 3 + 5i + 2 = 5 + 5i
        let prod = a * b;
        assert_eq!(prod, Complex::new(5.0, 5.0));

        // (1 + 2i) / (3 − i) = (1 + 2i)(3 + i) / |3 − i|² = (1 + 7i) / 10
        let quot = a / b;
        approx_eq(quot.re, 0.1);
        approx_eq(quot.im, 0.7);
    }

    #[test]
    fn magnitude_and_phase() {
        let z = Complex::new(3.0, 4.0);

        // |3 + 4i| = 5
        approx_eq(z.norm(), 5.0);
        // |z|² = 25
        approx_eq(z.norm_sqr(), 25.0);
        // arg(3 + 4i) = atan2(4, 3)
        approx_eq(z.arg(), 4f64.atan2(3.0));
    }

    #[test]
    fn conjugate() {
        let z = Complex::new(1.5, -2.5);
        let z_conj = z.conj();
        assert_eq!(z_conj, Complex::new(1.5, 2.5));

        // z * z̄ is real and equals |z|²
        let prod = z * z_conj;
        approx_eq(prod.re, z.norm_sqr());
        approx_eq(prod.im, 0.0);
    }

    #[test]
    fn zero_and_one() {
        // num-complex provides Zero/One traits — handy for generic kernels.
        let z = Complex::<f64>::new(0.0, 0.0);
        assert!(z.re == 0.0 && z.im == 0.0);

        let one = Complex::<f64>::new(1.0, 0.0);
        let two = one + one;
        assert_eq!(two, Complex::new(2.0, 0.0));
    }

    #[test]
    fn repr_c_layout() {
        // ADR 0001 leans on `Complex<f64>` being `#[repr(C)]` for future
        // FFI to `cuDoubleComplex`. Sanity-check size and field order.
        assert_eq!(core::mem::size_of::<Complex<f64>>(), 16);
        let z = Complex::new(1.0_f64, 2.0_f64);
        let bytes: [u8; 16] = unsafe { core::mem::transmute(z) };
        let re = f64::from_ne_bytes(bytes[..8].try_into().unwrap());
        let im = f64::from_ne_bytes(bytes[8..].try_into().unwrap());
        assert_eq!(re, 1.0);
        assert_eq!(im, 2.0);
    }
}
