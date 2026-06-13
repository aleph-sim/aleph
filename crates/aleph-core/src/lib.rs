//! `aleph-core`: Core primitives: Complex, StateVector, Gate, Circuit.
//!
//! Phase 0 — currently exposes the project-wide [`Complex`] type alias
//! and the [`AMPLITUDE_TOL`] tolerance constant. State-vector / gate /
//! circuit types land in later issues.

/// Project-wide complex-number type.
///
/// Defaults to `f64` precision. See `docs/decisions/0001-complex-type.md`
/// at the repo root for why we alias [`num_complex::Complex`] rather than
/// rolling our own. (Path-relative rustdoc links to files outside the
/// crate don't resolve under `cargo doc` output, hence the prose
/// reference instead of a link.)
///
/// The generic parameter exists so a future GPU backend can use
/// `Complex<f32>` (or `Complex<f16>` via a different crate) without
/// renaming the alias. By project convention `T` is expected to
/// implement [`num_traits::Float`]; the alias itself can't carry trait
/// bounds (Rust limitation on type aliases), so this is enforced
/// downstream wherever a function actually performs arithmetic.
///
/// # Examples
///
/// ```
/// use aleph_core::Complex;
///
/// let z = Complex::new(3.0, 4.0);
/// assert_eq!(z.norm(), 5.0);
/// ```
// The alias is the single place in the workspace that's allowed to
// name `num_complex::Complex` directly — clippy's `disallowed-types`
// rule blocks every other reference. See clippy.toml at repo root.
#[allow(clippy::disallowed_types)]
pub type Complex<T = f64> = num_complex::Complex<T>;

pub mod aligned;
pub use aligned::{AlignedBuf, CACHE_LINE};

pub mod gate;
pub use gate::{Gate, GateError, GateInstance, GateMatrix, Param, SymbolId};

pub mod pauli;
pub use pauli::{Pauli, PauliError, PauliString, PauliSum, PauliSumError};

pub mod perm;
pub use perm::bit_permute_buf;

/// Project-wide tolerance for amplitude comparisons in FP64.
///
/// CLAUDE.md § Testing Requirements pins this at `1e-10`. Use this
/// constant in oracle / kernel tests rather than hard-coding a literal
/// — keeps tolerance policy in one place.
pub const AMPLITUDE_TOL: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::{Complex, AMPLITUDE_TOL};

    // Tolerance helper. NaN inputs panic loudly instead of silently
    // returning false (which would mask kernel bugs that produce NaN).
    fn approx_eq(a: f64, b: f64) {
        assert!(!a.is_nan() && !b.is_nan(), "NaN in approx_eq: a={a}, b={b}");
        assert!(
            (a - b).abs() < AMPLITUDE_TOL,
            "assertion failed: |{a} - {b}| = {} >= {AMPLITUDE_TOL}",
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
    fn f32_precision_works() {
        // ADR 0001 sells `Complex<f32>` as the GPU-precision path. Lock
        // in the contract so a future feature-flag change can't silently
        // remove the `Float` impl we need for f32.
        let z = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(z.norm(), 5.0);
        assert_eq!(z.conj(), Complex::<f32>::new(3.0, -4.0));
    }

    #[test]
    fn repr_c_layout() {
        // ADR 0001 leans on `Complex<f64>` being `#[repr(C)]` with field
        // order `(re, im)` for future FFI to `cuDoubleComplex`. Cite the
        // source of truth so the next num-complex bump is easy to audit:
        // num-complex 0.4.x: `#[repr(C)] pub struct Complex<T> { pub re: T, pub im: T }`.
        assert_eq!(core::mem::size_of::<Complex<f64>>(), 16);
        let z = Complex::new(1.0_f64, 2.0_f64);
        // SAFETY: `Complex<f64>` is `#[repr(C)] { re: f64, im: f64 }`, total
        // size 16 bytes with no padding. Transmuting to `[u8; 16]` is a
        // bit-for-bit reinterpret with matching size and alignment — sound.
        let bytes: [u8; 16] = unsafe { core::mem::transmute(z) };
        let re = f64::from_ne_bytes(bytes[..8].try_into().unwrap());
        let im = f64::from_ne_bytes(bytes[8..].try_into().unwrap());
        assert_eq!(re, 1.0);
        assert_eq!(im, 2.0);
    }
}
