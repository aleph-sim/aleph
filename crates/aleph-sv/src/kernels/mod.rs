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
