//! GPU stabilizer (CHP tableau) Clifford backend (P5-07).
//!
//! A device-resident Aaronson-Gottesman tableau with word-parallel Clifford
//! kernels. The qubit-major layout mirrors the CPU backend's ColMajor
//! orientation, so the GPU tableau is bit-for-bit equal to `aleph-stab`'s after
//! the same gate sequence (the oracle in `tests/stab_oracle.rs`).

mod tableau;

pub use tableau::{op, CudaStab, CudaStabState, Generators, StabOp};
