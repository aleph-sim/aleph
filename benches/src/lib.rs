//! `aleph-benches`: workspace-level benchmark harness.
//!
//! The actual benchmarks live in `benches/*.rs` and are run via
//! `cargo bench --bench <name>` or `cargo bench --workspace`.
//! This `lib.rs` exists only so the crate is well-formed; it hosts
//! shared fixture builders used by individual bench files.
//!
//! See [`docs/benchmarking.md`](../../docs/benchmarking.md) for the
//! benchmark policy (when to add, how to interpret results, how
//! bencher.dev tracks history).

use aleph_core::Complex;

/// Allocate a state-vector amplitude buffer of length `2^n_qubits`,
/// initialised to the computational-basis |0…0⟩ state (first amplitude
/// is `1 + 0i`, all others `0 + 0i`).
///
/// This is the standard starting point for every benchmark fixture in
/// this crate. Once P0-09 lands the real `Backend` trait, the bench
/// bodies will hand the buffer (or a strongly-typed `StateVector`
/// wrapper) to `backend.apply_circuit(...)`; today they exercise the
/// allocation + initialisation paths so we have a non-trivial baseline
/// for criterion to measure.
#[must_use]
pub fn zero_state(n_qubits: u32) -> Vec<Complex> {
    let dim = 1usize << n_qubits;
    let mut amps = vec![Complex::new(0.0, 0.0); dim];
    amps[0] = Complex::new(1.0, 0.0);
    amps
}
