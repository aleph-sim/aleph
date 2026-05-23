//! `aleph-benches`: workspace-level benchmark harness.
//!
//! The actual benchmarks live in `benches/*.rs` and are run via
//! `cargo bench --bench <name>` or `cargo bench --workspace`.
//! This `lib.rs` exists only so the crate is well-formed; it hosts
//! shared fixture builders used by individual bench files.
//!
//! See `docs/benchmarking.md` at the repo root for the benchmark
//! policy (when to add, how to interpret results, how bencher.dev
//! tracks history). The link is intentionally not a `cargo doc`
//! intra-doc link because the file lives outside the crate.

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
///
/// # Panics
///
/// Panics (with a clear message) if `n_qubits >= usize::BITS` (i.e. ≥
/// 64 on 64-bit targets, ≥ 32 on 32-bit targets). The implicit
/// `1usize << n_qubits` would otherwise overflow into a confusing OOB
/// on the subsequent index write.
///
/// Also (unavoidably) panics if the allocation itself fails — at
/// `n_qubits = 30` the buffer is 16 GiB, beyond that grows exponentially.
///
/// # Examples
///
/// ```
/// use aleph_benches::zero_state;
///
/// let amps = zero_state(3); // 2^3 = 8 amplitudes
/// assert_eq!(amps.len(), 8);
/// assert_eq!(amps[0].re, 1.0);
/// assert_eq!(amps[1].re, 0.0);
/// ```
#[must_use]
pub fn zero_state(n_qubits: u32) -> Vec<Complex> {
    assert!(
        n_qubits < usize::BITS,
        "zero_state: n_qubits={n_qubits} >= usize::BITS={} (would overflow `1usize << n_qubits`)",
        usize::BITS
    );
    let dim = 1usize << n_qubits;
    let mut amps = vec![Complex::new(0.0, 0.0); dim];
    amps[0] = Complex::new(1.0, 0.0);
    amps
}
