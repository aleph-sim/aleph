//! BACKLOG P0-11 acceptance: sampling distribution converges to
//! `|ψ|²` at 1 000 000 shots within a 10σ band per outcome.
//!
//! Wall time: ~50 ms with the alias-method sampler. If a future
//! refactor pushes this past 500 ms, mark `#[ignore]` and add a
//! nightly-CI job instead of widening the band.

use aleph_backend::Backend;
use aleph_core::{Gate, GateInstance};
use aleph_sv::NaiveSvBackend;
use smallvec::smallvec;

#[test]
fn bell_state_1m_shots_converges_to_uniform_on_phi_plus() {
    let mut b = NaiveSvBackend::with_seed(0);
    let mut s = b.allocate(2).unwrap();
    b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32]))
        .unwrap();
    b.apply_gate(
        &mut s,
        &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32]),
    )
    .unwrap();

    const N: u32 = 1_000_000;
    let shots = b.sample(&s, N).unwrap();
    let mut hist = [0u64; 4];
    for v in &shots {
        hist[*v as usize] += 1;
    }
    assert_eq!(hist[1], 0, "Bell |Φ+⟩ produced a |01⟩ sample");
    assert_eq!(hist[2], 0, "Bell |Φ+⟩ produced a |10⟩ sample");
    // σ = √(N · p · (1-p)) = √(1e6 · 0.25) = 500; 10σ = 5000.
    let band = 5000.0;
    for &k in &[0usize, 3] {
        let dev = (hist[k] as f64 - 500_000.0).abs();
        assert!(
            dev <= band,
            "outcome {k}: count {} deviates by {dev} > {band}",
            hist[k]
        );
    }
}
