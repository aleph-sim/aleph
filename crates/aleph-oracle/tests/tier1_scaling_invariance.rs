//! P2-05: the rayon-parallel kernels must be correct on the full Tier-1 set
//! (GHZ / QFT / Grover / random) that the `tier1_scaling` bench measures, on
//! BOTH the raw and the fused (optimized) circuit paths the bench times.
//!
//! Compares the AoS `NaiveSvBackend` against the SoA `SoaSvBackend` within
//! 1e-12 per amplitude — two independent memory layouts and kernel families,
//! so a parallel-block disjointness bug in either path breaks the equality.
//! The optimized variant additionally exercises the dense fused kernels under
//! the same parallel driver (the bench's `tier1_scaling_fused` group). Run
//! under `scripts/p2-05-thread-sweep.sh` (ALEPH_PAR_MIN_AMPS=0 forces the
//! parallel path at this small n across RAYON_NUM_THREADS ∈ {1,2,4,8}); a
//! thread-count-dependent failure would fail the assert. Same idiom as
//! `soa_vs_naive.rs`, on the canonical n=15 Tier-1 fixtures under
//! `scripts/qiskit-baseline/circuits/`.

use aleph_backend::run;
use aleph_ir::Circuit;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};

/// Canonical Tier-1 fixtures at n=15 (fast; 32768 amplitudes). These are the
/// n=15 siblings of the n=25 circuits the bench measures.
const FIXTURES: &[&str] = &[
    "ghz_n15",
    "qft_n15",
    "grover_n15_iters5",
    "random_brickwall_n15_d20",
];

/// Run `circuit` through both backends and assert every amplitude agrees
/// within 1e-12. `label` distinguishes the raw vs optimized variant in
/// failure messages.
fn assert_backends_agree(name: &str, label: &str, circuit: &Circuit) {
    let mut naive = NaiveSvBackend::with_seed(0);
    let naive_state =
        run(&mut naive, circuit).unwrap_or_else(|e| panic!("naive run {name}/{label}: {e}"));
    let naive_amps = naive_state.amplitudes();

    let mut soa = SoaSvBackend::with_seed(0);
    let soa_state =
        run(&mut soa, circuit).unwrap_or_else(|e| panic!("soa run {name}/{label}: {e}"));
    let soa_re = soa_state.re();
    let soa_im = soa_state.im();

    assert_eq!(
        naive_amps.len(),
        soa_re.len(),
        "{name}/{label}: amp count mismatch"
    );
    assert_eq!(
        soa_re.len(),
        soa_im.len(),
        "{name}/{label}: re/im length mismatch"
    );

    for i in 0..naive_amps.len() {
        let a = naive_amps[i];
        let dr = a.re - soa_re[i];
        let di = a.im - soa_im[i];
        let delta = (dr * dr + di * di).sqrt();
        assert!(
            delta < 1e-12,
            "fixture {name}/{label} amp[{i}]: naive ({}, {}) vs soa ({}, {}); |Δ| = {:.3e}",
            a.re,
            a.im,
            soa_re[i],
            soa_im[i],
            delta,
        );
    }
}

// ~90 s in debug (8 n=15 simulations: 4 fixtures × {raw, optimized}), well over
// the CLAUDE.md 30 s bar, so it is `#[ignore]` and excluded from the default
// `cargo test --workspace`. Run it explicitly — `cargo test -p aleph-oracle
// --test tier1_scaling_invariance -- --ignored`, or across thread counts via
// `./scripts/p2-05-thread-sweep.sh` (which passes `--ignored`). CI runs ignored
// tests on the nightly schedule.
#[test]
#[ignore = "~90s; run via scripts/p2-05-thread-sweep.sh or `-- --ignored`"]
fn tier1_fixtures_match_across_backends() {
    for &name in FIXTURES {
        let path =
            aleph_oracle::workspace_path(&format!("scripts/qiskit-baseline/circuits/{name}.qasm"));
        let qasm = aleph_oracle::load_qasm(&path).unwrap_or_else(|e| panic!("load {name}: {e}"));
        let circuit = aleph_parser::parse(&qasm).unwrap_or_else(|e| panic!("parse {name}: {e}"));

        // Raw path (the bench's `tier1_scaling` group).
        assert_backends_agree(name, "raw", &circuit);

        // Fused path (the bench's `tier1_scaling_fused` group) — exercises the
        // dense fused kernels under the same parallel driver.
        let mut optimized = circuit.clone();
        optimized
            .optimize()
            .unwrap_or_else(|e| panic!("optimize {name}: {e:?}"));
        assert_backends_agree(name, "optimized", &optimized);
    }
}
