//! Workhorse SoA ≡ AoS equivalence: every committed oracle circuit
//! produces the same state vector through both `NaiveSvBackend` and
//! `SoaSvBackend` within 1e-12 per amplitude.
//!
//! Lives in `aleph-oracle` rather than `aleph-sv` because the oracle
//! crate already depends on `aleph-sv` + `aleph-parser`; adding
//! `aleph-oracle` as an `aleph-sv` dev-dep would create a Cargo
//! dev-dep cycle. Runs in roughly a second over all 28 fixtures
//! (max n=10). Catches kernel divergence without needing the Qiskit
//! oracle harness.

use aleph_backend::run;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};

const FIXTURES: &[&str] = &[
    "awkward_angles",
    "bell_phi_plus",
    "ghz_10",
    "ghz_3",
    "ghz_5",
    "grover_2q_mark11",
    "identity_1q",
    "kernel_ccx",
    "kernel_cx",
    "kernel_cz",
    "kernel_h",
    "kernel_p",
    "kernel_rx",
    "kernel_ry",
    "kernel_rz",
    "kernel_s",
    "kernel_sdg",
    "kernel_swap",
    "kernel_t",
    "kernel_tdg",
    "kernel_u3",
    "kernel_x",
    "kernel_y",
    "kernel_z",
    // P1-08 multi-controlled fixtures.
    "multi_ctrl_ccx_permuted",
    "multi_ctrl_grover_ccz_3q",
    "qft_3",
    "qft_5",
    "random_clifford_n4_d20",
    "random_nonclifford_n4_d20",
];

#[test]
fn all_fixtures_match_naive() {
    for &name in FIXTURES {
        let qasm_path = aleph_oracle::workspace_path(&format!("oracle/circuits/{name}.qasm"));
        let qasm =
            aleph_oracle::load_qasm(&qasm_path).unwrap_or_else(|e| panic!("load qasm {name}: {e}"));
        let circuit = aleph_parser::parse(&qasm).unwrap_or_else(|e| panic!("parse {name}: {e}"));

        let mut naive = NaiveSvBackend::with_seed(0);
        let naive_state =
            run(&mut naive, &circuit).unwrap_or_else(|e| panic!("naive run {name}: {e}"));
        let naive_amps = naive_state.amplitudes();

        let mut soa = SoaSvBackend::with_seed(0);
        let soa_state = run(&mut soa, &circuit).unwrap_or_else(|e| panic!("soa run {name}: {e}"));
        let soa_re = soa_state.re();
        let soa_im = soa_state.im();

        assert_eq!(naive_amps.len(), soa_re.len(), "{name}: amp count mismatch");
        assert_eq!(soa_re.len(), soa_im.len(), "{name}: re/im length mismatch");

        for i in 0..naive_amps.len() {
            let a = naive_amps[i];
            let dr = a.re - soa_re[i];
            let di = a.im - soa_im[i];
            let delta = (dr * dr + di * di).sqrt();
            assert!(
                delta < 1e-12,
                "fixture {name} amp[{i}]: naive ({}, {}) vs soa ({}, {}); |Δ| = {:.3e}",
                a.re,
                a.im,
                soa_re[i],
                soa_im[i],
                delta,
            );
        }
    }
}
