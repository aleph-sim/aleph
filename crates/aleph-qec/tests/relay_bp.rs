//! Q5-03 relay-BP oracle tests: relay-BP decodes the gross code well at low p and beats the Q5-02
//! BP+OSD baseline below the error floor. Hermetic.

use aleph_qec::{
    run_dem_experiment, BBCode, CircuitNoise, OsdDecoder, RelayBpDecoder, RelayBpOsdDecoder,
};

fn bb(l: usize) -> BBCode {
    BBCode::new(l, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
}

/// relay-BP decodes the gross code with a low logical-error rate well below threshold.
#[test]
fn relay_bp_decodes_gross_at_low_p() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let relay = RelayBpDecoder::new(&dem);
    let r = run_dem_experiment(&dem, 6_000, &relay, 9).expect("relay");
    assert!(
        r.rate < 0.01,
        "relay-BP logical rate at p=0.03 should be < 1% (got {:.4})",
        r.rate
    );
}

/// relay-BP beats BP+OSD below the error floor: at p=0.03 its logical rate is lower with CI
/// separation. This is the Q5-03 improvement-vs-Q5-02 acceptance, made hermetic.
#[test]
#[ignore = "Monte-Carlo (~2 min); nightly"]
fn relay_bp_beats_bposd_below_floor() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let relay = RelayBpDecoder::new(&dem);
    let bposd = OsdDecoder::with_params(&dem, aleph_qec::DEFAULT_MAX_ITER, 0.875, 10);
    let rr = run_dem_experiment(&dem, 40_000, &relay, 4).expect("relay");
    let ro = run_dem_experiment(&dem, 40_000, &bposd, 4).expect("bposd");
    assert!(
        rr.rate + rr.ci95 < ro.rate - ro.ci95,
        "relay-BP ({:.4}) must beat BP+OSD ({:.4}) below the floor with CI separation",
        rr.rate,
        ro.rate
    );
}

/// Q5-05: relay-BP+OSD decodes the **circuit-level** DEM (depth-7 syndrome extraction) of the
/// [[72,12,6]] BB code with a low logical-error rate. Smoke/quality gate (hermetic, fast).
#[test]
fn relay_bp_osd_decodes_circuit_level() {
    // A short 2-cycle experiment + light combination-sweep keep this under the CI budget; the heavy
    // improvement-vs-baseline comparison at d=6/rounds=6 is the `#[ignore]`d test below.
    let dem = bb(6)
        .circuit_level_dem(2, CircuitNoise::uniform(0.002))
        .expect("circuit-level dem");
    let ro = RelayBpOsdDecoder::new(&dem, 6);
    let r = run_dem_experiment(&dem, 1_000, &ro, 9).expect("relay+osd");
    assert!(
        r.rate < 0.02,
        "relay-BP+OSD circuit-level rate at p=0.002 should be small (got {:.4})",
        r.rate
    );
}

/// Q5-05 improvement-vs-baseline: on the circuit-level DEM, relay-BP+OSD beats plain BP+OSD with
/// CI separation. This is the acceptance criterion for the Q5-05 decoder improvement.
#[test]
#[ignore = "Monte-Carlo (~minutes); nightly"]
fn relay_bp_osd_beats_bposd_circuit_level() {
    let dem = bb(6)
        .circuit_level_dem(6, CircuitNoise::uniform(0.003))
        .expect("circuit-level dem");
    let ro = RelayBpOsdDecoder::new(&dem, 40);
    let bposd = OsdDecoder::with_params(&dem, aleph_qec::DEFAULT_MAX_ITER, 0.875, 40);
    let rr = run_dem_experiment(&dem, 20_000, &ro, 4).expect("relay+osd");
    let rb = run_dem_experiment(&dem, 20_000, &bposd, 4).expect("bposd");
    assert!(
        rr.rate + rr.ci95 < rb.rate - rb.ci95,
        "relay-BP+OSD ({:.4}) must beat BP+OSD ({:.4}) with CI separation",
        rr.rate,
        rb.rate
    );
}
