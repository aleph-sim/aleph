//! Q5-03 relay-BP oracle tests: relay-BP decodes the gross code well at low p and beats the Q5-02
//! BP+OSD baseline below the error floor. Hermetic.

use aleph_qec::{run_dem_experiment, BBCode, OsdDecoder, RelayBpDecoder};

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
