//! Q5-02 BP+OSD oracle tests on the gross code: the decoder achieves a low logical-error rate at
//! low physical rate, and the larger code suppresses errors more below threshold (the prerequisite
//! for a threshold crossing). Hermetic and fast.

use aleph_qec::{run_dem_experiment, BBCode, OsdDecoder};

fn bb(l: usize) -> BBCode {
    BBCode::new(l, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
}

/// BP+OSD decodes the gross code well below threshold: at p=0.02 the logical-error rate is small.
/// A broken decoder (wrong coset, bad ordering) blows this up to tens of percent.
#[test]
fn bposd_decodes_gross_code_at_low_p() {
    let code = bb(12); // [[144,12,12]]
    let dem = code.code_capacity_dem(0.02);
    let osd = OsdDecoder::new(&dem); // normalised min-sum α=0.875, OSD-0
    let r = run_dem_experiment(&dem, 10_000, &osd, 7).expect("decode");
    assert!(
        r.rate < 0.01,
        "BP+OSD logical rate at p=0.02 should be < 1% (got {:.4})",
        r.rate
    );
}

/// Below threshold the distance-12 code suppresses logical errors more than the distance-6 code —
/// the ordering that makes the threshold crossing meaningful.
#[test]
fn larger_distance_suppresses_more_below_threshold() {
    let p = 0.04;
    let r6 = {
        let dem = bb(6).code_capacity_dem(p); // [[72,12,6]]
        run_dem_experiment(&dem, 20_000, &OsdDecoder::new(&dem), 11).expect("d6")
    };
    let r12 = {
        let dem = bb(12).code_capacity_dem(p); // [[144,12,12]]
        run_dem_experiment(&dem, 20_000, &OsdDecoder::new(&dem), 11).expect("d12")
    };
    assert!(
        r12.rate + r12.ci95 < r6.rate - r6.ci95,
        "below threshold d=12 ({:.4}) must beat d=6 ({:.4}) with CI separation",
        r12.rate,
        r6.rate
    );
}

/// The combination sweep never increases the logical-error rate beyond OSD-0 by more than noise —
/// higher order is a strict (soft-weight) refinement. Checks order-12 ≤ order-0 within CI at a
/// moderate p where OSD actually runs.
#[test]
#[ignore = "Monte-Carlo (~30s); nightly"]
fn combination_sweep_does_not_regress() {
    let p = 0.05;
    let dem = bb(12).code_capacity_dem(p);
    let r0 = run_dem_experiment(&dem, 40_000, &OsdDecoder::new(&dem).with_order(0), 3).unwrap();
    let r12 = run_dem_experiment(&dem, 40_000, &OsdDecoder::new(&dem).with_order(12), 3).unwrap();
    assert!(
        r12.rate <= r0.rate + (r0.ci95 + r12.ci95),
        "OSD order-12 ({:.4}) must not regress vs OSD-0 ({:.4})",
        r12.rate,
        r0.rate
    );
}
