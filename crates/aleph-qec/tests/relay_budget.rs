//! Explicit relay schedules preserve validity/projection and the legacy default on real shots.
use aleph_qec::{
    sample_shots, BBCode, Decoder, DetectorErrorModel, OsdDecoder, RelayBpDecoder,
    RelayBpOsdDecoder, Syndrome,
};
use rayon::prelude::*;

const TRAP: &str = "error(0.15) D0 D2 D3 L0
error(0.07) D0 D1 D3 L1
error(0.03) D1 D3 D4 L0
error(0.07) D0 D1 D2 D3 L1
error(0.07) D0 D1 D2 D4 L0
error(0.07) D0 D1 D3
error(0.07) D0 D2 D3 D4 L0
error(0.25) D0 D1 D2 D3 D4 L1";

#[test]
fn early_exit_is_first_valid_not_best_over_the_full_schedule() {
    let dem = DetectorErrorModel::parse(TRAP).unwrap();
    let full = RelayBpDecoder::with_budget(&dem, 3, 8, 0.875, (-0.3, 0.9), 7);
    let early = full.clone().with_early_exit(true);
    let short = RelayBpDecoder::with_budget(&dem, 3, 1, 0.875, (-0.3, 0.9), 7);
    let s = Syndrome::new(5, vec![1, 3]);
    assert_eq!(full.decode_errors(&s), (vec![0, 0, 1, 1, 0, 0, 0, 1], true));
    assert_eq!(
        early.decode_errors(&s),
        (vec![0, 1, 1, 1, 0, 1, 0, 1], true)
    );
    assert_eq!(short.decode_errors(&s), early.decode_errors(&s));
    // Both estimates satisfy H, but this fixture's observable class also changes.
    assert_eq!(full.decode(&s).observable_flips, [true, false]);
    assert_eq!(early.decode(&s).observable_flips, [true, true]);
    for relay in [full, early, short] {
        let expected = relay.decode_errors(&s);
        let osd = RelayBpOsdDecoder::with_parts(relay, OsdDecoder::new(&dem));
        assert_eq!(osd.decode_errors(&s), expected);
    }
}

#[test]
fn default_budget_and_configured_parallel_decode_match_on_gross_shots() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let (syndromes, _) = sample_shots(&dem, 256, 514);
    let default = RelayBpDecoder::new(&dem);
    let explicit = RelayBpDecoder::with_budget(&dem, 4, 25, 0.875, (-0.3, 0.9), 0x5E1A_4B9C)
        .with_early_exit(false);
    for s in &syndromes {
        let a = default.decode_soft(s);
        let b = explicit.decode_soft(s);
        assert_eq!((a.ehat, a.converged), (b.ehat, b.converged));
        assert_eq!(
            a.llr.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.llr.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let early =
        RelayBpDecoder::with_budget(&dem, 6, 10, 0.875, (-0.3, 0.9), 7).with_early_exit(true);
    let serial: Vec<_> = syndromes.iter().map(|s| early.decode_errors(s)).collect();
    let parallel: Vec<_> = syndromes
        .par_iter()
        .map(|s| early.decode_errors(s))
        .collect();
    assert_eq!(serial, parallel);
    for (s, (estimate, valid)) in syndromes.iter().zip(serial) {
        let mut syndrome = vec![false; dem.detectors];
        let mut obs = vec![false; dem.observables];
        for (e, &bit) in dem.errors.iter().zip(&estimate) {
            if bit != 0 {
                for &d in &e.dets {
                    syndrome[d as usize] ^= true;
                }
                for &o in &e.obs {
                    obs[o as usize] ^= true;
                }
            }
        }
        if valid {
            assert_eq!(Syndrome::from_bits(&syndrome), *s);
        }
        assert_eq!(obs, early.decode(s).observable_flips);
    }
}
