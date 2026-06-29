//! Logical-error-rate Monte-Carlo harness (Q0-04).
//!
//! This is the reusable measurement instrument for the whole decoder track: every later
//! decoder (MWPM in Q1, Union-Find in Q2, GPU variants in Q3) plugs into it unchanged. Given a
//! [`DetectorErrorModel`] and a [`Decoder`], it samples many noisy shots, decodes each, and
//! reports the logical-error rate with a confidence interval ([`LogicalErrorResult`]).
//!
//! # Why we sample from the DEM, not the circuit
//!
//! The harness draws shots directly from the DEM — for each independent error mechanism it
//! flips a biased coin and XORs the mechanism's detector/observable support into the shot. This
//! is exactly how Stim/Sinter sample a memory experiment, and it makes the generative model and
//! the decoder's model *the same object*: the decoder is built from the DEM the shots came
//! from, so any threshold it produces is meaningful rather than an artefact of two noise models
//! disagreeing. (It is also distributionally identical to inserting each raw mechanism into the
//! circuit and running the Q0-02 frame sampler: merging mechanisms of identical support with
//! the odd-parity rule `p₁⊕p₂` — which [`build_dem`](crate::build_dem) already does — preserves
//! the per-support firing probability, and disjoint supports stay independent.)
//!
//! Shots are independent, so the loop is embarrassingly parallel: each shot derives its own RNG
//! stream from `(seed, shot_index)` via a SplitMix64 mix, which makes the result deterministic
//! for a fixed `seed` regardless of how rayon schedules the work.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::builder::build_dem;
use crate::decoder::{Decoder, LogicalErrorResult};
use crate::dem::DetectorErrorModel;
use crate::error::Result;
use crate::surface::SurfaceCode;
use crate::syndrome::Syndrome;

/// Phenomenological circuit noise for the surface-code memory experiment: an independent `X`
/// error of probability `p_data` on every data qubit before each round and the final readout,
/// and a measurement flip of probability `p_meas` on every ancilla measurement.
///
/// This is the noise model [`MemoryExperiment::phenomenological_mechanisms`] enumerates, so the
/// DEM the harness builds from it is exact and graphlike — the standard pre-threshold benchmark
/// (`crate::surface`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhenomenologicalNoise {
    /// Per-data-qubit `X` error probability per round (and at final readout).
    pub p_data: f64,
    /// Per-ancilla-measurement flip probability.
    pub p_meas: f64,
}

impl PhenomenologicalNoise {
    /// Distinct data and measurement error probabilities.
    pub fn new(p_data: f64, p_meas: f64) -> Self {
        PhenomenologicalNoise { p_data, p_meas }
    }

    /// A single error probability shared by data and measurement noise (`p_data == p_meas`).
    pub fn uniform(p: f64) -> Self {
        PhenomenologicalNoise {
            p_data: p,
            p_meas: p,
        }
    }
}

/// Run the surface-code memory experiment and return its logical-error rate.
///
/// Builds the `rounds`-round memory-Z experiment for `code`, derives its phenomenological DEM
/// under `noise`, constructs a decoder for that DEM via `make_decoder`, and Monte-Carlos
/// `shots` decoded shots.
///
/// The decoder is supplied as a *factory* (`&DetectorErrorModel -> Decoder`) rather than a bare
/// `&dyn Decoder` because a decoder is constructed against the very DEM this function builds —
/// the caller cannot build the decoder before the harness has built the DEM. For example:
///
/// ```no_run
/// use aleph_qec::{run_memory_experiment, NullDecoder, PhenomenologicalNoise, SurfaceCode};
/// let code = SurfaceCode::new(3);
/// let res = run_memory_experiment(
///     &code,
///     &PhenomenologicalNoise::uniform(0.01),
///     /* rounds */ 3,
///     /* shots  */ 10_000,
///     /* seed   */ 1,
///     |dem| Ok(NullDecoder::new(dem.observables)),
/// )?;
/// println!("logical error rate = {} ± {}", res.rate, res.ci95);
/// # Ok::<(), aleph_qec::Error>(())
/// ```
///
/// # Errors
/// Propagates DEM-construction errors ([`crate::Error::Propagation`]), any error from
/// `make_decoder`, and decode-time errors (e.g. an external oracle subprocess failing).
pub fn run_memory_experiment<D, F>(
    code: &SurfaceCode,
    noise: &PhenomenologicalNoise,
    rounds: usize,
    shots: u64,
    seed: u64,
    make_decoder: F,
) -> Result<LogicalErrorResult>
where
    D: Decoder,
    F: FnOnce(&DetectorErrorModel) -> Result<D>,
{
    let exp = code.memory_z_experiment(rounds);
    let mechs = exp.phenomenological_mechanisms(noise.p_data, noise.p_meas);
    let dem = build_dem(&exp.annotated, &mechs)?;
    let decoder = make_decoder(&dem)?;
    run_dem_experiment(&dem, shots, &decoder, seed)
}

/// Monte-Carlo a logical-error rate for an arbitrary [`DetectorErrorModel`] and [`Decoder`].
///
/// Samples `shots` shots from `dem` (each independent error mechanism fires with its own
/// probability, XORing its support into the shot's detectors and observables), decodes them all
/// through [`Decoder::decode_batch`], and counts shots where the decoder's predicted observable
/// flips differ from the truth.
///
/// Deterministic for a fixed `seed`. The sampling is rayon-parallel over shots; the decode is a
/// single batch call (so an external batch decoder pays one round-trip, not one per shot).
///
/// # Errors
/// Propagates any error from the decoder's [`decode_batch`](Decoder::decode_batch).
pub fn run_dem_experiment(
    dem: &DetectorErrorModel,
    shots: u64,
    decoder: &dyn Decoder,
    seed: u64,
) -> Result<LogicalErrorResult> {
    if shots == 0 {
        return Ok(LogicalErrorResult::new(0, 0));
    }

    // Sample every shot in parallel; each shot's RNG is seeded only from (seed, index), so the
    // collected order — and thus the result — is independent of rayon's scheduling.
    let (syndromes, truths): (Vec<Syndrome>, Vec<Vec<bool>>) = sample_shots(dem, shots, seed);

    let predictions = decoder.decode_batch(&syndromes)?;

    let logical_errors = predictions
        .iter()
        .zip(&truths)
        .filter(|(pred, truth)| mispredicted(&pred.observable_flips, truth, dem.observables))
        .count() as u64;

    Ok(LogicalErrorResult::new(shots, logical_errors))
}

/// Sample `shots` independent shots from `dem`, returning their syndromes and *true* observable
/// flips as parallel vectors (shot `i` is `(syndromes[i], truths[i])`).
///
/// This is the same sampler [`run_dem_experiment`] decodes, exposed so callers that need the raw
/// shot stream — e.g. the Q6-21 sim↔RTL co-simulation harness, which dumps these exact shots to
/// drive a Verilated decoder — sample *identically* to the software baseline: same `(seed, index)`
/// SplitMix64 derivation per shot, same DEM order, so the RTL and software decoders see the very
/// same Monte-Carlo stream and their logical-error rates are comparable shot-for-shot.
///
/// Deterministic for a fixed `seed` regardless of rayon scheduling (each shot seeds only from
/// `(seed, index)`).
pub fn sample_shots(
    dem: &DetectorErrorModel,
    shots: u64,
    seed: u64,
) -> (Vec<Syndrome>, Vec<Vec<bool>>) {
    (0..shots)
        .into_par_iter()
        .map(|s| sample_shot(dem, seed, s))
        .unzip()
}

/// Draw one shot from the DEM: returns its measured syndrome and the *true* observable flips.
fn sample_shot(dem: &DetectorErrorModel, seed: u64, shot: u64) -> (Syndrome, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(splitmix64(seed, shot));
    let mut det = vec![false; dem.detectors];
    let mut obs = vec![false; dem.observables];
    for e in &dem.errors {
        // One coin per mechanism, in DEM order, so the stream is reproducible per shot.
        if rng.gen::<f64>() < e.prob {
            for &d in &e.dets {
                det[d as usize] ^= true;
            }
            for &o in &e.obs {
                obs[o as usize] ^= true;
            }
        }
    }
    (Syndrome::from_bits(&det), obs)
}

/// Whether predicted observable flips differ from the truth on any observable.
fn mispredicted(pred: &[bool], truth: &[bool], observables: usize) -> bool {
    (0..observables)
        .any(|o| pred.get(o).copied().unwrap_or(false) != truth.get(o).copied().unwrap_or(false))
}

/// SplitMix64 finalizer over `base + shot * φ⁻¹`, giving each shot a well-mixed, independent
/// seed from the run seed and its index (Steele et al., "Fast Splittable PRNGs", 2014).
fn splitmix64(base: u64, shot: u64) -> u64 {
    let mut z = base.wrapping_add(shot.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::NullDecoder;
    use crate::dem::DemError;

    /// A DEM with a single observable-flipping edge, used to check sampler marginals and the
    /// NullDecoder's "every flip is a logical error" behaviour.
    fn single_edge_dem(prob: f64) -> DetectorErrorModel {
        DetectorErrorModel {
            detectors: 1,
            observables: 1,
            errors: vec![DemError::new(prob, vec![0], vec![0])],
        }
    }

    #[test]
    fn p_zero_gives_exactly_zero_rate() {
        // Every edge has probability 0 → no error ever fires → no observable is ever flipped,
        // so the rate is exactly 0 for any decoder. (Acceptance: p=0 ⇒ rate 0.)
        let code = SurfaceCode::new(3);
        let res = run_memory_experiment(
            &code,
            &PhenomenologicalNoise::uniform(0.0),
            3,
            5_000,
            42,
            |dem| Ok(NullDecoder::new(dem.observables)),
        )
        .unwrap();
        assert_eq!(res.logical_errors, 0);
        assert_eq!(res.rate, 0.0);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let code = SurfaceCode::new(3);
        let run = |seed| {
            run_memory_experiment(
                &code,
                &PhenomenologicalNoise::uniform(0.03),
                3,
                20_000,
                seed,
                |dem| Ok(NullDecoder::new(dem.observables)),
            )
            .unwrap()
        };
        assert_eq!(run(7), run(7), "same seed ⇒ identical result");
        assert_ne!(
            run(7).logical_errors,
            run(8).logical_errors,
            "different seeds ⇒ different sample (overwhelmingly likely)"
        );
    }

    #[test]
    fn null_decoder_rate_matches_observable_flip_probability() {
        // With one edge that flips observable L0 with probability p, the NullDecoder (predicts
        // no flip) is wrong exactly when the edge fires. So the empirical rate ≈ p.
        let p = 0.2;
        let dem = single_edge_dem(p);
        let dec = NullDecoder::new(dem.observables);
        let res = run_dem_experiment(&dem, 200_000, &dec, 123).unwrap();
        assert!(
            (res.rate - p).abs() < 0.01,
            "rate {} should be ≈ {p}",
            res.rate
        );
        // The true rate lies inside the reported 95% CI.
        assert!((res.rate - p).abs() < 4.0 * res.ci95 + 1e-9);
    }

    #[test]
    fn sampler_detector_frequency_matches_edge_probability() {
        // Two independent edges touch detector D0 (probs p, q); D0 fires with the odd-parity
        // combination p⊕q. This checks the per-shot Bernoulli sampling and XOR accumulation.
        let (p, q) = (0.1, 0.05);
        let dem = DetectorErrorModel {
            detectors: 1,
            observables: 0,
            errors: vec![
                DemError::new(p, vec![0], vec![]),
                DemError::new(q, vec![0], vec![]),
            ],
        };
        let shots = 300_000u64;
        let fired = (0..shots)
            .map(|s| sample_shot(&dem, 99, s).0.weight())
            .filter(|&w| w == 1)
            .count();
        let emp = fired as f64 / shots as f64;
        let pred = p + q - 2.0 * p * q;
        assert!((emp - pred).abs() < 0.005, "emp {emp} vs predicted {pred}");
    }

    #[test]
    fn zero_shots_is_empty_result() {
        let dem = single_edge_dem(0.1);
        let res = run_dem_experiment(&dem, 0, &NullDecoder::new(1), 1).unwrap();
        assert_eq!(res.shots, 0);
        assert_eq!(res.rate, 0.0);
        assert_eq!(res.ci95, 0.0);
    }
}
