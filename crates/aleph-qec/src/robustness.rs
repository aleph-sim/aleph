//! Opt-in paired decoder experiments under independent priors, drift and temporal bursts.
//! Hoeffding bounds use independent stream means, never correlated shots as Bernoulli trials.
use crate::{
    BinomialInterval95, DetectorErrorModel, Error, FailureWitness, ProbabilityBounds, Result,
    WitnessDecoder,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::Instant;
fn invalid(s: &str) -> Error {
    Error::Robustness(s.into())
}
/// Bounded stream length also enforced during artifact replay.
pub const MAX_ROBUSTNESS_STEPS: u64 = 100_000;
/// Generating process over original DEM mechanisms, without changing decoder priors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum NoiseProcess {
    /// Independent static Bernoulli mechanisms, identical RNG to ordinary DEM sampling.
    Static,
    /// Linear probabilities from the base DEM at t=0 to these values at t=steps-1.
    LinearDrift {
        /// One endpoint probability per original mechanism; a one-step stream uses the base.
        end_probabilities: Vec<f64>,
    },
    /// Inactive initially; before each shot transition with onset/recovery probabilities.
    /// Active duration is geometric with mean 1/recovery; probabilities are capped at one.
    MarkovBurst {
        /// Per-inactive-step activation probability.
        onset: f64,
        /// Per-active-step recovery probability.
        recovery: f64,
        /// Sorted unique original mechanism indices affected by the common latent state.
        affected: Vec<usize>,
        /// Active probability multiplier (>=1); 1 removes the effect but preserves state draws.
        multiplier: f64,
    },
}
/// Require identical ordered mechanism identities, including correlated component hints.
/// Equal detector/observable counts alone do not establish compatible identities.
pub fn validate_noise_compatibility(
    generating: &DetectorErrorModel,
    assumed: &DetectorErrorModel,
) -> Result<()> {
    crate::witness::validate_dem(generating)?;
    crate::witness::validate_dem(assumed)?;
    if generating.detectors != assumed.detectors
        || generating.observables != assumed.observables
        || generating.errors.len() != assumed.errors.len()
        || generating
            .errors
            .iter()
            .zip(&assumed.errors)
            .any(|(a, b)| a.dets != b.dets || a.obs != b.obs || a.components != b.components)
    {
        return Err(invalid(
            "ordered detector/observable mechanism identities differ",
        ));
    }
    Ok(())
}
fn probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}
impl NoiseProcess {
    /// Validate endpoint dimensions, probabilities and original mechanism indices.
    pub fn validate(&self, dem: &DetectorErrorModel) -> Result<()> {
        crate::witness::validate_dem(dem)?;
        match self {
            Self::Static => (),
            Self::LinearDrift { end_probabilities } => {
                if end_probabilities.len() != dem.errors.len()
                    || end_probabilities.iter().any(|&p| !probability(p))
                {
                    return Err(invalid("invalid drift endpoints"));
                }
            }
            Self::MarkovBurst {
                onset,
                recovery,
                affected,
                multiplier,
            } => {
                if !probability(*onset)
                    || !probability(*recovery)
                    || !multiplier.is_finite()
                    || *multiplier < 1.0
                    || affected.windows(2).any(|w| w[0] >= w[1])
                    || affected.iter().any(|&i| i >= dem.errors.len())
                {
                    return Err(invalid("invalid burst law or mechanism indices"));
                }
            }
        }
        Ok(())
    }
}
/// Exact ordinary seed for stream zero; other independent streams use a versioned seed map.
pub fn robustness_stream_seed(seed: u64, stream: u64) -> u64 {
    if stream == 0 {
        seed
    } else {
        crate::experiment::splitmix64(seed ^ 0x5460_5EED_A11C_0001, stream)
    }
}
/// Generated shot metadata. Faults refer to original DEM columns, never physical circuit faults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseShot {
    /// Absolute index inside this stream.
    pub time: u64,
    /// Common latent state after the transition preceding this shot.
    pub burst_active: bool,
    /// Sorted fired original mechanisms.
    pub faults: Vec<usize>,
}
struct Cursor<'a> {
    base: &'a DetectorErrorModel,
    process: &'a NoiseProcess,
    steps: u64,
    seed: u64,
    latent: StdRng,
    active: bool,
}
impl<'a> Cursor<'a> {
    fn new(
        base: &'a DetectorErrorModel,
        process: &'a NoiseProcess,
        steps: u64,
        seed: u64,
    ) -> Result<Self> {
        if steps == 0 || steps > MAX_ROBUSTNESS_STEPS {
            return Err(invalid("stream length outside 1..=100000"));
        }
        process.validate(base)?;
        Ok(Self {
            base,
            process,
            steps,
            seed,
            latent: StdRng::seed_from_u64(crate::experiment::splitmix64(seed, 0x5460_B0757)),
            active: false,
        })
    }
    fn next(&mut self, t: u64) -> NoiseShot {
        let mut dem = self.base.clone();
        match self.process {
            NoiseProcess::Static => (),
            NoiseProcess::LinearDrift { end_probabilities } => {
                let f = if self.steps == 1 {
                    0.0
                } else {
                    t as f64 / (self.steps - 1) as f64
                };
                for (e, &end) in dem.errors.iter_mut().zip(end_probabilities) {
                    if e.prob != end {
                        e.prob = ((1.0 - f) * e.prob + f * end).clamp(0.0, 1.0);
                    }
                }
            }
            NoiseProcess::MarkovBurst {
                onset,
                recovery,
                affected,
                multiplier,
            } => {
                let u = self.latent.gen::<f64>();
                self.active = if self.active {
                    u >= *recovery
                } else {
                    u < *onset
                };
                if self.active {
                    for &i in affected {
                        dem.errors[i].prob = (dem.errors[i].prob * multiplier).min(1.0);
                    }
                }
            }
        }
        NoiseShot {
            time: t,
            burst_active: self.active,
            faults: crate::witness::sample_faults(&dem, self.seed, t),
        }
    }
}
/// Materialize one bounded stream. Zero drift/burst strength preserves ordinary shots exactly.
pub fn generate_noise_stream(
    base: &DetectorErrorModel,
    process: &NoiseProcess,
    steps: u64,
    seed: u64,
) -> Result<Vec<NoiseShot>> {
    let mut c = Cursor::new(base, process, steps, seed)?;
    Ok((0..steps).map(|t| c.next(t)).collect())
}
/// Replay one stateful shot from its independent-stream seed; work is bounded by 100000 steps.
pub(crate) fn replay_noise_shot(
    base: &DetectorErrorModel,
    process: &NoiseProcess,
    steps: u64,
    seed: u64,
    time: u64,
) -> Result<NoiseShot> {
    if time >= steps {
        return Err(invalid("witness time outside stream"));
    }
    let mut c = Cursor::new(base, process, steps, seed)?;
    for t in 0..time {
        c.next(t);
    }
    Ok(c.next(time))
}
/// Predeclared independent stream count and fixed endpoints; no adaptive refresh/stopping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustnessOptions {
    /// Number of independently seeded streams (>0).
    pub streams: u64,
    /// Shots per stream, 1..=100000.
    pub steps: u64,
    /// Sampling seed, independent of decoder disorder seeds.
    pub seed: u64,
    /// Marginal confidence level for each reported estimand, not simultaneous across a sweep.
    pub confidence: f64,
    /// Global maximum number of failed assumed-decoder witnesses retained.
    pub witness_limit: usize,
}
/// Per-independent-stream paired tally, including latent state summary for process checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustnessStream {
    /// Independent stream index.
    pub stream: u64,
    /// Actual seed for this stream.
    pub seed: u64,
    /// Assumed-decoder logical mismatches.
    pub assumed_failures: u64,
    /// Reference-decoder logical mismatches.
    pub reference_failures: u64,
    /// Assumed failure with reference success.
    pub worse: u64,
    /// Assumed success with reference failure.
    pub better: u64,
    /// Assumed invalid syndrome decisions, separate from logical errors (None if unavailable).
    pub assumed_invalid: Option<u64>,
    /// Reference invalid syndrome decisions, separate from logical errors.
    pub reference_invalid: Option<u64>,
    /// Steps in active burst state.
    pub active_steps: u64,
    /// Inactive-to-active transitions, including onset before time zero.
    pub burst_starts: u64,
    /// Active-to-active adjacent pairs (denominator steps-1 across streams).
    pub active_adjacent_pairs: u64,
}
/// Paired robustness result. Both decoders are fixed for the entire campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustnessResult {
    /// Full generating model before scheduled changes.
    pub generating_dem: String,
    /// Canonical generating model SHA256.
    pub generating_sha256: String,
    /// Actual assumed decoder model, separate even when probabilities match.
    pub assumed_dem: String,
    /// Canonical assumed model SHA256.
    pub assumed_sha256: String,
    /// Fixed base-model reference priors; under drift/bursts this is not an oracle refresh.
    pub reference_dem: String,
    /// Canonical reference priors SHA256.
    pub reference_sha256: String,
    /// Generating process and complete parameters.
    pub process: NoiseProcess,
    /// Assumed decoder configuration and disorder seed.
    pub assumed_decoder: crate::DecoderSpec,
    /// Reference decoder configuration and disorder seed.
    pub reference_decoder: crate::DecoderSpec,
    /// Fixed campaign definition.
    pub options: RobustnessOptions,
    /// Independence unit and interval construction.
    pub interval_method: String,
    /// Assumed logical error probability (time-averaged for dynamic processes).
    pub assumed_rate: f64,
    /// Reference logical error probability.
    pub reference_rate: f64,
    /// Assumed marginal confidence bounds.
    pub assumed_interval: ProbabilityBounds,
    /// Reference marginal confidence bounds.
    pub reference_interval: ProbabilityBounds,
    /// Signed difference assumed minus reference (negative means assumed does better).
    pub paired_difference: f64,
    /// Paired signed-difference interval; endpoints can be negative.
    pub difference_interval: ProbabilityBounds,
    /// Full per-independent-stream tallies.
    pub streams: Vec<RobustnessStream>,
    /// Scored predicates, one per decoder per shot; fixed diagnostics may do extra internal work.
    /// Replay/import costs are excluded.
    pub decoder_calls: u64,
    /// Assumed decoding wall time, including its available validity diagnostics.
    pub assumed_decode_seconds: f64,
    /// Reference decoding wall time, including its available validity diagnostics.
    pub reference_decode_seconds: f64,
    /// Total runtime excluding prebuilt decoder construction and subsequent artifact imports.
    pub elapsed_seconds: f64,
    /// Failed assumed-decoder P7-01 artifacts, with both priors and stateful provenance.
    pub witnesses: Vec<FailureWitness>,
}
/// Evaluate fixed decoders on the same generated shots. Static iid rates use exact binomial
/// bounds; paired differences bound two discordant probabilities with Bonferroni allocation.
/// Dynamic processes use Hoeffding over independent stream means in [0,1] (differences [-1,1]).
/// No rounds within a correlated stream are counted as independent trials for uncertainty.
///
/// ```
/// use aleph_qec::*;
/// let dem=DetectorErrorModel::parse("error(0.1) D0 L0")?;
/// let d=WitnessDecoder::new(&dem,DecoderSpec::Mwpm)?;
/// let opts=RobustnessOptions{streams:2,steps:16,seed:546,confidence:0.95,witness_limit:1};
/// let r=run_robustness(&dem,&NoiseProcess::Static,&d,&d,&opts)?;
/// assert_eq!(r.paired_difference,0.0);
/// # Ok::<(),aleph_qec::Error>(())
/// ```
pub fn run_robustness(
    base: &DetectorErrorModel,
    process: &NoiseProcess,
    assumed: &WitnessDecoder,
    reference: &WitnessDecoder,
    options: &RobustnessOptions,
) -> Result<RobustnessResult> {
    let start = Instant::now();
    validate_noise_compatibility(base, assumed.model())?;
    validate_noise_compatibility(base, reference.model())?;
    process.validate(base)?;
    let total = options
        .streams
        .checked_mul(options.steps)
        .filter(|&n| n > 0 && n <= 2_000_000)
        .ok_or_else(|| invalid("campaign requires 1..=2000000 total shots"))?;
    if options.steps == 0
        || options.steps > MAX_ROBUSTNESS_STEPS
        || !options.confidence.is_finite()
        || options.confidence <= 0.0
        || options.confidence >= 1.0
    {
        return Err(invalid("invalid stream length or confidence"));
    }
    let alpha = 1.0 - options.confidence;
    BinomialInterval95::from_counts_alpha(1, 0, alpha / 2.0)?;
    let mut rows = Vec::new();
    let mut witnesses = Vec::new();
    let mut a_seconds = 0.0;
    let mut b_seconds = 0.0;
    for stream in 0..options.streams {
        let seed = robustness_stream_seed(options.seed, stream);
        let mut cursor = Cursor::new(base, process, options.steps, seed)?;
        let mut row = RobustnessStream {
            stream,
            seed,
            assumed_failures: 0,
            reference_failures: 0,
            worse: 0,
            better: 0,
            assumed_invalid: None,
            reference_invalid: None,
            active_steps: 0,
            burst_starts: 0,
            active_adjacent_pairs: 0,
        };
        let mut previous = false;
        for time in 0..options.steps {
            let shot = cursor.next(time);
            row.active_steps += u64::from(shot.burst_active);
            row.burst_starts += u64::from(shot.burst_active && !previous);
            row.active_adjacent_pairs += u64::from(shot.burst_active && previous);
            previous = shot.burst_active;
            let t = Instant::now();
            let a = assumed.decode_faults(&shot.faults)?;
            a_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = reference.decode_faults(&shot.faults)?;
            b_seconds += t.elapsed().as_secs_f64();
            row.assumed_failures += u64::from(a.logical_failure);
            row.reference_failures += u64::from(b.logical_failure);
            row.worse += u64::from(a.logical_failure && !b.logical_failure);
            row.better += u64::from(!a.logical_failure && b.logical_failure);
            if let Some(v) = a.valid {
                *row.assumed_invalid.get_or_insert(0) += u64::from(!v);
            }
            if let Some(v) = b.valid {
                *row.reference_invalid.get_or_insert(0) += u64::from(!v);
            }
            if a.logical_failure && witnesses.len() < options.witness_limit {
                witnesses.push(crate::witness::robustness_witness(
                    assumed, base, process, options, stream, &shot, a,
                )?);
            }
        }
        rows.push(row);
    }
    let a_count = rows.iter().map(|r| r.assumed_failures).sum::<u64>();
    let b_count = rows.iter().map(|r| r.reference_failures).sum::<u64>();
    let a_rate = a_count as f64 / total as f64;
    let b_rate = b_count as f64 / total as f64;
    let bounds = |n, e, al| -> Result<ProbabilityBounds> {
        let c = BinomialInterval95::from_counts_alpha(n, e, al)?;
        Ok(ProbabilityBounds {
            lower: c.lower,
            upper: c.upper,
        })
    };
    let (a_ci, b_ci, d_ci, method) = if *process == NoiseProcess::Static {
        let plus = bounds(total, rows.iter().map(|r| r.worse).sum(), alpha / 2.0)?;
        let minus = bounds(total, rows.iter().map(|r| r.better).sum(), alpha / 2.0)?;
        (
            bounds(total, a_count, alpha)?,
            bounds(total, b_count, alpha)?,
            ProbabilityBounds {
                lower: (plus.lower - minus.upper).next_down().max(-1.0),
                upper: (plus.upper - minus.lower).next_up().min(1.0),
            },
            "iid Clopper-Pearson marginals; paired discordant CP with Bonferroni",
        )
    } else {
        let radius = ((2.0 / alpha).ln() / (2.0 * options.streams as f64)).sqrt();
        let bounded = |v: f64, r: f64, lo: f64, hi: f64| ProbabilityBounds {
            lower: (v - r).next_down().max(lo),
            upper: (v + r).next_up().min(hi),
        };
        (
            bounded(a_rate, radius, 0.0, 1.0),
            bounded(b_rate, radius, 0.0, 1.0),
            bounded(a_rate - b_rate, 2.0 * radius, -1.0, 1.0),
            "Hoeffding independent stream means; paired stream differences range [-1,1]",
        )
    };
    let generating_dem = base.to_dem_string();
    let assumed_dem = assumed.model().to_dem_string();
    let reference_dem = reference.model().to_dem_string();
    Ok(RobustnessResult {
        generating_sha256: crate::witness::digest(generating_dem.as_bytes()),
        assumed_sha256: crate::witness::digest(assumed_dem.as_bytes()),
        reference_sha256: crate::witness::digest(reference_dem.as_bytes()),
        generating_dem,
        assumed_dem,
        reference_dem,
        process: process.clone(),
        assumed_decoder: assumed.configuration().clone(),
        reference_decoder: reference.configuration().clone(),
        options: options.clone(),
        interval_method: method.into(),
        assumed_rate: a_rate,
        reference_rate: b_rate,
        paired_difference: a_rate - b_rate,
        assumed_interval: a_ci,
        reference_interval: b_ci,
        difference_interval: d_ci,
        streams: rows,
        decoder_calls: 2 * total,
        assumed_decode_seconds: a_seconds,
        reference_decode_seconds: b_seconds,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        witnesses,
    })
}
