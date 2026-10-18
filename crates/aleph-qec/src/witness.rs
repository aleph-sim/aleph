//! Opt-in DEM mechanism capture, deterministic replay and budgeted 1-minimal reduction.
//! Ordinary sampling and decoder hot paths do not call this module.
use crate::{
    Decoder, DetectorErrorModel, Error, FixedRelayBp, MwpmDecoder, RelayBpDecoder, Result,
    Syndrome, UnionFindDecoder,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

fn invalid(message: impl Into<String>) -> Error {
    Error::Witness(message.into())
}
pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Complete, versioned choices supported by the replay tool. Priors always come from the DEM.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum DecoderSpec {
    /// Matching with the default locality policy; components are split only for decoding.
    Mwpm,
    /// Unweighted Union-Find; components are split only for decoding.
    UnionFind,
    /// Floating-point relay BP with fixed disorder seed and explicit full/early schedule.
    Relay {
        /// Number of relay legs.
        legs: usize,
        /// Sweeps per leg.
        iters: u32,
        /// Disorder seed, independent of the sampling seed.
        seed: u64,
        /// Stop at the first syndrome-valid decision.
        early_exit: bool,
    },
    /// Q-format relay BP, alpha=7/8, gamma=(-0.3,0.9).
    FixedRelay {
        /// Number of relay legs.
        legs: usize,
        /// Sweeps per leg.
        iters: u32,
        /// Disorder seed.
        seed: u64,
        /// Signed message width.
        msg_bits: u32,
        /// Fractional bits.
        frac_bits: u32,
        /// Stop at the first syndrome-valid decision.
        early_exit: bool,
    },
}

enum Engine {
    Mwpm(MwpmDecoder),
    Uf(UnionFindDecoder),
    Relay(RelayBpDecoder),
    Fixed(FixedRelayBp),
}

/// A decoder built once against immutable original priors. Reduction never rebuilds it.
pub struct WitnessDecoder {
    spec: DecoderSpec,
    dem: DetectorErrorModel,
    engine: Engine,
}
impl WitnessDecoder {
    /// Immutable original target model, including fixed decoder priors.
    pub fn model(&self) -> &DetectorErrorModel {
        &self.dem
    }
    /// Fixed decoder configuration, including the independent disorder seed.
    pub fn configuration(&self) -> &DecoderSpec {
        &self.spec
    }
    /// Evaluate a counterfactual set of original DEM mechanisms, without artifact allocation.
    pub fn decode_faults(&self, faults: &[usize]) -> Result<WitnessOutcome> {
        self.outcome(faults)
    }
    /// Validate the DEM and configuration, then build the replay engine.
    pub fn new(dem: &DetectorErrorModel, spec: DecoderSpec) -> Result<Self> {
        validate_dem(dem)?;
        let schedule = |legs: usize, iters: u32| -> Result<()> {
            if legs == 0 || iters == 0 || legs > 1024 || u64::from(iters) * legs as u64 > 1_000_000
            {
                return Err(invalid(
                    "schedule must be positive, <=1024 legs and <=1000000 sweeps",
                ));
            }
            Ok(())
        };
        let engine = match spec {
            DecoderSpec::Mwpm => Engine::Mwpm(MwpmDecoder::new_with_split_components(dem)?),
            DecoderSpec::UnionFind => {
                Engine::Uf(UnionFindDecoder::new_with_split_components(dem)?.weighted(false))
            }
            DecoderSpec::Relay {
                legs,
                iters,
                seed,
                early_exit,
            } => {
                schedule(legs, iters)?;
                Engine::Relay(
                    RelayBpDecoder::with_budget(dem, legs, iters, 0.875, (-0.3, 0.9), seed)
                        .with_early_exit(early_exit),
                )
            }
            DecoderSpec::FixedRelay {
                legs,
                iters,
                seed,
                msg_bits,
                frac_bits,
                early_exit,
            } => {
                schedule(legs, iters)?;
                if !(2..=31).contains(&msg_bits) || frac_bits >= msg_bits {
                    return Err(invalid("invalid fixed-point width"));
                }
                Engine::Fixed(
                    FixedRelayBp::with_budget(
                        dem,
                        legs,
                        iters,
                        (-0.3, 0.9),
                        seed,
                        msg_bits,
                        frac_bits,
                    )
                    .with_early_exit(early_exit),
                )
            }
        };
        Ok(Self {
            spec,
            dem: dem.clone(),
            engine,
        })
    }
    fn outcome(&self, faults: &[usize]) -> Result<WitnessOutcome> {
        let (syndrome, truth) = reconstruct_faults(&self.dem, faults)?;
        let (prediction, valid, first_valid_iter) = match &self.engine {
            Engine::Mwpm(d) => (d.decode(&syndrome), None, None),
            Engine::Uf(d) => (d.decode(&syndrome), None, None),
            Engine::Relay(d) => {
                let (p, v) = d.decode_relay(&syndrome);
                (p, Some(v), None)
            }
            Engine::Fixed(d) => {
                let (p, v) = d.decode_fixed(&syndrome);
                let (_, i) = d.iters_to_valid(&syndrome);
                (p, Some(v), Some(i))
            }
        };
        if prediction.observable_flips.len() != self.dem.observables {
            return Err(invalid("prediction width mismatch"));
        }
        let logical_failure = prediction.observable_flips != truth;
        Ok(WitnessOutcome {
            syndrome: syndrome.fired,
            truth,
            prediction: prediction.observable_flips,
            logical_failure,
            valid,
            first_valid_iter,
        })
    }
    /// Replay a witness under this engine's configuration. The model must match exactly.
    pub fn replay(&self, witness: &FailureWitness) -> Result<WitnessOutcome> {
        let dem = witness.validate_structure()?;
        if dem != self.dem {
            return Err(invalid("replay DEM mismatch"));
        }
        self.outcome(&witness.faults)
    }
}

pub(crate) fn validate_dem(dem: &DetectorErrorModel) -> Result<()> {
    if dem.detectors > u32::MAX as usize || dem.observables > 64 {
        return Err(invalid("unsupported DEM dimensions"));
    }
    for e in &dem.errors {
        if !e.prob.is_finite()
            || !(0.0..=1.0).contains(&e.prob)
            || e.dets.iter().any(|&d| d as usize >= dem.detectors)
            || e.obs.iter().any(|&o| o as usize >= dem.observables)
        {
            return Err(invalid("invalid DEM probability or target"));
        }
    }
    // Reject inconsistent component caches instead of silently changing the sampling model.
    if DetectorErrorModel::parse(&dem.to_dem_string())? != *dem {
        return Err(invalid("DEM does not round-trip exactly"));
    }
    Ok(())
}

/// Reconstruct by XORing original mechanisms, including duplicate supports and logical-only faults.
/// Fault indices must be strictly increasing; repeated indices are malformed, not extra faults.
pub fn reconstruct_faults(
    dem: &DetectorErrorModel,
    faults: &[usize],
) -> Result<(Syndrome, Vec<bool>)> {
    if faults.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("fault indices must be sorted and unique"));
    }
    let mut det = vec![false; dem.detectors];
    let mut obs = vec![false; dem.observables];
    for &i in faults {
        let e = dem
            .errors
            .get(i)
            .ok_or_else(|| invalid("mechanism index out of range"))?;
        for &d in &e.dets {
            let bit = det
                .get_mut(d as usize)
                .ok_or_else(|| invalid("detector out of range"))?;
            *bit ^= true;
        }
        for &o in &e.obs {
            let bit = obs
                .get_mut(o as usize)
                .ok_or_else(|| invalid("observable out of range"))?;
            *bit ^= true;
        }
    }
    Ok((Syndrome::from_bits(&det), obs))
}

/// Replay result: convergence is a diagnostic and is never the logical-failure predicate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessOutcome {
    /// Fired detectors.
    pub syndrome: Vec<u32>,
    /// True logical flips.
    pub truth: Vec<bool>,
    /// Predicted logical flips.
    pub prediction: Vec<bool>,
    /// Prediction differs from truth, even when syndrome is empty.
    pub logical_failure: bool,
    /// Syndrome validity, if the engine supplies it.
    pub valid: Option<bool>,
    /// First-valid sweep or full schedule length; available for fixed relay only.
    pub first_valid_iter: Option<u32>,
}

/// Sampler provenance: v2 conditional/enumeration, v3 robustness. Version 1 uses ordinary draws.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum WitnessSampling {
    /// Stateful generating process with decoder priors retained in the enclosing witness.
    RobustnessV1 {
        /// Original generating model (may differ from assumed decoder probabilities).
        generating_dem: String,
        /// Complete deterministic schedule or latent-state law.
        process_json: String,
        /// Original campaign sampling seed.
        campaign_seed: u64,
        /// Independent stream identity.
        stream: u64,
        /// Declared length, needed for drift interpolation and bounded state replay.
        steps: u64,
        /// State after the transition before this shot; verified on import.
        burst_active: bool,
    },
    /// Log-suffix conditional RNG v1; seed/shot identify a conditional draw, not an ordinary shot.
    FaultCountV1 {
        /// Total fired mechanisms, including p=1 mechanisms.
        faults: usize,
    },
    /// Lexicographic enumeration v1; shot is the rank, seed is retained campaign metadata.
    FaultCountEnumerationV1 {
        /// Total fired mechanisms.
        faults: usize,
    },
}

/// Portable v1/v2/v3 artifact. Indices identify DEM mechanisms, not physical circuit locations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureWitness {
    /// Schema version: 1 for ordinary, 2 for stratified, 3 for robustness provenance.
    pub version: u32,
    /// Absent for legacy ordinary draws; explicit conditional/rank (v2) or generating-process (v3) provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<WitnessSampling>,
    /// Exact canonical decoder-prior DEM including correlated components; v3 generation is in sampling.
    pub dem: String,
    /// SHA-256 of UTF-8 DEM bytes.
    pub dem_sha256: String,
    /// Original sampling seed.
    pub seed: u64,
    /// Original ordinary/conditional sample index, or enumeration rank (see sampling).
    /// Reduced faults no longer describe a random draw at this index.
    pub shot: u64,
    /// Original fired mechanisms, retained through reduction.
    pub original_faults: Vec<usize>,
    /// Current fired mechanisms, an ordered subset of original_faults.
    pub faults: Vec<usize>,
    /// Complete source decoder configuration.
    pub decoder: DecoderSpec,
    /// Explicit provenance: v1 supports only "dem-mechanisms".
    pub provenance: String,
    /// Original capture outcome, retained through reduction.
    pub original_outcome: WitnessOutcome,
    /// Current outcome under the source decoder.
    pub outcome: WitnessOutcome,
    /// Reduction status and budgets, absent for an unreduced capture.
    pub minimization: Option<MinimizationReceipt>,
}
impl FailureWitness {
    fn validate_structure(&self) -> Result<DetectorErrorModel> {
        if !matches!(
            (self.version, &self.sampling),
            (1, None)
                | (
                    2,
                    Some(
                        WitnessSampling::FaultCountV1 { .. }
                            | WitnessSampling::FaultCountEnumerationV1 { .. }
                    )
                )
                | (3, Some(WitnessSampling::RobustnessV1 { .. }))
        ) || self.provenance != "dem-mechanisms"
            || digest(self.dem.as_bytes()) != self.dem_sha256
        {
            return Err(invalid("version, provenance or DEM hash mismatch"));
        }
        let dem = DetectorErrorModel::parse(&self.dem)?;
        validate_dem(&dem)?;
        let original = match &self.sampling {
            Some(WitnessSampling::RobustnessV1 {
                generating_dem,
                process_json,
                campaign_seed,
                stream,
                steps,
                burst_active,
            }) => {
                let generating = DetectorErrorModel::parse(generating_dem)?;
                let process: crate::NoiseProcess =
                    serde_json::from_str(process_json).map_err(|e| invalid(e.to_string()))?;
                crate::validate_noise_compatibility(&generating, &dem)?;
                if self.seed != crate::robustness_stream_seed(*campaign_seed, *stream) {
                    return Err(invalid("robustness stream seed mismatch"));
                }
                let shot = crate::robustness::replay_noise_shot(
                    &generating,
                    &process,
                    *steps,
                    self.seed,
                    self.shot,
                )?;
                if shot.burst_active != *burst_active {
                    return Err(invalid("burst state does not replay"));
                }
                shot.faults
            }
            None => sample_faults(&dem, self.seed, self.shot),
            Some(WitnessSampling::FaultCountV1 { faults }) => {
                crate::FaultCountPlan::new(&dem, *faults, crate::stratified::MAX_CONDITIONAL_CELLS)?
                    .sample(*faults, self.seed, self.shot)?
            }
            Some(WitnessSampling::FaultCountEnumerationV1 { faults }) => {
                crate::FaultCountPlan::new(&dem, *faults, crate::stratified::MAX_CONDITIONAL_CELLS)?
                    .enumerated_faults(*faults, self.shot)?
            }
        };
        if original != self.original_faults {
            return Err(invalid("original faults do not match sampling seed/shot"));
        }
        for (faults, outcome) in [
            (&self.original_faults, &self.original_outcome),
            (&self.faults, &self.outcome),
        ] {
            let (s, t) = reconstruct_faults(&dem, faults)?;
            if s.fired != outcome.syndrome
                || t != outcome.truth
                || outcome.prediction.len() != dem.observables
                || outcome.logical_failure != (t != outcome.prediction)
            {
                return Err(invalid("stored outcome does not reconstruct"));
            }
        }
        if self
            .faults
            .iter()
            .any(|f| self.original_faults.binary_search(f).is_err())
        {
            return Err(invalid("reduced faults not a subset"));
        }
        Ok(dem)
    }
    /// Serialize structurally validated data; import also verifies source decoder outcomes.
    pub fn to_json(&self) -> Result<String> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self).map_err(|e| invalid(e.to_string()))
    }
    /// Import, reconstruct and verify both original and current source-decoder outcomes.
    pub fn from_json(text: &str) -> Result<Self> {
        let witness: Self = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        let dem = witness.validate_structure()?;
        let decoder = WitnessDecoder::new(&dem, witness.decoder.clone())?;
        if decoder.outcome(&witness.original_faults)? != witness.original_outcome
            || decoder.outcome(&witness.faults)? != witness.outcome
        {
            return Err(invalid("source decoder replay mismatch"));
        }
        if let Some(receipt) = &witness.minimization {
            if receipt.decoder_calls > receipt.call_budget {
                return Err(invalid("reduction exceeds declared call budget"));
            }
            if receipt.status == MinimizeStatus::OneMinimal {
                if !witness.outcome.logical_failure || receipt.decoder_calls == 0 {
                    return Err(invalid("invalid 1-minimal claim"));
                }
                for i in 0..witness.faults.len() {
                    let mut faults = witness.faults.clone();
                    faults.remove(i);
                    if decoder.outcome(&faults)?.logical_failure {
                        return Err(invalid("1-minimality check failed"));
                    }
                }
            }
        }
        Ok(witness)
    }
    /// Exact-example signature: SHA-256 of DEM hash, decoder config and current mechanism indices.
    /// Different fault sets sharing a syndrome deliberately remain separate examples.
    pub fn signature(&self) -> Result<String> {
        self.validate_structure()?;
        let bytes = serde_json::to_vec(&(&self.dem_sha256, &self.decoder, &self.faults))
            .map_err(|e| invalid(e.to_string()))?;
        Ok(digest(&bytes))
    }
}

/// Opt-in companion sampler. Uses the ordinary sampler's exact RNG and one coin per mechanism.
/// The decoder can be retained across calls; ordinary capture-disabled sampling is untouched.
///
/// ```
/// use aleph_qec::{capture_shot, DecoderSpec, DetectorErrorModel, WitnessDecoder,
///     FailureWitness};
/// let dem = DetectorErrorModel::parse("error(0.1) D0 L0")?;
/// let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm)?;
/// let witness = capture_shot(&decoder, 543, 0)?;
/// let imported = FailureWitness::from_json(&witness.to_json()?)?;
/// assert_eq!(decoder.replay(&imported)?, witness.outcome);
/// # Ok::<(), aleph_qec::Error>(())
/// ```
pub fn capture_shot(decoder: &WitnessDecoder, seed: u64, shot: u64) -> Result<FailureWitness> {
    let faults = sample_faults(&decoder.dem, seed, shot);
    let outcome = decoder.outcome(&faults)?;
    let dem = decoder.dem.to_dem_string();
    Ok(FailureWitness {
        version: 1,
        sampling: None,
        dem_sha256: digest(dem.as_bytes()),
        dem,
        seed,
        shot,
        original_faults: faults.clone(),
        faults,
        decoder: decoder.spec.clone(),
        provenance: "dem-mechanisms".into(),
        minimization: None,
        original_outcome: outcome.clone(),
        outcome,
    })
}

/// Limits checked before each predicate call. Time cannot preempt a decoder already running.
#[derive(Clone, Copy, Debug)]
pub struct MinimizeBudget {
    /// Maximum predicate calls, including initial replay and all final certification calls.
    pub decoder_calls: u64,
    /// Wall-time deadline; may be exceeded by one in-flight predicate call.
    pub time: Duration,
}
/// Whether deletion-minimality was fully certified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MinimizeStatus {
    /// Every single-fault deletion of the final set was tested and did not fail.
    OneMinimal,
    /// Call or wall-time budget exhausted. No minimality claim is made.
    Incomplete,
}
/// Serializable reduction receipt; counts are predicate calls, not internal BP sweeps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MinimizationReceipt {
    /// A completed final single-deletion sweep or interrupted work.
    pub status: MinimizeStatus,
    /// Maximum predicate evaluations allowed.
    pub call_budget: u64,
    /// Time budget in milliseconds, rounded down.
    pub time_budget_ms: u128,
    /// Predicate evaluations consumed, including verification and certification.
    pub decoder_calls: u64,
    /// Elapsed wall time in microseconds.
    pub elapsed_us: u128,
}
/// Reduction receipt; the original capture remains embedded in the resulting witness.
pub struct MinimizeResult {
    /// Reduced failing witness.
    pub witness: FailureWitness,
    /// Completed certification or interrupted work.
    pub status: MinimizeStatus,
    /// Number of predicate evaluations (fixed relay diagnostics perform a second internal pass).
    pub decoder_calls: u64,
    /// Elapsed wall time, including initial verification.
    pub elapsed: Duration,
}
impl WitnessDecoder {
    /// Deterministic complement chunk deletion, then restart-on-success single deletions.
    /// The final sweep is an explicit 1-minimality check, not a global minimum proof.
    /// Chunk strategy follows Zeller/Hildebrandt's ddmin complement reduction:
    /// <https://www.st.cs.uni-saarland.de/publications/files/zeller-tse-2002.pdf>.
    pub fn minimize(
        &self,
        witness: &FailureWitness,
        budget: MinimizeBudget,
    ) -> Result<MinimizeResult> {
        let start = Instant::now();
        let dem = witness.validate_structure()?;
        if dem != self.dem || witness.decoder != self.spec {
            return Err(invalid("minimization source decoder/DEM mismatch"));
        }
        if !witness.outcome.logical_failure {
            return Err(invalid("minimization requires a logical failure"));
        }
        if budget.decoder_calls == 0 || budget.time.is_zero() {
            return Ok(MinimizeResult {
                witness: incomplete_witness(witness, budget, 0, start.elapsed()),
                status: MinimizeStatus::Incomplete,
                decoder_calls: 0,
                elapsed: start.elapsed(),
            });
        }
        let mut calls = 0;
        let mut evaluate = |faults: &[usize]| -> Result<Option<WitnessOutcome>> {
            if calls >= budget.decoder_calls || start.elapsed() >= budget.time {
                return Ok(None);
            }
            calls += 1;
            Ok(Some(self.outcome(faults)?))
        };
        let Some(initial) = evaluate(&witness.faults)? else {
            return Ok(MinimizeResult {
                witness: incomplete_witness(witness, budget, calls, start.elapsed()),
                status: MinimizeStatus::Incomplete,
                decoder_calls: calls,
                elapsed: start.elapsed(),
            });
        };
        if initial != witness.outcome {
            return Err(invalid("source replay mismatch"));
        }
        let mut reduced = witness.clone();
        let mut complete = true;
        let mut chunks = 2;
        'chunks: while reduced.faults.len() >= 2 {
            let width = reduced.faults.len().div_ceil(chunks);
            for offset in (0..reduced.faults.len()).step_by(width) {
                let mut candidate = reduced.faults.clone();
                candidate.drain(offset..(offset + width).min(candidate.len()));
                let Some(outcome) = evaluate(&candidate)? else {
                    complete = false;
                    break 'chunks;
                };
                if outcome.logical_failure {
                    reduced.faults = candidate;
                    reduced.outcome = outcome;
                    chunks = 2;
                    continue 'chunks;
                }
            }
            if chunks >= reduced.faults.len() {
                break;
            }
            chunks = (chunks * 2).min(reduced.faults.len());
        }
        if complete {
            let mut i = 0;
            while i < reduced.faults.len() {
                let mut candidate = reduced.faults.clone();
                candidate.remove(i);
                let Some(outcome) = evaluate(&candidate)? else {
                    complete = false;
                    break;
                };
                if outcome.logical_failure {
                    reduced.faults = candidate;
                    reduced.outcome = outcome;
                    i = 0;
                } else {
                    i += 1;
                }
            }
        }
        let status = if complete {
            MinimizeStatus::OneMinimal
        } else {
            MinimizeStatus::Incomplete
        };
        let elapsed = start.elapsed();
        reduced.minimization = Some(MinimizationReceipt {
            status,
            call_budget: budget.decoder_calls,
            time_budget_ms: budget.time.as_millis(),
            decoder_calls: calls,
            elapsed_us: elapsed.as_micros(),
        });
        Ok(MinimizeResult {
            witness: reduced,
            status,
            decoder_calls: calls,
            elapsed,
        })
    }
}

fn incomplete_witness(
    witness: &FailureWitness,
    budget: MinimizeBudget,
    calls: u64,
    elapsed: Duration,
) -> FailureWitness {
    let mut w = witness.clone();
    w.minimization = Some(MinimizationReceipt {
        status: MinimizeStatus::Incomplete,
        call_budget: budget.decoder_calls,
        time_budget_ms: budget.time.as_millis(),
        decoder_calls: calls,
        elapsed_us: elapsed.as_micros(),
    });
    w
}

impl WitnessDecoder {
    /// Export a single block-layout golden shot through the existing `.syn`/v2 `.ref` codec.
    /// Requires the source fixed decoder, <=16 observables and <=32767 scheduled sweeps.
    /// Syndrome bit d is LE u32 word d/32, bit d%32, with zero padding. No windowing or
    /// hardware graph remapping is applied: the consumer must use this artifact's exact
    /// DEM ordering, priors and decoder configuration. This does not assert RTL support
    /// for arbitrary graphs; see the gross code-capacity example's supported layout.
    pub fn fixed_reference(&self, witness: &FailureWitness) -> Result<(Vec<u8>, Vec<u8>)> {
        if !matches!(self.spec, DecoderSpec::FixedRelay { .. }) || self.spec != witness.decoder {
            return Err(invalid("reference export requires source fixed decoder"));
        }
        let outcome = self.replay(witness)?;
        if outcome != witness.outcome || outcome.truth.len() > 16 {
            return Err(invalid("reference outcome mismatch or >16 observables"));
        }
        let (legs, iters) = match self.spec {
            DecoderSpec::FixedRelay { legs, iters, .. } => (legs, iters),
            _ => return Err(invalid("reference export requires fixed decoder")),
        };
        if legs as u64 * u64::from(iters) > 32767 {
            return Err(invalid("reference schedule exceeds 15-bit iteration field"));
        }
        let mut words = vec![0u32; self.dem.detectors.div_ceil(32)];
        for &d in &outcome.syndrome {
            words[d as usize / 32] |= 1 << (d % 32);
        }
        let syn = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mask = |bits: &[bool]| {
            bits.iter()
                .enumerate()
                .fold(0u16, |v, (i, &b)| v | (u16::from(b) << i))
        };
        let record = crate::RefRecord {
            true_obs: mask(&outcome.truth),
            sw_obs: mask(&outcome.prediction),
            valid: outcome.valid.ok_or_else(|| invalid("missing validity"))?,
            iters: outcome
                .first_valid_iter
                .ok_or_else(|| invalid("missing iteration diagnostic"))? as u16,
        };
        let mut reference = Vec::new();
        crate::write_ref(&mut reference, &[record]).map_err(|e| invalid(e.to_string()))?;
        Ok((syn, reference))
    }
}

pub(crate) fn sample_faults(dem: &DetectorErrorModel, seed: u64, shot: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(crate::experiment::splitmix64(seed, shot));
    dem.errors
        .iter()
        .enumerate()
        .filter_map(|(i, e)| (rng.gen::<f64>() < e.prob).then_some(i))
        .collect()
}

/// Internal constructor reuses the single scored outcome: exporting does not spend a second predicate call.
pub(crate) fn fault_count_witness(
    decoder: &WitnessDecoder,
    k: usize,
    enumerated: bool,
    seed: u64,
    shot: u64,
    faults: Vec<usize>,
    outcome: WitnessOutcome,
) -> FailureWitness {
    let dem = decoder.dem.to_dem_string();
    FailureWitness {
        version: 2,
        sampling: Some(if enumerated {
            WitnessSampling::FaultCountEnumerationV1 { faults: k }
        } else {
            WitnessSampling::FaultCountV1 { faults: k }
        }),
        dem_sha256: digest(dem.as_bytes()),
        dem,
        seed,
        shot,
        original_faults: faults.clone(),
        faults,
        decoder: decoder.spec.clone(),
        provenance: "dem-mechanisms".into(),
        original_outcome: outcome.clone(),
        outcome,
        minimization: None,
    }
}

impl Decoder for WitnessDecoder {
    fn decode(&self, syndrome: &Syndrome) -> crate::Correction {
        match &self.engine {
            Engine::Mwpm(d) => d.decode(syndrome),
            Engine::Uf(d) => d.decode(syndrome),
            Engine::Relay(d) => d.decode(syndrome),
            Engine::Fixed(d) => d.decode(syndrome),
        }
    }
}

// Reuse the scored outcome; capture incurs no hidden source-decoder call.
pub(crate) fn robustness_witness(
    decoder: &WitnessDecoder,
    generating: &DetectorErrorModel,
    process: &crate::NoiseProcess,
    options: &crate::RobustnessOptions,
    stream: u64,
    shot: &crate::NoiseShot,
    outcome: WitnessOutcome,
) -> Result<FailureWitness> {
    let dem = decoder.model().to_dem_string();
    Ok(FailureWitness {
        version: 3,
        sampling: Some(WitnessSampling::RobustnessV1 {
            generating_dem: generating.to_dem_string(),
            process_json: serde_json::to_string(process).map_err(|e| invalid(e.to_string()))?,
            campaign_seed: options.seed,
            stream,
            steps: options.steps,
            burst_active: shot.burst_active,
        }),
        dem_sha256: digest(dem.as_bytes()),
        dem,
        seed: crate::robustness_stream_seed(options.seed, stream),
        shot: shot.time,
        original_faults: shot.faults.clone(),
        faults: shot.faults.clone(),
        decoder: decoder.spec.clone(),
        provenance: "dem-mechanisms".into(),
        original_outcome: outcome.clone(),
        outcome,
        minimization: None,
    })
}
