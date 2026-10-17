//! Fixed-budget fault-count stratification of independent original DEM mechanisms.
//! Positive interval DP bounds masses; log-suffix DP conditions without probability underflow.
//! Conditional Bernoulli background: Chen & Liu (1997), Statistica Sinica 7, 875–892:
//! https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n44/j7n44.htm
//! Here the sequential conditional update follows directly from the suffix recurrence below.
use crate::{
    BinomialInterval95, DetectorErrorModel, Error, FailureWitness, Result, WitnessDecoder,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::time::Instant;

fn error(s: impl Into<String>) -> Error {
    Error::RareEvent(s.into())
}
/// Hard numeric-workspace limit used also for witness verification (32 MB of f64 cells).
pub const MAX_CONDITIONAL_CELLS: usize = 4_000_000;
/// Enclosing probability bounds, including outward rounding of positive DP arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbabilityBounds {
    /// Conservative lower bound.
    pub lower: f64,
    /// Conservative upper bound.
    pub upper: f64,
}
impl ProbabilityBounds {
    fn exact(p: f64) -> Self {
        Self { lower: p, upper: p }
    }
    fn plus(self, other: Self) -> Self {
        if self.upper == 0.0 {
            return other;
        }
        if other.upper == 0.0 {
            return self;
        }
        Self {
            lower: (self.lower + other.lower).next_down().max(0.0),
            upper: (self.upper + other.upper).next_up().min(1.0),
        }
    }
    fn times(self, other: Self) -> Self {
        if self.upper == 0.0 || other.upper == 0.0 {
            return Self::exact(0.0);
        }
        Self {
            lower: (self.lower * other.lower).next_down().max(0.0),
            upper: (self.upper * other.upper).next_up().min(1.0),
        }
    }
    fn complement(p: f64) -> Self {
        Self {
            lower: (1.0 - p).next_down().max(0.0),
            upper: (1.0 - p).next_up().min(1.0),
        }
    }
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let hi = a.max(b);
    hi + (-(a - b).abs()).exp().ln_1p()
}

/// Immutable original model with a resource-bounded truncated Poisson-binomial suffix table.
/// p=0 mechanisms never fire; p=1 mechanisms are forced and included in total fault counts.
/// Equal-probability models use the same DP (which reduces to binomial weights).
pub struct FaultCountPlan {
    dem: DetectorErrorModel,
    forced: Vec<usize>,
    variables: Vec<(usize, f64)>,
    width: usize,
    max_faults: usize,
    suffix: Vec<f64>,
    weights: Vec<ProbabilityBounds>,
    tail: ProbabilityBounds,
}
impl FaultCountPlan {
    /// Build through max_faults, enforcing both the caller's cell limit and a 4M-cell ceiling.
    /// The cell budget includes the log table, positive mass rows and stratum bounds;
    /// it excludes the owned DEM and mechanism-index metadata.
    /// Rejects max_faults below the forced count or above the number of mechanisms.
    ///
    /// ```
    /// use aleph_qec::{DetectorErrorModel, FaultCountPlan};
    /// let dem=DetectorErrorModel::parse("error(0.01) D0 L0\nerror(0.2) D1")?;
    /// let plan=FaultCountPlan::new(&dem,2,100)?;
    /// assert_eq!(plan.sample(1,544,0)?.len(),1);
    /// assert!(plan.weight(1)?.lower > 0.2);
    /// # Ok::<(),aleph_qec::Error>(())
    /// ```
    pub fn new(dem: &DetectorErrorModel, max_faults: usize, max_cells: usize) -> Result<Self> {
        crate::witness::validate_dem(dem)?;
        let forced: Vec<_> = dem
            .errors
            .iter()
            .enumerate()
            .filter_map(|(i, e)| (e.prob == 1.0).then_some(i))
            .collect();
        let variables: Vec<_> = dem
            .errors
            .iter()
            .enumerate()
            .filter_map(|(i, e)| (e.prob > 0.0 && e.prob < 1.0).then_some((i, e.prob)))
            .collect();
        if max_faults < forced.len() || max_faults > dem.errors.len() {
            return Err(error(
                "max_faults must include forced faults and be <= mechanism count",
            ));
        }
        let degree = (max_faults - forced.len()).min(variables.len());
        let width = degree + 1;
        let cells = (variables.len() + 1)
            .checked_mul(width)
            .ok_or_else(|| error("DP size overflow"))?;
        let numeric_cells = (max_faults + 1)
            .checked_mul(2)
            .and_then(|v| v.checked_add(width * 2))
            .and_then(|v| v.checked_add(cells))
            .ok_or_else(|| error("numeric workspace size overflow"))?;
        if numeric_cells > max_cells.min(MAX_CONDITIONAL_CELLS) {
            return Err(error("conditional DP cell budget exceeded"));
        }
        let mut suffix = vec![f64::NEG_INFINITY; cells];
        suffix[variables.len() * width] = 0.0;
        // P(K_i=r)=q_i P(K_{i+1}=r)+p_i P(K_{i+1}=r-1), in log space.
        for i in (0..variables.len()).rev() {
            let p = variables[i].1;
            let lq = (-p).ln_1p();
            let lp = p.ln();
            for r in 0..width {
                suffix[i * width + r] = log_add(
                    lq + suffix[(i + 1) * width + r],
                    if r > 0 {
                        lp + suffix[(i + 1) * width + r - 1]
                    } else {
                        f64::NEG_INFINITY
                    },
                );
            }
        }
        // Separate positive DP avoids subtracting a near-one CDF to obtain a tiny tail.
        let mut mass = vec![ProbabilityBounds::exact(0.0); width];
        mass[0] = ProbabilityBounds::exact(1.0);
        let mut tail = ProbabilityBounds::exact(0.0);
        for &(_, p) in &variables {
            let take = ProbabilityBounds::exact(p);
            let skip = ProbabilityBounds::complement(p);
            tail = tail.plus(mass[degree].times(take));
            for r in (0..width).rev() {
                mass[r] = mass[r].times(skip).plus(if r > 0 {
                    mass[r - 1].times(take)
                } else {
                    ProbabilityBounds::exact(0.0)
                });
            }
        }
        let mut weights = vec![ProbabilityBounds::exact(0.0); max_faults + 1];
        weights[forced.len()..forced.len() + width].copy_from_slice(&mass);
        Ok(Self {
            dem: dem.clone(),
            forced,
            variables,
            width,
            max_faults,
            suffix,
            weights,
            tail,
        })
    }
    /// Original DEM used for both sampling and immutable decoder priors.
    pub fn model(&self) -> &DetectorErrorModel {
        &self.dem
    }
    /// Enclosing unconditional P(K=k); out-of-plan counts are rejected.
    pub fn weight(&self, k: usize) -> Result<ProbabilityBounds> {
        self.weights
            .get(k)
            .copied()
            .ok_or_else(|| error("stratum outside plan"))
    }
    /// Natural log P(K=k), retained even when the linear probability underflows.
    /// None denotes an impossible stratum, not an underflowed positive mass.
    pub fn log_weight(&self, k: usize) -> Option<f64> {
        k.checked_sub(self.forced.len())
            .filter(|&r| r < self.width)
            .map(|r| self.suffix[r])
            .filter(|v| v.is_finite())
    }
    /// Enclosing omitted P(K>max_faults), accumulated directly rather than by cancellation.
    pub fn tail(&self) -> ProbabilityBounds {
        self.tail
    }
    /// Draw P(F | |F|=k) using original heterogeneous priors and a log-suffix recurrence.
    /// Seed identity depends only on (seed,k,absolute sample), independent of scheduling.
    /// "Exact conditional" means no importance proposal/uniform-subset approximation;
    /// f64 transcendental evaluation and finite random-bit resolution still apply.
    pub fn sample(&self, k: usize, seed: u64, sample: u64) -> Result<Vec<usize>> {
        if self.log_weight(k).is_none() {
            return Err(error("cannot sample an impossible/out-of-plan stratum"));
        }
        let mut remaining = k - self.forced.len();
        let mut faults = self.forced.clone();
        let mixed = crate::experiment::splitmix64(seed ^ 0xA544_5EED_0147_CAFE, k as u64);
        let mut rng = StdRng::seed_from_u64(crate::experiment::splitmix64(mixed, sample));
        for (i, &(index, p)) in self.variables.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            let a = p.ln() + self.suffix[(i + 1) * self.width + remaining - 1];
            let b = (-p).ln_1p() + self.suffix[(i + 1) * self.width + remaining];
            let prob = if a == f64::NEG_INFINITY {
                0.0
            } else if b == f64::NEG_INFINITY {
                1.0
            } else if a >= b {
                1.0 / (1.0 + (b - a).exp())
            } else {
                let x = (a - b).exp();
                x / (1.0 + x)
            };
            if !prob.is_finite() || !(0.0..=1.0).contains(&prob) {
                return Err(error("conditional probability evaluation failed"));
            }
            if rng.gen_bool(prob) {
                faults.push(index);
                remaining -= 1;
            }
        }
        if remaining != 0 {
            return Err(error("conditional sample did not reach requested count"));
        }
        faults.sort_unstable();
        Ok(faults)
    }
    /// Lexicographic rank within the possible k-fault sets (forced faults included).
    pub fn enumerated_faults(&self, k: usize, mut rank: u64) -> Result<Vec<usize>> {
        if self.log_weight(k).is_none() {
            return Err(error("impossible/out-of-plan enumeration"));
        }
        let mut r = k - self.forced.len();
        let n = self.variables.len();
        if count_capped(n, r, rank).is_some() {
            return Err(error("enumeration rank out of range"));
        }
        let mut faults = self.forced.clone();
        for i in 0..n {
            if r == 0 {
                break;
            }
            match count_capped(n - i - 1, r - 1, rank) {
                None => {
                    faults.push(self.variables[i].0);
                    r -= 1;
                }
                Some(c) if rank < c => {
                    faults.push(self.variables[i].0);
                    r -= 1;
                }
                Some(c) => rank -= c,
            }
        }
        if r != 0 || rank != 0 {
            return Err(error("invalid enumeration rank"));
        }
        faults.sort_unstable();
        Ok(faults)
    }
    fn fault_mass(&self, faults: &[usize]) -> ProbabilityBounds {
        self.variables
            .iter()
            .fold(ProbabilityBounds::exact(1.0), |mass, &(i, p)| {
                mass.times(if faults.binary_search(&i).is_ok() {
                    ProbabilityBounds::exact(p)
                } else {
                    ProbabilityBounds::complement(p)
                })
            })
    }
}
// Return the exact combination count iff <=cap; stop early before large products.
fn count_capped(n: usize, k: usize, cap: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut count = 1u128;
    for j in 0..k {
        count = count * (n - j) as u128 / (j + 1) as u128;
        if count > u128::from(cap) {
            return None;
        }
    }
    if count > u128::from(cap) {
        None
    } else {
        Some(count as u64)
    }
}

/// Predeclared per-stratum call budgets; no output-dependent allocation or early stopping.
#[derive(Clone, Debug)]
pub struct StratifiedOptions {
    /// Indexed by total k, with exactly max_faults+1 entries. Zero leaves a stratum unsampled.
    pub budgets: Vec<u64>,
    /// Total call limit, checked against the sum of declared budgets before any decode.
    pub max_decoder_calls: u64,
    /// Reproducible campaign seed.
    pub seed: u64,
    /// Simultaneous confidence level (strictly between 0 and 1).
    pub confidence: f64,
    /// Enumerate all possible sets if their count fits the predeclared stratum budget.
    pub enumerate_small: bool,
    /// Bounded number of failed-shot witnesses to retain. Capture reuses the scored outcome.
    pub witness_limit: usize,
    /// Optional requested final interval width; assessed only after the fixed campaign finishes.
    pub target_width: Option<f64>,
}
/// How a stratum was evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StratumMode {
    /// All possible sets, correctly weighted even for unequal priors.
    ExactEnumeration,
    /// Independent conditional draws with a binomial interval.
    ConditionalMonteCarlo,
    /// Positive mass, no decoder calls.
    Unsampled,
    /// No possible fault set.
    Impossible,
}
/// One stratum's complete accounting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratumResult {
    /// Total original mechanism count.
    pub faults: usize,
    /// Enclosing unconditional mass.
    pub weight: ProbabilityBounds,
    /// Log mass, or None for impossible rows.
    pub log_weight: Option<f64>,
    /// Predeclared budget.
    pub requested_calls: u64,
    /// Evaluation mode.
    pub mode: StratumMode,
    /// Actual decoder predicate calls, also the number of draws/sets evaluated.
    pub decoder_calls: u64,
    /// Calls with logical mismatch (not non-convergence alone).
    pub failures: u64,
    /// Available invalid/nonconverged diagnostic count.
    pub nonconverged: u64,
    /// Conditional failure bounds (0..1 for an unsampled possible row).
    pub conditional: ProbabilityBounds,
    /// Bounds on this row's unconditional failure contribution.
    pub contribution: ProbabilityBounds,
    /// Nominal evaluated failure mass; never includes an assumed-zero tail implicitly.
    pub failure_mass_estimate: f64,
}
/// Fixed-budget completion, with optional precision assessment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StratifiedStatus {
    /// No target was requested.
    FixedBudgetComplete,
    /// Requested width was reached after the declared campaign.
    TargetMet,
    /// Declared campaign completed but its bound is wider than requested.
    BudgetExhausted,
}
/// Stratified estimate with simultaneous intervals and explicit unobserved mass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StratifiedResult {
    /// SHA-256 of the canonical original decoder DEM (including priors).
    pub dem_sha256: String,
    /// Fixed target decoder settings and disorder seed.
    pub decoder: crate::DecoderSpec,
    /// Conditional generator identity; ordinary experiment RNG is not reused implicitly.
    pub sampler: String,
    /// Sum of the predeclared per-stratum call budgets; exact enumeration may spend less.
    pub declared_decoder_calls: u64,
    /// Requested interval width, assessed only at the fixed endpoint.
    pub target_width: Option<f64>,
    /// Fixed campaign seed.
    pub seed: u64,
    /// Simultaneous confidence level.
    pub confidence: f64,
    /// Interval construction; exact rows need no confidence allocation.
    pub interval_method: String,
    /// Row-level statistics, including impossible/unsampled rows.
    pub strata: Vec<StratumResult>,
    /// Omitted mass from uncalled rows plus all K above the plan.
    pub unsampled_mass: ProbabilityBounds,
    /// Failure-probability bounds including worst-case failure on omitted mass.
    pub failure_probability: ProbabilityBounds,
    /// Contribution estimated from evaluated rows only; not a full-tail point estimate.
    pub sampled_failure_mass_estimate: f64,
    /// Sum of actual predicate calls. Witness construction adds no decoder calls.
    pub decoder_calls: u64,
    /// Assessment at the fixed endpoint.
    pub status: StratifiedStatus,
    /// Run wall time, excluding prebuilt plan/decoder construction and subsequent artifact import.
    pub elapsed_seconds: f64,
    /// Failed-shot P7-01 artifacts with conditional/enumeration provenance.
    pub witnesses: Vec<FailureWitness>,
}

/// Evaluate immutable target priors at a predeclared endpoint. Uses Bonferroni-adjusted
/// Clopper–Pearson intervals for conditional Bernoulli draws; combines mass bounds outward.
/// Enumeration contributes exact weighted fault-set sums up to enclosed numerical rounding.
pub fn run_stratified(
    plan: &FaultCountPlan,
    decoder: &WitnessDecoder,
    options: &StratifiedOptions,
) -> Result<StratifiedResult> {
    let start = Instant::now();
    if plan.model() != decoder.model() {
        return Err(error("sampling model and decoder priors differ"));
    }
    let (declared, modes, alpha) = prepare(plan, options)?;
    let mut strata = Vec::new();
    let mut witnesses = Vec::new();
    let mut total_calls = 0;
    let mut unsampled = plan.tail();
    let mut total = ProbabilityBounds::exact(0.0);
    let mut estimate = 0.0;
    for (k, &mode) in modes.iter().enumerate() {
        let budget = options.budgets[k];
        let weight = plan.weight(k)?;
        let calls = match mode {
            StratumMode::ExactEnumeration => {
                count_capped(plan.variables.len(), k - plan.forced.len(), budget)
                    .ok_or_else(|| error("enumeration count changed"))?
            }
            StratumMode::ConditionalMonteCarlo => budget,
            _ => 0,
        };
        let mut failures = 0;
        let mut nonconverged = 0;
        let mut exact_mass = ProbabilityBounds::exact(0.0);
        for sample in 0..calls {
            let faults = if mode == StratumMode::ExactEnumeration {
                plan.enumerated_faults(k, sample)?
            } else {
                plan.sample(k, options.seed, sample)?
            };
            let outcome = decoder.decode_faults(&faults)?;
            total_calls += 1;
            nonconverged += u64::from(outcome.valid == Some(false));
            if outcome.logical_failure {
                failures += 1;
                if mode == StratumMode::ExactEnumeration {
                    exact_mass = exact_mass.plus(plan.fault_mass(&faults));
                }
                if witnesses.len() < options.witness_limit {
                    witnesses.push(crate::witness::fault_count_witness(
                        decoder,
                        k,
                        mode == StratumMode::ExactEnumeration,
                        options.seed,
                        sample,
                        faults,
                        outcome,
                    ));
                }
            }
        }
        let conditional = match mode {
            StratumMode::ExactEnumeration => ProbabilityBounds {
                lower: if weight.upper > 0.0 {
                    (exact_mass.lower / weight.upper).next_down().max(0.0)
                } else {
                    0.0
                },
                upper: if weight.lower > 0.0 {
                    (exact_mass.upper / weight.lower).next_up().min(1.0)
                } else {
                    1.0
                },
            },
            StratumMode::ConditionalMonteCarlo => {
                let b = BinomialInterval95::from_counts_alpha(calls, failures, alpha)?;
                ProbabilityBounds {
                    lower: b.lower,
                    upper: b.upper,
                }
            }
            StratumMode::Unsampled => ProbabilityBounds {
                lower: 0.0,
                upper: 1.0,
            },
            StratumMode::Impossible => ProbabilityBounds::exact(0.0),
        };
        let contribution = if mode == StratumMode::ExactEnumeration {
            exact_mass
        } else {
            weight.times(conditional)
        };
        let point = if mode == StratumMode::ExactEnumeration {
            exact_mass.lower + (exact_mass.upper - exact_mass.lower) * 0.5
        } else if calls > 0 {
            plan.log_weight(k).map_or(0.0, f64::exp) * failures as f64 / calls as f64
        } else {
            0.0
        };
        if mode == StratumMode::Unsampled {
            unsampled = unsampled.plus(weight);
        }
        total = total.plus(contribution);
        estimate += point;
        strata.push(StratumResult {
            faults: k,
            weight,
            log_weight: plan.log_weight(k),
            requested_calls: budget,
            mode,
            decoder_calls: calls,
            failures,
            nonconverged,
            conditional,
            contribution,
            failure_mass_estimate: point,
        });
    }
    total.upper = total.plus(plan.tail()).upper;
    let status = match options.target_width {
        None => StratifiedStatus::FixedBudgetComplete,
        Some(width) if total.upper - total.lower <= width => StratifiedStatus::TargetMet,
        Some(_) => StratifiedStatus::BudgetExhausted,
    };
    let interval_method = "Bonferroni simultaneous Clopper-Pearson; exact weighted enumeration; outward mass bounds; omitted mass [0,w_upper]".into();
    Ok(StratifiedResult {
        dem_sha256: crate::witness::digest(decoder.model().to_dem_string().as_bytes()),
        decoder: decoder.configuration().clone(),
        sampler: "fault-count-log-suffix-v1 / rand-0.8 StdRng".into(),
        declared_decoder_calls: declared,
        target_width: options.target_width,
        seed: options.seed,
        confidence: options.confidence,
        interval_method,
        strata,
        unsampled_mass: unsampled,
        failure_probability: total,
        sampled_failure_mass_estimate: estimate.min(1.0),
        decoder_calls: total_calls,
        status,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        witnesses,
    })
}
/// Capture one conditional draw as a P7-01-compatible v2 witness (one source predicate call).
pub fn capture_stratified_shot(
    plan: &FaultCountPlan,
    decoder: &WitnessDecoder,
    k: usize,
    seed: u64,
    sample: u64,
) -> Result<FailureWitness> {
    if plan.model() != decoder.model() {
        return Err(error("sampling/decoder model mismatch"));
    }
    let faults = plan.sample(k, seed, sample)?;
    let outcome = decoder.decode_faults(&faults)?;
    Ok(crate::witness::fault_count_witness(
        decoder, k, false, seed, sample, faults, outcome,
    ))
}

fn prepare(
    plan: &FaultCountPlan,
    options: &StratifiedOptions,
) -> Result<(u64, Vec<StratumMode>, f64)> {
    if options.budgets.len() != plan.max_faults + 1
        || !options.confidence.is_finite()
        || options.confidence <= 0.0
        || options.confidence >= 1.0
        || options
            .target_width
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
    {
        return Err(error("invalid budgets/confidence/precision target"));
    }
    let declared = options
        .budgets
        .iter()
        .try_fold(0u64, |a, &b| a.checked_add(b))
        .ok_or_else(|| error("call budget overflow"))?;
    if declared > options.max_decoder_calls {
        return Err(error("declared budgets exceed decoder call limit"));
    }
    let mut modes = Vec::new();
    for (k, &budget) in options.budgets.iter().enumerate() {
        let possible = plan.log_weight(k).is_some();
        let mode = if !possible {
            if budget > 0 {
                return Err(error("positive budget for impossible stratum"));
            }
            StratumMode::Impossible
        } else if budget == 0 {
            StratumMode::Unsampled
        } else if options.enumerate_small
            && count_capped(plan.variables.len(), k - plan.forced.len(), budget).is_some()
        {
            StratumMode::ExactEnumeration
        } else {
            if budget >= (1u64 << 53) {
                return Err(error(
                    "Monte Carlo budget exceeds representable binomial counts",
                ));
            }
            StratumMode::ConditionalMonteCarlo
        };
        modes.push(mode);
    }
    let stochastic = modes
        .iter()
        .filter(|&&m| m == StratumMode::ConditionalMonteCarlo)
        .count();
    let alpha = (1.0 - options.confidence) / stochastic.max(1) as f64;
    // Validate confidence representability before spending decoder calls.
    BinomialInterval95::from_counts_alpha(1, 0, alpha)?;
    Ok((declared, modes, alpha))
}

/// Method chosen before any stochastic observations or decoder calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RareEventMethod {
    /// Ordinary independent DEM Monte Carlo.
    Direct,
    /// Fault-count conditional sampling with exact small strata.
    Stratified,
}
/// Deterministic central-count width proxies, not guarantees of observed precision or runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RareEventDecision {
    /// Selected method; stratification requires a factor-two score advantage.
    pub method: RareEventMethod,
    /// Shared declared call envelope. Exact strata may spend fewer calls.
    pub declared_decoder_calls: u64,
    /// Direct Clopper–Pearson width at floor(N/2) failures.
    pub direct_width_score: f64,
    /// Weighted central-count row widths plus all unobserved mass.
    pub stratified_width_score: f64,
}
/// Select without a pilot, random outcomes, or assumptions about correctable fault counts.
/// Exact rows have only numerical uncertainty; unsampled rows and tails contribute their
/// full mass. Unknown conditional failure rates can make this heuristic choose poorly.
/// The factor-two margin is fixed, not fitted to observations. Planning cost is not scored.
pub fn select_rare_event_method(
    plan: &FaultCountPlan,
    options: &StratifiedOptions,
) -> Result<RareEventDecision> {
    let (declared, modes, alpha) = prepare(plan, options)?;
    if declared >= (1u64 << 53) {
        return Err(error(
            "automatic direct budget exceeds representable binomial counts",
        ));
    }
    let central_width = |n, a| -> Result<f64> {
        let ci = BinomialInterval95::from_counts_alpha(n, n / 2, a)?;
        Ok(ci.upper - ci.lower)
    };
    let direct_width_score = central_width(declared, 1.0 - options.confidence)?;
    let mut score = plan.tail().upper;
    for (k, mode) in modes.iter().enumerate() {
        let weight = plan.weight(k)?;
        score += match mode {
            StratumMode::Impossible => 0.0,
            StratumMode::Unsampled => weight.upper,
            StratumMode::ExactEnumeration => weight.upper - weight.lower,
            StratumMode::ConditionalMonteCarlo => {
                weight.upper * central_width(options.budgets[k], alpha)? + weight.upper
                    - weight.lower
            }
        };
    }
    let stratified_width_score = score.min(1.0);
    Ok(RareEventDecision {
        method: if stratified_width_score <= 0.5 * direct_width_score {
            RareEventMethod::Stratified
        } else {
            RareEventMethod::Direct
        },
        declared_decoder_calls: declared,
        direct_width_score,
        stratified_width_score,
    })
}
/// Fixed-endpoint automatic run. Direct mode emits no witness artifacts; `witness_limit`
/// applies only to stratified mode. No outcome-dependent switching takes place.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutomaticRareEventResult {
    /// Pre-observation decision and scores.
    pub decision: RareEventDecision,
    /// Confidence bounds using the selected method at the requested confidence level.
    pub failure_probability: ProbabilityBounds,
    /// Actual calls, excluding planning (which makes no decoder calls).
    pub decoder_calls: u64,
    /// Precision assessment at the fixed endpoint.
    pub status: StratifiedStatus,
    /// Detailed stratification/witness receipts, present only when selected.
    pub stratified: Option<Box<StratifiedResult>>,
    /// Ordinary Monte Carlo failure tally, present only when direct is selected.
    pub direct_failures: Option<u64>,
}
/// Opt-in dispatcher. The existing direct harness and explicit stratified API remain available.
/// Selection uses only immutable model masses and predeclared budgets. The direct branch
/// spends the declared envelope; exact stratification can spend less. Actual intervals,
/// rather than planning scores, determine target completion.
pub fn run_automatic_rare_event(
    plan: &FaultCountPlan,
    decoder: &WitnessDecoder,
    options: &StratifiedOptions,
) -> Result<AutomaticRareEventResult> {
    if plan.model() != decoder.model() {
        return Err(error("sampling model and decoder priors differ"));
    }
    let decision = select_rare_event_method(plan, options)?;
    if decision.method == RareEventMethod::Stratified {
        let result = run_stratified(plan, decoder, options)?;
        return Ok(AutomaticRareEventResult {
            failure_probability: result.failure_probability,
            decoder_calls: result.decoder_calls,
            status: result.status,
            stratified: Some(Box::new(result)),
            direct_failures: None,
            decision,
        });
    }
    let result = crate::run_dem_experiment(
        plan.model(),
        decision.declared_decoder_calls,
        decoder,
        options.seed,
    )?;
    let ci = BinomialInterval95::from_counts_alpha(
        result.shots,
        result.logical_errors,
        1.0 - options.confidence,
    )?;
    let bounds = ProbabilityBounds {
        lower: ci.lower,
        upper: ci.upper,
    };
    let status = match options.target_width {
        None => StratifiedStatus::FixedBudgetComplete,
        Some(width) if bounds.upper - bounds.lower <= width => StratifiedStatus::TargetMet,
        Some(_) => StratifiedStatus::BudgetExhausted,
    };
    Ok(AutomaticRareEventResult {
        decision,
        failure_probability: bounds,
        decoder_calls: result.shots,
        status,
        stratified: None,
        direct_failures: Some(result.logical_errors),
    })
}
