//! Deterministic FIFO trace replay, not a closed-loop quantum/controller simulator.
//! Whole timing/outcome tuples stay ordered; no independent resampling or inferred silicon time.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
fn invalid(s: &str) -> Error {
    Error::TraceReplay(s.into())
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("time/counter overflow"))
}
/// Source of service/outcome pairs; host golden timing is not silicon timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceOrigin {
    /// Deliberately constructed workload.
    Synthetic,
    /// Existing decoder golden implementation timed on a host.
    GoldenModelHost,
    /// Per-job board measurements with explicit provenance.
    MeasuredSilicon,
}
/// Scheduling/commit unit of one record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceUnit {
    /// Independently generated complete blocks; parallelism is allowed.
    IndependentBlock,
    /// Sequential commits; every later job requires its prior commit.
    StreamingWindow,
}
/// Service-only cost; transfer and interface costs are explicit replay configuration fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ServiceTime {
    /// Measured or synthetic nanoseconds, as identified by trace provenance.
    Nanoseconds {
        /// Service time in ns (zero is supported).
        ns: u64,
    },
    /// Convert with ceiling(cycles*1e9/clock_hz), rejecting zero clock or overflow.
    Cycles {
        /// Kernel cycles.
        cycles: u64,
        /// Actual/assumed clock; provenance must identify which.
        clock_hz: u64,
    },
}
impl ServiceTime {
    /// Convert without floating-point truncation.
    pub fn nanoseconds(self) -> Result<u64> {
        match self {
            Self::Nanoseconds { ns } => Ok(ns),
            Self::Cycles { cycles, clock_hz } => {
                if clock_hz == 0 {
                    return Err(invalid("zero cycle clock"));
                }
                let ns = (u128::from(cycles) * 1_000_000_000).div_ceil(u128::from(clock_hz));
                u64::try_from(ns).map_err(|_| invalid("cycle time overflow"))
            }
        }
    }
}
/// A measured policy-specific service and outcome tuple.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimedDecision {
    /// Kernel/host service cost, excluding replay-supplied transfer costs.
    pub service: ServiceTime,
    /// Syndrome validity, separate from logical correctness.
    pub valid: bool,
    /// Observed logical mismatch; None means genuinely unavailable, never assumed success.
    pub logical_failure: Option<bool>,
}
/// One ordered arrival, with optional prerequisite correction commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeTraceJob {
    /// Unique stable input/round identity.
    pub id: u64,
    /// Independent stream identity.
    pub stream: u64,
    /// Consecutive sequence inside this stream, starting at zero.
    pub sequence: u64,
    /// Nondecreasing arrival timestamp in ns; equal-time order is vector order.
    pub arrival_ns: u64,
    /// Relative correction deadline; None explicitly disables deadline testing.
    pub deadline_after_ns: Option<u64>,
    /// Earlier same-stream job that must commit before this job starts.
    pub predecessor: Option<u64>,
    /// Actual primary policy timing and output.
    pub primary: TimedDecision,
    /// Conditional fallback output/time from this same input, only where measured.
    pub fallback: Option<TimedDecision>,
}
/// Versioned trace. Immutable ordered tuples preserve timing/error/flag correlations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeTrace {
    /// Schema version; currently one.
    pub version: u32,
    /// Job meaning, including any window/commit lengths.
    pub unit: TraceUnit,
    /// Captured correction/fallback policy for dependent windows; None for independent blocks.
    pub streaming_policy: Option<ReplayPolicy>,
    /// Timing/outcome origin.
    pub origin: TraceOrigin,
    /// Exact source, platform, clock assumptions and unmeasured components.
    pub provenance: String,
    /// Code/window configuration, noise/seed and actual primary/fallback policy settings.
    pub configuration: String,
    /// Input fingerprint shared across policy-specific traces; not a claim of independent jobs.
    pub input_sha256: String,
    /// Ordered original tuples, at most 100000.
    pub jobs: Vec<DecodeTraceJob>,
}
impl DecodeTrace {
    /// Validate ordered identities, dependency chains, clocks and time ranges.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.jobs.len() > 100_000
            || self.provenance.trim().is_empty()
            || self.configuration.trim().is_empty()
            || self.input_sha256.len() != 64
            || !self.input_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err(invalid(
                "invalid trace version, metadata, fingerprint or size",
            ));
        }
        if (self.unit == TraceUnit::StreamingWindow) != self.streaming_policy.is_some() {
            return Err(invalid("dependent-window trace must declare its captured policy; independent blocks must not"));
        }
        let mut ids = HashMap::new();
        let mut streams = HashMap::<u64, (u64, u64)>::new();
        let mut previous_arrival = 0;
        for (i, j) in self.jobs.iter().enumerate() {
            if j.arrival_ns < previous_arrival || ids.contains_key(&j.id) {
                return Err(invalid("arrivals must be ordered and job IDs unique"));
            }
            match streams.get(&j.stream) {
                None if j.sequence != 0 || j.predecessor.is_some() => {
                    return Err(invalid(
                        "first stream job must have sequence zero and no predecessor",
                    ))
                }
                Some(&(seq, id))
                    if seq.checked_add(1) != Some(j.sequence)
                        || j.predecessor.is_some_and(|p| p != id)
                        || (self.unit == TraceUnit::StreamingWindow
                            && j.predecessor != Some(id)) =>
                {
                    return Err(invalid("invalid sequence or commit dependency"))
                }
                _ => (),
            }
            j.primary.service.nanoseconds()?;
            if let Some(f) = &j.fallback {
                f.service.nanoseconds()?;
            }
            if let Some(d) = j.deadline_after_ns {
                add(j.arrival_ns, d)?;
            }
            ids.insert(j.id, i);
            streams.insert(j.stream, (j.sequence, j.id));
            previous_arrival = j.arrival_ns;
        }
        Ok(())
    }
    /// Serialize validated trace data.
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| Error::TraceReplay(e.to_string()))
    }
    /// Import without silently accepting malformed units, dependencies or clocks.
    pub fn from_json(text: &str) -> Result<Self> {
        let t: Self = serde_json::from_str(text).map_err(|e| Error::TraceReplay(e.to_string()))?;
        t.validate()?;
        Ok(t)
    }
}
/// Primary output is emitted even when flagged, or an actually measured conditional tail runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayPolicy {
    /// No fallback; preserve actual primary policy's outcome.
    FlagOnly,
    /// Run measured fallback only on invalid primary decisions.
    FallbackOnFlag,
}
/// What happens if the separate fallback waiting queue is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackOverflowPolicy {
    /// Commit the flagged primary decision.
    KeepPrimary,
    /// No correction is committed for this job.
    DropJob,
}
/// Late work always runs to completion (no unmeasured preempt/cancel benefit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LateCommitPolicy {
    /// Commit even after deadline, allowing dependent work to start.
    CommitLate,
    /// Discard completed late output; dependents cancel when considered at FIFO head.
    DiscardLate,
}
/// Explicit policy-dependent operational metric, not a physical logical-error assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalFailurePolicy {
    /// Only observed decoded logical mismatches; missing outcomes are unknown.
    DecodedLogicalOnly,
    /// Union of mismatch, late or missing committed correction, counted once per job.
    LogicalOrLateOrMissing,
}
/// Finite waiting queues exclude executing jobs. Zero capacity means immediate service or drop.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    /// Primary resources; 1..=1024.
    pub workers: usize,
    /// Primary waiting slots, 0..=100000.
    pub queue_capacity: usize,
    /// Conditional tail resources; positive even if unused.
    pub fallback_workers: usize,
    /// Separate fallback waiting slots.
    pub fallback_queue_capacity: usize,
    /// Flag-only or conditional measured fallback.
    pub policy: ReplayPolicy,
    /// Fallback queue overflow handling.
    pub fallback_overflow: FallbackOverflowPolicy,
    /// Commit/discard policy for late completed corrections.
    pub late_commit: LateCommitPolicy,
    /// How the optional combined operational count is interpreted.
    pub operational_failure: OperationalFailurePolicy,
    /// Ingress cost occupying the primary resource, in ns.
    pub input_overhead_ns: u64,
    /// Egress cost occupying each primary/fallback service resource, in ns.
    pub output_overhead_ns: u64,
    /// Transfer/dispatch cost occupying the fallback resource, in ns.
    pub fallback_transfer_ns: u64,
    /// Origin of supplied overheads, including explicitly assumed zero components.
    pub overhead_provenance: String,
}
/// Terminal job disposition. Drops and dependency cancellation are not guessed logical errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayDisposition {
    /// Correction became available after final service.
    Committed,
    /// Final output completed late and was discarded.
    DiscardedLate,
    /// Primary waiting capacity was exhausted.
    PrimaryOverflow,
    /// Conditional fallback capacity exhausted under DropJob policy.
    FallbackOverflow,
    /// Required earlier correction never committed.
    DependencyCanceled,
}
/// Per-job deterministic event receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayJobResult {
    /// Original ID.
    pub id: u64,
    /// Original arrival timestamp.
    pub arrival_ns: u64,
    /// Primary service start, absent for unstarted jobs.
    pub primary_start_ns: Option<u64>,
    /// Primary completion/flag availability.
    pub primary_end_ns: Option<u64>,
    /// Conditional tail start, including its transfer overhead.
    pub fallback_start_ns: Option<u64>,
    /// Terminal event, including drop/cancel/commit.
    pub terminal_ns: u64,
    /// Terminal classification.
    pub disposition: ReplayDisposition,
    /// Primary invalid flag was observed (only after primary completion).
    pub flagged: bool,
    /// Fallback overflow happened, including KeepPrimary cases.
    pub fallback_overflow: bool,
    /// Final service completed after the relative deadline; dropped work is separate.
    pub deadline_missed: bool,
    /// Final decoded output mismatch where known, including late discarded output.
    pub decoded_logical_failure: Option<bool>,
    /// Validity of the final decoded output, absent for uncompleted/dropped work.
    pub final_valid: Option<bool>,
    /// A committed correction with history compatible with this trace, permitting dependent jobs.
    pub commit_satisfies_dependency: bool,
}
/// Nearest-rank statistics (ceil(p*n)), absent for an empty population.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// Population size.
    pub count: usize,
    /// Median ns.
    pub p50_ns: u64,
    /// 95th percentile ns.
    pub p95_ns: u64,
    /// 99th percentile ns.
    pub p99_ns: u64,
    /// Maximum ns.
    pub max_ns: u64,
}
fn percentiles(mut v: Vec<u64>) -> Option<LatencyPercentiles> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let n = v.len();
    let at = |p: usize| v[(p * n).div_ceil(100).saturating_sub(1)];
    Some(LatencyPercentiles {
        count: n,
        p50_ns: at(50),
        p95_ns: at(95),
        p99_ns: at(99),
        max_ns: v[n - 1],
    })
}
/// Queue occupancy after each stable event timestamp; in-service work is counted separately.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayQueueLevel {
    /// Timestamp ns.
    pub time_ns: u64,
    /// Waiting primary jobs.
    pub primary_waiting: usize,
    /// Waiting conditional tail jobs.
    pub fallback_waiting: usize,
    /// Busy primary resources.
    pub primary_running: usize,
    /// Busy tail resources.
    pub fallback_running: usize,
}
/// Complete fixed-trace replay, retaining distinct decoded and operational outcomes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayResult {
    /// Input trace canonical serialized SHA256, including tuple order and arrivals.
    pub trace_sha256: String,
    /// Exact scheduling assumptions.
    pub config: ReplayConfig,
    /// Per-job receipts in original input order.
    pub jobs: Vec<ReplayJobResult>,
    /// Stable event trajectory (not resampled).
    pub queue_levels: Vec<ReplayQueueLevel>,
    /// Final event minus first arrival; zero for empty/instantaneous work.
    pub horizon_ns: u64,
    /// Waiting before primary start, population = jobs that actually started primary service.
    pub primary_wait: Option<LatencyPercentiles>,
    /// Waiting for fallback after the primary completed, population = started tail jobs.
    pub fallback_wait: Option<LatencyPercentiles>,
    /// Sum of executing service including declared overhead, for jobs reaching a final decision.
    pub service: Option<LatencyPercentiles>,
    /// Arrival-to-final-output time for completed decisions, including discarded late outputs.
    pub end_to_end: Option<LatencyPercentiles>,
    /// Time-weighted primary waiting occupancy, absent for zero horizon.
    pub mean_queue: Option<f64>,
    /// Maximum observed waiting occupancy (including admission-time peaks).
    pub max_queue: usize,
    /// Maximum separate fallback waiting occupancy.
    pub max_fallback_queue: usize,
    /// Fraction of available primary resource time busy; absent for zero horizon.
    pub utilization: Option<f64>,
    /// Separate fallback busy fraction.
    pub fallback_utilization: Option<f64>,
    /// Committed corrections per second; absent for zero horizon.
    pub committed_throughput_per_second: Option<f64>,
    /// Primary decisions observed invalid.
    pub flags: usize,
    /// Completed final decisions that missed their deadline.
    pub deadline_misses: usize,
    /// Primary overflows.
    pub overflows: usize,
    /// Tail queue overflows, including KeepPrimary decisions.
    pub fallback_overflows: usize,
    /// Jobs canceled because their prerequisite never committed.
    pub dependency_cancellations: usize,
    /// Committed corrections.
    pub committed: usize,
    /// Known decoded logical mismatches, not necessarily applied physical corrections.
    pub decoded_logical_failures: usize,
    /// Jobs with known final decoded logical outcome (denominator for decoded error rate).
    pub decoded_known: usize,
    /// Final decoded outputs still invalid after the selected policy.
    pub final_invalid: usize,
    /// Policy-dependent union count; a late logical failure is counted once.
    pub operational_failures: usize,
    /// Unknown outcomes under that operational policy, never silently counted as successes.
    pub operational_unknown: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    New,
    Waiting,
    Primary,
    TailWaiting,
    Tail,
    Done,
}
struct Engine<'a> {
    trace: &'a DecodeTrace,
    config: &'a ReplayConfig,
    states: Vec<State>,
    rows: Vec<ReplayJobResult>,
    dep: Vec<Option<usize>>,
    primary: VecDeque<usize>,
    tail: VecDeque<usize>,
    events: BinaryHeap<Reverse<(u64, u8, usize)>>,
    busy: usize,
    tail_busy: usize,
    busy_ns: u64,
    tail_busy_ns: u64,
    max_queue: usize,
    max_tail: usize,
}
impl Engine<'_> {
    fn ready(&self, i: usize) -> bool {
        self.dep[i].is_none_or(|p| {
            self.states[p] == State::Done && self.rows[p].commit_satisfies_dependency
        })
    }
    fn broken(&self, i: usize) -> bool {
        self.dep[i].is_some_and(|p| {
            self.states[p] == State::Done && !self.rows[p].commit_satisfies_dependency
        })
    }
    fn drop_job(&mut self, i: usize, t: u64, disposition: ReplayDisposition) {
        self.states[i] = State::Done;
        self.rows[i].terminal_ns = t;
        self.rows[i].disposition = disposition;
    }
    fn primary_cost(&self, i: usize) -> Result<u64> {
        add(
            add(
                self.trace.jobs[i].primary.service.nanoseconds()?,
                self.config.input_overhead_ns,
            )?,
            self.config.output_overhead_ns,
        )
    }
    fn tail_cost(&self, i: usize) -> Result<u64> {
        let f = self.trace.jobs[i]
            .fallback
            .as_ref()
            .ok_or_else(|| invalid("missing conditional fallback timing/outcome"))?;
        add(
            add(f.service.nanoseconds()?, self.config.fallback_transfer_ns)?,
            self.config.output_overhead_ns,
        )
    }
    fn start(&mut self, i: usize, t: u64, tail: bool) -> Result<()> {
        let cost = if tail {
            self.tail_cost(i)?
        } else {
            self.primary_cost(i)?
        };
        let end = add(t, cost)?;
        if tail {
            self.states[i] = State::Tail;
            self.tail_busy += 1;
            self.tail_busy_ns = add(self.tail_busy_ns, cost)?;
            self.rows[i].fallback_start_ns = Some(t);
        } else {
            self.states[i] = State::Primary;
            self.busy += 1;
            self.busy_ns = add(self.busy_ns, cost)?;
            self.rows[i].primary_start_ns = Some(t);
        }
        if end == t {
            if tail {
                self.tail_busy -= 1;
            } else {
                self.busy -= 1;
            }
            self.complete(t, u8::from(tail), i)?;
        } else {
            self.events.push(Reverse((end, u8::from(tail), i)));
        }
        Ok(())
    }
    fn finish(&mut self, i: usize, t: u64, tail: bool) -> Result<()> {
        let decision = if tail {
            self.trace.jobs[i]
                .fallback
                .as_ref()
                .ok_or_else(|| invalid("fallback disappeared"))?
        } else {
            &self.trace.jobs[i].primary
        };
        let missed = self.trace.jobs[i]
            .deadline_after_ns
            .is_some_and(|d| t - self.trace.jobs[i].arrival_ns > d);
        self.rows[i].deadline_missed = missed;
        self.rows[i].decoded_logical_failure = decision.logical_failure;
        self.rows[i].final_valid = Some(decision.valid);
        let disposition = if missed && self.config.late_commit == LateCommitPolicy::DiscardLate {
            ReplayDisposition::DiscardedLate
        } else {
            ReplayDisposition::Committed
        };
        self.rows[i].commit_satisfies_dependency = disposition == ReplayDisposition::Committed
            && !(self.trace.unit == TraceUnit::StreamingWindow && self.rows[i].fallback_overflow);
        self.drop_job(i, t, disposition);
        Ok(())
    }
    fn complete(&mut self, t: u64, stage: u8, i: usize) -> Result<()> {
        if stage == 1 {
            return self.finish(i, t, true);
        }
        self.rows[i].primary_end_ns = Some(t);
        self.rows[i].flagged = !self.trace.jobs[i].primary.valid;
        if self.config.policy == ReplayPolicy::FallbackOnFlag && self.rows[i].flagged {
            if self.tail_busy < self.config.fallback_workers && self.tail.is_empty() {
                self.start(i, t, true)?;
            } else if self.tail.len() < self.config.fallback_queue_capacity {
                self.states[i] = State::TailWaiting;
                self.tail.push_back(i);
                self.max_tail = self.max_tail.max(self.tail.len());
            } else {
                self.rows[i].fallback_overflow = true;
                if self.config.fallback_overflow == FallbackOverflowPolicy::KeepPrimary {
                    self.finish(i, t, false)?;
                } else {
                    self.drop_job(i, t, ReplayDisposition::FallbackOverflow);
                }
            }
        } else {
            self.finish(i, t, false)?;
        }
        Ok(())
    }
    fn dispatch_tail(&mut self, t: u64) -> Result<()> {
        while self.tail_busy < self.config.fallback_workers {
            if let Some(i) = self.tail.pop_front() {
                self.start(i, t, true)?;
            } else {
                break;
            }
        }
        Ok(())
    }
    fn dispatch(&mut self, t: u64) -> Result<()> {
        self.dispatch_tail(t)?;
        while let Some(&i) = self.primary.front() {
            if self.broken(i) {
                self.primary.pop_front();
                self.drop_job(i, t, ReplayDisposition::DependencyCanceled);
                continue;
            }
            if self.busy >= self.config.workers || !self.ready(i) {
                break;
            }
            self.primary.pop_front();
            self.start(i, t, false)?;
        }
        Ok(())
    }
    fn settle(&mut self, t: u64) -> Result<()> {
        loop {
            let mut completed = Vec::new();
            while self.events.peek().is_some_and(|e| e.0 .0 == t) {
                let Reverse((_, stage, i)) =
                    self.events.pop().ok_or_else(|| invalid("event vanished"))?;
                if stage == 1 {
                    self.tail_busy -= 1;
                } else {
                    self.busy -= 1;
                }
                completed.push((stage, i));
            }
            // All resources finishing now are released before capacity decisions. Older tail
            // waiters keep FIFO priority over newly flagged primary completions.
            self.dispatch_tail(t)?;
            for (stage, i) in completed {
                self.complete(t, stage, i)?;
            }
            self.dispatch(t)?;
            if !self.events.peek().is_some_and(|e| e.0 .0 == t) {
                break;
            }
        }
        Ok(())
    }
    fn admit(&mut self, i: usize, t: u64) -> Result<()> {
        if self.broken(i) {
            self.drop_job(i, t, ReplayDisposition::DependencyCanceled);
        } else if self.primary.is_empty() && self.busy < self.config.workers && self.ready(i) {
            self.start(i, t, false)?;
        } else if self.primary.len() < self.config.queue_capacity {
            self.states[i] = State::Waiting;
            self.primary.push_back(i);
            self.max_queue = self.max_queue.max(self.primary.len());
        } else {
            self.drop_job(i, t, ReplayDisposition::PrimaryOverflow);
        }
        Ok(())
    }
}
/// Replay all ordered work to terminal disposition. At ties, completions precede arrivals;
/// primary completions precede fallback completions, then original vector index breaks ties.
/// Existing FIFO waiters dispatch before new arrivals. Newly created zero-time events settle
/// before the next same-time arrival. A blocked head is not bypassed; dependencies can cause
/// head-of-line blocking across streams. Running work is never preempted at a deadline.
///
/// ```no_run
/// use aleph_qec::{DecodeTrace, replay_decode_trace};
/// # fn demo(json:&str, config:&aleph_qec::ReplayConfig)->aleph_qec::Result<()> {
/// let trace=DecodeTrace::from_json(json)?;
/// let result=replay_decode_trace(&trace,config)?;
/// assert_eq!(result.jobs.len(),trace.jobs.len());
/// # Ok(()) }
/// ```
pub fn replay_decode_trace(trace: &DecodeTrace, config: &ReplayConfig) -> Result<ReplayResult> {
    trace.validate()?;
    if trace
        .streaming_policy
        .is_some_and(|policy| policy != config.policy)
    {
        return Err(invalid(
            "dependent-window policy differs from captured correction history",
        ));
    }
    if config.workers == 0
        || config.workers > 1024
        || config.fallback_workers == 0
        || config.fallback_workers > 1024
        || config.queue_capacity > 100_000
        || config.fallback_queue_capacity > 100_000
        || config.overhead_provenance.trim().is_empty()
    {
        return Err(invalid("invalid resources or overhead provenance"));
    }
    let n = trace.jobs.len();
    let ids: HashMap<_, _> = trace
        .jobs
        .iter()
        .enumerate()
        .map(|(i, j)| (j.id, i))
        .collect();
    let dep = trace
        .jobs
        .iter()
        .map(|j| j.predecessor.and_then(|p| ids.get(&p).copied()))
        .collect();
    let rows = trace
        .jobs
        .iter()
        .map(|j| ReplayJobResult {
            id: j.id,
            arrival_ns: j.arrival_ns,
            primary_start_ns: None,
            primary_end_ns: None,
            fallback_start_ns: None,
            terminal_ns: j.arrival_ns,
            disposition: ReplayDisposition::DependencyCanceled,
            flagged: false,
            fallback_overflow: false,
            deadline_missed: false,
            decoded_logical_failure: None,
            final_valid: None,
            commit_satisfies_dependency: false,
        })
        .collect();
    let mut e = Engine {
        trace,
        config,
        states: vec![State::New; n],
        rows,
        dep,
        primary: VecDeque::new(),
        tail: VecDeque::new(),
        events: BinaryHeap::new(),
        busy: 0,
        tail_busy: 0,
        busy_ns: 0,
        tail_busy_ns: 0,
        max_queue: 0,
        max_tail: 0,
    };
    for (i, job) in trace.jobs.iter().enumerate() {
        e.primary_cost(i)?;
        if config.policy == ReplayPolicy::FallbackOnFlag && !job.primary.valid {
            e.tail_cost(i)?;
        }
    }
    let first = trace.jobs.first().map_or(0, |j| j.arrival_ns);
    let mut previous = first;
    let mut index = 0;
    let mut levels = Vec::new();
    let mut area = 0.0;
    while index < n || !e.events.is_empty() {
        let t = match (trace.jobs.get(index), e.events.peek()) {
            (Some(j), Some(ev)) => j.arrival_ns.min(ev.0 .0),
            (Some(j), None) => j.arrival_ns,
            (None, Some(ev)) => ev.0 .0,
            _ => break,
        };
        area += (t - previous) as f64 * e.primary.len() as f64;
        previous = t;
        e.settle(t)?;
        while index < n && trace.jobs[index].arrival_ns == t {
            e.admit(index, t)?;
            index += 1;
            e.settle(t)?;
        }
        levels.push(ReplayQueueLevel {
            time_ns: t,
            primary_waiting: e.primary.len(),
            fallback_waiting: e.tail.len(),
            primary_running: e.busy,
            fallback_running: e.tail_busy,
        });
    }
    if e.states.iter().any(|s| *s != State::Done) {
        return Err(invalid("unresolved dependency/queue at replay end"));
    }
    let horizon = previous - first;
    let denom = horizon as f64;
    let flags = e.rows.iter().filter(|r| r.flagged).count();
    let committed = e
        .rows
        .iter()
        .filter(|r| r.disposition == ReplayDisposition::Committed)
        .count();
    let mut operational_failures = 0;
    let mut operational_unknown = 0;
    for r in &e.rows {
        let outcome = if config.operational_failure
            == OperationalFailurePolicy::LogicalOrLateOrMissing
            && (r.disposition != ReplayDisposition::Committed || r.deadline_missed)
        {
            Some(true)
        } else {
            r.decoded_logical_failure
        };
        match outcome {
            Some(true) => operational_failures += 1,
            None => operational_unknown += 1,
            _ => (),
        }
    }
    let final_rows: Vec<_> = e
        .rows
        .iter()
        .filter(|r| {
            matches!(
                r.disposition,
                ReplayDisposition::Committed | ReplayDisposition::DiscardedLate
            )
        })
        .collect();
    let mut service = Vec::new();
    for r in &final_rows {
        let i = ids[&r.id];
        let cost = if r.fallback_start_ns.is_some() {
            add(e.primary_cost(i)?, e.tail_cost(i)?)?
        } else {
            e.primary_cost(i)?
        };
        service.push(cost);
    }
    let end_to_end = percentiles(
        final_rows
            .iter()
            .map(|r| r.terminal_ns - r.arrival_ns)
            .collect(),
    );
    let primary_wait = percentiles(
        e.rows
            .iter()
            .filter_map(|r| r.primary_start_ns.map(|s| s - r.arrival_ns))
            .collect(),
    );
    let fallback_wait = percentiles(
        e.rows
            .iter()
            .filter_map(|r| {
                r.fallback_start_ns
                    .zip(r.primary_end_ns)
                    .map(|(s, p)| s - p)
            })
            .collect(),
    );
    Ok(ReplayResult {
        trace_sha256: crate::witness::digest(trace.to_json()?.as_bytes()),
        config: config.clone(),
        flags,
        deadline_misses: e.rows.iter().filter(|r| r.deadline_missed).count(),
        overflows: e
            .rows
            .iter()
            .filter(|r| r.disposition == ReplayDisposition::PrimaryOverflow)
            .count(),
        fallback_overflows: e.rows.iter().filter(|r| r.fallback_overflow).count(),
        dependency_cancellations: e
            .rows
            .iter()
            .filter(|r| r.disposition == ReplayDisposition::DependencyCanceled)
            .count(),
        committed,
        decoded_logical_failures: e
            .rows
            .iter()
            .filter(|r| r.decoded_logical_failure == Some(true))
            .count(),
        decoded_known: e
            .rows
            .iter()
            .filter(|r| r.decoded_logical_failure.is_some())
            .count(),
        final_invalid: e
            .rows
            .iter()
            .filter(|r| r.final_valid == Some(false))
            .count(),
        operational_failures,
        operational_unknown,
        jobs: e.rows,
        queue_levels: levels,
        horizon_ns: horizon,
        primary_wait,
        fallback_wait,
        service: percentiles(service),
        end_to_end,
        mean_queue: if horizon > 0 {
            Some(area / denom)
        } else {
            None
        },
        max_queue: e.max_queue,
        max_fallback_queue: e.max_tail,
        utilization: if horizon > 0 {
            Some(e.busy_ns as f64 / (denom * config.workers as f64))
        } else {
            None
        },
        fallback_utilization: if horizon > 0 {
            Some(e.tail_busy_ns as f64 / (denom * config.fallback_workers as f64))
        } else {
            None
        },
        committed_throughput_per_second: if horizon > 0 {
            Some(committed as f64 * 1e9 / denom)
        } else {
            None
        },
    })
}
