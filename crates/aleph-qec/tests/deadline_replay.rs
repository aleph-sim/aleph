//! Hand-computed timelines, queue conservation and causal commit/flag semantics.
use aleph_qec::*;
fn decision(ns: u64, valid: bool, error: Option<bool>) -> TimedDecision {
    TimedDecision {
        service: ServiceTime::Nanoseconds { ns },
        valid,
        logical_failure: error,
    }
}
fn trace(arrivals: &[u64], cost: u64) -> DecodeTrace {
    DecodeTrace {
        version: 1,
        unit: TraceUnit::IndependentBlock,
        streaming_policy: None,
        origin: TraceOrigin::Synthetic,
        provenance: "hand-computed synthetic times/outcomes".into(),
        configuration: "independent test jobs".into(),
        input_sha256: "a".repeat(64),
        jobs: arrivals
            .iter()
            .enumerate()
            .map(|(i, &t)| DecodeTraceJob {
                id: i as u64,
                stream: 0,
                sequence: i as u64,
                arrival_ns: t,
                deadline_after_ns: None,
                predecessor: None,
                primary: decision(cost, true, Some(false)),
                fallback: None,
            })
            .collect(),
    }
}
fn config() -> ReplayConfig {
    ReplayConfig {
        workers: 1,
        queue_capacity: 10,
        fallback_workers: 1,
        fallback_queue_capacity: 10,
        policy: ReplayPolicy::FlagOnly,
        fallback_overflow: FallbackOverflowPolicy::DropJob,
        late_commit: LateCommitPolicy::CommitLate,
        operational_failure: OperationalFailurePolicy::LogicalOrLateOrMissing,
        input_overhead_ns: 0,
        output_overhead_ns: 0,
        fallback_transfer_ns: 0,
        overhead_provenance: "assumed zero synthetic transfer/host costs".into(),
    }
}
fn starts(r: &ReplayResult) -> Vec<Option<u64>> {
    r.jobs.iter().map(|j| j.primary_start_ns).collect()
}
#[test]
fn fifo_constant_service_matches_lindley_recursion_and_utilization() {
    let t = trace(&[0, 5, 10], 3);
    let r = replay_decode_trace(&t, &config()).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(5), Some(10)]);
    assert_eq!(r.horizon_ns, 13);
    assert!((r.utilization.unwrap() - 9.0 / 13.0).abs() < 1e-12);
    assert_eq!(r.primary_wait.unwrap().max_ns, 0);
    let t = trace(&[0, 2, 4, 6], 5);
    let r = replay_decode_trace(&t, &config()).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(5), Some(10), Some(15)]);
    assert_eq!(r.end_to_end.unwrap().max_ns, 14);
    assert_eq!(r.max_queue, 2);
    assert!((r.mean_queue.unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(r.committed, 4);
}
#[test]
fn finite_capacity_and_simultaneous_completion_precedes_arrival() {
    let mut c = config();
    c.queue_capacity = 0;
    let t = trace(&[0, 1, 5], 5);
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.overflows, 1);
    assert_eq!(starts(&r), vec![Some(0), None, Some(5)]);
    assert_eq!(r.jobs[1].disposition, ReplayDisposition::PrimaryOverflow);
    assert_eq!(r.operational_failures, 1);
    assert_eq!(r.decoded_known, 2);
    c.queue_capacity = 1;
    let t = trace(&[0, 0, 0], 5);
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(5), None]);
    assert_eq!(r.jobs.len(), r.committed + r.overflows);
}
#[test]
fn multiple_workers_keep_fifo_and_overheads_occupy_resources() {
    let mut c = config();
    c.workers = 2;
    c.input_overhead_ns = 1;
    c.output_overhead_ns = 2;
    let t = trace(&[0, 0, 0, 0], 5);
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(0), Some(8), Some(8)]);
    assert_eq!(r.horizon_ns, 16);
    assert_eq!(r.service.unwrap().p50_ns, 8);
    assert!((r.utilization.unwrap() - 1.0).abs() < 1e-12);
}
#[test]
fn fallback_contention_uses_its_own_fifo_and_same_time_resource_release() {
    let mut t = trace(&[0, 0, 0], 1);
    for j in &mut t.jobs {
        j.primary = decision(1, false, Some(true));
        j.fallback = Some(decision(10, true, Some(false)));
    }
    let mut c = config();
    c.workers = 3;
    c.policy = ReplayPolicy::FallbackOnFlag;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(
        r.jobs
            .iter()
            .map(|j| j.fallback_start_ns)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(11), Some(21)]
    );
    assert_eq!(r.jobs[2].terminal_ns, 31);
    assert_eq!(r.fallback_wait.unwrap().max_ns, 20);
    assert_eq!(r.flags, 3);
    assert_eq!(r.final_invalid, 0);
    assert_eq!(r.decoded_logical_failures, 0);
    assert_eq!(r.max_fallback_queue, 2);
    // A tail finishing at t=11 must release before another primary completion at t=11.
    t.jobs.truncate(2);
    t.jobs[1].arrival_ns = 10;
    c.workers = 1;
    c.fallback_queue_capacity = 0;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.fallback_overflows, 0);
    assert_eq!(r.jobs[1].fallback_start_ns, Some(11));
}
#[test]
fn fallback_overflow_and_missing_trace_never_invent_a_rescue() {
    let mut t = trace(&[0, 0], 1);
    for j in &mut t.jobs {
        j.primary = decision(1, false, Some(true));
        j.fallback = Some(decision(10, true, Some(false)));
    }
    let mut c = config();
    c.workers = 2;
    c.policy = ReplayPolicy::FallbackOnFlag;
    c.fallback_queue_capacity = 0;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.fallback_overflows, 1);
    assert_eq!(r.jobs[1].disposition, ReplayDisposition::FallbackOverflow);
    assert_eq!(r.decoded_known, 1);
    assert_eq!(r.operational_failures, 1);
    c.fallback_overflow = FallbackOverflowPolicy::KeepPrimary;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.committed, 2);
    assert_eq!(r.decoded_logical_failures, 1);
    assert_eq!(r.final_invalid, 1);
    t.jobs[1].fallback = None;
    assert!(replay_decode_trace(&t, &c).is_err());
}
#[test]
fn deadline_is_strict_and_combined_failure_is_a_union_not_a_sum() {
    let mut t = trace(&[0, 0], 5);
    for j in &mut t.jobs {
        j.deadline_after_ns = Some(5);
        j.primary.logical_failure = Some(true);
    }
    let r = replay_decode_trace(&t, &config()).unwrap();
    assert_eq!(r.deadline_misses, 1);
    assert_eq!(r.decoded_logical_failures, 2);
    assert_eq!(r.operational_failures, 2);
    assert!(!r.jobs[0].deadline_missed);
    let mut c = config();
    c.late_commit = LateCommitPolicy::DiscardLate;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.jobs[1].disposition, ReplayDisposition::DiscardedLate);
    assert_eq!(r.committed, 1);
    t.jobs[0].primary.logical_failure = None;
    t.jobs[1].primary.logical_failure = None;
    c.operational_failure = OperationalFailurePolicy::DecodedLogicalOnly;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.operational_unknown, 2);
    assert_eq!(r.operational_failures, 0);
    assert_eq!(r.decoded_known, 0);
}
#[test]
fn streaming_dependencies_wait_for_final_commit_and_cascade_after_missing_output() {
    let mut t = trace(&[0, 1, 2], 2);
    t.unit = TraceUnit::StreamingWindow;
    t.streaming_policy = Some(ReplayPolicy::FallbackOnFlag);
    t.jobs[1].predecessor = Some(0);
    t.jobs[2].predecessor = Some(1);
    t.jobs[0].primary.valid = false;
    t.jobs[0].fallback = Some(decision(8, true, Some(false)));
    let mut c = config();
    c.workers = 3;
    c.policy = ReplayPolicy::FallbackOnFlag;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(10), Some(12)]);
    t.jobs[0].deadline_after_ns = Some(5);
    c.late_commit = LateCommitPolicy::DiscardLate;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.dependency_cancellations, 2);
    assert_eq!(r.committed, 0);
    assert_eq!(r.jobs[1].primary_start_ns, None);
    assert_eq!(r.operational_failures, 3);
    t.jobs[2].predecessor = None;
    assert!(t.validate().is_err());
}
#[test]
fn trace_roundtrip_cycle_conversion_empty_zero_and_overflow_cases() {
    assert_eq!(
        ServiceTime::Cycles {
            cycles: 1,
            clock_hz: 133_000_000
        }
        .nanoseconds()
        .unwrap(),
        8
    );
    assert!(ServiceTime::Cycles {
        cycles: 1,
        clock_hz: 0
    }
    .nanoseconds()
    .is_err());
    assert!(ServiceTime::Cycles {
        cycles: u64::MAX,
        clock_hz: 1
    }
    .nanoseconds()
    .is_err());
    let t = trace(&[0, 0, 0], 0);
    assert_eq!(DecodeTrace::from_json(&t.to_json().unwrap()).unwrap(), t);
    let mut c = config();
    c.queue_capacity = 0;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.committed, 3);
    assert_eq!(r.utilization, None);
    assert_eq!(r.horizon_ns, 0);
    let t = trace(&[], 1);
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.committed, 0);
    assert!(r.end_to_end.is_none());
    let t = trace(&[u64::MAX], 1);
    assert!(replay_decode_trace(&t, &c).is_err());
    let mut t = trace(&[0, 1], 1);
    t.jobs[1].id = 0;
    assert!(t.validate().is_err());
}
#[test]
fn late_missing_and_unobserved_flags_remain_distinct() {
    let mut t = trace(&[0, 0, 0], 10);
    for j in &mut t.jobs {
        j.primary.valid = false;
        j.primary.logical_failure = Some(false);
        j.deadline_after_ns = Some(1);
    }
    let mut c = config();
    c.queue_capacity = 0;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.flags, 1);
    assert_eq!(r.overflows, 2);
    assert_eq!(r.deadline_misses, 1);
    assert_eq!(r.decoded_logical_failures, 0);
    assert_eq!(r.operational_failures, 3);
    c.operational_failure = OperationalFailurePolicy::DecodedLogicalOnly;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.operational_failures, 0);
    assert_eq!(r.operational_unknown, 2);
}

#[test]
fn older_fallback_waiter_dispatches_before_new_same_time_flag() {
    let mut t = trace(&[0, 0, 10], 1);
    for j in &mut t.jobs {
        j.primary.valid = false;
        j.fallback = Some(decision(10, true, Some(false)));
    }
    let mut c = config();
    c.workers = 2;
    c.policy = ReplayPolicy::FallbackOnFlag;
    c.fallback_queue_capacity = 1;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.fallback_overflows, 0);
    assert_eq!(
        r.jobs
            .iter()
            .map(|j| j.fallback_start_ns)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(11), Some(21)]
    );
}
#[test]
fn blocked_fifo_head_does_not_reorder_another_stream() {
    let mut t = trace(&[0, 1, 2], 2);
    t.jobs[0].primary.valid = false;
    t.jobs[0].fallback = Some(decision(10, true, Some(false)));
    t.jobs[1].predecessor = Some(0);
    t.jobs[2].stream = 1;
    t.jobs[2].sequence = 0;
    let mut c = config();
    c.workers = 2;
    c.policy = ReplayPolicy::FallbackOnFlag;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(starts(&r), vec![Some(0), Some(12), Some(12)]);
}
#[test]
fn zero_time_primary_and_tail_release_without_artificial_queue_pressure() {
    let mut t = trace(&[0, 0, 0], 0);
    for j in &mut t.jobs {
        j.primary.valid = false;
        j.fallback = Some(decision(0, true, Some(false)));
    }
    let mut c = config();
    c.policy = ReplayPolicy::FallbackOnFlag;
    c.queue_capacity = 0;
    c.fallback_queue_capacity = 0;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.committed, 3);
    assert_eq!(r.fallback_overflows, 0);
    assert_eq!(r.horizon_ns, 0);
}

#[test]
fn dependent_traces_reject_policy_switch_and_cancel_changed_fallback_history() {
    let mut t = trace(&[0, 0, 2], 1);
    t.unit = TraceUnit::StreamingWindow;
    t.streaming_policy = Some(ReplayPolicy::FallbackOnFlag);
    t.jobs[1].stream = 1;
    t.jobs[1].sequence = 0;
    t.jobs[2].stream = 1;
    t.jobs[2].sequence = 1;
    t.jobs[2].predecessor = Some(1);
    for j in &mut t.jobs[..2] {
        j.primary.valid = false;
        j.fallback = Some(decision(20, true, Some(false)));
    }
    let mut c = config();
    assert!(replay_decode_trace(&t, &c).is_err());
    c.policy = ReplayPolicy::FallbackOnFlag;
    c.workers = 2;
    c.fallback_queue_capacity = 0;
    c.fallback_overflow = FallbackOverflowPolicy::KeepPrimary;
    let r = replay_decode_trace(&t, &c).unwrap();
    assert_eq!(r.jobs[1].disposition, ReplayDisposition::Committed);
    assert!(!r.jobs[1].commit_satisfies_dependency);
    assert_eq!(r.jobs[2].disposition, ReplayDisposition::DependencyCanceled);
}
