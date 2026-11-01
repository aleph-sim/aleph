//! Replay cost observations; synthetic traces, no decoder/RTL throughput claim.
use aleph_qec::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
fn bench(c: &mut Criterion) {
    let cfg = ReplayConfig {
        workers: 2,
        queue_capacity: 16,
        fallback_workers: 1,
        fallback_queue_capacity: 8,
        policy: ReplayPolicy::FallbackOnFlag,
        fallback_overflow: FallbackOverflowPolicy::KeepPrimary,
        late_commit: LateCommitPolicy::CommitLate,
        operational_failure: OperationalFailurePolicy::LogicalOrLateOrMissing,
        input_overhead_ns: 0,
        output_overhead_ns: 0,
        fallback_transfer_ns: 0,
        overhead_provenance: "Synthetic zero overhead".into(),
    };
    for n in [512, 4096] {
        let trace = DecodeTrace {
            version: 1,
            unit: TraceUnit::IndependentBlock,
            streaming_policy: None,
            origin: TraceOrigin::Synthetic,
            provenance: "Synthetic replay-engine benchmark".into(),
            configuration: "two primary workers, one fallback, flags every16 jobs".into(),
            input_sha256: "a".repeat(64),
            jobs: (0..n)
                .map(|i| DecodeTraceJob {
                    id: i,
                    stream: 0,
                    sequence: i,
                    arrival_ns: 50 * i,
                    deadline_after_ns: Some(500),
                    predecessor: None,
                    primary: TimedDecision {
                        service: ServiceTime::Nanoseconds { ns: 100 },
                        valid: i % 16 != 0,
                        logical_failure: Some(i % 32 == 0),
                    },
                    fallback: Some(TimedDecision {
                        service: ServiceTime::Nanoseconds { ns: 800 },
                        valid: true,
                        logical_failure: Some(false),
                    }),
                })
                .collect(),
        };
        c.bench_function(&format!("deadline_replay/{n}_jobs"), |b| {
            b.iter(|| black_box(replay_decode_trace(&trace, &cfg).unwrap()))
        });
    }
}
fn config() -> Criterion {
    let c = Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    match std::env::var_os("ALEPH_BENCH_OUTPUT") {
        Some(p) => c.output_directory(&std::path::PathBuf::from(p)),
        None => c,
    }
}
criterion_group! {name=benches;config=config();targets=bench}
criterion_main!(benches);
