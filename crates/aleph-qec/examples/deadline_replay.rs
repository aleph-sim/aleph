//! Q7-07 golden adapter and fixed ordered queue/deadline campaigns.
//! cargo run --release -p aleph-qec --example deadline_replay -- campaign OUTDIR [shots=2048]
//! cargo run --release -p aleph-qec --example deadline_replay -- replay TRACE CONFIG OUT.json
use aleph_qec::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, path::Path, time::Instant};
type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;
fn ns(start: Instant) -> AnyResult<u64> {
    Ok(u64::try_from(start.elapsed().as_nanos())?.max(1))
}
fn decision(ns: u64, valid: bool, error: bool) -> TimedDecision {
    TimedDecision {
        service: ServiceTime::Nanoseconds { ns },
        valid,
        logical_failure: Some(error),
    }
}
fn config(workers: usize, policy: ReplayPolicy) -> ReplayConfig {
    ReplayConfig{workers,queue_capacity:32,fallback_workers:1,fallback_queue_capacity:8,policy,fallback_overflow:FallbackOverflowPolicy::KeepPrimary,late_commit:LateCommitPolicy::CommitLate,operational_failure:OperationalFailurePolicy::LogicalOrLateOrMissing,input_overhead_ns:1000,output_overhead_ns:1000,fallback_transfer_ns:10000,overhead_provenance:"SENSITIVITY ASSUMPTIONS: 1us ingress, 1us egress per service, 10us fallback transfer; no board/host-interface timing is inferred. Kernel/OSD service separately measured on host.".into()}
}
fn golden(shots: u64, out: &Path) -> AnyResult<Vec<DecodeTrace>> {
    if shots == 0 || shots > 100_000 {
        return Err("shots outside 1..=100000".into());
    }
    let dem = BBCode::gross().circuit_level_dem(1, CircuitNoise::uniform(0.007))?;
    let (syndromes, truths) = sample_shots(&dem, shots, 2024);
    let raw = serde_json::to_vec(
        &json!({"syndromes":syndromes.iter().map(|s|&s.fired).collect::<Vec<_>>(),"truths":truths}),
    )?;
    let input_sha256 = format!("{:x}", Sha256::digest(&raw));
    let osd = OsdDecoder::new(&dem).with_order(0);
    let mut traces = Vec::new();
    for early in [false, true] {
        let fx = FixedRelayBp::with_budget(&dem, 6, 10, (-0.3, 0.9), 0x5E1A_4B9C, 8, 3)
            .with_early_exit(early);
        let mut jobs = Vec::new();
        for (i, (syn, truth)) in syndromes.iter().zip(&truths).enumerate() {
            let start = Instant::now();
            let soft = fx.decode_fixed_soft(syn);
            let correction = fx.correction_of(&soft.ehat);
            let primary_ns = ns(start)?;
            let primary = decision(
                primary_ns,
                soft.converged,
                correction.observable_flips != *truth,
            );
            let fallback = if !soft.converged {
                let start = Instant::now();
                let ehat = osd.ehat_from_soft(syn, &soft);
                let valid = osd.check_satisfied(syn, &ehat);
                let correction = fx.correction_of(&ehat);
                let cost = ns(start)?;
                Some(decision(cost, valid, correction.observable_flips != *truth))
            } else {
                None
            };
            jobs.push(DecodeTraceJob {
                id: i as u64,
                stream: 0,
                sequence: i as u64,
                arrival_ns: i as u64,
                deadline_after_ns: None,
                predecessor: None,
                primary,
                fallback,
            });
        }
        let name = if early {
            "golden-early-exit"
        } else {
            "golden-fixed-budget"
        };
        let trace=DecodeTrace{version:1,unit:TraceUnit::IndependentBlock,streaming_policy:None,origin:TraceOrigin::GoldenModelHost,provenance:format!("Q7-07 adapter: native FixedRelayBp golden, measured host Instant durations on {} {}; chronological unshuffled input prefix from sample_shots(seed=2024). Full/early policies measured separately. Not measured silicon. No transfer timing measured.",std::env::consts::OS,std::env::consts::ARCH),configuration:format!("Gross [[144,12,12]], circuit-level rounds=1 p=.007; FixedRelayBp 6x10, msg_bits=8 frac_bits=3, gamma=(-.3,.9), disorder seed=0x5E1A4B9C, early_exit={early}; conditional host OSD order0 consuming this same primary soft output"),input_sha256:input_sha256.clone(),jobs};
        std::fs::write(out.join(format!("{name}-trace.json")), trace.to_json()?)?;
        traces.push(trace);
    }
    Ok(traces)
}
fn synthetic(name: &str) -> DecodeTrace {
    let mut jobs = Vec::new();
    for i in 0..512u64 {
        let episode = (128..144).contains(&i);
        let spike = i % 128 == 127;
        let (cost, flagged, error) = match name {
            "rare-spikes" if spike => (5000, true, true),
            "correlated-flags" if episode => (100, true, i % 2 == 0),
            _ => (100, false, false),
        };
        jobs.push(DecodeTraceJob {
            id: i,
            stream: 0,
            sequence: i,
            arrival_ns: 160 * i,
            deadline_after_ns: Some(500),
            predecessor: None,
            primary: decision(cost, !flagged, error),
            fallback: if flagged {
                Some(decision(2000, true, false))
            } else {
                None
            },
        });
    }
    DecodeTrace{version:1,unit:TraceUnit::IndependentBlock,streaming_policy:None,origin:TraceOrigin::Synthetic,provenance:"Constructed correlated tuples: 100ns baseline, rare 5000ns flagged failures every128 jobs, or contiguous 16 flagged jobs with actual synthetic 2000ns successful fallback. No measured hardware claim.".into(),configuration:name.into(),input_sha256:format!("{:x}",Sha256::digest(name.as_bytes())),jobs}
}
fn compact(name: &str, result: &ReplayResult, trace_file: &str) -> Value {
    json!({"name":name,"trace_file":trace_file,"trace_sha256":result.trace_sha256,"config":result.config,"submitted":result.jobs.len(),"horizon_ns":result.horizon_ns,"primary_wait":result.primary_wait,"fallback_wait":result.fallback_wait,"service":result.service,"end_to_end":result.end_to_end,"mean_queue":result.mean_queue,"max_queue":result.max_queue,"max_fallback_queue":result.max_fallback_queue,"utilization":result.utilization,"fallback_utilization":result.fallback_utilization,"committed_throughput_per_second":result.committed_throughput_per_second,"committed":result.committed,"flags":result.flags,"final_invalid":result.final_invalid,"deadline_misses":result.deadline_misses,"overflows":result.overflows,"fallback_overflows":result.fallback_overflows,"dependency_cancellations":result.dependency_cancellations,"decoded_logical_failures":result.decoded_logical_failures,"decoded_known":result.decoded_known,"operational_failures":result.operational_failures,"operational_unknown":result.operational_unknown})
}
fn campaign(out: &Path, shots: u64) -> AnyResult<()> {
    std::fs::create_dir_all(out)?;
    let traces = golden(shots, out)?;
    // Same arrival-rate/deadline sweep for both measured policies, from the declared early
    // trace mean service. Never shuffle service times away from flags or logical outcomes.
    let mean = traces[1]
        .jobs
        .iter()
        .map(|j| j.primary.service.nanoseconds())
        .collect::<Result<Vec<_>>>()?
        .iter()
        .map(|&x| u128::from(x))
        .sum::<u128>()
        / u128::from(shots);
    let base = u64::try_from(mean)?.max(1);
    let mut cases = Vec::new();
    for (profile, original) in traces.iter().enumerate() {
        let file = if profile == 0 {
            "golden-fixed-budget-trace.json"
        } else {
            "golden-early-exit-trace.json"
        };
        for (load, period) in [
            ("low", base.saturating_mul(2)),
            ("near", base),
            ("high", (base / 2).max(1)),
        ] {
            let mut trace = original.clone();
            for (i, j) in trace.jobs.iter_mut().enumerate() {
                j.arrival_ns = (i as u64).checked_mul(period).ok_or("arrival overflow")?;
                j.deadline_after_ns = Some(period.checked_mul(5).ok_or("deadline overflow")?);
            }
            let arrivals_file = format!("golden-f{profile}-{load}-trace.json");
            std::fs::write(out.join(&arrivals_file), trace.to_json()?)?;
            for workers in [1, 2] {
                for policy in [ReplayPolicy::FlagOnly, ReplayPolicy::FallbackOnFlag] {
                    let c = config(workers, policy);
                    let r = replay_decode_trace(&trace, &c)?;
                    let name = format!("golden-f{profile}-{load}-w{workers}-{policy:?}");
                    println!(
                        "{name}: submitted={} committed={} late={} overflow={} logical={}/{}",
                        r.jobs.len(),
                        r.committed,
                        r.deadline_misses,
                        r.overflows,
                        r.decoded_logical_failures,
                        r.decoded_known
                    );
                    if workers == 1 && load == "near" {
                        std::fs::write(
                            out.join(format!("{name}-replay.json")),
                            serde_json::to_string_pretty(&r)?,
                        )?;
                    }
                    let mut row = compact(&name, &r, &arrivals_file);
                    row["capture_trace_file"] = json!(file);
                    row["arrival_period_ns"] = json!(period);
                    row["deadline_after_ns"] = json!(period * 5);
                    cases.push(row);
                }
            }
        }
        // No queue or deadline loss control: exact original policy outcomes must survive.
        let mut no_loss = original.clone();
        let max = no_loss
            .jobs
            .iter()
            .map(|j| {
                j.primary.service.nanoseconds().unwrap_or(0)
                    + j.fallback
                        .as_ref()
                        .and_then(|f| f.service.nanoseconds().ok())
                        .unwrap_or(0)
            })
            .max()
            .unwrap_or(1)
            .saturating_add(100_000);
        for (i, j) in no_loss.jobs.iter_mut().enumerate() {
            j.arrival_ns = i as u64 * max;
            j.deadline_after_ns = None;
        }
        std::fs::write(
            out.join(format!("golden-f{profile}-no-loss-trace.json")),
            no_loss.to_json()?,
        )?;
        let mut c = config(1, ReplayPolicy::FlagOnly);
        c.input_overhead_ns = 0;
        c.output_overhead_ns = 0;
        c.fallback_transfer_ns = 0;
        let r = replay_decode_trace(&no_loss, &c)?;
        assert_eq!(r.overflows, 0);
        assert_eq!(r.deadline_misses, 0);
        assert_eq!(
            r.decoded_logical_failures,
            original
                .jobs
                .iter()
                .filter(|j| j.primary.logical_failure == Some(true))
                .count()
        );
        std::fs::write(
            out.join(format!("golden-f{profile}-no-loss-control.json")),
            serde_json::to_string_pretty(&r)?,
        )?;
        let _ = file;
    }
    for name in ["constant-stable", "rare-spikes", "correlated-flags"] {
        let t = synthetic(name);
        let file = format!("{name}-trace.json");
        std::fs::write(out.join(&file), t.to_json()?)?;
        for policy in [ReplayPolicy::FlagOnly, ReplayPolicy::FallbackOnFlag] {
            let mut c = config(1, policy);
            c.queue_capacity = 16;
            c.input_overhead_ns = 0;
            c.output_overhead_ns = 0;
            c.fallback_transfer_ns = 0;
            c.overhead_provenance="Synthetic sensitivity: zero transfer costs; synthetic fallback service is explicitly 2000ns, not a host measurement".into();
            let r = replay_decode_trace(&t, &c)?;
            let tag = format!("{name}-{policy:?}");
            std::fs::write(
                out.join(format!("{tag}-replay.json")),
                serde_json::to_string_pretty(&r)?,
            )?;
            cases.push(compact(&tag, &r, &file));
        }
    }
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(
            &json!({"protocol":"Fixed ordered traces; same input fingerprint across actual full/early policies; no resampling; shared arrival/deadline sweep. Overheads are declared assumptions, not silicon measurements.","early_mean_service_ns":base,"cases":cases}),
        )?,
    )?;
    Ok(())
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("campaign") => campaign(
            Path::new(args.get(2).ok_or("OUTDIR required")?),
            args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(2048),
        ),
        Some("replay") => {
            let trace = DecodeTrace::from_json(&std::fs::read_to_string(
                args.get(2).ok_or("TRACE required")?,
            )?)?;
            let cfg: ReplayConfig = serde_json::from_str(&std::fs::read_to_string(
                args.get(3).ok_or("CONFIG required")?,
            )?)?;
            let r = replay_decode_trace(&trace, &cfg)?;
            std::fs::write(
                args.get(4).ok_or("OUT required")?,
                serde_json::to_string_pretty(&r)?,
            )?;
            Ok(())
        }
        _ => Err("usage: deadline_replay campaign OUTDIR [shots] | replay TRACE CONFIG OUT".into()),
    }
}
