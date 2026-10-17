//! Reproducible fixed-budget P7-04 paired campaigns; cargo run --release -p aleph-qec
//! --example decoder_robustness -- OUTDIR [streams=64] [steps=128]. No adaptive fitting.
use aleph_qec::*;
use serde_json::{json, Value};
use std::{error::Error, path::Path, time::Instant};
type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;
fn configs(bb: bool) -> Vec<DecoderSpec> {
    if bb {
        vec![
            DecoderSpec::Relay {
                legs: 6,
                iters: 10,
                seed: 0x5E1A_4B9C,
                early_exit: false,
            },
            DecoderSpec::FixedRelay {
                legs: 6,
                iters: 10,
                seed: 0x5E1A_4B9C,
                msg_bits: 8,
                frac_bits: 3,
                early_exit: false,
            },
        ]
    } else {
        vec![DecoderSpec::Mwpm, DecoderSpec::UnionFind]
    }
}
fn surface(data: f64, meas: f64) -> AnyResult<DetectorErrorModel> {
    let e = SurfaceCode::new(3).memory_z_experiment(3);
    Ok(build_dem(
        &e.annotated,
        &e.phenomenological_mechanisms(data, meas),
    )?)
}
fn record(
    out: &Path,
    name: &str,
    base: &DetectorErrorModel,
    assumed: &DetectorErrorModel,
    process: &NoiseProcess,
    spec: &DecoderSpec,
    opts: &RobustnessOptions,
) -> AnyResult<Value> {
    let start = Instant::now();
    let a = WitnessDecoder::new(assumed, spec.clone())?;
    let b = WitnessDecoder::new(base, spec.clone())?;
    let setup = start.elapsed().as_secs_f64();
    let result = run_robustness(base, process, &a, &b, opts)?;
    for (i, w) in result.witnesses.iter().enumerate() {
        let path = out.join(format!("{name}-witness-{i}.json"));
        std::fs::write(path, w.to_json()?)?;
        FailureWitness::from_json(&w.to_json()?)?;
    }
    println!(
        "{name} assumed={:.6} reference={:.6} delta={:.6} interval=[{:.6},{:.6}] calls={}",
        result.assumed_rate,
        result.reference_rate,
        result.paired_difference,
        result.difference_interval.lower,
        result.difference_interval.upper,
        result.decoder_calls
    );
    let row = json!({"name":name,"setup_seconds":setup,"result":result});
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_string_pretty(&row)?,
    )?;
    Ok(row)
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let out = Path::new(
        args.get(1)
            .ok_or("usage: decoder_robustness OUTDIR [streams] [steps]")?,
    );
    std::fs::create_dir_all(out)?;
    let opts = RobustnessOptions {
        streams: args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(64),
        steps: args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(128),
        seed: 546,
        confidence: 0.95,
        witness_limit: 1,
    };
    let mut rows = Vec::new();
    for bb in [false, true] {
        let code = if bb { "gross-bb" } else { "surface-d3-r3" };
        for p in [0.01, 0.03] {
            let base = if bb {
                BBCode::gross().code_capacity_dem(p)
            } else {
                surface(p, p)?
            };
            for (family, spec) in configs(bb).iter().enumerate() {
                for ratio in [0.25, 1.0, 4.0] {
                    let assumed = if bb {
                        BBCode::gross().code_capacity_dem(p * ratio)
                    } else {
                        surface(p * ratio, p * ratio)?
                    };
                    rows.push(record(
                        out,
                        &format!("{code}-f{family}-p{p}-ratio{ratio}"),
                        &base,
                        &assumed,
                        &NoiseProcess::Static,
                        spec,
                        &opts,
                    )?);
                }
                if !bb {
                    // Actual physical data/measurement mismatch, retaining exact mechanism identities.
                    let assumed = surface(p * 4.0, p * 0.25)?;
                    rows.push(record(
                        out,
                        &format!("{code}-f{family}-p{p}-data-meas-bias"),
                        &base,
                        &assumed,
                        &NoiseProcess::Static,
                        spec,
                        &opts,
                    )?);
                }
            }
        }
        let base = if bb {
            BBCode::gross().code_capacity_dem(0.01)
        } else {
            surface(0.01, 0.01)?
        };
        let mut wrong = base.clone();
        for (i, e) in wrong.errors.iter_mut().enumerate() {
            e.prob *= if i % 2 == 0 { 0.25 } else { 4.0 };
        }
        for (family, spec) in configs(bb).iter().enumerate() {
            for process in [
                NoiseProcess::LinearDrift {
                    end_probabilities: base
                        .errors
                        .iter()
                        .map(|e| (4.0 * e.prob).min(1.0))
                        .collect(),
                },
                NoiseProcess::MarkovBurst {
                    onset: 0.05,
                    recovery: 0.2,
                    affected: (0..base.errors.len()).step_by(2).collect(),
                    multiplier: 8.0,
                },
            ] {
                let label = if matches!(process, NoiseProcess::LinearDrift { .. }) {
                    "drift"
                } else {
                    "bursts"
                };
                rows.push(record(
                    out,
                    &format!("{code}-f{family}-{label}"),
                    &base,
                    &wrong,
                    &process,
                    spec,
                    &opts,
                )?);
            }
        }
    }
    // Independent tiny-model exact oracle input; no empirical rare-event extrapolation.
    let tiny = DetectorErrorModel::parse(
        "error(0.05) D0 L0\nerror(0.3) D0 L1\nerror(0.15) L0\nerror(0.2) D1 L1",
    )?;
    let mut wrong = tiny.clone();
    wrong.errors[0].prob = 0.4;
    wrong.errors[1].prob = 0.01;
    let tiny_row = record(
        out,
        "tiny-mismatch",
        &tiny,
        &wrong,
        &NoiseProcess::Static,
        &DecoderSpec::Mwpm,
        &opts,
    )?;
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(
            &json!({"protocol":"fixed seeds, paired shots, no fitting/refresh; physical surface priors and BB capacity priors; dynamic comparator is fixed base priors, not an oracle", "cases":rows,"tiny":tiny_row}),
        )?,
    )?;
    Ok(())
}
