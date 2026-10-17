//! Offline fixed-seed coverage against exact tiny-model static/drift/Markov expectations.
//! cargo run --release -p aleph-qec --example robustness_coverage -- OUT.json [trials=1000]
use aleph_qec::*;
use serde_json::json;
fn exact(dem: &DetectorErrorModel, d: &WitnessDecoder) -> Result<f64> {
    let mut total = 0.0;
    for mask in 0..1usize << dem.errors.len() {
        let faults: Vec<_> = (0..dem.errors.len())
            .filter(|i| mask >> i & 1 != 0)
            .collect();
        let p = dem
            .errors
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if mask >> i & 1 != 0 {
                    e.prob
                } else {
                    1.0 - e.prob
                }
            })
            .product::<f64>();
        if d.decode_faults(&faults)?.logical_failure {
            total += p;
        }
    }
    Ok(total)
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let out = args
        .get(1)
        .ok_or("usage: robustness_coverage OUT.json [trials]")?;
    let trials: u64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1000);
    if trials == 0 {
        return Err("positive trials required".into());
    }
    let base = DetectorErrorModel::parse(
        "error(0.05) D0 L0\nerror(0.3) D0 L1\nerror(0.15) L0\nerror(0.2) D1 L1",
    )?;
    let mut wrong = base.clone();
    wrong.errors[0].prob = 0.4;
    wrong.errors[1].prob = 0.01;
    let a = WitnessDecoder::new(&wrong, DecoderSpec::Mwpm)?;
    let b = WitnessDecoder::new(&base, DecoderSpec::Mwpm)?;
    let mut campaigns = Vec::new();
    for process in [
        NoiseProcess::Static,
        NoiseProcess::LinearDrift {
            end_probabilities: vec![0.2; 4],
        },
        NoiseProcess::MarkovBurst {
            onset: 0.15,
            recovery: 0.25,
            affected: vec![0, 1, 2],
            multiplier: 2.0,
        },
    ] {
        let steps = 16;
        let mut true_a = 0.0;
        let mut true_b = 0.0;
        let mut q = 0.0;
        let mut truth_curve = Vec::new();
        for t in 0..steps {
            let mut effective = base.clone();
            match &process {
                NoiseProcess::Static => (),
                NoiseProcess::LinearDrift { end_probabilities } => {
                    for (e, &end) in effective.errors.iter_mut().zip(end_probabilities) {
                        let f = t as f64 / (steps - 1) as f64;
                        e.prob = (1.0 - f) * e.prob + f * end;
                    }
                }
                NoiseProcess::MarkovBurst {
                    onset,
                    recovery,
                    affected,
                    multiplier,
                } => {
                    q = (1.0 - q) * onset + q * (1.0 - recovery);
                    for &i in affected {
                        effective.errors[i].prob = (effective.errors[i].prob * multiplier).min(1.0);
                    }
                }
            }
            let (pa, pb) = if matches!(process, NoiseProcess::MarkovBurst { .. }) {
                (
                    (1.0 - q) * exact(&base, &a)? + q * exact(&effective, &a)?,
                    (1.0 - q) * exact(&base, &b)? + q * exact(&effective, &b)?,
                )
            } else {
                (exact(&effective, &a)?, exact(&effective, &b)?)
            };
            true_a += pa / steps as f64;
            true_b += pb / steps as f64;
            truth_curve.push(json!({"time":t,"active_probability":q,"assumed":pa,"reference":pb}));
        }
        let mut counts = [0u64; 3];
        let mut intervals = Vec::new();
        for seed in 0..trials {
            let r = run_robustness(
                &base,
                &process,
                &a,
                &b,
                &RobustnessOptions {
                    streams: 32,
                    steps,
                    seed: 54600 + seed,
                    confidence: 0.95,
                    witness_limit: 0,
                },
            )?;
            let values = [true_a, true_b, true_a - true_b];
            let bounds = [
                r.assumed_interval,
                r.reference_interval,
                r.difference_interval,
            ];
            for (i, (v, b)) in values.iter().zip(bounds).enumerate() {
                counts[i] += u64::from(b.lower <= *v && *v <= b.upper);
            }
            intervals.push(json!({"seed":54600+seed,"assumed":bounds[0],"reference":bounds[1],"difference":bounds[2]}));
        }
        let coverage_ci = counts
            .iter()
            .map(|&n| BinomialInterval95::from_counts(trials, n))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|c| json!({"lower":c.lower,"upper":c.upper}))
            .collect::<Vec<_>>();
        campaigns.push(json!({"process":process,"streams":32,"steps":steps,"truth_curve":truth_curve,"expected_assumed":true_a,"expected_reference":true_b,"expected_difference":true_a-true_b,"trials":trials,"covered":counts,"coverage_binomial95":coverage_ci,"intervals":intervals}));
    }
    std::fs::write(
        out,
        serde_json::to_string_pretty(
            &json!({"base_dem":base.to_dem_string(),"assumed_dem":wrong.to_dem_string(),"reference_dem":base.to_dem_string(),"confidence":0.95,"protocol":"1000 or caller-declared independent seeds, fixed 32 streams x16 shots; no filtering; truth enumerates fault sets and analytically marginalizes the two-state law", "campaigns":campaigns}),
        )?,
    )?;
    Ok(())
}
