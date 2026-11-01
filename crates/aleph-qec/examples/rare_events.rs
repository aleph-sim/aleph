//! Fixed, reproducible matched-call campaigns and offline tiny-model coverage validation.
//! cargo run --release -p aleph-qec --example rare_events -- OUTDIR [coverage-trials] [campaign-seeds]
use aleph_qec::*;
use serde_json::{json, Value};
use std::{error::Error, path::Path, time::Instant};
type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;
fn opts(budgets: Vec<u64>, seed: u64) -> StratifiedOptions {
    StratifiedOptions {
        budgets,
        max_decoder_calls: 100_000,
        seed,
        confidence: 0.95,
        enumerate_small: true,
        witness_limit: 1,
        target_width: Some(1e-6),
    }
}
fn relay() -> DecoderSpec {
    DecoderSpec::Relay {
        legs: 6,
        iters: 10,
        seed: 0x5E1A_4B9C,
        early_exit: false,
    }
}
fn compare(
    name: &str,
    dem: &DetectorErrorModel,
    spec: DecoderSpec,
    max_k: usize,
    per_row: u64,
    seeds: u64,
    out: &Path,
) -> AnyResult<Value> {
    let setup = Instant::now();
    let decoder = WitnessDecoder::new(dem, spec.clone())?;
    let plan = FaultCountPlan::new(dem, max_k, 4_000_000)?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let direct_setup = Instant::now();
    let direct_decoder = make_direct(dem, &spec)?;
    let direct_setup_seconds = direct_setup.elapsed().as_secs_f64();
    let automatic_options = opts(vec![per_row; max_k + 1], 544);
    let automatic = run_automatic_rare_event(&plan, &decoder, &automatic_options)?;
    let mut runs = Vec::new();
    for seed in 0..seeds {
        let result = run_stratified(&plan, &decoder, &opts(vec![per_row; max_k + 1], 544 + seed))?;
        let direct_start = Instant::now();
        let mut errors = 0;
        // Same target priors and same *actual* predicate count, with ordinary DEM draws.
        // Capture artifacts are not built on this path; only the scored shot is evaluated.
        let (syndromes, truths) = sample_shots(dem, result.decoder_calls, 544 + seed);
        let predictions = direct_decoder.decode_batch(&syndromes)?;
        for (prediction, truth) in predictions.iter().zip(&truths) {
            errors += u64::from(prediction.observable_flips != *truth);
        }
        let interval = BinomialInterval95::from_counts(result.decoder_calls, errors)?;
        let direct_seconds = direct_start.elapsed().as_secs_f64();
        let row = json!({"seed":544+seed,"stratified":result,"direct":{"decoder_calls":result.decoder_calls,"errors":errors,"rate":errors as f64/result.decoder_calls as f64,"lower":interval.lower,"upper":interval.upper,"seconds":direct_seconds},"interval_width_ratio_stratified_over_direct":(result.failure_probability.upper-result.failure_probability.lower)/(interval.upper-interval.lower)});
        println!("case={name} seed={} calls={} strat_width={:.6e} direct_width={:.6e} omitted_upper={:.6e}",544+seed,result.decoder_calls,result.failure_probability.upper-result.failure_probability.lower,interval.upper-interval.lower,result.unsampled_mass.upper);
        runs.push(row);
    }
    let report = json!({"case":name,"automatic":automatic,"dem":dem.to_dem_string(),"mechanisms":dem.errors.len(),"detectors":dem.detectors,"observables":dem.observables,"decoder":spec,"max_faults":max_k,"per_row_declared_budget":per_row,"setup_seconds":setup_seconds,"direct_setup_seconds":direct_setup_seconds,"runs":runs});
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(report)
}
fn make_direct(dem: &DetectorErrorModel, spec: &DecoderSpec) -> AnyResult<Box<dyn Decoder>> {
    match *spec {
        DecoderSpec::Mwpm => Ok(Box::new(MwpmDecoder::new_with_split_components(dem)?)),
        DecoderSpec::Relay {
            legs,
            iters,
            seed,
            early_exit,
        } => Ok(Box::new(
            RelayBpDecoder::with_budget(dem, legs, iters, 0.875, (-0.3, 0.9), seed)
                .with_early_exit(early_exit),
        )),
        _ => Err("unsupported direct campaign configuration".into()),
    }
}
fn coverage(trials: u64, mixed_exact: bool) -> AnyResult<Value> {
    let dem = DetectorErrorModel::parse(
        "error(0.05) D0 L0\nerror(0.3) D0 L1\nerror(0.15) L0\nerror(0.2) D1 L1",
    )?;
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm)?;
    let plan = FaultCountPlan::new(&dem, 4, 100)?;
    let mut exact = 0.0;
    for mask in 0..16 {
        let faults = (0..4).filter(|i| mask >> i & 1 != 0).collect::<Vec<_>>();
        if decoder.decode_faults(&faults)?.logical_failure {
            exact += (0..4)
                .map(|i| {
                    if mask >> i & 1 != 0 {
                        dem.errors[i].prob
                    } else {
                        1.0 - dem.errors[i].prob
                    }
                })
                .product::<f64>();
        }
    }
    let start = Instant::now();
    let mut covered = 0;
    let mut intervals = Vec::new();
    for seed in 0..trials {
        let mut o = opts(
            if mixed_exact {
                vec![8, 8, 5, 8, 8]
            } else {
                vec![32; 5]
            },
            seed,
        );
        o.enumerate_small = mixed_exact;
        o.witness_limit = 0;
        o.target_width = None;
        let r = run_stratified(&plan, &decoder, &o)?;
        covered +=
            u64::from(r.failure_probability.lower <= exact && exact <= r.failure_probability.upper);
        intervals.push([r.failure_probability.lower, r.failure_probability.upper]);
    }
    let b = BinomialInterval95::from_counts(trials, covered)?;
    Ok(
        json!({"trials":trials,"nominal_confidence":0.95,"covered":covered,"observed_coverage":covered as f64/trials as f64,"coverage_binomial95":{"lower":b.lower,"upper":b.upper},"dem":dem.to_dem_string(),"exact_failure_probability":exact,"fixed_calls_per_trial":if mixed_exact {15}else{160},"enumerate_small":mixed_exact,"intervals":intervals,"seconds":start.elapsed().as_secs_f64()}),
    )
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let out = Path::new(
        args.get(1)
            .ok_or("usage: rare_events OUTDIR [coverage-trials] [campaign-seeds]")?,
    );
    std::fs::create_dir_all(out)?;
    let trials = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(1000);
    let seeds = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(16);
    if trials == 0 || seeds == 0 {
        return Err("positive trial/seed counts required".into());
    }
    let mut cases = Vec::new();
    for (name, p, k, budget) in [
        ("surface-low", 0.0001, 5, 256),
        ("surface-accessible", 0.03, 5, 256),
    ] {
        let exp = SurfaceCode::new(3).memory_z_experiment(3);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p))?;
        cases.push(compare(
            name,
            &dem,
            DecoderSpec::Mwpm,
            k,
            budget,
            seeds,
            out,
        )?);
    }
    for (name, p, k, budget) in [("bb-low", 0.001, 5, 256), ("bb-accessible", 0.03, 8, 256)] {
        cases.push(compare(
            name,
            &BBCode::gross().code_capacity_dem(p),
            relay(),
            k,
            budget,
            seeds,
            out,
        )?);
    }
    let coverage =
        json!({"all_conditional":coverage(trials,false)?,"mixed_exact":coverage(trials,true)?});
    std::fs::write(
        out.join("coverage.json"),
        serde_json::to_string_pretty(&coverage)?,
    )?;
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&json!({"cases":cases,"coverage":coverage}))?,
    )?;
    Ok(())
}
