//! Bounded integration study: ordinary draws, conditional estimates, witnesses and replay.
//! This is an example/test helper, not a new sampling or decoder implementation.
use aleph_qec::*;
use serde_json::{json, Value};
use std::{error::Error, path::Path, time::Duration};
type AnyResult<T> = std::result::Result<T, Box<dyn Error>>;
fn write_json(path: &Path, value: &impl serde::Serialize) -> AnyResult<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}
fn export(
    out: &Path,
    name: &str,
    source: &WitnessDecoder,
    witness: &FailureWitness,
) -> AnyResult<Value> {
    let reduced = source.minimize(
        witness,
        MinimizeBudget {
            decoder_calls: 512,
            time: Duration::from_secs(30),
        },
    )?;
    if reduced.status != MinimizeStatus::OneMinimal {
        return Err("bounded witness did not finish certification".into());
    }
    let file = format!("{name}.json");
    std::fs::write(out.join(&file), reduced.witness.to_json()?)?;
    let imported = FailureWitness::from_json(&std::fs::read_to_string(out.join(&file))?)?;
    if imported != reduced.witness || !source.replay(&imported)?.logical_failure {
        return Err("witness round-trip/source failure changed".into());
    }
    // An explicit final deletion sweep independently checks the integration's certificate use.
    for i in 0..imported.faults.len() {
        let mut faults = imported.faults.clone();
        faults.remove(i);
        if source.decode_faults(&faults)?.logical_failure {
            return Err("reduced witness is not deletion-minimal".into());
        }
    }
    let mut replays = Vec::new();
    for spec in [DecoderSpec::Mwpm, DecoderSpec::UnionFind] {
        let decoder = WitnessDecoder::new(source.model(), spec.clone())?;
        replays.push(json!({"decoder":spec,"outcome":decoder.replay(&imported)?}));
    }
    Ok(
        json!({"file":file,"schema_version":imported.version,"sampling":imported.sampling,"dem_sha256":imported.dem_sha256,"signature":imported.signature()?,"seed":imported.seed,"shot_or_rank":imported.shot,"original_faults":imported.original_faults.len(),"reduced_faults":imported.faults.len(),"status":reduced.status,"reduction_calls":reduced.decoder_calls,"source_failure":true,"cross_decoder_replays":replays}),
    )
}
fn stratified_options(max_k: usize, calls: u64) -> StratifiedOptions {
    StratifiedOptions {
        budgets: vec![calls; max_k + 1],
        max_decoder_calls: (max_k as u64 + 1) * calls,
        seed: 542,
        confidence: 0.95,
        enumerate_small: true,
        witness_limit: 1,
        target_width: None,
    }
}
pub fn study(out: &Path) -> AnyResult<Value> {
    // A new directory prevents silent mixing with a previous run's artifacts.
    std::fs::create_dir(out)?;
    let exp = SurfaceCode::new(3).memory_z_experiment(3);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03))?;
    std::fs::write(out.join("surface.dem"), dem.to_dem_string())?;
    let decoder = WitnessDecoder::new(&dem, DecoderSpec::Mwpm)?;
    let mut failures = 0;
    let mut first = None;
    // Fixed endpoint; retain the earliest failing shot for diagnosis, never stop the estimator there.
    for shot in 0..512 {
        let w = capture_shot(&decoder, 542, shot)?;
        if w.outcome.logical_failure {
            failures += 1;
            if first.is_none() {
                first = Some(w);
            }
        }
    }
    let ordinary =
        first.ok_or("no surface failure in the predeclared 512-shot diagnostic batch")?;
    std::fs::write(out.join("surface-original.json"), ordinary.to_json()?)?;
    let reduced = export(out, "surface-reduced", &decoder, &ordinary)?;
    let interval = BinomialInterval95::from_counts(512, failures)?;
    let plan = FaultCountPlan::new(&dem, 3, 4_000_000)?;
    let stratified = run_stratified(&plan, &decoder, &stratified_options(3, 64))?;
    if stratified.dem_sha256 != ordinary.dem_sha256 {
        return Err("surface model identity changed across workflows".into());
    }
    write_json(&out.join("surface-stratified.json"), &stratified)?;
    let mut conditional = Vec::new();
    for (i, w) in stratified.witnesses.iter().enumerate() {
        conditional.push(export(
            out,
            &format!("surface-conditional-{i}"),
            &decoder,
            w,
        )?);
    }

    // Independent analytical control: matching prefers the p=.2 no-logical boundary edge.
    // Failure is parity of the independent .1 and .03 logical mechanisms: .1+.03-2*.1*.03=.124.
    let tiny = DetectorErrorModel::parse("error(0.1) D0 L0\nerror(0.2) D0\nerror(0.03) L0")?;
    let tiny_decoder = WitnessDecoder::new(&tiny, DecoderSpec::Mwpm)?;
    let exact = run_stratified(
        &FaultCountPlan::new(&tiny, 3, 100)?,
        &tiny_decoder,
        &stratified_options(3, 8),
    )?;
    let truth = 0.124;
    if exact.decoder_calls != 8
        || exact.failure_probability.lower > truth
        || exact.failure_probability.upper < truth
    {
        return Err("complete enumeration disagrees with analytical parity control".into());
    }
    write_json(&out.join("tiny-exact.json"), &exact)?;
    let mut tiny_witnesses = Vec::new();
    for (i, w) in exact.witnesses.iter().enumerate() {
        tiny_witnesses.push(export(
            out,
            &format!("tiny-conditional-{i}"),
            &tiny_decoder,
            w,
        )?);
    }

    // Separate experiment with declared changed priors and correlated generating process.
    // Conditional rare-event weights are NOT reused for this temporal process.
    let mut assumed = tiny.clone();
    assumed.errors[0].prob = 0.2;
    assumed.errors[1].prob = 0.1;
    let assumed_decoder = WitnessDecoder::new(&assumed, DecoderSpec::Mwpm)?;
    let process = NoiseProcess::MarkovBurst {
        onset: 0.1,
        recovery: 0.25,
        affected: vec![2],
        multiplier: 3.,
    };
    let robustness = run_robustness(
        &tiny,
        &process,
        &assumed_decoder,
        &tiny_decoder,
        &RobustnessOptions {
            streams: 4,
            steps: 32,
            seed: 542,
            confidence: 0.95,
            witness_limit: 1,
        },
    )?;
    write_json(&out.join("tiny-robustness.json"), &robustness)?;
    let mut dynamic = Vec::new();
    for (i, w) in robustness.witnesses.iter().enumerate() {
        dynamic.push(export(
            out,
            &format!("tiny-dynamic-{i}"),
            &assumed_decoder,
            w,
        )?);
    }
    // Priors are part of witness identity. Cross-decoder replay is allowed; changed-model replay is not.
    if assumed_decoder.replay(&ordinary).is_ok() || tiny_decoder.replay(&ordinary).is_ok() {
        return Err("mismatched witness model accepted".into());
    }
    let result = json!({"study_version":1,"seed":542,"surface":{"model_file":"surface.dem","dem_sha256":ordinary.dem_sha256,"sampling":"ordinary independent original DEM mechanisms; seed/shot indexed","shots":512,"failures":failures,"interval95":{"lower":interval.lower,"upper":interval.upper},"selected_shot":ordinary.shot,"witness":reduced,"stratified_file":"surface-stratified.json","stratified_actual_calls":stratified.decoder_calls,"stratified_declared_calls":stratified.declared_decoder_calls,"stratified_interval":stratified.failure_probability,"omitted_mass":stratified.unsampled_mass,"conditional_witnesses":conditional},"tiny_exact":{"truth":truth,"decoder_calls":exact.decoder_calls,"interval":exact.failure_probability,"witnesses":tiny_witnesses},"tiny_dynamic":{"result_file":"tiny-robustness.json","independent_streams":4,"steps_per_stream":32,"decoder_calls":robustness.decoder_calls,"difference":robustness.paired_difference,"difference_interval":robustness.difference_interval,"witnesses":dynamic},"scope":"postselected witnesses diagnose one failure, not relative decoder LER; reduced faults are not fresh random draws; dynamic process is separate from independent stratification; no physical circuit-fault mapping or silicon timing"});
    write_json(&out.join("study.json"), &result)?;
    Ok(result)
}
