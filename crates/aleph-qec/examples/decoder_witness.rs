//! Capture, minimize and compare real surface/BB logical failures, or replay saved JSON.
//! cargo run --release -p aleph-qec --example decoder_witness -- capture OUTDIR
//! cargo run --release -p aleph-qec --example decoder_witness -- replay FILE.json
use aleph_qec::*;
use std::{
    error::Error,
    path::Path,
    time::{Duration, Instant},
};
fn fixed() -> DecoderSpec {
    DecoderSpec::FixedRelay {
        legs: 6,
        iters: 10,
        seed: 0x5E1A_4B9C,
        msg_bits: 8,
        frac_bits: 3,
        early_exit: false,
    }
}
fn compare(w: &FailureWitness) -> std::result::Result<(), Box<dyn Error>> {
    let dem = DetectorErrorModel::parse(&w.dem)?;
    let specs = if dem
        .errors
        .iter()
        .all(|e| e.dets.len() <= 2 || !e.components.is_empty())
    {
        vec![DecoderSpec::Mwpm, DecoderSpec::UnionFind]
    } else {
        vec![
            DecoderSpec::Relay {
                legs: 6,
                iters: 10,
                seed: 0x5E1A_4B9C,
                early_exit: false,
            },
            fixed(),
        ]
    };
    for spec in specs {
        let out = WitnessDecoder::new(&dem, spec.clone())?.replay(w)?;
        println!(
            "decoder={spec:?} failure={} valid={:?} prediction={:?}",
            out.logical_failure, out.valid, out.prediction
        );
    }
    Ok(())
}
fn capture(
    name: &str,
    dem: &DetectorErrorModel,
    spec: DecoderSpec,
    dir: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let decoder = WitnessDecoder::new(dem, spec)?;
    let search = Instant::now();
    let mut found = None;
    for shot in 0..100_000 {
        let w = capture_shot(&decoder, 543, shot)?;
        if w.outcome.logical_failure
            && w.faults.len() >= if name.starts_with("surface") { 4 } else { 9 }
        {
            found = Some(w);
            break;
        }
    }
    let w = found.ok_or("no failure in 100000 shots")?;
    let search_elapsed = search.elapsed();
    let reduced = decoder.minimize(
        &w,
        MinimizeBudget {
            decoder_calls: 100_000,
            time: Duration::from_secs(60),
        },
    )?;
    println!("case={name} shot={} faults={}->{} calls={} status={:?} search_us={} minimize_us={} signature={}",w.shot,w.faults.len(),reduced.witness.faults.len(),reduced.decoder_calls,reduced.status,search_elapsed.as_micros(),reduced.elapsed.as_micros(),reduced.witness.signature()?);
    let json = reduced.witness.to_json()?;
    let imported = FailureWitness::from_json(&json)?;
    if imported != reduced.witness {
        return Err("JSON round-trip changed witness".into());
    }
    std::fs::write(dir.join(format!("{name}.json")), json)?;
    compare(&imported)?;
    if matches!(imported.decoder, DecoderSpec::FixedRelay { .. }) {
        let (syn, reference) = decoder.fixed_reference(&imported)?;
        let rec = read_ref(&mut reference.as_slice())?;
        let bits: Vec<_> = syn
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| {
                let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                (0..32).map(move |i| word >> i & 1 != 0)
            })
            .take(dem.detectors)
            .collect();
        let source = FixedRelayBp::with_budget(dem, 6, 10, (-0.3, 0.9), 0x5E1A_4B9C, 8, 3);
        let syndrome = Syndrome::from_bits(&bits);
        let (correction, valid) = source.decode_fixed(&syndrome);
        let mask = correction
            .observable_flips
            .iter()
            .enumerate()
            .fold(0u16, |v, (i, &b)| v | (u16::from(b) << i));
        if mask != rec[0].sw_obs
            || valid != rec[0].valid
            || source.iters_to_valid(&syndrome).1 != u32::from(rec[0].iters)
        {
            return Err("fixed golden reference mismatch".into());
        }
        std::fs::write(dir.join(format!("{name}.syn")), syn)?;
        std::fs::write(dir.join(format!("{name}.ref")), reference)?;
        println!("fixed_reference=PASS layout=block gross_code_capacity C={} V={} OBS={} words={} Q5.3 schedule=6x10 full priors=0.03",dem.detectors,dem.errors.len(),dem.observables,dem.detectors.div_ceil(32));
    }
    Ok(())
}
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("capture") => {
            let dir = Path::new(args.get(2).ok_or("capture requires output directory")?);
            std::fs::create_dir_all(dir)?;
            let exp = SurfaceCode::new(3).memory_z_experiment(3);
            let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03))?;
            capture("surface-d3-r3", &dem, DecoderSpec::Mwpm, dir)?;
            capture(
                "bb-gross-capacity",
                &BBCode::gross().code_capacity_dem(0.03),
                fixed(),
                dir,
            )?;
        }
        Some("replay") => {
            let w = FailureWitness::from_json(&std::fs::read_to_string(
                args.get(2).ok_or("replay requires JSON path")?,
            )?)?;
            compare(&w)?;
        }
        _ => return Err("usage: decoder_witness capture OUTDIR | replay FILE.json".into()),
    }
    Ok(())
}
