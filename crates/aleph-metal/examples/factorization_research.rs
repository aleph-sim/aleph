//! Explicit P8-11 matrix campaign. Requires actual Metal; writes JSONL receipts.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_metal::factorization_research::Case;
    use std::io::Write;
    let chis = std::env::var("ALEPH_RESEARCH_CHIS").unwrap_or_else(|_| "32,64,128,256,512".into());
    let fixtures = std::env::var("ALEPH_RESEARCH_FIXTURES")
        .unwrap_or_else(|_| "well,clustered,ill,rank_deficient".into());
    let shapes =
        std::env::var("ALEPH_RESEARCH_SHAPES").unwrap_or_else(|_| "square,tall,wide".into());
    let repeats: usize = std::env::var("ALEPH_RESEARCH_REPEATS")
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    for chi in chis.split(',').map(str::parse::<usize>) {
        let chi = chi?;
        for shape in shapes.split(',') {
            let (m, n) = match shape {
                "square" => (2 * chi, 2 * chi),
                "tall" => (2 * chi, chi),
                "wide" => (chi, 2 * chi),
                _ => return Err("unknown shape".into()),
            };
            for fixture in fixtures.split(',') {
                let mut case = Case::new(m, n, fixture)?;
                case.bench_factor(false)?;
                case.bench_factor(true)?;
                // Frozen ABBA order; each receipt repeats CPU oracles/stage probes.
                for block in 0..repeats {
                    for candidate in if block % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let receipt = case.run(candidate)?;
                        println!("{}", serde_json::to_string(&receipt)?);
                        std::io::stdout().flush()?;
                    }
                }
            }
        }
    }
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("P8-11 requires actual Metal hardware");
}
