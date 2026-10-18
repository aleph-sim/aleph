//! Phase 7 observable integration. Usage: research_observables NEW_OUTPUT_DIRECTORY [--smoke].
#[path = "support/research_observables.rs"]
mod integration;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 || (args.len() == 2 && args[1] != "--smoke") {
        return Err("usage: research_observables NEW_OUTPUT_DIRECTORY [--smoke]".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&integration::study(
            std::path::Path::new(&args[0]),
            args.len() == 2
        )?)?
    );
    Ok(())
}
