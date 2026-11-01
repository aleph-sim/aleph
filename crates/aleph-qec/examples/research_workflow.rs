//! Phase 7 QEC integration. Usage: research_workflow NEW_OUTPUT_DIRECTORY.
#[path = "support/research_qec.rs"]
mod integration;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: research_workflow NEW_OUTPUT_DIRECTORY".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&integration::study(std::path::Path::new(&args[0]))?)?
    );
    Ok(())
}
