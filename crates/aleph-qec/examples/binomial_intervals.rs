//! Batch confidence-bound evaluator for offline independent reference/coverage checks.
//! Input: one `shots errors` pair per line. Output: CSV, with explicit numerical failures.

use std::io::{self, BufRead};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("shots,errors,lower,upper,status");
    for line in io::stdin().lock().lines() {
        let line = line?;
        let counts: Vec<u64> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if counts.len() != 2 {
            return Err("expected shots errors".into());
        }
        let (n, k) = (counts[0], counts[1]);
        match aleph_qec::BinomialInterval95::from_counts(n, k) {
            Ok(b) => println!("{n},{k},{},{},ok", b.lower, b.upper),
            Err(e) => println!("{n},{k},,,{e}"),
        }
    }
    Ok(())
}
