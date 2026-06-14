//! `aleph` — quantum circuit simulator CLI.  See
//! `docs/superpowers/specs/2026-05-25-p0-12-cli-design.md`.

use std::io::{self, Write};

use anyhow::Result;
use clap::Parser;

use aleph_cli::cli::{Cli, Cmd};
use aleph_cli::exec::{bench_circuit, run_circuit};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match cli.cmd {
        Cmd::Run {
            qasm,
            shots,
            statevector,
            force_statevector,
            expectation,
            seed,
            precision,
            backend,
            max_bond,
            max_error,
            noise,
        } => run_circuit(
            &qasm,
            shots,
            statevector,
            force_statevector,
            &expectation,
            seed,
            precision,
            backend,
            max_bond,
            max_error,
            &noise,
            &mut out,
        )?,
        Cmd::Bench { qasm, seed } => bench_circuit(&qasm, seed, &mut out)?,
    }
    out.flush()?;
    Ok(())
}
