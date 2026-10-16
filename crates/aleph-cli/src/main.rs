//! `aleph` — quantum circuit simulator CLI.  See
//! `docs/superpowers/specs/2026-05-25-p0-12-cli-design.md`.

use std::io::{self, Write};

use anyhow::Result;
use clap::Parser;

use aleph_cli::cli::{Cli, Cmd};
use aleph_cli::exec::{bench_circuit, run_circuit_with_execution_options};

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
            noise_memory_budget_bytes,
            noise_max_workers,
            cuda_raw,
            cuda_diagnostics,
        } => run_circuit_with_execution_options(
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
            aleph_sv::noise::NoiseOptions {
                memory_budget_bytes: noise_memory_budget_bytes.unwrap_or_else(|| {
                    aleph_sv::noise::NoiseOptions::default().memory_budget_bytes
                }),
                max_workers: noise_max_workers,
            },
            aleph_cli::exec::CudaOptions {
                raw: cuda_raw,
                diagnostics: cuda_diagnostics,
            },
            &mut out,
        )?,
        Cmd::Bench { qasm, seed } => bench_circuit(&qasm, seed, &mut out)?,
    }
    out.flush()?;
    Ok(())
}
