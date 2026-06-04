//! clap derive types.  See spec §4.1.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Floating-point precision for the state-vector backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Precision {
    /// Double precision (default). Oracle-reference accuracy (~1e-10).
    #[default]
    F64,
    /// Single precision. ~2× less memory traffic at large n; ~1e-6 accuracy.
    F32,
}

/// `aleph` — quantum circuit simulator command-line tool.
#[derive(Debug, Parser)]
#[command(name = "aleph", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Debug, Subcommand)]
pub enum Cmd {
    /// Run a QASM circuit and print results.
    ///
    /// With no view flag, defaults to `--shots 1024`.  `--shots`,
    /// `--statevector`, and `--expectation` can be combined; one
    /// backend run, multiple views off the resulting state.
    Run {
        /// Path to the OpenQASM 3.0 source file.
        qasm: PathBuf,

        /// Sample N shots from the final state.  Default is 1024
        /// when no other view flag is given.
        #[arg(long)]
        shots: Option<u32>,

        /// Print the full final state vector.  Capped at 10 qubits;
        /// use --force-statevector to print larger.
        #[arg(long)]
        statevector: bool,

        /// Bypass the 10-qubit cap on --statevector.  Foot-gun; n=20
        /// is 1 048 576 lines.
        #[arg(long)]
        force_statevector: bool,

        /// Compute ⟨ψ|P|ψ⟩ for a Pauli string like `ZZ`, `IXZI`, or
        /// `1.5*ZZ`.  Position 0 is qubit 0.  Repeatable for several
        /// observables in one run.
        #[arg(long)]
        expectation: Vec<String>,

        /// RNG seed for reproducibility.  Defaults to entropy.
        /// Reproducibility holds for a given aleph version; the
        /// sample-RNG-sequence contract is not stable across
        /// versions.
        #[arg(long)]
        seed: Option<u64>,

        /// State-vector precision: `f64` (default) or `f32` (faster at
        /// large n, lower accuracy).
        #[arg(long, value_enum, default_value_t = Precision::F64)]
        precision: Precision,
    },

    /// Run a QASM circuit once and print parse / execute / sample
    /// wall-times.  Single iteration; criterion benches under
    /// `crates/aleph-sv/benches/` are the source of truth for
    /// regression-tracking.
    Bench {
        /// Path to the OpenQASM 3.0 source file.
        qasm: PathBuf,

        /// RNG seed for the (always-on) 1024-shot sample phase.
        #[arg(long)]
        seed: Option<u64>,
    },
}
