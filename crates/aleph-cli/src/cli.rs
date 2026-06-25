//! clap derive types.  See spec §4.1.

use std::path::PathBuf;

use aleph_backend::{BackendKind, BackendRequest, Reach, SelectEnv, Selection};
use clap::{Parser, Subcommand};

/// Resolve a parsed [`BackendRequest`] into a [`Selection`] (kind + [`Reach`]).
///
/// `Auto` runs the [`aleph_backend::select_explained_full`] reach-aware heuristic
/// over `env` (GPU availability + precision preference); an auto pick of
/// `Stabilizer` is downgraded to `Statevector` when `wants_amplitudes` is set
/// (`--statevector`/`--force-statevector`), because the stabilizer backend has no
/// dense state vector. A `Fixed` choice keeps its kind verbatim (manual override)
/// but still gets its [`Reach`] computed from `n` (so `--backend cuda` at n>cap
/// pages). Diagnostics go to stderr so stdout stays pipeable.
///
/// Backend-name parsing lives once in [`BackendRequest::from_user_str`], shared
/// with the Python binding (P4-12); this is only the CLI-side resolution.
pub fn resolve_backend(
    request: BackendRequest,
    circuit: &aleph_ir::Circuit,
    wants_amplitudes: bool,
    env: &SelectEnv,
) -> Selection {
    match request {
        BackendRequest::Fixed(kind) => Selection {
            kind,
            reach: Reach::for_kind(kind, circuit.num_qubits()),
            reason: "explicit --backend override",
        },
        BackendRequest::Auto => {
            // `env` carries the caller's runtime GPU probes (Metal / CUDA) and the
            // precision preference; the reach-aware heuristic routes large dense
            // circuits to the GPU and pages past the in-core cap (P5.6-07, P5.11-06).
            let sel = aleph_backend::select_explained_full(circuit, env);
            if sel.kind == BackendKind::Stabilizer && wants_amplitudes {
                eprintln!(
                    "auto-selected backend: state vector \
                     (downgraded from stabilizer: --statevector needs amplitudes \
                     the stabilizer backend cannot provide)"
                );
                Selection {
                    kind: BackendKind::Statevector,
                    reach: Reach::InCore,
                    reason: "downgraded from stabilizer for --statevector",
                }
            } else {
                eprintln!("auto-selected backend: {} ({})", sel.kind, sel.reason);
                sel
            }
        }
    }
}

/// Floating-point precision for the state-vector backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Precision {
    /// Let the backend choose (default). The CPU/`auto`-CPU path stays FP64
    /// (oracle accuracy); an `auto` GPU pick resolves to FP32 — 2× throughput and
    /// one extra qubit of in-core reach, the P5.10/5.11 default GPU trade.
    #[default]
    Auto,
    /// Force double precision. Oracle-reference accuracy (~1e-10).
    F64,
    /// Force single precision. ~2× less memory traffic at large n; ~1e-6 accuracy.
    F32,
}

impl From<Precision> for aleph_backend::Precision {
    fn from(p: Precision) -> Self {
        match p {
            Precision::Auto => aleph_backend::Precision::Auto,
            Precision::F64 => aleph_backend::Precision::F64,
            Precision::F32 => aleph_backend::Precision::F32,
        }
    }
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

        /// State-vector precision: `auto` (default — FP64 on CPU, FP32 on an
        /// `auto` GPU pick), `f64`, or `f32` (faster at large n, lower accuracy).
        #[arg(long, value_enum, default_value_t = Precision::Auto)]
        precision: Precision,

        /// Simulation backend: `auto` (default — picks from circuit
        /// structure), `statevector` (alias `sv`), `stabilizer` (alias
        /// `stab`; Clifford-only, rejects non-Clifford gates and
        /// --statevector), `mps` (tensor network; bounded entanglement,
        /// rejects --statevector), `metal` (alias `gpu`; FP32 Apple Silicon
        /// GPU — needs a macOS `--features metal` build), or `cuda` /
        /// `cuda-f32` (NVIDIA GPU state vector, FP64 / FP32 — needs a Linux
        /// `--features cuda` build; out-of-core paging past the in-core cap).
        /// `auto` picks a CUDA GPU backend for large dense circuits only when
        /// a CUDA build reports a device. Same names as the Python `backend=`.
        #[arg(long, default_value = BackendRequest::AUTO, value_parser = BackendRequest::from_user_str)]
        backend: BackendRequest,

        /// MPS max bond dimension χ (only used by `--backend mps`).
        #[arg(long, default_value_t = 128)]
        max_bond: usize,

        /// MPS error-bounded truncation: keep the discarded weight per bond
        /// below ε (only `--backend mps`; overrides fixed-χ, with `--max-bond`
        /// as a safety cap).
        #[arg(long)]
        max_error: Option<f64>,

        /// Apply a built-in noise preset, repeatable. Format `<preset>:<p>`
        /// with `p` in [0,1]. Presets: `depol:<p>` (depolarizing on every 1q
        /// and 2q gate in the circuit) and `readout:<p>` (symmetric readout
        /// flip on every qubit). Forces the state-vector backend; cannot be
        /// combined with --statevector or --expectation. Full NoiseModel
        /// construction is available in the Python API.
        #[arg(long)]
        noise: Vec<String>,
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

#[cfg(test)]
mod backend_choice_tests {
    use super::*;
    use aleph_backend::BackendKind;
    use aleph_core::{Gate, GateInstance};
    use aleph_ir::Circuit;

    fn clifford() -> Circuit {
        let mut c = Circuit::new(2, 0);
        c.add_gate(GateInstance::new(Gate::H, vec![0u32])).unwrap();
        c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
            .unwrap();
        c
    }

    use aleph_backend::SelectEnv;

    /// SelectEnv with only Metal available (the pre-P5.11-06 `auto` probe shape).
    fn metal_env(on: bool) -> SelectEnv {
        SelectEnv {
            metal_available: on,
            ..SelectEnv::default()
        }
    }

    #[test]
    fn explicit_choice_overrides_without_analysis() {
        let c = clifford();
        assert_eq!(
            resolve_backend(
                BackendRequest::Fixed(BackendKind::Mps),
                &c,
                false,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Mps
        );
        assert_eq!(
            resolve_backend(
                BackendRequest::Fixed(BackendKind::Statevector),
                &c,
                false,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Statevector
        );
    }

    #[test]
    fn auto_picks_stabilizer_for_clifford() {
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                &clifford(),
                false,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Stabilizer
        );
    }

    #[test]
    fn auto_downgrades_to_sv_when_amplitudes_requested() {
        // Clifford would be stabilizer, but --statevector needs amplitudes.
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                &clifford(),
                true,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Statevector
        );
    }

    // P5.6-07: a large dense (non-Clifford) circuit auto-routes to Metal when the
    // caller reports the GPU available, and stays on the CPU SV otherwise.
    #[test]
    fn auto_routes_large_dense_to_metal_when_available() {
        let mut big = aleph_ir::Circuit::new(aleph_backend::GPU_PREFER_N, 0);
        big.h(0).unwrap();
        big.t(0).unwrap(); // non-Clifford ⇒ not stabilizer
        assert_eq!(
            resolve_backend(BackendRequest::Auto, &big, false, &metal_env(true)).kind,
            BackendKind::Metal,
            "GPU available + n>=GPU_PREFER_N ⇒ Metal"
        );
        assert_eq!(
            resolve_backend(BackendRequest::Auto, &big, false, &metal_env(false)).kind,
            BackendKind::Statevector,
            "GPU unavailable ⇒ CPU state vector"
        );
    }

    // Below the GPU threshold, auto stays on the CPU SV even when Metal is there.
    #[test]
    fn auto_keeps_small_dense_on_cpu_even_with_metal() {
        let mut small = aleph_ir::Circuit::new(aleph_backend::GPU_PREFER_N - 1, 0);
        small.h(0).unwrap();
        small.t(0).unwrap();
        assert_eq!(
            resolve_backend(BackendRequest::Auto, &small, false, &metal_env(true)).kind,
            BackendKind::Statevector
        );
    }

    // P5.11-06: with CUDA available, a large dense circuit auto-routes to the FP32
    // CUDA backend, paging past the in-core cap.
    #[test]
    fn auto_routes_large_dense_to_cuda_and_pages() {
        let cuda_env = SelectEnv {
            cuda_available: true,
            ..SelectEnv::default()
        };
        // n=32 > FP32 in-core cap ⇒ CudaF32, paged. A long-range CNOT keeps it off
        // the MPS route (otherwise a shallow nearest-neighbor circuit picks MPS).
        let mut big = aleph_ir::Circuit::new(32, 0);
        big.h(0).unwrap();
        big.t(0).unwrap();
        big.cnot(0, 16).unwrap();
        let sel = resolve_backend(BackendRequest::Auto, &big, false, &cuda_env);
        assert_eq!(sel.kind, BackendKind::CudaF32);
        assert!(matches!(sel.reach, Reach::Paged { .. }));
    }

    // An explicit `--backend cuda` at n past the FP64 cap still computes a paged reach.
    #[test]
    fn explicit_cuda_pages_past_cap() {
        let mut big = aleph_ir::Circuit::new(aleph_backend::MAX_CUDA_QUBITS + 1, 0);
        big.h(0).unwrap();
        let sel = resolve_backend(
            BackendRequest::Fixed(BackendKind::Cuda),
            &big,
            false,
            &SelectEnv::default(),
        );
        assert_eq!(sel.kind, BackendKind::Cuda);
        assert!(matches!(sel.reach, Reach::Paged { .. }));
    }

    // The `--backend` default string parses to Auto (parity with the Python
    // default and the former clap ValueEnum default).
    #[test]
    fn default_backend_string_parses_to_auto() {
        assert_eq!(
            BackendRequest::from_user_str(BackendRequest::AUTO),
            Ok(BackendRequest::Auto)
        );
    }

    // The CLI now accepts the same aliases as Python (`sv`, `stab`).
    #[test]
    fn cli_accepts_python_aliases() {
        assert_eq!(
            resolve_backend(
                BackendRequest::from_user_str("sv").unwrap(),
                &clifford(),
                false,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Statevector
        );
        assert_eq!(
            resolve_backend(
                BackendRequest::from_user_str("stab").unwrap(),
                &clifford(),
                false,
                &SelectEnv::default()
            )
            .kind,
            BackendKind::Stabilizer
        );
    }
}
