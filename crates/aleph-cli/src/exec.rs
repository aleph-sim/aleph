//! Business logic: run_circuit / bench_circuit.  See spec §4.2.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use aleph_backend::{run, Backend, BackendKind, BackendRequest};
use aleph_core::Complex;
use aleph_mps::{MpsBackend, TruncationPolicy};
use aleph_stab::StabilizerBackend;
use aleph_sv::{Fp32SvBackend, NaiveSvBackend};

use crate::cli::{resolve_backend, Precision};
use crate::output;
use crate::pauli::parse_pauli_arg;

/// Bridges the two backends' differently-typed amplitude buffers
/// (`&[Complex<f64>]` vs `&[Complex<f32>]`) to a uniform `Vec<Complex<f64>>`
/// for the `--statevector` view.  The f64 path clones verbatim (byte-identical
/// to the historical `state.amplitudes()` dump); the f32 path widens via the
/// backend's `to_aos_f64()`.
trait AmpsF64 {
    fn amps_f64(&self) -> Vec<Complex>;
}

impl AmpsF64 for aleph_sv::CpuState {
    fn amps_f64(&self) -> Vec<Complex> {
        self.amplitudes().to_vec()
    }
}

impl AmpsF64 for aleph_sv::Fp32CpuState {
    fn amps_f64(&self) -> Vec<Complex> {
        self.to_aos_f64()
    }
}

// The Metal GPU state is FP32; widen to f64 for the `--statevector` view exactly
// like the CPU FP32 backend. Only compiled when the `metal` feature is on (macOS).
#[cfg(all(target_os = "macos", feature = "metal"))]
impl AmpsF64 for aleph_metal::MetalSvState {
    fn amps_f64(&self) -> Vec<Complex> {
        self.to_aos_f64()
    }
}

/// Default shot count when the user passes no view flag.
pub const DEFAULT_SHOTS: u32 = 1024;

/// State-vector cap that gates `--statevector` without
/// `--force-statevector`.  10 qubits = 1024 amplitudes, comfortable
/// on a terminal; 28 qubits = 256 M lines, decidedly not.
pub const STATEVECTOR_CAP_QUBITS: u32 = 10;

/// Parse + run + print views (counts / statevector / expectation) per
/// the user's flags.  Validates all CLI arguments BEFORE running the
/// circuit so a bad input does not produce partial output.
#[allow(clippy::too_many_arguments)]
pub fn run_circuit<W: Write>(
    qasm_path: &Path,
    shots_opt: Option<u32>,
    print_statevector: bool,
    force_statevector: bool,
    expectations: &[String],
    seed: Option<u64>,
    precision: Precision,
    backend: BackendRequest,
    max_bond: usize,
    max_error: Option<f64>,
    noise: &[String],
    out: &mut W,
) -> Result<()> {
    // 1. Read + parse.
    let source = std::fs::read_to_string(qasm_path)
        .with_context(|| format!("reading QASM file: {}", qasm_path.display()))?;
    let circuit =
        aleph_parser::parse(&source).with_context(|| format!("parsing {}", qasm_path.display()))?;
    let n = circuit.num_qubits();

    // Noise preset path: build a NoiseModel from --noise presets and run the
    // Monte-Carlo trajectory engine. SV-only, shots-only (no statevector /
    // expectation view under a mixed state). Returns before the normal
    // backend dispatch.
    if !noise.is_empty() {
        if !backend.allows_noise() {
            // Report the canonical name of the backend the user pinned, not the
            // Debug variant name. !allows_noise admits only an explicit
            // Stabilizer / Mps / Metal (Auto and Statevector allow noise).
            let requested = match backend {
                BackendRequest::Fixed(k) => k.canonical_name(),
                BackendRequest::Auto => "stabilizer",
            };
            return Err(anyhow!(
                "--noise is only supported on the state-vector backend; \
                 remove --backend {requested}"
            ));
        }
        if print_statevector || force_statevector || !expectations.is_empty() {
            return Err(anyhow!(
                "--noise is shots-only in v1; it cannot be combined with \
                 --statevector, --force-statevector, or --expectation"
            ));
        }
        let model = build_noise_model(noise, &circuit, n)?;
        let shots = shots_opt.unwrap_or(DEFAULT_SHOTS);
        let run_seed = seed.unwrap_or_else(rand::random::<u64>);
        let seed_label = match seed {
            Some(s) => format!("seed={s}"),
            None => "seed=entropy".to_string(),
        };
        let hist = aleph_sv::noise::run_noisy(&circuit, &model, shots, run_seed)
            .context("running noisy circuit")?;
        output::format_counts_hist(out, &hist, shots, n, &seed_label)?;
        return Ok(());
    }

    // 2. Parse + range-check every --expectation BEFORE running.
    let mut paulis: Vec<(String, aleph_core::PauliString)> = Vec::with_capacity(expectations.len());
    for raw in expectations {
        let ps = parse_pauli_arg(raw)
            .with_context(|| format!("parsing --expectation argument {raw:?}"))?;
        if let Some(&(q, _)) = ps.terms.last() {
            if q >= n {
                return Err(anyhow!(
                    "--expectation {raw:?} references qubit {q} but circuit has {n} qubits"
                ));
            }
        }
        paulis.push((raw.clone(), ps));
    }

    // 3a. Validate --max-error if provided.
    if let Some(e) = max_error {
        if !e.is_finite() || e <= 0.0 {
            return Err(anyhow!(
                "--max-error must be a positive finite number, got {e}"
            ));
        }
    }

    // 3b. Resolve the backend (runs the auto heuristic for `--backend auto`).
    //     The requested output view gates an auto stabilizer pick: a
    //     state-vector view needs amplitudes the stabilizer backend lacks.
    let wants_amplitudes = print_statevector || force_statevector;
    let resolved = resolve_backend(backend, &circuit, wants_amplitudes, metal_available());

    // The dense state-vector family (CPU SV + Metal GPU SV) shares the qubit-cap
    // and `--statevector` print-cap policy; the stabilizer/MPS backends don't.
    let sv_family = matches!(resolved, BackendKind::Statevector | BackendKind::Metal);

    // Too-large soft warning: an exact dense run past the soft cap may exhaust
    // memory. We warn and proceed (the user stayed in control by not narrowing
    // the backend); this mirrors the SV soft-cap-warns-not-refuses convention.
    if sv_family && n > aleph_backend::SV_EXACT_CAP {
        eprintln!(
            "warning: n={n} exceeds the {}-qubit state-vector soft cap; \
             this run may exhaust memory (override with a different --backend)",
            aleph_backend::SV_EXACT_CAP
        );
    }

    // 3. Statevector cap check. Skipped for the stabilizer backend, which
    //    has no dense state vector at all — `run_stabilizer` rejects
    //    `--statevector` with a clearer, backend-specific message below.
    if sv_family && print_statevector && !force_statevector && n > STATEVECTOR_CAP_QUBITS {
        let dim = 1u64 << n;
        return Err(anyhow!(
            "state vector has 2^{n} = {dim} amplitudes; pass --force-statevector to print anyway"
        ));
    }

    // 4. Default view: --shots 1024 if no view flag was passed.
    let effective_shots = match (shots_opt, print_statevector, !paulis.is_empty()) {
        (Some(s), _, _) => Some(s),
        (None, false, false) => Some(DEFAULT_SHOTS),
        _ => None,
    };

    // 5. Allocate backend + run circuit once + print views.  The two
    //    precision backends have different associated `State` types, so
    //    the run-and-view pipeline is factored into a generic helper
    //    (`run_with_backend`) instantiated per concrete backend here.
    let seed_label = match seed {
        Some(s) => format!("seed={s}"),
        None => "seed=entropy".to_string(),
    };

    if resolved == aleph_backend::BackendKind::Stabilizer {
        return run_stabilizer(
            &circuit,
            effective_shots,
            print_statevector || force_statevector,
            &paulis,
            n,
            seed,
            &seed_label,
            out,
        );
    }

    if resolved == aleph_backend::BackendKind::Mps {
        return run_mps(
            &circuit,
            effective_shots,
            print_statevector || force_statevector,
            &paulis,
            n,
            seed,
            max_bond,
            max_error,
            &seed_label,
            out,
        );
    }

    if resolved == BackendKind::Metal {
        return run_metal(
            &circuit,
            effective_shots,
            print_statevector,
            &paulis,
            n,
            seed,
            &seed_label,
            out,
        );
    }

    match precision {
        Precision::F64 => {
            let backend = match seed {
                Some(s) => NaiveSvBackend::with_seed(s),
                None => NaiveSvBackend::new(),
            };
            run_with_backend(
                backend,
                &circuit,
                effective_shots,
                print_statevector,
                &paulis,
                n,
                &seed_label,
                out,
            )
        }
        Precision::F32 => {
            let backend = match seed {
                Some(s) => Fp32SvBackend::with_seed(s),
                None => Fp32SvBackend::new(),
            };
            run_with_backend(
                backend,
                &circuit,
                effective_shots,
                print_statevector,
                &paulis,
                n,
                &seed_label,
                out,
            )
        }
    }
}

/// Run `circuit` on `backend` once, then emit the requested views
/// (sampling / statevector / expectation).  Generic over the backend so
/// both the f64 and f32 state-vector paths share one implementation; the
/// only state-type-specific operation, the `--statevector` amplitude dump,
/// is routed through the `AmpsF64` bridge.
#[allow(clippy::too_many_arguments)]
fn run_with_backend<B, W>(
    mut backend: B,
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    print_statevector: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed_label: &str,
    out: &mut W,
) -> Result<()>
where
    B: Backend,
    B::State: AmpsF64,
    W: Write,
{
    let state = run(&mut backend, circuit).context("running circuit")?;

    if let Some(shots) = effective_shots {
        let samples = backend
            .sample(&state, shots)
            .context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, seed_label)?;
    }
    if print_statevector {
        let amps = state.amps_f64();
        output::format_statevector(out, &amps, n)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in paulis {
            let v = backend
                .expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    Ok(())
}

/// Stabilizer-backend run path. Supports `--shots` and `--expectation`;
/// rejects `--statevector` (a tableau has no dense amplitudes).
#[allow(clippy::too_many_arguments)]
fn run_stabilizer<W: Write>(
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    statevector_requested: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed: Option<u64>,
    seed_label: &str,
    out: &mut W,
) -> Result<()> {
    if statevector_requested {
        return Err(anyhow!(
            "the stabilizer backend has no dense state vector; drop --statevector \
             (use --shots and/or --expectation instead)"
        ));
    }
    let mut backend = match seed {
        Some(s) => StabilizerBackend::with_seed(s),
        None => StabilizerBackend::new(),
    };
    let state = run(&mut backend, circuit).context("running circuit (stabilizer)")?;

    if let Some(shots) = effective_shots {
        let samples = backend
            .sample(&state, shots)
            .context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, seed_label)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in paulis {
            let v = backend
                .expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    Ok(())
}

/// MPS-backend run path. Supports `--shots` and `--expectation`; rejects
/// `--statevector` (an MPS exposes no dense amplitude vector).
#[allow(clippy::too_many_arguments)]
fn run_mps<W: Write>(
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    statevector_requested: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed: Option<u64>,
    max_bond: usize,
    max_error: Option<f64>,
    seed_label: &str,
    out: &mut W,
) -> Result<()> {
    if statevector_requested {
        return Err(anyhow!(
            "the MPS backend has no dense state vector; drop --statevector \
             (use --shots and/or --expectation instead)"
        ));
    }
    let policy = match max_error {
        Some(epsilon) => TruncationPolicy::ErrorBounded { epsilon, max_bond },
        None => TruncationPolicy::FixedBond(max_bond),
    };
    let mut backend = match seed {
        Some(s) => MpsBackend::with_seed(s),
        None => MpsBackend::new(),
    }
    .with_truncation(policy);
    let state = run(&mut backend, circuit).context("running circuit (mps)")?;

    if let Some(shots) = effective_shots {
        let samples = backend
            .sample(&state, shots)
            .context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, seed_label)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in paulis {
            let v = backend
                .expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    writeln!(
        out,
        "truncation error: {:.3e}; max bond χ: {}",
        state.truncation_error(),
        state.max_bond_reached()
    )?;
    Ok(())
}

/// Whether the Metal GPU backend can actually run here — used by the `auto`
/// heuristic to decide whether to route large dense circuits to the GPU
/// (P5.6-07). True only on a macOS+`metal` build where a Metal device is
/// acquirable, so headless macOS (e.g. CI) and every other platform correctly
/// stay on the CPU. Probing constructs a backend (acquire device + compile
/// pipelines); that cost is paid once per `auto` run and avoided entirely when
/// the user fixes `--backend` explicitly.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_available() -> bool {
    aleph_metal::MetalSvBackend::new().is_ok()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn metal_available() -> bool {
    false
}

/// Metal GPU state-vector run path (macOS + `metal` feature). FP32; supports
/// `--shots`, `--statevector`, and `--expectation` through the shared
/// `run_with_backend` helper (the Metal state implements [`AmpsF64`]). The
/// backend is built fallibly — GPU/device acquisition can fail — and a failure
/// surfaces as a clear error rather than a silent CPU fallback.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn run_metal<W: Write>(
    circuit: &aleph_ir::Circuit,
    effective_shots: Option<u32>,
    print_statevector: bool,
    paulis: &[(String, aleph_core::PauliString)],
    n: u32,
    seed: Option<u64>,
    seed_label: &str,
    out: &mut W,
) -> Result<()> {
    let backend = match seed {
        Some(s) => aleph_metal::MetalSvBackend::with_seed(s),
        None => aleph_metal::MetalSvBackend::new(),
    }
    .map_err(|e| anyhow!("initializing the Metal GPU backend: {e}"))?;
    run_with_backend(
        backend,
        circuit,
        effective_shots,
        print_statevector,
        paulis,
        n,
        seed_label,
        out,
    )
}

/// Stub when the Metal backend is not compiled in (non-macOS, or built without
/// `--features metal`). `--backend metal` then fails with a clear, actionable
/// message instead of silently running on the CPU.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
fn run_metal<W: Write>(
    _circuit: &aleph_ir::Circuit,
    _effective_shots: Option<u32>,
    _print_statevector: bool,
    _paulis: &[(String, aleph_core::PauliString)],
    _n: u32,
    _seed: Option<u64>,
    _seed_label: &str,
    _out: &mut W,
) -> Result<()> {
    Err(anyhow!(
        "the Metal GPU backend is not available in this build; \
         rebuild aleph with `--features metal` on macOS (Apple Silicon)"
    ))
}

/// Run a circuit once and report parse / execute / sample(1024)
/// wall-times.  No statistics; criterion is the source of truth for
/// regression-tracking perf numbers.
pub fn bench_circuit<W: Write>(qasm_path: &Path, seed: Option<u64>, out: &mut W) -> Result<()> {
    let source = std::fs::read_to_string(qasm_path)
        .with_context(|| format!("reading QASM file: {}", qasm_path.display()))?;
    let qasm_name = qasm_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| qasm_path.display().to_string());

    let parse_start = Instant::now();
    let circuit =
        aleph_parser::parse(&source).with_context(|| format!("parsing {}", qasm_path.display()))?;
    let parse_dt = parse_start.elapsed();

    let n = circuit.num_qubits();
    let mut backend = match seed {
        Some(s) => NaiveSvBackend::with_seed(s),
        None => NaiveSvBackend::new(),
    };

    let run_start = Instant::now();
    let state = run(&mut backend, &circuit).context("running circuit")?;
    let run_dt = run_start.elapsed();

    let sample_start = Instant::now();
    let _ = backend
        .sample(&state, 1024)
        .context("sampling final state")?;
    let sample_dt = sample_start.elapsed();

    let total = parse_dt + run_dt + sample_dt;
    output::format_bench_header(out, &qasm_name, n)?;
    output::format_bench_phase(out, "parse", parse_dt)?;
    output::format_bench_phase(out, "run", run_dt)?;
    output::format_bench_phase(out, "sample(1024)", sample_dt)?;
    output::format_bench_phase(out, "total", total)?;
    Ok(())
}

/// Build a `NoiseModel` from `--noise <preset>:<p>` strings. `depol:<p>`
/// attaches depolarizing error to every distinct 1q and 2q gate name present
/// in `circuit`; `readout:<p>` attaches a symmetric readout flip to every
/// qubit. 3q+ gates get no preset noise (depolarizing_error supports 1q/2q).
fn build_noise_model(
    presets: &[String],
    circuit: &aleph_ir::Circuit,
    n: u32,
) -> Result<aleph_sv::noise::NoiseModel> {
    use aleph_sv::noise::{depolarizing_error, NoiseModel, ReadoutError};

    let mut nm = NoiseModel::new();
    for raw in presets {
        let (kind, val) = raw
            .split_once(':')
            .ok_or_else(|| anyhow!("--noise expects <preset>:<p>, got {raw:?}"))?;
        let p: f64 = val
            .parse()
            .map_err(|_| anyhow!("--noise {raw:?}: {val:?} is not a number"))?;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(anyhow!("--noise {raw:?}: p must be in [0,1], got {p}"));
        }
        match kind {
            "depol" => {
                // Distinct gate name -> arity for the 1q/2q gates present.
                // Gate::name() is &'static, so this borrows nothing from the
                // circuit.
                let mut seen: std::collections::BTreeMap<&'static str, u8> =
                    std::collections::BTreeMap::new();
                for inst in circuit.instructions() {
                    if let aleph_ir::Instruction::Gate(gi) = inst {
                        let arity = gi.qubits.len();
                        if arity == 1 || arity == 2 {
                            seen.insert(gi.gate.name(), arity as u8);
                        }
                    }
                }
                for (name, arity) in seen {
                    nm.add_all_qubit_quantum_error(depolarizing_error(p, arity), &[name]);
                }
            }
            "readout" => {
                let re = ReadoutError::new([[1.0 - p, p], [p, 1.0 - p]]);
                for q in 0..n {
                    nm.add_readout_error(re, q);
                }
            }
            other => {
                return Err(anyhow!(
                    "unknown --noise preset {other:?} (expected depol or readout)"
                ))
            }
        }
    }
    Ok(nm)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QASM_BELL: &str =
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n";

    /// Tiny stdlib-only temp directory, RAII-cleaned on drop.
    ///
    /// Uniqueness comes from PID + a process-local atomic counter.
    /// Cargo runs tests in parallel by default; two threads sampling
    /// `SystemTime::now().as_nanos()` simultaneously on macOS could
    /// produce the same value and collide on the directory name,
    /// causing flaky test failures via `create_dir(...).unwrap()`.
    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut p = std::env::temp_dir();
            p.push(format!("aleph-cli-test-{}-{n}", std::process::id()));
            std::fs::create_dir(&p).unwrap();
            Self { path: p }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Write the QASM into a temp file and return its path.  Caller
    /// owns the returned `(PathBuf, TempDir)` and the file is removed
    /// when `TempDir` drops.
    fn temp_qasm(src: &str) -> (std::path::PathBuf, TempDir) {
        let dir = TempDir::new();
        let path = dir.path.join("circuit.qasm");
        std::fs::write(&path, src).unwrap();
        (path, dir)
    }

    #[test]
    fn default_view_is_shots_1024() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        run_circuit(
            &path,
            None,
            false,
            false,
            &[],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("counts (1024 shots, seed=0):"));
    }

    #[test]
    fn seeded_run_is_reproducible() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut a = Vec::new();
        let mut b = Vec::new();
        run_circuit(
            &path,
            Some(256),
            false,
            false,
            &[],
            Some(42),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut a,
        )
        .unwrap();
        run_circuit(
            &path,
            Some(256),
            false,
            false,
            &[],
            Some(42),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut b,
        )
        .unwrap();
        assert_eq!(a, b, "two seed=42 invocations diverged");
    }

    #[test]
    fn statevector_cap_rejects_n11() {
        // Build an 11-qubit circuit: empty body, identity state.
        let qasm = "OPENQASM 3.0;\nqubit[11] q;\n";
        let (path, _dir) = temp_qasm(qasm);
        let mut out = Vec::new();
        let err = run_circuit(
            &path,
            None,
            true,
            false,
            &[],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("2^11"), "msg = {msg}");
        assert!(msg.contains("--force-statevector"), "msg = {msg}");
    }

    #[test]
    fn statevector_force_bypasses_cap() {
        let qasm = "OPENQASM 3.0;\nqubit[11] q;\n";
        let (path, _dir) = temp_qasm(qasm);
        let mut out = Vec::new();
        run_circuit(
            &path,
            None,
            true,
            true,
            &[],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("statevector (11 qubits"));
    }

    #[test]
    fn expectation_oor_qubit_errors() {
        // Bell circuit is 2 qubits; --expectation "ZZZ" references q2.
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        let err = run_circuit(
            &path,
            None,
            false,
            false,
            &["ZZZ".to_string()],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("references qubit 2"));
        assert!(msg.contains("circuit has 2 qubits"));
    }

    #[test]
    fn expectation_bad_pauli_errors() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        let err = run_circuit(
            &path,
            None,
            false,
            false,
            &["ABC".to_string()],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("parsing --expectation"));
        assert!(msg.contains("'A'"));
    }

    #[test]
    fn expectation_zz_on_bell_is_one() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        run_circuit(
            &path,
            None,
            false,
            false,
            &["ZZ".to_string()],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("expectation values:"));
        assert!(s.contains("ZZ"));
        assert!(s.contains("+1.000"));
    }

    #[test]
    fn bench_writes_all_four_lines() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        bench_circuit(&path, Some(0), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("bench circuit.qasm (n=2):"));
        assert!(s.contains("parse"));
        assert!(s.contains("run"));
        assert!(s.contains("sample(1024)"));
        assert!(s.contains("total"));
    }

    /// The Metal GPU backend runs a GHZ circuit through the CLI path and yields
    /// the expected state vector. Device-or-skip: on a device-less mac the
    /// backend init fails and we accept the clear error instead. Only compiled
    /// with `--features metal` on macOS.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_backend_runs_ghz_statevector() {
        let qasm = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\n\
                    h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\n";
        let (path, _dir) = temp_qasm(qasm);
        let mut out = Vec::new();
        let res = run_circuit(
            &path,
            None,
            true,
            false,
            &[],
            Some(1),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Metal),
            128,
            None,
            &[],
            &mut out,
        );
        match res {
            Ok(()) => {
                let s = String::from_utf8(out).unwrap();
                assert!(s.contains("statevector (3 qubits"), "out = {s}");
                // GHZ: |000⟩ and |111⟩ ≈ 0.7071 (FP32), all others ≈ 0.
                assert!(s.contains("|000⟩") && s.contains("|111⟩"), "out = {s}");
            }
            Err(e) => {
                // No Metal device on this box — accept the device-init error.
                let msg = format!("{e:#}");
                assert!(msg.contains("Metal GPU backend"), "unexpected error: {msg}");
            }
        }
    }

    /// On a build without the Metal backend (non-macOS, or no `--features
    /// metal`), `--backend metal` fails with a clear, actionable message rather
    /// than silently running on the CPU. Runs in default CI.
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn metal_backend_unavailable_errors_clearly() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut out = Vec::new();
        let err = run_circuit(
            &path,
            None,
            false,
            false,
            &[],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Metal),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Metal GPU backend is not available"),
            "msg = {msg}"
        );
        assert!(msg.contains("--features metal"), "msg = {msg}");
    }

    #[test]
    fn missing_file_error_names_path() {
        let mut out = Vec::new();
        let err = run_circuit(
            std::path::Path::new("/definitely/does/not/exist.qasm"),
            None,
            false,
            false,
            &[],
            Some(0),
            Precision::F64,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("reading QASM file"));
        assert!(msg.contains("does/not/exist.qasm"));
    }
}
