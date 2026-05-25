//! Business logic: run_circuit / bench_circuit.  See spec §4.2.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use aleph_backend::{run, Backend};
use aleph_sv::NaiveSvBackend;

use crate::output;
use crate::pauli::parse_pauli_arg;

/// Default shot count when the user passes no view flag.
pub const DEFAULT_SHOTS: u32 = 1024;

/// State-vector cap that gates `--statevector` without
/// `--force-statevector`.  10 qubits = 1024 amplitudes, comfortable
/// on a terminal; 28 qubits = 256 M lines, decidedly not.
pub const STATEVECTOR_CAP_QUBITS: u32 = 10;

/// Parse + run + print views (counts / statevector / expectation) per
/// the user's flags.  Validates all CLI arguments BEFORE running the
/// circuit so a bad input does not produce partial output.
pub fn run_circuit<W: Write>(
    qasm_path: &Path,
    shots_opt: Option<u32>,
    print_statevector: bool,
    force_statevector: bool,
    expectations: &[String],
    seed: Option<u64>,
    out: &mut W,
) -> Result<()> {
    // 1. Read + parse.
    let source = std::fs::read_to_string(qasm_path)
        .with_context(|| format!("reading QASM file: {}", qasm_path.display()))?;
    let circuit =
        aleph_parser::parse(&source).with_context(|| format!("parsing {}", qasm_path.display()))?;
    let n = circuit.num_qubits();

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

    // 3. Statevector cap check.
    if print_statevector && !force_statevector && n > STATEVECTOR_CAP_QUBITS {
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

    // 5. Allocate backend + run circuit once.
    let mut backend = match seed {
        Some(s) => NaiveSvBackend::with_seed(s),
        None => NaiveSvBackend::new(),
    };
    let state = run(&mut backend, &circuit).context("running circuit")?;

    // 6. Print requested views.
    let seed_label = match seed {
        Some(s) => format!("seed={s}"),
        None => "seed=entropy".to_string(),
    };
    if let Some(shots) = effective_shots {
        let samples = backend
            .sample(&state, shots)
            .context("sampling final state")?;
        output::format_counts(out, &samples, shots, n, &seed_label)?;
    }
    if print_statevector {
        output::format_statevector(out, state.amplitudes(), n)?;
    }
    if !paulis.is_empty() {
        writeln!(out, "expectation values:")?;
        for (raw, ps) in &paulis {
            let v = backend
                .expectation_value(&state, ps)
                .with_context(|| format!("computing expectation value for {raw:?}"))?;
            output::format_expectation(out, raw, v)?;
        }
    }
    Ok(())
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
        run_circuit(&path, None, false, false, &[], Some(0), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("counts (1024 shots, seed=0):"));
    }

    #[test]
    fn seeded_run_is_reproducible() {
        let (path, _dir) = temp_qasm(QASM_BELL);
        let mut a = Vec::new();
        let mut b = Vec::new();
        run_circuit(&path, Some(256), false, false, &[], Some(42), &mut a).unwrap();
        run_circuit(&path, Some(256), false, false, &[], Some(42), &mut b).unwrap();
        assert_eq!(a, b, "two seed=42 invocations diverged");
    }

    #[test]
    fn statevector_cap_rejects_n11() {
        // Build an 11-qubit circuit: empty body, identity state.
        let qasm = "OPENQASM 3.0;\nqubit[11] q;\n";
        let (path, _dir) = temp_qasm(qasm);
        let mut out = Vec::new();
        let err = run_circuit(&path, None, true, false, &[], Some(0), &mut out).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("2^11"), "msg = {msg}");
        assert!(msg.contains("--force-statevector"), "msg = {msg}");
    }

    #[test]
    fn statevector_force_bypasses_cap() {
        let qasm = "OPENQASM 3.0;\nqubit[11] q;\n";
        let (path, _dir) = temp_qasm(qasm);
        let mut out = Vec::new();
        run_circuit(&path, None, true, true, &[], Some(0), &mut out).unwrap();
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
            &mut out,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("reading QASM file"));
        assert!(msg.contains("does/not/exist.qasm"));
    }
}
