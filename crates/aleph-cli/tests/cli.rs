//! `assert_cmd` end-to-end coverage of the `aleph` binary.  Reuses
//! oracle/circuits/ fixtures for the happy paths; synthesises one
//! 12-qubit QASM inline for the statevector-cap tests.

use assert_cmd::Command;
use predicates::str::contains;

fn aleph() -> Command {
    Command::cargo_bin("aleph").unwrap()
}

fn bell_path() -> std::path::PathBuf {
    aleph_oracle::workspace_path("oracle/circuits/bell_phi_plus.qasm")
}

fn ghz3_path() -> std::path::PathBuf {
    aleph_oracle::workspace_path("oracle/circuits/ghz_3.qasm")
}

fn surface_code_path() -> std::path::PathBuf {
    aleph_oracle::workspace_path("oracle/circuits/surface_code_cycle.qasm")
}

/// Unique temp-file path; avoids collision when tests run in
/// parallel by using a process-local atomic counter.
fn unique_tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("aleph-cli-it-{}-{n}-{name}", std::process::id()))
}

#[test]
fn bell_state_counts() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--shots", "1024", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("counts (1024 shots, seed=0):"))
        .stdout(contains("|00⟩"))
        .stdout(contains("|11⟩"));
}

#[test]
fn bell_statevector() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--statevector"])
        .assert()
        .success()
        .stdout(contains("statevector (2 qubits, 4 amplitudes):"))
        .stdout(contains("|00⟩"))
        .stdout(contains("|11⟩"))
        .stdout(contains("|a|² = 0.500000"));
}

#[test]
fn bell_expectation_zz() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--expectation", "ZZ"])
        .assert()
        .success()
        .stdout(contains("expectation values:"))
        .stdout(contains("ZZ"))
        .stdout(contains("+1.000"));
}

#[test]
fn bell_multiple_expects() {
    let out = aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--expectation", "ZZ", "--expectation", "XX"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    let pos_zz = s.find("ZZ").expect("ZZ line missing");
    let pos_xx = s.find("XX").expect("XX line missing");
    assert!(pos_zz < pos_xx, "expectations not in arg order");
}

#[test]
fn ghz_3_default_shots() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--seed", "0"])
        .assert()
        .success()
        .stdout(contains("counts (1024 shots, seed=0):"))
        .stdout(contains("|000⟩"))
        .stdout(contains("|111⟩"));
}

#[test]
fn bench_prints_phases() {
    aleph()
        .args(["bench"])
        .arg(bell_path())
        .args(["--seed", "0"])
        .assert()
        .success()
        .stdout(contains("bench bell_phi_plus.qasm (n=2):"))
        .stdout(contains("parse"))
        .stdout(contains("run"))
        .stdout(contains("sample(1024)"))
        .stdout(contains("total"));
}

#[test]
fn seed_reproducibility() {
    let one = aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--shots", "512", "--seed", "0"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let two = aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--shots", "512", "--seed", "0"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(one, two, "two seeded runs diverged");
}

#[test]
fn missing_file_error() {
    aleph()
        .args(["run", "/definitely/does/not/exist.qasm"])
        .assert()
        .failure()
        .stderr(contains("reading QASM file"));
}

#[test]
fn parse_error_propagates() {
    let tmp = unique_tmp("parse-error.qasm");
    std::fs::write(&tmp, "this is not qasm\n").unwrap();
    let result = aleph().args(["run"]).arg(&tmp).assert().failure();
    let _ = std::fs::remove_file(&tmp);
    result.stderr(contains("parsing"));
}

#[test]
fn expectation_oor_qubit_error() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--expectation", "ZZZ"])
        .assert()
        .failure()
        .stderr(contains("references qubit 2"))
        .stderr(contains("circuit has 2 qubits"));
}

#[test]
fn expectation_bad_pauli_error() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--expectation", "ABC"])
        .assert()
        .failure()
        .stderr(contains("parsing --expectation"))
        .stderr(contains("'A'"));
}

#[test]
fn statevector_n_cap_rejects() {
    let qasm = "OPENQASM 3.0;\nqubit[12] q;\n";
    let tmp = unique_tmp("bigsv.qasm");
    std::fs::write(&tmp, qasm).unwrap();
    let result = aleph()
        .args(["run"])
        .arg(&tmp)
        .args(["--statevector"])
        .assert()
        .failure();
    let _ = std::fs::remove_file(&tmp);
    result
        .stderr(contains("2^12"))
        .stderr(contains("--force-statevector"));
}

#[test]
fn statevector_force_bypasses_cap() {
    let qasm = "OPENQASM 3.0;\nqubit[11] q;\n";
    let tmp = unique_tmp("force.qasm");
    std::fs::write(&tmp, qasm).unwrap();
    let result = aleph()
        .args(["run"])
        .arg(&tmp)
        .args(["--statevector", "--force-statevector"])
        .assert()
        .success();
    let _ = std::fs::remove_file(&tmp);
    result.stdout(contains("statevector (11 qubits"));
}

#[test]
fn precision_f32_ghz_statevector() {
    // f32 GHZ-3: |000⟩ and |111⟩ carry 1/√2 ≈ 0.7071, |a|² ≈ 0.5.
    // Single precision (~1e-6) easily resolves this to the 6-digit
    // |a|² formatting, so the populated states still show 0.500000.
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--statevector", "--precision", "f32"])
        .assert()
        .success()
        .stdout(contains("statevector (3 qubits, 8 amplitudes):"))
        .stdout(contains("|000⟩"))
        .stdout(contains("|111⟩"))
        .stdout(contains("|a|² = 0.500000"));
}

#[test]
fn precision_f32_ghz_counts_are_all_or_nothing() {
    // GHZ samples only ever collapse to |000⟩ or |111⟩; the f32 path
    // must preserve that — no intermediate basis states.
    // Explicit --backend statevector is required: without it, `auto` would
    // route this all-Clifford circuit to the stabilizer backend, which
    // ignores --precision f32 and wouldn't exercise the f32 SV path.
    let out = aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args([
            "--backend",
            "statevector",
            "--shots",
            "256",
            "--seed",
            "0",
            "--precision",
            "f32",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("counts (256 shots, seed=0):"));
    // Only the two GHZ basis states may appear.
    for forbidden in ["|001⟩", "|010⟩", "|011⟩", "|100⟩", "|101⟩", "|110⟩"] {
        assert!(
            !s.contains(forbidden),
            "unexpected basis state {forbidden}: {s}"
        );
    }
}

#[test]
fn precision_default_equals_explicit_f64() {
    // The default (no --precision) must be byte-identical to --precision f64.
    let default_out = aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--statevector"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let f64_out = aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--statevector", "--precision", "f64"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(
        default_out, f64_out,
        "default precision diverged from explicit f64"
    );
}

#[test]
fn help_lists_subcommands() {
    aleph()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("run"))
        .stdout(contains("bench"));
}

#[test]
fn version_flag() {
    aleph()
        .arg("--version")
        .assert()
        .success()
        .stdout(contains("aleph 0.0.0"));
}

#[test]
fn stabilizer_backend_runs_surface_code() {
    aleph()
        .args(["run"])
        .arg(surface_code_path())
        .args(["--backend", "stabilizer", "--shots", "1024", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("counts (1024 shots, seed=0):"));
}

#[test]
fn stabilizer_backend_rejects_non_clifford() {
    // qft_5 contains non-Clifford rz gates.
    let qft = aleph_oracle::workspace_path("oracle/circuits/qft_5.qasm");
    aleph()
        .args(["run"])
        .arg(qft)
        .args(["--backend", "stabilizer", "--shots", "16"])
        .assert()
        .failure()
        .stderr(contains("not supported"));
}

#[test]
fn stabilizer_backend_rejects_statevector() {
    aleph()
        .args(["run"])
        .arg(surface_code_path())
        .args(["--backend", "stabilizer", "--statevector"])
        .assert()
        .failure()
        .stderr(contains("no dense state vector"));
}

#[test]
fn mps_backend_runs_bell() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--backend", "mps", "--shots", "1024", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("counts (1024 shots, seed=0):"))
        .stdout(contains("|00⟩"))
        .stdout(contains("|11⟩"));
}

#[test]
fn mps_backend_expectation_zz_on_bell() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--backend", "mps", "--expectation", "ZZ", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("expectation values:"))
        // ⟨ZZ⟩ = +1 on |Φ+⟩, but the MPS path accumulates ~1e-16 SVD/QR
        // rounding (prints +0.999…), so this CLI test only confirms the
        // expectation path is wired; the exact value is pinned in the
        // unit/oracle tests.
        .stdout(contains("ZZ"));
}

#[test]
fn mps_backend_max_bond_runs_ghz() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args([
            "--backend",
            "mps",
            "--max-bond",
            "8",
            "--shots",
            "64",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(contains("counts (64 shots, seed=0):"));
}

#[test]
fn mps_backend_rejects_statevector() {
    aleph()
        .args(["run"])
        .arg(bell_path())
        .args(["--backend", "mps", "--statevector"])
        .assert()
        .failure()
        .stderr(contains("no dense state vector"));
}

#[test]
fn mps_backend_reports_truncation() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--backend", "mps", "--shots", "64", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("truncation error:"))
        .stdout(contains("max bond"));
}

#[test]
fn mps_backend_max_error_runs() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args([
            "--backend",
            "mps",
            "--max-error",
            "1e-8",
            "--shots",
            "64",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .stdout(contains("truncation error:"));
}

#[test]
fn mps_backend_rejects_nonpositive_max_error() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--backend", "mps", "--max-error", "0"])
        .assert()
        .failure()
        .stderr(contains("--max-error must be a positive"));
}
