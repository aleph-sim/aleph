//! Public CLI raw/optimized final-state and structural regression checks.
use assert_cmd::Command;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture(body: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "aleph-cli-optimized-{}-{}.qasm",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(
        &path,
        format!("OPENQASM 3.0; include \"stdgates.inc\"; {body}"),
    )
    .unwrap();
    path
}

fn output(path: &PathBuf, backend: &str, precision: &str, raw: bool) -> std::process::Output {
    let mut cmd = Command::cargo_bin("aleph").unwrap();
    cmd.arg("run").arg(path).args([
        "--backend",
        backend,
        "--precision",
        precision,
        "--shots",
        "32",
        "--seed",
        "7",
        "--statevector",
        "--expectation",
        "YII",
        "--expectation",
        "IZI",
        "--execution-stats",
    ]);
    if raw {
        cmd.arg("--raw");
    }
    cmd.assert().success().get_output().clone()
}

fn amplitudes(bytes: &[u8]) -> Vec<(f64, f64)> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|l| l.contains("|a|²"))
        .map(|line| {
            let parts: Vec<_> = line.split_whitespace().collect();
            (
                parts[1].parse().unwrap(),
                parts[2].trim_end_matches('i').parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn public_views_preserve_measurements_controls_phases_and_barriers() {
    let sources = [
        "qubit[3] q; bit[1] c; x q[1]; h q[0]; s q[0]; barrier q; measure q[0] -> c[0]; h q[2]; s q[2];",
        "qubit[3] q; h q[0]; h q[1]; s q[0]; t q[1]; ccx q[0],q[1],q[2]; barrier q; cx q[2],q[0]; h q[2];",
        "qubit[3] q; h q[0]; s q[0]; x q[1]; h q[2]; t q[2]; s q[2];",
    ];
    let mut backends = vec![("sv", "f64", 1e-11), ("sv", "f32", 2e-5)];
    if std::env::var_os("ALEPH_TEST_CLI_METAL").is_some() {
        backends.push(("metal", "auto", 2e-5));
    }
    for source in sources {
        let path = fixture(source);
        let exact = output(&path, "sv", "f64", true);
        let reference = amplitudes(&exact.stdout);
        for &(backend, precision, tol) in &backends {
            let opt = output(&path, backend, precision, false);
            let raw = output(&path, backend, precision, true);
            assert!(String::from_utf8_lossy(&opt.stderr).contains("final-state optimized"));
            assert!(String::from_utf8_lossy(&raw.stderr).contains("pipeline: raw"));
            for (bytes, tolerance) in [(&opt.stdout, tol), (&raw.stdout, tol)] {
                let got = amplitudes(bytes);
                assert_eq!(got.len(), reference.len());
                for ((re, im), (rr, ri)) in got.iter().zip(&reference) {
                    assert!((re - rr).abs() < tolerance && (im - ri).abs() < tolerance);
                }
            }
            // Compare counts and expectation views, not just amplitudes.
            let views = |bytes: &[u8]| {
                String::from_utf8_lossy(bytes)
                    .lines()
                    .filter(|l| !l.contains("|a|²") && !l.starts_with("statevector"))
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            let a = views(&raw.stdout);
            let b = views(&opt.stdout);
            // Histograms must use the same seed stream; expectations allow FP roundoff.
            assert_eq!(
                &a[..a.iter().position(|l| l == "expectation values:").unwrap()],
                &b[..b.iter().position(|l| l == "expectation values:").unwrap()]
            );
            for (x, y) in a.iter().zip(&b).filter(|(x, _)| {
                x.trim_start().starts_with("YII") || x.trim_start().starts_with("IZI")
            }) {
                let value = |s: &str| s.split_whitespace().last().unwrap().parse::<f64>().unwrap();
                assert!((value(x) - value(y)).abs() < tol);
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn public_optimized_path_reports_retained_fusion_and_smaller_plan() {
    let path =
        fixture("qubit[3] q; h q[0]; s q[0]; t q[0]; s q[0]; cx q[0],q[1]; h q[1]; cx q[1],q[2];");
    for precision in ["f64", "f32"] {
        let out = output(&path, "sv", precision, false);
        let stats = String::from_utf8_lossy(&out.stderr);
        let metric = |name: &str| {
            stats
                .split(&format!("{name}="))
                .nth(1)
                .unwrap()
                .split(';')
                .next()
                .unwrap()
                .trim()
                .parse::<usize>()
                .unwrap()
        };
        assert!(metric("planned_instructions") < metric("input_instructions"));
        assert!(metric("diagonal_phases") + metric("fused_gates") + metric("tiled_blocks") > 0);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unsupported_reset_is_not_optimized_away() {
    let path = fixture("qubit[1] q; x q[0]; reset q[0];");
    for precision in ["f64", "f32"] {
        for raw in [false, true] {
            let mut cmd = Command::cargo_bin("aleph").unwrap();
            cmd.arg("run")
                .arg(&path)
                .args(["--backend", "sv", "--precision", precision]);
            if raw {
                cmd.arg("--raw");
            }
            cmd.assert().failure();
        }
    }
    std::fs::remove_file(path).unwrap();
}
