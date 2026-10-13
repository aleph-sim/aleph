//! Criterion over the real CLI process (parse, planning, execute, sample, format).
//! Build a release CLI first and set ALEPH_BENCH_CLI to its absolute path.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn public_pipeline(c: &mut Criterion) {
    let binary = std::env::var("ALEPH_BENCH_CLI")
        .expect("set ALEPH_BENCH_CLI to a previously built release aleph binary");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workloads = [
        ("bell2", root.join("oracle/circuits/bell_phi_plus.qasm")),
        (
            "qft15",
            root.join("scripts/qiskit-baseline/circuits/qft_n15.qasm"),
        ),
        (
            "brickwall15_d20",
            root.join("scripts/qiskit-baseline/circuits/random_brickwall_n15_d20.qasm"),
        ),
    ];
    for precision in ["f64", "f32"] {
        let mut group = c.benchmark_group(format!("public_cli/{precision}"));
        group
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2));
        for (name, path) in &workloads {
            for raw in [true, false] {
                let mode = if raw { "raw" } else { "optimized" };
                group.bench_function(BenchmarkId::new(*name, mode), |b| {
                    b.iter(|| {
                        let mut cmd = Command::new(&binary);
                        cmd.arg("run")
                            .arg(path)
                            .args([
                                "--backend",
                                "sv",
                                "--precision",
                                precision,
                                "--shots",
                                "32",
                                "--seed",
                                "7",
                            ])
                            .stdout(Stdio::null())
                            .stderr(Stdio::null());
                        if raw {
                            cmd.arg("--raw");
                        }
                        assert!(cmd.status().expect("launch public CLI").success());
                    })
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, public_pipeline);
criterion_main!(benches);
