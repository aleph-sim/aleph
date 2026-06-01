//! P2-04 sweep instrument. Applies ONE gate class at ONE target on a
//! large state, repeatedly, so a driver can vary ALEPH_PAR_MIN_AMPS /
//! ALEPH_PAR_GRAIN (the env override in `tuning::resolve_policy`) and
//! read criterion's median per grid point.
//!
//! Env:
//!   ALEPH_TUNE_GATE   = h|zdiag|x|dense|cnot|cz|swap|cphase
//!   ALEPH_TUNE_TARGET = target qubit index (default 12)
//!   ALEPH_TUNE_N      = qubit count (default 25)
//!   ALEPH_PAR_MIN_AMPS / ALEPH_PAR_GRAIN = the knobs under test

use aleph_core::Complex;
use aleph_sv::kernels;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn seeded_state(n: u32) -> Vec<Complex> {
    (0..(1usize << n))
        .map(|k| {
            let h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let theta = (h >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
            Complex::new(theta.cos(), theta.sin())
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let gate = std::env::var("ALEPH_TUNE_GATE").unwrap_or_else(|_| "cphase".into());
    let t = env_u32("ALEPH_TUNE_TARGET", 12);
    let n = env_u32("ALEPH_TUNE_N", 25);
    let mut s = seeded_state(n);
    let id = format!("chunk_tune/{gate}/t{t}/n{n}");

    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    let sq = std::f64::consts::FRAC_1_SQRT_2;
    // second qubit for 2q gates: a distinct qubit near the target.
    let t2 = if t == 0 { 1 } else { t - 1 };
    let q2 = [t, t2];

    macro_rules! run1 {
        ($m:expr) => {{
            let m = $m;
            c.bench_function(&id, |b| {
                b.iter(|| kernels::aos::apply_1q(black_box(&mut s), t, &[], &m))
            });
        }};
    }
    macro_rules! run2 {
        ($m:expr) => {{
            let m = $m;
            c.bench_function(&id, |b| {
                b.iter(|| kernels::aos::apply_2q(black_box(&mut s), q2, &[], &m))
            });
        }};
    }

    match gate.as_str() {
        "h" => run1!([
            [Complex::new(sq, 0.0), Complex::new(sq, 0.0)],
            [Complex::new(sq, 0.0), Complex::new(-sq, 0.0)]
        ]),
        "zdiag" => run1!([[o, z], [z, Complex::new(-1.0, 0.0)]]),
        "x" => run1!([[z, o], [o, z]]),
        "cphase" => run2!([
            [o, z, z, z],
            [z, o, z, z],
            [z, z, o, z],
            [z, z, z, Complex::new(0.0, 1.0)]
        ]),
        "cnot" => run2!([[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]),
        "cz" => run2!([
            [o, z, z, z],
            [z, o, z, z],
            [z, z, o, z],
            [z, z, z, Complex::new(-1.0, 0.0)]
        ]),
        "swap" => run2!([[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, o]]),
        "dense" => run2!([
            [Complex::new(0.5, 0.5), z, z, Complex::new(0.5, -0.5)],
            [z, o, z, z],
            [z, z, o, z],
            [Complex::new(0.5, -0.5), z, z, Complex::new(0.5, 0.5)]
        ]),
        other => panic!("unknown ALEPH_TUNE_GATE={other}"),
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
