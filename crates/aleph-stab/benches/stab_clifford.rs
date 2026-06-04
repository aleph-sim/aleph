//! P3-01 perf AC: a 1000-qubit, depth-100 random Clifford circuit must
//! run in < 1s. Run on a *verified-idle* box (CLAUDE.md idle-check).
//!
//! Run: cargo bench -p aleph-stab --bench stab_clifford
//! For the <1s assertion as a test: the bench prints per-iter time;
//! compare against the 1s budget in the PR writeup.

use aleph_stab::Tableau;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Deterministic xorshift so the bench is reproducible without an RNG dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn run_circuit(n: usize, depth: usize) {
    let mut t = Tableau::new(n);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..depth {
        // one layer ≈ n gates, ~half 2-qubit
        for _ in 0..n {
            match rng.below(6) {
                0 => t.h(rng.below(n as u64) as usize).unwrap(),
                1 => t.s(rng.below(n as u64) as usize).unwrap(),
                2 => t.x_gate(rng.below(n as u64) as usize).unwrap(),
                3 => t.z_gate(rng.below(n as u64) as usize).unwrap(),
                _ => {
                    let a = rng.below(n as u64) as usize;
                    let mut b = rng.below(n as u64) as usize;
                    if a == b {
                        b = (b + 1) % n;
                    }
                    t.cnot(a, b).unwrap();
                }
            }
        }
    }
    black_box(t.stabilizers());
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("stab_clifford");
    group.sample_size(10);
    group.bench_function("n1000_depth100", |b| {
        b.iter(|| run_circuit(black_box(1000), black_box(100)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
