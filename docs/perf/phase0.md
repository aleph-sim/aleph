# Phase 0 — performance report

Closes the ROADMAP § 7 Phase-0 exit criterion: *"benchmark harness
produces reports"*.  Numbers below are the **local-dev baseline**
(macOS, Apple-silicon M-series, release build, no `target-cpu=native`)
captured at the close of Phase 0 — they are **indicative**, not
authoritative.  The authoritative baseline for Phase-1 perf claims is
the [self-hosted Linux x86_64 runner] (`aleph-bench-server`), tracked
continuously on [bencher.dev] via `.github/workflows/bench.yml`.

[self-hosted Linux x86_64 runner]: https://github.com/aleph-sim/aleph/actions/workflows/bench.yml
[bencher.dev]: https://bencher.dev

## What Phase 0 shipped

A working pipeline: **OpenQASM 3.0 source → IR → naive single-thread
state-vector backend → measurement primitives**.  Plus an oracle
harness that compares every committed circuit against Qiskit Aer at
`1e-10` per amplitude.

Phase 0's perf goal was modest: produce a baseline so Phase 1 (SoA,
SIMD, gate fusion, multi-threading) has something to beat.  This
report is that baseline.

## Tier-1 algorithm benchmarks

End-to-end `parse → run → final state` time via
`aleph_backend::run(&mut NaiveSvBackend, &Circuit)`, measured by
criterion (1 warm-up + 1 sample per `--quick` run, ~100 samples per
default run).

| Benchmark | n | Time (median) | Throughput |
|---|---|---|---|
| `ghz/10`    | 10 | 23.2 µs   | 44.2 Mamp/s |
| `ghz/15`    | 15 | 1.08 ms   | 30.2 Mamp/s |
| `ghz/20`    | 20 | 47.0 ms   | 22.3 Mamp/s |
| `ghz/25`    | 25 | 1.90 s    | 17.6 Mamp/s |
| `qft/10`    | 10 | 53.3 µs   | 192 Melem/s |
| `qft/15`    | 15 | 3.42 ms   | 144 Melem/s |
| `qft/20`    | 20 | ~ 100 ms  | (run in CI) |
| `random/n20 d=20` | 20 | 1.75 s | 12.0 Melem/s |
| `bell` (n=2) | 2 | 189 ns    | — |

(QFT throughput is `n · 2^n` per the circuit's complexity; GHZ is `2^n`.)

### Reading the throughput numbers

GHZ throughput **decreases** with n because the state-vector backend
sweeps 2^n amplitudes per CNOT, but cache misses dominate once the
working set leaves L2 (~1 MiB at n=17) and then L3 (~16 MiB at n=21).
The 22→17 Mamp/s drop from n=20 → n=25 is a memory-bandwidth ceiling,
exactly the regime Phase 1 SoA + SIMD targets.

## Sampling and expectation micro-benches

From P0-11 (`crates/aleph-sv/benches/{sample,expectation}.rs`),
captured during that PR (commit `ee9998e`):

| Bench | Time | Notes |
|---|---|---|
| `sample/uniform_n10_shots100k` | 689 µs | post-alias (P0-11 swap from inverse-CDF saved 56%) |
| `sample/uniform_n16_shots100k` | 895 µs | post-alias, 75% faster than CDF |
| `expectation/exp_z_chain_n10` | 1.69 µs | Z fast path (P0-11) — 86% faster than slow copy+apply |
| `expectation/exp_x_chain_n10` | 12.3 µs | slow path (X requires kernel apply) |

Pre-P0-11 baselines and the full discussion are in
`docs/superpowers/specs/2026-05-25-p0-11-primitives-design.md` § 7.

## Oracle correctness — not a perf number but worth recording

28 fixture circuits × 2 (state-vector + distribution oracles) = 56
generated tests; all pass.  State-vector tolerance is `1e-10` per
amplitude; distribution oracle uses a `5σ + 1e-6` band over 100k
samples per fixture.  Total wall time under `cargo test -p
aleph-oracle`: ~0.55 s.

## How to refresh this report

```bash
# Full benches (slow): ~ several minutes
cargo bench --workspace

# Quick sanity: ~ 10s per file (criterion-quick mode)
cargo bench -p aleph-benches -- --quick

# 25-qubit GHZ (gated #[ignore]; release-only practical)
cargo test --release -p aleph-sv --test tier1 -- --ignored ghz_25_runs
```

For canonical numbers, push to `main` and read the bencher.dev
timeline — CI runs `bench.yml` on the self-hosted EPYC box.

## What Phase 1 will measure against

Phase 1 (ROADMAP § Phase 1) targets:

* P1-01 SoA layout — expects ~1.5–2× on `qft/20` purely from cache effects.
* P1-02 bit-manipulation 1q gate — expects 2–3× on `qft/20`.
* P1-03 AVX2 1q — expects further 2–4× on AVX2 hardware.
* P1-05 specialised Pauli-X — expects 3–10× over the generic 1q kernel.
* P1-09/P1-10 gate fusion — expects per-pass amplitude-walk savings.
* P1-14 Phase 1 performance report — successor to this document.

Each Phase-1 PR ships before/after criterion comparisons against the
baseline numbers above (or rather, against the CI-tracked baseline on
the EPYC runner, which is the source of truth).
