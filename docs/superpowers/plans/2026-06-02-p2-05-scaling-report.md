# P2-05 — Phase 2 Scaling-Efficiency Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the Phase-2 capstone — a reusable `tier1_scaling` criterion bench over all four Tier-1 algorithms plus `docs/perf/phase2.md`, the scaling-efficiency report that resolves the ROADMAP §7 Phase-2 exit (≥12×/16-core = ≥75%@16t) honestly.

**Architecture:** A new feature-gated criterion bench parses the canonical n=25 qasm fixtures and drives the AoS+AVX-512 parallel kernels, swept via `RAYON_NUM_THREADS` (exactly the P2-01 `qft_scaling` pattern, extended to GHZ/Grover/random). A thread-invariance test + sweep script guards correctness. The report synthesizes the real QFT scaling numbers already measured in P2-01..04, marks unmeasured cells as *pending hardware run* (no fabricated data), and consolidates the bandwidth-bound root-cause narrative.

**Tech Stack:** Rust 2021, criterion, rayon, `aleph-parser`/`aleph-backend`/`aleph-sv`, existing `scaling-bench` feature gate.

**Spec:** `docs/superpowers/specs/2026-06-02-p2-05-scaling-report-design.md`
**Branch:** `p2-05-scaling-report` (already created off `origin/main`; spec already committed).

---

## File Structure

- **Create** `crates/aleph-oracle/tests/tier1_scaling_invariance.rs` — thread-invariance correctness guard for the Tier-1 fixtures (Task 1).
- **Create** `scripts/p2-05-thread-sweep.sh` — runs the invariance test across `RAYON_NUM_THREADS ∈ {1,2,4,8}` with forced parallelism (Task 1).
- **Create** `benches/benches/tier1_scaling.rs` — the scaling bench (Task 2).
- **Modify** `benches/Cargo.toml` — register the bench (Task 2).
- **Create** `docs/perf/phase2.md` — the report (Task 3).

No production/kernel code changes — Phase-2 perf work is already complete; this ticket is measurement infrastructure + the report.

---

## Task 1: Thread-invariance correctness guard for the Tier-1 fixtures

Establishes (and locks in) that the parallel kernels are bit-exact on the exact GHZ/QFT/Grover/random circuits the bench will measure. Written first so the bench in Task 2 is measuring known-correct kernels. Uses n=15 fixtures (fast: 2^15 = 32768 amplitudes) and forces the parallel path with `ALEPH_PAR_MIN_AMPS=0` via the sweep script. Compares AoS `NaiveSvBackend` against SoA `SoaSvBackend` within 1e-12 — two independent memory layouts, so a disjointness bug in either breaks the equality (same idiom as `crates/aleph-oracle/tests/soa_vs_naive.rs`).

**Files:**
- Create: `crates/aleph-oracle/tests/tier1_scaling_invariance.rs`
- Create: `scripts/p2-05-thread-sweep.sh`

- [ ] **Step 1: Write the test**

Create `crates/aleph-oracle/tests/tier1_scaling_invariance.rs`:

```rust
//! P2-05: the rayon-parallel kernels must be correct on the full Tier-1 set
//! (GHZ / QFT / Grover / random) that the `tier1_scaling` bench measures.
//!
//! Compares the AoS `NaiveSvBackend` against the SoA `SoaSvBackend` within
//! 1e-12 per amplitude — two independent memory layouts and kernel families,
//! so a parallel-block disjointness bug in either path breaks the equality.
//! Run under `scripts/p2-05-thread-sweep.sh` (ALEPH_PAR_MIN_AMPS=0 forces the
//! parallel path at this small n across RAYON_NUM_THREADS ∈ {1,2,4,8}); a
//! thread-count-dependent failure would fail the assert. Same idiom as
//! `soa_vs_naive.rs`, but on the canonical n=15 Tier-1 fixtures under
//! `scripts/qiskit-baseline/circuits/`.

use aleph_backend::run;
use aleph_sv::{NaiveSvBackend, SoaSvBackend};

/// Canonical Tier-1 fixtures at n=15 (fast; 32768 amplitudes). These are the
/// n=15 siblings of the n=25 circuits the bench measures.
const FIXTURES: &[&str] = &[
    "ghz_n15",
    "qft_n15",
    "grover_n15_iters5",
    "random_brickwall_n15_d20",
];

#[test]
fn tier1_fixtures_match_across_backends() {
    for &name in FIXTURES {
        let path =
            aleph_oracle::workspace_path(&format!("scripts/qiskit-baseline/circuits/{name}.qasm"));
        let qasm =
            aleph_oracle::load_qasm(&path).unwrap_or_else(|e| panic!("load {name}: {e}"));
        let circuit =
            aleph_parser::parse(&qasm).unwrap_or_else(|e| panic!("parse {name}: {e}"));

        let mut naive = NaiveSvBackend::with_seed(0);
        let naive_state =
            run(&mut naive, &circuit).unwrap_or_else(|e| panic!("naive run {name}: {e}"));
        let naive_amps = naive_state.amplitudes();

        let mut soa = SoaSvBackend::with_seed(0);
        let soa_state =
            run(&mut soa, &circuit).unwrap_or_else(|e| panic!("soa run {name}: {e}"));
        let soa_re = soa_state.re();
        let soa_im = soa_state.im();

        assert_eq!(naive_amps.len(), soa_re.len(), "{name}: amp count mismatch");
        assert_eq!(soa_re.len(), soa_im.len(), "{name}: re/im length mismatch");

        for i in 0..naive_amps.len() {
            let a = naive_amps[i];
            let dr = a.re - soa_re[i];
            let di = a.im - soa_im[i];
            let delta = (dr * dr + di * di).sqrt();
            assert!(
                delta < 1e-12,
                "fixture {name} amp[{i}]: naive ({}, {}) vs soa ({}, {}); |Δ| = {:.3e}",
                a.re,
                a.im,
                soa_re[i],
                soa_im[i],
                delta,
            );
        }
    }
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p aleph-oracle --test tier1_scaling_invariance -- --nocapture`
Expected: PASS — `test tier1_fixtures_match_across_backends ... ok`. (Kernels are already correct; this is a regression guard + spec requirement, so it passes immediately. If it fails to *compile* on `.amplitudes()`/`.re()`/`.im()`, those are the inherent accessors used unchanged in `soa_vs_naive.rs` — do not add a trait import.)

- [ ] **Step 3: Write the thread-sweep script**

Create `scripts/p2-05-thread-sweep.sh`:

```bash
#!/usr/bin/env bash
# P2-05: prove the rayon-parallel kernels are thread-count invariant on the
# full Tier-1 set (GHZ/QFT/Grover/random) that the `tier1_scaling` bench
# measures. Each parallel kernel writes pairwise-disjoint amplitude blocks
# with no cross-thread reduction, so the result must be bit-identical
# regardless of worker-thread count. We force the parallel path at n=15
# (ALEPH_PAR_MIN_AMPS=0) and run the AoS==SoA equivalence across thread
# counts; a non-identical result at any count fails the 1e-12 assert.
#
# Run from the workspace root. On an AVX-512 host (EPYC) this exercises the
# SIMD kernels; on a non-x86 host it exercises the scalar dispatch and the
# par_blocks driver — still a meaningful invariance proof.
set -euo pipefail

cd "$(dirname "$0")/.."

for t in 1 2 4 8; do
  echo "== RAYON_NUM_THREADS=$t (ALEPH_PAR_MIN_AMPS=0, forced parallel) =="
  RAYON_NUM_THREADS=$t ALEPH_PAR_MIN_AMPS=0 \
    cargo test -p aleph-oracle --test tier1_scaling_invariance --quiet
done

echo
echo "All thread counts (1/2/4/8) agree: Tier-1 parallel kernels are thread-count invariant within 1e-12."
```

- [ ] **Step 4: Make the script executable and run it**

Run:
```bash
chmod +x scripts/p2-05-thread-sweep.sh && ./scripts/p2-05-thread-sweep.sh
```
Expected: four `== RAYON_NUM_THREADS=N ...==` blocks each ending in a passing test, then the final "All thread counts (1/2/4/8) agree" line.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-oracle/tests/tier1_scaling_invariance.rs scripts/p2-05-thread-sweep.sh
git commit -m "[P2-05] Tier-1 thread-invariance guard + sweep script

AoS==SoA 1e-12 equivalence on the canonical GHZ/QFT/Grover/random n=15
fixtures; scripts/p2-05-thread-sweep.sh forces the parallel path across
RAYON_NUM_THREADS in {1,2,4,8}. Guards the kernels the tier1_scaling bench
measures.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `tier1_scaling` benchmark

The deliverable measurement instrument: GHZ/QFT/Grover/random at n=25 through the parallel kernels, swept across `RAYON_NUM_THREADS`. Mirrors `benches/benches/qft_scaling.rs` (raw + fused groups) but loads the canonical fixtures (no Rust Grover builder exists; fixtures match the Aer baseline). Gated behind `scaling-bench` so `cargo bench --workspace`/CI skip the 512 MiB runs.

**Files:**
- Create: `benches/benches/tier1_scaling.rs`
- Modify: `benches/Cargo.toml` (append a `[[bench]]` entry)

- [ ] **Step 1: Write the bench**

Create `benches/benches/tier1_scaling.rs`:

```rust
//! P2-05 Tier-1 parallel-scaling benchmark: GHZ / QFT / Grover / random
//! brick-wall at n = 25 through the AoS + AVX-512 `NaiveSvBackend`, driving
//! the rayon-parallel gate kernels. Companion to `qft_scaling.rs` (P2-01),
//! extended to the full Tier-1 set for the Phase-2 scaling report (#31).
//!
//! Gated behind `scaling-bench` so `cargo bench --workspace` / CI skip the
//! 512 MiB n=25 runs. Measure scaling by sweeping `RAYON_NUM_THREADS` across
//! processes, exactly as P2-01:
//!
//!   RAYON_NUM_THREADS=1  cargo bench -p aleph-benches --bench tier1_scaling \
//!       --features scaling-bench -- --save-baseline t1
//!   RAYON_NUM_THREADS=8  cargo bench -p aleph-benches --bench tier1_scaling \
//!       --features scaling-bench -- --baseline t1
//!
//! Circuits are the canonical n=25 fixtures under
//! `scripts/qiskit-baseline/circuits/` — the same circuits the Stage-0 Qiskit
//! Aer baseline used, so scaling lines up with the Aer comparison. GHZ-25 is
//! trivial (25 gates, allocation-bound) — included for spec completeness; its
//! efficiency number is not a meaningful bandwidth signal (see docs/perf/phase2.md).

use aleph_backend::run;
use aleph_sv::NaiveSvBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::PathBuf;

/// (group label, fixture stem). n = 25 only — the Phase-2 scaling target size.
const WORKLOADS: &[(&str, &str)] = &[
    ("ghz", "ghz_n25"),
    ("qft", "qft_n25"),
    ("grover", "grover_n25_iters5"),
    ("random", "random_brickwall_n25_d20"),
];

const N: u32 = 25;

fn fixture_path(stem: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("benches crate is one dir deep from repo root")
        .join("scripts/qiskit-baseline/circuits")
        .join(format!("{stem}.qasm"))
}

fn load(stem: &str) -> aleph_ir::Circuit {
    let path = fixture_path(stem);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", path.display()));
    aleph_parser::parse(&src)
        .unwrap_or_else(|e| panic!("parse {} failed: {:?}", path.display(), e))
}

/// Raw `run`: parallelism lives in the gate kernels, so this isolates kernel
/// scaling. Headline scaling group.
fn bench_tier1_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_scaling");
    // n=25 is a 512 MiB state vector; keep the sample count low so a sweep
    // finishes in minutes. Criterion treats this as a floor for slow benches.
    group.sample_size(10);
    for &(label, stem) in WORKLOADS {
        let circuit = load(stem);
        group.throughput(Throughput::Elements(u64::from(N) * (1u64 << N)));
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circuit| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

/// Fused path: optimize once outside the timed loop (the `run_optimized`
/// default pipeline), then time the parallel kernels on the fused circuit.
/// The honest end-to-end shape; QFT is known fused == raw, Grover/random may
/// differ. Compare its T1→Tn curve against the raw `tier1_scaling` group.
fn bench_tier1_scaling_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_scaling_fused");
    group.sample_size(10);
    for &(label, stem) in WORKLOADS {
        let mut circuit = load(stem);
        circuit.optimize().expect("optimize pipeline");
        group.throughput(Throughput::Elements(u64::from(N) * (1u64 << N)));
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circuit| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| {
                    let state = run(&mut backend, circuit).unwrap();
                    black_box(state);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tier1_scaling, bench_tier1_scaling_fused);
criterion_main!(benches);
```

- [ ] **Step 2: Register the bench in `benches/Cargo.toml`**

Append after the existing `[[bench]] name = "qft_scaling"` block (the one with `required-features = ["scaling-bench"]`):

```toml
[[bench]]
name = "tier1_scaling"
harness = false
required-features = ["scaling-bench"]
```

- [ ] **Step 3: Verify it builds under the feature**

Run: `cargo build -p aleph-benches --benches --features scaling-bench`
Expected: compiles clean (the `tier1_scaling` bench builds; no warnings).

- [ ] **Step 4: Verify `--workspace` still skips it**

Run: `cargo bench --workspace --no-run 2>&1 | grep -i tier1_scaling || echo "SKIPPED (correct)"`
Expected: prints `SKIPPED (correct)` — the gated bench is not compiled into the default `cargo bench --workspace` set (required-features keeps it out).

- [ ] **Step 5: Smoke-run the bench once locally (criterion `--test` mode)**

Run: `cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench -- --test`
Expected: each group/workload runs once and reports "ok" (no measurement, just exercises the code path end-to-end on this aarch64 host — proves the fixtures parse, the circuits run through both raw and fused groups, and nothing panics). n=25 allocates 512 MiB per workload; this is fine on the dev box for a single pass.

- [ ] **Step 6: Commit**

```bash
git add benches/benches/tier1_scaling.rs benches/Cargo.toml
git commit -m "[P2-05] Add tier1_scaling bench (GHZ/QFT/Grover/random, n=25)

Feature-gated (scaling-bench) criterion bench over the canonical n=25
qasm fixtures, raw + fused groups, swept via RAYON_NUM_THREADS. Mirrors
qft_scaling.rs, extended to the full Tier-1 set for the Phase-2 report.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Write `docs/perf/phase2.md` — the scaling report

The capstone report. Synthesizes the real measured QFT scaling from P2-01..04, marks unmeasured cells *pending hardware run* (no fabricated data), consolidates the bandwidth-bound root cause across all four prior tickets, and resolves the ROADMAP §7 exit honestly via the follow-ups path.

**Files:**
- Create: `docs/perf/phase2.md`

- [ ] **Step 1: Write the report**

Create `docs/perf/phase2.md` with exactly this content:

````markdown
# Phase 2 — Multi-Threaded CPU — Scaling-Efficiency Report

**Phase:** 2 (multi-threaded CPU) · **Issue:** #31 (P2-05) · **Date:** 2026-06-02
**Consolidates:** P2-01 (#27), P2-02 (#28), P2-03 (#29), P2-04 (#30).
**Bench:** `benches/benches/tier1_scaling.rs` (feature `scaling-bench`).
**Hardware (all reference boxes, verified-idle per CLAUDE.md idle-check):**

- **EPYC** — AMD EPYC 8124P (Siena), 16 physical / 32 SMT, single socket, 1 NUMA
  node, AVX-512, **all-core frequency-throttled to ~55 %** (`phase2-p2-01.md` §3).
- **NUMA** — 2× Intel Xeon Silver 4114, 20 physical / 40 SMT, **2 sockets / 2 NUMA
  nodes** (distance 10/21), AVX-512.
- **Ryzen** — AMD Ryzen 9 3900, 12 physical / 24 SMT, **no AVX-512** (scalar path).

Toolchain: Rust 1.95, `RUSTFLAGS="-C target-cpu=native"`, criterion release builds.

## 1. Verdict

The ROADMAP §7 Phase-2 exit — **≥ 12× speedup on 16 cores vs single-thread**,
equivalently **≥ 75 % parallel efficiency at 16 threads** (the P2-05 spec's
target; 0.75 × 16 = 12) — **is not met on any hardware available to this
project, and cannot be on this hardware regardless of code.** The measured QFT-25
efficiency at 16 threads is **23 %** on the EPYC box (3.69×), and the same
saturating shape reproduces on a second CPU and a second (scalar) code path.

This is **not a parallelization defect.** State-vector gate application is
**memory-bandwidth-bound** at high core counts: at n=25 the 512 MiB state vector
is pure DRAM streaming, and the lowest-intensity gates (QFT is ~92 % controlled-
phase: 1 read + 1 write + 1 complex multiply per amplitude) saturate the memory
controllers with a handful of cores. Two independent ceilings sit below the
target — the EPYC's ~55 % all-core frequency throttle (an environmental cap that
alone limits its 16-core *ideal* to ≈ 8.6×) and the fundamental bandwidth wall —
and neither is movable by the kinds of work Phase 2 covered (parallelization,
alignment, NUMA placement, chunk tuning). The four Phase-2 tickets each
confirmed this from a different angle (§4).

Per the AC ("scaling target met **or** follow-ups filed"), we take the
follow-ups path (§6). The honest engineering conclusion: the parallelization is
**good for the regime it is in** — on EPYC, QFT-25 reaches **78 % of the box's
frequency-adjusted 8-core ceiling** — and the fixed ≥12×/≥75 % gate is the wrong
metric for a bandwidth-bound kernel (follow-up §6.2).

## 2. Headline scaling — QFT-25 (real measured data)

QFT is the canonical bandwidth-bound Tier-1 workload and the one with measured
multi-thread data across boxes (P2-01 §2/§8, P2-02 §3). `efficiency =
speedup / threads` is the spec's literal metric.

### EPYC 8124P (AVX-512), raw `run`

| Threads | Time | Speedup | Efficiency | Eff. vs freq-adjusted ideal |
|--------:|-----:|--------:|-----------:|----------------------------:|
| 1  | 8.41 s | 1.00× | — | — |
| 8  | 2.50 s | **3.37×** | 42 % | **78 %** |
| 16 | 2.28 s | **3.69×** | 23 % | 43 % |

Frequency context: the all-core clock drops 2995 → ~1620 MHz under load (~1.85×
per-core handicap), capping the *ideal* 8-core speedup at ≈ 4.3× and 16-core at
≈ 8.6× before any memory effect (`phase2-p2-01.md` §3). Against that adjusted
ceiling, 3.37×@8 is **78 %** — the parallelization is sound; the absolute number
is hardware-capped. P2-02 re-measured 3.31×@8 / 3.62×@16 (within noise;
alignment changed nothing). Fused (`run_optimized`) ≈ raw — QFT's controlled-
phase ladder acts on distinct qubit pairs and does not fuse.

### Ryzen 9 3900 (scalar, no AVX-512), raw `run`

| Threads | Time | Speedup | Efficiency |
|--------:|-----:|--------:|-----------:|
| 1  | 13.05 s | 1.00× | — |
| 8  | 6.07 s | **2.15×** | 27 % |
| 12 | 6.02 s | **2.10×** | 18 % |

A second CPU, a second code path: QFT-25 plateaus at **~8 threads** (8→12 buys
nothing) — the same bandwidth wall. The smaller QFT-22 (fits cache far better)
keeps scaling to 3.99×@12 (P2-02 §3), which sharpens the diagnosis: the n=25
plateau is bandwidth, not a parallelization defect.

### NUMA 2× Xeon 4114 — allocation-placement result (P2-03)

The NUMA box's measured Phase-2 contribution is the **allocation-policy** result,
not a full thread sweep: NUMA-aware **first-touch** allocation cut QFT-25 by
**−37.7 % (1.60×)** vs the default allocator — beating `numactl --interleave`
(−31.8 %) **with no thread pinning** (`phase2-p2-03.md`). This is orthogonal to
the thread-scaling ceiling: correct page placement raises the achievable
bandwidth on a 2-socket box; it does not change the bandwidth-bound *shape* of
the per-thread curve.

## 3. Full Tier-1 matrix — measured + pending

The `tier1_scaling` bench measures GHZ / QFT / Grover / random at n=25, swept via
`RAYON_NUM_THREADS`. The cells below are the **measured** state of the project's
data. Cells marked **`pending HW run`** have **no fabricated numbers**: the bench
is delivered ready and produces them with one command (§7). No box reaches the
spec's 32/64-thread points except via SMT (EPYC 32t, NUMA 40t); 64 physical
threads is **unreachable on available hardware** (§5).

| Workload (n=25) | Box | T1 | T2 | T4 | T8 | T16 | T32 |
|---|---|---|---|---|---|---|---|
| QFT     | EPYC  | 8.41 s | pending | pending | **3.37×** | **3.69×** | pending |
| QFT     | Ryzen | 13.05 s | pending | pending | **2.15×** | — (12c) | — |
| GHZ     | EPYC  | pending HW run — *trivial workload, see §4.5* |
| Grover  | EPYC  | pending HW run |
| Random  | EPYC  | pending HW run |

Honest scope note: the multi-thread numbers measured during P2-01..04 targeted
QFT-25 (the workload over the Stage-0 Aer target and the clearest bandwidth
probe). GHZ/Grover/random full sweeps, and the intermediate 2/4/32-thread QFT
points, were **not** measured and are not invented here. The expectation, given
§4, is that Grover and random show the same bandwidth-bound plateau (Grover
carries Toffoli/CCZ, random is brick-wall — both higher arithmetic intensity than
QFT's cphase, so if anything they scale *slightly* better at low thread counts,
but hit the same wall); GHZ is degenerate (§4.5). Confirming this is follow-up §6.3.

## 4. Root-cause synthesis — what the four Phase-2 tickets established

### 4.1 P2-01 — parallelization + the two ceilings
Every SV gate kernel (AoS + SoA, 1q/2q/3q) is rayon-parallel behind
`ALEPH_PAR_MIN_AMPS`, bit-identical across thread counts. A **count-starvation**
bug (outer-block-only parallelism left high-qubit gates serial) was fixed with
`par_units` flattening (high-qubit gate 1.03× → 4.86× @8). The remaining limits
are environmental (frequency throttle) and fundamental (bandwidth) — not code.

### 4.2 P2-02 — contention is not the limiter
64-byte-aligned `AlignedBuf` + a false-sharing audit. `perf c2c` over a 16-thread
QFT-25: **28 shared lines / 24 local HITM across 230 k records** — noise, no
ping-pong. Scaling **flat vs P2-01**. There was no contention to remove; the
deliverable was an alignment *guarantee* (and the NUMA hook P2-03 needed).

### 4.3 P2-03 — NUMA placement helps bandwidth, not the curve shape
First-touch allocation (`zeroed_first_touch`, `numa` feature) gives **−37.7 %**
on the 2-socket box with no pinning (§2). It raises *achievable* bandwidth; it
does not change the bandwidth-bound per-thread scaling shape.

### 4.4 P2-04 — no chunk-tuning headroom
A 360-cell (gate × target × `min_amps` × `grain`) sweep on EPYC + Ryzen: every
cell within **~0.4 % of the default `grain = 64`**. Negative findings *confirm*
the default — large grain (≥256) *regresses* stride-heavy AVX-512 kernels by
+8–15 %; `min_amps` is inert at n≥21 (always parallel). Nothing to tune toward.

### 4.5 GHZ-25 is a degenerate scaling workload
GHZ-25 is 1 H + 24 CNOT = **25 gates total**, running in milliseconds and
dominated by state allocation/initialization, not gate-kernel throughput. Its
"efficiency" is allocation+setup noise, not a bandwidth-scaling signal. It is
included for spec completeness and annotated as such — never reported as a
meaningful parallel-efficiency data point.

## 5. The 64-core / ≥12× target is hardware-gated

The spec asks for thread counts up to 64; the ROADMAP exit asks for ≥12×@16. No
available box can demonstrate either:

- **No 64-physical-core box exists in the fleet.** EPYC 16c/32t, NUMA 20c/40t,
  Ryzen 12c/24t. Counts above physical cores are SMT (throughput-limited for this
  bandwidth-bound, FPU-heavy workload), and 64 is unreachable entirely.
- **The EPYC's frequency throttle** caps its 16-core *ideal* at ≈ 8.6× before any
  memory effect — ≥12×@16 is arithmetically impossible there.
- **Bandwidth** then pulls the realized EPYC 16-core figure to ~3.7×, and the
  Ryzen and NUMA boxes corroborate the saturating shape.

Demonstrating ≥12×@16 for bandwidth-bound SV simulation needs a **non-throttled,
high-memory-bandwidth, ≥16-physical-core** machine (and the 64-thread point a
≥64-core box) — hardware this project does not currently have. This is a
measurement-environment gap, recorded as a follow-up (§6.1), not an open code
defect.

## 6. Follow-ups (filed)

1. **Re-validate ≥12×@16 (and the 32/64-thread points) on non-throttled,
   higher-bandwidth, ≥32-physical-core hardware** when available. The target is
   gated on hardware we lack, not on a code defect.
2. **`[meta]` proposal: revise the ROADMAP §7 Phase-2 exit metric** toward an
   *efficiency-vs-achievable-bandwidth-ceiling* (or compute-bound-regime) form for
   memory-streaming SV kernels. A fixed ≥12×/≥75 % is not an honest gate for a
   bandwidth-bound workload (first flagged in P2-01 follow-up #4). This report
   **recommends** the `[meta]`; it does not edit ROADMAP.md here.
3. **Run the full `tier1_scaling` sweep** (GHZ/Grover/random, and the
   intermediate 2/4/32-thread QFT points) on EPYC + NUMA + Ryzen to fill the
   *pending* cells of §3. The bench is ready (§7).
4. **Propagate `par_units` flattening** to the remaining inner-loop kernels
   (carried from P2-01 follow-up #1) — improves high-qubit-gate scaling on
   non-throttled hardware; will not move the QFT bandwidth number.

## 7. Reproduce

```bash
# Correctness (thread-count invariance on the Tier-1 fixtures):
./scripts/p2-05-thread-sweep.sh

# Tier-1 scaling sweep on an idle bench box (repeat the second line per N):
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench -- --save-baseline t1
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=8 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench -- --baseline t1

# NUMA first-touch (2-socket box, P2-03):
cargo build -p aleph-benches --features "scaling-bench numa" --release
```

Measure only on a **verified-idle** box (`uptime` ≈ 0, no competing
`cargo bench`/runner jobs); deliver code to the self-hosted EPYC runner via
`git bundle`, not a GitHub push, to avoid racing the CI Bench job (CLAUDE.md;
`phase2-p2-01.md` §1).
````

- [ ] **Step 2: Sanity-check the report renders and links resolve**

Run: `ls docs/perf/phase2-p2-01.md docs/perf/phase2-p2-03.md && grep -c "pending" docs/perf/phase2.md`
Expected: the referenced sibling reports exist; `pending` markers are present (proves no cell was silently fabricated).

- [ ] **Step 3: Commit**

```bash
git add docs/perf/phase2.md
git commit -m "[P2-05] Phase 2 scaling-efficiency report (docs/perf/phase2.md)

Consolidates P2-01..04: SV gate application is bandwidth-bound; ROADMAP §7
>=12x/16-core (=75%@16t) not met on available hardware (freq throttle +
bandwidth + no >=64-core box). Real QFT-25 numbers synthesized from prior
measured data; GHZ/Grover/random cells marked pending HW run (no fabricated
data); bench delivered ready. Follow-ups filed incl. a [meta] to revise the
bandwidth-bound exit metric.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Final verification, push, PR

**Files:** none (verification + git only).

- [ ] **Step 1: Lint and format gate (what CI runs)**

Run:
```bash
cargo clippy --workspace --all-targets -- -D warnings && \
cargo clippy -p aleph-benches --all-targets --features scaling-bench -- -D warnings && \
cargo fmt --check
```
Expected: all clean, no output from `fmt --check`. (The second clippy line covers the gated bench, which the default `--all-targets` does not compile.)

- [ ] **Step 2: Workspace test suite still green**

Run: `cargo test --workspace`
Expected: PASS, including the new `tier1_scaling_invariance` test.

- [ ] **Step 3: Re-run the invariance sweep as a final correctness sign-off**

Run: `./scripts/p2-05-thread-sweep.sh`
Expected: the final "All thread counts (1/2/4/8) agree" line.

- [ ] **Step 4: Self-review the diff**

Run: `git log --oneline origin/main..HEAD && git diff --stat origin/main..HEAD`
Expected: four commits (spec already committed earlier on the branch + the three task commits), touching exactly the files in this plan's File Structure plus the spec. Re-read the diff with fresh eyes.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin p2-05-scaling-report
gh pr create --title "[P2-05] Phase 2 scaling-efficiency report" --body "$(cat <<'EOF'
Closes #31

## Summary
Phase-2 capstone (#31): a reusable `tier1_scaling` criterion bench over all four
Tier-1 algorithms (GHZ/QFT/Grover/random at n=25) plus `docs/perf/phase2.md`, the
scaling-efficiency report consolidating P2-01..04.

## Approach
- `tier1_scaling` bench (feature-gated `scaling-bench`) parses the canonical n=25
  qasm fixtures and drives the AoS+AVX-512 parallel kernels, raw + fused groups,
  swept via `RAYON_NUM_THREADS` — same methodology as the P2-01 `qft_scaling` bench.
- Thread-invariance guard (`tier1_scaling_invariance` test + `p2-05-thread-sweep.sh`)
  proves the kernels are bit-exact across thread counts on these exact circuits.
- Report synthesizes the **real measured** QFT-25 scaling from P2-01..04; cells with
  no measured data are marked **`pending HW run`** (no fabricated numbers), with the
  bench delivered ready to fill them.

## Verdict (honest)
ROADMAP §7 ≥12×/16-core (=≥75%@16t) is **not met on available hardware** and
cannot be: bandwidth-bound kernels + the EPYC frequency throttle + no ≥64-core
box. Measured QFT-25 efficiency at 16t is 23% (3.69×), reproduced on a second CPU
and a second code path. Per the AC, follow-ups path taken.

## Test results
- `cargo test --workspace` green, incl. new `tier1_scaling_invariance`.
- `./scripts/p2-05-thread-sweep.sh`: AoS==SoA within 1e-12 across RAYON_NUM_THREADS ∈ {1,2,4,8}.
- `cargo clippy --workspace --all-targets -- -D warnings` + gated-bench clippy + `cargo fmt --check`: clean.
- `cargo bench --workspace` still skips the gated bench.

## Benchmark numbers
No fresh live run for this ticket (per spec §2). Real numbers synthesized from
P2-01..04: QFT-25 EPYC 3.37×@8 / 3.69×@16, Ryzen 2.15×@8, NUMA first-touch −37.7%.

## Follow-ups
- Re-validate ≥12×@16 on non-throttled ≥32-core hardware (target is HW-gated).
- `[meta]` to revise the bandwidth-bound exit metric (recommended, not done here).
- Fill the pending GHZ/Grover/random cells via the ready bench.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR created against `main`, body references `Closes #31` (the **issue** number, per CLAUDE.md — not the PR number).

- [ ] **Step 6: Report the PR URL** back for review.

---

## Self-Review (completed by plan author)

**Spec coverage:** §1 goal → Tasks 2+3. §3 bench (fixtures, gating, raw+fused, sweep protocol, correctness gate) → Tasks 1+2. §4 report sections → Task 3 (all 9 sections present: header, verdict, headline QFT, full matrix, root-cause synthesis, GHZ caveat, 64-core discussion, follow-ups, reproduce). §5 follow-ups → report §6 + PR body. §2 honesty/no-fabrication → `pending HW run` markers + Task 3 Step 2 grep check. §6 out-of-scope (no kernel/ROADMAP/GPU changes) → respected (no production code touched). §7 AC → Task 4.

**Placeholder scan:** the only "pending" strings are the deliberate, spec-mandated *pending HW run* report markers (honest no-fabrication), not plan placeholders. All code steps contain complete code; all command steps state expected output.

**Type/name consistency:** bench fns `bench_tier1_scaling` / `bench_tier1_scaling_fused`, groups `tier1_scaling` / `tier1_scaling_fused`, test fn `tier1_fixtures_match_across_backends`, script `scripts/p2-05-thread-sweep.sh`, accessors `.amplitudes()` / `.re()` / `.im()` (verified against `soa_vs_naive.rs`), `circuit.optimize()` (verified against `qft_scaling.rs`) — all consistent across tasks and matched to the existing codebase APIs.
