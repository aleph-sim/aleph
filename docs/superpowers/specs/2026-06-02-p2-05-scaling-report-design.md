# P2-05 — Phase 2 Scaling-Efficiency Report — Design

**Issue:** #31 (P2-05) · **Milestone:** Phase 2 · **Date:** 2026-06-02
**Depends on:** P2-01 (#27), P2-02 (#28), P2-03 (#29), P2-04 (#30) — all merged.

## 1. Goal

Deliver the Phase-2 capstone:

1. A reusable, gated criterion bench (`tier1_scaling`) that measures
   thread-scaling of **all four Tier-1 algorithms** (GHZ, QFT, Grover, random)
   at n = 25, swept across `RAYON_NUM_THREADS`.
2. `docs/perf/phase2.md` — the scaling-efficiency report that resolves the
   ROADMAP §7 Phase-2 exit criterion (**≥ 12× on 16 cores = ≥ 75 % efficiency
   at 16 threads**; the P2-05 spec's "≥75 % at 16 threads" is the same target)
   and files honest follow-ups.

## 2. Constraints & honesty framing (read first)

- **No fresh live benchmark run for this ticket.** The bench boxes are not being
  driven live here. The report therefore **synthesizes the real, already-measured
  numbers** from P2-01..04 and clearly marks any unmeasured cell as
  *pending hardware run* — never fabricated. The bench is delivered fully working
  and locally correctness-verified so those cells can be filled by a later pinned
  run with one command.
- **Real data we have (measured, idle boxes):**
  - EPYC 8124P (AVX-512): QFT-25 — T1 8.41 s, **3.37×@8**, **3.69×@16** (P2-01
    clean numbers; P2-02 re-measured 3.31×@8 / 3.62×@16 within noise).
  - Ryzen 9 3900 (scalar, no AVX-512): QFT-25 — **2.11×@8**, plateau 2.10×@12.
  - NUMA 2× Xeon 4114 (AVX-512): QFT-25 first-touch **−37.7 %** (1.60×) vs default
    allocator (P2-03).
- **No box reaches 64 physical threads.** EPYC 16c/32t, NUMA 20c/40t, Ryzen
  12c/24t. The spec's 64-thread point is **hardware-gated** — reported explicitly,
  follow-up filed. (Also: EPYC all-core frequency throttle ~55 % caps even its
  16-core *ideal* at ≈ 8.6×, per P2-01 §3 — the absolute ≥12× target is not
  reachable on available hardware regardless of code.)
- **GHZ-25 is trivial** (1 H + 24 CNOT = 25 gates, runs in ms, allocation-bound).
  Included for spec completeness but annotated as not a meaningful
  bandwidth-scaling workload — never silently dropped.
- The established Phase-2 conclusion (P2-01..04, two boxes, two code paths): SV
  gate application is **memory-bandwidth-bound** at high core counts; scaling
  plateaus well below linear. This report consolidates that evidence; it does not
  re-litigate it.

## 3. Component 1 — `tier1_scaling` bench

- New file `benches/benches/tier1_scaling.rs`, registered in `benches/Cargo.toml`
  with `harness = false` and `required-features = ["scaling-bench"]` (the existing
  opt-in gate, so `cargo bench --workspace` / CI skip the 512 MiB n=25 runs).
- **Circuits:** parse the four canonical fixtures
  `scripts/qiskit-baseline/circuits/{ghz_n25,qft_n25,grover_n25_iters5,random_brickwall_n25_d20}.qasm`
  via `aleph_parser`. Rationale: there is no Rust `grover_circuit` builder, and
  these are the exact circuits the Stage-0 Qiskit Aer baseline used — parsing them
  keeps all four workloads consistent with the Aer comparison and avoids a
  hand-rolled Grover.
- **Structure:** one criterion group per algorithm (`ghz`, `qft`, `grover`,
  `random`), `sample_size(10)` (criterion floor for slow benches), `Throughput`
  set per circuit. Driven through `NaiveSvBackend::with_seed(0)` + `run` — the
  AoS + AVX-512 path whose kernels carry the rayon parallelism. Mirrors
  `qft_scaling.rs` exactly.
- **Sweep protocol** (documented in the module header, matches P2-01):
  `RAYON_NUM_THREADS=1 cargo bench … --features scaling-bench -- --save-baseline t1`,
  then `RAYON_NUM_THREADS=N … -- --baseline t1` for N ∈ the box's thread counts.
- **Fused mirror:** a `*_fused` variant per algorithm (circuit `.optimize()`'d once
  outside the timed loop) so the honest end-to-end (`run_optimized`) scaling can be
  compared. QFT is known fused == raw; Grover/random may differ. Intended primarily
  for the EPYC representative run.
- **Correctness gate:** an integration test (or a `#[test]` in the bench's adjacent
  module) asserting **thread-count invariance** — each Tier-1 circuit produces a
  bit-identical (1e-12) final state across `RAYON_NUM_THREADS ∈ {1,2,4,8}` vs the
  Naive backend, reusing the P2-01 `scripts/p2-01-thread-sweep.sh` pattern. No
  silent wrong answers from the sweep. Verified locally on aarch64 (scalar path
  runs there).

## 4. Component 2 — `docs/perf/phase2.md`

Capstone report. Sections:

1. **Header** — phase, issue, date, the three boxes, toolchain, idle-check note.
2. **Summary / verdict** — the ROADMAP §7 ≥12×/16-core (= ≥75 %@16t) target is
   **not met** on available hardware; root cause is memory bandwidth +
   (EPYC) frequency throttle, established across 3 boxes and 2 code paths; AC
   satisfied via the follow-ups path.
3. **Headline scaling** — QFT-25 table (real measured): per box, columns
   threads / time / speedup-vs-T1 / **efficiency = speedup/threads** (the spec's
   literal metric) / efficiency-vs-frequency-adjusted-ideal (EPYC). This is the
   anchored, real-data section.
4. **All-four-algorithm matrix** — a table per box for ghz/qft/grover/random.
   QFT row(s) filled from real data; the other rows marked **`pending hardware
   run`** with the exact `tier1_scaling` command to produce them. Honest and
   explicit; no fabricated cells.
5. **Root-cause synthesis** — folds in the four prior tickets:
   - P2-01: count-starvation fix (`par_units`) + frequency throttle (the
     environmental ceiling) + bandwidth (the fundamental ceiling).
   - P2-02: no false sharing (`perf c2c` noise-level) — contention is not the
     limiter; alignment is a guarantee, not a speedup.
   - P2-03: NUMA first-touch −37.7 % on the 2-socket box, no pinning needed.
   - P2-04: no chunk-size headroom (≤0.4 % across a 360-cell grid); default
     grain is near-optimal; large grain *regresses* stride-heavy kernels.
6. **GHZ caveat** — why GHZ-25's efficiency number is not meaningful.
7. **Hardware-gated 64-core / ≥12× discussion** — Amdahl + bandwidth + the
   absence of a non-throttled ≥32-physical-core / high-bandwidth box; why the
   absolute target cannot be demonstrated on current hardware.
8. **Follow-ups (filed).**
9. **Reproduce** — exact commands.

## 5. Follow-ups the report files

1. Re-validate ≥12×/16-core (and the 32/64-thread points) on **non-throttled,
   higher-memory-bandwidth, ≥32-physical-core hardware** when available — the
   target is gated on hardware we do not currently have, not on a code defect.
2. **`[meta]` proposal** to revise the ROADMAP §7 Phase-2 exit metric toward an
   *efficiency-vs-achievable-bandwidth-ceiling* (or compute-bound-regime) form for
   memory-streaming SV kernels — a fixed ≥12×/≥75 % is not an honest gate for a
   bandwidth-bound workload (already flagged in P2-01 follow-up #4). The report
   **recommends** this `[meta]`; it does **not** edit ROADMAP.md in this PR.
3. Run the full `tier1_scaling` sweep (GHZ/Grover/random) on the EPYC + NUMA +
   Ryzen boxes to fill the *pending* cells; the bench is ready.

## 6. Out of scope (YAGNI)

- No new kernels or perf optimizations — Phase-2 perf work is complete; this is the
  measurement + report ticket.
- No ROADMAP.md edit in this PR (only the `[meta]` recommendation).
- No GPU, no distributed.
- No 64-core emulation / oversubscription beyond noting SMT thread counts above
  physical-core counts.
- No live bench run on the hosted boxes as part of this ticket (per the constraint
  in §2) — the bench is delivered ready, numbers synthesized from existing data.

## 7. Acceptance criteria (from BACKLOG)

- [x] **Report committed** — `docs/perf/phase2.md`.
- [x] **Scaling target met or follow-ups filed** — target **not** met on available
  hardware (honestly documented); follow-ups filed (§5).

Plus, for this implementation:

- [x] `tier1_scaling` bench builds, is feature-gated, and is registered in
  `benches/Cargo.toml`.
- [x] Thread-count-invariance correctness check passes locally (aarch64 scalar).
- [x] No fabricated benchmark numbers; every unmeasured cell marked *pending*.
- [x] `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt
  --check` clean; `cargo bench --workspace` still skips the gated bench.
