# Phase 1 baseline: aleph vs Qiskit Aer (EPYC, single thread)

**Date:** 2026-05-27
**Host:** AMD EPYC 8124P (16 cores / 32 threads, Zen 4), 123 GiB RAM,
kernel 7.0.0-15-generic (Ubuntu)
**Toolchain:** Rust 1.95.0 (2026-04-14), `RUSTFLAGS="-C target-cpu=native"`,
AVX-512 emission verified (`objdump` shows 34× `vmulpd zmm` in bench binary)
**Python:** 3.12.13, **Qiskit:** 1.2.4, **Aer:** 0.15.1, **numpy:** 2.1.3,
**scipy:** 1.17.1
**Pin:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 0 python …` (Aer); cargo bench run un-pinned but on the same
otherwise-idle runner. Aer `max_parallel_threads=1` verified at runtime.
**Reproducibility:** see `scripts/qiskit-baseline/README.md`

## Headline — `NaiveSvBackend` (AoS + AVX-512, canonical fast x86 path post-P1-03)

| Workload                              |  aleph (ms) |  Aer (ms) | aleph / Aer | ROADMAP § 7 target |
|---------------------------------------|------------:|----------:|------------:|:------------------:|
| `qft_n20`                             |   **1 098** |       459 |  **2.39 ×** |  ≤ 2× ❌ (1.20× over) |
| `grover_n20_iters5`                   |  **92 111** |   115 598 |  **0.80 ×** |  ≤ 2× ✅ (aleph faster) |
| `random_brickwall_n20_d20`            |     **822** |     1 138 |  **0.72 ×** |  ≤ 2× ✅ (aleph faster) |

Times are criterion medians (30 samples, `--sample-size 30 --measurement-time 15`);
Aer medians are over 10 timed iterations (1 warm-up). Lower is faster.
Ratios > 1 mean aleph is slower than Aer.

## Appendix — full backend matrix

| Workload                              | `NaiveSvBackend` (ms) | `SoaSvBackend` (ms) | Aer (ms)   | gates (post-transpile) |
|---------------------------------------|---------------------:|--------------------:|-----------:|-----------------------:|
| `qft_n20`                             |             1 098.11 |            2 553.54 |     459.34 |                    970 |
| `grover_n20_iters5`                   |            92 111.33 |          211 096.40 | 115 597.58 |                 96 210 |
| `random_brickwall_n20_d20`            |               821.58 |            2 238.38 |   1 138.31 |                    990 |

Relative standard deviation (stdev ÷ median):

| Workload                              | NaiveSv | SoaSv | Aer    |
|---------------------------------------|--------:|------:|-------:|
| `qft_n20`                             |  0.08 % | 0.05 %| 0.64 % |
| `grover_n20_iters5`                   |  0.34 % | 0.08 %| 0.03 % |
| `random_brickwall_n20_d20`            |  0.07 % | 0.08 %| 0.09 % |

All measurements clean — worst stdev 0.64 % (Aer QFT-20).

## Interpretation

Stage 0 surfaces a sharp split across the three canonical workloads.

**QFT-20 is the weak point (2.39× Aer).** The QFT kernel is dominated by
controlled-Phase gates, which our backend currently routes through the generic
two-qubit `apply_2q` kernel — no diagonal-gate fast path, no controlled-phase
specialisation, and no AVX-512 yet on the 2-qubit kernel (P1-03 only covered
1-qubit AoS+AVX-512). That's exactly what **P1-06** (diagonal-gate kernel) and
**P1-07** (2-qubit kernel + CNOT/CZ specialisations) target. With a working
diagonal-gate fast path and AVX-512 on the 2q kernel, this should land at or
under the 2× target on QFT.

**Grover and random circuits already beat Aer (0.80×, 0.72×).** Both workloads
are dominated by 1q gates flowing through P1-03's AoS+AVX-512 `apply_1q_avx512`
kernel — that one wins. The Grover headline (~20 % faster than Aer at 96 210
gates and ~92 s wall-clock) is the largest absolute headroom we've seen so far,
and it confirms the **AoS substrate decision in ADR 0008**: SoA is 2.3× slower
than AoS on QFT and 2.3× slower on Grover, consistent with the layout-dominates-
SIMD finding.

**Aer ≤ 2× exit criterion is partially met.** ROADMAP § 7 says "single-thread
within 2× of Qiskit Aer for QFT, Grover, random circuits at 25 qubits". At
n=20 we're already over on QFT, comfortably under on the other two. **n=25 gets
re-measured at Phase 1 closure (P1-14)** — the gate-count ratios should hold,
so QFT will still be the weak point.

**Stage 1 priority shifts.** The Phase 1 plan's original ordering (P1-05 Pauli-X
→ P1-06 diagonal → P1-07 2q → P1-08 multi-controlled) was reasonable but
QFT-driven priority would put **P1-06 + P1-07 first**, because QFT is the
benchmark closest to the ≤ 2× line and both specialisations directly attack
it. Pauli-X (P1-05) and multi-controlled (P1-08) help Grover and random circuits
where we already beat Aer — diminishing-returns territory until QFT is fixed.

**Stage 2 (IR-opt) is the bigger long-term lever.** Per ADR 0008's hierarchy,
gate fusion + cancellation + commutation analysis collapse the gate count
upstream, so they multiply with whatever SIMD wins land in Stage 1. QFT
specifically has tons of trivially fusible neighbours (`H · cP · H · cP · …`)
that could shave 30–50 % of the gate count.

**Phase 1 proceeds to Stage 1 regardless of this report** — these numbers
are informational, not a gate. The user's explicit decision was to ship the
full Phase 1 backlog.

## Caveats

1. **n=20, not n=25.** ROADMAP target is 25 qubits, measured here at 20 because
   it matches the bencher.dev anchor and keeps Grover wall-clock under ~2 min.
   25-qubit numbers land in P1-14 (Phase 1 closure).
2. **Manual measurement, not bencher.dev-driven.** Stage 0's cargo bench was
   run with `--sample-size 30 --measurement-time 15` directly on EPYC, not via
   the Bench CI workflow (the default sample size would have busted the 30-min
   workflow timeout). The bench file (`benches/benches/qiskit_baseline.rs`) now
   ships with reduced sample budgets so future Bench CI completes within budget,
   meaning subsequent measurements will appear on bencher.dev automatically.
3. **No taskset on the cargo bench side.** Aer was pinned with `taskset -c 0`
   because Python threading otherwise migrates work between cores. Cargo bench
   ran un-pinned (the `cargo` shim in PATH and `taskset` don't compose cleanly
   — see PR notes). All measurements happened on an otherwise-idle runner
   (CI/Bench drained before measuring), so the absence of pinning is a small
   noise risk, not a correctness one. Future CI Bench runs (post-fix) and Stage
   1 re-measurements will eventually re-pin via a proper wrapper script.
4. **`optimization_level=0` transpile.** Qiskit's transpiler is disabled to
   keep the comparison engine-vs-engine, not transpiler-vs-engine. Aleph has
   no equivalent optimiser yet; that's exactly what Stage 2 builds. When both
   sides have optimisers, this measurement gets re-done with whatever Aleph's
   optimiser produces vs whatever Qiskit's level-3 transpiler produces.

## Reproducing this report

See `scripts/qiskit-baseline/README.md`. Quick form on EPYC:

```bash
ssh root@195.154.249.85
cd /tmp/aleph-forensics
git clone <repo> && cd aleph && git checkout meta-phase1-plan

# Aer side (pinned)
cd scripts/qiskit-baseline
python3.12 -m venv .venv  # or via uv: uv venv -p 3.12 .venv
source .venv/bin/activate && pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python run.py

# aleph side (un-pinned, see caveat 3)
cd ../..
RUSTFLAGS="-C target-cpu=native" cargo bench --bench qiskit_baseline -- \
  --sample-size 30 --measurement-time 15 --save-baseline phase1-stage0
```

Pinned versions in `scripts/qiskit-baseline/requirements.txt`. Aer/Qiskit
versions printed at the top of this report come from `pip freeze`.

## Related work

- **P1-03** AVX-512 packed-complex 1q kernel: PR #80 (`f596e9a`).
- **ADR 0008** — AoS + AVX-512 beats SoA on Zen 4: `docs/decisions/0008-aos-avx512-beats-soa-simd.md`.
- **ADR 0007** — SoA-on-x86 perf finding: `docs/decisions/0007-soa-x86-perf-finding.md`.
- **Phase 1 plan:** `docs/superpowers/plans/2026-05-26-phase1-completion.md`.
- **Stage 0 spec:** `docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md`.
