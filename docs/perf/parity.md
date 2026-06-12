# Phase 4.5 competitive parity matrix (P4.5-01 baseline)

**Date:** 2026-06-11/12 (single session, 19:53–03:03 UTC+1).
**Box:** AMD EPYC 8124P (16c/32t, AVX-512), self-hosted runner host, idle-verified
before the session (load 0.02, `pgrep -af "cargo bench|bencher run|Runner.Worker"`
clean, no md resync) and confirmed undisturbed after (no workflow runs in the
measurement window per `gh run list`; last Bench run finished 19:16, session
started 19:53).
**Toolchain:** rustc 1.95.0 (59807616e 2026-04-14), `RUSTFLAGS=-C target-cpu=native`;
Python 3.12.13 (uv venv), qiskit 1.2.4, qiskit-aer 0.15.1 (same pins as the
Stage-0/P1-14 baseline); Stim row imported from `docs/perf/surface_code.md`
(Stim 1.16.0), not re-measured.
**Bar:** every cell ≤ 1.2× its reference, or a documented structural exception
(spec § 2: `docs/superpowers/specs/2026-06-11-phase-4.5-cpu-parity-design.md`).

## Protocol

Both sides of every cell consume **byte-identical QASM fixtures** on the same
box in the same session, strictly sequentially (nothing else ran between rows).

- **SV row:** aleph = `tier1_scaling_fused` criterion group (`RAYON_NUM_THREADS=16`,
  circuit `.optimize()`d once outside the timed loop, backend construction in
  untimed setup); Aer = `AerSimulator(method='statevector', max_parallel_threads=16)`,
  `sim.run()` timed, transpile outside, **default gate fusion ON** —
  default-vs-default on both sides (aleph's default pipeline vs Aer's default
  config), disclosed per spec § 6. Directional note: aleph's one-time
  `optimize()` cost is excluded from its timed region while Aer pays its
  fusion inside `sim.run()` — an asymmetry in aleph's favor, negligible at
  these gate counts (microseconds vs hundreds-of-ms cells) and unable to flip
  any verdict. Fixtures: `scripts/qiskit-baseline/circuits/`.
- **MPS row:** aleph = `mps_parity` criterion bench, sequential default build
  (no `parallel` feature); Aer = `matrix_product_state` method,
  `max_bond_dimension` matched per family, `truncation_threshold=1e-16`,
  `max_parallel_threads=1` — sequential both sides. Fixtures:
  `scripts/mps-baseline/circuits/`. Aer's timed region includes the mandatory
  `save_matrix_product_state()` serialization, which aleph does not pay (see
  the timing caveat in `scripts/mps-baseline/run.py`); given every MPS cell
  lands ≥ 10× below the bar, this asymmetry cannot affect any verdict.
- **Runs:** Aer SV — `--min-runs 5` (cost-budgeted up to 10 for cheap cells);
  Aer MPS — 10 runs; aleph — criterion `sample_size(10)`. Medians throughout;
  warm-up run/iterations excluded on both sides.

## Row 1 — state vector, 16 threads both sides, n = 25

| workload | gates | aleph (ms) | Aer (ms) | aleph/Aer | verdict (≤ 1.2×) |
|---|---|---|---|---|---|
| ghz_n25 | 25 | 441.5 | 430.0 | **1.03×** | ✅ PASS |
| qft_n25 | 1525 | 1 317.5 | 1 628.6 | **0.81×** | ✅ PASS |
| grover_n25_iters5 | 160 615 | 308 329 | 513 366 | **0.60×** | ✅ PASS |
| random_brickwall_n25_d20 | 1240 | 2 120.4 | 3 566.8 | **0.59×** | ✅ PASS |

aleph is faster on 3 of 4 cells; GHZ is allocation-bound on both sides
(~0.42–0.44 s floor, see `docs/perf/phase2.md`) and lands at a statistical tie.
`grover_n25_iters5` is the **iteration-capped** legacy Phase-1 fixture (5
Grover iterations); the asymptotically optimal-iteration circuit is ~2 900×
larger and intractable for a timing matrix (P2-05 measured ~13 CPU-h
single-thread) — no extrapolation to the full circuit is claimed.

For reference, the *unfused* aleph path (raw `tier1_scaling` group, same
session): ghz 422.3 ms, qft 15 067.9 ms, grover 1 603 600 ms, random
14 548.8 ms. The default-pipeline fusion is worth 11.4× on QFT, 5.2× on
Grover, 6.9× on random at 16 threads — comparing raw aleph against
fusion-enabled Aer would be the wrong (non-default) comparison.

## Row 2 — MPS, sequential both sides

| workload | n | χ | gates | aleph (ms) | Aer (ms) | aleph/Aer | verdict (≤ 1.2×) |
|---|---|---|---|---|---|---|---|
| brickwork_n128_d6 | 128 | 64 | 2039 | 8.41 | 37.35 | **0.23×** | ✅ PASS |
| long_range_n12_dist4 | 12 | 64 | 46 | 0.074 | 1.078 | **0.07×** | ✅ PASS |
| long_range_n12_dist8 | 12 | 64 | 46 | 0.105 | 1.129 | **0.09×** | ✅ PASS |
| long_range_n12_dist11 | 12 | 64 | 46 | 0.113 | 1.155 | **0.10×** | ✅ PASS |
| wide_bond_n26_d12 | 26 | 256 | 3900 | 11 430 | 73 503 | **0.16×** | ✅ PASS |

aleph-mps is 4.4–14× faster than Aer MPS on every cell. Fidelity equality:
brickwork (max bond 8 ≪ χ=64) and long_range (χ=64 = 2^(n/2) exact at n=12)
truncate on **neither** side — equal fidelity by construction. wide_bond
saturates the χ=256 cap on both sides and truncation semantics differ between
implementations; at 0.16× the caveat cannot flip the verdict, but the cell is
a *throughput* comparison at equal bond cap, not a proven-equal-fidelity one.

## Row 3 — stabilizer (imported from `docs/perf/surface_code.md`)

| workload | qubits | aleph (ms/cycle) | Stim (ms/cycle) | aleph/Stim | verdict (≤ 1.2×) |
|---|---|---|---|---|---|
| surface-code d=11 syndrome cycle | 241 | — | — | **1.64×** | ❌ GAP |

Measured by P3-11 (single-thread both sides, Stim `TableauSimulator`); see
`surface_code.md` for the full d=3..11 table. Known levers, deferred from
P3-11: orientation-transpose (~30% of the d=11 cycle) and `zero_row`/`copy_row`
(~33%), both still scalar.

## Gap list (scopes P4.5-06)

**No SV-MT or MPS gaps.** Every cell in rows 1–2 is at or below 1.03×, most
well below 1.0×. **P4.5-06 closes as a no-op** (per its acceptance criteria:
"if the matrix shows no cell > 1.2×, close as no-op with a comment linking
the report").

The only gap in the matrix is the **stabilizer row (1.64× Stim @ d=11)**,
already scoped as **P4.5-02** (#155) with identified levers.

## Raw data

Aer JSON outputs (medians/stdev over the stated runs):

```
results-qiskit-t16.json  (aer_threads=16)
  ghz_n25                   median 430.0 ms   stdev 21.4 ms
  qft_n25                   median 1628.6 ms  stdev 1.4 ms
  grover_n25_iters5         median 513365.5 ms stdev 1263.1 ms
  random_brickwall_n25_d20  median 3566.8 ms  stdev 132.0 ms

results-aer-mps.json  (sequential, truncation_threshold=1e-16)
  brickwork_n128_d6   chi=64  median 37.354 ms   stdev 0.207 ms
  long_range_n12_dist4  chi=64  median 1.078 ms  stdev 0.020 ms
  long_range_n12_dist8  chi=64  median 1.129 ms  stdev 0.018 ms
  long_range_n12_dist11 chi=64  median 1.155 ms  stdev 0.011 ms
  wide_bond_n26_d12   chi=256 median 73502.9 ms  stdev 107.3 ms
```

aleph criterion medians (point estimates, same session):

```
tier1_scaling_fused/ghz     441.46 ms     tier1_scaling/ghz     422.32 ms
tier1_scaling_fused/qft     1317.47 ms    tier1_scaling/qft     15067.87 ms
tier1_scaling_fused/grover  308329 ms     tier1_scaling/grover  1603600 ms
tier1_scaling_fused/random  2120.43 ms    tier1_scaling/random  14548.85 ms

mps_parity/brickwork_n128_d6      8.4133 ms
mps_parity/long_range_n12_dist4   0.073708 ms
mps_parity/long_range_n12_dist8   0.105391 ms
mps_parity/long_range_n12_dist11  0.112870 ms
mps_parity/wide_bond_n26_d12      11429.5 ms
```

Commands (run on the box, in order, nothing in parallel):

```bash
# row 1a (Aer SV, 16 threads) — from scripts/qiskit-baseline/
python run.py --threads 16 --min-runs 5 --out results-qiskit-t16.json \
  --from-qasm circuits/ghz_n25.qasm circuits/qft_n25.qasm \
              circuits/grover_n25_iters5.qasm circuits/random_brickwall_n25_d20.qasm
# row 1b (aleph SV, 16 threads)
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=16 \
  cargo bench -p aleph-benches --bench tier1_scaling --features scaling-bench
# row 2a (Aer MPS, sequential)
python run.py --runs 10 --out results-aer-mps.json     # scripts/mps-baseline
# row 2b (aleph MPS, sequential)
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-mps --bench parity
```

## Verdict summary

| row | cells | result |
|---|---|---|
| SV multi-thread vs Aer | 4 | **4/4 ≤ 1.2×** (3 of 4 faster than Aer) |
| MPS vs Aer | 5 | **5/5 ≤ 1.2×** (all ≥ 4× faster than Aer) |
| Stabilizer vs Stim | 1 | **gap: 1.64×** → P4.5-02 (#155) |

Phase 4.5 exit therefore hinges entirely on **P4.5-02**. P4.5-06 is a no-op.
