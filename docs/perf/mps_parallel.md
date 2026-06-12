# P3-09: MPS lazy SWAP routing + multithreading

**Date:** 2026-06-11 · **Issue:** #125 · **Branch:** `p3-09-mps-lazy-swap-multithreading` @ `8f13106` vs `main` @ `cae92a3`

**Boxes:**

- **EPYC** 8124P, 16c, AVX-512 (`195.154.249.85`), verified idle before and after every run (load 0.00, no competing `cargo bench`/CI).
- **Ryzen** 9 3900, 12c/24t, no AVX-512 (`49.12.173.85`), branch shipped as a fresh git bundle, HEAD verified.

All numbers are criterion medians (`cargo bench -p aleph-mps`), default build = sequential
faer (the production configuration after this PR); thread sweeps use
`--features parallel` + `RAYON_NUM_THREADS`.

## AC-1 — lazy SWAP permutation routing

A long-range 2q gate now applies `(d−1)` nearest-neighbor SWAPs instead of `2(d−1)`
(no swap-back), and consecutive long-range gates on the same pair amortize to zero
(`swaps_applied()` counter: CNOT(0,4) on n=5 costs 3 SWAPs vs 6 on `main`; repeating it
costs 0). Correctness is pinned by the SV oracle suite at 1e-10, including dedicated
non-identity-permutation read tests and a 256-case long-range proptest.

`long_range` bench (n=12, χ=32, one CNOT(0,dist) after an NN ladder), EPYC, default
sequential build vs `main`:

| dist | main | P3-09 | change |
|---|---|---|---|
| 1 | 20.83 µs | 23.32 µs | **+11.9 %** (see honest notes) |
| 4 | 31.96 µs | 29.08 µs | **−9.1 %** |
| 8 | 44.63 µs | 37.13 µs | **−16.9 %** |
| 11 | 55.27 µs | 43.27 µs | **−21.6 %** |

`nn_qaoa` guard (NN-only circuits, χ=64 — no long-range gates, isolates the faer
hot-path rewrite), EPYC sequential vs `main`:

| cell | change |
|---|---|
| n10 | −1.8 % |
| n20 | −4.2 % |
| n30 | −5.0 % |
| n20 fixed_chi64 | −4.2 % |
| n20 error_1e-8 | −3.8 % |

**AC-1 MET**: 1e-10 oracle equivalence, strictly fewer SWAPs (counter), and wall-clock
wins growing with distance.

## AC-2 — multithreading

The 2q hot path (theta gemm, truncated SVD, thin-QR center moves, bond absorption) runs
on faer; the `parallel` cargo feature enables faer's rayon backend.

`wide_bond` brickwall, `--features parallel`, `RAYON_NUM_THREADS` sweep
\[editor's note: since P3-13 the bench is runtime-gated — prefix the command
with `WIDE_BOND=1` (and `WIDE_BOND_CHI512=1` for the χ=512 cell) to reproduce\]:

| threads | EPYC n20 χ128 | EPYC n24 χ256 | EPYC n26 χ512 | Ryzen n20 χ128 | Ryzen n24 χ256 |
|---|---|---|---|---|---|
| 1 | 322.9 ms | 3.843 s | 46.50 s | 270.3 ms | 3.289 s |
| 2 | 752.8 ms | 4.756 s | — | 448.3 ms | 3.618 s |
| 4 | 713.3 ms | 4.664 s | — | 403.4 ms | 3.116 s |
| 8 | 760.6 ms | 4.750 s | 37.02 s | 457.1 ms | 3.176 s |
| 12/16 | 748.6 ms | 4.533 s | **29.61 s** | 495.1 ms | 3.444 s |

Default sequential build (no feature): EPYC 321.8 ms / 3.848 s, Ryzen 271.7 ms /
3.311 s — within noise of the t=1 rows, confirming `Par::rayon(1)` ≈ `Par::Seq`.

**Speedup exists only at wide bonds**: χ=512 gives **1.26× @8T** and **1.57× @16T**
(criterion-grade, 10 samples). At χ≤256 the rayon pool is a *pessimization* on both
boxes (up to 2.3× slower at χ=128), and with small bonds (χ=32–64, NN workloads) the
default pool was 2.5×–19× slower — which is why parallelism is **opt-in, default off**
(the default `MpsBackend` is χ=128). The crossover sits between χ=256 and χ=512: per-op
matrices below ~512×1024 are too small for fork-join overhead to pay off, and most gates
in a circuit touch bonds far below the cap.

**AC-2 MET with that scoping**: measured speedup on the wide-bond bench (χ=512 cell,
env-gated `WIDE_BOND_CHI512=1`), truncation-error oracle (ε=0 ⇒ exact) and the
Par::Seq-vs-Par::rayon invariance oracle (1e-10) both pass.

## Honest notes

- **`long_range` dist1 is +11.9 % on EPYC** (sequential). This cell is a 20 µs
  microcircuit at χ≤32 where faer's call overhead (gemm/QR dispatch, workspace setup) is
  visible against the old hand-rolled loops. The realistic NN workload (`nn_qaoa`,
  χ=64) *improves* by 2–5 %, so the rewrite is a net win everywhere except
  microsecond-scale toy circuits; we accept the trade.
- **No parallel win below χ≈512.** If wide-bond MPS becomes a primary workload, the
  next levers are per-op size-thresholded parallelism (pass `Par` explicitly instead of
  the global) and parallelizing the read-path transfer contractions
  (`overlap`/`probabilities`, still nalgebra) — both deferred.
- Ryzen never reaches a parallel win in the measured range (χ≤256; the χ=512 cell was
  not run there — 12 scalar cores at ~47 s/iter × 10 samples was not worth the box
  time given EPYC already demonstrates the crossover).
- The first EPYC sweep (rayon default build) recorded `nn_qaoa` at **+155–225 %** and
  `long_range` at **+1300–1900 %** vs main — preserved here as the evidence for why
  default-on parallelism was rejected.

## P3-13: per-op size-thresholded parallelism, `parallel` default ON

**Date:** 2026-06-12/13 · **Issue:** #149 · **Branch:** `p3-13-mps-size-thresholded-par`
(measurement HEADs: `2e39ed9` sweep 1, `3be8ef1` sweep 2, `d8451cf` sweep 3)

**Box:** EPYC 8124P, 16c, AVX-512 (`195.154.249.85`), verified idle before every run.
All numbers are criterion medians; `--features parallel` builds use `RAYON_NUM_THREADS`
as stated.

P3-09 shipped parallelism opt-in because faer's process-global rayon pool pessimized
every χ≤256 cell (up to 2.3× at χ=128). P3-13 replaces the global control plane with a
per-op decision (`linalg::par_for`): an operand runs `Par::Seq` unless
`rows·cols ≥ PAR_MIN_ELEMS = 2^18 + 1` — i.e. strictly above the measured 2^18
pessimization band. With small ops structurally protected, the `parallel` cargo feature
is now **default ON**; `default-features = false` is the opt-out.

### Threshold sweep — `wide_bond` n26 χ=512, @16T

Reference: t=1 parallel build = **46.755 s** (P3-09 sequential: 46.50 s, same within
noise — nothing regressed in the rewiring; likewise χ128/χ256 t=1 at 321.82 ms /
3.837 s vs P3-09's 322.9 ms / 3.843 s).

| `PAR_MIN_ELEMS` | wall @16T | vs t=1 | diagnosis |
|---|---|---|---|
| 2^20 (sweep 1, `2e39ed9`) | 38.046 s | 1.23× | only the saturated 1024×1024 theta/SVD crossed the threshold; the 1024×512 thin-QR and 512×1024 absorption operands (= 2^19 elements exactly) ran Seq |
| 2^19 (sweep 2, `3be8ef1`) | 31.535 s | 1.48× | remaining gap is the bond-ramp band (2^18, 2^19) — e.g. 768×512 ops while bonds grow toward saturation |
| **2^18+1 (sweep 3, FINAL, `d8451cf`)** | **30.817 s** (CI 30.36–31.31 s) | **1.52×** | P3-09 all-parallel reference: 29.61 s = 1.57× |

### Small/medium-cell guard — parallelism compiled in, @16T, final threshold

χ128/χ256 are band-identical across 2^19 and 2^18+1 — their largest ops (256×256 =
65 536 and 512×512 = 262 144 elements) sit below both thresholds — so the sweep-2/3
numbers are valid for the final config.

| cell | P3-09 global rayon @12/16T | P3-13 @16T | vs sequential reference |
|---|---|---|---|
| wide_bond n20 χ128 | 748.6 ms | 322.20 ms | +0.1 % vs t=1 321.82 ms (p=0.16, noise) |
| wide_bond n24 χ256 | 4.533 s | 3.835 s | −0.05 % vs t=1 3.837 s (p=0.67, noise) |
| nn_qaoa n10 χ64 | +155–225 % class | 328.71 µs | −0.43 % vs seq 330.07 µs |
| nn_qaoa n20 χ64 | +155–225 % class | 825.95 µs | −0.32 % vs seq 829.05 µs |
| nn_qaoa n30 χ64 | +155–225 % class | 1.3169 ms | −1.38 % vs seq 1.3357 ms |
| nn_qaoa n20 fixed_chi64 | +155–225 % class | ~825 µs | −1.09 % |
| nn_qaoa n20 error_1e-8 | +155–225 % class | ~748 µs | −1.25 % |

This is the headline: under P3-09's global rayon, χ=128 ran **748.6 ms** at 12/16T
(2.3× slower than sequential) and nn_qaoa regressed +155–225 % — now every χ≤256 cell
is **within noise of sequential with the pool compiled in and 16 threads live**. The
pessimization that forced default-OFF is gone.

### `long_range` (n=12, χ=32) — build flavor, not pool usage

@16T parallel build vs the sequential (no-feature) build, sweep 1. Threshold-independent:
every op in this bench is ≤ 64×64 = 4 096 elements, so all run Seq at every threshold —
the delta is parallel-*build* codegen flavor, not pool usage.

| dist | seq build | parallel build | change |
|---|---|---|---|
| 1 | 22.637 µs | 23.873 µs | +5.5 % |
| 4 | 29.040 µs | 29.697 µs | +2.2 % |
| 8 | 36.700 µs | 37.890 µs | +3.3 % |
| 11 | 42.085 µs | 42.674 µs | +1.4 % |

These are 20–40 µs microcells; cf. P3-09's accepted +11.9 % on dist1 from the faer
rewrite itself. P3-14 (scratch arena) is the ticket aimed at this class.

### Verdict

- **AC-1a MET** — `nn_qaoa` χ=64 and `wide_bond` χ=128/256 within noise of sequential
  @16T with parallelism compiled in.
- **AC-1b NOT met to the letter** — 1.52× achieved vs the P3-09 1.57× bar (96.7 % of
  the all-parallel win retained). See honest notes.
- **AC-2 MET** — ε=0 oracle unchanged-green; the new `state_invariant_seq_vs_rayon`
  oracle compares `Par::Seq` vs rayon as plain arguments via `MpsState::par_override`
  (no global toggle — fixing the P3-09 test-isolation wart).

### Honest notes

- **The residual ~3 % on χ=512 is structural for a single size threshold.** It comes
  from sub-2^18 ops that are measured pessimizations *in isolation* (the χ=256 cell)
  but benefit from rayon pool warmth *inside* a wide-bond circuit; one global cutoff
  cannot capture that without re-importing the χ≤256 pessimization. The three-point
  sweep (1.23× → 1.48× → 1.52×) hit diminishing returns and was stopped per the plan's
  stop rule.
- **`parallel` is now a default feature.** Users who want the rayon-free build (and the
  ~1–5 % smaller `long_range` microcell times) use
  `aleph-mps = { ..., default-features = false }`; CI keeps that configuration
  compiling and green.

### Re-tuning `PAR_MIN_ELEMS`

Edit the const in `crates/aleph-mps/src/linalg.rs`, then re-measure the win cell with
`WIDE_BOND=1 WIDE_BOND_CHI512=1 RAYON_NUM_THREADS=16 cargo bench -p aleph-mps --bench
wide_bond` AND re-run the guard cells — `nn_qaoa` against a `--no-default-features`
`--save-baseline`, plus the `wide_bond` χ=128/χ=256 cells against their t=1 rows — to
confirm the new threshold introduces no small-cell pessimization. Note the crossover
above was measured at 16T on EPYC only; it is thread-count- and machine-dependent.
