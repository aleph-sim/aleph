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

`wide_bond` brickwall, `--features parallel`, `RAYON_NUM_THREADS` sweep:

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
