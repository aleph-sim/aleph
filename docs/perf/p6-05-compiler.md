# P6-05: Communication-aware compiler for distributed SV

Spec: `docs/superpowers/specs/2026-10-04-p6-05-comm-aware-compiler-design.md`. Issue #59.

This report grows over three PRs.
- PR 1 (this one) adds the commutation DAG, the reordering scheduler (`Router::Reorder`) and initial placement, and
  reports CPU communication counts.
- PR 2 adds the calibrated GPU cost model.
- PR 3 adds `compile` and the predicted-time benchmark.

## 1. Communication counts (CPU, PR 1)

`cargo run --release -p aleph-sv --example dist_comm_counts`

| circuit | g | strategy | exch | × slice | local swaps | vs lookahead |
|---|---|---|---|---|---|---|
| GHZ-32 | 2 | naive | 2 | 1.0 | 0 | 0.75× |
| GHZ-32 | 2 | lookahead | 1 | 0.8 | 0 | 1.00× |
| GHZ-32 | 2 | reorder | 1 | 0.8 | 0 | 1.00× |
| GHZ-32 | 2 | reorder k=1 | 2 | 1.0 | 0 | 0.75× |
| GHZ-32 | 2 | reorder+place | 1 | 0.8 | 0 | 1.00× |
| GHZ-32 | 3 | naive | 3 | 1.5 | 0 | 0.58× |
| GHZ-32 | 3 | lookahead | 1 | 0.9 | 0 | 1.00× |
| GHZ-32 | 3 | reorder | 1 | 0.9 | 0 | 1.00× |
| GHZ-32 | 3 | reorder k=1 | 3 | 1.5 | 0 | 0.58× |
| GHZ-32 | 3 | reorder+place | 1 | 0.9 | 0 | 1.00× |
| QFT-32 | 2 | naive | 3 | 1.5 | 0 | 1.00× |
| QFT-32 | 2 | lookahead | 2 | 1.5 | 2 | 1.00× |
| QFT-32 | 2 | reorder | 2 | 1.5 | 2 | 1.00× |
| QFT-32 | 2 | reorder k=1 | 3 | 1.5 | 1 | 1.00× |
| QFT-32 | 2 | reorder+place | 1 | 0.8 | 0 | 2.00× |
| QFT-32 | 3 | naive | 4 | 2.0 | 0 | 0.88× |
| QFT-32 | 3 | lookahead | 2 | 1.8 | 3 | 1.00× |
| QFT-32 | 3 | reorder | 2 | 1.8 | 3 | 1.00× |
| QFT-32 | 3 | reorder k=1 | 4 | 2.0 | 1 | 0.88× |
| QFT-32 | 3 | reorder+place | 1 | 0.9 | 0 | 2.00× |
| random-30 d=20 | 2 | naive | 89 | 44.5 | 0 | 0.37× |
| random-30 d=20 | 2 | lookahead | 22 | 16.5 | 42 | 1.00× |
| random-30 d=20 | 2 | reorder | 1 | 0.8 | 2 | 22.00× |
| random-30 d=20 | 2 | reorder k=1 | 2 | 1.0 | 2 | 16.50× |
| random-30 d=20 | 2 | reorder+place | 1 | 0.8 | 2 | 22.00× |
| random-30 d=20 | 3 | naive | 119 | 59.5 | 0 | 0.34× |
| random-30 d=20 | 3 | lookahead | 23 | 20.1 | 67 | 1.00× |
| random-30 d=20 | 3 | reorder | 1 | 0.9 | 3 | 23.00× |
| random-30 d=20 | 3 | reorder k=1 | 3 | 1.5 | 3 | 13.42× |
| random-30 d=20 | 3 | reorder+place | 1 | 0.9 | 3 | 23.00× |
| Grover-20 (5 iters) | 2 | naive | 117 | 58.5 | 0 | 0.21× |
| Grover-20 (5 iters) | 2 | lookahead | 20 | 12.5 | 24 | 1.00× |
| Grover-20 (5 iters) | 2 | reorder | 22 | 11.0 | 16 | 1.14× |
| Grover-20 (5 iters) | 2 | reorder k=1 | 22 | 11.0 | 16 | 1.14× |
| Grover-20 (5 iters) | 2 | reorder+place | 22 | 11.0 | 16 | 1.14× |
| Grover-20 (5 iters) | 3 | naive | 293 | 146.5 | 0 | 0.14× |
| Grover-20 (5 iters) | 3 | lookahead | 30 | 20.6 | 28 | 1.00× |
| Grover-20 (5 iters) | 3 | reorder | 41 | 20.8 | 32 | 0.99× |
| Grover-20 (5 iters) | 3 | reorder k=1 | 42 | 21.0 | 32 | 0.98× |
| Grover-20 (5 iters) | 3 | reorder+place | 41 | 20.8 | 32 | 0.99× |

`× slice` is the amplitudes moved per rank divided by the slice size `2^m`. `vs lookahead` is lookahead's moved volume
divided by this row's (> 1 = less traffic).

**Reading.**
- **Random-30 d=20 is a 1D nearest-neighbour brickwall.** Reorder needs 1 exchange for 20 layers (lookahead: 22–23),
  moving 0.8–0.9 slices against 16.5–20.1, a 22–23× reduction. Even `k=1` gives 16.5× (g=2) and 13.4× (g=3).
  - Why: the depth (20) is less than the distance from at least g of the local qubits to the global ones (here roughly
    q ≤ 7 at n=30), so gates form a light cone. That is all the mechanism needs.
    Qubits far from the global ones can finish all 20 layers before any global-qubit gate is needed. They then become
    free eviction victims (never needed again), so one exchange makes the global qubits local for the rest of the
    circuit.
  - This is specific to shallow 1D brickwalls (depth < distance to the global qubits), not a general ~22× gain.
    Deeper or all-to-all circuits will not collapse this way.
  - Correctness of these reorders is covered by the DAG soundness proptest (`prop_every_dag_order_is_equivalent`) and
    the Reorder oracle proptests in `crates/aleph-sv/tests/dist_oracle.rs`.
- **QFT** is unchanged by full-k reorder (the `reorder` row: 1.00×, same 2 exchanges). `reorder k=1` is 1.00× at g=2
  (but 3 exchanges vs 2) and 0.88× at g=3. Reorder+place cuts it to 1 exchange and 0 local swaps, 2.00× less data than
  lookahead at both g.
- **GHZ**: reorder and reorder+place tie lookahead (1.00×). `reorder k=1` moves more (0.75× at g=2, 0.58× at g=3,
  the same as naive; 2 and 3 exchanges vs lookahead's 1), because capping the exchange width at 1 bit forbids batching.
- **Grover moves slightly more data than lookahead at g=3.** Reorder moves 20.8 slices vs 20.6 (0.99×), with 41
  exchanges vs 30 and 32 local swaps vs 28. `k=1` is 0.98× (42 exchanges). Reorder+place is identical to reorder here
  (placement does not help this circuit). At g=2 reorder moves less (11.0 vs 12.5, 1.14×) but uses more exchanges
  (22 vs 20).
- Comm volume is only half of the objective. Reordering also lengthens local segments, which matters for per-rank
  fusion, and that is measured in PR 3. `compile` keeps Lookahead as a candidate, so a row where reorder moves more
  data is not a regression of the final compiler.
- No GPU timing is claimed here; these are planning-level counts only.

## 2. Calibrated GPU cost model (PR 2)

Hardware: RTX 4000 SFF Ada (sm_89, 20 GiB, 70 W cap), one card. Date: 2026-10-04.

### 2.1 Calibration

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture`

Method: on a `2^m_ref` state after an H layer, 32 × (`H` on qubit `i % 8`, then one launch of the kind), best of 5,
minus the same circuit with the `H`s alone; divided by 32. Each constant is the mean of two runs. The values are
seconds per launch at `m_ref` and scale by `2^(m − m_ref)`. They are committed in `crates/aleph-cuda/src/dist/cost.rs`.

| kind | kernel | FP64 (m_ref = 27) | FP64 run spread | FP32 (m_ref = 28) | FP32 run spread |
|---|---|---|---|---|---|
| `dense1` | `apply_1q` | 1.755645e-2 | −0.1 % | 1.763479e-2 | −0.0 % |
| `dense2` | `apply_kq_tiled` k=2 | 1.840509e-2 | +2.0 % | 1.781169e-2 | −0.3 % |
| `dense3` | `apply_kq_tiled` k=3 | 3.436177e-2 | +2.4 % | 1.797886e-2 | −0.3 % |
| `diag1` | `apply_diag_1q` | 1.774244e-2 | −0.2 % | 1.769642e-2 | −0.2 % |
| `diag_k` | `apply_diag` k=2/3 | 1.759981e-2 | −0.1 % | 1.771855e-2 | −0.3 % |
| `cnot` | `apply_cnot` | 1.080041e-2 | −0.1 % | 1.295351e-2 | −0.3 % |
| `phase_base` | `apply_phase_poly` (per launch) | 2.147593e-2 | +2.0 % | 4.623386e-2 | +0.0 % |
| `phase_term` | per term, ≤ 1 cond | 6.712988e-4 | +0.2 % | 1.285620e-3 | +0.0 % |
| `phase_term_multi` | per term, AND of ≥ 2 conds | 4.776938e-4 | −0.2 % | 8.999014e-4 | −0.2 % |

Notes:
- `phase_term` is fitted from 1 vs 64 single-cond terms (one 2-bit parity cond). `phase_term_multi` is the slope of
  the same 1-vs-64 fit on terms `[1 << (n−1), 1 << s]` (AND of two 1-bit conds); the model reuses `phase_base`.
- `phase_base` includes the per-launch device allocation and upload of the phase terms.
- `dense1`, `diag1` and `diag_k` are within ~1 % of each other (the single-pass bandwidth floor), so their ordering is
  noise.

### 2.2 First attempt (back-to-back calibration, one phase per-term price)

The first calibration timed 32 identical launches back to back and priced every phase term alike (FP64:
`dense2` 2.113696e-2, `dense3` 3.630713e-2, `phase_base` 2.331010e-2, `phase_term` 6.563838e-4). With it the gate
**failed on QFT only**: model / measured = 1.152 / 1.156 (D=2 / D=4) in one run and 1.164 / 1.166 in another, worst
16.6 %. The other cells were GHZ 1.045–1.071, random d=10 0.951–0.960, QAOA 0.994–0.999, CCZ ladder 1.005–1.012.

A separate diagnosis (FP64, m=27) tested four hypotheses:
- **H1, roofline `max(floor, base + c·t)`: rejected.** A 1-term phase launch already takes 21–23 ms, above the
  17.6 ms bandwidth floor, so there is no plateau. The curve is mildly concave, which would under-predict, not over.
- **H2, term shape: confirmed.** 701 of QFT's 727 phase terms at g=1 are an AND of two 1-bit conds (fire on 1/4 of
  amplitudes); the calibration used one 2-bit parity cond (fires on 1/2). Measured per-term cost: 0.666 ms for the
  calibration form vs 0.497 ms for the QFT form. QAOA and the CCZ ladder have no multi-cond terms.
- **H3, per-launch allocation/upload: rejected.** At m=10 a phase launch costs 7.7–29 µs, under 0.1 % of QFT.
- **H4, back-to-back calibration: confirmed, affects every kind.** The same kernels interleaved with an `H` run 2–10 %
  cheaper per launch than 32 back to back at the 70 W cap (e.g. Iswap 21.34 → 19.27 ms, dense3 36.29 → 35.44 ms).
  On QFT's own rank programs, natural order took 2.374 s vs 2.580 s for phase-only plus rest-only back to back.
- The 0.36 s QFT gap split into ~0.16 s from H2 and ~0.21 s from H4.

So the model changed for two reasons that are measured on microbenchmarks, not fitted to QFT: phase terms are priced
by shape (`phase_term` / `phase_term_multi`), and every kind is calibrated interleaved (§2.1). The 0.10 bound, the
workloads and the gate method are unchanged.

### 2.3 Model accuracy gate (spec §6.3)

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture`

n=28, FP64, `Router::Lookahead`, all D ranks on one card with `LocalExchange`, best of 3.
- **measured compute** = `T(plan) − T(exchange-only plan)`, where the exchange-only plan is the same plan with every
  `Local` step emptied.
- **model all-ranks** = `Σ_steps Σ_r rank_segment(step, r)` (the gated quantity).
- **R·model(R−1)** = the representative-rank estimate `compile` will use (reported, not gated).

Workloads: QFT, GHZ, random = 1D brickwall d=10 (`Rx`/`Rz` layers + alternating nearest-neighbour `CNOT`s), QAOA
Max-Cut p=2 on a ring `(i, i+1 mod n)` plus 7 chords `(i, i + n/2)` for even `i < n/2` (not a regular graph), and the
CCZ ladder d=4.

Two runs on an idle box (load 0.22 and 0.31, GPU util 0 %, 1339 MiB resident before each). Run 2:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio |
|---|---|---|---|---|---|---|
| QFT | 2 | 2.357 | 2.485 | 1.054 | 2.507 | 1.064 |
| QFT | 4 | 2.357 | 2.500 | 1.061 | 2.510 | 1.065 |
| GHZ | 2 | 0.926 | 0.911 | 0.984 | 0.929 | 1.003 |
| GHZ | 4 | 0.882 | 0.875 | 0.992 | 0.883 | 1.001 |
| random d=10 | 2 | 7.065 | 6.066 | 0.859 | 6.066 | 0.859 |
| random d=10 | 4 | 8.165 | 7.112 | 0.871 | 7.288 | 0.893 |
| QAOA p=2 | 2 | 3.290 | 3.256 | 0.989 | 3.324 | 1.010 |
| QAOA p=2 | 4 | 3.323 | 3.249 | 0.978 | 3.300 | 0.993 |
| CCZ ladder d=4 | 2 | 1.055 | 1.062 | 1.007 | 1.062 | 1.007 |
| CCZ ladder d=4 | 4 | 1.057 | 1.060 | 1.003 | 1.060 | 1.003 |

Run 1's all-ranks ratios (D=2 / D=4): QFT 1.058 / 1.064, GHZ 0.990 / 1.001, random 0.864 / 0.876, QAOA 0.989 / 0.978,
CCZ ladder 1.008 / 1.003.

Worst |model/measured − 1| = **14.1 %** (run 2), **13.6 %** (run 1), both on random d=10 D=2.

**Gate FAILED** in both runs: the two random d=10 cells are under-predicted by more than 10 %. The other eight cells are
within ±6.4 % in both runs.

Model all-ranks compute by kind (s; the model is deterministic, so both runs agree):

| circuit | D | dense1 | dense2 | dense3 | diag1 | cnot | phase | exchange-only plan (s, run 2) |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 0.983 | 0.037 | 0 | 0.018 | 0 | 1.448 | 0.070 |
| QFT | 4 | 0.983 | 0.074 | 0 | 0.044 | 0 | 1.398 | 0.096 |
| GHZ | 2 | 0.018 | 0 | 0.893 | 0 | 0 | 0 | 0.043 |
| GHZ | 4 | 0 | 0.018 | 0.825 | 0 | 0.032 | 0 | 0.056 |
| random d=10 | 2 | 1.299 | 3.975 | 0.619 | 0 | 0.173 | 0 | 0.284 |
| random d=10 | 4 | 2.317 | 3.939 | 0.619 | 0 | 0.238 | 0 | 0.457 |
| QAOA p=2 | 2 | 2.879 | 0.037 | 0 | 0.071 | 0 | 0.268 | 0.150 |
| QAOA p=2 | 4 | 2.669 | 0.258 | 0 | 0.053 | 0 | 0.269 | 0.217 |
| CCZ ladder d=4 | 2 | 0.983 | 0 | 0 | 0 | 0 | 0.079 | 0.043 |
| CCZ ladder d=4 | 4 | 0.983 | 0 | 0 | 0 | 0 | 0.077 | 0.056 |

`diag_k` is 0 on every cell and is omitted.

### 2.4 Reading

- **Errors ranked (run 2, |model/measured − 1|):** random D=2 14.1 %, random D=4 12.9 %, QFT D=4 6.1 %, QFT D=2
  5.4 %, QAOA D=4 2.2 %, GHZ D=2 1.6 %, QAOA D=2 1.1 %, GHZ D=4 0.8 %, CCZ D=2 0.7 %, CCZ D=4 0.3 %. Run 1 has the
  same top four (random 13.6 % / 12.4 %, QFT 6.4 % / 5.8 %).
- **QFT now passes**, at +5.4 to +6.4 % over both runs (was +15.2 to +16.6 %). Its phase term fell from 1.666 to
  1.448 s at D=2 and from 1.605 to 1.398 s at D=4.
- **Random d=10 now fails low.** It is `dense2`-dominated (3.975 of 6.066 s at D=2, 66 %; 3.939 of 7.112 s at D=4,
  55 %). Interleaved calibration lowered FP64 `dense2` by 12.9 % (2.113696e-2 → 1.840509e-2), which moved this cell
  from 0.951–0.960 (first attempt) to 0.859–0.876. The gap (0.957–1.053 s) equals 24–27 % of the model's `dense2`
  term.
  - The diagnosis predicted this risk (random was already −4 to −5 % and Iswap dropped ~10 % interleaved).
  - Residual cause not isolated. One candidate: the `dense2` calibration payload (`Iswap` on qubits 0–5, interleaved
    with 1q `H`) is not representative of random's fused k=2 blocks; this run does not test it.
- **The other cells hold:** GHZ (`dense3`, 0.893 of 0.911 s at D=2) −1.6 to +0.1 %; QAOA (`dense1`, 2.879 of 3.256 s
  at D=2) −2.2 to −1.1 %; CCZ ladder (`dense1`, 0.983 of ~1.06 s) +0.3 to +0.8 %.
- **The representative-rank estimate over-predicts the all-ranks model by 0–2.5 %.** `R·model(R−1)` / model all-ranks
  is 1.000 (random D=2, CCZ both D) up to 1.025 (random D=4: 7.288 vs 7.112 s). Against measured compute it lands at
  0.859–1.065 (run 2).
- The exchange-only plan costs 0.043–0.457 s (run 2) on one card. That is on-card copy time, not interconnect time.
