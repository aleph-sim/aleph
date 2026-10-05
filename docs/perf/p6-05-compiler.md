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
minus the same circuit with the `H`s alone; divided by 32. `dense2` alone runs on a scrambled state: the H layer is
followed by one `Rx(0.3 + 0.17q)` and one `Rz(0.7 + 0.11q)` per qubit, in both the full and the baseline circuit
(§2.4). Each constant is the mean of two runs. The values are seconds per launch at `m_ref` and scale by
`2^(m − m_ref)`. They are committed in `crates/aleph-cuda/src/dist/cost.rs`.

| kind | kernel | FP64 (m_ref = 27) | FP64 run spread | FP32 (m_ref = 28) | FP32 run spread |
|---|---|---|---|---|---|
| `dense1` | `apply_1q` | 1.755645e-2 | −0.1 % | 1.763479e-2 | −0.0 % |
| `dense2` (scrambled state) | `apply_kq_tiled` k=2 | 2.168771e-2 | +3.3 % | 1.773724e-2 | +0.6 % |
| `dense3` | `apply_kq_tiled` k=3 | 3.436177e-2 | +2.4 % | 1.797886e-2 | −0.3 % |
| `diag1` | `apply_diag_1q` | 1.774244e-2 | −0.2 % | 1.769642e-2 | −0.2 % |
| `diag_k` | `apply_diag` k=2/3 | 1.759981e-2 | −0.1 % | 1.771855e-2 | −0.3 % |
| `cnot` | `apply_cnot` | 1.080041e-2 | −0.1 % | 1.295351e-2 | −0.3 % |
| `phase_base` | `apply_phase_poly` (per launch) | 2.147593e-2 | +2.0 % | 4.623386e-2 | +0.0 % |
| `phase_term` | per term, ≤ 1 cond | 6.712988e-4 | +0.2 % | 1.285620e-3 | +0.0 % |
| `phase_term_multi` | per term, AND of ≥ 2 conds | 4.776938e-4 | −0.2 % | 8.999014e-4 | −0.2 % |

Notes:
- `dense2` comes from a later pair of runs (with the scrambled state) than the other constants (uniform state). That
  later pair also re-printed the other constants. Their means moved from the committed values by at most +0.6 % for FP64
  (`dense3`) and +1.4 % for FP32 (`phase_term_multi`); the committed values were left unchanged.
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
- **H4, back-to-back calibration: confirmed on the uniform state.** Kernels interleaved with an `H` ran 2–10 %
  cheaper per launch than 32 back to back at the 70 W cap (e.g. Iswap 21.34 → 19.27 ms, dense3 36.29 → 35.44 ms).
  On QFT's own rank programs, natural order took 2.374 s vs 2.580 s for phase-only plus rest-only back to back.
- The 0.36 s QFT gap split into ~0.16 s from H2 and ~0.21 s from H4.

So the model changed for reasons measured on microbenchmarks, not fitted to QFT: phase terms are priced by shape
(`phase_term` / `phase_term_multi`), and every kind is calibrated interleaved.

### 2.3 Second attempt (interleaved, uniform state for every kind)

With interleaved calibration on the uniform H state (FP64 `dense2` 1.840509e-2, 12.9 % below the first attempt), QFT
passed (1.054–1.064) but **random d=10 failed low**: 0.859 / 0.871 (D=2 / D=4) in one run and 0.864 / 0.876 in
another, worst 14.1 %. GHZ was 0.984–1.001, QAOA 0.978–0.989, CCZ ladder 1.003–1.008.

A second diagnosis (FP64, m=27, random's rank R−1 programs replayed) found:
- **H5, qubit position: rejected.** Moving every Dense2 of random's real program onto the calibration qubits changed
  the natural-order replay by 0.2 % (3.612 → 3.606 s at g=1).
- **H6, block mix: rejected.** Random's Dense2 are `Unitary2q` blocks on the same kernel as Iswap; everything except
  Dense2 replayed at model/measured 0.973 (g=1) and 0.982 (g=2). The gap sits inside Dense2.
- **H8, data-dependent FP64 time at the power cap: confirmed.** Dense2 time depends on the amplitude values. Iswap
  costs 18.445 ms on the uniform H state and 22.052 ms on a scrambled (generic complex) state, ×1.196; random's
  in-situ Dense2 costs 23.3 ms. On the scrambled state, interleaved and back-to-back agree within 2 %, so the H4
  discount was a low-entropy-state effect. Dense3 shows the same effect (permutation payload 35.9 → 40.3 ms, ×1.12);
  Dense1 barely moves (17.55 → 17.81 ms, +1.5 %).

### 2.4 Third revision (Dense2 on a scrambled state)

Only `dense2` is now calibrated on the scrambled state (§2.1). Payload (Iswap), qubit pattern, interleaving and
best of 5 × 32 are unchanged; every other kind stays on the uniform state.

**Caveat: this per-kind choice was made after seeing the gate results.** Scrambling Dense3 as well (×1.12) would
push GHZ, whose state stays low-entropy (two nonzero amplitudes, 0/1 permutation blocks), to ~1.11 by the diagnosis'
estimate. On this power-capped card the FP64 cost of a kernel is partly a property of the workload's state, which a
per-kind constant cannot fully express: each constant carries an irreducible state-dependent error. Dense3 is the
known residual: random's 9 Dense3 blocks per rank run on a generic state at 42.5–42.9 ms in situ against the
34.4 ms constant, ≈ +0.15 s over all ranks (diagnosis estimate).

The 0.10 bound, the workloads and the gate method are unchanged across all three attempts.

### 2.5 Model accuracy gate (spec §6.3)

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture`

n=28, FP64, `Router::Lookahead`, all D ranks on one card with `LocalExchange`, best of 3.
- **measured compute** = `T(plan) − T(exchange-only plan)`, where the exchange-only plan is the same plan with every
  `Local` step emptied.
- **model all-ranks** = `Σ_steps Σ_r rank_segment(step, r)` (the gated quantity).
- **R·model(R−1)** = the representative-rank estimate `compile` will use (reported, not gated).

Workloads: QFT, GHZ, random = 1D brickwall d=10 (`Rx`/`Rz` layers + alternating nearest-neighbour `CNOT`s), QAOA
Max-Cut p=2 on a ring `(i, i+1 mod n)` plus 7 chords `(i, i + n/2)` for even `i < n/2` (not a regular graph), the
CCZ ladder d=4, and Grover with K=3 iterations (the `gpu_report_bench` construction: oracle and diffusion each use a
Z on qubit 27 controlled by qubits 0–7; K is fixed small because the gate measures per-launch model accuracy, not the
algorithm).

Before the Grover cell was added, two runs of the first ten cells gave worst 6.7 % and 6.9 % (QFT D=4). The final
gate below includes Grover. Two runs on an idle box (load 0.23 and 0.43, GPU util 0 %, 1339 MiB resident before
each). Run 2:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio |
|---|---|---|---|---|---|---|
| QFT | 2 | 2.359 | 2.492 | 1.056 | 2.514 | 1.065 |
| QFT | 4 | 2.359 | 2.513 | 1.065 | 2.523 | 1.070 |
| GHZ | 2 | 0.927 | 0.911 | 0.983 | 0.929 | 1.002 |
| GHZ | 4 | 0.882 | 0.879 | 0.996 | 0.890 | 1.009 |
| random d=10 | 2 | 7.055 | 6.775 | 0.960 | 6.775 | 0.960 |
| random d=10 | 4 | 8.144 | 7.815 | 0.960 | 7.990 | 0.981 |
| QAOA p=2 | 2 | 3.288 | 3.262 | 0.992 | 3.330 | 1.013 |
| QAOA p=2 | 4 | 3.323 | 3.295 | 0.991 | 3.346 | 1.007 |
| CCZ ladder d=4 | 2 | 1.060 | 1.062 | 1.002 | 1.062 | 1.002 |
| CCZ ladder d=4 | 4 | 1.055 | 1.060 | 1.004 | 1.060 | 1.004 |
| Grover K=3 | 2 | 6.741 | 6.969 | 1.034 | 6.969 | 1.034 |
| Grover K=3 | 4 | 6.540 | 6.808 | 1.041 | 6.808 | 1.041 |

Run 1's all-ranks ratios (D=2 / D=4): QFT 1.074 / 1.081, GHZ 1.013 / 1.025, random 0.980 / 0.976, QAOA 0.997 / 0.999,
CCZ ladder 1.010 / 1.005, Grover 1.035 / 1.042. Run 1 measured 0.1–3.0 % less compute than run 2 on every cell
(e.g. QFT D=2 2.319 vs 2.359 s, GHZ D=2 0.899 vs 0.927 s), so its ratios sit higher.

Worst |model/measured − 1| = **6.5 %** (run 2), **8.1 %** (run 1), both on QFT D=4.

**Gate PASSED** in both runs: every cell is within ±8.1 % (ratios 0.960–1.081 over both runs).

Model all-ranks compute by kind (s; the model is deterministic, so both runs agree):

| circuit | D | dense1 | dense2 | dense3 | diag1 | cnot | phase | exchange-only plan (s, run 2) |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 0.983 | 0.043 | 0 | 0.018 | 0 | 1.448 | 0.070 |
| QFT | 4 | 0.983 | 0.087 | 0 | 0.044 | 0 | 1.398 | 0.096 |
| GHZ | 2 | 0.018 | 0 | 0.893 | 0 | 0 | 0 | 0.043 |
| GHZ | 4 | 0 | 0.022 | 0.825 | 0 | 0.032 | 0 | 0.056 |
| random d=10 | 2 | 1.299 | 4.685 | 0.619 | 0 | 0.173 | 0 | 0.285 |
| random d=10 | 4 | 2.317 | 4.641 | 0.619 | 0 | 0.238 | 0 | 0.457 |
| QAOA p=2 | 2 | 2.879 | 0.043 | 0 | 0.071 | 0 | 0.268 | 0.151 |
| QAOA p=2 | 4 | 2.669 | 0.304 | 0 | 0.053 | 0 | 0.269 | 0.217 |
| CCZ ladder d=4 | 2 | 0.983 | 0 | 0 | 0 | 0 | 0.079 | 0.043 |
| CCZ ladder d=4 | 4 | 0.983 | 0 | 0 | 0 | 0 | 0.077 | 0.056 |
| Grover K=3 | 2 | 6.496 | 0.260 | 0 | 0.213 | 0 | 0 | 0.204 |
| Grover K=3 | 4 | 6.075 | 0.521 | 0 | 0.213 | 0 | 0 | 0.296 |

`diag_k` is 0 on every cell and is omitted: **the gate never exercises `DiagK`** (the CCZ ladder's diagonals fuse into
the phase polynomial), so the `diag_k` constant is unvalidated.

### 2.6 Reading

- **Errors ranked (run 2, |model/measured − 1|):** QFT D=4 6.5 %, QFT D=2 5.6 %, Grover D=4 4.1 %, random D=2 4.0 %,
  random D=4 4.0 %, Grover D=2 3.4 %, GHZ D=2 1.7 %, QAOA D=4 0.9 %, QAOA D=2 0.8 %, GHZ D=4 0.4 %, CCZ D=4 0.4 %,
  CCZ D=2 0.2 %. In run 1 the order is QFT D=4 8.1 %, QFT D=2 7.4 %, Grover D=4 4.2 %, Grover D=2 3.5 %, GHZ D=4
  2.5 %, random D=4 2.4 %, random D=2 2.0 %, then the rest ≤ 1.3 %.
- **QFT is the worst cell, over-predicted by 5.6–8.1 %.** It is the phase-dominated workload (phase 1.448 of 2.492 s
  at D=2, 58 %). Run 1's 8.1 % is the closest any cell came to the ±10 % edge.
- **Random d=10 is under-predicted by 2.0–4.0 %.** `dense2` is 4.685 of 6.775 s (69 %) at D=2 and 4.641 of 7.815 s
  (59 %) at D=4. The diagnosis' Dense3 state estimate (≈ +0.15 s) accounts for part of the gap; the rest is not
  isolated.
- **Grover: external controls are priced as a full pass.** The model prices a gate with local external controls by
  its target kind over the whole slice, although the kernels skip amplitudes whose control bits are clear. Grover's
  controlled-Z launches are the 0.213 s `diag1` column (3 % of its model); the cell is dominated by `dense1` (6.496 of
  6.969 s at D=2, 93 %). It is over-predicted by 3.4–4.2 %, which bounds this conservative over-estimate on this
  workload.
- **GHZ holds with the uniform-state Dense3** (0.983–1.025; `dense3` 0.893 of 0.911 s at D=2).
- **QAOA** (`dense1`, 2.879 of 3.262 s at D=2) is within 0.9 %; the **CCZ ladder** (`dense1`, 0.983 of ~1.06 s) within
  1.0 %.
- **The representative-rank estimate over-predicts the all-ranks model by 0–2.2 %.** `R·model(R−1)` / model all-ranks
  is 1.000 (random D=2, CCZ both D, Grover both D) up to 1.022 (random D=4: 7.990 vs 7.815 s). Against measured
  compute it lands at 0.960–1.070 (run 2). Rank R−1 is representative (typically the busiest), not a strict bound.
- The exchange-only plan costs 0.043–0.457 s (run 2) on one card. That is on-card copy time, not interconnect time.
- Limit: the constants are per-kind, but at the 70 W cap FP64 cost also depends on the state (§2.4). The gate passes
  on these six workloads; a workload whose state entropy differs from what its dominant kind was calibrated on can
  miss by up to the measured state factor (×1.2 for Dense2, ×1.12 for Dense3). The model is valid near `m_ref`; it
  has no launch-latency floor, so small-m costs are not meaningful.
