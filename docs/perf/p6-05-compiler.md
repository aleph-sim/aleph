# P6-05: Communication-aware compiler for distributed SV

Spec: `docs/superpowers/specs/2026-10-04-p6-05-comm-aware-compiler-design.md`. Issue #59.

This report grows over three PRs.
- PR 1 adds the commutation DAG, the reordering scheduler (`Router::Reorder`) and initial placement, and reports CPU
  communication counts.
- PR 2 adds the calibrated GPU cost model.
- PR 3 adds `compile`, `run_compiled` and the predicted-time benchmark (§3).

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

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture` (replaced by `dist_cost_states`, §4.1)

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

## 3. Compiler (PR 3)

`compile` builds {Naive, Lookahead, Reorder{1..=g}} × {identity, initial placement}, prices each with the PR 2
`GpuCostModel`, and runs the cheapest (`DistSvBackend::run_compiled`). Bench:
`cargo test --release -p aleph-cuda --features cuda --test dist_compile_bench -- --ignored --nocapture`
(RTX 4000 SFF Ada, driver 580.178.04 as reported by `nvidia-smi` on 2026-10-05, two idle-box runs on 2026-10-05, each followed by a re-run of the PR 2 gate).

**Metric (spec §8).** `T_pred = (T_full − T_comm)/D + Σ bytes/BW(k)`, with measured compute from one card running
all D ranks, exchange copies subtracted with the exchange-only plan, and the AWS g6 link table (FP64 7.16 / 4.35 GB/s
for one- / two-bit exchanges). The model's own compute estimate chooses the plan but is not the evidence.

Columns: `exch N/L/C` = exchanges of the Naive / Lookahead / compiled plan; `model/measured (C)` = the model's
all-ranks compute for the compiled plan over its measured compute; `compile (ms)` = one cold `compile_detailed` call;
`measured all N/L/C` = measured all-ranks compute (`T_full − T_comm`) of the three plans.

### 3.1 FP64, n=28

Run 2:

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.500 | 1.496 | 1.335 | 0.893 | 0.893 | 1.033 | 1.0 | 2.401/2.392/2.371 |
| QFT | 4 | naive+place | 3/2/2 | 0.824 | 0.971 | 0.729 | 0.885 | 0.751 | 1.031 | 1.4 | 2.396/2.402/2.318 |
| GHZ | 2 | naive | 1/1/1 | 0.626 | 0.627 | 0.626 | 1.000 | 0.998 | 0.957 | 0.1 | 0.952/0.954/0.952 |
| GHZ | 4 | naive | 2/1/2 | 0.379 | 0.412 | 0.379 | 1.000 | 0.920 | 0.962 | 0.1 | 0.916/0.907/0.916 |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 7.641 | 5.120 | 2.937 | 0.574 | 0.574 | 0.899 | 1.1 | 9.284/7.240/5.573 |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 6.414 | 4.105 | 1.547 | 0.377 | 0.377 | 0.898 | 1.5 | 12.458/8.274/5.589 |
| QAOA p=2 | 2 | reorder k=1 | 9/5/2 | 3.004 | 2.401 | 1.917 | 0.799 | 0.799 | 0.990 | 0.4 | 3.308/3.302/3.234 |
| QAOA p=2 | 4 | reorder k=1 | 16/5/4 | 2.061 | 1.759 | 1.100 | 0.626 | 0.626 | 0.994 | 0.5 | 3.446/3.332/3.201 |
| CCZ ladder d=4 | 2 | naive | 1/1/1 | 0.680 | 0.680 | 0.680 | 1.000 | 1.000 | 1.002 | 0.5 | 1.060/1.060/1.060 |
| CCZ ladder d=4 | 4 | naive | 2/1/2 | 0.415 | 0.450 | 0.415 | 1.000 | 0.922 | 1.001 | 0.6 | 1.059/1.058/1.059 |
| Grover K=3 | 2 | reorder k=1 | 13/7/1 | 5.409 | 4.422 | 1.598 | 0.361 | 0.361 | 1.068 | 0.5 | 6.920/6.745/2.896 |
| Grover K=3 | 4 | reorder k=1 | 20/7/2 | 3.230 | 2.932 | 0.874 | 0.298 | 0.298 | 1.067 | 0.6 | 6.921/6.545/2.898 |

Run 1 (same choices and exchange counts):

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.466 | 1.466 | 1.305 | 0.891 | 0.891 | 1.060 | 1.0 | 2.332/2.331/2.311 |
| QFT | 4 | naive+place | 3/2/2 | 0.808 | 0.956 | 0.716 | 0.885 | 0.748 | 1.057 | 1.4 | 2.334/2.344/2.262 |
| GHZ | 2 | naive | 1/1/1 | 0.609 | 0.611 | 0.609 | 1.000 | 0.997 | 0.992 | 0.1 | 0.919/0.923/0.919 |
| GHZ | 4 | naive | 2/1/2 | 0.372 | 0.404 | 0.372 | 1.000 | 0.919 | 0.994 | 0.1 | 0.887/0.877/0.887 |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 7.578 | 5.059 | 2.899 | 0.573 | 0.573 | 0.912 | 1.1 | 9.158/7.118/5.497 |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 6.409 | 4.100 | 1.544 | 0.377 | 0.377 | 0.900 | 1.5 | 12.439/8.255/5.577 |
| QAOA p=2 | 2 | reorder k=1 | 9/5/2 | 3.003 | 2.400 | 1.917 | 0.799 | 0.799 | 0.990 | 0.4 | 3.307/3.300/3.235 |
| QAOA p=2 | 4 | reorder k=1 | 16/5/4 | 2.062 | 1.758 | 1.100 | 0.626 | 0.626 | 0.994 | 0.5 | 3.447/3.330/3.201 |
| CCZ ladder d=4 | 2 | naive | 1/1/1 | 0.680 | 0.680 | 0.680 | 1.000 | 1.000 | 1.002 | 0.5 | 1.060/1.060/1.060 |
| CCZ ladder d=4 | 4 | naive | 2/1/2 | 0.414 | 0.450 | 0.414 | 1.000 | 0.922 | 1.003 | 0.6 | 1.058/1.057/1.058 |
| Grover K=3 | 2 | reorder k=1 | 13/7/1 | 5.410 | 4.423 | 1.599 | 0.362 | 0.362 | 1.067 | 0.5 | 6.921/6.747/2.899 |
| Grover K=3 | 4 | reorder k=1 | 20/7/2 | 3.230 | 2.933 | 0.875 | 0.298 | 0.298 | 1.067 | 0.7 | 6.922/6.547/2.899 |

Every candidate's model `T` (s; deterministic, identical in both runs). The chosen candidate is in bold; ties within
1e-9 go to fewer exchanges, then fewer `Local` instructions, then the earlier candidate.

| circuit | D | naive | lookahead | reorder k=1 | reorder k=2 | naive+place | lookahead+place | reorder k=1+place | reorder k=2+place |
|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 1.548 | 1.557 | 1.557 | – | **1.385** | 1.385 | 1.385 | – |
| QFT | 4 | 0.849 | 1.001 | 0.853 | 1.001 | **0.750** | 0.794 | 0.750 | 0.794 |
| GHZ | 2 | **0.614** | 0.614 | 0.614 | – | – | – | – | – |
| GHZ | 4 | **0.379** | 0.408 | 0.379 | 0.408 | – | – | – | – |
| random d=10 | 2 | 7.648 | 4.887 | **2.656** | – | – | – | – | – |
| random d=10 | 4 | 6.520 | 4.034 | **1.405** | 1.431 | – | – | – | – |
| QAOA p=2 | 2 | 3.000 | 2.415 | **1.901** | – | – | – | – | – |
| QAOA p=2 | 4 | 2.059 | 1.762 | **1.095** | 1.164 | – | – | – | – |
| CCZ ladder d=4 | 2 | **0.681** | 0.681 | 0.681 | – | – | – | – | – |
| CCZ ladder d=4 | 4 | **0.415** | 0.450 | 0.416 | 0.451 | – | – | – | – |
| Grover K=3 | 2 | 5.497 | 4.534 | **1.696** | – | – | – | – | – |
| Grover K=3 | 4 | 3.273 | 2.998 | **0.923** | 0.958 | – | – | – | – |

`–`: not built (`reorder k=2` needs g=2; the placed set is skipped when `initial_placement` is the identity).

### 3.2 FP32, n=28 (reported, not gated)

Run 2 (run 1 in brackets where it differs):

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.168 [1.169] | 1.160 [1.162] | 1.076 [1.077] | 0.927 | 0.927 | 1.006 | 1.0 | 2.033/2.018/2.001 [2.036/2.020/2.002] |
| QFT | 4 | naive+place | 3/2/2 | 0.621 | 0.685 | 0.562 | 0.905 | 0.821 | 1.007 | 1.3 | 2.031/1.991/1.946 [2.030/1.992/1.946] |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 3.672 [3.670] | 2.165 [2.164] | 0.824 | 0.381 [0.380] | 0.381 [0.380] | 0.981 [0.982] | 1.1 | 4.314/2.815/1.497 [4.310/2.814/1.496] |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 3.218 | 1.864 | 0.453 | 0.243 | 0.243 | 0.974 [0.973] | 1.5 | 6.209/3.341/1.507 [6.207/3.343/1.510] |

Model `T` per candidate (FP32): QFT D=2 naive 1.178, lookahead 1.183, reorder k=1 1.183, all three placed 1.098
(naive+place chosen); QFT D=4 naive 0.627, lookahead 0.694, reorder k=1 0.629, reorder k=2 0.694, naive+place and
reorder k=1+place 0.569 (naive+place chosen), lookahead+place and reorder k=2+place 0.591; random D=2 naive 3.774,
lookahead 2.174, reorder k=1 0.810; random D=4 naive 3.349, lookahead 1.892, reorder k=1 0.443, reorder k=2 0.456.

### 3.3 Exit criteria

| criterion | result |
|---|---|
| 1. compiled ≤ min(Naive, Lookahead) + 3 % on every cell | **PASS** in both runs. Worst cell 1.000 (GHZ and CCZ ladder, where the compiled plan *is* Naive); best 0.298 (Grover D=4). |
| 2. random d=10 D=2 ≥ 15 % better than Lookahead | **PASS**: 42.6 % (run 2: 2.937 vs 5.120 s), 42.7 % (run 1: 2.899 vs 5.059 s). |
| 3a. model gate (§6.3) on Lookahead plans, re-run (the spec's exit 3) | **PASS**: worst \|model/measured − 1\| 5.6 % (run 2), 5.4 % (run 1), both on random d=10 D=2 (table below). |
| 3b. additional plan-level check (not part of the spec's exit): compiled plans' model/measured within ±10 % | **Borderline MISS on random d=10.** PASS on 10 of 12 cells (run 2; run 1: 12/12, D=4 exactly on the bound at 0.900). Run 2: 0.899 / 0.898 (D=2 / D=4), i.e. 10.1 % / 10.2 % low. Run 1: 0.912 / 0.900. Every other cell: 0.957–1.068. |
| compile time, ~1k gates (brickwall d=15, 1043 gates), g=2, best of 5 (target < 50 ms) | **2.2 ms** in both runs. |

Exit 3 is read as the spec §8 defines it, "the model gate (§6.3) holds on every cell", and the §6.3 gate is defined
on Lookahead plans; the compiled-plan check (3b) is reported alongside as an additional plan-level check. The PR 3
plan had bundled the two into exit 3, and this narrowing was decided after seeing these runs (under the plan's
wording, run 2 misses exit 3). The 3a worst cell, random d=10 D=2, moved from 2.0–4.0 % in PR 2 (run 1 2.0 %, run 2 4.0 %; §2.5/§2.6) to 5.4–5.6 % in
this re-run.

Note: the raw logs of these runs print the old `exit3` label for what this report calls 3b.

Gate re-run (spec exit 3a; `dist_cost_gate`, Lookahead plans, best of 3), run 2 with run 1 in brackets:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio |
|---|---|---|---|---|---|---|
| QFT | 2 | 2.389 [2.393] | 2.492 | 1.043 [1.041] | 2.514 | 1.052 [1.050] |
| QFT | 4 | 2.387 [2.389] | 2.513 | 1.053 [1.052] | 2.523 | 1.057 [1.056] |
| GHZ | 2 | 0.940 [0.941] | 0.911 | 0.970 [0.968] | 0.929 | 0.988 [0.987] |
| GHZ | 4 | 0.896 [0.895] | 0.879 | 0.981 [0.982] | 0.890 | 0.993 [0.994] |
| random d=10 | 2 | 7.180 [7.164] | 6.775 | 0.944 [0.946] | 6.775 | 0.944 [0.946] |
| random d=10 | 4 | 8.237 [8.218] | 7.815 | 0.949 [0.951] | 7.990 | 0.970 [0.972] |
| QAOA p=2 | 2 | 3.299 [3.298] | 3.262 | 0.989 [0.989] | 3.330 | 1.009 [1.010] |
| QAOA p=2 | 4 | 3.332 [3.328] | 3.295 | 0.989 [0.990] | 3.346 | 1.004 [1.005] |
| CCZ ladder d=4 | 2 | 1.061 [1.061] | 1.062 | 1.001 [1.001] | 1.062 | 1.001 [1.001] |
| CCZ ladder d=4 | 4 | 1.057 [1.057] | 1.060 | 1.002 [1.002] | 1.060 | 1.002 [1.002] |
| Grover K=3 | 2 | 6.746 [6.745] | 6.969 | 1.033 [1.033] | 6.969 | 1.033 [1.033] |
| Grover K=3 | 4 | 6.546 [6.544] | 6.808 | 1.040 [1.040] | 6.808 | 1.040 [1.040] |

The model's per-kind split for these Lookahead plans is unchanged from §2.5.

### 3.4 Plan structure (model only)

Why each plan wins or loses, from the plans themselves: exchange widths, local swaps, `Local` segments, the kernel
launches rank 0 issues after specialisation and fusion (`DistSvBackend::rank_pass_count`), and the model's all-ranks
compute by kind (s, with the all-ranks launch count of that kind in brackets). Model only, no timing:
`cargo test --release -p aleph-cuda --features cuda --test dist_compile_kinds -- --ignored --nocapture`. Launch
counts are counted, not derived from seconds: `all_ranks` under a model whose constant is 1 for one kind and 0 for
the rest, at `m_ref = m` (PhasePoly: `phase_base` = 1, per-term costs 0). `diag_k` is 0 on every Lookahead and
compiled plan and is omitted; the test also prints Naive's full split (only its exchange widths are listed here).

| circuit | D | plan | exchange widths | local swaps | `Local` segs | rank-0 launches | model all-ranks (s) | dense1 | dense2 | dense3 | diag1 | cnot | phase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive | 1,1 | | | | | | | | | | |
| QFT | 2 | lookahead | 1,1 | 1 | 3 | 55 | 2.492 | 0.983 [56] | 0.043 [2] | 0 | 0.018 [1] | 0 | 1.448 [51] |
| QFT | 2 | compiled (naive+place) | 1 | 0 | 2 | 54 | 2.448 | 0.983 [56] | 0 | 0 | 0.018 [1] | 0 | 1.448 [51] |
| QFT | 4 | naive | 1,1,1 | | | | | | | | | | |
| QFT | 4 | lookahead | 2,2 | 2 | 3 | 56 | 2.513 | 0.983 [112] | 0.087 [8] | 0 | 0.044 [5] | 0 | 1.398 [99] |
| QFT | 4 | compiled (naive+place) | 1,1 | 0 | 3 | 53 | 2.391 | 0.983 [112] | 0 | 0 | 0.009 [1] | 0 | 1.398 [99] |
| GHZ | 2 | lookahead = compiled | 1 | 0 | 2 | 13 | 0.911 | 0.018 [1] | 0 | 0.893 [26] | 0 | 0 | 0 |
| GHZ | 4 | lookahead | 2 | 0 | 2 | 14 | 0.879 | 0 | 0.022 [2] | 0.825 [48] | 0 | 0.032 [6] | 0 |
| GHZ | 4 | compiled (naive) | 1,1 | 0 | 3 | 13 | 0.881 | 0.035 [4] | 0 | 0.825 [48] | 0 | 0.022 [4] | 0 |
| random d=10 | 2 | naive | 1 ×20 | | | | | | | | | | |
| random d=10 | 2 | lookahead | 1 ×10 | 9 | 11 | 162 | 6.775 | 1.299 [74] | 4.685 [216] | 0.619 [18] | 0 | 0.173 [16] | 0 |
| random d=10 | 2 | compiled (reorder k=1) | 1 | 1 | 2 | 82 | 5.012 | 0.281 [16] | 0.607 [28] | 4.123 [120] | 0 | 0 | 0 |
| random d=10 | 4 | naive | 1 ×44 | | | | | | | | | | |
| random d=10 | 4 | lookahead | 2 ×11 | 20 | 12 | 188 | 7.815 | 2.317 [264] | 4.641 [428] | 0.619 [36] | 0 | 0.238 [44] | 0 |
| random d=10 | 4 | compiled (reorder k=1) | 1,1 | 2 | 3 | 82 | 5.020 | 0.246 [28] | 0.651 [60] | 4.123 [240] | 0 | 0 | 0 |
| QAOA p=2 | 2 | naive | 1 ×9 | | | | | | | | | | |
| QAOA p=2 | 2 | lookahead | 1 ×5 | 1 | 6 | 87 | 3.262 | 2.879 [164] | 0.043 [2] | 0 | 0.071 [4] | 0 | 0.268 [8] |
| QAOA p=2 | 2 | compiled (reorder k=1) | 1,1 | 2 | 3 | 87 | 3.202 | 2.844 [162] | 0.087 [4] | 0 | 0 | 0 | 0.271 [8] |
| QAOA p=2 | 4 | naive | 1 ×16 | | | | | | | | | | |
| QAOA p=2 | 4 | lookahead | 2 ×5 | 7 | 6 | 87 | 3.295 | 2.669 [304] | 0.304 [28] | 0 | 0.053 [6] | 0 | 0.269 [16] |
| QAOA p=2 | 4 | compiled (reorder k=1) | 1 ×4 | 4 | 5 | 86 | 3.181 | 2.739 [312] | 0.174 [16] | 0 | 0 | 0 | 0.268 [16] |
| CCZ ladder d=4 | 2 | lookahead = compiled | 1 | 0 | 2 | 29 | 1.062 | 0.983 [56] | 0 | 0 | 0 | 0 | 0.079 [2] |
| CCZ ladder d=4 | 4 | lookahead | 2 | 0 | 2 | 29 | 1.060 | 0.983 [112] | 0 | 0 | 0 | 0 | 0.077 [4] |
| CCZ ladder d=4 | 4 | compiled (naive) | 1,1 | 0 | 3 | 29 | 1.061 | 0.983 [112] | 0 | 0 | 0 | 0 | 0.078 [4] |
| Grover K=3 | 2 | naive | 1 ×13 | | | | | | | | | | |
| Grover K=3 | 2 | lookahead | 1 ×7 | 6 | 8 | 197 | 6.969 | 6.496 [370] | 0.260 [12] | 0 | 0.213 [12] | 0 | 0 |
| Grover K=3 | 2 | compiled (reorder k=1) | 1 | 0 | 2 | 88 | 3.092 | 2.879 [164] | 0 | 0 | 0.213 [12] | 0 | 0 |
| Grover K=3 | 4 | naive | 1 ×20 | | | | | | | | | | |
| Grover K=3 | 4 | lookahead | 2 ×7 | 12 | 8 | 191 | 6.808 | 6.075 [692] | 0.521 [48] | 0 | 0.213 [24] | 0 | 0 |
| Grover K=3 | 4 | compiled (reorder k=1) | 1,1 | 0 | 3 | 88 | 3.092 | 2.879 [328] | 0 | 0 | 0.213 [24] | 0 | 0 |

At GHZ and CCZ D=2 Naive, Lookahead and the compiled plan have one 1-bit exchange each (identical model `T`). The
model-all-ranks column for the compiled random plans is what `model/measured (C)` in §3.1 divides by its measurement.

**Random d=10 under the compiled plan: a Dense3 test.** The model prices each kind with one per-kind constant; Dense3
is calibrated on the uniform H state, and §2.3–2.4 measured it at ×1.12 on a generic state. Raising only the model's
`dense3` term by ×1.12, everything else unchanged, gives:

| plan | D | model all-ranks (s) | dense3 share (s) | model with dense3 × 1.12 (s) | measured run 2 [run 1] (s) | ratio as gated, run 2 [run 1] | ratio with dense3 × 1.12, run 2 [run 1] | dense3 factor that would close the gap, run 2 [run 1] |
|---|---|---|---|---|---|---|---|---|
| compiled (reorder k=1) | 2 | 5.012 | 4.123 | 5.507 | 5.573 [5.497] | 0.899 [0.912] | 0.988 [1.002] | 1.136 [1.118] |
| compiled (reorder k=1) | 4 | 5.020 | 4.123 | 5.515 | 5.589 [5.577] | 0.898 [0.900] | 0.987 [0.989] | 1.138 [1.135] |
| lookahead | 2 | 6.775 | 0.619 | 6.849 | 7.240 [7.118] | 0.936 [0.952] | 0.946 [0.962] | 1.751 [1.554] |
| lookahead | 4 | 7.815 | 0.619 | 7.889 | 8.274 [8.255] | 0.945 [0.947] | 0.954 [0.956] | 1.742 [1.711] |

(Measured = the bench's `measured all` for that plan in §3.1. "Factor that would close the gap" =
(measured − model without dense3) / dense3 share; it assumes every other kind is priced exactly.)

### 3.5 Reading

- **What wins, per workload** (§3.1 and §3.4 tables):
  - **QFT picks naive+place.** Placement moves the qubits QFT needs late onto the global slots, so the plan needs one
    1-bit exchange at D=2 (Lookahead: two, plus 1 local swap) and two 1-bit exchanges at D=4 (Lookahead: two 2-bit
    exchanges, plus 2 local swaps; Naive: three 1-bit). Compute barely moves (rank-0 launches 55 → 54 and 56 → 53;
    Lookahead's Dense2 launches, 2 and 8 all-ranks, go away with its local swaps). The gain is mostly link time: compiled / L = 0.893
    (D=2) and 0.751 (D=4) in run 2. At D=2 all three placed candidates tie at 1.385 and the earliest, naive+place,
    is kept; at D=4 naive+place and reorder k=1+place tie at 0.750.
  - **GHZ and the CCZ ladder pick Naive, which ties the others.** At D=2 Naive, Lookahead and reorder k=1 are the same
    one-exchange plan (model 0.614 and 0.681). At D=4 Naive (and reorder k=1) use two 1-bit exchanges where Lookahead
    uses one 2-bit exchange. That is cheaper on the AWS table: a 2-bit exchange moves 3/4 of a slice at 4.35 GB/s, two
    1-bit exchanges move 1/2 each at 7.16 GB/s, and 0.75/4.35 > 2 × 0.5/7.16. Compute is unchanged (GHZ 0.881 vs
    0.879 s model, CCZ 1.061 vs 1.060 s), so compiled / L = 0.920 and 0.922. Exit 1's worst cell (1.000) is these
    ties: the compiled plan *is* Naive there.
  - **Random d=10 picks reorder k=1.** One exchange instead of 10 at D=2, two 1-bit exchanges instead of eleven 2-bit
    ones at D=4; local swaps 9 → 1 and 20 → 2. The shallow brickwall's light cone lets the far qubits finish all 10
    layers first (§1). The longer `Local` segments (2 instead of 11) also fuse much better: rank-0 launches 162 → 82
    (D=2) and 188 → 82 (D=4), and measured all-ranks compute falls 7.240 → 5.573 s (D=2, run 2). Both levers together
    give compiled / L = 0.574 and 0.377.
  - **QAOA picks reorder k=1.** It is a communication win only: 2 exchanges instead of 5 at D=2, four 1-bit instead
    of five 2-bit at D=4, with the same launch count (87 → 87, 87 → 86) and measured compute within 2–4 % (3.302 →
    3.234 s, 3.332 → 3.201 s). Compiled / L = 0.799 and 0.626.
  - **Grover picks reorder k=1, and the win is mostly compute.** Exchanges 7 → 1 (D=2) and seven 2-bit → two 1-bit
    (D=4), local swaps 6 → 0 and 12 → 0. But measured all-ranks compute also drops 6.745 → 2.896 s (D=2, run 2),
    2.3× less. Mechanism, verified by the counted launches in §3.4: the multi-controlled Z acts on qubits 0–7 and 27, so the H/X
    runs on the other 19 qubits commute with every MCZ. Lookahead keeps program order, so each MCZ splits every
    qubit's H/X sequence into 7 runs (initial H, then H·X and X·H around each of the 6 MCZs). Reorder emits a free
    qubit's whole sequence back to back, and the 1q fuser collapses it into one gate. The compiled plan's Dense1
    count per rank is then 19 × 1 + 9 × 7 = 82, which is exactly what §3.4 shows (164 all-ranks at D=2 = 82 per rank;
    328 at D=4 = 82 per rank), plus 6 MCZ (`diag1`) launches per rank: 88 rank-0 launches against Lookahead's 197 (D=2)
    and 191 (D=4). The model sees the same thing (6.969 → 3.092 s all-ranks).
- **reorder k=1 beats k=2 wherever both exist** (every D=4 cell: random 1.405 vs 1.431, QAOA 1.095 vs 1.164, Grover
  0.923 vs 0.958, GHZ 0.379 vs 0.408, CCZ 0.416 vs 0.451, QFT 0.853 vs 1.001 and with placement 0.750 vs 0.794). The
  reason is the same link asymmetry as GHZ above: on this AWS table two 1-bit exchanges cost less than one 2-bit
  exchange, so a 2-bit exchange pays only where it replaces more than two 1-bit ones; on these cells k=1 is cheaper
  every time.
- **The model's choice agrees with the measurement on every cell.** In each row of §3.1 the compiled `T_pred` is the
  lowest of the three measured plans (or ties Naive). The candidates the bench did not measure (e.g. reorder k=2) are
  ranked by the model only.
- **Exit 3b, random d=10 (borderline MISS): the cause is the Dense3 constant on a generic state.** The compiled
  random plan is a different kernel mix from the Lookahead plan the gate checks: its longer segments fuse into
  120 Dense3 launches (D=2, all ranks) instead of 18, so `dense3` is 4.123 of 5.012 s of its model (82 %) against
  0.619 of 6.775 s (9 %) for Lookahead. Dense3 is calibrated on the uniform H state, and the brickwall's state is
  generic. Raising only `dense3` by the ×1.12 that §2.3 measured on a generic state brings the compiled ratios to
  0.987–1.002 in both runs (§3.4 table); the factor that would close the gap exactly is 1.118–1.138. So the hypothesis
  is **confirmed** by the breakdown. It does not explain Lookahead's smaller under-prediction (0.936–0.952 here,
  0.944–0.951 in the gate re-run): there Dense3 is too small a share (it would need ×1.55–1.75), so that residual is
  elsewhere, as §2.6 already said. The constants were **not** re-calibrated against this result (that would be
  in-sample tuning again).
  - The miss did not change the choice: the compiled random plan's predicted time (`T_pred`) is 42.6–42.7 % lower
    than Lookahead's (exit 2), and its measured compute is lower too (5.573 vs 7.240 s at D=2, run 2). An
    under-priced Dense3 favours fusion-heavy plans, so a better constant could move a close call; none of these cells
    is close (reorder k=1 model 2.656 vs lookahead 4.887 at D=2).
  - Supporting evidence: FP32 has no such gap. Its `dense3` constant sits on the bandwidth floor (§2.1), and the same
    compiled random plans land at 0.973–0.982 (§3.2).
  - Follow-up (#538): calibrate Dense3 for generic states, or price kernels by state entropy (a per-workload state factor),
    and re-run both gates out of sample.
- **FP32** makes the same choices as FP64 (naive+place for QFT, reorder k=1 for random). Compiled / L is 0.927 and
  0.821 for QFT and 0.381 and 0.243 for random (run 2). The random gains are larger than FP64's because the compiled
  plan's measured compute falls further (2.815 → 1.497 s at D=2, against 7.240 → 5.573 s for FP64): FP32 Dense3 is not
  compute-bound (§2.1). Model/measured for the compiled plans is 0.973–1.007 over both runs.
- **Run-to-run spread.** Run 2 is slower than run 1 on the cells measured first: measured compute +2.5–3.6 % on QFT and
  GHZ (e.g. GHZ D=2 0.919 → 0.952 s compiled), +1.4–1.7 % on random D=2, and ≤ 0.25 % on random D=4, QAOA, CCZ ladder,
  Grover and every FP32 cell. All choices, exchange counts and verdicts are identical across runs except exit 3b on
  random d=10, which sits on the ±10 % edge in both runs (run 1 0.912 / 0.900, run 2 0.899 / 0.898). The gate re-run
  moved by ≤ 0.3 % per cell.
- **Compile time** is 2.2 ms for 1043 gates at g=2 (best of 5), well under the 50 ms target. Single cold calls
  on the §3.1 workloads take 0.1–1.5 ms.
- On one card `T_pred` is a prediction: compute is measured here, the link term is the AWS g6 table, and no
  multi-GPU run of the compiled plans was made.

## 4. State-class cost model (#538)

Spec: `docs/superpowers/specs/2026-10-05-p6-05-state-class-cost-design.md`. The rule and the constants below were
fixed and pushed (commit `ab514f7`) before any Stage C run.

### 4.1 Stage A: per-kind time by state

Bench: `cargo test --release -p aleph-cuda --features cuda --test dist_cost_states -- --ignored --nocapture`
(RTX 4000 SFF Ada, 2026-10-05, idle box, two passes in one invocation; raw log
`p6-05-state-class/stage-a.log`). Cells are ms per launch, mean of the two passes (pass 1 / pass 2). States (spec §2):
(a) uniform H; (b) (a) + Rz per qubit (equal magnitudes, varied phases); (c) (a) + Ry per qubit (real, varied
magnitudes); (d) (a) + Rx + Rz per qubit (generic complex); (e) GHZ; (f) 4 Clifford layers.

**FP64, m_ref = 27:**

| kind | a | b | c | d | e | f | d/a |
|---|---|---|---|---|---|---|---|
| dense1 | 17.549 (17.520/17.579) | 17.745 (17.723/17.767) | 17.639 (17.662/17.615) | 17.806 (17.778/17.834) | 17.548 (17.547/17.549) | 17.560 (17.609/17.510) | 1.015 |
| dense2 | 18.659 (18.039/19.280) | 22.701 (22.626/22.776) | 22.709 (22.828/22.590) | 22.684 (22.763/22.604) | 18.965 (18.910/19.019) | 18.962 (18.916/19.009) | 1.216 |
| dense3 | 34.657 (33.663/35.651) | 40.058 (39.997/40.119) | 37.981 (38.033/37.930) | 40.087 (40.205/39.969) | 35.584 (35.590/35.579) | 35.682 (35.762/35.601) | 1.157 |
| diag1 | 17.758 (17.724/17.792) | 17.636 (17.641/17.631) | 17.812 (17.794/17.830) | 17.652 (17.670/17.635) | 17.684 (17.670/17.697) | 17.646 (17.630/17.662) | 0.994 |
| diag_k | 17.619 (17.609/17.628) | 17.624 (17.632/17.617) | 17.639 (17.647/17.630) | 17.673 (17.627/17.720) | 17.627 (17.660/17.594) | 17.679 (17.675/17.683) | 1.003 |
| cnot | 10.770 (10.752/10.788) | 10.663 (10.668/10.658) | 10.775 (10.758/10.792) | 10.708 (10.715/10.701) | 10.798 (10.814/10.781) | 10.777 (10.834/10.721) | 0.994 |
| phase_base | 21.346 (20.933/21.759) | 24.958 (24.991/24.925) | 24.989 (25.014/24.964) | 24.850 (24.881/24.820) | 20.950 (20.972/20.929) | 20.949 (20.909/20.989) | 1.164 |
| phase_term | 0.670 (0.665/0.674) | 0.630 (0.631/0.629) | 0.631 (0.632/0.630) | 0.630 (0.631/0.629) | 0.640 (0.642/0.637) | 0.638 (0.638/0.639) | 0.941 |
| phase_term_multi | 0.477 (0.476/0.478) | 0.462 (0.463/0.461) | 0.468 (0.469/0.467) | 0.462 (0.464/0.461) | 0.463 (0.460/0.466) | 0.462 (0.462/0.463) | 0.970 |

| state | dense2 / a | measured | R1 predicts | R2 predicts |
|---|---|---|---|---|
| A | 1.000 | simple | simple | simple |
| B | 1.217 | generic | simple | generic |
| C | 1.217 | generic | generic | generic |
| D | 1.216 | generic | generic | generic |
| E | 1.016 | simple | simple | simple |
| F | 1.016 | simple | simple | simple |

**FP32, m_ref = 28:**

| kind | a | b | c | d | e | f | d/a |
|---|---|---|---|---|---|---|---|
| dense1 | 17.601 (17.596/17.606) | 17.678 (17.691/17.664) | 17.598 (17.601/17.596) | 17.661 (17.658/17.665) | 17.548 (17.567/17.530) | 17.593 (17.595/17.591) | 1.003 |
| dense2 | 17.785 (17.782/17.788) | 17.725 (17.741/17.708) | 17.829 (17.813/17.845) | 17.741 (17.741/17.740) | 17.769 (17.800/17.739) | 17.781 (17.814/17.749) | 0.998 |
| dense3 | 17.987 (17.999/17.975) | 17.334 (17.325/17.343) | 17.946 (17.954/17.938) | 17.274 (17.258/17.290) | 17.967 (17.936/17.997) | 17.863 (17.835/17.891) | 0.960 |
| diag1 | 17.711 (17.734/17.687) | 17.608 (17.565/17.651) | 17.692 (17.695/17.689) | 17.599 (17.585/17.612) | 17.645 (17.652/17.638) | 17.575 (17.476/17.673) | 0.994 |
| diag_k | 17.660 (17.672/17.647) | 17.681 (17.696/17.667) | 17.734 (17.736/17.732) | 17.698 (17.673/17.722) | 17.680 (17.700/17.661) | 17.698 (17.712/17.684) | 1.002 |
| cnot | 12.939 (12.928/12.950) | 12.962 (12.959/12.965) | 12.933 (12.957/12.908) | 12.989 (12.992/12.987) | 12.973 (12.967/12.978) | 12.924 (12.909/12.940) | 1.004 |
| phase_base | 46.156 (46.091/46.221) | 49.670 (49.690/49.650) | 49.625 (49.751/49.500) | 49.589 (49.696/49.481) | 45.390 (45.515/45.265) | 45.294 (45.406/45.181) | 1.074 |
| phase_term | 1.281 (1.281/1.281) | 1.233 (1.231/1.234) | 1.236 (1.239/1.234) | 1.231 (1.233/1.229) | 1.206 (1.209/1.203) | 1.204 (1.208/1.200) | 0.961 |
| phase_term_multi | 0.907 (0.906/0.908) | 0.885 (0.886/0.885) | 0.891 (0.893/0.890) | 0.885 (0.887/0.883) | 0.895 (0.895/0.894) | 0.896 (0.896/0.895) | 0.975 |

| state | dense2 / a | measured | R1 predicts | R2 predicts |
|---|---|---|---|---|
| A | 1.000 | simple | simple | simple |
| B | 0.997 | simple | simple | generic |
| C | 1.002 | simple | generic | generic |
| D | 0.998 | simple | generic | generic |
| E | 0.999 | simple | simple | simple |
| F | 1.000 | simple | simple | simple |

`measured` is rule 1 (generic iff `dense2` ≥ 1.05 × its (a) time); the R1/R2 columns are each candidate rule's
prediction from the state's preparation circuit.

**Decision (pre-registered rules, spec §2):**
- **Rule 1, FP64:** (b), (c), (d) are generic (dense2 1.216–1.217 × (a)); (e) and (f) are simple (1.016).
- **Rule 3, FP64: R2.** State (b) has equal magnitudes and only non-π/4 phases. It measured generic (×1.217), which
  only R2 predicts; R1 calls it simple. R2 matches all six states. So at the 70 W cap the phases alone are enough to
  slow the dense kernels: the effect is about the amplitude *values*, not their magnitudes.
- **Rule 2, FP64:** two constants for `dense2` (d/a 1.216), `dense3` (1.157) and `phase_base` (1.164). One constant
  for `dense1` (1.015), `diag1` (0.994), `diag_k` (1.003), `cnot` (0.994), `phase_term` (0.941) and
  `phase_term_multi` (0.970).
- **FP32 — the pre-registered stop fired.** No dense kind moves with the state (dense2 0.997–1.002 × (a) on every
  state), so rule 1 calls every state simple and neither R1 nor R2 matches (plan ruling 3). But rule 2 splits one
  kind, `phase_base` (d/a 1.074), and its per-state times follow the R2 classes exactly: 49.59–49.67 ms on
  (b)/(c)/(d) against 45.29–46.16 ms on (a)/(e)/(f). **The user decided, after seeing these tables,** to use R2 for
  FP32 too, with the `phase_base` split as printed. This is the one post-measurement choice in #538; it touches only
  FP32, which no exit criterion gates.
- **Rule 4:** the constants are the printed literals, unedited, in `crates/aleph-cuda/src/dist/cost.rs`
  (`RTX4000_FP64`, `RTX4000_FP32`).

Compared with the PR 2 constants (§2.4), the simple FP64 `dense2` drops from 21.69 ms (a scrambled-state value) to
18.66 ms (uniform), and the generic value is 22.68 ms. `dense3` gains a generic value of 40.09 ms (simple 34.66 ms,
PR 2 34.36 ms).

**Run-to-run noise.** The largest spread between the two passes is on state (a), FP64 dense2, dense3 and
phase_base. Pass 2 is the slower one in all three, and (a) is measured first in each pass. Every other cell moves by
≤ 1.4 %. The class decisions are not near the 5 % threshold: the generic states sit at 1.216–1.217 and the simple ones
at 1.000–1.016. Spread = |pass 1 − pass 2| / mean, from `p6-05-state-class/stage-a.log` (the whole Stage A invocation
took 5375 s):

| kind | precision | state | pass 1 (ms) | pass 2 (ms) | mean (ms) | spread |
|---|---|---|---|---|---|---|
| dense2 | FP64 | (a) | 18.039 | 19.280 | 18.659 | 6.6 % |
| dense3 | FP64 | (a) | 33.663 | 35.651 | 34.657 | 5.7 % |
| phase_base | FP64 | (a) | 20.933 | 21.759 | 21.346 | 3.9 % |
| phase_term | FP64 | (a) | 0.665 | 0.674 | 0.670 | 1.3 % |
| phase_term_multi | FP64 | (e) | 0.460 | 0.466 | 0.463 | 1.3 % |
| diag1 | FP32 | (f) | 17.476 | 17.673 | 17.575 | 1.1 % |

### 4.2 Stage C: old cells (seen during design)

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture`

Same method as §2.5 (n=28, FP64, `Router::Lookahead`, all D ranks on one card, best of 3), with the state-class model
and frozen constants of §4.1. `generic steps` = `Local` steps priced generic / all `Local` steps (the class walk;
deterministic, identical in both runs). Two runs back to back on 2026-10-05 (raw logs
`p6-05-state-class/stage-c-gate-run{1,2}.log`). **These runs did not satisfy spec §4's idle-box precondition.** The
`uptime` line logged before each gate run (raw: `p6-05-state-class/stage-c-runner.out`):

| run | time | 1-min load | 5-min load | 15-min load |
|---|---|---|---|---|
| 1 | 12:25:57 | 1.57 | 0.88 | 0.85 |
| 2 | 12:41:10 | 2.36 | 1.82 | 1.39 |

The load is residual from this session's own builds and runs, with no foreign workload: run 1's is the tail of a
compile that had just finished (GPU idle), run 2's is run 1's gate and compile bench just finishing. The verdicts
below are reported as measured; the deviation can only make them less clean, and the exit-1 MISS stands.

Run 1:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | verdict |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 2.323 | 2.657 | 1.144 | 2.682 | 1.155 | 2/3 | **MISS** |
| QFT | 4 | 2.325 | 2.667 | 1.147 | 2.679 | 1.152 | 2/3 | **MISS** |
| GHZ | 2 | 0.899 | 0.919 | 1.021 | 0.936 | 1.041 | 0/2 | PASS |
| GHZ | 4 | 0.860 | 0.883 | 1.027 | 0.891 | 1.036 | 0/2 | PASS |
| random d=10 | 2 | 6.910 | 7.092 | 1.026 | 7.092 | 1.026 | 11/11 | PASS |
| random d=10 | 4 | 8.014 | 8.129 | 1.014 | 8.305 | 1.036 | 12/12 | PASS |
| QAOA p=2 | 2 | 3.271 | 3.282 | 1.003 | 3.350 | 1.024 | 5/6 | PASS |
| QAOA p=2 | 4 | 3.298 | 3.326 | 1.008 | 3.377 | 1.024 | 5/6 | PASS |
| CCZ ladder d=4 | 2 | 1.052 | 1.069 | 1.016 | 1.069 | 1.016 | 1/2 | PASS |
| CCZ ladder d=4 | 4 | 1.054 | 1.066 | 1.011 | 1.066 | 1.011 | 1/2 | PASS |
| Grover K=3 | 2 | 6.733 | 6.930 | 1.029 | 6.930 | 1.029 | 0/8 | PASS |
| Grover K=3 | 4 | 6.530 | 6.733 | 1.031 | 6.733 | 1.031 | 0/8 | PASS |

Run 2:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | verdict |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 2.440 | 2.657 | 1.089 | 2.682 | 1.099 | 2/3 | PASS |
| QFT | 4 | 2.433 | 2.667 | 1.096 | 2.679 | 1.101 | 2/3 | PASS |
| GHZ | 2 | 0.961 | 0.919 | 0.956 | 0.936 | 0.974 | 0/2 | PASS |
| GHZ | 4 | 0.914 | 0.883 | 0.966 | 0.891 | 0.975 | 0/2 | PASS |
| random d=10 | 2 | 7.263 | 7.092 | 0.976 | 7.092 | 0.976 | 11/11 | PASS |
| random d=10 | 4 | 8.272 | 8.129 | 0.983 | 8.305 | 1.004 | 12/12 | PASS |
| QAOA p=2 | 2 | 3.300 | 3.282 | 0.994 | 3.350 | 1.015 | 5/6 | PASS |
| QAOA p=2 | 4 | 3.332 | 3.326 | 0.998 | 3.377 | 1.014 | 5/6 | PASS |
| CCZ ladder d=4 | 2 | 1.061 | 1.069 | 1.008 | 1.069 | 1.008 | 1/2 | PASS |
| CCZ ladder d=4 | 4 | 1.058 | 1.066 | 1.007 | 1.066 | 1.007 | 1/2 | PASS |
| Grover K=3 | 2 | 6.743 | 6.930 | 1.028 | 6.930 | 1.028 | 0/8 | PASS |
| Grover K=3 | 4 | 6.543 | 6.733 | 1.029 | 6.733 | 1.029 | 0/8 | PASS |

Worst |model/measured − 1| on old cells: **14.7 %** (run 1, QFT D=4), **9.6 %** (run 2, QFT D=4).

All-ranks ratio against the earlier models (D=2 / D=4; PR 2 = §2.5, PR 3 = the §3.3 gate re-run, same constants as
PR 2):

| circuit | PR 2 run 1 | PR 2 run 2 | PR 3 run 1 | PR 3 run 2 | #538 run 1 | #538 run 2 | generic steps (D=2 / D=4) |
|---|---|---|---|---|---|---|---|
| QFT | 1.074 / 1.081 | 1.056 / 1.065 | 1.041 / 1.052 | 1.043 / 1.053 | 1.144 / 1.147 | 1.089 / 1.096 | 2/3 / 2/3 |
| GHZ | 1.013 / 1.025 | 0.983 / 0.996 | 0.968 / 0.982 | 0.970 / 0.981 | 1.021 / 1.027 | 0.956 / 0.966 | 0/2 / 0/2 |
| random d=10 | 0.980 / 0.976 | 0.960 / 0.960 | 0.946 / 0.951 | 0.944 / 0.949 | 1.026 / 1.014 | 0.976 / 0.983 | 11/11 / 12/12 |
| QAOA p=2 | 0.997 / 0.999 | 0.992 / 0.991 | 0.989 / 0.990 | 0.989 / 0.989 | 1.003 / 1.008 | 0.994 / 0.998 | 5/6 / 5/6 |
| CCZ ladder d=4 | 1.010 / 1.005 | 1.002 / 1.004 | 1.001 / 1.002 | 1.001 / 1.002 | 1.016 / 1.011 | 1.008 / 1.007 | 1/2 / 1/2 |
| Grover K=3 | 1.035 / 1.042 | 1.034 / 1.041 | 1.033 / 1.040 | 1.033 / 1.040 | 1.029 / 1.031 | 1.028 / 1.029 | 0/8 / 0/8 |

Model all-ranks compute, PR 2/3 model → #538 model (s; deterministic):

| circuit | D=2 | D=4 |
|---|---|---|
| QFT | 2.492 → 2.657 | 2.513 → 2.667 |
| GHZ | 0.911 → 0.919 | 0.879 → 0.883 |
| random d=10 | 6.775 → 7.092 | 7.815 → 8.129 |
| QAOA p=2 | 3.262 → 3.282 | 3.295 → 3.326 |
| CCZ ladder d=4 | 1.062 → 1.069 | 1.060 → 1.066 |
| Grover K=3 | 6.969 → 6.930 | 6.808 → 6.733 |

#538 model all-ranks compute by kind (s; identical in both runs; `t_comm` = exchange-only plan, run 1; `diag_k` is 0
on every old cell and omitted):

| circuit | D | dense1 | dense2 | dense3 | diag1 | cnot | phase | t_comm (s) |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 0.983 | 0.037 | 0 | 0.018 | 0 | 1.619 | 0.070 |
| QFT | 4 | 0.983 | 0.075 | 0 | 0.044 | 0 | 1.565 | 0.096 |
| GHZ | 2 | 0.018 | 0 | 0.901 | 0 | 0 | 0 | 0.043 |
| GHZ | 4 | 0 | 0.019 | 0.832 | 0 | 0.032 | 0 | 0.056 |
| random d=10 | 2 | 1.299 | 4.900 | 0.722 | 0 | 0.172 | 0 | 0.283 |
| random d=10 | 4 | 2.317 | 4.854 | 0.722 | 0 | 0.237 | 0 | 0.457 |
| QAOA p=2 | 2 | 2.878 | 0.037 | 0 | 0.071 | 0 | 0.295 | 0.150 |
| QAOA p=2 | 4 | 2.668 | 0.310 | 0 | 0.053 | 0 | 0.296 | 0.217 |
| CCZ ladder d=4 | 2 | 0.983 | 0 | 0 | 0 | 0 | 0.086 | 0.043 |
| CCZ ladder d=4 | 4 | 0.983 | 0 | 0 | 0 | 0 | 0.083 | 0.056 |
| Grover K=3 | 2 | 6.493 | 0.224 | 0 | 0.213 | 0 | 0 | 0.203 |
| Grover K=3 | 4 | 6.072 | 0.448 | 0 | 0.213 | 0 | 0 | 0.297 |

**QFT MISS breakdown** (gate `kinds:` lines, run 1; run 2 identical except `t_full` 2.510 / 2.529):

```
kinds: QFT D=2 t_full=2.393 t_comm=0.070 dense1=0.983 dense2=0.037 dense3=0.000 diag1=0.018 diag_k=0.000 cnot=0.000 phase=1.619
kinds: QFT D=4 t_full=2.422 t_comm=0.096 dense1=0.983 dense2=0.075 dense3=0.000 diag1=0.044 diag_k=0.000 cnot=0.000 phase=1.565
```

Measured compute run 1 → run 2 (the order the gate measures cells in):

| circuit | D=2 (s) | change | D=4 (s) | change |
|---|---|---|---|---|
| QFT | 2.323 → 2.440 | +5.0 % | 2.325 → 2.433 | +4.6 % |
| GHZ | 0.899 → 0.961 | +6.9 % | 0.860 → 0.914 | +6.3 % |
| random d=10 | 6.910 → 7.263 | +5.1 % | 8.014 → 8.272 | +3.2 % |
| QAOA p=2 | 3.271 → 3.300 | +0.9 % | 3.298 → 3.332 | +1.0 % |
| CCZ ladder d=4 | 1.052 → 1.061 | +0.9 % | 1.054 → 1.058 | +0.4 % |
| Grover K=3 | 6.733 → 6.743 | +0.1 % | 6.530 → 6.543 | +0.2 % |
| HEA d=4 | 8.705 → 8.755 | +0.6 % | 8.703 → 8.735 | +0.4 % |
| random d=20 | 13.968 → 14.012 | +0.3 % | 16.300 → 16.312 | +0.1 % |
| Clifford brickwall d=10 | 6.048 → 6.037 | −0.2 % | 7.244 → 7.238 | −0.1 % |
| QAOA p=2 skip-7 | 3.560 → 3.559 | −0.0 % | 3.646 → 3.647 | +0.0 % |

The same Lookahead QFT plan's measured all-ranks compute, from every run on this branch and the PR 2/3 runs
(`dist_compile_bench`'s `measured all` L column is the same quantity, `T_full − T_comm` of the Lookahead plan):

| source | QFT D=2 (s) | QFT D=4 (s) | #538 model / it (D=2 / D=4) |
|---|---|---|---|
| PR 2 gate run 1 (§2.5) | 2.319 | – | – |
| PR 2 gate run 2 (§2.5) | 2.359 | 2.359 | – |
| PR 3 gate re-run, run 1 / run 2 (§3.3) | 2.393 / 2.389 | 2.389 / 2.387 | – |
| #538 gate run 1 | 2.323 | 2.325 | 1.144 / 1.147 |
| #538 gate run 2 | 2.440 | 2.433 | 1.089 / 1.096 |
| #538 compile bench run 1 (L column) | 2.449 | 2.444 | 1.085 / 1.091 |
| #538 compile bench run 2 (L column) | 2.445 | 2.438 | 1.087 / 1.094 |

### 4.3 Stage C: held-out cells (never used to choose anything)

Spec §4: HEA = `build_hea(28, 4, params)`, `params[i] = 0.1 + 0.07·i`; random d=20 = `brickwall_bench(28, 20)`;
Clifford brickwall d=10 = the state (f) construction at depth 10; QAOA p=2 skip-7 = ring `(i, i+1 mod n)` plus
`(i, i+7 mod n)`, γ = [0.4, 0.7], β = [0.3, 0.5].

Run 1:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | verdict |
|---|---|---|---|---|---|---|---|---|
| HEA d=4 | 2 | 8.705 | 8.918 | 1.024 | 8.918 | 1.024 | 10/10 | PASS |
| HEA d=4 | 4 | 8.703 | 8.973 | 1.031 | 8.973 | 1.031 | 10/10 | PASS |
| random d=20 | 2 | 13.968 | 13.581 | 0.972 | 13.581 | 0.972 | 21/21 | PASS |
| random d=20 | 4 | 16.300 | 15.924 | 0.977 | 16.310 | 1.001 | 23/23 | PASS |
| Clifford brickwall d=10 | 2 | 6.048 | 5.777 | 0.955 | 5.848 | 0.967 | 0/6 | PASS |
| Clifford brickwall d=10 | 4 | 7.244 | 6.984 | 0.964 | 7.126 | 0.984 | 0/7 | PASS |
| QAOA p=2 skip-7 | 2 | 3.560 | 3.543 | 0.995 | 3.662 | 1.029 | 7/8 | PASS |
| QAOA p=2 skip-7 | 4 | 3.646 | 3.643 | 0.999 | 3.744 | 1.027 | 7/8 | PASS |

Run 2:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | verdict |
|---|---|---|---|---|---|---|---|---|
| HEA d=4 | 2 | 8.755 | 8.918 | 1.019 | 8.918 | 1.019 | 10/10 | PASS |
| HEA d=4 | 4 | 8.735 | 8.973 | 1.027 | 8.973 | 1.027 | 10/10 | PASS |
| random d=20 | 2 | 14.012 | 13.581 | 0.969 | 13.581 | 0.969 | 21/21 | PASS |
| random d=20 | 4 | 16.312 | 15.924 | 0.976 | 16.310 | 1.000 | 23/23 | PASS |
| Clifford brickwall d=10 | 2 | 6.037 | 5.777 | 0.957 | 5.848 | 0.969 | 0/6 | PASS |
| Clifford brickwall d=10 | 4 | 7.238 | 6.984 | 0.965 | 7.126 | 0.984 | 0/7 | PASS |
| QAOA p=2 skip-7 | 2 | 3.559 | 3.543 | 0.996 | 3.662 | 1.029 | 7/8 | PASS |
| QAOA p=2 skip-7 | 4 | 3.647 | 3.643 | 0.999 | 3.744 | 1.027 | 7/8 | PASS |

Worst |model/measured − 1| on held-out cells: **4.5 %** (run 1), **4.3 %** (run 2), both Clifford brickwall D=2.
No held-out cell missed. Per-kind model (s; identical in both runs; `t_comm` run 1):

| circuit | D | dense1 | dense2 | dense3 | diag1 | cnot | phase | t_comm (s) |
|---|---|---|---|---|---|---|---|---|
| HEA d=4 | 2 | 4.703 | 0.045 | 4.169 | 0 | 0 | 0 | 0.257 |
| HEA d=4 | 4 | 4.422 | 0.499 | 4.009 | 0 | 0.043 | 0 | 0.379 |
| random d=20 | 2 | 1.650 | 10.344 | 1.523 | 0 | 0.065 | 0 | 0.551 |
| random d=20 | 4 | 3.896 | 10.434 | 1.443 | 0 | 0.151 | 0 | 0.898 |
| Clifford brickwall d=10 | 2 | 0.316 | 2.985 | 1.871 | 0.142 | 0.022 | 0.440 | 0.150 |
| Clifford brickwall d=10 | 4 | 0.720 | 2.575 | 2.218 | 0.657 | 0.022 | 0.793 | 0.258 |
| QAOA p=2 skip-7 | 2 | 2.808 | 0.264 | 0 | 0.071 | 0 | 0.400 | 0.203 |
| QAOA p=2 skip-7 | 4 | 2.597 | 0.491 | 0 | 0.107 | 0 | 0.448 | 0.298 |

`diag_k` is 0 on every held-out cell too, so the gate still never exercises `DiagK` (§2.5).

### 4.4 Stage C: compiled plans (spec exit 2) and compile bench (spec exit 3)

`cargo test --release -p aleph-cuda --features cuda --test dist_compile_bench -- --ignored --nocapture`, two runs
(raw logs `p6-05-state-class/stage-c-compile-run{1,2}.log`), same method and columns as §3. The log lines labelled
`exit3b` are #538's exit 2 (compiled-plan model/measured within ±10 %); the lines labelled `exit1` are #538's exit 3.
The labels keep P6-05's numbering. (The test source now prints `exit3b (compiled-plan model/measured; #538 exit 2)`; the committed logs carry the old
label `exit3b (plan-level compiled-plan check; spec exit 3 = dist_cost_gate)`.)

**FP64, run 1:**

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.532 | 1.524 | 1.362 | 0.894 | 0.894 | 1.081 | 1.1 | 2.465/2.449/2.424 |
| QFT | 4 | naive+place | 3/2/2 | 0.835 | 0.981 | 0.738 | 0.884 | 0.752 | 1.087 | 1.4 | 2.438/2.444/2.352 |
| GHZ | 2 | naive | 1/1/1 | 0.633 | 0.633 | 0.633 | 1.000 | 0.999 | 0.952 | 0.1 | 0.965/0.967/0.965 |
| GHZ | 4 | naive | 2/1/2 | 0.381 | 0.415 | 0.381 | 1.000 | 0.919 | 0.961 | 0.1 | 0.924/0.918/0.924 |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 7.451 | 5.028 | 2.959 | 0.588 | 0.588 | 1.019 | 1.2 | 8.903/7.056/5.617 |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 6.414 | 4.106 | 1.548 | 0.377 | 0.377 | 1.026 | 1.5 | 12.460/8.280/5.593 |
| QAOA p=2 | 2 | reorder k=1 | 9/5/2 | 3.004 | 2.399 | 1.917 | 0.799 | 0.799 | 0.999 | 0.4 | 3.308/3.298/3.234 |
| QAOA p=2 | 4 | reorder k=1 | 16/5/4 | 2.062 | 1.759 | 1.100 | 0.626 | 0.626 | 1.004 | 0.6 | 3.448/3.333/3.202 |
| CCZ ladder d=4 | 2 | naive | 1/1/1 | 0.680 | 0.680 | 0.680 | 1.000 | 1.000 | 1.008 | 0.5 | 1.060/1.061/1.060 |
| CCZ ladder d=4 | 4 | naive | 2/1/2 | 0.415 | 0.450 | 0.415 | 1.000 | 0.923 | 1.007 | 0.7 | 1.060/1.058/1.060 |
| Grover K=3 | 2 | reorder k=1 | 13/7/1 | 5.407 | 4.421 | 1.598 | 0.361 | 0.361 | 1.067 | 0.6 | 6.916/6.743/2.896 |
| Grover K=3 | 4 | reorder k=1 | 20/7/2 | 3.230 | 2.932 | 0.874 | 0.298 | 0.298 | 1.067 | 0.7 | 6.922/6.545/2.898 |

**FP64, run 2:**

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.528 | 1.522 | 1.358 | 0.892 | 0.892 | 1.084 | 1.1 | 2.455/2.445/2.416 |
| QFT | 4 | naive+place | 3/2/2 | 0.833 | 0.980 | 0.736 | 0.883 | 0.751 | 1.092 | 1.5 | 2.430/2.438/2.342 |
| GHZ | 2 | naive | 1/1/1 | 0.632 | 0.632 | 0.632 | 1.000 | 1.000 | 0.954 | 0.1 | 0.963/0.964/0.963 |
| GHZ | 4 | naive | 2/1/2 | 0.381 | 0.413 | 0.381 | 1.000 | 0.921 | 0.963 | 0.1 | 0.922/0.913/0.922 |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 7.660 | 5.128 | 2.950 | 0.575 | 0.575 | 1.022 | 1.2 | 9.321/7.257/5.600 |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 6.413 | 4.104 | 1.545 | 0.376 | 0.376 | 1.028 | 1.6 | 12.453/8.272/5.580 |
| QAOA p=2 | 2 | reorder k=1 | 9/5/2 | 3.003 | 2.400 | 1.918 | 0.799 | 0.799 | 0.999 | 0.4 | 3.307/3.301/3.235 |
| QAOA p=2 | 4 | reorder k=1 | 16/5/4 | 2.061 | 1.759 | 1.100 | 0.626 | 0.626 | 1.004 | 0.6 | 3.446/3.331/3.201 |
| CCZ ladder d=4 | 2 | naive | 1/1/1 | 0.681 | 0.680 | 0.681 | 1.000 | 1.000 | 1.007 | 0.5 | 1.062/1.061/1.062 |
| CCZ ladder d=4 | 4 | naive | 2/1/2 | 0.415 | 0.449 | 0.415 | 1.000 | 0.924 | 1.005 | 0.7 | 1.062/1.057/1.062 |
| Grover K=3 | 2 | reorder k=1 | 13/7/1 | 5.409 | 4.421 | 1.598 | 0.361 | 0.361 | 1.068 | 0.6 | 6.918/6.743/2.896 |
| Grover K=3 | 4 | reorder k=1 | 20/7/2 | 3.229 | 2.931 | 0.874 | 0.298 | 0.298 | 1.067 | 0.7 | 6.919/6.542/2.898 |

**The chosen candidate and the exchange counts are unchanged from §3.1 on every FP64 cell** (QFT naive+place, GHZ
and CCZ ladder naive, random/QAOA/Grover reorder k=1). The model `T` of each candidate moved, not the ranking
(deterministic, identical in both runs; chosen in bold, §3.1 value in brackets):

| circuit | D | naive | lookahead | reorder k=1 | reorder k=2 | naive+place | lookahead+place | reorder k=1+place | reorder k=2+place |
|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 1.635 [1.548] | 1.641 [1.557] | 1.641 [1.557] | – | **1.472** [1.385] | 1.472 [1.385] | 1.472 [1.385] | – |
| QFT | 4 | 0.892 [0.849] | 1.040 [1.001] | 0.895 [0.853] | 1.040 [1.001] | **0.792** [0.750] | 0.836 [0.794] | 0.792 [0.750] | 0.836 [0.794] |
| GHZ | 2 | **0.618** [0.614] | 0.618 [0.614] | 0.618 [0.614] | – | – | – | – | – |
| GHZ | 4 | **0.381** [0.379] | 0.408 [0.408] | 0.381 [0.379] | 0.408 [0.408] | – | – | – | – |
| random d=10 | 2 | 7.713 [7.648] | 5.046 [4.887] | **3.013** [2.656] | – | – | – | – | – |
| random d=10 | 4 | 6.517 [6.520] | 4.113 [4.034] | **1.584** [1.405] | 1.611 [1.431] | – | – | – | – |
| QAOA p=2 | 2 | 3.015 [3.000] | 2.425 [2.415] | **1.916** [1.901] | – | – | – | – | – |
| QAOA p=2 | 4 | 2.067 [2.059] | 1.770 [1.762] | **1.104** [1.095] | 1.172 [1.164] | – | – | – | – |
| CCZ ladder d=4 | 2 | **0.684** [0.681] | 0.684 [0.681] | 0.684 [0.681] | – | – | – | – | – |
| CCZ ladder d=4 | 4 | **0.417** [0.415] | 0.452 [0.450] | 0.417 [0.416] | 0.452 [0.451] | – | – | – | – |
| Grover K=3 | 2 | 5.496 [5.497] | 4.515 [4.534] | **1.696** [1.696] | – | – | – | – | – |
| Grover K=3 | 4 | 3.273 [3.273] | 2.979 [2.998] | **0.923** [0.923] | 0.958 [0.958] | – | – | – | – |

Compiled-plan model/measured (spec exit 2), against PR 3 (§3.1):

| circuit | PR 3 run 1 (D=2 / D=4) | PR 3 run 2 | #538 run 1 | #538 run 2 |
|---|---|---|---|---|
| QFT | 1.060 / 1.057 | 1.033 / 1.031 | 1.081 / 1.087 | 1.084 / 1.092 |
| GHZ | 0.992 / 0.994 | 0.957 / 0.962 | 0.952 / 0.961 | 0.954 / 0.963 |
| random d=10 | 0.912 / 0.900 | 0.899 / 0.898 | 1.019 / 1.026 | 1.022 / 1.028 |
| QAOA p=2 | 0.990 / 0.994 | 0.990 / 0.994 | 0.999 / 1.004 | 0.999 / 1.004 |
| CCZ ladder d=4 | 1.002 / 1.003 | 1.002 / 1.001 | 1.008 / 1.007 | 1.007 / 1.005 |
| Grover K=3 | 1.067 / 1.067 | 1.068 / 1.067 | 1.067 / 1.067 | 1.068 / 1.067 |
| FP32 QFT (not gated) | 1.006 / 1.007 | 1.006 / 1.007 | 1.049 / 1.052 | 1.051 / 1.052 |
| FP32 random d=10 (not gated) | 0.982 / 0.973 | 0.981 / 0.974 | 0.983 / 0.975 | 0.985 / 0.974 |

**FP32** (reported, not gated; run 2, run 1 in brackets where it differs):

| circuit | D | chosen | exch N/L/C | T_pred naive (s) | T_pred lookahead (s) | T_pred compiled (s) | compiled / min(N,L) | compiled / L | model/measured (C) | compile (ms) | measured all N/L/C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 2/2/1 | 1.168 [1.170] | 1.162 | 1.075 [1.077] | 0.926 [0.927] | 0.926 [0.927] | 1.051 [1.049] | 1.1 [1.0] | 2.033/2.021/1.999 [2.037/2.020/2.002] |
| QFT | 4 | naive+place | 3/2/2 | 0.621 | 0.684 | 0.561 | 0.904 | 0.821 [0.820] | 1.052 | 1.4 | 2.028/1.988/1.942 [2.029/1.990/1.943] |
| random d=10 | 2 | reorder k=1 | 20/10/1 | 3.669 [3.670] | 2.164 | 0.821 [0.823] | 0.379 [0.380] | 0.379 [0.380] | 0.985 [0.983] | 1.1 [1.2] | 4.309/2.814/1.491 [4.310/2.814/1.494] |
| random d=10 | 4 | reorder k=1 | 44/11/2 | 3.217 | 1.865 [1.864] | 0.453 [0.452] | 0.243 | 0.243 | 0.974 [0.975] | 1.6 | 6.205/3.345/1.509 [6.206/3.344/1.507] |

FP32 choices are unchanged from §3.2 (QFT naive+place, random reorder k=1). FP32 QFT model/measured rose from
1.006 / 1.007 (§3.2, run 2) to 1.049–1.052: QFT's phase launches are now priced at the FP32 generic `phase_base`
(49.59 vs 46.16 ms, §4.1). Model `T` per FP32 candidate is in the table below (deterministic, identical in both runs; from the `candidates:`
lines of the compile-run logs).

| circuit | D | naive | lookahead | reorder k=1 | reorder k=2 | naive+place | lookahead+place | reorder k=1+place | reorder k=2+place |
|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 1.223 | 1.227 | 1.227 | – | 1.143 | 1.143 | 1.143 | – |
| QFT | 4 | 0.649 | 0.715 | 0.651 | 0.715 | 0.591 | 0.613 | 0.591 | 0.613 |
| random d=10 | 2 | 3.772 | 2.176 | 0.810 | – | – | – | – | – |
| random d=10 | 4 | 3.346 | 1.892 | 0.443 | 0.456 | – | – | – | – |

Other bench lines (both runs):

| line | run 1 | run 2 | PR 3 (run 1 / run 2) |
|---|---|---|---|
| P6-05 exit 2: random d=10 D=2 compiled vs Lookahead `T_pred` | 41.2 % better (2.959 vs 5.028 s) | 42.5 % better (2.950 vs 5.128 s) | 42.7 % / 42.6 % |
| compile time, brickwall d=15 (1043 gates), g=2, best of 5 (target < 50 ms) | 2.3 ms | 2.3 ms | 2.2 ms / 2.2 ms |

### 4.5 Exit criteria and Reading

| spec §4 exit | run 1 | run 2 |
|---|---|---|
| 1. every old and held-out cell within ±10 % (`dist_cost_gate`) | **MISS**: QFT 1.144 / 1.147 (D=2 / D=4); every other old cell 1.003–1.031, every held-out cell 0.955–1.031 | **PASS**: worst QFT D=4 1.096 (9.6 %); held-out worst 4.3 % |
| 2. compiled-plan check (3b) within ±10 % on every FP64 cell | **PASS**: 0.952–1.087 | **PASS**: 0.954–1.092 |
| 3. `dist_compile_bench` exit 1, compiled ≤ min(Naive, Lookahead) + 3 % | **PASS**: worst 1.000 (GHZ, CCZ ladder), best 0.298 (Grover D=4) | **PASS**: worst 1.000, best 0.298 |

**Idle-box precondition not met.** Spec §4 requires an idle box. Neither Stage C gate run met it (1-min load before
each run, from `p6-05-state-class/stage-c-runner.out`):

| run | time | 1-min load | 5-min load | 15-min load |
|---|---|---|---|---|
| 1 | 12:25:57 | 1.57 | 0.88 | 0.85 |
| 2 | 12:41:10 | 2.36 | 1.82 | 1.39 |

The load was residual from this session's own builds and runs (no foreign workload), and the GPU was idle. The runs
were not repeated. The verdict below is reported as measured and is not softened.

**Spec exit 1 is MISSED** (it needs every cell in both runs; run 1's QFT cells are 14.4 % and 14.7 % high). Exits 2
and 3 pass in both runs, and all eight held-out cells pass in both runs. Per spec §4 the rule and the constants are
**not re-tuned in this PR; the next step is the user's call.**

**Reading**

- **The state-class model fixed the cell it was built for.** The compiled random d=10 plan, PR 3's borderline 3b
  miss (0.899 / 0.898 in PR 3 run 2), now reads 1.019–1.028 in both runs. On the Lookahead gate, random d=10 moved
  from 0.944–0.951 (PR 3 re-run) to 0.976–1.026. Its model rose from 6.775 to 7.092 s (D=2): every step is generic
  (11/11), so `dense2` (4.900 s) and `dense3` (0.722 s) are priced at their generic values.
- **Held-out cells: all within 4.5 %.** The generic-dominated ones (HEA 10/10, random d=20 21/21 and 23/23 generic
  steps) land at 0.969–1.031; QAOA skip-7 (7/8) at 0.995–0.999. The Clifford brickwall is priced all simple (0/6,
  0/7) and is the most under-predicted held-out cell, 0.955–0.965.
- **Cells the walk leaves simple barely move.** Grover (0/8) gets slightly cheaper (6.969 → 6.930 s at D=2) because
  the simple `dense2` is now the uniform-state value; its ratio goes 1.033 → 1.028–1.029. GHZ (0/2) spans 0.956–1.027
  against 0.968–1.025 before (its model rose 0.911 → 0.919 s at D=2, via `dense3`
  re-measured on state (a)). QAOA (5/6) and the CCZ ladder (1/2) stay within 1.6 %.
- **QFT is the cell the model made clearly worse**: 2.492 → 2.657 s model at D=2 (2/3 generic steps), so its ratio rose
  from 1.041–1.081 (PR 2/3) to 1.089–1.147. **Diagnosis: a blind spot of the gate-based rule** (the same kind of
  error spec §6 accepts under "Un-scrambling", but here the state never leaves simple at all). QFT's controlled-phase
  angles π/2^k (k ≥ 3) are not multiples of π/4, so R2 marks every step holding them generic and prices
  `phase_base` at 24.85 ms instead of 21.35 ms (§4.1). But the gate runs QFT from |0…0⟩, and `qft()`
  in `tests/common/dist.rs` applies the H on qubit j just before the controlled phases between j and every k < j, so
  qubit k is still |0⟩ when each phase acts. A controlled phase only touches amplitudes with both qubits at 1, so
  every phase acts as the identity: the state is a product of |+⟩ and |0⟩ factors throughout, i.e. state (a)'s class.
  The rule is judged on gates, not amplitudes, so it cannot see this.
- **Diagnostic counterfactual (not a re-tune):** price QFT's phase launches at the simple `phase_base`, leaving the
  per-term costs (which did not split) and every other kind as they are. The all-ranks phase launch counts are the
  ones counted in §3.4 (QFT lookahead and compiled plans: 51 at D=2, 99 at D=4). Each launch at m_ref = 27 is
  re-priced from the generic 24.850 ms to the simple 21.346 ms (`cost.rs`, §4.1), saving 3.504 ms; at D=4 each rank
  holds m = 26 and a launch costs half as much, so it saves 1.752 ms.

  | D | phase launches (§3.4) | saving per launch (ms) | #538 phase (s) | phase at simple base (s) | model (s) | counterfactual model (s) | Δ (s) | run 1 ratio | run 1 counterfactual | run 2 ratio | run 2 counterfactual |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 2 | 51 | 3.504 | 1.619 | 1.440 | 2.657 | 2.478 | 0.179 | 1.144 | 1.067 | 1.089 | 1.016 |
  | 4 | 99 | 1.752 | 1.565 | 1.392 | 2.667 | 2.494 | 0.173 | 1.147 | 1.072 | 1.096 | 1.025 |

  The counterfactual puts both runs inside ±10 % and back near the PR 2 range (run 1 1.074 / 1.081, §2.5). It
  explains 0.173–0.179 s of the over-prediction, not all of run 1's: run 1 is still 1.067–1.072 against run 2's
  1.016–1.025, because run 1 also measured QFT 4.6–5.0 % faster.
- **Run 1 measured the first cells fast.** QFT, GHZ and random d=10 (the first three cells) measured 3.2–6.9 % less
  compute in run 1 than in run 2; every later cell agrees within 1.0 %. Run 1's QFT (2.323 / 2.325 s) is the low
  outlier of the four measurements of the same Lookahead plan on this branch: both runs of the compile bench measured
  2.438–2.449 s, close to gate run 2. PR 2's run 1 showed the same first-run pattern (§2.5, QFT 2.319 vs 2.359 s).
  Run 1 started with a residual 1-min load of 1.57 (§4.2); that is noted, not shown to be the cause. Even at run 2's (or the compile bench's) QFT timings the cell sits at 1.085–1.096, near the bound.
- **The compiled plans did not change.** Every FP64 and FP32 chosen candidate and exchange count matches §3.1 / §3.2;
  the generic pricing raised each candidate's model `T` (random reorder k=1 2.656 → 3.013 at D=2) without changing
  any ranking. Compiled QFT's 3b ratio rose 1.031–1.060 → 1.081–1.092 for the same phase-pricing reason; FP32 QFT
  (1.006 → 1.049–1.052) shows it too.
- **What exit 1 needs is the user's decision**, not a change made here: e.g. a state-aware treatment of diagonal
  phases that act on |0⟩ qubits (an amplitude-free "touched-qubit" walk), or accepting QFT as a documented
  exception. Neither is applied in this PR, and either would have to be re-validated out of sample.

### 4.6 Zero-tracking walk (QFT follow-up)

Spec: `docs/superpowers/specs/2026-10-05-p6-05-zero-tracking-design.md` (commit `b9bd87d`). The code (commits
`8b23b25`, `9abbc76`) was pushed before any run. Stage A constants and `STATE_RULE = R2` are unchanged.

**The rule.** `state_classes` now tracks **Z0**, the physical qubits known to be |0⟩. It starts as all qubits. A gate
acts trivially on the state, and leaves the class alone, in two cases:

- one of its external controls is in Z0;
- its target matrix, restricted to the input columns that the Z0 qubits allow, is the identity (tolerance 1e-9).

Otherwise R2 is judged on the restricted columns only. Targets leave Z0 when the restricted output can set their bit.
A `Swap` moves a |0⟩ to the other qubit. A `DiagonalPhase` term whose cond mask lies inside Z0 is dead. An `Exchange`
swaps the Z0 membership of the bits it pairs.

`makes_generic` (no qubit assumed |0⟩) is unchanged. It drives the "old-rule generic" column below.

**Runs.** `dist_cost_gate` ×2 and `dist_compile_bench` ×2 on the RTX 4000 SFF Ada, 2026-10-05, in one script
(raw logs `p6-05-state-class/zt-{gate,compile}-run{1,2}.log`, runner output `zt-runner.out`). Before each run the
script waited until the 1-minute load was below 0.1 and GPU utilisation was 0 %, then logged:

| run | UTC start | 1-min / 5-min / 15-min load | GPU util | GPU mem resident | power (W) |
|---|---|---|---|---|---|
| gate 1 | 12:43:17 | 0.09 / 0.34 / 0.25 | 0 % | 6600 MiB | 23.67 |
| compile 1 | 12:56:08 | 0.09 / 0.63 / 0.68 | 0 % | 13828 MiB | 6.58 |
| gate 2 | 13:07:09 | 0.09 / 0.58 / 0.72 | 0 % | 13828 MiB | 6.67 |
| compile 2 | 13:18:37 | 0.06 / 0.45 / 0.69 | 0 % | 13828 MiB | 6.57 |

The resident memory belongs to services that share the card (an Ollama LLM server, a text-embeddings server).
The check runs only at the start of each run. It cannot see work that these services, or another project's CI runner
on the same host, start during a run (see "Gate run 2's HEA D=2 cell" below).

**Classes, expected vs measured.** "generic steps" is the new walk; "old-rule" is the pre-change walk on the same
plan. Each column is identical in both runs; the new walk differs from the old only on QFT, H2 and H3.

| cell | expected (spec §3/§4) | generic steps, new | old-rule generic |
|---|---|---|---|
| QFT (in-sample) | all simple | 0/3 | 2/3 |
| GHZ, random d=10, QAOA p=2, CCZ ladder, Grover | unchanged | 0/2, 11/11 · 12/12, 5/6, 1/2, 0/8 | same |
| HEA, random d=20, Clifford brickwall, QAOA skip-7 | unchanged | 10/10, 21/21 · 23/23, 0/6 · 0/7, 7/8 | same |
| H1 QFT on X-odd input | generic | 2/3 | 2/3 |
| H2 \|0⟩-controlled phase ladder | simple | 0/2 | 1/2 |
| H3 \|0⟩-controlled CRx | simple | 0/2 | 1/2 |

Every class came out as the spec stated before the runs. The only cells the change re-prices are QFT, H2 and H3.

**Gate run 1:**

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | old-rule generic | verdict |
|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 2.328 | 2.478 | 1.064 | 2.500 | 1.074 | 0/3 | 2/3 | PASS |
| QFT | 4 | 2.326 | 2.493 | 1.072 | 2.504 | 1.076 | 0/3 | 2/3 | PASS |
| GHZ | 2 | 0.901 | 0.919 | 1.020 | 0.936 | 1.039 | 0/2 | 0/2 | PASS |
| GHZ | 4 | 0.860 | 0.883 | 1.026 | 0.891 | 1.035 | 0/2 | 0/2 | PASS |
| random d=10 | 2 | 6.914 | 7.092 | 1.026 | 7.092 | 1.026 | 11/11 | 11/11 | PASS |
| random d=10 | 4 | 8.011 | 8.129 | 1.015 | 8.305 | 1.037 | 12/12 | 12/12 | PASS |
| QAOA p=2 | 2 | 3.272 | 3.282 | 1.003 | 3.350 | 1.024 | 5/6 | 5/6 | PASS |
| QAOA p=2 | 4 | 3.300 | 3.326 | 1.008 | 3.377 | 1.023 | 5/6 | 5/6 | PASS |
| CCZ ladder d=4 | 2 | 1.056 | 1.069 | 1.012 | 1.069 | 1.012 | 1/2 | 1/2 | PASS |
| CCZ ladder d=4 | 4 | 1.054 | 1.066 | 1.011 | 1.066 | 1.011 | 1/2 | 1/2 | PASS |
| Grover K=3 | 2 | 6.734 | 6.930 | 1.029 | 6.930 | 1.029 | 0/8 | 0/8 | PASS |
| Grover K=3 | 4 | 6.530 | 6.733 | 1.031 | 6.733 | 1.031 | 0/8 | 0/8 | PASS |
| HEA d=4 | 2 | 8.702 | 8.918 | 1.025 | 8.918 | 1.025 | 10/10 | 10/10 | PASS |
| HEA d=4 | 4 | 8.554 | 8.973 | 1.049 | 8.973 | 1.049 | 10/10 | 10/10 | PASS |
| random d=20 | 2 | 13.997 | 13.581 | 0.970 | 13.581 | 0.970 | 21/21 | 21/21 | PASS |
| random d=20 | 4 | 16.334 | 15.924 | 0.975 | 16.310 | 0.999 | 23/23 | 23/23 | PASS |
| Clifford brickwall d=10 | 2 | 6.058 | 5.777 | 0.954 | 5.848 | 0.965 | 0/6 | 0/6 | PASS |
| Clifford brickwall d=10 | 4 | 7.258 | 6.984 | 0.962 | 7.126 | 0.982 | 0/7 | 0/7 | PASS |
| QAOA p=2 skip-7 | 2 | 3.562 | 3.543 | 0.995 | 3.662 | 1.028 | 7/8 | 7/8 | PASS |
| QAOA p=2 skip-7 | 4 | 3.648 | 3.643 | 0.999 | 3.744 | 1.026 | 7/8 | 7/8 | PASS |
| H1 QFT on X-odd input | 2 | 2.908 | 3.113 | 1.071 | 3.138 | 1.079 | 2/3 | 2/3 | PASS |
| H1 QFT on X-odd input | 4 | 2.834 | 3.053 | 1.077 | 3.065 | 1.082 | 2/3 | 2/3 | PASS |
| H2 \|0⟩-controlled phase ladder | 2 | 0.763 | 0.729 | 0.955 | 0.740 | 0.969 | 0/2 | 1/2 | PASS |
| H2 \|0⟩-controlled phase ladder | 4 | 0.744 | 0.720 | 0.967 | 0.727 | 0.977 | 0/2 | 1/2 | PASS |
| H3 \|0⟩-controlled CRx | 2 | 1.616 | 1.519 | 0.940 | 1.519 | 0.940 | 0/2 | 1/2 | PASS |
| H3 \|0⟩-controlled CRx | 4 | 1.906 | 1.817 | 0.954 | 1.817 | 0.953 | 0/2 | 1/2 | PASS |

Worst |model/measured − 1| for run 1, by table:

- old cells: 7.2 % (QFT D=4);
- held-out §4.3 cells: 4.9 % (HEA D=4);
- H1–H3: 7.7 % (H1 D=4).

The gate test passed.

**Gate run 2:**

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio | generic steps | old-rule generic | verdict |
|---|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 2.341 | 2.478 | 1.059 | 2.500 | 1.068 | 0/3 | 2/3 | PASS |
| QFT | 4 | 2.340 | 2.493 | 1.065 | 2.504 | 1.070 | 0/3 | 2/3 | PASS |
| GHZ | 2 | 0.916 | 0.919 | 1.003 | 0.936 | 1.022 | 0/2 | 0/2 | PASS |
| GHZ | 4 | 0.871 | 0.883 | 1.013 | 0.891 | 1.022 | 0/2 | 0/2 | PASS |
| random d=10 | 2 | 7.003 | 7.092 | 1.013 | 7.092 | 1.013 | 11/11 | 11/11 | PASS |
| random d=10 | 4 | 8.104 | 8.129 | 1.003 | 8.305 | 1.025 | 12/12 | 12/12 | PASS |
| QAOA p=2 | 2 | 3.283 | 3.282 | 1.000 | 3.350 | 1.020 | 5/6 | 5/6 | PASS |
| QAOA p=2 | 4 | 3.314 | 3.326 | 1.004 | 3.377 | 1.019 | 5/6 | 5/6 | PASS |
| CCZ ladder d=4 | 2 | 1.059 | 1.069 | 1.009 | 1.069 | 1.009 | 1/2 | 1/2 | PASS |
| CCZ ladder d=4 | 4 | 1.056 | 1.066 | 1.009 | 1.066 | 1.009 | 1/2 | 1/2 | PASS |
| Grover K=3 | 2 | 6.739 | 6.930 | 1.028 | 6.930 | 1.028 | 0/8 | 0/8 | PASS |
| Grover K=3 | 4 | 6.538 | 6.733 | 1.030 | 6.733 | 1.030 | 0/8 | 0/8 | PASS |
| HEA d=4 | 2 | **13.008** | 8.918 | **0.686** | 8.918 | 0.686 | 10/10 | 10/10 | **MISS** |
| HEA d=4 | 4 | 8.736 | 8.973 | 1.027 | 8.973 | 1.027 | 10/10 | 10/10 | PASS |
| random d=20 | 2 | 14.064 | 13.581 | 0.966 | 13.581 | 0.966 | 21/21 | 21/21 | PASS |
| random d=20 | 4 | 16.358 | 15.924 | 0.973 | 16.310 | 0.997 | 23/23 | 23/23 | PASS |
| Clifford brickwall d=10 | 2 | 6.063 | 5.777 | 0.953 | 5.848 | 0.965 | 0/6 | 0/6 | PASS |
| Clifford brickwall d=10 | 4 | 7.261 | 6.984 | 0.962 | 7.126 | 0.981 | 0/7 | 0/7 | PASS |
| QAOA p=2 skip-7 | 2 | 3.559 | 3.543 | 0.995 | 3.662 | 1.029 | 7/8 | 7/8 | PASS |
| QAOA p=2 skip-7 | 4 | 3.647 | 3.643 | 0.999 | 3.744 | 1.027 | 7/8 | 7/8 | PASS |
| H1 QFT on X-odd input | 2 | 2.913 | 3.113 | 1.069 | 3.138 | 1.077 | 2/3 | 2/3 | PASS |
| H1 QFT on X-odd input | 4 | 2.830 | 3.053 | 1.079 | 3.065 | 1.083 | 2/3 | 2/3 | PASS |
| H2 \|0⟩-controlled phase ladder | 2 | 0.756 | 0.729 | 0.965 | 0.740 | 0.979 | 0/2 | 1/2 | PASS |
| H2 \|0⟩-controlled phase ladder | 4 | 0.743 | 0.720 | 0.969 | 0.727 | 0.979 | 0/2 | 1/2 | PASS |
| H3 \|0⟩-controlled CRx | 2 | 1.615 | 1.519 | 0.940 | 1.519 | 0.940 | 0/2 | 1/2 | PASS |
| H3 \|0⟩-controlled CRx | 4 | 1.903 | 1.817 | 0.955 | 1.817 | 0.955 | 0/2 | 1/2 | PASS |

Worst |model/measured − 1| for run 2, by table:

- old cells: 6.5 % (QFT D=4);
- held-out §4.3 cells: **31.4 %** (HEA D=2);
- H1–H3: 7.9 % (H1 D=4).

**The gate test FAILED.** Its assert covers all three tables.

**Gate run 2's HEA D=2 cell.** The cell measured 13.008 s. The other measurements of the same Lookahead plan read
8.702–8.755 s: Stage C runs 1–2 (§4.3) and zero-tracking gate run 1. The cell's model is identical under both rules
(10/10 generic, 8.918 s), so the zero-tracking change cannot move this ratio. The gate takes best-of-3, so all three
repetitions of this cell were slow.

The Ollama container's log lists every `/api/chat` request on the shared GPU between 12:40 and 13:30 UTC. The raw
source is `p6-05-state-class/zt-ollama-requests.log` (the container's `[GIN]` request log for 12:40–13:30 UTC; its
other lines are short `GET` calls). Run windows
are taken from each idle-check time plus the test's wall time.

| request window (UTC) | duration (s) | falls in |
|---|---|---|
| 12:46:09–12:46:13 | 3.3 | gate run 1 (12:43:17–12:50:07) |
| 12:46:23–12:46:58 | 34.8 (includes a model load) | gate run 1 |
| 12:55:06 | 0.8 | between runs |
| 12:56:29–12:56:32 | 3.1 | compile run 1 (12:56:08–13:04:39), start |
| 13:09:02 | 0.8 | gate run 2 (13:07:09–13:14:06) |
| 13:09:33–13:09:48 | 14.6 | gate run 2 |
| 13:09:33–13:10:14 | 41.0 | gate run 2 |

Compile run 2 (13:18:37–13:27:08) saw none. Gate run 2's 41 s request is the longest generation with the model
already loaded. Gate run 1's 34.8 s request includes the model load, and no gate run 1 cell stands out.

The gate does not timestamp its cells. Pro-rating run 2's wall time by measured compute puts the first held-out
cell (HEA D=2) at about 13:09:40–13:10:30 UTC. That would overlap the 41 s request. This is an estimate from timing,
not a proof.

**Compile bench** (FP64; deterministic model, so the chosen candidate and exchange counts are the same in both runs):

| circuit | D | chosen | compiled / min(N,L), run 1 | run 2 | 3b model/measured (C), run 1 | run 2 | §4.4 3b (runs 1/2) |
|---|---|---|---|---|---|---|---|
| QFT | 2 | naive+place | 0.886 | 0.892 | 1.062 | 1.054 | 1.081 / 1.084 |
| QFT | 4 | naive+place | 0.885 | 0.884 | 1.048 | 1.051 | 1.087 / 1.092 |
| GHZ | 2 | naive | 1.000 | 1.000 | 0.993 | 0.998 | 0.952 / 0.954 |
| GHZ | 4 | naive | 1.000 | 1.000 | 0.998 | 0.998 | 0.961 / 0.963 |
| random d=10 | 2 | reorder k=1 | 0.573 | 0.573 | 1.037 | 1.039 | 1.019 / 1.022 |
| random d=10 | 4 | reorder k=1 | 0.377 | 0.376 | 1.029 | 1.029 | 1.026 / 1.028 |
| QAOA p=2 | 2 | reorder k=1 | 0.799 | 0.799 | 0.999 | 0.999 | 0.999 / 0.999 |
| QAOA p=2 | 4 | reorder k=1 | 0.626 | 0.626 | 1.004 | 1.004 | 1.004 / 1.004 |
| CCZ ladder d=4 | 2 | naive | 1.000 | 1.001 | 1.008 | 1.008 | 1.008 / 1.007 |
| CCZ ladder d=4 | 4 | naive | 1.000 | 1.000 | 1.008 | 1.008 | 1.007 / 1.005 |
| Grover K=3 | 2 | reorder k=1 | 0.361 | 0.362 | 1.067 | 1.067 | 1.067 / 1.068 |
| Grover K=3 | 4 | reorder k=1 | 0.298 | 0.298 | 1.067 | 1.066 | 1.067 / 1.067 |

Every FP64 cell's chosen candidate and exchange counts match §4.4 and §3.1. Only QFT's candidate model `T` moved:

| QFT candidate | D=2 §4.4 | D=2 now | D=4 §4.4 | D=4 now |
|---|---|---|---|---|
| naive | 1.635 | 1.544 | 0.892 | 0.847 |
| lookahead | 1.641 | 1.550 | 1.040 | 0.996 |
| naive+place (chosen) | 1.472 | 1.381 | 0.792 | 0.748 |

FP32 compiled QFT 3b reads 1.003 / 1.006 (run 1, D=2 / D=4) and 1.006 / 1.006 (run 2), against 1.049–1.052 in §4.4.
FP32 random d=10 reads 0.971–0.983.

Both compile-bench runs passed: every `exit1`, every `exit3b`, the random d=10 `exit2`, and the compile-time check
(2.3 ms).

**Exit criteria (zero-tracking spec §4)**

| exit | run 1 | run 2 |
|---|---|---|
| 1. H1–H3 within ±10 % | **PASS**: 0.940–1.077 | **PASS**: 0.940–1.079 |
| 2. all ten old cells within ±10 % (§4.2 six cells + §4.3 four held-out cells) | **PASS**: 0.954–1.072 | **MISS**: HEA D=2 0.686 (31.4 %); every other cell 0.953–1.065 |
| 3. compile bench exit1 and 3b on every FP64 cell | **PASS**: exit1 ≤ 1.000; 3b 0.993–1.067 | **PASS**: exit1 ≤ 1.001; 3b 0.998–1.067 |
| 4. Stage A constants and `STATE_RULE` unchanged | **PASS** | **PASS** |

Exits 1, 3 and 4 pass in both runs. Exit 2 passes in run 1 and **MISSES in run 2** on one cell, HEA D=2 (0.686).

The #538 exit 1 (§4.5, old + held-out cells) has the same verdict:

- run 1 **PASSES**, every cell within 7.2 %;
- run 2 is a **MISS** on the same single cell, HEA D=2 at 0.686.

The gate test asserts over all three tables, so it reported run 2 as FAILED. Per spec, nothing is re-tuned and no run
is repeated here. Whether to re-run with the shared GPU services paused is the user's call.

**Reading**

- **The QFT blind spot is closed.** QFT now walks all simple (0/3, against 2/3 before), as spec §3 predicted. Its
  model drops 2.657 → 2.478 s at D=2 and 2.667 → 2.493 s at D=4, which matches the §4.5
  counterfactual (2.478 / 2.494) to within 0.001 s. Its ratio is 1.059–1.072 in both runs, against 1.089–1.147 in Stage C. QFT is **in-sample**: the
  rule was designed after the Stage C QFT miss.
- **The out-of-sample cells behave as stated in advance.**
  - H1 (X on odd qubits, then QFT): the controlled phases really fire, and the class stays generic. The model is not
    under-pricing it: 1.069–1.079 in both runs.
  - H2 and H3: the old rule called one of two steps generic; the new walk calls both simple. They read 0.940–0.969.
  - H3, the |0⟩-controlled CRx, is the most under-predicted cell at 0.940 in both runs. It is still inside the bound,
    and it reads in the same direction as the all-simple Clifford brickwall (0.953–0.962).
- **Nothing else moved.** Every other cell's class and model value is bit-identical to §4.2–4.3. Their ratios
  (1.000–1.031 old, 0.953–1.049 held-out, excluding run 2's HEA D=2) are within run-to-run noise of Stage C.
- **The compiler's choices did not change.** Compiled QFT's 3b improved from 1.081–1.092 to 1.048–1.062 (FP32:
  1.049–1.052 to 1.003–1.006).
- **The idle-wait precondition is necessary but not sufficient on this box.** The card is shared with an LLM server
  and an embeddings server. A check at the start of each run cannot prevent a request that arrives mid-run. Gate
  run 2's one MISS coincides in time with such a request (estimated, see above). A clean run would need those services
  paused, or a per-cell GPU-utilisation log.

### 4.7 Gate re-run with the shared GPU services paused

The user authorised a re-run of `dist_cost_gate` (×2, code `c33d7dd`) after §4.6's run-2 MISS, with the box's GPU
services out of the way. Raw logs: `p6-05-state-class/zt-gate-rerun{1,2}.log` and `zt-rerun-runner.out`.
Container events are in `zt-rerun-docker-events.log` and the watchdog journal is in `zt-rerun-watchdog.log`.
Nothing in the model changed.

**What actually happened to the services.**

- 13:45:14Z: the script ran `docker pause ollama neatkept-embed`. A `trap` would unpause them on any exit.
- The box also runs `neatkept-watchdog.timer`. Every 2 min it runs `docker exec <c> nvidia-smi -L` and probes an
  embedding. A paused container fails that check, so the watchdog restarted both services:
  - `neatkept-embed` at 13:46:49Z and again at 13:51:34Z;
  - `ollama` at 13:47:00Z (its log line: "restarted ollama: it had lost the GPU").

So the services were **running, not paused**, for almost all of both re-runs. The idle-check label in both logs
still says "paused"; the container-state line under it shows the truth ("paused" before re-run 1, "running" before
re-run 2). Re-run 1 also contains the three restarts.

What the services did while running:

- Ollama logged **no** `/api/chat` request between 13:44 and 14:02Z, and its model stayed unloaded after the restart.
- The embeddings server served only the watchdog's ~10 ms probes, one every ~2 min.

Re-run 1 started at 13:46:14Z, re-run 2 at 13:55:26Z. Both started at a 1-minute load of 0.08, GPU util 0 %.

**Results** (ratio = model all-ranks / measured compute; class columns identical to §4.6):

| circuit | D | measured (s), rerun 1 | ratio, rerun 1 | measured (s), rerun 2 | ratio, rerun 2 |
|---|---|---|---|---|---|
| QFT | 2 | 2.329 | 1.064 | 2.343 | 1.057 |
| QFT | 4 | 2.327 | 1.071 | 2.342 | 1.065 |
| GHZ | 2 | 0.904 | 1.016 | 0.914 | 1.005 |
| GHZ | 4 | 0.858 | 1.029 | 0.873 | 1.011 |
| random d=10 | 2 | 6.936 | 1.023 | 6.995 | 1.014 |
| random d=10 | 4 | 8.064 | 1.008 | 8.107 | 1.003 |
| QAOA p=2 | 2 | 3.275 | 1.002 | 3.284 | 0.999 |
| QAOA p=2 | 4 | 3.304 | 1.007 | 3.318 | 1.002 |
| CCZ ladder d=4 | 2 | 1.054 | 1.014 | 1.058 | 1.010 |
| CCZ ladder d=4 | 4 | 1.055 | 1.010 | 1.056 | 1.009 |
| Grover K=3 | 2 | 6.736 | 1.029 | 6.738 | 1.029 |
| Grover K=3 | 4 | 6.532 | 1.031 | 6.537 | 1.030 |
| HEA d=4 | 2 | 8.716 | 1.023 | 8.745 | 1.020 |
| HEA d=4 | 4 | 8.727 | 1.028 | 8.732 | 1.028 |
| random d=20 | 2 | 14.026 | 0.968 | 14.054 | 0.966 |
| random d=20 | 4 | 16.354 | 0.974 | 16.348 | 0.974 |
| Clifford brickwall d=10 | 2 | 6.070 | 0.952 | 6.065 | 0.953 |
| Clifford brickwall d=10 | 4 | 7.267 | 0.961 | 7.262 | 0.962 |
| QAOA p=2 skip-7 | 2 | 3.563 | 0.994 | 3.564 | 0.994 |
| QAOA p=2 skip-7 | 4 | 3.652 | 0.998 | 3.652 | 0.998 |
| H1 QFT on X-odd input | 2 | 2.916 | 1.068 | 2.917 | 1.067 |
| H1 QFT on X-odd input | 4 | 2.832 | 1.078 | 2.836 | 1.076 |
| H2 \|0⟩-controlled phase ladder | 2 | 0.761 | 0.957 | 0.762 | 0.957 |
| H2 \|0⟩-controlled phase ladder | 4 | 0.744 | 0.968 | 0.736 | 0.978 |
| H3 \|0⟩-controlled CRx | 2 | 1.616 | 0.940 | 1.618 | 0.939 |
| H3 \|0⟩-controlled CRx | 4 | 1.905 | 0.954 | 1.903 | 0.955 |

Worst |model/measured − 1| by table:

| table | rerun 1 | rerun 2 |
|---|---|---|
| old | 7.1 % | 6.5 % |
| held-out §4.3 | 4.8 % | 4.7 % |
| zero-tracking held-out (H1–H3) | 7.8 % | 7.6 % |

**Both gate tests passed.** HEA D=2 measured 8.716 / 8.745 s (ratio 1.023 / 1.020), in line with the 8.70–8.76 s of
every run except §4.6's run 2.

**Exit criteria on the re-run pair:**

| exit | rerun 1 | rerun 2 |
|---|---|---|
| 1. H1–H3 within ±10 % | PASS (0.940–1.078) | PASS (0.939–1.076) |
| 2. ten old cells within ±10 % | PASS (0.952–1.071) | PASS (0.953–1.065) |
| 3. compile bench | not re-run (passed both §4.6 runs) | not re-run |
| 4. constants and rule frozen | PASS | PASS |

The original pair (§4.6) still stands as recorded, with exit 2 a MISS in run 2. The re-run pair passes every gate
exit. This is consistent with the run-2 HEA cell being an external disturbance rather than a model error, but
the re-run was made after seeing the MISS, so it is supplementary evidence, not a replacement.

The re-runs did not get a GPU that was free in practice either. A truly isolated run on this box would also need
`neatkept-watchdog.timer` stopped for its duration.
