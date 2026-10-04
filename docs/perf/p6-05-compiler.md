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

Hardware: RTX 4000 SFF Ada (sm_89, 20 GiB), one card. Date: 2026-10-04.

### 2.1 Calibration

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_calibrate -- --ignored --nocapture`

Each kind is timed as the best of 5 over 32 launches on a `2^m_ref` state, with an H-layer baseline subtracted. The
values are seconds per launch at `m_ref` and scale by `2^(m − m_ref)`. They are committed in
`crates/aleph-cuda/src/dist/cost.rs` (run 2 of two calibration runs).

| kind | kernel | FP64 (m_ref = 27) | FP32 (m_ref = 28) |
|---|---|---|---|
| `dense1` | `apply_1q` | 1.756583e-2 | 1.762976e-2 |
| `dense2` | `apply_kq_tiled` k=2 | 2.113696e-2 | 1.767234e-2 |
| `dense3` | `apply_kq_tiled` k=3 | 3.630713e-2 | 1.715475e-2 |
| `diag1` | `apply_diag_1q` | 1.773637e-2 | 1.764512e-2 |
| `diag_k` | `apply_diag` k=2/3 | 1.772850e-2 | 1.760776e-2 |
| `cnot` | `apply_cnot` | 1.071684e-2 | 1.297517e-2 |
| `phase_base` | `apply_phase_poly` (per launch) | 2.331010e-2 | 4.622262e-2 |
| `phase_term` | `apply_phase_poly` (per term) | 6.563838e-4 | 1.289533e-3 |

Notes:
- Run-to-run spread between the two calibration runs exceeded 5 % on three FP64 kinds: `dense2` +8.4 %, `dense3`
  +5.1 %, `phase_base` +5.2 % (FP32 `phase_base` +1.9 %).
- `phase_base` includes the per-launch device allocation and upload of the phase terms.
- `dense1`, `diag1` and `diag_k` are within ~1 % of each other (the single-pass bandwidth floor), so their ordering is
  noise.

### 2.2 Model accuracy gate (spec §6.3)

`cargo test --release -p aleph-cuda --features cuda --test dist_cost_gate -- --ignored --nocapture`

n=28, FP64, `Router::Lookahead`, all D ranks on one card with `LocalExchange`, best of 3.
- **measured compute** = `T(plan) − T(exchange-only plan)`, where the exchange-only plan is the same plan with every
  `Local` step emptied.
- **model all-ranks** = `Σ_steps Σ_r rank_segment(step, r)` (the gated quantity).
- **R·model(R−1)** = the representative-rank estimate `compile` will use (reported, not gated).

Two runs on an idle box (load ≤ 0.71 and decaying, GPU util 0 %, 1339 MiB resident before each). Run 2:

| circuit | D | measured compute (s) | model all-ranks (s) | ratio | R·model(R−1) (s) | ratio |
|---|---|---|---|---|---|---|
| QFT | 2 | 2.353 | 2.710 | 1.152 | 2.733 | 1.161 |
| QFT | 4 | 2.352 | 2.717 | 1.156 | 2.729 | 1.160 |
| GHZ | 2 | 0.920 | 0.962 | 1.045 | 0.979 | 1.064 |
| GHZ | 4 | 0.878 | 0.925 | 1.053 | 0.935 | 1.065 |
| random d=10 | 2 | 7.029 | 6.690 | 0.952 | 6.690 | 0.952 |
| random d=10 | 4 | 8.130 | 7.731 | 0.951 | 7.907 | 0.973 |
| QAOA p=2 | 2 | 3.288 | 3.275 | 0.996 | 3.343 | 1.017 |
| QAOA p=2 | 4 | 3.320 | 3.301 | 0.994 | 3.352 | 1.010 |
| CCZ ladder d=4 | 2 | 1.060 | 1.066 | 1.005 | 1.066 | 1.005 |
| CCZ ladder d=4 | 4 | 1.057 | 1.063 | 1.006 | 1.063 | 1.006 |

Worst |model/measured − 1| = **15.6 %** (run 2), **16.6 %** (run 1). Run 1's all-ranks ratios: QFT 1.164 / 1.166,
GHZ 1.059 / 1.071, random 0.958 / 0.960, QAOA 0.999 / 0.999, CCZ 1.012 / 1.008 (D=2 / D=4).

**Gate FAILED** in both runs: the two QFT cells exceed +10 %. The other eight cells are within ±7.1 % in both runs.

Model all-ranks compute by kind (s; identical in both runs, since the model is deterministic):

| circuit | D | dense1 | dense2 | dense3 | diag1 | cnot | phase | exchange-only plan (s, run 2) |
|---|---|---|---|---|---|---|---|---|
| QFT | 2 | 0.984 | 0.042 | 0 | 0.018 | 0 | 1.666 | 0.070 |
| QFT | 4 | 0.984 | 0.085 | 0 | 0.044 | 0 | 1.605 | 0.097 |
| GHZ | 2 | 0.018 | 0 | 0.944 | 0 | 0 | 0 | 0.043 |
| GHZ | 4 | 0 | 0.021 | 0.871 | 0 | 0.032 | 0 | 0.056 |
| random d=10 | 2 | 1.300 | 4.566 | 0.654 | 0 | 0.171 | 0 | 0.283 |
| random d=10 | 4 | 2.319 | 4.523 | 0.654 | 0 | 0.236 | 0 | 0.456 |
| QAOA p=2 | 2 | 2.881 | 0.042 | 0 | 0.071 | 0 | 0.281 | 0.150 |
| QAOA p=2 | 4 | 2.670 | 0.296 | 0 | 0.053 | 0 | 0.282 | 0.216 |
| CCZ ladder d=4 | 2 | 0.984 | 0 | 0 | 0 | 0 | 0.082 | 0.043 |
| CCZ ladder d=4 | 4 | 0.984 | 0 | 0 | 0 | 0 | 0.079 | 0.056 |

`diag_k` is 0 on every cell and is omitted.

### 2.3 Reading

- **The gate fails on QFT only.** The model over-predicts QFT compute by 15.2–16.6 % (0.357–0.386 s on 2.33–2.35 s
  measured) at both D, in both runs. Per spec §6.3 the per-kind model must be revised and re-checked before `compile`
  lands; that revision is not part of this section.
- **QFT is the only cell where `PhasePoly` dominates.** It is 1.666 of 2.710 s (61 %) at D=2 and 1.605 of 2.717 s (59 %)
  at D=4. The QFT excess (0.357–0.386 s) equals 21–24 % of that phase term. `phase_base` moved +5.2 % between the two
  calibration runs, and it includes a per-launch allocation and upload.
  The phase kernel is therefore the first suspect, but this run does not isolate it.
- **The other cells are dominated by one dense kind each**, and the model holds there:
  - random d=10: `dense2` (4.566 of 6.690 s at D=2, 68 %); under-predicted by 4.0–4.9 %.
  - GHZ: `dense3` (0.944 of 0.962 s at D=2, 0.871 of 0.925 s at D=4); over-predicted by 4.5–7.1 %.
  - QAOA p=2: `dense1` (2.881 of 3.275 s at D=2, 2.670 of 3.301 s at D=4); within 0.6 %.
  - CCZ ladder d=4: `dense1` (0.984 of ~1.065 s); within 1.2 %.
- **Near the ±10 % edge:** GHZ at D=4, at +7.1 % in run 1 (+5.3 % in run 2). The next largest error is random d=10 at
  −4.9 % (run 2, D=4).
- **The representative-rank estimate over-predicts the all-ranks model by 0–2.3 %.** `R·model(R−1)` / model all-ranks
  is 1.000 (random D=2, CCZ both D) up to 1.023 (random D=4: 7.907 vs 7.731 s). Against measured compute it lands at
  0.952–1.161 (run 2), so it inherits the QFT failure and slightly widens it (1.160–1.161 vs 1.152–1.156).
- The exchange-only plan costs 0.043–0.456 s (run 2) on one card. That is on-card copy time, not interconnect time.
