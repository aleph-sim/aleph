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
