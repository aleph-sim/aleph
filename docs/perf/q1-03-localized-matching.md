# Q1-03 — Localized MWPM matching (savings reformulation)

**Status:** partial. Delivers an exact, weight-identical speedup of the Q1-02 MWPM decoder;
does **not** yet reach the issue's ≥10× target, which needs a Sparse-Blossom rewrite (see
"Remaining gap"). Tracked as a follow-up.

## What changed

Two exact, correctness-preserving levers over the Q1-02 dense decoder
(`crates/aleph-qec/src/mwpm.rs`, `crates/aleph-qec/src/blossom.rs`):

1. **Savings reformulation (clone-free matching).** Q1-02 encoded "matching with a boundary" as a
   maximum-cardinality maximum-weight *perfect* matching on `2n` nodes — `n` defects plus one
   private boundary clone each, with a full `O(n²)` clone clique. Q1-03 instead solves a
   **non-perfect** maximum-weight matching on just the `n` defect nodes, where each edge carries
   the *savings* of pairing two defects rather than sending both to the boundary:

   ```text
   savings(i,j) = b_i + b_j − dist(i,j)            (b_i = defect i's distance to the boundary)
   total cost   = Σ b_i − Σ_matched savings(i,j)   (an unmatched defect goes to the boundary)
   ```

   Minimising cost ⇔ maximising total savings. This halves the node count, removes the clone
   clique entirely, and keeps only positive-savings edges — which is exactly the boundary prune
   `dist(i,j) < b_i + b_j` and is **weight-exact** (it never drops an edge the optimum needs).

2. **Thread-local reusable blossom solver.** The blossom engine previously allocated ~`O(n)`
   vectors per call (one `max_weight_matching` per decoded shot). It is now a reusable solver
   (`Blossom::load` reuses all buffer capacity) held in a thread-local, plus a trivial-vertex fast
   path in `assign_label` that skips a per-stage `blossom_leaves` allocation. This benefits both
   the dense reference path and the localized path.

Both paths are kept: `decode_dense` (Q1-02 reference) and `decode` (Q1-03 localized, default).

## Results

Benchmark: `cargo bench -p aleph-benches --bench mwpm_decode` (256 surface-code memory-Z
syndromes per distance, phenomenological `p = 0.03`, near threshold). Machine: local M4 Mac
(dev box, not the idle EPYC — treat absolute times as indicative; ratios are the signal).

Same-run, same-machine **dense (Q1-02 algorithm) vs local (Q1-03)** — isolates the savings
reformulation (both run the same reusable-solver blossom engine):

| d  | avg defects | dense    | local    | speedup | local throughput |
|----|-------------|----------|----------|---------|------------------|
| 7  | 27          | 14.61 ms | 3.93 ms  | 3.72×   | 65.2 K syn/s     |
| 9  | 56          | 67.77 ms | 18.49 ms | 3.66×   | 13.8 K syn/s     |
| 11 | 104         | 267.4 ms | 73.75 ms | 3.62×   | 3.47 K syn/s     |
| 13 | 169         | 897.5 ms | 235.5 ms | 3.81×   | 1.09 K syn/s     |

The reusable-solver change additionally sped the shared blossom engine: the original frozen
Q1-02 decode measured **350.8 ms** at d=11 before any change, vs **73.75 ms** for the localized
path now (≈4.8× end-to-end, with a machine-load caveat). The 3.6–3.8× figure above is the clean,
reproducible apples-to-apples number.

## Correctness

- **Differential vs the dense Q1-02 decoder** (`local_matches_dense_weight_and_corrections`):
  the localized matching reaches the **same minimum weight on 100% of shots** for d ∈ {3,5,7,9,11}
  — the rigorous optimality invariant — proving the savings prune never drops a needed edge.
  Corrections are identical except on genuine MWPM ties (equal-weight matchings in different
  homology classes; ≈1.4% of non-empty shots at d=5, fewer at larger d), which is the same
  degeneracy the PyMatching oracle exhibits and which leaves the logical-error rate unchanged.
- The blossom engine's `maxcardinality = false` (non-perfect) path — newly relied on by the
  savings formulation — is validated against a brute-force subset-DP optimum on 200 random graphs
  per size up to n=11, including negative edges (`nonperfect_matches_brute_force`).

## Profiling — why it plateaus at ~3.7×

At d=11 (avg n=103 defects, ~1324 positive-savings edges), the decode splits as **build ≈ 58 µs,
blossom ≈ 221 µs** per shot. Removing every remaining allocation in the hot blossom internals
(`add_blossom`/`scan_blossom`) moved the number by 0% (within noise) and was reverted — so the
221 µs is **compute**, not allocation: the textbook blossom restarts its tree search and re-scans
all tight edges on each of the ~`n` augmentation stages (`O(n·m)`).

## Remaining gap to ≥10×

Closing 3.7×→10× needs eliminating the per-stage restart, i.e. local **region-growing with
persistent search state** — the Sparse Blossom algorithm (Higgott & Gidney, arXiv:2303.15933) /
Blossom-V (Kolmogorov). That is a from-scratch, research-grade engine, deferred to a dedicated
follow-up; it is also the change that makes the decoder scale to high distance.
