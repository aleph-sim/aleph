# P6-02: State-vector partitioning strategy (distributed SV)

Issue #56. Design: [`docs/superpowers/specs/2026-10-03-multi-gpu-sv-design.md`](../superpowers/specs/2026-10-03-multi-gpu-sv-design.md).
Code: `crates/aleph-ir/src/dist/` (planner, backend-agnostic) and `crates/aleph-sv/src/dist_ref.rs` (CPU reference
executor).

## Strategy

- `R = 2^g` ranks. Physical index = `(rank << m) | local`, with `m = n − g`. Physical qubits `≥ m` are **global**: they
  select the rank. Qubits `< m` are **local**.
- The logical→physical qubit map is lazy and never unwound. A user `Swap` is an O(1) relabel.
- **Free on global qubits:**
  - Every diagonal gate. It is rank-specialised into a local diagonal or a per-rank phase. The phase depends on the
    rank, so it is never dropped as a "global phase".
  - Every control, external or in-qubit (`Cnot`/`CRx`/`CRy`/`Toffoli`). It becomes skip or drop per rank.
  - `DiagonalPhase`. Global parities fold to per-rank constants, and negated local conditions expand by
    inclusion–exclusion, `[A ∧ ¬B] = [A] − [A ∧ B]`.
- **An exchange is needed only for a non-diagonal target on a global qubit.** The exchange is a physical SWAP of that
  global bit with the top local bit `m−1`. That is one contiguous half-slice per rank, `2^(m−1)` amplitudes each way.
  If the top slot holds a qubit the same gate needs, a local `Swap` moves it down first.
- **Naive router (this PR):** one qubit per exchange, on demand. Lookahead and multi-bit routing are P6-03 (#57).

Reference: Häner & Steiger, "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit" (SC'17), §3.

## Correctness

`crates/aleph-sv/tests/dist_oracle.rs` checks the CPU reference executor (ranks as separate `2^m` slices, the exchange
done by the contiguous-chunk algorithm) against `NaiveSvBackend` at **1e-10**. It covers:

- GHZ-8, QFT-8 and random brickwall-9 for g = 0, 1, 2, 3;
- global-control Toffoli, a controlled-H with a global external control, and CCZ;
- a rank-dependent `Rz`/`CZ` on the top qubit;
- a trailing local↔global `Swap` relabel;
- Iswap, controlled Iswap and a dense `Unitary2q` on {top local slot, global} and on two globals, checked on amplitudes;
- every diagonal gate kind (Z/S/Sdg/T/Tdg/Rz/Phase/`Unitary1qDiag`/Cz/CRz/Ccz) with a global qubit in every operand
  position, with and without controls;
- Grover-8 (13 iterations) for g = 0..3;
- a 64-case proptest over arbitrary unitary circuits (n=6, g=0..3);
- a 96-case proptest over every gate family the planner distinguishes (incl. CRx/CRy/CRz, Iswap/IswapDg, U3,
  `Unitary1q`/`Unitary2q`, Toffoli, Ccz) with 0–2 random external controls (n=6, g=0..2).

Two mutation checks show the oracle has teeth:

- Dropping the last exchange of a plan is detected by `mutation_dropping_an_exchange_breaks_oracle`. Dropping the
  *first* one is not a valid mutation: it acts on `|0…0⟩`, where swapping two `|0⟩` qubits is the identity.
- Inverting `DistLayout::rank_bit` by hand fails 7 of the 8 original oracle tests. Flipping the matrix bit order in
  `diag_reduce` (MSB↔LSB) by hand is caught only by the diagonal-gate table and the all-families proptest. Both were
  manual checks and are not committed.

`specialize` has 10 unit tests in `crates/aleph-ir/src/dist/specialize.rs`. Among them, `DiagonalPhase` is checked
against the full-index `phase_at` for every rank and local index. The planner has 9 unit tests in `plan.rs`, including saturation of the traffic counter at n=64. `DistLayout` rejects
`g > 31` (ranks are `u32`).

## Communication count (naive router)

`cargo run --release -p aleph-sv --example dist_comm_counts`

| circuit | gates | g | exchanges | amps moved / rank | × slice | local swaps | relabels |
|---|---|---|---|---|---|---|---|
| GHZ-32 | 32 | 2 | 2 | 1073741824 | 1.0 | 0 | 0 |
| GHZ-32 | 32 | 3 | 3 | 805306368 | 1.5 | 0 | 0 |
| QFT-32 | 544 | 2 | 3 | 1610612736 | 1.5 | 0 | 16 |
| QFT-32 | 544 | 3 | 4 | 1073741824 | 2.0 | 0 | 16 |
| random-30 d=20 | 1490 | 2 | 89 | 11945377792 | 44.5 | 0 | 0 |
| random-30 d=20 | 1490 | 3 | 119 | 7985954816 | 59.5 | 0 | 0 |
| Grover-20 (5 iters) | 96210 | 2 | 117 | 15335424 | 58.5 | 0 | 0 |
| Grover-20 (5 iters) | 96210 | 3 | 293 | 19202048 | 146.5 | 0 | 0 |

`× slice` is the number of amplitudes moved per rank divided by the per-rank slice size `2^m`. Each `k=1` exchange
moves half a slice, so `× slice = exchanges / 2`.

## What this means for P6-03

- **QFT-32 is already near the floor:** g+1 exchanges for the whole circuit. All 496 controlled-phases are free, and the
  16 final swaps are relabels. QFT is diagonal-dominated, so its distributed cost is about one H per global qubit.
- **GHZ** costs exactly g exchanges, one per CNOT target that is global.
- **Random brickwall and Grover are exchange-bound.**
  - The brickwall applies a non-diagonal `Rx` to *every* qubit in *every* layer: 89 exchanges over 20 layers at g=2,
    about 4.5 per layer. The naive router evicts the top local qubit, and the very next layer needs it again.
  - A lookahead router that exchanges all `g` needed qubits in one `k=g` step should get close to one exchange per
    layer: `(1 − 2^−g)` of a slice per layer, about 15 slices instead of 44.5 at g=2.
  - Reordering commuting gates (#59) is the next lever after that.
- **Grover-20 is small here** (the slice is only `2^18`). Its gate count, not its byte count, dominates. It does show
  that multi-controlled decompositions hit global targets often (117–293 exchanges).
