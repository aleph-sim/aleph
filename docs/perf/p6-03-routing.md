# P6-03: Lookahead multi-bit routing (distributed SV)

Issue #57. This builds on P6-02 ([`p6-02-partitioning.md`](p6-02-partitioning.md)). The code is
`Router::Lookahead` in `crates/aleph-ir/src/dist/plan.rs` (`exchange_lookahead`), plus `dist/next_use.rs`.

## Algorithm

- **Batch:** when a gate needs global qubits, all of them come in with **one k-bit exchange** (k ≤ g). That costs
  `(1 − 2^−k)` of a slice per rank, instead of `k/2` for k separate single-bit swaps.
- **Prefetch:** further global qubits come in on the same exchange, soonest next use first. A qubit qualifies only if
  it is needed *before* the next use of the local qubit it would evict.
- **Victims:** the local qubits the current gate does not need, farthest next use first (Belady). Ties prefer a higher
  physical slot.
  - Victims move into the top-k local slots with local `Swap`s, so the exchange stays a contiguous-chunk all-to-all
    (`global_bits[j] ↔ m−k+j`).
  - A local swap is one on-device bandwidth pass. That is cheap next to a link transfer.
- **Next use** is precomputed once per qubit's *data*: the positions where `required_local` needs it.
  - A user `Swap(a, b)` is an O(1) relabel that hands label `a`'s data to label `b`, so uses are recorded per
    data track. Labels are mapped to tracks by replaying the swaps at build time, and by `NextUse::relabel` while
    planning.
  - Keying by label instead mispredicts after a relabel. On random circuits with relabels, that made lookahead move up
    to 2.5× (4.5× on a wider search) more data than naive. `lookahead_bounded_regression_with_relabels` pins this
    case.
- `Router::Naive` (P6-02) is unchanged and stays as the baseline.

## Correctness

`crates/aleph-sv/tests/dist_oracle.rs` (17 tests) runs **every** P6-02 oracle case under both routers. All are checked
against `NaiveSvBackend` at 1e-10:

- GHZ, QFT, brickwall and Grover-8 for g = 0..3;
- every diagonal gate kind on global positions;
- Iswap and `Unitary2q` on {top slot, global};
- global-control Toffoli;
- a trailing relabel.

New in this PR:

- `exchange_cpu_multi_bit_is_product_of_swaps`: the 2- and 3-bit exchange primitive (bit orders `[3,4]`, `[4,3]`,
  `[5,3,4]`) equals the explicit product of physical SWAPs, element by element.
- `tight_m_prefetch_cap_oracle` and `required_qubit_in_top_slot_k2_oracle`: the prefetch cap at m=2 and a required
  qubit in a top slot during a k=2 exchange, checked on amplitudes.
- `mid_circuit_relabel_then_lookahead`: local↔global `Swap` relabels in the middle of a brickwall, then more layers,
  for g = 1..3.
- Both proptests draw the router at random. One covers 64 arbitrary circuits; the other covers 96 circuits over all
  gate families with external controls.
- Planner unit tests (`aleph-ir`):
  - every exchange is ≤ g distinct global bits;
  - a gate's own qubit is never evicted;
  - the tight m=2 case does not prefetch and does not error;
  - lookahead never moves more than naive on GHZ/QFT for g = 1..3;
  - on 1,200 deterministic random 30-gate circuits mixing relabel `Swap`s, Iswap, CNOT, H, Rx, Rz and CZ
    (n ∈ {6,8,10,12}, g = 1..3), lookahead moves at most 1.3× what naive moves;
  - a 2× reduction bound on a 16-qubit brickwall.

**Mutation check (manual):** I paired the k global bits with the top slots in reverse order inside
`exchange_lookahead`. Five of the then-15 oracle tests failed: brickwall, Grover, mid-circuit relabel and both proptests. I
then reverted it.

## Measured reduction

`cargo run --release -p aleph-sv --example dist_comm_counts`

| circuit | g | naive exch | naive × slice | lookahead exch | lookahead × slice | reduction | la local swaps |
|---|---|---|---|---|---|---|---|
| GHZ-32 | 2 | 2 | 1.0 | 1 | 0.8 | 1.33× | 0 |
| GHZ-32 | 3 | 3 | 1.5 | 1 | 0.9 | 1.71× | 0 |
| QFT-32 | 2 | 3 | 1.5 | 2 | 1.5 | 1.00× | 2 |
| QFT-32 | 3 | 4 | 2.0 | 2 | 1.8 | 1.14× | 3 |
| random-30 d=20 | 2 | 89 | 44.5 | 22 | 16.5 | 2.70× | 42 |
| random-30 d=20 | 3 | 119 | 59.5 | 23 | 20.1 | 2.96× | 67 |
| Grover-20 (5 iters) | 2 | 117 | 58.5 | 20 | 12.5 | 4.68× | 24 |
| Grover-20 (5 iters) | 3 | 293 | 146.5 | 30 | 20.6 | 7.10× | 28 |

`× slice` is the amplitudes moved per rank divided by the per-rank slice size `2^m`. `reduction` is naive divided by
lookahead.

## Reading

- **Random brickwall** applies a non-diagonal `Rx` to every qubit in every layer, so at least one exchange per layer
  is unavoidable without reordering. Lookahead reaches **22–23 exchanges for 20 layers**, close to that floor, and moves
  **2.7–3.0× less** data than naive.
  - The cost is 42–67 local swaps, one on-device pass each.
  - On a PCIe box the device bandwidth is roughly 30× the link (P5.10-02 measured ~12 GB/s PCIe vs ~360 GB/s
    on-card). On that basis 42 local swaps are worth under 2 slice transfers, well under the ~28 slices saved.
- **Grover** gains most (4.7–7.1×). Its multi-controlled decompositions keep touching the same few global targets, and
  naive keeps evicting exactly the qubit needed next.
- **GHZ** and **QFT** were already near their floor (P6-02). Lookahead ties or wins: QFT at g=2 is 1.00×, the same
  bytes in fewer exchanges, which means fewer synchronisation points. A unit test pins "never worse" for these two.
- **Lookahead is a heuristic, not a guarantee.** A prefetch trades the prefetched qubit's future fetch for its
  victim's, so on general circuits it can lose modestly. The final review's search over 4,800 small random circuits
  found 26 regressions, the worst 1.25×.
  - Allowing prefetch only when the victim is never used again removes every regression, but costs the headline:
    random-30 drops to 2.25×/2.03× and Grover to 4.50×/6.44×.
  - The current rule is the better trade.
- **What is left for #59** (commutation-aware reordering): today the router keeps gate order, so the floor is about
  one exchange per layer of non-diagonal gates on global qubits. Gates on disjoint qubits commute. A reordering pass
  could sink a run of local-only gates past a global-qubit gate, or hoist global-qubit gates from several layers
  together, so that one k=g exchange serves more of them. How much that buys is circuit-dependent and still
  unmeasured. Real GPU timings (exchange vs local-swap cost on a
  link) come with the multi-GPU backend and the AWS session; this PR is planning-only.
