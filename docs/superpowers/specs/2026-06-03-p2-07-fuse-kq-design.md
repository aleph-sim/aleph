# [P2-07] Deep k-qubit gate fusion (FuseKq, k ≤ 5) — design

> Status: approved (brainstorm 2026-06-03). Implements GitHub issue #107.
> Depends on: P1-09 (`Fuse1qRuns`), P1-10 (`Fuse2q`), P1-07 (2q dense kernel +
> renormalised outer-walk), P2-06 (`FuseDiagonalRuns`, pipeline placement).

## 1. Problem and motivation

P2-05 (`docs/perf/phase2.md`) established that state-vector simulation is
memory-bandwidth-bound: each gate streams the full `2^n` state with near-zero
arithmetic intensity, so wall-clock is dominated by the *number of passes over
memory*. P2-06 attacked this for QFT by collapsing diagonal+`cx` ladders into a
single `DiagonalPhase` pass. The **general** antidote is raising arithmetic
intensity: one pass that does a `2^k × 2^k` matrix–vector product per
`2^k`-amplitude block does `O(2^k)` FLOPs per amplitude moved instead of `O(1)`
per single gate — turning `M` separate full-state passes into one.

This is the standard "gate fusion" of Qiskit Aer and qsim, and is the biggest
lever for the **fusible** Tier-1 workloads — VQE / QAOA / random brick-wall —
which P2-05 measured at 2.5–2.8×@16 (better than QFT but far from linear)
because they currently apply many small gates. `Fuse1qRuns` + `Fuse2q` already
fuse up to 2 qubits; this extends the same idea to a configurable `k ≤ 5`.

Unlike P2-06's *diagonal* kernel (pure memory streaming, where AVX-512 gave
~0 over scalar on bandwidth-bound hardware — P2-04), a dense `k`-qubit matvec
does `2^k` complex-muls per amplitude. It **raises** arithmetic intensity, which
is the regime where SIMD genuinely pays — so this design ships scalar **and**
AVX-512 kernels.

## 2. Scope decision (brainstorm)

- **Complementary pass.** `FuseKq` is a NEW pass running **after**
  `Fuse1q`/`Fuse2q`/`FuseDiagonal`. Its inputs are therefore mostly the
  `Unitary1q`/`Unitary1qDiag`/`Unitary2q` blocks those passes already produced,
  plus leftover specialized gates (lone `Cnot`/`Cz`/`Swap`, `H`, …) and
  `DiagonalPhase` fences. It greedily merges adjacent blocks that chain across
  qubits (e.g. `Unitary2q(0,1)` then `Unitary2q(1,2)` → one `Unitary3q(0,1,2)`)
  into dense `k`-qubit blocks. It does **not** rewrite the existing passes
  (lowest risk; one-issue-one-PR).
- **Full kernel rigor:** scalar + AVX-512 generic-`k` dense kernel, AoS + SoA,
  EPYC-validated.
- **Cost-model boundary:** a closed block becomes a dense `UnitaryKq` only when
  its span is **≥ 3 qubits** *and* it absorbed **≥ 2 gates**; otherwise its
  members are re-emitted verbatim. So 1q/2q stay as the existing passes left
  them and lone `Cnot`/`Cz`/`Swap` keep their specialized kernels — dense fusion
  never makes a cheap gate more expensive.
- **`max_qubits` caps at 5** (32×32, 16 KiB matrix); the *default* (4 vs 5) is
  chosen by EPYC measurement in the perf task (deferred to data, as P2-04 did
  for grain).

## 3. IR representation

New `Gate` variant (a dense `k`-qubit unitary acts on a fixed, small,
relocatable qubit set — exactly like `Unitary2q` — so it belongs in `Gate`, not
as an `Instruction` like the full-register `DiagonalPhase`):

```rust
// crates/aleph-core/src/gate/kinds.rs
Gate::UnitaryKq { k: u8, data: Box<[Complex]> }   // row-major 2^k × 2^k, len == 4^k
```

- **Qubit convention** generalizes `Unitary2q`: for operands
  `[qubits[0], …, qubits[k-1]]`, matrix index bit `k-1` (MSB) = `qubits[0]`,
  bit `0` (LSB) = `qubits[k-1]`. Row `i`, col `j` is `⟨i|U|j⟩`, flattened
  `data[i * 2^k + j]`.
- `arity() == k as usize`. `is_diagonal() == false`, `is_clifford() == false`.
- `inverse()` = conjugate-transpose of the `2^k × 2^k` matrix (same `k`).
- `name() == "UnitaryKq"`.
- `matrix()` returns `Err(GateError::Unrepresentable)` for `k > 3` (the
  `GateMatrix` enum stops at `M8x8`); the SV backend never calls `matrix()` for
  `UnitaryKq` — it dispatches on the variant and reads `data` directly. For
  `k ≤ 3` we still do NOT route through `GateMatrix` (keep one code path). Add a
  `GateError` variant if one does not already fit.
- Constraints asserted at construction by the producer (the pass): `2 ≤ k ≤ 5`
  and `data.len() == 1 << (2*k)`. (`k` is `u8`; `2^k` and `4^k` fit easily.)

### 3.1 Blast radius
- `Gate` exhaustive matches gain a `UnitaryKq` arm: `arity`, `name`,
  `is_diagonal`, `is_clifford`, `inverse`, `matrix` (in `kinds.rs`), and any
  match in `commute.rs` (treat as non-diagonal, conservative — does not commute
  except trivially). Most other passes run *before* `FuseKq` and use
  `arity()`/`used_qubits()` generically.
- SV backends (`NaiveSvBackend` AoS, `SoaSvBackend` SoA): a dispatch arm sending
  `UnitaryKq` to the new `apply_kq` kernel, keyed on the variant **before** the
  arity dispatch.
- `aleph-parser::emit` refuses it (`UnsupportedGate { name: "UnitaryKq" }`),
  like `Unitary2q`.
- `GateInstance::qubits` carries the `k` target qubits; uniqueness already
  validated by `Circuit::add_gate`. No external `controls` on a `UnitaryKq`.

## 4. The pass — `passes::FuseKq`

New file `crates/aleph-ir/src/passes/fuse_kq.rs`.

```rust
pub struct FuseKq { pub max_qubits: usize }   // 2 ≤ max_qubits ≤ 5
impl Default for FuseKq { fn default() -> Self { Self { max_qubits: 4 } } } // default tuned in perf task
```

**Greedy, dependency-respecting block growth.** Walk the instruction stream once
maintaining a set of **open blocks**, each = `{ qubits: BitSet, members: Vec<usize> }`
(member instruction indices). Per-qubit, track which open block currently "owns"
that qubit (its most recent writer). For each gate instruction `g` with support
`S = used_qubits(g)`:

1. **Fences** (`DiagonalPhase`, `Barrier`, `Measure`, `Reset`, or any
   non-`Gate` instruction): close every open block whose qubits intersect the
   fence's qubits (a `Barrier`/`Measure` on a subset closes only the touching
   blocks; a global barrier closes all). Then emit the fence verbatim.
2. Let `B` = the union of open blocks owning any qubit in `S` (the blocks `g`
   depends on). Candidate merged support = `S ∪ (⋃ B.qubits)`.
   - If `|candidate| ≤ max_qubits`: merge all blocks in `B` into one, append `g`,
     set that block as owner of every qubit in the merged support.
   - Else: **close** the blocks in `B` (emit them), then start a **new** block
     `{ qubits: S, members: [g] }` owning `S`.
3. At end of stream, close all remaining open blocks.

Closing a block runs the **cost model**: if `block.qubits.len() ≥ 3` *and*
`block.members.len() ≥ 2`, build the dense matrix (below) and emit a single
`UnitaryKq`; otherwise re-emit the block's member instructions **verbatim, in
original order** (preserving specialized kernels). Either way the relative order
of non-overlapping blocks is preserved by keying output position on the block's
**first** member index (stable sort), mirroring the P1-10 output-ordering fix.

**Dense matrix build.** For a block over sorted qubits `Q = [q0<…<q_{k-1}]`
(`k = block.qubits.len()`), start from the `2^k × 2^k` identity and, for each
member gate in circuit order, **left-multiply** its lifted matrix: take the
gate's own matrix (via `gate.matrix()` for standard/`Unitary1q/2q`/`Unitary1qDiag`,
or its `data` for a nested `UnitaryKq` — possible if `max_qubits` lets two
`UnitaryKq`s merge), embed it into the `k`-qubit space by tensoring with
identity on `Q ∖ gate.qubits` and permuting rows/cols so the gate's operand
order maps to `Q`'s MSB-first positions. This is `O(member_count · (2^k)^3)` at
build time — once per block, negligible vs the `2^n` apply.

Helper `lift_to_block(gate_matrix, gate_qubits, block_qubits) -> 2^k×2^k` is the
delicate part; it gets its own unit tests (identity placement + operand-order
permutation verified against a brute-force basis-state simulation).

**Determinism:** no `HashMap` in output ordering; block list and member lists
are index-ordered.

## 5. Backend kernel — generic `apply_kq`

`apply_kq(state, qubits: &[u32], k, data: &[Complex])` applies the dense
`2^k × 2^k` matrix in one pass:

- Sort `qubits` ascending → bit positions `Q`. Precompute `offsets[0..2^k]`: for
  mask index `m`, `offsets[m] = Σ_{b: m has bit b} (1 << Q[mapped])`, i.e. the
  amplitude-index displacement produced by the `k` target bits in MSB-first
  operand order. (Generalizes P1-07's 2q `offsets`.)
- Outer loop over the `2^(n-k)` **free-bit** combinations (all non-target
  qubits) → a `base` index with the `k` target bits cleared. Generalize
  `expand_with_fixed` to splice the free bits around the `k` fixed (cleared)
  positions.
- Inner: gather the `2^k` amplitudes `ψ[base | offsets[m]]`, compute the matvec
  `out[i] = Σ_j data[i*2^k + j] · in[j]`, scatter back.
- `par_blocks`/`par_units` over the `2^(n-k)` outer dimension (blocks are
  pairwise-disjoint → safe; bit-identical regardless of thread count).

**Scalar** first (correct reference). **AVX-512** (AoS + SoA): vectorize the
inner matvec across the `2^k` gathered amplitudes / across lanes; reuse the
complex-multiply and de-interleave patterns from P1-07's 2q kernel and P2-06's
kernel, generalized to a runtime `2^k`. Gated on `is_x86_feature_detected!`.
Every `unsafe` carries a `// SAFETY:` block; the disjoint-block argument is the
soundness basis.

**Mandatory indexing-coverage test** (the P1-07 SIGSEGV lesson): an integer-only
reproduction asserting `{ base | offsets[m] }` are pairwise-disjoint and cover
each amplitude exactly once, for every `(n, k, qubit-placement)` including
`k`-qubit blocks touching the top qubits and `n−k < LANES` edge cases.

## 6. Pipeline integration

`default_pipeline()` →
`[Cancel, DCE, FuseDiagonal, Fuse1q, Fuse2q, FuseKq::default()]`.
`FuseKq` runs last so it consumes the dense 1q/2q blocks the others produced.
`UnitaryKq` is opaque to every earlier pass (they run before it) and is a fence
for `FuseKq` itself (a `UnitaryKq` already at `max_qubits` cannot grow), so the
pipeline still converges (idempotence caveat identical to P2-06: convergent,
documented).

## 7. Testing requirements

1. **`lift_to_block` unit tests:** identity placement and operand-order
   permutation, verified vs brute-force basis-state application (catches the
   MSB/LSB and tensor-position bugs that plagued P1-07/P1-10).
2. **Property test:** a fused `k`-block ≡ sequential application of its members
   on a **generic** input state (not `|0…0⟩` — P1-13 lesson), for k = 3, 4, 5,
   to 1e-12.
3. **Indexing-coverage test** (§5).
4. **Standalone pass test** (run `FuseKq` directly) **+ pipeline idempotence /
   convergence test** (P1-10/P2-06 lesson: a `pub` pass must be tested
   standalone, not only via the pipeline).
5. **Cost-model tests:** a 2-qubit-only run is NOT fused (left to `Fuse2q`); a
   single multi-qubit gate is NOT fused; a lone `Cnot` between fences is
   re-emitted verbatim.
6. **Oracle:** equivalence vs unfused across Tier-1 fixtures, raw pass and via
   `default_pipeline`, 1e-12 incl. global phase, through both SV backends.
7. **Scalar ↔ AVX-512 equivalence** on the EPYC box.
8. **Benchmark:** VQE / QAOA / random brick-wall before/after on EPYC
   (pass-count + wall-clock), plus a `max_qubits ∈ {2,3,4,5}` sweep to pick the
   `default()`.

## 8. Acceptance criteria (BACKLOG #107) and how this meets them

- [ ] Configurable `max_qubits`; `default_pipeline()` uses a sensible default
  (4 or 5, picked by §7.8 sweep). → §4, §6.
- [ ] VQE/QAOA/random pass-count reduction and criterion speedup on EPYC,
  reported in the PR. → §7.8.
- [ ] Oracle equivalence vs unfused within 1e-12 across Tier-1 fixtures. → §7.6.

## 9. Out of scope / follow-ups

- Cross-pass cluster capture that the "complement, run-after" choice forgoes
  (e.g. a 1q+2q+2q span the linear passes split): a future unified fuser
  (the "Replace" option) could recover it.
- Cache-aware block scheduling and qubit relabelling — that is **P2-09**
  (`apply_kq` is the natural place those tiles will later be applied).
- Commutation-aware reordering to enlarge fusible blocks (uses
  `passes::gates_commute`; deferred).
