# Q7-07 — Non-convergence fallback policy (`valid_flag=0` path)

**Issue:** #458 · **Backlog:** `docs/qec/BACKLOG.md` § Q7-07 · **Depends on:** Q7-02, Q7-06
**Date:** 2026-07-25 · **Status:** design approved, not yet implemented

## Problem

Relay-BP occasionally emits a hard decision that violates the syndrome (`H ê ≠ s`). Both the
software golden and the RTL already detect and report this — the core sets `valid_flag = found`
(`hw/bp_relay_banked.sv:968`) and emits the best-kept decision anyway (`:956`), and
`FixedRelayBp::decode_fixed` returns the same bool. What does not exist is a **policy**: a
measured statement of how often this happens at the shipped operating points, how much of the
logical error budget it accounts for, and whether any post-processing is worth its cost.

## Acceptance criteria (from the backlog)

- AC-1: non-convergence rate quantified per operating point.
- AC-2: fallback policy chosen with data (including do-nothing-but-flag if rates are negligible);
  the LER impact of the chosen policy measured in software.

## Prior art in-tree — what is already settled

Q7-07 must not re-litigate these:

- **OSD-0 does not help and order-12 is intractable.** `docs/perf/qec-q7-fixed-bp.md:587–640`
  (Q7-02 M5-followup): at circuit level, rounds=6, 3 000 shots, OSD-0 makes LER *worse*
  (2.7e-2 vs 2.0e-2 plain at p=0.003), order-4 is roughly break-even, order-12 wins but costs
  4096 reliability-ordered GF(2) solves per failure shot. Verdict recorded there: Q7-02 ships
  pure relay-BP.
- **The RTL is bit-exact to the software golden.** Q7-06 AC-2: 10⁶ shots × 3 circuit-level rates,
  `rtl_err == sw_err` exactly, 0 divergence at every point (`docs/qec/q7-06-ac1-batched-dma.md`).
  This is the licence to measure convergence statistics in software at arbitrary scale — the
  software golden's answer *is* the silicon's answer.
- **Convergence is already near-total on the block path at low p.** `qec_q7_early` (circuit-level
  rounds=1, 40 000 shots): 100.0 / 100.0 / 99.9 / 99.2 % converged at p = 0.001 / 0.002 / 0.003 /
  0.005 (`docs/perf/qec-q7-fixed-bp.md:968–973`).
- **The window path is a different regime.** M9b, rounds=12, W=6/C=2: 11.82 / 66.82 / 96.01 % of
  *shots* see ≥1 non-converged window at p = 1/3/5 × 10⁻³, while only 0.33 / 8.91 / 51.02 %
  discard a non-empty commit region (`commit_clean`) — most non-converged windows still drain
  cleanly, so the per-shot non-convergence figure overstates the problem.

## Why the obvious measurement does not work

The natural reading of AC-2 — "run the campaign with and without the fallback, compare LER" — is
statistically hopeless. At p=0.003 the campaign LER is 8.3e-4 with a 95 % CI of ±1.13e-4 at 10⁶
shots, and the non-convergence rate is order 0.1 %. Any fallback's effect on overall LER is
buried several orders of magnitude under the campaign CI; resolving it directly would need
~10⁸ shots per arm.

Q7-07 therefore measures **conditionally**, and propagates to overall LER analytically.

## Measurement architecture

### Level 1 — rate

Per operating point, `r(p) = P(valid_flag = 0)` with a Wald 95 % CI (`LogicalErrorResult::new`
in `crates/aleph-qec/src/decoder.rs:74` already computes this shape). Cheap: `decode_batch`
(`fixed_bp.rs:495`) is rayon-parallel, so 10⁶–10⁷ shots per point is routine on the EPYC box.
Reported alongside the iteration-count distribution from `iters_to_valid` (`fixed_bp.rs:357`):
mean / p50 / p99 / max.

### Level 2 — attributable fraction (the ceiling)

Split logical errors by the flag:

- `P(err | valid = 0)` — is a non-converged decode actually a lost shot?
- `P(err | valid = 1)` — the converged-but-wrong background.
- `A(p) = (# logical errors with valid = 0) / (# logical errors total)`.

**`A(p)` is the hard ceiling on any fallback.** A fallback only ever acts on `valid = 0` shots, so
even an oracle that decodes every one of them perfectly reduces LER by exactly `A(p)`. If `A(p)`
is a few percent, no candidate can move the needle and the verdict follows from arithmetic
rather than from a candidate sweep.

### Level 3 — conditional rescue (only if the ceiling is material)

Decode a large stream, retain **only** the `valid = 0` shots as a fixed corpus on disk, and run
each candidate against that dense subset. This yields a tight rescue rate from ~10³ non-converged
shots instead of 10⁸ full campaign shots. Corpus target is 10³ retained shots per operating point,
so the stream length is `10³ / r(p)` — Level 1's rate measurement sizes Level 1's own stream and
Level 3's in one pass. Propagate:

```
ΔLER(p) = r(p) · [ P(err | valid=0) − P(err | valid=0, fallback) ]
```

Significance on the subset is tested paired (McNemar), matching the convention M9a/M9b adopted
for the same reason (`docs/perf/qec-q7-fixed-bp.md:1315`).

## The two decoder paths

Measured separately and never pooled — the flag means different things.

| | block decoder (**primary**) | sliding window (**secondary**) |
|---|---|---|
| source | Q7-02 `bp_relay_banked`, Q7-06 campaign vehicle | Q7-04 M9b `HwSlidingWindowBp` |
| unit of the flag | one decode | one window; one shot spans many windows |
| shipped schedule | 6 legs × 10 iters, Q5.3, early-exit | same core on the W=6/C=2 truncated graph |
| operating points | p = 0.003 / 0.005 / 0.007, rounds = 1 | p = 0.001 / 0.003 / 0.005, rounds = 12 |
| existing figures | 99.9 / 99.2 % converged (rounds=1, p=0.003/0.005) | 11.82 / 66.82 / 96.01 % shots with ≥1 non-converged window |

The block path is primary: it is what the AC's `Depends on: Q7-02` names and what Q7-06 qualified
on silicon. The window path is reported because it is the deployable multi-round decoder and its
rates are not negligible.

For the window path, report **both** a per-window rate and the per-shot "≥1 non-converged window"
rate — the existing 12/67/96 % figures are the latter. Carry `StreamStats.residual` and
`WindowTrace.commit_clean` (`relay_window.rs:100`, `:446`) alongside, since M9b already argues
discarded-bits is the sharper health signal.

## Latency constraint on any fallback

Independent of LER, and potentially decisive. Q7-01 targets **1 µs/round**; early-exit decode is
1.81 µs/shot on silicon (Q7-06 AC-1). A PS-side tail is amortized-cheap when `r` is small, but
real-time QEC is governed by the **worst case**, not the mean: a tail that fires on 0.1 % of shots
still sets worst-case latency. Every candidate is therefore costed in two units — GF(2)
eliminations per shot, and measured PS microseconds — and a candidate that wins on LER while
breaking the latency budget is recorded as **rejected-on-latency**, with the arithmetic shown.

## Candidates

The baseline is not a candidate; it is the reference every candidate is measured against.

**Baseline — do-nothing-but-flag.** Emit the best-kept decision, set `valid_flag = 0`. This is
current RTL and software behaviour.

Cost ladder, all reusing `OsdDecoder::correction_from_soft` (`crates/aleph-qec/src/osd.rs:104`):

1. **OSD-0** — most-reliable-basis solve, no combination sweep. 1 GF(2) elimination/shot. Known
   to lose; measured again at the *shipped* operating points so the rejection is current.
2. **OSD-w, w ∈ {2, 4}** — bounded global sweep, `2^w` solves.
3. **Residual-restricted OSD-w** — the literal reading of "OSD-lite on the residual" and the only
   genuinely new candidate: run the order-`w` combination sweep only over variables in the support
   of the **unsatisfied** checks, not the whole 144-variable basis. Because the unsatisfied set is
   small at sub-threshold `p`, this buys a much higher effective order for the same solve budget.
   Implemented as a filter on the sweep index set in `osd.rs`.

Explicitly **out of scope** (not selected): retry-with-fresh-disorder (extra relay legs), and
flag-and-escalate-to-a-host-decoder.

## Pre-registered decision rule

Fixed before the data is seen, so the verdict is not fitted to it.

- `A(p) < 5 %` at every operating point → **do-nothing-but-flag**. Ship `valid_flag` as a
  health/telemetry signal only.
- `A(p) ≥ 5 %` **and** some candidate shows a significant conditional rescue (paired McNemar on
  the non-converged subset) **and** its worst-case latency fits the 1 µs/round budget → that
  candidate is chosen and implemented.
- Wins on LER, breaks latency → **rejected-on-latency**, arithmetic recorded.

**Implementation boundary.** If a candidate wins, the implementation for any OSD variant is a
**PS-side tail on `!valid_flag` shots** — the RTL already exports the flag, and the ARM runs the
Gauss–Jordan. No RTL change, no re-synthesis, no timing re-close. If the data instead points at a
hardware-side policy (e.g. extra relay legs on non-convergence), that is a materially larger
change and returns to the user for a scope decision before any RTL is touched.

## Changes

### Software

**`crates/aleph-qec/examples/qec_q7_bp_graph.rs`** — `emit_sil_vectors` (`:2153`) currently writes
`.ref` as two u16 per shot (`true_obs`, `sw_obs`, `:2194–2210`). Add a third u16 carrying
`sw_valid` (from `decode_fixed_ehat`'s third return) packed with the iteration count. Bump a
version byte in the emitted header so a stale `.ref` cannot be silently misread by the new driver
— the same footgun class as #478.

**`crates/aleph-qec/examples/qec_q7_nonconv.rs`** (new) — the campaign. Positional args
`mode(block|window) rounds shots seed`, sweeping the mode's operating points. Emits
`docs/perf/data/qec-q7-nonconv.csv` with, per point: `r`, `r_ci95`, `P(err|valid=0)`,
`P(err|valid=1)`, `A`, iteration percentiles, and for the window mode the per-window rate,
per-shot rate, `residual`, and `commit_clean` fractions. Writes the retained non-converged subset
to disk so Level-3 candidates run against a fixed corpus rather than re-sampling.

**`crates/aleph-qec/src/osd.rs`** — add the residual-restricted sweep variant (candidate 3).

**`crates/aleph-qec/src/fixed_bp.rs`** — `decode_fixed_osd` (`:541`) calls `correction_from_soft`
unconditionally and relies on `OsdDecoder` internally passing through already-valid inputs. For a
policy measurement that implicit gating must not be depended on: add an explicit
`if !soft.converged` guard so the measured tail cost is the real tail cost.

### Board

**`hw/sw/bp_stream_banked_ler_kv260.py`** — `run_chunk` (`:99`) masks `(word >> 20) & obs_mask`
and discards the rest. Also capture `(word >> 19) & 1` (`valid_flag`) and `word & 0xFFFF`
(`latency_cycles`), and add `rtl_valid` / `valid_mismatch` columns to the per-point report. The
status word layout is `hw/bp_stream_banked_core.sv:28`; `hw/sw/bp_stream_banked_kv260.py:216`
already reads bit 19 the same way and is the reference for the change.

No RTL change. No new bitstream.

## Verification

**Software.** Unit test that `valid` from `decode_fixed_ehat`, `converged` from
`decode_fixed_soft`, and `iters_to_valid`'s bool agree on the same syndromes — three APIs that
must not drift. Round-trip test for the extended `.ref` format (write, read, compare, and confirm
the version byte rejects the old layout). Existing gates unchanged: `make -C hw bpbanked`,
`bpbanked-highweight`, `bpstreambanked`.

**Board confirmation** — deliberately minimal, since bit-exactness is already proven at 10⁶ × 3:

1. Pre-check that the board is reachable and the matched p=0.005 overlay is still resident.
2. Regenerate `.syn`/`.ref` at p=0.005 with the new third u16.
3. Run the patched driver at ~10⁵ shots against that overlay.
4. **Gate: `rtl_valid == sw_valid` on 100 % of shots, 0 mismatches.**

That gate is the only new hardware claim in the ticket. If the board or overlay is unavailable,
it degrades to a Verilator co-sim gate on the same vectors — no re-synthesis is in scope.

## Deliverables

- `crates/aleph-qec/examples/qec_q7_nonconv.rs` + `docs/perf/data/qec-q7-nonconv.csv`
- The emitter, OSD, `fixed_bp` and driver changes above
- `docs/qec/q7-07-nonconvergence-policy.md` — rates per operating point, attributable fractions,
  candidate table with LER and cost columns, the chosen policy, and the latency arithmetic
- PR `[Q7-07] Non-convergence fallback policy (valid_flag=0 path)`, body `Closes #458`

## Open risk

The pre-registered rule may well land on do-nothing-but-flag, in which case the ticket's output is
data and a documented rejection rather than a new decoder feature. That is an explicitly
acceptable AC-2 outcome ("incl. do-nothing-but-flag if rates are negligible") and is the reason
Level 2 exists: it converts "we tried some candidates and none helped" into "no candidate *can*
help by more than `A(p)`", which is a much stronger statement.
