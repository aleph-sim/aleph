# Q7-02 M8 — squeeze the KV260: pipeline the banked core, rehabilitate 16/48

**Status:** design (2026-07-11). Continues the Q7-02 FPGA relay-BP track (Advances #322).
**Predecessor:** M7 (PR #453) shipped `bp_relay_banked` (β-split check-major LUTRAM banking, W/V=12/36)
on KV260 silicon: **32.8 µs worst-case / 2.5 µs early-exit mean @75 MHz**, 40/40 bit-exact.

## 1. Motivation & the two corrections M8 starts from

1. **The M7 DSP verdict was wrong.** M7's OOC tcl counted `REF_NAME =~ DSP*`, which matches the ~9
   sub-primitives a DSP48E2 macro expands into. Real usage (hierarchical report): 8/24 → **82 DSP
   (6.6 %)**, 12/36 → **124 (10 %)**, 16/48 → **164 (13 %)**. The 16/48 config was never DSP-bound —
   its only constraint is **LUT 90 %** (105.3 k / 117 120). It is back on the table.
2. **The M7 clock left ~1.4× on the table by design.** The measured critical path (bank RAMD32 read →
   gather mux → `check_minsum` stage-1 tournament, 10.77 ns OOC / ~12.6 ns post-route) is one long
   unregistered chain. Splitting it is a pure re-timing — values unchanged.

## 2. Goal

Minimize decode latency on the SAME silicon and precision envelope. Honest target band:
**~17–22 µs worst-case** (from 32.8) and **~1.5–1.6 µs early-exit mean** (from 2.5), decided by which
(config, FCLK) closes post-route timing. Correctness bar unchanged: **bit-exact in VALUES to the same
`FixedRelayBp` golden** — cycle counts change (documented; the TB and doc record the new numbers).

## 3. Levers (all four; L1/L3 are the RTL work)

- **L1 — bank-output pipeline registers.** Register `qmcm`, `qa_ecm`/`qb_ecm`, `qmvm` before the
  gathers. Submodule operands arrive one cycle later → the FSM's scatter lag moves `pc−2 → pc−3` and
  every phase's drain window grows by 1. RMW safety holds: m_vm reads row `pc` while writing row
  `pc−3` — still disjoint groups in the software pipeline; m_cm/e_cm keep their read-phase/write-phase
  separation.
- **L3 — 3-stage `check_minsum`.** Split the 5-level tournament tree with a register plane after
  level 3. **Parameterized `STAGES` (default 2)** so `bp_relay_unroll_pipe` / `bp_unroll_skeleton`
  (which hard-code the 2-cycle latency) are untouched; `bp_relay_banked` instantiates `STAGES=3`.
  `var_update` stays 2-stage (its adder tree and blend paths are shallow; the blend multiply lives in
  a DSP). With L1+L3 the expected worst path is ~5–6 ns OOC → board FCLK candidates on the PS grid:
  **125 / 115.4 / 107.1 MHz** (1500/12, /13, /14).
- **L4 — 16/48 rehabilitation.** OOC-probe and board-build 16/48 alongside 12/36 with L1+L3. Cycle
  model (to confirm in co-sim): 16/48 ≈ **~2 300 cyc** (GC=9, GV=18, deeper drains), 12/36 ≈ ~2 840.
  16/48 at 90 % LUT is a real congestion risk — post-route decides; 12/36 is the fallback. Pick the
  fastest (config × FCLK) that closes.
- **L2 — impl strategy.** The board tcl gains a strategy tclarg: default first; on TIMING_VIOLATED
  retry `Performance_Explore` before stepping the FCLK down one grid notch.

Latency arithmetic at the candidates: 16/48 @125 → **18.4 µs**, @107 → 21.5 µs; 12/36 @125 → 22.7 µs,
@107 → 26.5 µs. Early-exit mean scales with cycles/iter (~176 cyc at 16/48) → **~1.4–1.6 µs**.

## 4. Tooling & record fixes (part of the milestone)

- `hw/syn/ooc_banked.tcl`: count DSPs as `REF_NAME == DSP48E2` (exact), keep the old cell count out of
  the RESULT line. Keep `-hierarchical` reporting.
- `docs/perf/qec-q7-fixed-bp.md`: a correction note in the M7 section (DSP figures ÷9; "16/48 no-fit
  (DSP)" → "LUT-bound at 90 %, rehabilitated in M8") + the eventual M8 section. The correction ships
  with M8's PR — the merged record must not stay wrong.

## 5. Correctness & verification (M7 discipline, unchanged)

- Both RTL levers are pure re-timings: same arithmetic, same order, same widths — **bit-exact in
  values**; only latency counts move. Any decode-result diff = bug.
- Gates: `checkminsum` TB extended to run **both STAGES=2 and STAGES=3** (10 000 vectors each);
  `bpbanked` tri-config co-sim 40/40 (Mac 8/24; EPYC all three — new cycle counts recorded);
  `bpaxibanked` AXI gate; regressions `bpunrollpipe` (STAGES=2 path) + `bpbramdp`; the elaboration
  guards stay.
- Synthesis: OOC probes (both configs, L1+L3) at 5 ns with the fixed DSP count → board builds at the
  highest closing PS-grid FCLK (L2 escalation) → silicon via the M7 runner (`--idcode 0x42500003`,
  both modes) → doc + PR `Advances #322`.

## 6. Deliverables

- RTL: `hw/bp_relay_banked.sv` (L1 + FSM lag updates), `hw/check_minsum.sv` (STAGES param + 3-stage
  plane), Makefile gate updates.
- Synth: fixed `ooc_banked.tcl`, board tcl strategy arg; probe + build results for both configs.
- Silicon numbers (both modes) at the chosen config/FCLK.
- Doc: M7 correction + M8 section; PR `[Q7-02] M8: …`, **Advances #322**.

## 7. Honest scope & risks

- 90 % LUT may simply not route at speed — then M8 ships 12/36 with L1+L3 (~23–27 µs) and the honest
  finding. Either way the DSP record gets corrected.
- L1+L3 shift every FSM lag by up to 2 cycles — the most delicate RTL of the milestone; the co-sim
  gate (bit-exact values + expected new latencies) is the net, exactly as it caught nothing-by-luck
  in M7 (all schedule changes were caught by the golden).
- Post-route degradation ate 91→75 MHz in M7; the same ~15 % haircut is priced into the target band
  (worst path ~6 ns OOC → ~7 ns routed → 125–107 MHz board). If even 107 fails post-route, the floor
  is the M7 result at 75 — M8 cannot regress silicon, only decline to improve it.
