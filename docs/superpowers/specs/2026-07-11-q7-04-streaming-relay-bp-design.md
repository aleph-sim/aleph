# Q7-04 M9 — multi-round sliding-window streaming relay-BP on FPGA (real-time M9)

**Status:** design (2026-07-11). Issue #455 (`docs/qec/BACKLOG.md` § Q7-04).
**Predecessors:** M8 banked core (PR #454: 15.64 µs worst / 0.85 µs median on KV260, rounds=1
single-shot); surface-code UF streaming (Q6-20/22/23, DONE: `sliding.rs`, `uf_streaming_decoder.sv`,
`uf_stream_win_core.sv`, 2.87 µs/window sustained on Arty).

## 1. Problem

The banked relay-BP core decodes one `rounds=1` circuit-level batch per START. Real-time QEC is a
continuous stream of measurement rounds (~1 µs/round) decoded in a sliding window with bounded
per-round latency. Everything needed exists in two unconnected halves:

- **Fast silicon core, no windowing:** `bp_relay_banked.sv` + graph-generic AXI wrapper + KV260
  runner, keyed on a baked rounds=1 hypergraph DEM (`bb_circuit_tanner.svh`: N=864, C=144, E=2952,
  16/48 banking = 88 % LUT).
- **Validated windowing, wrong decoder:** `sliding.rs` residual-carry W/C schedule + the streaming
  RTL wrapper pattern — hardwired to Union-Find over *graphlike* (surface) DEMs.

M9 bridges them: the BB-code analog of the UF streaming decoder, with the M8 core unchanged inside.

## 2. Decisions taken (brainstorm 2026-07-11)

1. **(W, C) chosen by data, not heuristic.** The surface rule W=3d/C=d (→ W=36 at d=12) cannot fit:
   the window graph scales ~W× and rounds=1 already eats 88 % LUT. A software LER sweep over small
   (W, C) picks the smallest acceptable config; only that goes to RTL.
2. **Seam state decided by the same sweep.** Residual-only (UF-style) vs residual + soft priors
   (BP posteriors of buffer rounds seeding the next window) are both implemented in the software
   golden; RTL carries the winner. Soft priors ride only on a clear LER win.
3. **Three staged PRs** under issue #455, mirroring M6→M7→M8: **M9a** software golden + sweep
   (AC-1) → **M9b** emitter + streaming RTL + bit-exact co-sim (AC-2) → **M9c** KV260 silicon +
   sustained measurement (AC-3).
4. **PL-side streaming wrapper** (exact `uf_streaming_decoder.sv` analog) around the UNCHANGED
   banked core; AXI-Stream/DMA front-end. PS stays out of the per-window loop so the sustained
   number measures the decoder, not the host (the Q7-06 lesson: AXI-Lite harness dominates).

## 3. Two BP-specific deltas vs the UF pattern

- **Hypergraph time cut.** UF cuts out-of-window edges with temporal-sink *nodes* (a matching graph
  needs somewhere to route to). A DEM error mechanism touches an arbitrary detector set, so the BP
  window graph instead **truncates each straddling mechanism's detector set to the in-window
  detectors** (open temporal boundary, standard for sliding-window BP/LDPC). Past cut: rounds that
  slid off no longer exist; the residual carries their unresolved seam detectors forward.
- **Commit on error-vars, not matching edges.** Commit rule: an error-var whose **earliest detector
  round < commit boundary** is committed — its observable mask XORs into the running logical, its
  detector set toggles the residual. The trailing W−C buffer rounds stay lit and are re-decoded by
  the next window with fresh future context. The UF validity-drain gate does not transfer (BP may
  not satisfy the syndrome); the co-sim gate is **bit-exactness** instead (§ 5-M9b).

## 4. The plan, per stage

### M9a — software golden + (W, C) sweep → AC-1

New module `crates/aleph-qec/src/relay_window.rs`: `SlidingWindowBp` — residual-carry sliding
window over a multi-round circuit-level BB DEM, base decoder `FixedRelayBp` (the same bit-exact
fixed-point pipeline the RTL implements; LEGS=6 × ITERS=10, Q5.3).

- Inputs already exist: `BBCode::circuit_level_dem(rounds, noise)` (any rounds ≥ 1) and
  `BBMemoryExperiment::detector_rounds()` give the multi-round DEM + per-detector round
  coordinates. The emitter (`qec_q7_bp_graph.rs`) already plumbs `rounds`, baking 1 today.
- Interior-window translation invariance → one compiled window DEM serves every steady-state
  window (the property that makes one baked RTL header possible).
- Boundary handling: interior windows + a boundary-aware final window that commits everything to
  the end (the `qec_q6_stream_ler.rs` software-baseline convention).
- Non-convergence inside a window (valid_flag=0): commit the best-kept decision anyway + flag.
  Policy tuning is Q7-07; M9 only plumbs the flag through to the output.
- **Sweep (EPYC):** (W, C) ∈ {(3,1), (4,2), (6,2), (6,3)} × seam ∈ {residual-only,
  residual+soft-priors} × p ∈ {0.001, 0.003, 0.005}, rounds=12 memory-X experiment, LER vs the
  batch `FixedRelayBp` decode of the **same shots** (difference = windowing cost, not sampling
  noise). Pick the smallest (W, C, seam) whose LER stays within the batch curve's CI (or a
  documented, explicitly-accepted gap).
- AC-1 evidence: sweep table + chosen config in the M9 section of `docs/perf/qec-q7-fixed-bp.md`.

### M9b — emitter + streaming RTL + co-sim → AC-2

- **Fit de-risk FIRST:** regenerate the window header for the chosen (W, C) at candidate bankings
  (8/24, then 4/12 if needed) and OOC-synth it **before writing the FSM**. rounds=1 at 16/48 is
  88 % LUT; the W-round graph needs a narrower config. If nothing fits at the sweep-chosen W, drop
  to the next (W, C) on the sweep table (documented trade).
- **Emitter** (`qec_q7_bp_graph.rs`, new modes):
  - `streamgraph rounds p bankW bankV W C` → window Tanner header (the existing BP_* CSR +
    banking-solve, now for the W-round window DEM) **plus streaming metadata**, the
    `qec_surface_uf_graph.rs` analog: `BP_DPR` (detectors/round = 72 for gross memory-X),
    `BP_SHIFT` (detector r → r−C slide map), `BP_LOAD_LO` (reload region), `BP_VAR_COMMIT`
    (1-bit per var: earliest detector round < C, i.e. this var commits), `BP_VAR_DET` (var →
    in-window detector toggle set; the commit path),
    plus the existing `BP_OBS_MASK`.
  - `streamvectors` → golden round-streams + per-window `{corr, obs, vflag}` decisions from the
    M9a software golden (interior windows), for the Verilator TB.
- **RTL** `bp_streaming_decoder.sv`: warm/run/commit/slide/reload FSM around the **unchanged**
  `bp_relay_banked` core (it just gets the window header): residual buffer over W·DPR detectors,
  combinational commit from `corr_out` through `BP_VAR_COMMIT`/`BP_VAR_DET`/`BP_OBS_MASK`,
  slide-by-C via `BP_SHIFT`, reload of C·DPR fresh bits. Early-exit input passes through.
- **AXI front-end** `bp_stream_win_core.sv` (lift of `uf_stream_win_core.sv`): one round =
  ⌈72/32⌉ = 3 MM2S beats (round-per-3-beats framing), output = one 32-bit S2MM word per committed
  window `{obs[12], vflag, spare, latency[16]}` — obs is the 12-bit committed-logical vector —
  1-deep result slot, tlast framing, per-frame re-arm (the Q6-20 mid-stream-resume bugfix
  transfers).
- **Co-sim gate (AC-2):** fixed-point BP is deterministic, so unlike UF (validity-drain because of
  tie-breaks) the gate is **bit-exact**: per-window `{corr committed, obs, vflag}` equal to the
  software windowed golden on 40 MC round-streams, plus back-pressure invariance (byte-identical
  output under random S2MM stalls) and frame independence (N frames back-to-back, no external
  reset), mirroring `tb_uf_stream_win.cpp`. Makefile targets `bpstream`, `bpstream-axi`, lint.

### M9c — KV260 silicon + sustained measurement → AC-3

- Block design: KV260 + AXI DMA (MM2S/S2MM, `arty_z7_dma_bd.tcl` pattern on `zynq_ultra_ps_e`),
  PL clock on the PS grid (start at M8's 133.332 MHz; step down if the wider residual/commit
  fabric costs timing).
- Driver `hw/sw/bp_stream_kv260.py` (raw `pynq.Bitstream` + MMIO/DMA, the M8 no-xclbin bypass):
  stream ≥10⁵-window MC round-streams end-to-end.
- **Measurements (AC-3):**
  - Sustained µs/window (one large decoder-bound transfer, setup amortized), full-schedule AND
    early-exit.
  - Per-round latency distribution (window latency ÷ C, plus the DMA-visible arrival-to-commit
    distribution).
  - **Max round rate with no unbounded backlog:** back-pressure is by construction (1-deep slot →
    `s_axis_tready` stalls the DMA; nothing is dropped); the measured sustained rate IS the max
    rate — reported as rounds/s = windows/s × C, against the ~1 µs/round target.
  - Bit-exact spot-check on-board: 40 golden streams, both modes, vs `streamvectors`.
- Results + honest verdict → `docs/perf/qec-q7-fixed-bp.md` § M9; BACKLOG AC boxes ticked.

## 5. Honest expectations (recorded up front)

M8 worst-case is 2085 cycles on the rounds=1 graph at 16/48. A W-round window at a narrower
banking plausibly lands at ~6–12 k cycles ≈ **45–90 µs/window** against a real-time budget of
C µs (at 1 µs/round, C=1–3). **Worst-case real-time at 1 MHz round rate is almost certainly out of
reach on the KV260** — early-exit medians may come close on quiet streams, and the deliverable is
the *architecture + the honestly measured max round rate*, which is exactly the Q7-01 de-risk
input (an ASIC buys ~an order of magnitude in clock and unrestricted banking width). If the
measured rate is embarrassing, that is a finding, not a failure — it prices the FPGA product
story and sharpens the ASIC case.

## 6. Correctness & verification summary

| gate | stage | criterion |
|------|-------|-----------|
| sweep validity | M9a | windowed LER within batch-decode CI (same shots) at chosen (W, C, seam) |
| golden determinism | M9a | `SlidingWindowBp` decode is a pure function of (stream, config) |
| header re-verify | M9b | emitter banking-solve asserts inverses on every regen (M7 discipline) |
| co-sim | M9b | bit-exact per-window `{corr, obs, vflag}` vs software golden, 40/40 |
| robustness | M9b | back-pressure invariance + frame independence (UF TB pattern) |
| silicon | M9c | 40/40 bit-exact on-board both modes; sustained + max-rate measured |

Out of scope: fallback policy beyond flagging (Q7-07), power (Q7-05), batched qualification
interface (Q7-06), parallel/even-odd windows (`parallel_window.rs` — the escape hatch if a single
streaming core's backlog math demands it; noted for Q7-01, not built here).
