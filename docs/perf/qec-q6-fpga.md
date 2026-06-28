# Phase Q6 — FPGA decoder: utilization, Fmax, latency

Synthesis results for the surface-code Union-Find decoder (`hw/uf_surface_decoder.sv`, the Q6-04
sequential FSM) on both target boards. Flow: `hw/syn/` (non-project out-of-context). This document is
the shared home for Q6-05 (d=3 synth), Q6-09 (d=5 scaling), and Q6-03 (GPU-vs-FPGA comparison).

**Hosts:** synthesis on `openwebgui` (Vivado, x86 Linux). Sim baseline (Verilator) on the M4 Mac.

## Target parts

| board | part | LUT | FF | BRAM36 | DSP | role |
|-------|------|-----|----|--------|----|------|
| Digilent Zybo Z7-20 | `xc7z020clg400-1` | 53 200 | 106 400 | 140 | 220 | small part — d=5 fit risk lives here |
| Xilinx Kria KV260 | `xck26-sfvc784-2LV-c` | ~256 200 | ~512 400 | 144 | 1 248 | headroom part |

## Q6-05 — d=3, sequential FSM (33-cycle decode)

Vivado 2024.2 on `openwebgui`, non-project out-of-context flow (`hw/syn/run.sh`), implemented
(synth → place → route). Numbers from `reports/{zybo,kv260}/{util_impl.rpt,fmax.txt}`.

| part | LUT | FF | BRAM36 | DSP | Fmax | decode latency (33 clk) | fits 1 µs? |
|------|-----|----|--------|----|------|--------------------------|------------|
| `xc7z020clg400-1` (Zybo) | 1178 (2.21%) | 268 (0.25%) | 0 | 0 | **58.7 MHz** (WNS −12.04 ns @ 200 MHz tgt) | **562 ns** | ✅ |
| `xck26-sfvc784-2LV-c` (KV260) | 1200 (1.02%) | 268 (0.11%) | 0 | 0 | **170.0 MHz** (WNS −2.88 ns @ 333 MHz tgt) | **194 ns** | ✅ |

**Budget check:** the surface-code round budget is ~1 µs; latency = `33 / Fmax`. Zybo
33 × 17.04 ns = **562 ns**, KV260 33 × 5.88 ns = **194 ns** — both within budget at d=3.

**Fit verdict (d=3):** the decoder is tiny — **~1.2k LUT, ~268 FF, zero BRAM, zero DSP** — so it
fits with enormous headroom on *both* parts (2.2% of the small XC7Z020; 1.0% of the XCK26). Fit is a
non-issue for d=3; d=5 (a larger matching graph) will grow LUTs but stays far inside both parts.

**Caveat — Fmax, not area, is the wall.** Neither part met its aggressive target (200/333 MHz):
WNS is negative, so the closed Fmax is 58.7 / 170 MHz. The critical paths are the long
combinational chains *inside* each FSM cycle — chiefly the peel sweep's 18-edge loop-carried update
and the union-find root-walk (depth N). They still clear the 1 µs budget at d=3, but **pipelining
those passes is the lever** for higher Fmax / margin and for d≥5 (tracked under Q6-09 and follow-on
timing work). Area is nowhere near a constraint; latency-per-cycle-depth is.

## Q6-06 — gate-level sign-off (xsim)

The same self-checking SV testbench (`hw/tb_uf_surface_xsim.sv`) replays all 256 syndromes against
three elaborations of the decoder; run via `hw/syn/gatesim.sh <part-dir>` on `openwebgui`. Every
stage must bit-match the frozen Q6-02 golden table, reproduce the syndrome, and drive no X.

| stage | what it catches | Zybo | KV260 |
|-------|-----------------|------|-------|
| behavioral RTL | TB sanity | PASS | PASS |
| **post-route functional netlist** | synth/sim mismatch, inferred latches | PASS | PASS |
| **post-route timing (SDF, 50 MHz)** | X-prop / setup at real cell delays | PASS | PASS |

All: **256/256 bit-match golden, valid, zero X** on both parts. The routed netlist behaves
bit-identically to the RTL and holds up under back-annotated cell delays — **no RTL fixes were
needed** for sim/synth parity (the Verilator TB stays green). Two sim-infra fixes folded in: compile
RTL and TB in separate compilation units (both `include the graph header), and hold reset past the
gate-level `glbl` GSR window (~100 ns) before the first decode. Timing sim is clocked at 50 MHz —
below both parts' closed Fmax (58.7 / 170 MHz) — so it reflects cell delays without spurious
over-clock violations.

## Q6-09 — d=5 scaling

The decoder is parametric in the generated matching graph, so d=5 is the same RTL with a larger graph
(`hw/uf_surface_graph_d5.svh`: N=25 / 24 detectors / M=54 edges, vs d=3's N=9 / M=18).

**Verification** (`hw/tb_uf_surface_scale.cpp`, `make -C hw surf-d5`). 2^24 syndromes can't be
enumerated, so we check every weight-1 and weight-2 error pattern (a distance-5 code must correct ≤2
errors):

| | validity | weight-1 (distance) | weight-≤2 logical errors |
|---|---|---|---|
| d=3 (cross-check) | 0 fail | 0 fail | 40 / 153 (d=3 corrects only weight-1) |
| **d=5** | **0 fail** | **0 fail** | **0 / 1431** ⇒ corrects *every* ≤2-error fault |

**Synthesis** (both parts, OOC, Vivado 2024.2):

| part | d | LUT | FF | Fmax | worst-case latency | budget (1 µs) |
|------|---|-----|----|------|--------------------|---------------|
| Zybo `xc7z020` | 3 | 1178 (2.2%) | 268 | 58.7 MHz | 47 clk → **801 ns** | ✅ |
| Zybo `xc7z020` | 5 | 6139 (11.5%) | 754 | 15.9 MHz | 109 clk → **6.86 µs** | ❌ |
| KV260 `xck26` | 3 | 1200 (1.0%) | 268 | 170 MHz | 47 clk → **276 ns** | ✅ |
| KV260 `xck26` | 5 | 6427 (5.5%) | 754 | 38.0 MHz | 109 clk → **2.87 µs** | ❌ |

(Worst-case latency is over all weight-≤2 syndromes; the "33 clk" quoted under Q6-05 was the
empty-syndrome *best* case. Latency is syndrome-dependent: more growth rounds ⇒ more cycles.)

**Scaling verdict.** d=3 meets the ~1 µs round budget on both boards. **d=5 misses it on both** with
the current single-bounded-pass-per-cycle FSM. Area is never the limit (d=5 is 11.5 % of the small
XC7Z020). The wall is two compounding factors as the graph grows: **Fmax collapses** ~3.7–4.5×
(58.7→15.9 MHz, 170→38 MHz — the per-cycle combinational chains are O(M): the 54-edge loop-carried
peel sweep and the CC/forest passes over 25 nodes) **and the cycle count grows** ~2.3× (47→109). The
fix is the lever flagged since Q6-05: **pipeline the per-cycle passes** — split the O(M) peel/CC
chains into bounded-depth sub-steps to restore Fmax — which is the precondition for real-time d≥5.
Max code distance per board *within budget*, as-is: **d=3 on both** (KV260 has ~3.6× the timing
headroom of Zybo, so it is the path to d=5 once pipelining lands).

## Q6-10 — pipelining attempt: parallel peel + the real critical path

Q6-09 named the per-cycle O(M) chains as the Fmax wall. Q6-10 rewrites the **peel** sweep from a
loop-carried M-edge chain (one cycle, O(M) depth) to a **parallel leaf-strip** — each round peels all
current non-boundary leaves at once via associative count/XOR reductions (bit-equivalent: the
per-edge correction is the leaf-side subtree parity, peel-order-independent; the d=3 golden still
matches, d=5 still corrects 0/1431).

Before (loop-carried peel) → after (parallel peel):

| part | d | LUT before→after | Fmax before→after |
|------|---|------------------|-------------------|
| Zybo `xc7z020` | 3 | 1178 → **872** (−26%) | 58.7 → 62.4 MHz |
| Zybo `xc7z020` | 5 | 6139 → 6061 | 15.9 → **15.4 MHz** (unchanged) |
| KV260 `xck26` | 3 | 1200 → 876 | 170 → 173.6 MHz |
| KV260 `xck26` | 5 | 6427 → 6111 | 38.0 → **38.2 MHz** (unchanged) |

**Finding: the peel was *not* the binding critical path.** Freeing it dropped LUTs ~26% at d=3 but
left d=5 Fmax flat. The d=5 post-route worst path is unambiguous:

```
Source: e_idx_reg → Destination: troot_reg[24]   (the spanning-forest phase)
Logic Levels: 78  (LUT6=62 …)   Data path 64.6 ns
```

That is the **union-find root-walk** in `S_FOREST` (`for k in 0..N: ra = troot[ra]`, done for both
endpoints): an N-deep chain of index-muxed `troot` lookups, unrolled to ~78 LUT levels at d=5. It is
a *true* serial dependency (each hop needs the previous lookup), so it cannot be balanced like the
peel reductions. **This is the actual wall for d≥5 Fmax**, and removing it needs a bounded-depth /
parallel union-find for the forest (path-compression/union-by-rank across cycles, or the parallel
cluster architecture of Liyanage et al., ArXiv:2301.08419) — a real redesign, not a chain rewrite.

Net: the parallel peel ships as a correctness-preserving area win and removes one chain that *would*
become critical once the find is fixed; the d≥5 real-time lever is now precisely targeted (the forest
find) and tracked in #372.

## Q6-03 — GPU vs FPGA

> Pending board bring-up (Q6-08) for measured on-board latency/throughput/power vs the Q3 GPU decoder.
