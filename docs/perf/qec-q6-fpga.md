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

> Pending Q6-09 (d=5 graph + re-synth). Will add d=5 rows for both parts and the max-distance-per-board
> verdict.

## Q6-03 — GPU vs FPGA

> Pending board bring-up (Q6-08) for measured on-board latency/throughput/power vs the Q3 GPU decoder.
