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

> **Pending the first Vivado run on `openwebgui`** (`hw/syn/run.sh`). Table filled from
> `reports/{zybo,kv260}/util_impl.rpt` + `fmax.txt`.

| part | LUT | FF | BRAM36 | DSP | Fmax | decode latency (33 clk) | fits? |
|------|-----|----|--------|----|------|--------------------------|-------|
| `xc7z020clg400-1` (Zybo) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ ns | _TBD_ |
| `xck26-sfvc784-2LV-c` (KV260) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ ns | _TBD_ |

**Budget check:** the surface-code round budget is ~1 µs. With a 33-cycle decode, the per-round
latency is `33 / Fmax`; e.g. at 100 MHz → 330 ns (within budget), at 200 MHz → 165 ns. Confirm
against the measured Fmax above.

**Fit verdict (d=3):** _TBD after run._

## Q6-09 — d=5 scaling

> Pending Q6-09 (d=5 graph + re-synth). Will add d=5 rows for both parts and the max-distance-per-board
> verdict.

## Q6-03 — GPU vs FPGA

> Pending board bring-up (Q6-08) for measured on-board latency/throughput/power vs the Q3 GPU decoder.
