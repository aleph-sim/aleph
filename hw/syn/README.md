# `hw/syn/` — Q6-05 Vivado dual-target synthesis flow

Non-project, **out-of-context** synth + implementation of `uf_surface_decoder` for **both** target
parts, producing utilization + Fmax. Board-independent — needs only Vivado on an x86 Linux host (we
run it on `openwebgui`), no physical board.

| file | role |
|------|------|
| `synth.tcl` | OOC synth → opt/place/route → utilization + timing reports + a one-line Fmax. Parameterised by `<part> <xdc> <outdir>`. |
| `zybo_z7_20.xdc` | clock constraint for the Zybo Z7-20 part (`xc7z020clg400-1`, 200 MHz target). |
| `kv260.xdc` | clock constraint for the Kria KV260 part (`xck26-sfvc784-2LV-c`, 333 MHz target). |
| `run.sh` | runs both parts and prints the Fmax + utilization summary. |

## Run

```bash
# on openwebgui (Vivado on PATH):
source /tools/Xilinx/Vivado/2024.2/settings64.sh
hw/syn/run.sh
```

Outputs land in `hw/syn/reports/{zybo,kv260}/` (git-ignored): `util_impl.rpt`,
`timing_impl.rpt`, `fmax.txt`, and routed/synth checkpoints.

## Why out-of-context

This is a **fit + Fmax** study of the decoder PL block before board bring-up. OOC skips I/O buffer
insertion and pin placement (the wide `correction`/`syndrome` ports become AXI in Q6-07, not chip
pins), so utilization reflects the core logic and timing reflects the internal critical path. Real
board pin constraints arrive with on-board bring-up (Q6-08).

## What the numbers decide

- **Fit:** does d=3 (then d=5 in Q6-09) fit the small part — XC7Z020 has 53 200 LUT / 140 BRAM36 /
  220 DSP. The KV260 K26 (~256k LUT) is the headroom reference.
- **Latency:** the Q6-04 FSM takes 33 cycles for d=3; `latency_ns = 33 / Fmax`, checked against the
  ~1 µs surface-code round budget.

Results are recorded in `docs/perf/qec-q6-fpga.md`.
