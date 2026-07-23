# Q7-06 AC-1 — batched AXI-DMA decoder path (on-silicon results)

**Status: AC-1 throughput target MET.** RTL + both KV260 overlays (full-schedule and early-exit) built
and validated on-silicon; the early-exit overlay reaches **163×** the per-word harness throughput
(≥100× cleared). Part of #457.

## What AC-1 is

Replace the M8 per-word AXI-Lite runner (`hw/sw/bp_circ_kv260.py`, one experiment = NS syndrome-word
MMIO writes + a Python `START`/poll loop + correction/obs reads) with a **batched AXI-DMA path**: a whole
batch of independent syndrome→result experiments streams through one DMA transfer. Target: **≥100×
harness throughput** over the per-word runner, and the batched-duty measurement that tightens the Q7-05
µJ/decode bound.

## Design

`hw/bp_stream_banked_core.sv` (board top `hw/bp_stream_banked.v`) — an AXI4-Stream batch shell around the
banked block decoder `bp_relay_banked`:

- **Input:** NS = ⌈BP_C/32⌉ = 5 MM2S beats per experiment (syndrome bits `[i*32 +: 32]`); the low BP_C
  bits of the assembled word feed the decoder.
- **Output:** one 32-bit S2MM status word per experiment — `[31:20]=obs_flip[11:0]`, `[19]=valid_flag`,
  `[15:0]=latency_cycles` (LER needs only the observable flips, not the 864-bit correction).
- Simpler than the sliding-window shell `bp_stream_win_core`: the block decoder is stateless between
  decodes, so there is **no per-frame reset and no drain-tail FIFO** — just a depth-2 output FIFO for
  S2MM back-pressure and a 1-deep input gate. `tlast` marks the batch end.

`hw/syn/kv260_bp_stream_banked_bd.tcl` — KV260 (Zynq UltraScale+) block design: `M_AXI_HPM0_FPD` →
AXI-DMA `S_AXI_LITE` (control); DMA `MM2S`/`S2MM` → `S_AXI_HP0_FPD` (SAXIGP2) → PS DDR (data); DMA
`M_AXIS_MM2S` → shell → DMA `S_AXIS_S2MM` (stream). `early_exit` is a build-time constant (arg 6:
0 = full schedule, 1 = product/early-exit mode).

`hw/sw/bp_stream_banked_kv260.py` — batched driver. Programs the PL (Overlay with a placeholder sidecar
`.xclbin` to dodge the Kria-PYNQ 3.0.1 stub-xclbinutil bug; raw-MMIO fallback), decodes the 40-shot
circuit golden as one batch (bit-exact gate), then benches experiments/sec.

## Verification

- **Verilator co-sim** (`make -C hw bpstreambanked`): 40/40 bit-identical to the golden as one batched
  AXIS transfer, exact `tlast` framing, back-pressure invariant.
- **On-silicon (KV260, full-schedule overlay, 100 MHz, WNS +1.079 ns):** 40/40 batched decodes match the
  golden. Confirmed at batch sizes n = 1, 2, 4, 40 and 20 000. The same `bp_relay_banked` core in the M8
  AXI-Lite overlay also passes 40/40, cross-checking the core independent of the DMA path.

## Throughput (KV260, 100 MHz, measured on-silicon)

| path | mode | exp/s | µs/exp | speedup vs per-word (same mode) |
|---|---|---|---|---|
| per-word AXI-Lite (`bp_circ_kv260.py`, `bp_m8.bit`) | full | 3 018 | 331 | 1× (Python+MMIO bound, ~94 % harness overhead) |
| per-word AXI-Lite | early | 3 392 | 295 | 1× (still harness-bound) |
| batched DMA (`bp_stream_banked_kv260.py`) | full | 43 380 | 23.1 | **14.4×** — hardware-decode-bound (2085 cyc = 20.85 µs) |
| **batched DMA** | **early** | **553 000** | **1.81** | **163× — ≥100× MET** (183× vs per-word full) |

**Reading it.** Batching removes ~99 % of the per-experiment Python/MMIO harness overhead in both modes;
the batched path's µs/exp then equals the raw hardware decode latency (23 µs ≈ the 2085-cycle full
schedule; 1.81 µs ≈ the ~153-cycle early-exit mean). So the *mode* sets the throughput ceiling once the
harness is out of the way:

- **Full schedule is decode-latency-bound at 14.4×** — 1/20.85 µs ≈ 48 k exp/s is the hard ceiling for a
  single, non-pipelined-across-experiments core; ≥100× is unreachable there because each decode itself
  costs 20.85 µs (the harness was never the *only* bottleneck at full schedule).
- **Early-exit (the product mode, spec D4) clears ≥100× with margin: 553 k exp/s = 163×** over the
  per-word early-exit baseline. This is the AC-1 result — batching converts the per-word runner's
  harness-bound rate into a hardware-decode-bound rate, and in the deployment mode that is a 163× gain.

Correctness on the early-exit overlay is 40/40 vs the *full-schedule* golden (the two agree on obs for all
40 sub-threshold shots); a strict early-exit gate would use the first-valid golden (`circvectorsearly`).

## Reproduce

```
# build (EPYC + Vivado 2024.2), full then early-exit:
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100            # full
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100 full default 1  # early-exit
# on the KV260 (root, pynq venv + XRT), from ~/q7stream with the .bit + a placeholder .xclbin sidecar:
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     bp_stream_banked_kv260.py bp_kv260_stream_banked.bit bp_circ_vectors.txt --bench-batch 20000
```
