# Q7-06 AC-1 — batched AXI-DMA decoder path (on-silicon results)

**Status: AC-1 throughput target MET** (163× per-word harness throughput, ≥100× cleared) on both KV260
overlays. **AC-2 (10⁶-shot LER) ran and surfaced a real synth-vs-sim divergence in the banked core** on
high-weight syndromes — bit-exact at p=0.003, RTL LER ~7–13 % worse at p≥0.005; isolated to
`bp_relay_banked` (not the DMA wrapper, not timing, not the RTL-as-simulated), so AC-2's within-CI gate
is not yet met and root-cause is scoped follow-up. Part of #457.

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

## AC-2 — 10⁶-shot on-silicon LER campaign (a real divergence found)

The campaign streams real DEM shots through the batched overlay and compares the RTL logical-error rate
to the software `FixedRelayBp` golden. Harness: the `silvectors` emitter (binary `.syn`/`.ref`) +
`hw/sw/bp_stream_banked_ler_kv260.py`. Run at **10⁶ shots × 3 circuit-level rates** (rounds=1 vehicle):

| point | n | sw LER | RTL LER | \|diff\| | comb 95% CI | divergence (rtl≠sw) | verdict |
|---|---|---|---|---|---|---|---|
| p=0.003 | 10⁶ | 8.32e-4 | 8.32e-4 | 0 | 1.1e-4 | **0 / 10⁶** | PASS |
| p=0.005 | 10⁶ | 7.05e-3 | 7.53e-3 | 4.8e-4 | 3.3e-4 | 7 067 / 10⁶ | FAIL |
| p=0.007 | 10⁶ | 2.88e-2 | 3.26e-2 | 3.8e-3 | 6.8e-4 | 30 703 / 10⁶ | FAIL |

**AC-2 is not met at p ≥ 0.005**: the silicon RTL is bit-exact to the software golden on low-weight
syndromes but diverges on high-weight ones (RTL LER ~7–13 % worse), and the divergence fraction grows
with the physical error rate. Every prior co-sim used a p=0.003 golden, so this is the first test that
exercised high-weight syndromes — and it surfaced a real divergence.

**Root-cause characterization** (four isolating experiments — the finding, not a guess):

1. **Core, not the DMA wrapper.** The same `bp_relay_banked` core in the M8 per-word AXI-Lite overlay
   (`bp_m8.bit`, no DMA path) diverges from software identically (200 / 30 000 on the p=0.005 subset).
   The AC-1 batched path is faithful; the divergence is upstream of it.
2. **Not the RTL design.** A Verilator co-sim of `bp_relay_banked` against the software golden at p=0.005
   is **2000 / 2000 bit-identical** — the RTL *as simulated* matches software even on high-weight shots.
3. **Not timing.** Re-running the p=0.005 subset at 100 / 77 / 50 / 25 MHz (PYNQ `Clocks.fclk0_mhz`)
   gives an **identical** divergence count at every clock. A setup violation would shrink as the clock
   slows; this is flat → not a timing path (consistent with the +1.08 ns WNS being real).
4. **Deterministic.** Two full 10⁶-shot runs give bit-identical error/divergence counts.

Together these point to a **synthesis-vs-simulation logic mismatch** in `bp_relay_banked`: Vivado
synthesizes it into hardware that computes a different result than Verilator simulates the same RTL,
deterministically, only when operand magnitudes are large — i.e. most likely in the **fixed-point
saturation / accumulator-width handling**, which high-weight syndromes (more simultaneously-firing
checks → larger accumulated magnitudes) exercise at its edges while p=0.003 never does.

**Follow-up (own issue):** audit the banked core's accumulator/saturation widths against the
`FixedRelayBp` reference arithmetic; reproduce with a **post-synth gate-level** sim on the high-weight
vectors (Verilator RTL sim cannot see it); fix; re-run the campaign. This is a pre-existing M7/M8 banked-
core issue exposed by the campaign, independent of the Q7-06 AC-1 batched-DMA work. The AC-2 harness
(`silvectors` + the LER driver) is in place and ready to re-verify once the core is fixed.

## Reproduce

```
# AC-2 vectors (per point) + campaign:
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 <p> 1000000 2024 <prefix>
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     bp_stream_banked_ler_kv260.py bp_kv260_stream_banked.bit p003 p005 p007
```

```
# build (EPYC + Vivado 2024.2), full then early-exit:
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100            # full
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100 full default 1  # early-exit
# on the KV260 (root, pynq venv + XRT), from ~/q7stream with the .bit + a placeholder .xclbin sidecar:
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     bp_stream_banked_kv260.py bp_kv260_stream_banked.bit bp_circ_vectors.txt --bench-batch 20000
```
