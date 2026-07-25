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

## AC-2 — 10⁶-shot on-silicon LER campaign (a harness bug, since root-caused)

The campaign streams real DEM shots through the batched overlay and compares the RTL logical-error rate
to the software `FixedRelayBp` golden. Harness: the `silvectors` emitter (binary `.syn`/`.ref`) +
`hw/sw/bp_stream_banked_ler_kv260.py`. Run at **10⁶ shots × 3 circuit-level rates** (rounds=1 vehicle):

| point | n | sw LER | RTL LER | \|diff\| | comb 95% CI | divergence (rtl≠sw) | verdict |
|---|---|---|---|---|---|---|---|
| p=0.003 | 10⁶ | 8.32e-4 | 8.32e-4 | 0 | 1.1e-4 | **0 / 10⁶** | PASS |
| p=0.005 | 10⁶ | 7.05e-3 | 7.53e-3 | 4.8e-4 | 3.3e-4 | 7 067 / 10⁶ | FAIL |
| p=0.007 | 10⁶ | 2.88e-2 | 3.26e-2 | 3.8e-3 | 6.8e-4 | 30 703 / 10⁶ | FAIL |

**The p ≥ 0.005 rows above are not a decoder result** — they compare **two different decoders**, and the
table is kept only as the record of what the campaign measured before that was understood.

**Root cause (#478): the golden's priors did not match the bitstream's.** `FixedRelayBp` derives its
per-variable prior `λ_v` from the DEM's error probabilities, and the RTL bakes those priors into
`BP_LAMBDA` at header-generation time. The shipped overlay's header comes from `circgraph 1 0.003 16 48`
→ **λ(p=0.003)**. The campaign's golden came from `silvectors 1 <p> ...`, which built its decoder from a
DEM at that same `<p>` → **λ(p=0.005) / λ(p=0.007)**. Same Tanner graph, different priors ⇒ a different
message trajectory on syndromes hard enough for the trajectory to decide the outcome. Hence exactly the
observed pattern: 0 divergence at p=0.003 (priors agree), growing monotonically with |p − 0.003|,
deterministic, frequency-flat, and identically present in RTL, funcsim and silicon.

Two independent reproductions:

1. **The 24 enriched on-silicon-diverging shots** (dumped by `hw/sw/bp_stream_banked_enrich_kv260.py`),
   re-decoded at a sweep of decoder-`p` — `qec_q7_bp_graph -- enrichprobe 1 <prefix>`:
   **24/24 match silicon at p=0.003**, 6/24 at p=0.005, 0/24 at p=0.007 — at the default keep-best
   selection and the shipped ITERS=10, with no schedule variant needed.
2. **Off-board Verilator co-sim on 2000 shots sampled at p=0.007** against the shipped p=0.003 header
   (`make -C hw bpbanked-highweight`): golden decoded at λ(0.007) → **210/2000 mismatch** (the campaign
   divergence, reproduced off-board for the first time); golden decoded at λ(0.003) → **PASS 2000/2000
   bit-identical**, worst latency 2085 cycles.

**The banked core is correct**, including on the highest-weight syndromes. Everything the earlier
investigation suspected — synthesis fidelity, DSP widening of the `var_update` MACC, marginal hold, the
16-bit blend wrap, final-ê selection, banking gather/scatter, the M8 register plane, `m_cm`/`m_vm` drift
— was ruled out or rendered moot.

**What the campaign actually exposed** is a coverage hole plus a harness footgun:

* Every simulation gate drove 40 shots at p=0.003, so high-weight syndromes had never been simulated.
  `make -C hw bpbanked-highweight` is now that gate (2000 shots at p=0.007, 2000/2000 required).
* The vector emitters conflated the **sampling** rate with the **decoder** rate. `circvectors` and
  `silvectors` now take an explicit trailing `decoder_p` and stamp `decoder-p=` into the emitted header,
  so a golden can no longer be silently paired with an RTL header built at another `p`.

**Re-running AC-2.** The gate needs the two sides configured alike, which is either one bitstream per
rate (matched priors at each point — the LER optimum, and the route taken) or one bitstream with goldens
emitted at its header's `p` (`silvectors 1 <p> 1000000 2024 p00X 0.003`), which is also the more
realistic deployment metric: real silicon bakes its priors and then meets whatever physical rate the
device sees.

## Reproduce

```
# AC-2 vectors (per point) + campaign. The trailing decoder_p is the p the TARGET BITSTREAM's header
# was generated at; omit it only when the bitstream was built at the same p as the shots (#478).
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 <p> 1000000 2024 <prefix> [decoder_p]
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     bp_stream_banked_ler_kv260.py bp_kv260_stream_banked.bit p003 p005 p007
```

```
# #478 checks, off-board:
make -C hw bpbanked-highweight                                          # 2000 high-weight shots, 2000/2000
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- enrichprobe 1 hw/bp_enrich_p007
```

`hw/bp_enrich_p007.{syn,ref,rtl}` are the 24 shots themselves (syndrome words, `true_obs`+`sw_obs`, and
the observable this silicon produced), kept in-tree so the reproduction above needs no board.

```
# build (EPYC + Vivado 2024.2), full then early-exit:
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100            # full
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> 100 full default 1  # early-exit
# on the KV260 (root, pynq venv + XRT), from ~/q7stream with the .bit + a placeholder .xclbin sidecar:
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     bp_stream_banked_kv260.py bp_kv260_stream_banked.bit bp_circ_vectors.txt --bench-batch 20000
```
