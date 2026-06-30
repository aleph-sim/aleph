# Q6-21 — board-free sim↔RTL co-simulation (hardware-in-the-loop without a board)

**Status: done (sim).** The whole decoder verification chain now closes on realistic noise *entirely
in software*:

```
noise model  →  Monte-Carlo syndromes  →  RTL decode (Verilated)  →  logical error rate
   (aleph-qec)        (aleph-qec)            (uf_surface_decoder)        (vs software UF)
```

This is the board-free form of hardware-in-the-loop and the concrete realisation of the ROADMAP §2.4
co-design differentiator: the simulator plays QPU, the **actual synthesizable RTL decoder** closes the
loop, and we measure the logical-error rate / threshold the RTL produces — confirming it tracks the
software Union-Find decoder. When a board arrives (Q6-08) the same syndrome stream drives the real
decoder over the Q6-07 AXI link instead of Verilator; nothing else in the harness changes.

## How it works

| piece | file |
|-------|------|
| syndrome stream + software baseline | `crates/aleph-qec/examples/qec_q6_cosim.rs` |
| Verilated co-sim testbench | `hw/tb_uf_cosim.cpp` |
| run targets | `make -C hw cosim` (d=3) · `make -C hw cosim-3d` (d=5×3) |
| matching graph (shared with the decoder) | `qec_surface_uf_graph -- graph <d> <rounds>` |

The driver draws shots from the **same** detector-error model the RTL's matching graph was generated
from, via the shared `aleph_qec::sample_shots` (each shot's RNG derived from `(seed, index)`). So the
software UF baseline and the RTL decode the *identical* Monte-Carlo stream — their logical-error rates
are directly comparable shot-for-shot, not just distributionally. The matching graph is
`p`-independent (its structure is which mechanisms exist, with uniform-noise edges), so one RTL build
serves the whole `p` sweep; the `.vec` file carries one block per `p`, each with its software
`sw_rate`/`sw_ci` in the header.

The testbench drives each shot through the decoder's multi-cycle `in_valid → out_valid` handshake,
collects `obs_flip`, and accumulates the RTL logical-error rate with a normal-approximation 95% CI
(the same formula as `LogicalErrorResult`). It checks `|rtl_rate − sw_rate|` against the combined CI.

## Results

### d=3, code-capacity (1 round) — full 2-D matching graph (8 detectors), 20 000 shots/cell

```
   p       rtl_rate     sw_rate     |diff|    combined_ci  verdict
  0.010   7.2500e-03  7.8000e-03  5.50e-04  2.40e-03   PASS
  0.020   2.6900e-02  2.7250e-02  3.50e-04  4.50e-03   PASS
  0.030   5.1800e-02  5.2400e-02  6.00e-04  6.16e-03   PASS
  0.040   8.2250e-02  8.1350e-02  9.00e-04  7.60e-03   PASS
  0.050   1.1545e-01  1.1400e-01  1.45e-03  8.83e-03   PASS
max decode latency = 30 clk
RESULT: PASS  (RTL logical-error rate matches software UnionFind within MC CI at every p)
```

The RTL decoder reproduces the software UF logical-error curve within Monte-Carlo CI at **every** `p`.

### d=5 × 3 rounds — multi-round phenomenological (3-D space-time) graph (48 detectors), 20 000 shots/cell

```
   p       rtl_rate     sw_rate     |diff|    combined_ci  verdict
  0.010   4.6000e-03  4.0500e-03  5.50e-04  1.82e-03   PASS
  0.020   3.5150e-02  3.0750e-02  4.40e-03  4.94e-03   PASS
  0.030   9.0300e-02  8.0600e-02  9.70e-03  7.75e-03   info (supra-threshold)
  0.040   1.6380e-01  1.4760e-01  1.62e-02  1.00e-02   info (supra-threshold)
  0.050   2.4695e-01  2.3295e-01  1.40e-02  1.18e-02   info (supra-threshold)
max decode latency = 112 clk
RESULT: PASS  (gated on the sub-threshold operating regime, p ≤ 0.02)
```

In the **sub-threshold operating regime** (`p ≤ 0.02`, below the ~3 % phenomenological surface-code
threshold) the RTL matches the software UF within CI — exactly the regime a real decoder runs in.

**Above threshold the RTL UF is modestly weaker** (higher LER, the gap growing with `p`). This is a
real, expected quality gap, not statistical noise: both decoders decode the *same* shots, yet the RTL
loses consistently and monotonically. The RTL is an **unweighted, bounded-per-cycle** Union-Find FSM;
above threshold (where weight-≥3 fault clusters dominate) it tie-breaks degenerate equal-weight cosets
more crudely than the CPU `UnionFindDecoder`, so it commits to a logical flip slightly more often.
Above threshold logical memory is failing regardless, so this gap does not affect the device's
operating point — but **surfacing it is precisely the value of hardware-in-the-loop**: the co-sim
catches a hardware-vs-software decoder-quality difference that neither side's unit tests would.

This is consistent with `hw/README.md` (the RTL UF and CPU UF agree bit-for-bit on 171/256 d=3
syndromes; the rest are logically-degenerate cosets where UF tie-breaks legitimately differ) — here we
quantify the aggregate effect across distance, rounds, and noise strength.

### Circuit-level noise (d=3 × 3 rounds) — the realistic graph with hook errors, 20 000 shots/cell

The phenomenological model above puts one data error per round + a measurement flip. The
**circuit-level** model (`docs/perf/qec-surface-circuit-dem.md`) makes every CNOT / idle / prep /
measurement a fault site — so the matching graph carries **hook-error edges** the phenomenological
graph lacks (here `M = 49` edges over 16 detectors, vs 18 over 8 for d=3×1 phenomenological). The RTL
decoder thus closes the loop on the *realistic* graph: `graph-circuit` builds the RTL graph from the
circuit-level DEM, `qec_q6_cosim … circuit` samples from the same model.

```
co-sim: graph N=17 M=49 dets=16 | circuit-level
   p       rtl_rate     sw_rate     |diff|    combined_ci  verdict
  0.002   2.2000e-03  2.2500e-03  5.00e-05  1.31e-03   PASS
  0.004   8.3500e-03  8.7500e-03  4.00e-04  2.55e-03   PASS
  0.006   1.6650e-02  1.7250e-02  6.00e-04  3.58e-03   PASS
  0.008   2.7800e-02  2.8600e-02  8.00e-04  4.59e-03   PASS
  0.010   3.9500e-02  4.0750e-02  1.25e-03  5.44e-03   PASS
max decode latency = 50 clk
RESULT: PASS  (gated on the sub-threshold operating regime, p ≤ 0.006)
```

The RTL decoder reproduces the software UF logical-error curve within CI across the whole circuit-level
prob grid (the circuit-level threshold is ~0.9%, so the grid is lower and we gate `p ≤ 0.006`). This is
the full board-free HiL chain on realistic gate noise: **circuit-level DEM → 3-D syndromes → RTL decode
→ LER**.

The same on the larger **d=5 × 3** circuit-level graph (`make -C hw cosim-circuit-3d`) — `N=49 M=165`
(vs 120 for d=5×3 phenomenological; the extra 45 edges are hook errors) — also passes at every p:

```
co-sim: graph N=49 M=165 dets=48 | circuit-level
   p       rtl_rate     sw_rate     |diff|    combined_ci  verdict
  0.002   2.5000e-04  1.5000e-04  1.00e-04  3.89e-04   PASS
  0.004   2.4000e-03  1.7000e-03  7.00e-04  1.25e-03   PASS
  0.006   7.9000e-03  7.6000e-03  3.00e-04  2.43e-03   PASS
  0.008   1.7300e-02  1.6500e-02  8.00e-04  3.57e-03   PASS
  0.010   3.1650e-02  2.8200e-02  3.45e-03  4.72e-03   PASS
max decode latency = 123 clk
RESULT: PASS
```

## Reproduce

```bash
make -C hw cosim          # d=3, phenomenological, p=0.01..0.05, all within CI
make -C hw cosim-3d       # d=5×3 (3-D phenomenological), gated on p ≤ 0.02
make -C hw cosim-circuit     # d=3×3, circuit-level (hook errors), gated on p ≤ 0.006
make -C hw cosim-circuit-3d  # d=5×3, circuit-level (denser graph, M=165), gated on p ≤ 0.006
# knobs: COSIM_SHOTS (default 20000), COSIM_SEED (default 2024)
make -C hw cosim COSIM_SHOTS=100000
```

## The AXI swap-in (Q6-08)

The `.vec` stream is the data plane. On hardware the same stream feeds the decoder over the Q6-07
AXI4-Stream link (`uf_axi_wrap.sv`): write each syndrome word on `s_axis`, read `{obs_flip,
correction}` back on `m_axis`, accumulate the LER identically. The Verilated `uf_surface_decoder` in
`tb_uf_cosim.cpp` is the only piece that gets replaced by the real board; the driver, the noise model,
and the comparison are unchanged. That makes this harness the board-free stand-in for full HiL and the
on-ramp for Q6-08 bring-up.
