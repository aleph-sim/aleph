# Track P2 — appliance v2 on a ZCU104: 36/144 at 125 MHz, 8.29 µs worst case

> Open-silicon program (`docs/qec/open-silicon-program.md`), Track P, Task P2. Built 2026-09-25 on the
> EPYC box with the free Vivado 2024.2. **Built and timing-closed, not yet run on a board** — nobody on
> the project owns a ZCU104. The first lab that has one runs `deploy.sh`, which will not report success
> without 40/40 bit-exact on that board.

## 1. Why the target moved from VU47P to ZCU104

The plan named the 144/864 build on the VU47P that Phase B had placed and routed. That artefact cannot
be published as a bitstream:

- Phase B implemented the **bare core out of context** — no PS, no DMA, no pins. There is nothing to
  program.
- The VU47P exists as a rentable part only inside AWS F2, and **every large part is licence-gated.**
  Vivado ML Standard (free) covers Zynq UltraScale+ only up to the ZU7EV. Virtex UltraScale+, RFSoC
  and Alveo all need a paid tier (Enterprise, or Core/Pro from 2026.1). The only free route to a VU47P
  bitstream is AWS's FPGA Developer AMI, which is licensed on EC2 for AWS parts only.
- B2 Step 4 (2026-07-31) had already moved v2 off 144/864 anyway: 1.09× faster than 64/192 for 4× the
  area.

The **ZU7EV** is the largest part the free toolchain builds. It sits on the ZCU104, which PYNQ supports
officially. It is also the same Zynq UltraScale+ family as the KV260, so v1's block design ports
unchanged and v2 keeps v1's host interface — the `interface-spec.md` §6 promise. The AWS F2 build
(large part, public AFI) follows as a separate step.

## 2. Picking the geometry — out-of-context fit on `xczu7ev-ffvc1156-2-e`

Product top `bp_stream_banked` (core plus the AXI-Stream shell), out of context, 5.0 ns target,
`hw/syn/ooc_banked.tcl` retargeted. Cycles come from the closed form
`60·(GC+GV) + 420 + (2·GV+GC)`, with `GC = ⌈144/W⌉` and `GV = ⌈864/V⌉`.

| W/V | GC/GV | cycles | CLB LUTs | % ZU7EV | DSP | Fmax (OOC) | verdict |
|---|---|---|---|---|---|---|---|
| 48/192 | 3/5 | 913 | 213,841 | 92.8 % | 593 | 153.4 MHz | no routing headroom |
| 48/144 | 3/6 | 975 | 197,333 | 85.7 % | 492 | 147.6 MHz | tight |
| **36/144** | **4/6** | **1036** | **174,039** | **75.5 %** | 493 | **214.5 MHz** (met) | **chosen** |

48/192 is the first geometry to try, not 64/192: it has the same 913 cycles, because GC = 3 either way
(`⌈144/48⌉ = ⌈144/64⌉`), while 64-wide check banks pad 48 dead lanes. It still does not fit.

A two-parameter area model from the KV260-part sweep, **LUT ≈ 48 k + 725·(W+V)**, predicted 48/192 to
within 3 % and picked the two follow-up points.

**Bit-exact at every candidate** (Verilator, 40-shot circuit-level golden at p = 0.003):

- 48/192: 913 cycles, 40/40
- 48/216: 851 cycles, 40/40
- 48/144: 975 cycles, 40/40
- 36/144: 1036 cycles, 40/40

36/144 is now in `make -C hw bpbankedscale`.

## 3. The PL-clock trap, and the fix

The first full build requested 150 MHz and missed it (WNS −0.617 ns). Its critical path is
`ehat_reg → best_e_reg`, 27 LUT levels, 7.27 ns — the best-kept candidate compare. The 133 MHz and
125 MHz builds met timing. **Both would have failed on a real ZCU104**, for a reason that has nothing
to do with the RTL:

- PYNQ's `Overlay.download()` copies **only PL0's `DIVISOR0/DIVISOR1`** out of the `.hwh`. It forces
  the source mux to IOPLL (`pynq/ps.py`, `PLX_CTRL_SRC_DEFAULT = 0`) and applies the divisors to the
  **board's** IOPLL.
- The KV260 script builds without a board preset. On the ZU7EV its PS model put PL0 on a 1050 MHz RPLL,
  so the "133 MHz" build carries 8×1 divisors.
- The ZCU104 PYNQ image runs IOPLL at ~1500 MHz (its `base.tcl` puts `DLL_REF` at 1499.98 MHz). So
  8×1 would have clocked the core at **187.5 MHz** against the 131.25 MHz it closed timing at — wrong
  decodes, and no error anywhere.

The fix is that `hw/syn/kv260_bp_stream_banked_bd.tcl` now takes `BP_BOARD`, which applies the board's
PS preset. With `xilinx.com:zcu104:part0:1.1`, the model runs IOPLL at 1500.000 MHz, and 125 MHz
becomes divisors **12×1** from IOPLL. The clock Vivado analyses and the clock the board produces are
now the same number. The preset also enables HPM1_FPD, which is now disabled explicitly.

The same arithmetic applied to v1. Its `.hwh` asks for 100 MHz and carries 11×1 against a modelled
RPLL, so its model says 96.97 MHz. On the KV260's ~1000 MHz IOPLL it really runs at **~90.9 MHz**,
which is why the self-test measures 23.0 µs per decode for 2085 cycles. That is below its closure
clock, so v1 is safe. But v1's quoted 15.64 µs belongs to the AXI-Lite overlay at 133.332 MHz, not to
the DMA image `deploy.sh` installs.

To stop this class of bug reaching a user, the driver now reads back the PL0 clock PYNQ programmed and
prints it. With `--max-mhz` it refuses to decode if the clock is faster than the closure clock.
`deploy.sh` passes `--max-mhz 125` on a ZCU104. On a KV260 the guard only reports for now: it has not
been run on that board yet.

## 4. The shipped build

`BP_PART=xczu7ev-ffvc1156-2-e BP_BOARD=xilinx.com:zcu104:part0:1.1 BP_OUTNAME=bp_zcu104_stream_banked`,
`-tclargs <proj> <out> 125 impl default 0` (full schedule, `early_exit = 0`, as v1), header
`circgraph 1 0.003 36 144`.

| | value |
|---|---|
| PL0 | **125.000 MHz** = IOPLL 1500 / (12·1) |
| setup | WNS **+0.041 ns**, TNS 0 — "All user specified timing constraints are met" |
| hold | WHS +0.010 ns, THS 0 |
| CLB LUTs | 176,017 / 230,400 = **76.4 %** |
| CLB registers | 45,357 = 9.8 % |
| DSP | 493 = 28.5 % |
| BRAM | 2 tiles |
| cycles | **1036**, fixed (full schedule) |
| **worst-case latency** | **8.29 µs** — against v1's 2085 cycles, which is 15.64 µs at 133.332 MHz and 22.9 µs at the v1 DMA image's real ~90.9 MHz |
| build time | 47 min on the EPYC box |

Checksums:

- `bp_zcu104_stream_banked.bit`: `7c17a503f8a70b348cb82cc1ac6dd30ac2e646c5a0ee7a05d094e38dbda2040c`
- `bp_zcu104_stream_banked.hwh`: `57e022b4480c5b0d2186895ea5c971f32ea5e050b684406688f5dbe887306566`

The 7.27–7.58 ns best-kept compare caps this core near 132 MHz on the ZU7EV. Pipelining it is the
obvious next clock lever. It is an RTL change that moves the cycle count, so it is not part of P2.

## 5. What stands in for a board

- **Verilator** RTL: 40/40 bit-exact against the fixed-point golden, 1036 cycles every shot.
- **xsim RTL**: 20/20 shots of the Q7-06 p = 0.003 campaign vectors match the software decoder
  (`net == sw`, 0 divergences).
- **xsim post-synthesis netlist**: the same 20 shots through the ZU7EV funcsim netlist of the core. This
  is the check that caught Q7-06's synthesis-vs-simulation question: _RESULT PENDING_.
- **Not done:** any run on a ZCU104. The release says so.

## 6. Reproduce

```bash
cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 36 144 > hw/bb_gross_tanner.svh
cd hw && BP_PART=xczu7ev-ffvc1156-2-e BP_BOARD=xilinx.com:zcu104:part0:1.1 BP_OUTNAME=bp_zcu104_stream_banked \
  vivado -mode batch -source syn/kv260_bp_stream_banked_bd.tcl -tclargs zcu104proj out_zcu104 125 impl default 0
```

The build log prints the PS clock actually modelled, e.g.
`PLCLK IOPLL_DLL_REF=1500.000000 PL0_ACT=125.000000 PL0_DIV0=12 PL0_DIV1=1 PL0_SRC=IOPLL`. If `PL0_ACT` is
not the frequency you asked for, the divisors will not produce it on the board either.
