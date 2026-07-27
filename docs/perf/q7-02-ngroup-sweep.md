# Q7 Task B2 — the `NGROUP` sweep: where the partial-unroll core's latency/area curve actually goes

**Status: in progress.** Two of the planned probe points are measured; the third is running. The model
and its prediction below were written **before** the third point returned, so that point is a test of the
model rather than a fit to it.

## What this measures and why

Decoder latency is `cycles / Fmax`. Task B0 left us with two cores at opposite ends of that curve, with
nothing in between measured:

| core | cycles | Fmax | latency | fits KV260 |
|---|---|---|---|---|
| banked 16/48 | 2085 | 133.3 MHz (silicon) | 15.64 µs | yes |
| M4 unrolled (`bp_relay_unrolled`) | 181 | 30.7 MHz (OOC) | 5.9 µs | no — 838 % LUT |

`bp_relay_unroll_pipe.sv` spans that gap: it stamps only `1/NGROUP` of the check and variable slots and
time-multiplexes each slot across `NGROUP` groups per BP phase. The sweep asks the obvious question —
**which `NGROUP` minimises `cycles / Fmax` while still fitting the KV260** (117,120 LUTs, 234,240 FFs,
14,640 CARRY8).

Method: `hw/syn/run_ngroup_sweep.sh` runs `hw/syn/ooc_core.tcl` once per `NGROUP`, serially, out-of-context
on `xck26-sfvc784-2LV-c` at a 5.0 ns target — the same part, flow and period as the B0 probe, so the
numbers are directly comparable. Cycles come from the Verilator co-simulation, which doubles as the
bit-exactness gate at every swept point.

## The cycle half: closed form, and bit-exact everywhere

The co-simulation (`tb_bp_unroll_pipe.cpp`, 40 circuit-level DEM shots, p = 0.003) is **40/40
bit-identical to the fixed-point golden at every `NGROUP` measured** — 4, 6, 8, 10, 12, 16, 24, 48, plus
15 further non-divisor values. `NGROUP` is a pure latency knob; it does not touch the decode.

Latency is an exact affine function of the knob:

```
cycles(NGROUP) = 122 · NGROUP + 240
```

| NGROUP | 4 | 6 | 8 | 10 | 12 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|---|
| cycles | 728 | 972 | 1216 | 1460 | 1704 | 2192 | 3168 | 6096 |

For scale: the banked 16/48 core's 2085 cycles sit between `NGROUP` 12 and 16.

Every shot takes the same number of cycles — worst equals median at every point. Unlike the banked core,
this core runs a fixed schedule with no early exit: it keeps the best satisfying decision across all
relay legs rather than stopping at the first. Worst-case is therefore the only fair basis for comparing
it against the banked core's 2085.

## The area/Fmax half: the knob does not do what it was built to do

| NGROUP | CLB LUTs | % of KV260 | F7 / F8 muxes | CARRY8 | FF | Fmax | latency |
|---|---|---|---|---|---|---|---|
| 48 | 1,042,940 | **890 %** | 420,002 / 198,526 | 1,680 (11 %) | 53,552 (23 %) | 17.3 MHz | 352 µs |
| 24 | 1,433,697 | **1224 %** | *(see report)* | 3,168 (22 %) | 57,084 (24 %) | 16.8 MHz | 189 µs |
| 16 | *running* | | | | | | |

Two things are already clear, and both are bad:

1. **Area moves the wrong way.** Halving `NGROUP` from 48 to 24 — which halves nothing in the intended
   sense, it doubles the stamped arithmetic — costs **+390,757 LUTs**. The smallest member of this family
   is the one with the *most* cycles.
2. **Fmax is flat at ~17 MHz** across a 2× change in stamped arithmetic. The critical path is not in the
   arithmetic. It is in the part that does not change with `NGROUP`.

## Why: the crossbar is `NGROUP`-invariant by construction

The utilisation report says the design is muxes, not maths: at `NGROUP=48`, F7/F8 muxes are at 717 % /
678 % of the device while CARRY8 sits at 11 % and DSPs at 9 %. Vivado also reports the message arrays as
`RAM dissolved into registers` — there is no memory here, only registers and the selection logic between
them and the slots.

That selection logic is invariant in `NGROUP`, and the RTL shows why. Each slot's gather
(`bp_relay_unroll_pipe.sv:94` and `:135`) selects among `NGROUP` groups' worth of constant edge indices,
and there are `BP_C/NGROUP` check slots and `BP_N/NGROUP` variable slots. The product cancels:

```
mux volume  ≈  (BP_C/NGROUP) · CHK_DEG · MSG · NGROUP  +  (BP_N/NGROUP) · VAR_DEG · MSG · NGROUP
            =   BP_C · CHK_DEG · MSG     +     BP_N · VAR_DEG · MSG        — no NGROUP left
```

So the area decomposes into a fixed crossbar plus arithmetic that scales as `1/NGROUP`:

```
LUT(NGROUP)  ≈  FLOOR  +  A / NGROUP
```

Fitting the two measured points gives **FLOOR ≈ 652,000 LUTs** (557 % of the KV260) and `A ≈ 1.88e7`.

**Prediction, recorded before the run finished:** `NGROUP = 16` should land near **1.82 M CLB LUTs**, with
Fmax still ~17 MHz. If instead it comes in near 1.04 M or below, the fixed-floor model is wrong and the
sweep must continue down the ladder.

If the model holds, the consequence is that **the feasible set is empty**: no `NGROUP` fits, because the
`NGROUP`-free floor alone is 5.6× the whole device — and buying closer to that floor costs cycles
linearly, with Fmax pinned at 17 MHz by the very logic that constitutes the floor.

This is the same wall the M9c fit note (PR #463) hit from the other side — a ~1.17 M-mux runtime gather —
and it is the reason the banked core exists. Banking replaces this crossbar with index-addressed LUTRAM
banks. It was never an optimisation on top of a working parallel design; it is what makes the design
implementable at all.
