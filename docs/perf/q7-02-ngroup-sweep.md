# Q7 Task B2 — the `NGROUP` sweep: the partial-unroll family has no feasible member

**Result: the knob does not work, and the banked core dominates the entire family on both axes at
once.** Five `NGROUP` values were synthesised out-of-context on the KV260 part. Every one of them is
simultaneously *over the device budget* and *slower than the banked core that already ships*. There is no
trade-off left to tune.

## What this measures and why

Decoder latency is `cycles / Fmax`. Task B0 left two cores at opposite ends of that curve with nothing
in between measured:

| core | cycles | Fmax | latency | fits KV260 |
|---|---|---|---|---|
| banked 16/48 | 2085 | 133.3 MHz (silicon) | 15.64 µs | yes |
| M4 unrolled (`bp_relay_unrolled`) | 181 | 30.7 MHz (OOC) | 5.9 µs | no — 838 % LUT |

`bp_relay_unroll_pipe.sv` spans that gap: it stamps only `1/NGROUP` of the check and variable slots and
time-multiplexes each slot across `NGROUP` groups per BP phase. The sweep asks the obvious question —
**which `NGROUP` minimises `cycles / Fmax` while still fitting the KV260** (117,120 LUTs, 234,240 FFs,
14,640 CARRY8)?

Method: `hw/syn/run_ngroup_sweep.sh` drives `hw/syn/ooc_core.tcl` once per `NGROUP`, serially,
out-of-context on `xck26-sfvc784-2LV-c` at a 5.0 ns target — the same part, flow and period as the B0
probe, so the numbers are directly comparable to it. Cycles come from the Verilator co-simulation, which
doubles as the bit-exactness gate at every swept point.

## The cycle half: closed form, bit-exact everywhere

The co-simulation (`tb_bp_unroll_pipe.cpp`, 40 circuit-level DEM shots at p = 0.003) is **40/40
bit-identical to the fixed-point golden at every `NGROUP` measured** — 4, 6, 8, 10, 12, 16, 24, 48, 72,
144, plus 15 further non-divisor values. `NGROUP` is a pure latency knob; it does not touch the decode.

Cycles are an exact affine function of it:

```
cycles(NGROUP) = 122 · NGROUP + 240
```

| NGROUP | 4 | 6 | 8 | 10 | 12 | 16 | 24 | 48 | 72 | 144 |
|---|---|---|---|---|---|---|---|---|---|---|
| cycles | 728 | 972 | 1216 | 1460 | 1704 | 2192 | 3168 | 6096 | 9024 | 17808 |

Every shot takes the same number of cycles — worst equals median at every point. Unlike the banked core,
this core runs a fixed schedule with no early exit: it keeps the best satisfying decision across all
relay legs rather than stopping at the first satisfied one. Worst-case is therefore the only fair basis
for comparing it against the banked core's 2085.

## The area/Fmax half: measured

| NGROUP | cycles | CLB LUTs | % of KV260 | CARRY8 | FF | Fmax | **latency** |
|---|---|---|---|---|---|---|---|
| 144 | 17,808 | 703,698 | **601 %** | 672 (5 %) | 51,256 (22 %) | 17.5 MHz | 1017.6 µs |
| 72 | 9,024 | 858,952 | **733 %** | 1,176 (8 %) | 52,368 (22 %) | 17.4 MHz | 518.6 µs |
| 48 | 6,096 | 1,042,940 | **890 %** | 1,680 (11 %) | 53,552 (23 %) | 17.3 MHz | 352.4 µs |
| 24 | 3,168 | 1,433,697 | **1224 %** | 3,168 (22 %) | 57,084 (24 %) | 16.8 MHz | 188.6 µs |
| 16 | 2,192 | 1,726,227 | **1474 %** | 4,596 (31 %) | 60,935 (26 %) | 16.5 MHz | 132.8 µs |
| *banked 16/48* | *2,085* | *fits* | *—* | *—* | *—* | *133.3 MHz* | ***15.64 µs*** |

Three facts, each independently fatal:

1. **Area moves the wrong way.** Shrinking `NGROUP` — the thing that is supposed to buy latency — makes
   the core *bigger*. The cheapest member is the one with the worst cycle count.
2. **Fmax is flat at 16.5–17.5 MHz across a 9× span of `NGROUP`** (6 % spread). The critical path is not
   in the arithmetic, because a 9× change in stamped arithmetic barely moves it.
3. **Nothing fits, and nothing is fast.** The best latency in the family is 133 µs at `NGROUP = 16` —
   **8.5× worse than the banked core that already runs on silicon**, at 14.7× the device's LUTs. The
   banked core wins on both axes at the same time.

## Why: the crossbar is `NGROUP`-invariant by construction

The utilisation report says the design is muxes, not maths: at `NGROUP = 48`, F7/F8 muxes are at 717 % /
678 % of the device while CARRY8 sits at 11 % and DSPs at 9 %. Vivado also reports the message arrays as
`RAM dissolved into registers` — there is no memory here, only registers and the selection logic between
them and the slots.

That selection logic does not shrink with `NGROUP`, and the RTL shows why. Each slot's gather
(`hw/bp_relay_unroll_pipe.sv:94` for checks, `:135` for variables) selects among `NGROUP` groups' worth
of constant edge indices — and there are `BP_C/NGROUP` check slots and `BP_N/NGROUP` variable slots. The
product cancels:

```
mux volume  ≈  (BP_C/NGROUP) · CHK_DEG · MSG · NGROUP  +  (BP_N/NGROUP) · VAR_DEG · MSG · NGROUP
            =   BP_C · CHK_DEG · MSG     +     BP_N · VAR_DEG · MSG      — no NGROUP left
```

So area is a fixed crossbar plus arithmetic that scales as `1/NGROUP`. Least squares over all five
measured points:

```
LUT(NGROUP)  =  615,985  +  18,415,474 / NGROUP        (residuals within 5.7 %)
```

The model was tested, not just fitted. With only `NGROUP` 48 and 24 in hand it predicted **1.82 M LUTs**
for `NGROUP = 16`; that prediction was committed in `9d2325d` *before* the run returned, and the measured
value was 1,726,227 — 5 % low. The floor was then measured directly rather than extrapolated:
**`NGROUP = 144`, where only one check slot and six variable slots are stamped, still needs 703,698 LUTs
— 601 % of the entire device** with the arithmetic essentially switched off.

That is the whole result. The crossbar is both the area floor *and* the critical path, which is why
6.0 device-budgets and ~17 MHz survive to the far end of the sweep where almost no arithmetic is left.

## What was not run, and why

The ladder was planned as 48 → 24 → 16 → 12 → 10 → 8 → 6 → 4 and **the downward tail (12, 10, 8, 6, 4)
was cut** after the first three points, in favour of the two floor probes (144, 72) that actually
discriminate. The tail is guaranteed worse on the axis that already disqualifies the family: the fitted
model puts `NGROUP = 8` at ~2400 % and `NGROUP = 4` at ~4100 % of the device. Roughly ten hours of
synthesis would have bought a steeper confirmation of a conclusion already established. Cycles at those
points *are* measured and bit-exact — only the synthesis was skipped.

## Consequences

- **The `NGROUP` knob is not a latency lever.** It trades cycles for area at a ruinous exchange rate and
  never touches Fmax, because it cannot touch the crossbar that sets Fmax.
- **The banked core is not one option among several — it is the only implementable one.** It dominates
  every measured member of this family on latency *and* fits. This is the second independent measurement
  saying so: PR #463's M9c fit note hit the same wall from the other side with a ~1.17 M-mux runtime
  gather. Banking replaces the flat register file plus crossbar with index-addressed LUTRAM banks; it was
  never an optimisation bolted onto a working parallel design, it is what makes the design implementable.
- **Sub-microsecond remains blocked, and not on this road.** B0 closed the fully-unrolled road (838 %,
  30.7 MHz); B2 now closes the partially-unrolled road at every knob setting. Any future attempt has to
  attack the message store and its access pattern — the thing banking already attacks — rather than the
  degree of unrolling on top of a flat register array.

## Reproducing

```bash
# cycles + bit-exactness at one NGROUP (Verilator >= 5.050)
cd hw && cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 > bb_gross_tanner.svh
cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > bp_circ_vectors.txt
verilator --cc --exe --build -Wall --Mdir obj_ng16 --top-module bp_relay_unroll_pipe -GNGROUP=16 \
  check_minsum.sv var_update.sv bp_relay_unroll_pipe.sv tb_bp_unroll_pipe.cpp -o sim_ng16
./obj_ng16/sim_ng16 bp_circ_vectors.txt

# area + Fmax (Vivado box; edit NG_LIST in the script, then run it detached)
#   nohup setsid ./run_ngroup_sweep.sh >/dev/null 2>&1 </dev/null &
```

Two traps cost real time here and are worth naming:

- **`GROUPS` is a special bash variable** holding the caller's group IDs, and assignments to it are
  silently ignored. A sweep parameterised on `GROUPS` runs over the *user's GIDs* — locally that was
  15 bogus points (20, 12, 61, …) and on the box, as root, exactly one: `NGROUP=0`. The runner uses
  `NG_LIST`.
- **The OOC utilisation report writes the row as `CLB LUTs*`**, with a star. The probe's original regex
  missed it and reported `CLBLUT=-1` on every run, including B0's — which is why B0 had to quote a raw
  cell count. Fixed in `hw/syn/ooc_core.tcl`.
