# Q7-01 de-risk — SKY130 open-PDK synthesis probe of the relay-BP cores

**Date:** 2026-07-16 · **Box:** EPYC 8124P (`195.154.249.85`), Yosys 0.67+40 (oss-cad-suite
2026-07-16, slang SV frontend) · **Library:** `sky130_fd_sc_hd__tt_025C_1v80.lib` (SkyWater 130 nm
HD, typical corner, from OpenROAD-flow-scripts)

## Question

The M9c area campaign closed with a terminal no-fit verdict: the fully-parallel streaming relay-BP
decoder is 162 % of a KV260's LUTs (`qec-q7-fixed-bp.md` § M9c). The ASIC track (Q7-01..03) has so
far only had FPGA numbers to reason from. This probe answers, with an open ASIC flow on measured
netlists, the first Q7-01 questions:

1. Is the RTL **portable** off the Xilinx toolchain (no vendor primitives / attributes load-bearing)?
2. What do the two production cores cost in **standard-cell area** on the cheapest open PDK
   (SkyWater 130 nm — the chipIgnite / Tiny Tapeout shuttle process)?
3. Is the KV260 no-fit a statement about the **design's size** or about the **LUT fabric**?

## Method

`hw/syn/asic_probe.sh` — Yosys with the slang frontend (`--unroll-limit` raised; the only flag the
RTL needs), generic synth, then SKY130 HD mapping:

- **Memories are inventoried and kept as `$mem_v2` blackbox boundaries**, not mapped: on an ASIC
  they become SRAM/ROM macros or register files, not standard cells. Their bit totals are reported
  separately from a parameter dump (`*_mems.txt`). *Deleting* them instead is a measurement bug we
  hit first: the message RAMs sit in the decode loop, so removing them let `opt_clean` sweep the
  whole dangling datapath — the tell was a 0.033 mm² "core" that was 91 % flops.
- `techmap` → `dfflibmap`/`abc -liberty` at a 10 ns target; `stat -liberty` gives standard-cell
  area, ABC `stime` the pre-P&R critical path. For the M8 core the ABC script was extended with
  `buffer; upsize; dnsize; stime` — see the fanout finding below.
- No place & route, no wire parasitics (`WireLoad none`), typical corner only. These are
  synthesis-quality numbers: good to a first order for area, honest only after buffering for delay,
  and silent on routing. An OpenROAD P&R pass is the follow-up that would harden them.

Sanity scale point: one combinational `check_minsum` block alone maps to 18,478 µm² (0.018 mm²),
25 % sequential — the flow round-trips a known-small module at a believable size.

## Results

| | **M8 core** (`bp_relay_banked`) | **streaming core** (`bp_relay_banked_bram_m` + Beneš + AS-Waksman) |
|---|---|---|
| source config | rounds=1 gross graph, 16/48 banking — the shipped KV260 overlay (15.64 µs worst / 0.85 µs median @ 133 MHz) | the M9c Step-5 file set verbatim (W=6 C=2 window header) — the design that is **162 % of a KV260** |
| std-cell logic area | **1.88 mm²** (256.5 k gates, 15 % sequential, ~13.1 k FF) | **6.45 mm²** (739.5 k gates, 25 % sequential, ~80 k FF) |
| memory bits (excluded from above) | 122.8 kbit in 1,499 arrays | 528.7 kbit in 1,736 arrays |
| memory shape | dominated by 1,108× 8b×9 + 164× 8b×18 message arrays → register-file class, not SRAM-macro class | same small-array tail (1,200× 8b×9) plus a handful of real ROM/RAM singles (916×9, 2× 6400×1, …) |
| pre-P&R critical path | 691 ns raw → **11.5 ns after ABC buffer/upsize/dnsize** (~87 MHz tt) | 36.5 ns raw (buffer pass not run) |
| slang/yosys wall time | ~7.5 min | ~47 min |

**Area accounting.** The small memories will not become SRAM macros — at 72–144 bits each they are
flop/latch register files. Budgeting them as flop arrays adds roughly 2–3 mm² (M8) / ~1.5 mm²
(streaming, whose big bits *do* macro), giving core totals of **~4–5 mm²** (M8) and **~10–11 mm²**
(streaming) in SKY130 HD. For placement-feasibility framing: chipIgnite's user area is ~10 mm² —
the M8 core fits with room, the full streaming core is right at the envelope (and its logic-only
6.45 mm² fits). On any modern node (22FDX / 16 nm) both cores are trivially small (~0.5–2 mm²).

**Timing artifact worth recording.** ABC's first `stime` reported a 691 ns "critical path" on the
M8 core. The printed path is only **10 gate levels** deep; the delay is unbuffered fanout — the
FSM state bits drive 1,700–2,600 loads from single gates (6.4 pF on one nand2b). One
`buffer; upsize; dnsize` pass (+11.9 k buffers, +2.9 % area) collapses it to 11.5 ns. Lesson for
every future ASIC number out of this flow: **never quote the unbuffered stime**; and the datapath
itself is shallow — the design's speed on an ASIC will be set by P&R quality, not logic depth.

**Memory truth vs BRAM tiles.** The KV260 report charges the streaming core 163 BRAM tiles ≈
5.9 Mbit of *capacity*; the actual stored state is **528.7 kbit** — 91 % of the BRAM budget was
36 Kbit-tile quantization over 1,736 mostly-tiny arrays. The two-constraint no-fit (LUT *and*
BRAM) is therefore doubly a fabric artifact: the LUT side was mux-fabric emulation of a
permutation, the BRAM side was tile granularity.

## Verdict

1. **The RTL is ASIC-portable as-is.** Zero Xilinx primitives; `rom_style`/`ram_style` attributes
   are advisory; slang elaborates everything with one `--unroll-limit` flag.
2. **The KV260 no-fit is a fabric statement, not a size statement.** The full streaming decoder —
   unfittable on the 117 k-LUT part — is a ~740 k-gate, 6.45 mm² SKY130 netlist: an unremarkable
   ASIC block even on a 130 nm open PDK, and a small one on any commercial node. This is the
   quantified version of Q6-03's "the ASIC removes the routing tax" GO argument.
3. **A silicon prototype is shuttle-class money, not tape-out-program money.** The M8 core (~4–5
   mm² all-in, ~87 MHz pre-P&R ⇒ 24 µs worst / 1.3 µs median early-exit at SKY130 speeds — KV260
   class on a 130 nm process) fits a chipIgnite-style ~$10 k shuttle with margin. The Q7-03
   feasibility study should price shuttles, not full mask sets, for the prototype step.
4. **Next hardening steps** (in Q7-01 spec order): OpenROAD P&R of the M8 core on sky130hd for
   placed-and-routed Fmax/area/power; a register-file/latch-array plan for the 8b×9 message
   arrays (the dominant memory shape on both cores); then the same two passes on a commercial-node
   PDK under NDA when one is in reach.

## OpenROAD P&R follow-up (sky130hd) — the § "next hardening steps" item 1, measured

**Date:** 2026-07-18/19 · **Flow:** OpenROAD-flow-scripts docker (`openroad/orfs:latest`) on the
EPYC box; M8 core pre-elaborated with the oss-cad-suite yosys+slang (`--unroll-limit`,
`memory_map` — the message arrays as flat DFF register files, i.e. the spec's D6 *baseline*
implementation), 10 ns clock, sky130hd.

**Synthesis (ORFS yosys):** 5.28 mm² total standard-cell — validating the probe's ~4–5 mm²
all-in estimate above. 98.8 k flops: 11,486 `dfxtp` + **87,274 `edfxtp`** (the enable-flops of
the flop-mapped message arrays).

**Place + CTS: clean.** Global/detail placement, resizing and clock-tree synthesis all
converge; TritonCTS builds the 98,760-sink clock net with 11,728 buffers.

**Global routing: fails at both tested utilizations.** The wall is met2 (the first vertical
routing layer — the read/write mux trees of 1,272 tiny arrays):

| CORE_UTILIZATION | wirelength | met2 usage | total overflow | verdict |
|---|---|---|---|---|
| 30 % | 86.1 M µm | **96.6 %** | 304,619 | GRT-0116 congestion fail |
| 20 % | 119.2 M µm | **83.9 %** | 45,798 | GRT-0116 congestion fail |

The 1.5× area increase buys a 6.6× overflow reduction — extrapolating, the flat-DFF netlist
might route near ~12 % utilization, a ~44 mm² die for a 5.3 mm² netlist. That is not a tuning
problem; it is a structural verdict: **flat-flop message arrays are route-infeasible on
sky130hd**, and the placed-Fmax/area/power numbers this section was meant to produce must come
from the restructured-memory core instead (Q7-08, issue #470 — where the first measured lever is
already in: latch arrays are 2.54× denser than DFF at 15.1 µm²/bit, and the m_vm arrays
consolidate mechanically into wide byte-masked banks because their read row is the global pc
cursor).

**Flow traps recorded** (cost a day of wall clock between them): ORFS `make` does not hash the
design config — after changing `CORE_UTILIZATION` the stale floorplan/place/CTS artifacts under
`WORK_HOME` are reused and only GRT re-runs (wipe `[2-6]_*` phase artifacts to actually re-place);
ORFS synthesis silently ignores yosys `write_verilog` elaborated netlists ("contains processes")
— feed it raw RTL, or pre-elaborate only when the frontend genuinely cannot (M8's slang
unroll limit); and long jobs on this box need `setsid` or they die with the launching ssh
session.

## Reproduce

```bash
# one-time setup on the synth box (downloads oss-cad-suite + the ORFS sky130 liberty)
mkdir -p /data/asicprobe && cd /data/asicprobe
curl -sLO https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-07-16/oss-cad-suite-linux-x64-20260716.tgz
tar xzf oss-cad-suite-linux-x64-20260716.tgz
curl -sLo sky130_fd_sc_hd__tt_025C_1v80.lib \
  https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib

# M8 core (stage hw/: bb_gross_tanner.svh + the three sources)
hw/syn/asic_probe.sh <staging-dir> bp_relay_banked 10000 \
  check_minsum.sv var_update.sv bp_relay_banked.sv

# streaming core (stage the M9c file set: stream header as bb_gross_tanner.svh)
hw/syn/asic_probe.sh <staging-dir> bp_relay_banked_bram_m 10000 \
  check_minsum.sv var_update.sv bp_benes.sv bp_asw.sv bp_relay_banked_bram_m.sv
```

Outputs per run: `<top>_sky130.log` (stat blocks + ABC stime), `<top>_mems.txt` ($mem_v2 parameter
dump; sum WIDTH×SIZE for the macro budget), `<top>_mapped.blif`.
