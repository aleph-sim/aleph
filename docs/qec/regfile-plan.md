# Q7-08 — Register-file/latch plan for the relay-BP message arrays

**Status:** v1 (2026-07-21). Closes issue #470 (spec `docs/qec/asic-architecture.md` D6/§ 8.2).
**Inputs:** the ORFS sky130hd flat-DFF verdict (`docs/perf/qec-q7-asic-sky130-probe.md` § "OpenROAD
P&R follow-up"): the flop-only core synthesizes to 5.28 mm² but **fails global routing on met2 at
30 % AND 20 % utilization** on sky130hd's five routable metals. Q7-08 asked whether restructuring
the message memories (latch arrays + wide byte-masked banks) fixes that.

**Outcome (honest, mixed):** the restructuring is a clear **area/clock-fanout win** (−14 % core
area, 43 k vs 98.8 k clock sinks, latch storage 2.54× denser than DFF) and a **marginal** signal-
congestion improvement — but it is **not, by itself, a routability fix**. At 20–30 % util on
sky130hd the restructured core is still met2-congested (78–82 % post-CTS), does not route cleanly,
and in full-flow P&R is actually *harder* than the flat core (its latch-dense clock tree livelocks
GRT on clock-NDR congestion). Restructured memories are **necessary but not sufficient** on
sky130hd. **This prediction is then validated on ASAP7** (open 7 nm predictive PDK, no NDA): the
*identical* netlist routes cleanly there — GRT 0 congestion on every layer, no clock-NDR livelock,
detailed route completes to a 0.163 mm² die at 45 % util (207 residual DRC, mostly PDK artifacts)
— proving **the routability wall is sky130's 130 nm node, not the design**. Bit-exactness (AC-2
co-sim) and the area de-risk (AC-1) are solid; the routability verdict is "sky130-limited, routes
on a modern node". Full data below (§ AC-2 sky130 + § AC-2 follow-up ASAP7).

## The decision table (AC-3)

RTL characterization of the three message-array cell types in `hw/bp_relay_banked.sv` (16/48
banking), and the chosen implementation per type:

| cell | shape | ports | access pattern | chosen implementation | why |
|---|---|---|---|---|---|
| `bp_mvm_cell` ×164 | 8b×18 | 1W + 1R async | read row = global `pc` cursor; write row = global group cursor; per-cell enable `vedge_at(h,i,d)≥0` | **consolidate: one wide byte-masked array per var slot** (`bp_mvm_rf` ×48, `VAR_DEG`×8b wide × 18 rows) | all lanes of a slot already share both cursors — the only per-(i,d) term is the present gate, which becomes the per-lane write mask. One row decoder per slot instead of `VAR_DEG` private ones: area-neutral, the win is write/read wiring |
| `bp_mcm_cell` ×~740 | 8b×9 | 1W + 1R async | read row = global `pc`; **write row is per-cell** (the edge's check group, from the shared m_cm scatter) | **latch array** (transparent-low rows) | cannot consolidate — the write row differs per half-bank in the same cycle. Latch storage is 2.54× denser than DFF and halves the storage-cell footprint under the same wiring |
| `bp_ecm_cell` ×~370 | 8b×9 | 1W + **2R async** | write row = global chk-group cursor, but **two fabric-driven read rows** (per-bank port-A/B addresses from the e_cm read scatter) | **latch array** | the dual distributed read ports are the point of the bank — consolidation would rebuild the runtime-index mux wall M7 killed. Same latch-density win |

Write-pulse discipline for the latch arrays (transparent-low: rows open during clk-low, `wa`/`wd`
must be settled by the falling edge) is a physical-design caveat, acceptable for area/routability
evidence now, and structurally safe in this schedule: **reads and writes of m_cm/e_cm happen in
different FSM states**, so the half-cycle-early latch write is invisible to the dataflow.

## AC-1 — per-implementation area, sky130 probe flow (measured 2026-07-18)

864-bit bank slice (6 lanes × 8b × 18 rows), identical harness through
`hw/syn/asic_probe.sh` (sky130_fd_sc_hd tt_025C_1v80):

| variant | area | µm²/bit |
|---|---|---|
| flat DFF (current elaboration) | 33,139 µm² | 38.4 |
| **latch array** | **13,053 µm²** | **15.1 (2.54×)** |
| wide byte-masked | 33,137 µm² | 38.4 (area-neutral — its win is routing, see below) |

A 48-bank micro-plane P&R proxy (dff vs wide) was **inconclusive** — ORFS width-collapsed the XOR
harness 8× and the structures became indistinguishable — so the routability verdict (AC-2) was
moved to the real core, below. (Trap recorded: periodic per-bank rotations cancel under
XOR-reduction; always check the synthesized flop count against expectation first.)

## AC-2 — the real-core prototype (`BP_RF_REGFILE`)

`hw/bp_relay_banked.sv` now carries the restructuring as an opt-in compile style (`-DBP_RF_REGFILE`;
the DFF baseline is untouched and stays the default): `bp_mvm_rf` wide byte-masked slot banks,
latch-array storage inside `bp_mcm_cell`/`bp_ecm_cell`. No schedule, quantisation or memory-map
change — storage only.

**Bit-exactness gate (`make -C hw bpbankedrf`): PASS.** 40/40 full decodes bit-identical to the
fixed-point golden at all three bank configs, worst latency identical to the DFF baseline
(8/24: 3750, 12/36: 2640, 16/48: 2085 cycles). The DFF baseline (`bpbanked`) still passes 40/40
at all three.

### SKY130 synth probe (16/48 M8 core, rounds=1 header — same vehicle as the flat probe)

| | flat DFF (baseline probe) | `BP_RF_REGFILE` |
|---|---|---|
| std-cell logic area (mapped) | 1.88 mm² (message arrays all blackboxed `$mem`) | 2.44 mm² (m_cm/e_cm decode+mux logic now explicit; excludes raw latch cells + m_vm arrays, below) |
| message-array storage | 122.8 kbit in 1,499 `$mem` arrays (1,108× 8b×9 + 164× 8b×18 + 227 misc) | **55,788 latch bits** (m_cm/e_cm, `$_DLATCH_P_`, ≈0.63 mm² at dlxtp) + **23,616 bits in 48 wide `$mem` arrays** (m_vm — exactly the baseline's 164×144 m_vm bits, consolidated) + 227 misc `$mem` |
| flops | ~13.1 k | 13.1 k (dfxtp 11,098 + edfxtp 1,983) — unchanged, as intended |

The latch conversion also *shrinks the stored-bit count*: yosys `opt_merge` deduplicates
55.8 kbit of live latches out of the 79.8 kbit flat m_cm/e_cm capacity (identical rows collapse
once storage is exposed as individual latches rather than opaque `$mem`).

## AC-2 — ORFS P&R routability verdict (sky130hd)

**ORFS synthesis: 4.54 mm² (−14 % vs flat-DFF's 5.28 mm²) and the clock-tree sink count drops
2.3×:** 11,486 `dfxtp` (bit-identical to the flat run — the core didn't change, only the storage)
+ 31,539 `edfxtp` (the flop-mapped wide m_vm arrays + core enables; was **87,274**) +
**55,737 `dlxtp` latches** (m_cm/e_cm storage, 0.837 mm² → 15.0 µm²/bit, matching the AC-1
micro-measurement). Net: 43 k clock-tree flop sinks instead of 98.8 k. (The synth-level sink-count
drop is real, but note the routing section below — the *placed* clock tree is where this bites.)

Full-flow GRT is unusable as the metric on either core (both fail to converge — see the two flow
notes below), so congestion is read from bounded `global_route` (`-allow_congestion`) probes at
matched conditions. **The honest result is mixed, and it does not support the original premise
that restructuring the memories alone clears the routability wall.**

**Pre-CTS signal congestion** (`3_5_place_dp.odb`, ideal clock, no clock-tree NDRs — isolates the
datapath), 3 congestion iterations, 20 % util:

| core | met2 usage | met2 congestion | congested GCells | Σ overflow | signal wirelength |
|---|---|---|---|---|---|
| flat-DFF | 74.9 % | 127,697 | 12,024 | 25,643 | 91.6 M µm |
| `BP_RF_REGFILE` | 75.8 % | 89,420 | 11,911 | 22,587 | 84.2 M µm |

On the datapath alone the restructured core is **marginally better** — −30 % met2 congestion,
−12 % overflow, −8 % wirelength — but met2 *usage* is essentially tied (~75 %). This is a modest
improvement, **not** the wall-clearing the memory-count reduction (1,272 tiny `$mem` mux trees →
55.7 k latch bits + 48 wide arrays) suggested it would be. (An earlier draft claimed "~4× less
overflow"; that was an artifact of comparing runs with mismatched congestion-iteration counts —
withdrawn.)

**Post-CTS congestion** (`4_cts.odb`, real clock tree, 0 congestion iterations — raw route), 20 %
util:

| core | met2 usage | met2 congestion | wirelength | clock NDRs disabled |
|---|---|---|---|---|
| flat-DFF | 78.0 % | 513,389 | 86.0 M µm | 0 |
| `BP_RF_REGFILE` | 81.7 % | 403,310 | 81.9 M µm | **1000+ (capped)** |

Adding the clock tree **inverts the modest pre-CTS edge**: RF's met2 *usage* is now slightly
*worse* (81.7 % vs 78.0 %), though its total congestion magnitude stays lower. Both cores are
severely met2-congested (78–82 %) — neither routes cleanly at 20 % util. And the RF core triggers
**1000+ clock-NDR disables where the flat core triggers none** — the restructuring introduces a
new clock-tree pathology (below) that the flat core did not have.

**Verdict: necessary for area, not sufficient for routability.** The memory restructuring is a
clear win on area (−14 %, § below) and clock fan-out (43 k vs 98.8 k sinks), and marginally helps
signal congestion — but at 20–30 % util on sky130hd the core is **still not routable**, and the
limiter has broadened from the met2 datapath-mux wall to include clock-tree congestion. The real
levers for a placed core are lower utilization, a clock-tree/NDR strategy (fewer, larger clock
buffers; relaxed clock NDR), and — most decisively — **a commercial node with more than sky130's
five routable metals**, which is the spec's actual prototype target (§ 7). sky130hd routability is
a genuine multi-layer-demand problem, not one that memory shape alone resolves.

**Two flow pathologies, both recorded (each cost real wall-clock):**

1. *CTS `repair_timing` crash.* ORFS CTS repair aborts with
   `[CRITICAL ODB-0445] No undo_updateField support for type dbTechNonDefaultRule` after ~32 min
   of grinding the unbuffered latch-D scatter paths (WNS −9.5 ns at 10 ns) and tripping the
   resizer's journal-undo on an NDR clock net. Workaround: `SKIP_CTS_REPAIR_TIMING = 1` + a 20 ns
   constraint clock (matches the design intent — latch writes get the full cycle).
2. *Full-flow GRT clock-tree livelock.* After CTS, `global_route` on the full RF netlist does
   **not** hard-fail like the flat core (which stops fast at `GRT-0116`) — it *livelocks*,
   disabling non-default routing on the 43 k-sink clock net one net at a time (1000+ `GRT-0273`
   messages, congestion-iteration counter restarting on each NDR disable) for 20 h+ without
   converging. The bounded 0-iteration probe above is the workaround that extracts a post-CTS
   number without the livelock. Counter-intuitively the *smaller* clock tree (43 k vs the flat
   core's 98.8 k sinks) triggers *more* NDR disabling — the latch-dense placement packs the clock
   sinks tighter, worsening local clock-net congestion. This is the practical reason full-flow
   P&R of the RF core is currently harder than the flat core, not easier.

**Fmax/power are not quotable from these runs** (repair skipped ⇒ paths are unbuffered-fanout
limited; post-CTS WNS −7.6 ns at 20 ns is the same unbuffered-fanout artifact the flat probe hit
at 691 ns→11.5 ns after ABC buffering). The honest logic-depth speed remains the buffered-ABC
~87 MHz from the synth probe; route timing/power await a repair-clean, routable flow.

## Impact on the spec § 5 area budget

The synthesis result **tightens the § 5 area estimate and confirms its direction**: the flop-only
M8 core is 5.28 mm², the restructured core 4.54 mm² (−14 %) in sky130hd standard cells, with the
message storage now 55.7 k latch bits (0.84 mm² at 15.0 µm²/bit) + 48 wide m_vm arrays. This
validates the spec's "~4–5 mm² all-in M8 at SKY130, ≪ 1 mm² at 22FDX" area line — area was never
the binding constraint and the latch conversion improves it.

The **routability** finding, however, revises the § 5/§ 8 plan: restructured memories are
necessary but do not by themselves yield a placed sky130hd core. The § 8 "OpenROAD P&R → placed
Fmax/area/power" item should be **retargeted to the commercial prototype node** (22FDX, § 7),
where the extra routing layers remove the multi-layer-demand wall that both core variants hit on
sky130's five metals. sky130hd remains useful for *area/synthesis* de-risking (done) but is not
the right vehicle for a placed-and-routed timing/power number.

## AC-2 follow-up — the routability wall is a node problem, not a design problem (ASAP7)

The sky130hd verdict above ("necessary but not sufficient; needs a node with more routing
resources") predicted that a modern node would route the exact same netlist. That prediction is
now **validated on an open proxy node — ASAP7** (7 nm predictive PDK, no NDA, standard ORFS
platform) — before any commercial-PDK spend.

Same pre-elaborated `BP_RF_REGFILE` netlist (`bp_m8rf_elab.v`), full ORFS flow, `asap7`
platform, CORE_UTILIZATION = 45 % (more than 2× the util at which sky130 failed), M2–M7 routing
(6 layers — only one more than sky130's 5).

**Synthesis mapped faithfully:** 43,025 `DFFHQNx1` flops (bit-identical to the sky130 flop count)
+ 55,703 `DHLx1` transparent latches (the m_cm/e_cm storage → ASAP7's D-latch cell) — the netlist
is technology-independent and re-maps cleanly. Cell area 0.074 mm², **placed-and-routed die
0.163 mm²** at 45 % util (vs sky130's 4.54 mm² synth and route-infeasible ~44 mm² extrapolation —
a ~60× node shrink and, more to the point, an *actually routed* die where sky130 has none).

**Global routing — clean, zero congestion, no livelock:**

| layer | usage | total congestion (overflow) |
|---|---|---|
| M2 | 51.2 % | **0** |
| M3 | 51.2 % | **0** |
| M4 | 38.9 % | **0** |
| M5 | 30.8 % | **0** |
| M6 | 43.4 % | **0** |
| M7 | 40.5 % | **0** |

GRT converged in 16 extra iterations (no restart), **0 clock-NDR disables** (the sky130 livelock
does **not** recur), routed 676,861 nets, total wirelength 7.17 M µm. Contrast the identical
netlist on sky130hd: flat-DFF hard-failed `GRT-0116` (met2 96.6 %); the RF core livelocked on
clock-NDR congestion for 20 h+ without converging. On ASAP7 the same core routes to **zero
overflow on every layer at 45 % util**.

**Detailed routing (TritonRoute) completes** — the design physically routes (677 k nets, routed
wirelength 6.17 M µm, **0 antenna violations**), which sky130hd never reached on this netlist
(it failed/​livelocked at global route). Residual DRC is **207 violations** (~0.03 % of nets):
49 Short + 46 Metal-spacing (the real ones, closable by a small util drop) and 111 ASAP7 Lef58
`EolKeepOut`/`EndOfLine`/`CutSpacingTable`/Cut — the predictive PDK's finicky cut/EOL rules, which
ASAP7 reference designs routinely leave in the dozens–hundreds. This is a *near-complete* route
with a fine-tuning residual, categorically different from sky130's "cannot route at all". A lower
util (≈35 %) or a production PDK closes the remainder; not pursued here — the routability verdict
is already unambiguous.

**Conclusion.** The routability wall is a property of **sky130's 130 nm routing budget**, not of
the decoder RTL. A modern node routes the same netlist comfortably — and notably it does so with
essentially the same layer count (6 vs 5), so the win is the finer **track pitch / density** of
the advanced node, not merely "more metals". This retires the § 8 doubt: the spec's 22FDX
prototype target (§ 7) is the right vehicle for a placed timing/power number, and this ASAP7
result is the open-tooling evidence that it will route — obtained at compute cost, before any
NDA/PDK commitment.

**Flow note.** ASAP7 post-GRT `repair_timing` segfaults on this netlist (same tool-bug class as
the sky130 CTS `repair_timing` ODB-0445 crash — the huge-fanout clock net / latch paths trip the
resizer). `SKIP_INCREMENTAL_REPAIR = 1` skips it; the routability verdict is GRT's, unaffected.
As on sky130, **timing/power are therefore not quotable** from this run — the finish-stage
setup/hold violation counts are the *unrepaired* state (no `repair_timing` ran) at an arbitrary
1 ns probe clock, not an achieved Fmax. This run answers routability only; a repair-clean timing
number is a separate follow-up (and best taken on the real 22FDX PDK).

## Reproduce

```bash
# co-sim gate (both styles), from repo root
make -C hw bpbankedrf && make -C hw bpbanked

# sky130 probe of the RF core (EPYC box staging under /data/asicprobe/m8rf)
hw/syn/asic_probe.sh <staging-dir> bp_relay_banked 10000 \
  -DBP_RF_REGFILE check_minsum.sv var_update.sv bp_relay_banked.sv

# ORFS: pre-elaborate (full `memory`, latches pass through as $dlatch), then
yosys -l elab_rf.log elab_rf.ys   # read_slang -DBP_RF_REGFILE ... ; proc; opt; memory; write_verilog
# full flow reaches CTS but GRT livelocks on clock-NDR congestion (see pathology 2):
docker run --rm -v /data/asicprobe:/work openroad/orfs:latest bash -c \
  "cd /OpenROAD-flow-scripts/flow && make WORK_HOME=/work/orfs_out_m8rf DESIGN_CONFIG=/work/orfs/m8rf/config.mk"

# congestion is therefore read from BOUNDED global_route probes on the flow's own odbs
# (stdbuf -oL for live output; openroad at tools/install/OpenROAD/bin, not on PATH):
#   pre-CTS signal congestion  — read_db 3_5_place_dp.odb; global_route -allow_congestion \
#                                -congestion_iterations 3 -congestion_report_file r.rpt
#   post-CTS (real clock tree) — read_db 4_cts.odb;        global_route -allow_congestion \
#                                -congestion_iterations 0   (0 iters avoids the NDR livelock)
# met2 "Usage (%)" + "Total Congestion" come from the GRT-0096 summary; Σ overflow = sum of the
# per-GCell `congestion:` values in the report file (note: report file caps at 20,000 violations).
```
