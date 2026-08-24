# Q7-02 Task B3 — the full-parallel geometry through the ASIC flow

**Result, 2026-08-01: 543 cycles at 614.59 MHz = 0.88 µs. Sub-microsecond survives.**

Task B2 Step 1 had just measured the full-parallel core losing **35 % of its clock** on FPGA when it
grew from 64/192 to 144/864, and the whole silicon case rested on a 686 MHz figure measured on the
**16/48** geometry — 8.5× smaller. That made one question decisive: does the clock collapse with
geometry on an ASIC too? It does not. The penalty is **10.4 %**.

-----

## 1. What was run

The same flow, platform and knobs that produced the 16/48 numbers, so the two are directly comparable:

- yosys elaboration to a tech-independent netlist, `-DBP_RF_REGFILE` (the Q7-08 regfile variant),
  peak 4.45 GB, 11.5 minutes, 37.4 MB netlist;
- OpenROAD-flow-scripts, platform **asap7**, `CORE_UTILIZATION = 45`, `PLACE_DENSITY = 0.55`,
  `SKIP_CTS_REPAIR_TIMING = 1`, `SKIP_INCREMENTAL_REPAIR = 1`, SDC target 1000 ps;
- ~20 hours wall clock on the EPYC box, `rc = 0`, flow completed through `6_finish`.

Config: `/data/asicprobe/orfs/fp864_asap7/config.mk`. Outputs: `/data/asicprobe/orfs_out_fp864_asap7/`.

## 2. Results, against the 16/48 baseline

| | 16/48 (shipped geometry) | **144/864 (full-parallel)** | ratio |
|---|---|---|---|
| cycles | 2085 | **543** | 0.26× |
| Design area @ 50 % util | 81,090 µm² | **434,450 µm²** | 5.36× |
| die at that utilisation | 0.162 mm² | **0.869 mm²** | 5.36× |
| **fmax** (`report_clock_min_period`) | **686.13 MHz** | **614.59 MHz** | **0.896×** |
| min period | 1457.44 ps | 1627.11 ps | |
| WNS at the 1000 ps target | −622.73 ps | −1117.97 ps | |
| TNS | −3,947,471 ps | −22,145,750 ps | 5.6× |
| **latency** | **3.04 µs** | **0.88 µs** | **3.44×** |
| **DRC violations** | **0** | **0** | clean both |

### The headline

**543 cycles ÷ 614.59 MHz = 0.884 µs.** The full-parallel configuration is sub-microsecond at ASAP7,
and it routes DRC-clean.

### The geometry penalty is an FPGA artefact, mostly

| | small geometry | full-parallel | clock lost |
|---|---|---|---|
| FPGA (VU47P, post-route) | 150.4 MHz @ 64/192 | 97.3 MHz | **−35.3 %** |
| **ASIC (ASAP7, post-route)** | 686.13 MHz @ 16/48 | 614.59 MHz | **−10.4 %** |

This is the hypothesis B2 Step 1 raised and could not test. The FPGA penalty came from fixed
interconnect with finite routing tracks and hard SLR boundaries crossed through Laguna registers; an
ASIC has neither, and its custom routing and buffer trees absorb the high-fanout control net that
dominated the FPGA critical path. The prediction was that the penalty would not transfer intact. It
did not — roughly a third of it did.

### Area scales sub-linearly

5.36× the area for 8.5× the logic. On FPGA the same pair of geometries cost 8.53× the CLB LUTs
(94,182 → 803,518), so the ASIC scales better, which is expected: no LUT quantisation.

Global placement decomposed the growth usefully. Of the +11.18 % the placer added over the raw netlist,
**routability inflation was only +1.28 %** and **timing-driven area was +9.90 %**. Congestion on ASAP7
is mild; the placer spent its area on critical paths, not on getting wires through.

## 3. What this does not settle, in order of how much it matters

### 3.1 ASAP7 is 7 nm predictive. The target is 28 nm.

**This is now the single largest gap between 0.88 µs and a chip.** ASAP7 is a predictive academic
7 nm PDK; Phase E targets TSMC 28 nm. A 28 nm implementation of the same logic will clock
substantially lower, and nothing in this repository measures by how much.

The project has been quoting 686 MHz as "the ASIC number" since the 16/48 run, so this document is
consistent with existing practice rather than introducing the assumption — but the assumption should be
named. Arithmetic, to show the sensitivity:

| if 28 nm is … | fmax | 543 cycles | sub-µs? |
|---|---|---|---|
| as fast as ASAP7 | 614.6 MHz | 0.88 µs | yes |
| 1.2× slower | 512 MHz | 1.06 µs | **no** |
| 1.5× slower | 410 MHz | 1.33 µs | **no** |
| 2× slower | 307 MHz | 1.77 µs | **no** |

**A 20 % node penalty is enough to lose the claim.** Sub-microsecond is now supported by measurement on
a proxy node, not established on the target one.

### 3.2 Neither run had timing repair

`SKIP_CTS_REPAIR_TIMING` and `SKIP_INCREMENTAL_REPAIR` are set in both configs, because
`repair_timing` segfaults on this netlist class (`docs/perf/q7-02-asap7-timing.md`). The comparison is
therefore like-for-like, and `report_clock_min_period` reports the clock the placed-and-routed design
can actually take — but **neither number is sign-off quality**.

| | 16/48 | 144/864 |
|---|---|---|
| setup violations | 31,581 | 105,311 |
| **hold violations** | 44,704 | **35,616** |
| max slew violations | 2,792 | 1,353 |
| max cap violations | 7 | 0 |
| max fanout violations | 0 | 0 |

Setup violations tripling is expected: the SDC constrains at 1 GHz and the design achieves 614 MHz, so
every near-critical endpoint reports a violation. Slowing the clock fixes those by definition.

**Hold violations do not work that way** — 35,616 of them are unaffected by clock period and need the
repair pass that was skipped. The baseline had more (44,704), so this is not a regression, but "0 DRC"
must not be read as "ready to tape out". A real tape-out needs the repair pass to work, which is
Phase A Task A3 and is still open.

### 3.3 Fab-cost implication

Task B3's original purpose was to replace a two-point log-node interpolation with real synthesis. The
7 nm point for 144/864 is now measured: **0.869 mm²**. Scaling the measured 5.36× onto the existing
16/48 28 nm estimate (~0.8 mm²) gives **~4.3 mm² at 28 nm**, consistent with the ~4.7 mm² and ~€45 k the
plan already carried. The plan's ±2× uncertainty band stands; this narrows the geometry half of it, not
the node half.

## 4. Verdict

1. **Sub-microsecond is measured, not projected — on ASAP7.** 0.88 µs, DRC-clean, at 0.869 mm².
2. **The clock is geometry-dependent, but only mildly on an ASIC** (−10.4 % against the FPGA's
   −35.3 %). The top risk entered after B2 Step 1 is substantially retired.
3. **It is replaced by a narrower and harder risk: the node.** 0.88 µs has 12 % margin, and a 20 %
   node penalty from 7 nm predictive to 28 nm production erases it. Establishing what 28 nm actually
   does to this design is now the decisive open question of the silicon track.
4. **Neither this run nor the baseline is sign-off clean.** 35,616 hold violations remain, unfixed
   because `repair_timing` segfaults. Phase A Task A3 is on the critical path to a real tape-out, not
   merely to a tidy report.
