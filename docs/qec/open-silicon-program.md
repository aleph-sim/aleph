# Open-Silicon QEC Decoder Program — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended)
> or `superpowers:executing-plans` to implement Phase A task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Track O, Track P (P1) and Track S (Phase A, Phase B) are implementable now. Phases C–E are
> money/decision gates executed by a human, and each names its exit criterion explicitly.

**Goal:** Ship a cheap, ready-to-use, openly-licensed real-time decoder for bivariate-bicycle qLDPC codes
— as a working product from day one, and as a sub-microsecond open ASIC at the end — so that any QEC lab
can deploy real-time decoding for a few hundred euro instead of the $10–30 k FPGA or $30 k+ GPU it
replaces. Design, data, results and bring-up software permanently open. Silicon funded from a fixed
personal budget of €50–100 k.

**Architecture:** Three tracks run **in parallel**, not in sequence.
**Track O (open)** makes the existing work usable and citable by outsiders — licence, CI, size, docs,
paper, data DOI, ecosystem plugins. **Track P (product)** ships a usable decoder appliance at every
stage of capability, starting with hardware that already exists and works today. **Track S (silicon)**
makes the numbers trustworthy, proves the full-parallel configuration on rented FPGA, rehearses the
tape-out flow on a free fully-open 130 nm shuttle, then tapes out the 28 nm part and gives the dies away.

The tracks are coupled at exactly one point: **Track P is Track S's demand gate.** We do not ask labs
whether they would use a decoder — we ship them one that works, count who actually deploys it, and let
that number authorise the silicon spend.

**Tech stack:** SystemVerilog RTL (`hw/`), Verilator co-simulation against the Rust golden model
(`crates/aleph-qec`), Vivado for FPGA, OpenROAD-flow-scripts for ASIC P&R, ASAP7 (predictive proxy) and
IHP SG13G2 / SKY130 (open PDKs) and TSMC 28 nm (production target).

## Global constraints

- **Everything published stays open.** RTL, scripts, campaign data, P&R configs, bring-up software.
  No NDA-encumbered artefact may become a dependency of the *published* flow.
- **Budget ceiling: €100 k total, self-funded.** Any stage that would exceed it stops the program instead.
- **No stage may start before its predecessor's exit criterion is met, in writing, in this file.**
- **Honest numbers only.** Every performance claim in public material names its measurement conditions
  and its comparison baseline, including where we lose. This is the project's reputational asset.
- **Correctness gates are non-negotiable.** Bit-exact co-simulation against the software golden model
  gates every RTL change (repo Golden Rule 1); a banking change is an RTL change.
- Existing repo conventions apply: no git worktrees, one issue one PR, `Closes #<issue-number>`,
  `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` must pass.

---

## 0. The decision that shapes the whole program

Two facts, established 2026-07-26, that cannot both be satisfied:

1. **Open PDKs stop at 130 nm.** SKY130, GF180MCU and IHP SG13G2 are the complete set. There is no
   open 28 nm or 22 nm PDK. At 130 nm this core runs ~90–150 MHz.
2. **Sub-microsecond decoding needs an ASIC clock.** The measured cycle model
   (`cycles = LEGS·ITERS·(GC+GV+7) + (2·GV+GC+1)`, `docs/qec/asic-architecture.md` §5) floors at
   **544 cycles** for the full-parallel 144/864 configuration. 544 cycles is 0.91 µs only at 600 MHz
   and 4.09 µs at the KV260's 133 MHz.

Therefore: **a fully-open-flow chip cannot be faster than the $300 FPGA we already have, and a chip that
beats it cannot have a fully-open flow.** The program resolves this by splitting "open" into two claims
and honouring both separately:

- **The design is open** — RTL, testbenches, campaign data, P&R scripts, bring-up software, results.
  This holds at every node, including 28 nm. Anyone with PDK access can re-run and re-fabricate.
- **The flow is open** — reproducible end-to-end by anyone with no NDA. This holds only at 130 nm, and
  is delivered by the Phase D shuttle, which is deliberately kept in the program for that reason alone.

The PDK itself is the one thing we cannot open, because it is not ours.

---

## 1. Who needs this chip and why — the case the money rests on

### 1.1 The one-line proposition

> A ~€500–1000 chip that does what a $10–30 k FPGA or a $30 k+ GPU does today: real-time relay-BP
> decoding of bivariate-bicycle qLDPC codes in under a microsecond — free to any lab that asks, with
> the RTL published.

### 1.2 Why an FPGA is not already the answer

It is, until you need sub-µs. Measured, on our own hardware:

| Platform | Config | Cycles | Clock | Worst-case latency |
|---|---|---|---|---|
| KV260 (~$300) — **shipped, measured** | 16/48 banked | 2085 | 133.332 MHz | **15.64 µs** (0.85 µs median early-exit) |
| KV260 | 144/864 full-parallel | 544 | — | **does not fit** (M9c: LUT 162 %, BRAM 113 %) |
| Large FPGA ($10–30 k), projected | 144/864 | 544 | ~200–300 MHz | 1.8–2.7 µs |
| **28 nm ASIC, projected** | 144/864 | 544 | 600 MHz–1 GHz | **0.54–0.91 µs** |

The full-parallel configuration — the only one that reaches the latency floor — **does not fit in an
affordable FPGA**. That is the entire technical justification for silicon, and it is a measured fact,
not an aspiration.

### 1.3 Named user segments, honestly ranked

1. **Superconducting-qubit QEC groups without an NVIDIA budget.** Round time ~1 µs, so they genuinely
   need sub-µs. Today their options are a large FPGA or a GPU on NVQLink. **This is the real user.**
2. **Control-electronics builders** (Qblox, Quantum Machines, QuantWare, the QICK/Fermilab community).
   They want a decode block inside the controller. They will take the RTL; some would take a chip.
3. **Neutral-atom and trapped-ion groups.** Round times of 100 µs–ms mean our *existing FPGA build is
   already real-time for them*. They need the design and the bitstream, **not** the chip. Do not
   count them in the silicon case.
4. **Teaching and reproducibility use.** Served by Phase D's fully-open 130 nm part, not by the 28 nm one.

### 1.4 What we are not

We are not first, and the plan must never claim otherwise:

- **Riverlane** has a hardware decoder (Local Clustering Decoder) in Nature Communications and Nature
  Electronics, <1 µs/round, deployed with Rigetti, OQC, Infleqtion, ORNL; Deltaflow 3 (FPGA/ASIC) is
  due late 2026. They have raised $120 M+ and employ ~198 people.
- **NVIDIA** ships relay-BP — our algorithm — inside CUDA-Q QEC 0.6 over NVQLink, adopted by
  Quantinuum, IQM, Pasqal, Alice&Bob, Q-CTRL.
- **QEC Labs** is a startup on hardware-native QLDPC decoders (GPU/FPGA/ASIC).
- In the *open* lane, arXiv:2603.16203 demonstrates a 446 ns open-source decode-feedback loop on ZCU216
  — but for the **surface code at d=3**, a different and easier code family than BB/gross qLDPC.

Our defensible claim is narrow and true: **the first open-source, silicon-validated, sub-µs relay-BP
decoder for bivariate-bicycle qLDPC codes, distributed free.** Everything public must stay inside that
claim.

### 1.5 Assets that make this credible

- Bit-exact co-simulation harness, 25+ Verilator targets in `hw/Makefile`, software golden ↔ RTL.
- **0 mismatches in 10⁶ × 3 shots on silicon** (Q7-06, matched-prior campaign).
- Q7-07: `valid_flag` heralds nearly every logical error (A(p) ≈ 1.0) — an independently publishable
  architectural result.
- A measured banking scaling law with a banking-invariant pipeline tail (Q7-08 / PR #475).
- OpenROAD P&R evidence on two nodes: sky130hd (4.54 mm², routing-infeasible at 5 metals) and ASAP7
  (die 0.163 mm² @45 % util, 0 GRT congestion, 207 residual DRC, route completes).

---

## 2. The three tracks at a glance

| Track | Phase | What | Direct cost | Exit criterion (written into this file before anything downstream starts) |
|---|---|---|---|---|
| **O** | F1–F8 | Licence, CI, repo size, docs, paper, DOI, CUDA-Q plugin, grants | €0 | `hw/` re-licensed with a patent grant, hardware gates green in CI, paper on arXiv, data DOI minted |
| **P** | **P1** | **Appliance v1 — KV260, ships now** | ~€0.3–3 k | One-command deploy; ≥ 1 external lab running it on their own hardware |
| **P** | P2 | Appliance v2 — large-FPGA build, sub-2 µs | €0 (bitstream only) | Published bitstream + utilisation/Fmax on a real large part |
| **P** | P3 | Appliance v3 — ASIC module on a carrier board | €5–15 k | Module BOM under €200 at 100 units, schematic and layout open |
| **S** | A | Trustworthy ASIC numbers (#322) | €0 (compute) | Post-route Fmax ≥ 600 MHz with timing repair actually run, **and** a gate-level-sim-derived power number |
| **S** | B | Full-parallel proof on rented FPGA | ~€100–500 | 64/192 **and** 144/864 pass bit-exact co-sim; P&R on a real large part under 90 % utilisation with a quoted Fmax |
| **S** | **C** | **Demand gate — measured, not surveyed** | €0 | **≥ 5 labs have deployed appliance v1/v2 on their own hardware; ≥ 2 named as early adopters for silicon** |
| **S** | D | Fully-open 130 nm shuttle (flow rehearsal) | €0–20 k | Working silicon back, co-sim vectors pass on it, measured power published |
| **S** | E | 28 nm part | €45–93 k fab | Dies distributed; measured sub-µs on silicon published |

**Start now, in parallel:** O (F1–F3), P1, S-A, S-B. Phase C is not a separate activity — it is a
counter on Track P that reaches its threshold or does not. Phases D and E are locked until it does.

Budget envelope: everything after Phase C ≤ €100 k inclusive of packaging, test boards and shuttle fees.

---

## Track P — the cheap ready product

**Design rule for the whole track:** at every capability level, something works and is documented well
enough that a stranger can deploy it in an afternoon. We never hold a release waiting for silicon.

**Latency tiers and who each one already serves:**

| Version | Hardware | Config | Latency | Real-time for |
|---|---|---|---|---|
| **v1 — now** | Kria KV260, ~$300 | 16/48 banked | 15.64 µs worst, 0.85 µs median early-exit | neutral atoms, trapped ions (round times 100 µs–ms) — **already real-time for them today** |
| v2 | large FPGA ($10–30 k) | 144/864 | ~1.8–2.7 µs projected | most platforms except the fastest superconducting loops |
| v3 | AD-1 ASIC module | 144/864 | 0.54–0.91 µs projected | superconducting, ~1 µs round times |

v1 is the important row: **for two of the four qubit modalities the product is finished and merely
undistributed.** That is the cheapest possible demand experiment.

### Task P1: Appliance v1 — make the KV260 build deployable by a stranger

**Files:**
- Create: `hw/product/README.md`
- Create: `hw/product/deploy.sh`
- Create: `hw/product/interface-spec.md`
- Modify: `.github/workflows/release.yml` (attach the bitstream to releases)

- [ ] **Step 1: Publish the prebuilt bitstream as a release artefact**

Do not ask users to run Vivado. The bitstream `bp_p005.bit` used for the Q7-07 silicon campaign already
exists and is validated at `mismatch = 0` over 100 000 shots.

- [ ] **Step 2: Write the interface specification**

The product's contract with the outside world, and the single most important design decision on this
track. Specify all three paths and state which is primary:
  - **AXI-Lite / AXI-DMA** for designs embedding the core in their own FPGA — already implemented
    (`hw/bp_axi_top_banked.v`, `hw/bp_axi_wrap_banked.sv`), and the Q7-06 batched DMA path is measured.
  - **Low-latency serial** (LVDS or Aurora over SFP+) for use as an external decoder box — the path a
    control-electronics vendor would actually wire up. Not yet implemented; specify before building.
  - **Slow/simple** (SPI or plain parallel) for atom and ion platforms where 15 µs is already ample.

- [ ] **Step 3: Write `deploy.sh`**

One command from a bare KV260 to a running decoder: flash, load bitstream, install the Python driver
(`hw/sw/bp_stream_banked_ler_kv260.py` is the working starting point), run a self-test against the
shipped vectors, print the measured latency.

- [ ] **Step 4: Verify on a fresh board, not on the development board**

Expected: a person who has never seen this repository gets a decoding self-test to pass without asking
us a question. If that fails, the product is not ready, regardless of what the RTL does.

- [ ] **Step 5: Commit and cut a release**

```bash
git add hw/product/
git commit -m "[product] Appliance v1: one-command KV260 deploy, bitstream in releases"
```

### Task P2: Appliance v2 — publish the large-FPGA build

Depends on Phase B, which produces exactly this artefact as a by-product.

- [ ] **Step 1: Publish the 144/864 bitstream and constraints for the part Phase B used**
- [ ] **Step 2: State the achieved Fmax and utilisation, and the resulting latency, in the release notes**
- [ ] **Step 3: Do not buy these boards for other people.** Publish; let the labs that own them use them.

### Task P3: Appliance v3 — the ASIC module

Locked until Phase E produces dies.

- [ ] **Step 1: Design an open carrier board** — chip, power, clock, the serial interface from P1 Step 2,
  and a USB or Ethernet control path. Schematic and layout published under CERN-OHL-S.
- [ ] **Step 2: Target a BOM under €200 at 100 units**, so the module genuinely replaces a $10–30 k FPGA
  rather than merely being smaller.
- [ ] **Step 3: Assemble a first batch of 20–50**, ship free to the Phase C early adopters first.
- [ ] **Step 4: Publish the recipient list and every measured latency number obtained on real setups.**

### Task P4: Support policy — the thing that makes it a product rather than a repo

- [ ] **Step 1: Write it down.** What we answer, how fast, what we do not support, how versions are
  numbered, what "stable" means for the interface spec. One page.
- [ ] **Step 2: Commit to interface stability across v1→v2→v3.** A lab that integrates against v1 must
  not have to rewrite when the ASIC arrives. This single promise is what makes the cheap product a
  bridge to the chip instead of a distraction from it.

---

## Phase A — Make the ASIC numbers trustworthy (issue #322)

**Why this is first:** every later number — the €45 k fab quote, the sub-µs claim, the datasheet —
descends from Fmax and power. Today both are provisional: the ASAP7 flow ran with
`SKIP_INCREMENTAL_REPAIR=1` and `SKIP_CTS_REPAIR_TIMING=1`, and power used ORFS's default switching
activity rather than our traffic.

**Root cause already established (2026-07-26):** `repair_timing` segfaults with a null dereference in
OpenSTA's CRPR arrival pruning, reached through GRT's incremental parasitics update:

```
rsz::Resizer::repairSetup → SetupLegacyPolicy::repairEndpoint
  → est::EstimateParasitics::updateParasitics → grt::GlobalRouter::updateDirtyRoutes
  → grt::FastRouteCore::layerAssignment → updateSlacks → sta::Sta::slack
  → sta::Search::findAllArrivals → ArrivalVisitor::pruneCrprArrivals → sta::Path::minMax  ← SIGSEGV
```

**Second finding:** the critical path is launched by a *latch* (`gmcm[250].u_mcm.mem[18]$_DLATCH_P_`)
whose clock insertion is **1.37 ns**, against 0.47 ns for an ordinary flop. TritonCTS stops at the
generic `AND3x1` gating cell and the 55,703-latch array beyond it is buffered by a *serial*
`load_slew` daisy-chain. The ASAP7 platform sets `DONT_USE_CELLS += ICG*`, so no integrated
clock-gating cell was available to synthesis. **Fmax is limited by a flow artefact, not by the datapath.**

### Task A1: Quantify the gated-clock penalty

**Files:**
- Create: `/data/asicprobe/clk_struct.tcl` (on the EPYC box, already staged)
- Create: `docs/perf/q7-02-asap7-timing.md`

- [ ] **Step 1: Collect the clock-network structure**

Run on `root@195.154.249.85`:

```bash
cd /data/asicprobe && setsid nohup ./run_clkstruct.sh </dev/null >/dev/null 2>&1 &
```

Expected output in `/data/asicprobe/clk_struct.log`: `DISTINCT_LATCH_CLK_NETS`, `DISTINCT_DFF_CLK_NETS`,
`LATCH_NET_SINK_TOP20`, `LATCH_SINKS_TOTAL`.

- [ ] **Step 2: Record how many distinct gated-clock nets exist**

If the count is small (< ~2000), the SDC route in Task A2 is viable. If it is very large, skip A2 and
go straight to A3 (re-synthesis with ICG cells enabled).

- [ ] **Step 3: Commit the finding**

```bash
git add docs/perf/q7-02-asap7-timing.md
git commit -m "[Q7-02] ASAP7 clock-network structure: gated-clock nets driving the latch array"
```

### Task A2: Declare the gated clocks and re-run from CTS

**Files:**
- Create: `/data/asicprobe/orfs/m8rf_asap7_gclk/config.mk`
- Create: `/data/asicprobe/orfs/m8rf_asap7_gclk/constraint.sdc`

- [ ] **Step 1: Generate the `create_generated_clock` statements**

For every net found in A1 driving latch `CLK` pins, emit into the SDC:

```tcl
create_generated_clock -name gclk_<n> -source [get_ports clk] -divide_by 1 [get_pins <gate>/Y]
```

- [ ] **Step 2: Re-run the flow from CTS onward with the new SDC**

TritonCTS builds balanced trees for declared clocks, which the plain AND gate previously prevented.

- [ ] **Step 3: Compare `report_clock_min_period` against the 686.13 MHz baseline**

Baseline for comparison (unrepaired, post-route, RCX parasitics, SDC period 1000 ps):
`period_min = 1457.44 ps, fmax = 686.13 MHz`, `wns max = -622.73`, setup skew 940.67 ps.

- [ ] **Step 4: Commit the result, pass or fail**

A negative result is a result — record it either way.

### Task A3: Work around the `repair_timing` segfault

Two independent hypotheses; test the cheap one first, one variable at a time.

**Files:**
- Create: `/data/asicprobe/repair_h1_crpr.tcl`
- Create: `/data/asicprobe/repair_h2_placement.tcl`

- [ ] **Step 1: H1 — disable CRPR, the exact function that crashed**

```tcl
read_db /work/orfs_out_m8rf_asap7/results/asap7/bp_relay_banked/base/5_1_grt.odb
set_crpr_enabled false
estimate_parasitics -global_routing
repair_timing -setup_margin 0 -hold_margin 0 -repair_tns 100 -verbose
```

Expected if H1 holds: the run completes instead of dying in `pruneCrprArrivals`.

- [ ] **Step 2: If H1 fails, H2 — avoid the GRT incremental path entirely**

```tcl
read_db /work/orfs_out_m8rf_asap7/results/asap7/bp_relay_banked/base/4_1_cts.odb
estimate_parasitics -placement
repair_timing -setup_margin 0 -hold_margin 0 -repair_tns 100 -verbose
```

The crash is reached through `updateDirtyRoutes`; placement-based parasitics never call it.

- [ ] **Step 3: If both fail, report upstream and stop**

File an OpenROAD issue with the stack trace and the ODB. Do not attempt a third workaround —
per `superpowers:systematic-debugging`, three failures means questioning the approach, and the
fallback (quote the unrepaired number with its conditions stated) is legitimate.

- [ ] **Step 4: Commit whichever outcome occurred**

### Task A4: Real switching-activity power via gate-level simulation

This simultaneously closes the long-standing gap that no gate-level co-simulation of an ASIC netlist
has ever been run (`docs/qec/asic-architecture.md` §6 claims the chain "retargets to gate-level
unchanged" — never tested).

**Files:**
- Create: `hw/tb_bp_gate_asap7.sv`
- Create: `hw/Makefile` target `bpgate-asap7`
- Modify: `docs/perf/q7-02-asap7-timing.md`

**Inputs available:** netlist `6_final.v` (78 MB), ASAP7 behavioural models at
`/OpenROAD-flow-scripts/flow/platforms/asap7/verilog/stdcell/*.v`, and the existing co-sim vectors
(`hw/bp_dec_vectors.txt`, `hw/bp_circ_vectors.txt`).

- [ ] **Step 1: Compile the netlist with Verilator, zero-delay functional mode**

Toggle counts do not need timing annotation. `specify` blocks are ignored by default.

- [ ] **Step 2: Drive one representative decode window and dump VCD**

Use ~300 cycles of steady-state BP iteration rather than the full 2085-cycle window — a full-window VCD
over 654 k cells is tens of GB. Dump a second, separate stretch covering the idle/early-exit regime.

- [ ] **Step 3: Verify the gate-level netlist reproduces the golden output**

This is the co-simulation gate, not an optional extra. Expected: bit-exact match on the same vectors
the RTL passes.

- [ ] **Step 4: Feed the VCD to OpenSTA and report power**

```tcl
read_power_activities -vcd /work/gate_window.vcd
report_power
```

Baseline to beat for honesty, not for pride: ORFS default-activity power was **0.655926 W**
(internal 0.339 W, switching 0.317 W, leakage 6.4e-5 W) with a PSM worst IR drop of 33.9 mV (4.40 %).

- [ ] **Step 5: Convert to energy per window and compare with the §5 budget**

`E = P × cycles / f`. Note in the write-up that energy is roughly **banking-invariant** while latency is
not, and that a 28 nm part will draw several times the 7 nm figure — the §5 "≤1 µJ per window" budget
almost certainly needs restating at the real target node.

- [ ] **Step 6: Commit**

### Task A5: Close #322 with the honest note

**Files:**
- Modify: `docs/qec/BACKLOG.md` (Q7-02 acceptance criteria, ~lines 1318–1319)
- Modify: `docs/qec/asic-architecture.md` §8 open items
- Create: `docs/perf/q7-02-asap7-timing.md` (final form)

- [ ] **Step 1: Write the closure note**

It must state, without softening: gate-level co-sim now exists (A4) or does not; the streaming core
never fit silicon (M9c: LUT 162 %, BRAM 113 %); runt frames (`slices < W`) remain uncovered by co-sim;
64/192 and 144/864 were cycle-counted in Verilator before Phase B, not synthesised; and none of the
hardware gates run in CI.

- [ ] **Step 2: Open the PR with `Closes #322`**

Use the issue number, not the PR number — P0-06/07/08/11 all merged with the wrong reference.

---

## Phase B — Prove the full-parallel configuration (unblocks everything downstream)

**Why:** the entire silicon case rests on 144/864 at 544 cycles. That number is currently a Verilator
cycle count, never synthesised. If it does not close timing or does not fit, the chip is not worth
building and we find out for ~€300 instead of ~€45 000.

### Task B1: Bit-exact co-simulation at 64/192 and 144/864

**Files:**
- Modify: `hw/Makefile` (extend the existing `bpbankedscale` geometry list)

- [ ] **Step 1: Add both geometries to the scale sweep**

`bpbankedscale` already covers 5 geometries; 8/24, 16/48, 32/96 and 64/192 have cycle counts
3750 / 2085 / 1283 / 913 respectively.

- [ ] **Step 2: Run and confirm bit-exactness**

```bash
make -C hw bpbankedscale
```

Expected: pass at every geometry, matching the model
`cycles = LEGS·ITERS·(GC+GV+7) + (2·GV+GC+1)`.

- [ ] **Step 3: Commit**

### Task B2: Synthesise and place-and-route on a large FPGA

**Files:**
- Create: `hw/syn/f1_144x864.tcl`
- Create: `docs/perf/q7-02-fullparallel-fpga.md`

- [ ] **Step 1: Rent the part**

AWS F1 (VU9P) with the FPGA Developer AMI carries Vivado licences for the part; the alternative is a
borrowed Alveo/VPK120. Budget ~€100–500 of instance time.

- [ ] **Step 2: Run synthesis and implementation for 64/192, then 144/864**

- [ ] **Step 3: Record utilisation and achieved Fmax for both**

- [ ] **Step 4: Decide**

If 144/864 fits a large FPGA under 90 % utilisation, its ASIC area estimate is trustworthy.
If it does not fit even there, revise the target configuration to 64/192 (913 cycles, 1.52 µs at
600 MHz) and re-cost — still a useful chip, no longer a sub-µs one.

- [ ] **Step 5: Commit both results**

### Task B3: Re-derive the 28 nm area and fab cost from real synthesis

**Files:**
- Modify: `docs/qec/asic-architecture.md` §7

Current estimate, from log-interpolating our own two ORFS data points (4.54 mm² @130 nm sky130hd,
0.163 mm² @7 nm ASAP7; fitted slope 1.139 in log-node):

| Config | Est. area @28 nm | mini@sic 28 nm fab cost |
|---|---|---|
| 16/48 | ~0.8 mm² | ~€10.6 k |
| 64/192 | ~2.4 mm² | ~€23.5 k |
| **144/864** | **~4.7 mm²** | **~€45 k** |

Pricing basis: TSMC 28 nm HPC+ RF mini@sic, €10,609 for the first mm² plus €919 per additional 0.1 mm²,
1 mm² minimum, registration 3 months ahead.

- [ ] **Step 1: Replace the interpolation with the Phase B synthesis-derived cell count**

- [ ] **Step 2: State the uncertainty band explicitly** — a two-point node interpolation is worth about
  ±2×; at the pessimistic end 144/864 is ~10 mm² ≈ €93 k, which is still inside budget but leaves no
  room for tooling.

- [ ] **Step 3: Commit**

---

## Phase C — Demand gate (measured by deployments, not by interviews)

**This phase costs nothing and blocks all silicon spending.** Its purpose is to answer "who needs it and
why" with names rather than reasoning — and, because Track P ships a working product first, with
*deployments* rather than opinions. A lab that has gone to the trouble of running our decoder on their
own hardware has demonstrated demand in a way no survey can.

### Task C1: Write the one-page chip spec sheet

**Files:**
- Create: `docs/qec/ad1-datasheet-draft.md`

- [ ] **Step 1: State, on one page:** target latency and the configuration that achieves it, host
  interface, power, die area, package, licence, price to the recipient (free), and what it does *not*
  do (surface-code MWPM, arbitrary codes, on-chip syndrome extraction).

- [ ] **Step 2: State the comparison honestly** — against a large FPGA, against GPU+NVQLink, and
  against Riverlane's published <1 µs.

### Task C2: Interview 12–20 named groups

- [ ] **Step 1: Build the target list** across the four segments in §1.3, prioritising superconducting
  groups not already committed to Riverlane (Rigetti, OQC, Infleqtion, ORNL, NQCC, QuEra, Atlantic
  Quantum) or to NVQLink (Quantinuum, IQM, Pasqal, Alice&Bob, Q-CTRL).

- [ ] **Step 2: Ask each the same four questions.** Do you decode in real time today, and with what?
  What is your per-QPU budget for decoding? Would a free chip plus open RTL change what you do?
  What interface would it have to speak?

- [ ] **Step 3: Record every answer verbatim** in `docs/qec/ad1-demand-validation.md`, including the
  refusals and the silences. A thin file is the most valuable possible outcome if it is true.

### Task C3: The gate

- [ ] **Decision.** Proceed to Phase D/E only if **≥ 5 labs have actually deployed appliance v1 or v2 on
  their own hardware, and ≥ 2 agree in writing to be named early adopters for the chip.** Interviews
  (C2) seed the funnel and shape the interface spec; they do not satisfy the gate on their own —
  deployments do.

- [ ] **Fallback if the gate fails.** Stop at FPGA plus published RTL, which is a complete and valuable
  outcome — `ROADMAP.md` §0 already says so — and redirect the budget into Track O and Track P: more
  seeded boards, better documentation, the paper, the CUDA-Q plugin. Note honestly that a failed gate
  most likely means the product is not yet good enough to deploy, not that nobody wants decoding; in
  that case the correct response is to fix the product, not to abandon the goal.

---

## Phase D — Fully-open 130 nm shuttle (flow rehearsal, and the only fully-open artefact)

**Purpose is explicitly *not* performance.** At 130 nm this core cannot beat the KV260. What it buys:
the whole tape-out flow learned end-to-end at near-zero PDK cost, a real measured-power datapoint, and
the one deliverable that is reproducible by anyone on Earth with no NDA.

- [ ] **Task D1:** Apply for IHP SG13G2 free MPW area. IHP offers free area in MPW runs for
  non-economic use (university education and research), the PDK is open, and OpenROAD already supports
  it. A sustainable free/low-cost MPW scheme for the open-source community was announced as under
  development for 2026 — confirm current terms at
  `https://www.ihp-microelectronics.com/services/research-and-prototyping-service/fast-design-enablement/open-source-pdk`.

- [ ] **Task D2:** If IHP free area is unavailable, fall back to ChipFoundry chipIgnite (SKY130,
  **$14,950**, ~10 mm² Caravel user area, 100 QFN packaged parts, ~5 months, 3 shuttles planned for
  2026). Note that Efabless, the original operator, shut down; ChipFoundry took the platform over.

- [ ] **Task D3:** Tape out a *subset* — one check-bank plus one variable-bank plus the regfile and the
  AXI interface — not the whole core. The goal is flow rehearsal and a power datapoint.

- [ ] **Task D4:** Bring-up on the existing PYNQ harness re-pointed at the chip; run the same co-sim
  vectors; publish the measured power.

**Exit criterion:** working silicon back, co-sim vectors pass on it, power published.

---

## Phase E — The 28 nm part (the chip that is actually useful)

**Entry:** Phase A, B and C exits all met. Not before.

- [ ] **Task E1:** Secure PDK and sign-off tooling. This is the real blocker, not the fab fee.
  TSMC 28 nm needs foundry sign-off DRC/LVS decks (Calibre-class), which OpenROAD does not replace.
  Routes worth pursuing, in order: an academic/institutional partner with Europractice membership;
  the Chips JU **EuroCDP** platform, which has a framework agreement with Siemens EDA explicitly aimed
  at lowering EDA cost for SMEs and start-ups; a commercial broker as the expensive fallback.

- [ ] **Task E2:** Harden the design for a real PDK — memory strategy in particular. The Q7-08 latch
  regfile avoids a memory-compiler licence, which is a real cost saving; verify it survives at 28 nm
  or budget for an SRAM compiler.

- [ ] **Task E3:** Register at least 3 months before the shuttle deadline (mini@sic requirement).

- [ ] **Task E4:** Tape out 144/864 (or 64/192 if Phase B forced the fallback). Fab cost ≈ €45 k at the
  central area estimate; ≈ €93 k at the pessimistic end.

- [ ] **Task E5:** Packaging, test board, bring-up. Budget €10–40 k. This is routinely underestimated.

- [ ] **Task E6:** Distribute the dies. 40 diced samples come with the mini@sic slot; order more.
  Publish a one-page application form; ship free to labs; keep a public recipient list.

**Exit criterion:** measured sub-µs decode latency on silicon, published, with dies in other people's hands.

---

## Track O — Openness and distribution (starts now, runs continuously)

This track is not a fallback and not a finishing touch. It is the precondition for Track P having any
users at all: an undiscoverable, unlicensed, untested repository ships to nobody, and a product with no
users cannot serve as Track S's demand gate. F1–F3 are prerequisites for the appliance v1 release.

- [ ] **Task F1: Re-licence `hw/`.** The repo is MIT, which has **no explicit patent grant** — a blocker
  for anyone whose lawyers must approve pulling RTL into a chip. Move `hw/` to **Apache-2.0** or
  **CERN-OHL-P v2**; keep the Rust crates MIT or dual MIT/Apache-2.0 as is idiomatic in that ecosystem.

- [ ] **Task F2: Put the hardware gates in CI.** `.github/workflows/` currently contains only
  `bench.yml`, `ci.yml` and `release.yml` — no `verilator`, no `make -C hw`. 58 make targets exist and
  none of them are enforced automatically. An open hardware project whose hardware is not tested in CI
  does not earn trust.

- [ ] **Task F3: Shrink `hw/`.** It is **2.8 GB**, including `_bp*build` directories. Move artefacts to
  releases; a multi-gigabyte clone deters every casual evaluator.

- [ ] **Task F4: Ship "decoder-in-a-box".** Pre-built KV260 bitstream, Python driver, one command,
  one page of documentation. This is the artefact most people will actually use.

- [ ] **Task F5: Publish the paper.** The M0→M9c ladder plus silicon campaigns plus the Q7-07 heralding
  result. `docs/perf/qec-q7-fixed-bp.md` is already a 1863-line master record. Mint a Zenodo DOI for
  the campaign CSVs so the data is citable.

- [ ] **Task F6: Write a CUDA-Q QEC decoder plugin.** NVQLink is where every real-time QEC integrator
  already is; a plugin lets them A/B our decoder against the GPU path on their own hardware. Using the
  competitor's platform as our distribution channel is the cheapest reach available.

- [ ] **Task F7: Register in the community indexes** — Error Correction Zoo decoder list,
  `qosf/awesome-quantum-software`.

- [ ] **Task F8: Apply for aligned funding.** Unitary Foundation microgrants ($4 k, worldwide, open
  quantum projects). NGI Zero Commons (NLnet, €21.6 M committed across 2026–27, **submissions currently
  paused** pending the EC Tech Sovereignty package of 2026-06-03 — watch for reopening).

---

## 3. Corrections owed to existing documents

`docs/qec/asic-architecture.md` §7 is now known to be wrong in three ways and must be fixed as part of
Phase A:

1. **Efabless is defunct.** chipIgnite is operated by ChipFoundry at **$14,950**, not "~$10–15 k" from
   Efabless, with 3 shuttles planned for 2026 rather than quarterly.
2. **22FDX is under-priced by roughly 3–5×.** The table says "~€15–60 k"; Europractice 2026 lists
   **€17,820/mm²** (€16,200 discounted) with a 4 mm² minimum — i.e. **~€71 k floor**.
3. **The TSMC mini@sic ladder is missing entirely**, and it is the cheapest real route to a
   performance-class node: 65 nm €4,491 + €419/0.1 mm²; 28 nm €10,609 + €919/0.1 mm²;
   16 nm €30,592 + €2,827/0.1 mm², each with a 1 mm² minimum.

`docs/qec/BACKLOG.md` Q7-03 (issue #323) must be rewritten: the acceptance criterion "explicit gate
decision recorded" is now satisfiable, but the decision is **not** the no-go of 2026-07-26. It is
**conditional go, open-hardware, staged, self-funded, gated on Phase C**. Record the date and the
reasoning, including that the commercial thesis was rejected and why.

`ROADMAP.md` §0 and §102 say the Q7 trigger is "funding + a committed QPU-company customer". That rule
was written for a commercial tape-out. It does not fit a community-funded open chip and must be amended
rather than quietly ignored — the replacement trigger is Phase C's gate.

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 144/864 does not fit / does not close timing | Medium | Kills the sub-µs claim | Phase B finds out for ~€300; fall back to 64/192 at 1.52 µs |
| No sign-off EDA access at 28 nm | **High** | Blocks Phase E entirely | Phase E Task E1 is a gate, not a step; EuroCDP and academic partnership are the routes; Phase D proves the flow at 130 nm regardless |
| First silicon dead on arrival | Medium (30–50 % is normal) | €45–95 k lost | Phase D rehearsal on a free/cheap shuttle; conservative interface design; on-chip observability |
| Demand does not materialise (Phase C fails) | Medium | Silicon stops; product and open corpus continue | A legitimate outcome, not a failure — budget redirects into Tracks O and P |
| Product ships but nobody can deploy it unaided | Medium | Phase C gate misreads as "no demand" | P1 Step 4 tests deployment on a fresh board by a fresh person; kill criterion 2 distinguishes a bad product from absent demand |
| GPU path wins before we ship (12–18 months) | **High** | Chip is obsolete on arrival | Accept it. The open RTL, data and paper retain value independently; do not let the chip become the only deliverable |
| Area estimate off by 2× | Medium | €45 k → €93 k | Still inside budget, but leaves nothing for tooling; Phase B replaces the estimate with synthesis |
| Dual-use / export-control friction | Low | Delay | Open publication is the safest posture; take advice before shipping dies across borders |

## 5. Kill criteria — when to stop, decided in advance

Stop the **silicon** track — Tracks O and P continue regardless — if **any** of these becomes true:

1. Phase B shows 144/864 needs more than ~10 mm² at 28 nm, pushing fab past €93 k with no tooling budget left.
2. Phase C's counter stalls below 5 real deployments **after** appliance v1 has been genuinely deployable
   for six months. (Below 5 *before* that, the fault is the product, not the demand — fix the product.)
3. Phase E Task E1 finds no viable sign-off-tooling route within 6 months.
4. Total committed spend would exceed €100 k.
5. A functionally equivalent open chip ships from someone else first — in which case help them instead.

Nothing on this list stops Track O or Track P. The cheap product and the open corpus are the deliverables
that survive every bad outcome, and they are the reason the silicon is worth attempting at all.

## 6. Budget summary

| Line | Low | High |
|---|---|---|
| Track O (licence, CI, repo hygiene, paper, DOI, plugin) | €0 | €0 |
| Track P — appliance v1, seed boards for early labs (5–10 × KV260) | €0 | €3 k |
| Track P — appliance v3 carrier board design + first 20–50 modules | €5 k | €15 k |
| Phase A (compute only) | €0 | €0 |
| Phase B (rented FPGA) | €100 | €500 |
| Phase C | €0 | €0 |
| Phase D (open 130 nm shuttle + test board) | €0 (IHP free area) | €20 k (chipIgnite + board) |
| Phase E fab (144/864 @28 nm) | €45 k | €93 k |
| Phase E packaging, test board, bring-up | €10 k | €40 k |
| **Total** | **~€60 k** | **exceeds ceiling — see kill criterion 4** |

The high column overshoots deliberately: at the pessimistic end of *both* area and tooling, this program
does not fit in €100 k, and the plan says so up front rather than discovering it at the shuttle deadline.
Phase B exists to collapse that uncertainty before any money moves — it costs about €300 and removes the
single largest unknown in the budget.

Note the shape of the spend: **everything that produces a usable product costs under €20 k**, and the
€45–93 k is bought purely to move 1.8–2.7 µs down to 0.54–0.91 µs. That is the honest framing of what the
chip is for, and it is worth restating to anyone who asks why an ASIC is needed when an FPGA exists.
