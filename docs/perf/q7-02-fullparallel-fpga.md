# Q7-02 Task B2 — the full-parallel configuration on FPGA

**Step 0 result, 2026-07-30.** The banked core was probed out-of-context across the whole banking
curve, from the geometry that ships to the full-parallel geometry the silicon case rests on. This is
the free half of Task B2: it costs nothing, it runs on hardware we own, and its job is to decide
whether renting a large part is worth doing at all.

**It is.** For the first time in this program a full-parallel configuration looks buildable.

-----

## 1. What was run

`hw/syn/ooc_banked.tcl`, top `bp_relay_banked`, part `xck26-sfvc784-2LV-c` (the KV260's Zynq
UltraScale+), out-of-context, `-flatten_hierarchy none`, clock period **5.0 ns** — the same target the
Task B0 and Task B2 `NGROUP` probes used, so every Fmax in this document is comparable with those.

Four geometries, ascending, serially, one staging directory each (`check_minsum.sv`, `var_update.sv`,
`bp_relay_banked.sv`, plus that geometry's generated `bb_gross_tanner.svh`). Ascending order was
deliberate: 144/864 had never been synthesised by anyone, and the fully-unrolled B0 probe had peaked at
47.9 GB, so the growth curve had to survive the last point failing. It did not fail — the whole sweep
took **70 minutes** and peaked at **9.1 GB**.

Raw reports: `/data/asicprobe/b2fp/{16x48,48x288,144x432,144x864}/` on the EPYC box.

## 2. Results

| W/V | GC/GV | cycles | CLB LUTs | % KV260 | LUTRAM | CLB regs | CARRY8 | DSP | BRAM | Fmax | peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16/48 *(ships)* | 9/18 | 2085 | 94,182 | 80.4 % | 12,640 | 17,579 | 2,945 | 164 | 0 | 177.7 MHz | 4.8 GB |
| 48/288 | 3/3 | 789 | 291,098 | 248.6 % | 37,856 | 67,274 | 18,045 | 698 | 0 | 155.8 MHz | 6.4 GB |
| 144/432 | 1/2 | 605 | 490,944 | 419.2 % | 11,808 | 175,964 | 32,845 | 717 | 0 | 164.3 MHz | 7.6 GB |
| **144/864** | **1/1** | **543** | **803,518** | **686.1 %** | 0 | 266,545 | 61,436 | 696 | 0 | **154.0 MHz** | 9.1 GB |

Percentages are against the KV260's 117,120 CLB LUTs. Only the 16/48 row fits that part, which is
exactly why it is the one that ships.

## 3. The finding: banking preserves the clock

This is the result that matters, and it is not what the previous two probes predicted.

**Fmax falls 13 % — 177.7 → 154.0 MHz — while the design grows 8.5× in LUTs.** Set that against the
other two members of this design family, measured on the same flow at the same period:

| core family | how it scales | Fmax across the family |
|---|---|---|
| fully unrolled (B0) | one shot | **30.7 MHz** |
| partially unrolled, `NGROUP` swept 9× (B2) | area *grows* as `NGROUP` shrinks | **flat 16.5–17.5 MHz** |
| **banked, this sweep** | area grows 8.5× | **177.7 → 154.0 MHz** |

The unrolled families are an order of magnitude slower and stay slower no matter which knob is turned,
because their critical path is a crossbar that does not care how much arithmetic is stamped around it.
The banked core does not have that path. Its critical path at 144/864 is
`pc_reg[2] → check min-sum (gmcm[7198]) → ehat_w_reg[9]`: 25 logic levels, 6.49 ns, of which **61 % is
routing estimate** — a normal deep-arithmetic path, not a fabric-wide mux.

That is the structural reason the banked core wins, and it is now measured across the full curve rather
than inferred from the one geometry that fits a $300 board.

## 4. Calibration — the most useful number here

An out-of-context Fmax is an estimate with no placement behind it, so on its own it means little. But
this sweep contains its own calibration point: **16/48 is a design we have actually built, routed and
run on silicon at 133.332 MHz.** OOC claimed 177.7 MHz for it.

> **OOC overstates achieved Fmax by 1.33× on this core, this flow, this part.**

Applying that same de-rate to the full-parallel geometry:

| | cycles | Fmax | latency |
|---|---|---|---|
| 16/48, OOC estimate | 2085 | 177.7 MHz | 11.7 µs |
| **16/48, achieved on silicon** | 2085 | **133.332 MHz** | **15.64 µs** |
| 144/864, OOC estimate | 543 | 154.0 MHz | 3.53 µs |
| **144/864, calibrated projection** | 543 | **~115 MHz** | **~4.7 µs** |

**~4.7 µs against the shipped 15.64 µs — 3.3× better.** The de-rate is conservative for a large part:
the KV260 is a `-2LV` low-voltage speed grade, and the datacentre parts below are `-2` or `-3`.

**Sub-microsecond is still not on the FPGA road.** 543 cycles in 1 µs requires **543 MHz**, and nothing
in this family is within 3.5× of that. The FPGA result narrows the sub-µs question to the ASIC clock
alone — where 543 cycles at our one measured ASAP7 figure of 686 MHz is 0.79 µs, with the gated-clock
caveat of `docs/perf/q7-02-asap7-timing.md` still unresolved.

## 5. Does it fit a large FPGA?

On paper, comfortably. AWS's current FPGA instance family (F2) carries the **Virtex UltraScale+ HBM
VU47P**:

| resource | 144/864 needs | VU47P has | utilisation |
|---|---|---|---|
| CLB LUTs | 803,518 | 1,303,680 | **61.6 %** |
| CLB registers | 266,545 | 2,607,360 | 10.2 % |
| CARRY8 | 61,436 | ~162,960 | 37.7 % |
| DSP | 696 | 9,024 | 7.7 % |
| BRAM / URAM | 0 | 2,016 / 960 | 0 % |

61.6 % is inside Task B2 Step 4's 90 % gate with room to spare, and every non-LUT resource is nearly
empty. Compare what the same gate said about the alternatives: the unrolled core needed **838 % of a
KV260**, and every `NGROUP` setting needed **601–1474 %**.

**Three reasons not to treat 61.6 % as settled:**

1. **This is synthesis, not implementation.** Every net in the timing report is marked `unplaced` and
   61 % of the critical path is an estimate. Post-route utilisation rises and post-route Fmax falls.
2. **The critical path already carries a fanout-7941 net** (`pc_reg[2]`). At 62 % occupancy on a large
   die that is where congestion lives. Register replication is the obvious mitigation and has not been
   tried.
3. **VU47P is a multi-die (SSI) part.** A monolithic 800k-LUT design gets partitioned across super
   logic regions, and SLR crossings are expensive in exactly the way this path cannot afford. The SLR
   count and the partitioning cost were **not** established here and must be checked before any
   conclusion about achieved Fmax on that part.

## 6. What to rent — and why it is not what the plan said

The program document's Task B2 Step 1 says "rent AWS F1 (VU9P), budget ~€100–500". Both halves of that
are now wrong.

### 6.1 F1 is gone

The F1 platform reached end of life at the end of 2025 and is not available to new users; AWS's
second-generation FPGA instances (**F2**, VU47P) replace it. `hw/syn/f1_144x864.tcl` — a file the plan
asks us to create — would target a part that can no longer be rented. Target VU47P instead.

### 6.2 We do not need an FPGA instance at all

Task B2 Steps 2 and 3 ask for **utilisation and achieved Fmax**. That is synthesis and implementation.
It is not a running bitstream, and nothing in this task requires one. AWS's own development kit
documentation is explicit that builds do not require an F-family instance — the FPGA is needed only to
*run* an image — and recommends ordinary compute instances with **≥ 4 vCPU and ≥ 32 GiB**, x86 only.

Our 144/864 synthesis peaked at 9.1 GB and took 30 minutes. Place-and-route on a 62 %-occupied VU47P
will want several times that; 64 GiB is the safe size.

| what | pick | why |
|---|---|---|
| instance | **z1d.2xlarge** — 8 vCPU, 64 GiB, ~$0.744/h | highest sustained clock AWS sells; Vivado P&R is largely single-threaded, so clock beats core count |
| alternative | m5.4xlarge / c5.4xlarge, 64 GiB class | cheaper per hour, slower wall-clock |
| AMI | **FPGA Developer AMI (F2)** | this is the entire reason AWS is involved — see below |
| storage | ~100 GB gp3 | checkpoints and reports |
| runtime | ~40 instance-hours for two configs (64/192 and 144/864) plus retries | synthesis 0.5 h each; P&R on a large part realistically 4–12 h |

**Estimated total: $30–60**, not €100–500. Even booking an `f2.6xlarge` at $1.98/h out of caution lands
near $80.

### 6.3 Why not Vultr, Hetzner, or our own box — the blocker is the licence, not the machine

This is the part worth being precise about, because the instinct to avoid AWS is otherwise correct and
cheap here.

Vivado ML **Standard** — the free edition — does not support Virtex UltraScale+ at all. A large-part
build needs Vivado ML **Enterprise**, which is roughly **$4,395 node-locked**. The one place that
licence comes bundled at no software charge is the **AWS FPGA Developer AMI**, and that licence is
valid only on EC2 and only for the parts AWS itself deploys.

So:

- **Vultr / DigitalOcean / Hetzner:** neither offers FPGA instances, but that is beside the point —
  they would be perfectly good build machines. What they cannot supply is the licence.
- **Our own EPYC box:** already exceeds AWS's recommended build spec by a wide margin, has 123 GB of
  RAM, and just ran the entire Step 0 sweep for free. It is blocked on exactly one thing: the licence.
- **Consequence:** if a Vivado Enterprise licence is ever obtained by another route — purchase, or the
  AMD University Program / Europractice academic route this program already contemplates for 28 nm
  sign-off — then **every future large-part build, including the Track P2 appliance-v2 bitstream, runs
  on hardware we already own at zero marginal cost.** At $4,395 against a €100k programme budget that
  is worth pricing deliberately rather than renting repeatedly by reflex.

The rent-now / buy-later split is therefore: **rent a build instance for ~$40 to answer Task B2**, and
treat the licence purchase as a separate decision driven by Track P2, not by this measurement.

## 7. Step 0 verdict

**Proceed to Step 1.** The rule written into the task was that another 838 %-class result would mean
falling back to 64/192 without spending anything. This is not that result: 61.6 % of a rentable part, a
clock that barely degrades across an 8.5× area growth, and a calibrated 4.7 µs against the shipped
15.64 µs.

What Step 1 must still establish, and Step 0 explicitly does not: **post-route** utilisation and Fmax on
VU47P including SLR cost; whether the fanout-7941 control net survives placement; and the same numbers
for 64/192 as the fallback. And what no FPGA result can establish: sub-microsecond. That needs 543 MHz
on 543 cycles, and it lives on the ASIC road or nowhere.

-----

# Step 1 result, 2026-07-31 — it fits, and the clock collapses

Both configurations were implemented end to end on the real part. **The fit gate passes and the latency
case for going full-parallel does not.**

## 8. What was run

AWS EC2 `z1d.2xlarge` with the FPGA Developer AMI (Vivado **2025.2**), part
**`xcvu47p-fsvh2892-2-e`** — the AMD Virtex UltraScale+ HBM device in AWS's F2 instances. Full flow:
`synth_design → opt_design → place_design → phys_opt_design → route_design`, then post-route
utilisation, timing and congestion. Script: `hw/syn/impl_vu47p.tcl`. Clock target 5.0 ns, default
directives, **no floorplanning and no fanout constraints** — remember that when reading the Fmax.

The instance lived 7.3 hours and cost about **$5.50**, against the $30–60 budgeted in §6.2 and the
€100–500 the program document originally assumed.

| | synth | opt | place | phys_opt | route | total |
|---|---|---|---|---|---|---|
| 64/192 | 9 min | 1.5 min | 30 min | 6 min | 18 min | **65 min** |
| 144/864 | 60 min | 6 min | **170 min** | 14 min | 80 min | **5 h 40 min** |

## 9. Results

| | cycles | CLB LUTs | % VU47P | CLB regs | CARRY8 | DSP | BRAM | Fmax (post-route) | **latency** |
|---|---|---|---|---|---|---|---|---|---|
| 64/192 | 913 | 247,434 | **19.0 %** | 60,561 | 10,893 | 593 | 0 | **150.4 MHz** | **6.07 µs** |
| 144/864 | 543 | 994,700 | **76.3 %** | 245,464 | 39,668 | 2,952 | 0 | **97.3 MHz** | **5.58 µs** |
| *(shipped 16/48, KV260)* | 2085 | — | — | — | — | — | — | 133.3 MHz | 15.64 µs |

**Post-route utilisation barely moves off post-synthesis** — 64/192 grew 244,941 → 247,434 (+1.0 %) and
144/864 actually *shrank*, 1,000,035 → 994,700. On this part, post-synthesis utilisation is a reliable
predictor of the final number.

### The fit gate passes

76.3 % is inside Step 4's 90 % criterion. **A 144/864 instance places and routes on a rentable FPGA**,
which is the thing Task B2 existed to find out and which no previous configuration in this program
managed. Memory is not a constraint anywhere: BRAM and LUTRAM are at zero and registers at 9.4 %. The
design is pure combinational logic and LUTs are the only resource that matters.

### The latency case for full-parallel does not pass

| | cycles | Fmax | latency |
|---|---|---|---|
| 64/192 | 913 | 150.4 MHz | 6.07 µs |
| 144/864 | 543 | 97.3 MHz | 5.58 µs |

**1.68× fewer cycles, 1.55× slower clock, net 1.09× — for 4.0× the area.** On this part, with this
flow, the full-parallel configuration is not worth building: 64/192 delivers 92 % of the latency in a
quarter of the device, leaving room for a host interface, and it implements in 65 minutes rather than
six hours.

## 10. Why the clock collapsed — and it is not what the area suggests

The obvious story would be "the design got bigger, so the logic got deeper". The timing report says the
opposite:

```
Slack (VIOLATED):     -5.273ns
Source:               pc_reg[1]/C
Destination:          gmcm[5562].u_mcm/mem_reg[6]/D
Data Path Delay:      10.251ns  (logic 0.750ns (7.3%)  route 9.501ns (92.7%))
Logic Levels:         8  (CARRY8=1 LUT3=1 LUT5=1 LUT6=5)
```

**Eight logic levels, and 92.7 % of the path is wire.** Logic contributes 0.75 ns of a 10.25 ns path.
For comparison the Step 0 out-of-context estimate had *25* logic levels and 61 % routing — the
implemented design is logically **shallower** and still three times slower.

The path is a small control counter, `pc_reg[1]`, reaching a min-sum cell with index 5562. That is one
register driving thousands of loads scattered across a die large enough to need three super logic
regions. The congestion report agrees: congested windows at **level 5–6** with 57–91 % LUT occupancy,
and `LAG_LAG` tiles — the Laguna registers that implement SLR crossings — appear inside them.

So the wall is **control-net distribution**, not computation. This is exactly the risk Step 0 named
(fanout 7941 on `pc_reg[2]`) and it materialised on the same net.

### 97.3 MHz is a floor for this design, not its ceiling

This was a default-directive run. Every standard mitigation for precisely this failure mode is
untried:

- `MAX_FANOUT` on the `pc` counter, or explicit per-bank replication in RTL — the textbook fix for one
  register feeding thousands of loads;
- SLR-aware floorplanning (a Pblock per bank group) so that crossings are chosen rather than left to
  the placer;
- `phys_opt_design` with aggressive directives, and higher-effort place/route;
- constraining at an achievable period rather than at 5.0 ns, which the tool missed by 5.3 ns.

None of these were attempted. A follow-up run that only replicates the control registers could
plausibly recover a large part of the 35 % clock loss, and it is cheap — about $4 of instance time.

## 11. Where Step 0's projections were wrong

Stated plainly, because both errors were in the flattering direction:

| | Step 0 projected | Step 1 measured | error |
|---|---|---|---|
| 144/864 utilisation | 61.6 % of VU47P | **76.3 %** | 15 points optimistic |
| 144/864 latency | ~4.7 µs | **5.58 µs** | 19 % optimistic |

The utilisation miss has two confounded causes that one run cannot separate: Step 0 synthesised on the
KV260's Zynq UltraScale+ part with Vivado **2024.2**, and Step 1 used Virtex UltraScale+ HBM with
**2025.2**. The same RTL produced 803,518 CLB LUTs there and 1,000,035 here, 24 % more. Cross-part
LUT-count extrapolation is therefore worth about ±25 %, and should be quoted that way in future.

The latency miss follows from the 1.33× de-rate calibrated on 16/48 being too gentle for a design that
is eight times larger and spans three dies. The calibration method was sound; it simply cannot see a
congestion regime that the calibration point never entered.

## 12. What this means for the ASIC — carefully

**Do not read the 35 % clock collapse as an ASIC prediction.** Its causes are specifically FPGA: fixed
interconnect with finite routing tracks, and hard SLR boundaries crossed through Laguna registers. An
ASIC has neither. Custom routing, buffer-tree insertion and clock-tree synthesis handle a
high-fanout control net far better than an FPGA fabric does.

**But do read it as a warning about an assumption the silicon case rests on.** The only ASIC-node
frequency this project has ever measured — 686 MHz on ASAP7 — was measured on the **16/48** geometry,
not on 144/864. The program has been quoting 543 cycles ÷ 600–1000 MHz as though the clock were
geometry-independent. This result is direct evidence that it is not: on real implementation, the
full-parallel geometry lost a third of its clock to distribution effects that scale with design size.

If even half that penalty transferred to 28 nm, 543 cycles at ~450 MHz would be **1.2 µs** and
sub-microsecond would be gone. That is now a measured risk rather than a hypothetical one, and it makes
**Task B3 — running 144/864 through the ASIC flow — the decisive measurement of the whole silicon
track**, ahead of any further FPGA work.

## 13. Step 1 verdict

1. **Fit: PASS.** 76.3 % post-route on a rentable part, under the 90 % gate. Task B2 Step 4's condition
   is met, so the ASIC *area* estimate is trustworthy in the sense the plan intended.
2. **Full-parallel on FPGA: not worth it.** 1.09× latency for 4.0× area. Track P2's appliance v2 should
   ship **64/192**: 6.07 µs at 19 % of the device, with the rest free for a host interface.
3. **The clock wall is control-net distribution, and it is unaddressed.** One cheap follow-up run with
   register replication would tell us whether 97.3 MHz is the design or the defaults.
4. **Sub-microsecond is untouched by all of this** and now looks harder, not easier. It needs 543 MHz on
   543 cycles; no FPGA is close, and the ASIC number it depends on was measured on a geometry eight
   times smaller than the one being sold.

-----

## Appendix — a tooling defect this sweep exposed

`ooc_banked.tcl` reported `CLBLUT=-1` on all four points. The utilisation report labels the row
`CLB LUTs*` — with a footnote asterisk — and the extraction pattern `LUTs\s*\|` does not match it, so
the authoritative count silently became -1 while the cell-level `cellLUT` count (which counts LUT
primitives, not packed CLB LUTs, and reads 33 % high) looked plausible enough to be mistaken for it.

Every CLB-LUT figure in this document was read out of `util_banked.rpt` by hand. The pattern is fixed
in this commit. `ooc_core.tcl` had the identical bug and was fixed during the `NGROUP` sweep; the fix
was never carried across.
