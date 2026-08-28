# Q7-01 — Decoder ASIC architecture spec + HW/SW partitioning

**Status:** v1 (2026-07-16). Closes the two Q7-01 acceptance criteria: (§ 2–6) architecture,
block diagram, budgets, partitioning rationale; (§ 7) MPW-shuttle cost/timeline. The commercial
gate stays Q7-03: **no silicon money moves before funding + a committed QPU-company customer**
(ROADMAP § 0/§ Q7). This document is the engineering half of that gate.

Every number here is measured, from: `docs/perf/qec-q6-fpga.md` (Q6-03 GPU-vs-FPGA verdict),
`docs/perf/qec-q7-fixed-bp.md` (M0–M9c ladder, Q7-05 power), and
`docs/perf/qec-q7-asic-sky130-probe.md` (open-PDK synthesis of the two production cores).
Estimates derived from them are labelled as such, with their scaling assumption stated.

-----

## 1. Product definition

**A real-time streaming qLDPC decoder chip**: fixed-point relay-BP over circuit-level detector
graphs of bivariate-bicycle codes (anchor workload: the [[144,12,12]] gross code), consuming a
continuous measurement-round stream through a sliding window (W=6, C=2, residual-only carry —
the M9a-swept operating point) with early-exit, and emitting committed corrections plus
per-window validity flags.

**What it is not** (scope fences, from ROADMAP § 2):

- Not a surface-code MWPM/UF accelerator — that market is served (Riverlane Collision
  Clustering, Google's in-house stack). The open frontier is qLDPC + real-time streaming.
- Not cryo-CMOS in v1. Room-temperature placement next to the control electronics; the syndrome
  bandwidth (§ 5) is trivially cable-class. Cryo integration is a later program that reuses the
  architecture, not a v1 requirement.
- Not a general-purpose BP engine. One code family per configuration load (§ 3, D2), one window
  geometry in hardware.

## 2. Measured baseline (the inputs this spec stands on)

| quantity | value | source |
|---|---|---|
| batch decode, rounds=1 gross graph, KV260 @133.3 MHz | 15.64 µs worst / **0.85 µs median** early-exit (2 085 / 113 cyc) | M8 |
| streaming window decode (W=6, C=2), KV260-class fabric | full **16 298 cyc** ≈ 122 µs; early-exit min 722 / **mean 4 898 cyc** ≈ 37 µs | M9b |
| streaming core fits a 117 k-LUT KV260? | **No — 162 % LUT / 113 % BRAM** after four gather realisations (terminal verdict) | M9c |
| same streaming core, SKY130 std cells | **6.45 mm² logic**, 739.5 k gates, 528.7 kbit real memory bits | SKY130 probe |
| M8 core, SKY130 std cells | **1.88 mm²**, 256.5 k gates, 11.5 ns buffered pre-P&R path (~87 MHz tt), 122.8 kbit mems | SKY130 probe |
| FPGA critical path composition (UF, same fabric class) | **76–82 % routing delay** — the tax std-cell P&R removes | Q6-03 |
| FPGA vs GPU energy/decode (UF) | 0.10–0.22 µJ vs ~29–120 µJ (**150–600×**); ASIC typically another 10–100× | Q6-03 |
| KV260 SOM power under decode load | +249 mW load delta, of which ~200 mW is the AXI host loop; PL datapath ≲ 48 mW; **≲ 17 µJ / ≲ 3 µJ per decode** (upper bounds, duty-limited) | Q7-05 |
| decode schedule | 6 legs × 10 iters, Q5.3 messages, γ-disorder ROMs, early-exit first-valid | M5/M8 |

The single most important measured fact: **the KV260 no-fit is a fabric artifact, not a design-size
statement.** The full streaming core is an unremarkable ~740 k-gate block even in 130 nm, and its
"163 BRAM tiles" hold only 528.7 kbit of real state (91 % was 36 Kb-tile quantization).

## 3. Architecture

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                        DECODER ASIC                         │
  syndrome stream      │  ┌───────────┐   ┌──────────────────────────────────────┐   │
  (LVDS/DDR par.,      │  │  ingest   │   │  WINDOW ENGINE (M9b FSM:             │   │
  ~0.3 Gb/s/qubit) ────┼─▶│  + frame  │──▶│  WARM/RUN/WAIT/COMMIT/SLIDE/RELOAD)  │   │
                       │  │  align    │   │  W=6 slice buffer (SRAM)             │   │
                       │  └───────────┘   └───────────────┬──────────────────────┘   │
                       │                                  │ per-window                │
                       │  ┌───────────────────────────────▼──────────────────────┐   │
                       │  │            RELAY-BP CORE (banked, M8 lineage)        │   │
                       │  │  ┌──────────────┐  AS-Waksman   ┌─────────────────┐  │   │
                       │  │  │ check banks  │◀─m_cm write──▶│  var banks      │  │   │
                       │  │  │ (minsum)     │   fabric      │  (var_update)   │  │   │
                       │  │  └──────┬───────┘               └────────┬────────┘  │   │
                       │  │         │        e_cm read fabric        │           │   │
                       │  │  msg regfiles (8b×9 class) · λ/γ tables · addr ROMs  │   │
                       │  │  early-exit syndrome check · leg/iter sequencer      │   │
                       │  └──────────────────────────┬───────────────────────────┘   │
                       │                             │                                │
  corrections /        │  ┌───────────┐   ┌──────────▼─────────┐   ┌─────────────┐   │
  valid flags   ◀──────┼──│  result   │◀──│ commit + residual  │   │ CSR + table │◀──┼── boot/config
  (to controller)      │  │  stream   │   │ unit (C=2/slide)   │   │ load port   │   │   (SPI/APB)
                       │  └───────────┘   └────────────────────┘   └─────────────┘   │
                       └─────────────────────────────────────────────────────────────┘
```

All blocks exist as verified RTL today (`bp_streaming_decoder.sv` window FSM,
`bp_relay_banked_bram[_m].sv` core, `bp_asw.sv`/`bp_benes.sv` fabrics, AXI front-end), bit-exact
against the software golden across 40 trials × 7 slots × 2 exit modes (M9b) — the chip is a
re-targeting of qualified RTL, not a new design.

**Design decisions:**

- **D1 — banked core, not full-parallel, as the v1 baseline.** The banked (16/48) core is the
  qualified, silicon-proven (M8) RTL. Banking scale-up (§ 4 ladder) is the knob for round-rate
  targets; full-parallel (M4-style) is a stretch variant to be qualified only if a customer's
  round budget demands it.
- **D2 — loadable tables, not baked.** All code-specific state (Tanner CSR, bank permutations,
  γ/λ, window seam tables) totals ≤ ~0.6 Mbit (measured): held in SRAM/regfiles, loaded at boot
  via the table-load port. One chip serves a BB-code *family*; the M9c lesson that runtime-data
  fabrics dominate area is unaffected (fabric geometry stays fixed; only contents load).
- **D3 — fixed-point Q5.3 message word, 6×10 relay schedule, γ-disorder ROMs** — the M0-golden
  bit-exact chain is the verification anchor; no precision changes on the silicon path.
- **D4 — early-exit is the product mode.** Median/mean latency is the deployable metric (M8:
  0.85 µs median; M9b: 236/280 windows exit early); full-schedule is the bounded worst case and
  sets the backlog analysis, not the marketing number.
- **D5 — room-temperature, controller-adjacent.** See scope fence; I/O budget in § 5.
- **D6 — memory implementation:** the dominant shape is 8b×9 / 8b×18 message arrays (1 200+
  instances) → flop/latch register files, *not* SRAM macros; only the handful of kbit-class
  tables (window slices, big ROMs) macro. This is a physical-design work item (§ 8), already
  quantified by the probe.

## 4. HW/SW partitioning

| layer | lives on | responsibility | rationale |
|---|---|---|---|
| DEM/graph compilation | host CPU (offline, aleph toolchain) | circuit-level DEM extraction, window graph + bank/fabric table generation, (W, C) calibration sweeps, γ schedules | already exists (`qec_q7_bp_graph` emitter); zero latency coupling |
| system control | controller FPGA / control stack | syndrome routing + framing, per-qubit decoder tiling, retry/fallback policy (Q7-07), telemetry | policy iterates faster than silicon; keeps ASIC datapath-only |
| decode datapath | **ASIC** | window engine + relay-BP core + commit (everything in the diagram) | the only latency/power-critical loop; 76–82 % of FPGA latency is routing tax the ASIC exists to remove |

The ASIC therefore has **no embedded CPU, no firmware** — CSR + tables in, syndromes in,
corrections out. Everything programmable-by-policy stays outside.

## 5. Budgets

### Latency / round rate (the defining budget)

Real-time condition: each window step commits C=2 rounds, so sustained decode requires
`t_window ≤ C · t_round`. At the standard t_round ≈ 1 µs: **t_window ≤ 2 µs**.

Cycle counts are the hardware invariant (measured, W=6): full 16 298; early mean 4 898; min 722.
Clock and banking are the two levers. Ladder (cycles ÷ banking-scale, ÷ f_clk; banking assumed
to scale cycles ~linearly — first-order, to be confirmed per step, M7→M8 measured 2.1× for its
combined step):

| config | f_clk | window mean (early) | per-round mean | window full (bound) | verdict |
|---|---|---|---|---|---|
| KV260 today, 16/48 | 133 MHz | 37 µs | 18.4 µs | 122 µs | 18× over budget (measured) |
| ASIC 16/48 | 600 MHz | 8.2 µs | 4.1 µs | 27 µs | clock alone is not enough |
| ASIC 64/192 (4× banks) | 600 MHz | 2.0 µs | 1.0 µs † | 6.8 µs † | linear-banking; optimistic — see below |
| ASIC 144/864 (full-par) | 600 MHz | 0.9 µs | 0.45 µs † | 3.0 µs † | linear-banking; optimistic — see below |

† These two rows used the "banking scales cycles linearly" first-order assumption. **That
assumption is now measured and does not hold** — see the qualification immediately below; the
64/192 and 144/864 worst-case numbers are ~1.75× / ~2.1× better than reality.

#### Banking-scaling qualification (measured, Q7-08 follow-up)

The banked relay-BP core (`bp_relay_banked`, rounds=1 M8 vehicle) was regenerated and
cycle-measured at five bank geometries via the 40-shot co-sim (worst-case full schedule, all
LEGS·ITERS = 60 iterations; latency is the hardware invariant, identical for the DFF and
`BP_RF_REGFILE` styles):

| banking (W/V) | check groups GC | var groups GV | worst-case cycles | µs @ 600 MHz |
|---|---|---|---|---|
| 8/24 | 18 | 36 | 3750 | 6.25 |
| 12/36 | 12 | 24 | 2640 | 4.40 |
| 16/48 (shipped) | 9 | 18 | 2085 | 3.48 |
| 32/96 | 5 | 9 | 1283 | 2.14 |
| 64/192 | 3 | 5 | **913** | **1.52** |

The counts fit an **exact** closed form: `cycles = 60·(GC+GV) + 60·7 + (2·GV+GC+1)`. The middle
term is the killer: **`60·7 = 420` cycles of per-iteration pipeline tail** (the M8 CHK phase runs
`pc = 0..GC+3` and VAR runs `pc = 0..GV+2` — 7 drain cycles per iteration that exist regardless of
banking). Only the `60·(GC+GV)` "useful work" term shrinks with banking; the 420-cycle tail is
**banking-invariant**, so scaling is sublinear and floored:

- **4× banking (16/48 → 64/192) delivers 2.28× fewer cycles, not 4×** (2085 → 913). The pipeline
  tail is 20 % of the 16/48 cycle budget but **47 %** of the 64/192 budget.
- **Hard floor:** even full-parallel 144/864 (GC = GV = 1) is `60·9 + 4 = 544` cycles = **0.91 µs
  @ 600 MHz** for the worst-case schedule — it cannot go lower without changing LEGS, ITERS, or the
  pipeline depth. The ladder's 0.45 µs full-parallel figure is unreachable by banking.
  **Now measured, not modelled (Q7-02 Task B0 Option A, 2026-07-30): 543 cycles**, 40/40 bit-identical
  to the golden on `bp_relay_banked` at 144/864. The closed form above reads exactly one cycle high at
  every measured point (544 vs 543 here, 2086 vs 2085 at 16/48, 914 vs 913 at 64/192, 3751 vs 3750 at
  8/24), so treat it as `… + (2·GV+GC)` rather than `… + (2·GV+GC+1)`. The floor is a *cycle-count*
  result only: whether a 144/864 instance can be placed and clocked anywhere is Task B2, still open.

**Consequence for the target.** Worst-case-schedule real-time at 64/192 @ 600 MHz is **≈ 1.5 µs**,
not the ~0.87 µs a linear 4× implies; even the full-parallel stretch bottoms out at ~0.91 µs. The
**1 µs/round worst-case budget is therefore not met by banking alone.** The dominant remaining
lever is no longer banking but **ITERS** — the 420-cycle floor is `LEGS·ITERS·7`, so halving the
BP iteration budget halves the floor. The minimum tolerable ITERS at a target LER is exactly what
**Q7-06's silicon-accelerated LER campaign qualifies** — so Q7-06 is now on the critical path to
the latency budget, not just the reliability budget. (The *early-exit mean* column may still scale
better than worst-case, since early exit cuts effective iterations; but that too is set by the
Q7-06 syndrome-distribution data, not assumable.) A secondary lever is a shallower submodule
pipeline (fewer than the M8 +2/+1 drain cycles), which trades Fmax.

**Spec target (revised): 64/192-class banking at ≥ 600 MHz reaches ~1.5 µs worst-case / ~1 µs
early-mean per round — the 1 µs *worst-case* budget additionally requires an ITERS reduction
(Q7-06-qualified) or accepting the early-exit mean.** 600 MHz is a plausible commercial-node
target given the measured 10-gate-level critical path (11.5 ns in 130 nm pre-P&R is
fanout/wire-dominated); it is **not yet a placed number** — and the sky130hd P&R attempt (§ 8,
Q7-08) showed the placed number needs a commercial node (routes clean on ASAP7, not sky130).

#### Iteration-budget qualification on the circuit-level DEM (Q7-06 software precursor)

The banking qualification named **ITERS** as the dominant remaining latency lever once banking is
maxed. This quantifies it at the *real* operating point. The earlier `qec_q7_budget` sweep ran on the
**code-capacity** DEM (one perfect round, independent Z, p≈0.05) — the wrong point for a latency claim.
`qec_q7_circuit_budget` re-runs the `(legs, iters)` budget study on the **circuit-level** DEM (depth-7
syndrome extraction, depolarizing CNOT/idle/init/measure noise, gross code, rounds = d = 12) in the
**sub-threshold** regime, at the shipped 6-leg Q5.3 hardware word (100 000 shots/point; the harness
decodes fixed-point relay-BP in parallel per P5.9-class batch). For each schedule it asks: is the LER
still within Monte-Carlo CI of the full 6×10 schedule, and what worst-case latency does it cost at
64/192 and 144/864 (via the banking model above)?

The smallest ITERS whose LER stays within CI of the full 6×10 schedule is **operating-point-dependent**:

| circuit-level p | LER (full 6×10) | min ITERS within CI | sweeps | worst-case @ 64/192 | @ 144/864 |
|---|---|---|---|---|---|
| 0.001 | 8.2 × 10⁻⁴ | **6** (of 10) | 36 | **0.92 µs** | 0.55 µs |
| 0.002 | 8.4 × 10⁻³ | **8** | 48 | 1.22 µs | 0.73 µs |
| 0.003 | 4.4 × 10⁻² | **10** (none cuttable) | 60 | 1.52 µs | 0.91 µs |

At 100 k shots the CI is tight enough that at p = 0.003, iters = 8 already departs the baseline
(1.09× LER), so no iteration can be shed within strict CI — the full 6×10 is required. At p = 0.001
the schedule tolerates a cut to 6 iterations (36 sweeps) with no statistically-resolvable LER cost.

**Refined conclusion (replaces the earlier indicative code-capacity estimate).** The old sketch —
"worst-case 1 µs needs full-parallel 144/864, not 64/192" — holds and is now grounded in circuit-DEM
data, with the mechanism made precise:

- **144/864 full-parallel meets the 1 µs worst-case budget at full reliability** — 0.91 µs at the full
  6×10 schedule, *no* ITERS reduction, at every p measured. This is the clean way to the 1 µs
  worst-case target; it is the floor established above, and the circuit-DEM LER confirms the full
  schedule is affordable there.
- **64/192 reaches ≤1 µs only by trading LER.** Its full-schedule worst case is 1.52 µs; dropping to
  iters ≤ 6 (36 sweeps → 0.92 µs) buys sub-1 µs but costs ≈ 1.2–1.4× LER at p ≥ 0.002 (outside strict
  CI). Within strict CI, 64/192 does **not** meet the 1 µs worst case at the p = 0.003 operating point.
- The **early-exit mean** (not worst-case) remains a separate, softer lever; its scaling is set by the
  on-silicon syndrome distribution the Q7-06 board campaign (AC-2) will measure, not assumable here.

Net: the spec target stands as **144/864-class banking at ≥ 600 MHz for a full-reliability 1 µs
worst-case round**, with 64/192 a lower-area option that meets 1 µs only under a modest, quantified LER
penalty. Reproduce: `cargo run --release -p aleph-qec --example qec_q7_circuit_budget -- 100000`.

### Area (per node; scaling from the measured 130 nm netlist by published node density ratios — indicative)

| node | streaming core logic | + mems/regfiles (est.) | note |
|---|---|---|---|
| SKY130 (130 nm, measured) | 6.45 mm² | ~10–11 mm² | measured synthesis; chipIgnite envelope |
| GF 22FDX (est. ~30× density) | ~0.2 mm² | **< 1 mm²** | performance-class prototype node |
| 16 nm FinFET (est. ~60×) | ~0.1 mm² | < 0.5 mm² | production node; only post-Q7-03 |

64/192 banking multiplies core logic ~2–3× (fabrics grow sub-linearly per M9c right-sizing data)
— still ≪ 1 mm² at 22FDX. **Area is not a constraint off the FPGA; one decoder per logical
qubit tiles at hundreds per reticle.** That is the machine-scale argument (Q6-03) made concrete.

### Power / energy per decode

Measured anchors: UF-FPGA 0.10–0.22 µJ/decode; relay-BP PL ≲ 17 µJ (full) / ≲ 3 µJ (early) as
duty-limited upper bounds (Q7-05); ASIC vs FPGA typically 10–100× (Q6-03). **Budget: ≤ 1 µJ per
window full-schedule, ≤ 0.2 µJ early-exit mean, ≤ ~100 mW sustained per decoder channel at the
1 µs/round rate.** These are targets consistent with the anchors, not yet simulated numbers;
the two hardening steps are Q7-06 (batched-duty FPGA re-measure → tight µJ) and P&R power
analysis with committed switching activity (§ 8).

### I/O

Per decoder channel, gross code: 144 detector bits/round @ 1 µs ≈ **0.15 Gb/s in** (×2 for X/Z
if both bases on one chip), corrections + flags ≈ ~0.02 Gb/s out. A handful of LVDS pairs or a
DDR parallel bus — no SerDes IP needed at v1; table-load + CSR over SPI/APB-class. Package:
low-pin-count BGA/QFN. This confirms D5: nothing here requires cryo placement or exotic I/O.

## 6. Verification & qualification plan (carried over, already in flight)

- Bit-exactness: the M0→M9b chain (software golden ↔ RTL co-sim, per-mode goldens) is the
  regression harness; it retargets to gate-level simulation unchanged — **verified 2026-08-28 on the
  ASAP7 routed netlist (`make -C hw bpgate-asap7`): the harness retargets, the netlist fails, because
  of 43 802 unrepaired hold violations on the latch register file
  (`docs/perf/q7-02-asap7-timing.md` §4b).**
- **Q7-06 (silicon-accelerated LER campaign) is the pre-freeze gate**: ≥10⁶ shots per operating
  point across a (p, legs, iters) grid on the FPGA prototype, LER within the golden's
  statistical band, *before* any tape-out netlist freeze.
- **Q7-07 (non-convergence fallback)** must be decided (even if "flag-and-report") before
  freeze, since it fixes the result-stream contract.

## 7. MPW-shuttle prototype: cost & timeline (AC-2)

Options, 2026 indicative pricing (re-quote at the Q7-03 gate):

| route | process | user area | cost | silicon back | what it buys |
|---|---|---|---|---|---|
| Tiny Tapeout | SKY130 | ~0.1 mm² tile | ~$300–600 | ~6–9 mo | a test structure (one check/var cell + regfile), tooling practice — not a decoder |
| **Efabless chipIgnite-class shuttle** | SKY130 | ~10 mm² | **~$10–15 k** (incl. ~100+ packaged parts) | ~6–9 mo after tape-in | **the M8 core with margin** (~4–5 mm² all-in, ~50–90 MHz class): physical-design de-risk, regfile plan validation, *measured* silicon power — not real-time performance |
| Europractice MPW | GF 22FDX (or similar) | 3 mm² min | ~€15–60 k + NDA/PDK | ~9–12 mo | the streaming core at performance-class clock: the actual § 5 ladder validated in silicon |
| production shuttle | 16 nm | per-quote | ~$100–300 k | 12+ mo | post-Q7-03 only (funding + customer) |

Recommended sequence (engineering view; money gates per Q7-03):

1. **$0 now:** OpenROAD P&R on sky130hd of the M8 core — placed Fmax/area/power without any
   shuttle commitment (§ 8). This is the highest-information next step and costs compute only.
2. **~$10–15 k, optional:** chipIgnite-class SKY130 prototype of the M8 core (12-month
   end-to-end: ~3–4 mo P&R hardening + DRC/LVS sign-off with open tools, quarterly shuttle
   cadence, ~2 mo bring-up on the existing PYNQ harness re-pointed at the chip). Justified as
   portfolio/validation (real silicon, real µJ numbers, proof the team ships silicon) — **not**
   as a performance demonstrator, and only if that value is wanted pre-gate.
3. **Post-Q7-03 gate:** 22FDX MPW of the streaming core at the § 5 target config —
   the first silicon that is actually real-time — then production node with the customer.

Total pre-gate exposure is bounded at ~$15 k (steps 1–2); the ROADMAP's "$1M+ tape-out"
figure applies only to the post-gate production program (step 3 onward + team + masks).

## 8. Open items (tracked, in dependency order)

1. ~~OpenROAD P&R pass~~ — done on ASAP7 (`docs/perf/q7-02-asap7-timing.md`): die 0.163 mm²,
   Fmax 528–686 MHz setup-only, **0.149 W / 0.31 µJ per window with real activity**. Open instead:
   **hold closure of the latch register file** (43 802 violations, clock-structure fix, not
   buffering) — the gate to any sign-off-able netlist.
2. **Register-file/latch plan** for the 8b×9/8b×18 message arrays (D6) — the one block where
   FPGA and ASIC implementations genuinely diverge.
3. **Banking scale-up qualification** (64/192 per § 5): regenerate tables, re-run the M7-style
   sweep + bit-exact gate at the new banking, confirm the cycles-vs-banks scaling assumption.
4. **(W, C) re-sweep at target config** for the worst-case-sustained story (M9a harness reruns
   as-is).
5. Q7-06 / Q7-07 as § 6 gates.
6. Commercial-node PDK access + re-synthesis at the Q7-03 gate.

-----

*AC-1 (architecture doc: block diagram § 3, budgets § 5, partitioning § 4) — this document.*
*AC-2 (MPW cost/timeline) — § 7.*
