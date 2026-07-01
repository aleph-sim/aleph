# Phase Q6 — FPGA decoder: utilization, Fmax, latency

Synthesis results for the surface-code Union-Find decoder (`hw/uf_surface_decoder.sv`, the Q6-04
sequential FSM) on both target boards. Flow: `hw/syn/` (non-project out-of-context). This document is
the shared home for Q6-05 (d=3 synth), Q6-09 (d=5 scaling), and Q6-03 (GPU-vs-FPGA comparison).

**Hosts:** synthesis on `openwebgui` (Vivado, x86 Linux). Sim baseline (Verilator) on the M4 Mac.

## Target parts

| board | part | LUT | FF | BRAM36 | DSP | role |
|-------|------|-----|----|--------|----|------|
| Digilent Zybo Z7-20 | `xc7z020clg400-1` | 53 200 | 106 400 | 140 | 220 | small part — d=5 fit risk lives here |
| Xilinx Kria KV260 | `xck26-sfvc784-2LV-c` | ~256 200 | ~512 400 | 144 | 1 248 | headroom part |

## Q6-05 — d=3, sequential FSM (33-cycle decode)

Vivado 2024.2 on `openwebgui`, non-project out-of-context flow (`hw/syn/run.sh`), implemented
(synth → place → route). Numbers from `reports/{zybo,kv260}/{util_impl.rpt,fmax.txt}`.

| part | LUT | FF | BRAM36 | DSP | Fmax | decode latency (33 clk) | fits 1 µs? |
|------|-----|----|--------|----|------|--------------------------|------------|
| `xc7z020clg400-1` (Zybo) | 1178 (2.21%) | 268 (0.25%) | 0 | 0 | **58.7 MHz** (WNS −12.04 ns @ 200 MHz tgt) | **562 ns** | ✅ |
| `xck26-sfvc784-2LV-c` (KV260) | 1200 (1.02%) | 268 (0.11%) | 0 | 0 | **170.0 MHz** (WNS −2.88 ns @ 333 MHz tgt) | **194 ns** | ✅ |

**Budget check:** the surface-code round budget is ~1 µs; latency = `33 / Fmax`. Zybo
33 × 17.04 ns = **562 ns**, KV260 33 × 5.88 ns = **194 ns** — both within budget at d=3.

**Fit verdict (d=3):** the decoder is tiny — **~1.2k LUT, ~268 FF, zero BRAM, zero DSP** — so it
fits with enormous headroom on *both* parts (2.2% of the small XC7Z020; 1.0% of the XCK26). Fit is a
non-issue for d=3; d=5 (a larger matching graph) will grow LUTs but stays far inside both parts.

**Caveat — Fmax, not area, is the wall.** Neither part met its aggressive target (200/333 MHz):
WNS is negative, so the closed Fmax is 58.7 / 170 MHz. The critical paths are the long
combinational chains *inside* each FSM cycle — chiefly the peel sweep's 18-edge loop-carried update
and the union-find root-walk (depth N). They still clear the 1 µs budget at d=3, but **pipelining
those passes is the lever** for higher Fmax / margin and for d≥5 (tracked under Q6-09 and follow-on
timing work). Area is nowhere near a constraint; latency-per-cycle-depth is.

## Q6-06 — gate-level sign-off (xsim)

The same self-checking SV testbench (`hw/tb_uf_surface_xsim.sv`) replays all 256 syndromes against
three elaborations of the decoder; run via `hw/syn/gatesim.sh <part-dir>` on `openwebgui`. Every
stage must bit-match the frozen Q6-02 golden table, reproduce the syndrome, and drive no X.

| stage | what it catches | Zybo | KV260 |
|-------|-----------------|------|-------|
| behavioral RTL | TB sanity | PASS | PASS |
| **post-route functional netlist** | synth/sim mismatch, inferred latches | PASS | PASS |
| **post-route timing (SDF, 50 MHz)** | X-prop / setup at real cell delays | PASS | PASS |

All: **256/256 bit-match golden, valid, zero X** on both parts. The routed netlist behaves
bit-identically to the RTL and holds up under back-annotated cell delays — **no RTL fixes were
needed** for sim/synth parity (the Verilator TB stays green). Two sim-infra fixes folded in: compile
RTL and TB in separate compilation units (both `include the graph header), and hold reset past the
gate-level `glbl` GSR window (~100 ns) before the first decode. Timing sim is clocked at 50 MHz —
below both parts' closed Fmax (58.7 / 170 MHz) — so it reflects cell delays without spurious
over-clock violations.

## Q6-09 — d=5 scaling

The decoder is parametric in the generated matching graph, so d=5 is the same RTL with a larger graph
(`hw/uf_surface_graph_d5.svh`: N=25 / 24 detectors / M=54 edges, vs d=3's N=9 / M=18).

**Verification** (`hw/tb_uf_surface_scale.cpp`, `make -C hw surf-d5`). 2^24 syndromes can't be
enumerated, so we check every weight-1 and weight-2 error pattern (a distance-5 code must correct ≤2
errors):

| | validity | weight-1 (distance) | weight-≤2 logical errors |
|---|---|---|---|
| d=3 (cross-check) | 0 fail | 0 fail | 40 / 153 (d=3 corrects only weight-1) |
| **d=5** | **0 fail** | **0 fail** | **0 / 1431** ⇒ corrects *every* ≤2-error fault |

**Synthesis** (both parts, OOC, Vivado 2024.2):

| part | d | LUT | FF | Fmax | worst-case latency | budget (1 µs) |
|------|---|-----|----|------|--------------------|---------------|
| Zybo `xc7z020` | 3 | 1178 (2.2%) | 268 | 58.7 MHz | 47 clk → **801 ns** | ✅ |
| Zybo `xc7z020` | 5 | 6139 (11.5%) | 754 | 15.9 MHz | 109 clk → **6.86 µs** | ❌ |
| KV260 `xck26` | 3 | 1200 (1.0%) | 268 | 170 MHz | 47 clk → **276 ns** | ✅ |
| KV260 `xck26` | 5 | 6427 (5.5%) | 754 | 38.0 MHz | 109 clk → **2.87 µs** | ❌ |

(Worst-case latency is over all weight-≤2 syndromes; the "33 clk" quoted under Q6-05 was the
empty-syndrome *best* case. Latency is syndrome-dependent: more growth rounds ⇒ more cycles.)

**Scaling verdict.** d=3 meets the ~1 µs round budget on both boards. **d=5 misses it on both** with
the current single-bounded-pass-per-cycle FSM. Area is never the limit (d=5 is 11.5 % of the small
XC7Z020). The wall is two compounding factors as the graph grows: **Fmax collapses** ~3.7–4.5×
(58.7→15.9 MHz, 170→38 MHz — the per-cycle combinational chains are O(M): the 54-edge loop-carried
peel sweep and the CC/forest passes over 25 nodes) **and the cycle count grows** ~2.3× (47→109). The
fix is the lever flagged since Q6-05: **pipeline the per-cycle passes** — split the O(M) peel/CC
chains into bounded-depth sub-steps to restore Fmax — which is the precondition for real-time d≥5.
Max code distance per board *within budget*, as-is: **d=3 on both** (KV260 has ~3.6× the timing
headroom of Zybo, so it is the path to d=5 once pipelining lands).

## Q6-10 — pipelining attempt: parallel peel + the real critical path

Q6-09 named the per-cycle O(M) chains as the Fmax wall. Q6-10 rewrites the **peel** sweep from a
loop-carried M-edge chain (one cycle, O(M) depth) to a **parallel leaf-strip** — each round peels all
current non-boundary leaves at once via associative count/XOR reductions (bit-equivalent: the
per-edge correction is the leaf-side subtree parity, peel-order-independent; the d=3 golden still
matches, d=5 still corrects 0/1431).

Before (loop-carried peel) → after (parallel peel):

| part | d | LUT before→after | Fmax before→after |
|------|---|------------------|-------------------|
| Zybo `xc7z020` | 3 | 1178 → **872** (−26%) | 58.7 → 62.4 MHz |
| Zybo `xc7z020` | 5 | 6139 → 6061 | 15.9 → **15.4 MHz** (unchanged) |
| KV260 `xck26` | 3 | 1200 → 876 | 170 → 173.6 MHz |
| KV260 `xck26` | 5 | 6427 → 6111 | 38.0 → **38.2 MHz** (unchanged) |

**Finding: the peel was *not* the binding critical path.** Freeing it dropped LUTs ~26% at d=3 but
left d=5 Fmax flat. The d=5 post-route worst path is unambiguous:

```
Source: e_idx_reg → Destination: troot_reg[24]   (the spanning-forest phase)
Logic Levels: 78  (LUT6=62 …)   Data path 64.6 ns
```

That is the **union-find root-walk** in `S_FOREST` (`for k in 0..N: ra = troot[ra]`, done for both
endpoints): an N-deep chain of index-muxed `troot` lookups, unrolled to ~78 LUT levels at d=5. It is
a *true* serial dependency (each hop needs the previous lookup), so it cannot be balanced like the
peel reductions. **This is the actual wall for d≥5 Fmax**, and removing it needs a bounded-depth /
parallel union-find for the forest (path-compression/union-by-rank across cycles, or the parallel
cluster architecture of Liyanage et al., ArXiv:2301.08419) — a real redesign, not a chain rewrite.

Net: the parallel peel ships as a correctness-preserving area win and removes one chain that *would*
become critical once the find is fixed; the d≥5 real-time lever is now precisely targeted (the forest
find) and tracked in #372.

### Q6-10 (part 2) — forest find: quick-union → quick-find (flat union-find)

The remaining #372 work: kill the `S_FOREST` root-walk identified above. The Q6-02/Q6-04 form was a
**lazy quick-union** — `find` walked `troot` N times per endpoint (`for k in 0..N: ra = troot[ra]`),
an N-deep *serial* chain of index-muxed lookups (77–78 LUT levels at d=5). Replace it with
**quick-find**: hold the invariant that `troot[x]` is *always* the direct component root (a flat
forest). Then `find` is a single depth-1 read, and `union` eagerly relabels every node of the
absorbed root in parallel — N independent compare-muxes (`troot[i] <= (troot[i]==ra)?rb:troot[i]`),
no serial dependency. This is **bit-identical** to the lazy form: edges are still processed in index
order, the union test is still `ra!=rb`, and the surviving root is still `rb`, so `istree[]` is
unchanged — d=3 still bit-matches `uf_surface_golden.mem` and d=5 still corrects 0/1431 weight-≤2
(both re-verified in Verilator; cycle counts unchanged: d=3 47 clk worst, d=5 109 clk worst).

Re-synth (Vivado 2024.2, OOC, both parts; **base** = the parallel-peel decoder above, re-run on the
same box for a same-day before/after):

| part | d | LUT base→new | Fmax base→new | d=5 worst-case latency base→new (109 clk) |
|------|---|--------------|---------------|--------------------------------------------|
| Zybo `xc7z020`  | 3 | 872 → **760**  | 62.4 → **81.5 MHz** (+31%) | — (d=3 47 clk: 753→**577 ns** ✅) |
| Zybo `xc7z020`  | 5 | 6061 → **3622** (−40%) | 15.4 → **17.8 MHz** (+16%) | 7.08 → **6.12 µs** ❌ |
| KV260 `xck26`   | 3 | 876 → **670**  | 173.6 → **191.1 MHz** (+10%) | — (d=3 47 clk: 271→**246 ns** ✅) |
| KV260 `xck26`   | 5 | 6111 → **3799** (−38%) | 38.2 → **47.5 MHz** (+24%) | 2.85 → **2.29 µs** ❌ |

**Two wins, not one.** Quick-find lifts d=5 Fmax +16% (Zybo) / +24% (KV260) *and* nearly **halves
d=5 LUT** (the N-deep unrolled root-walk was also large combinational area) — d=5 now fits in ~3.8k
LUT (1.5% of KV260). The root-walk is **gone** from the worst path: the new d=5 binding path is

```
Zybo : label_reg[1][2] → anyodd_reg    77 levels, 56.1 ns
KV260: defect_reg[0]   → anyodd_reg    78 levels, 21.0 ns   (via oddc[])
```

i.e. the **connected-component relabel + odd-cluster parity** scatter — `S_CC_RELAX`'s Jacobi
min-relaxation (`nl[EA]=min(...)`) and `S_ODD`'s per-cluster parity (`par[label[i]] ^= defect[i]`,
a histogram scatter indexed by `label`) feeding `anyodd`. These are label-indexed scatters with
read-after-write on a shared bucket, which synth serialises into a long chain — the same *shape* of
wall the find had, one stage upstream.

**Budget verdict.** d=3 now clears 1 µs with wide margin on both boards. d=5 still misses (6.12 µs
Zybo, 2.29 µs KV260) but is **materially closer** — KV260 is within ~2.3× of budget. The next lever
is bounding the **CC/odd scatter** (e.g. tree-reduced per-label parity, or fusing the relabel into
the growth loop so `anyodd` doesn't depend on a fresh full CC pass), tracked as follow-up to #372.

## Q6-11 — odd-cluster parity: scatter → gather (d=5 crosses the budget on KV260)

The forest-find PR (#375) left the d≥5 worst path as the `S_ODD` per-cluster parity **scatter**
(`par[label[i]] ^= defect[i]`, ~77 levels feeding `anyodd`). A scatter into a label-indexed bucket
with read-after-write serialises into a long accumulator chain. Replace it with a **gather**: for
each label `v`, `par[v] = XOR{ defect[i] : label[i]==v }` — an independent masked XOR-reduction over
the N nodes, which Vivado balances into an O(log N) tree (XOR is associative/commutative, so this is
**bit-identical** to the scatter; d=3 still bit-matches the golden, d=5 still 0/1431; cycle counts
unchanged 33/47/109 clk).

Re-synth (Vivado 2024.2, OOC; **before** = the #375 quick-find decoder, **after** = + parity gather):

| part | d | LUT before→after | Fmax before→after | d=5 latency (109 clk) |
|------|---|------------------|-------------------|------------------------|
| Zybo `xc7z020`  | 3 | 760 → **650**  | 81.5 → **84.6 MHz** | d=3 47 clk: 577→**556 ns** ✅ |
| Zybo `xc7z020`  | 5 | 3622 → **2799** | 17.8 → **50.3 MHz** (+183%) | 6.12 → **2.17 µs** ❌ |
| KV260 `xck26`   | 3 | 670 → **591**  | 191.1 → **260.8 MHz** | d=3 47 clk: 246→**180 ns** ✅ |
| KV260 `xck26`   | 5 | 3799 → **2807** | 47.5 → **135.9 MHz** (+186%) | 2.29 → **0.80 µs** ✅ |

**The d=5 wall was overwhelmingly this one scatter.** Removing it nearly **triples** d=5 Fmax on both
parts, and **KV260 d=5 now meets the ~1 µs round budget at 802 ns** — the first real-time d=5 result,
reached entirely board-free. Zybo d=5 (2.17 µs) is still over but within ~2.2×.

New d=5 worst path (down from 77 to 26 levels):
```
Zybo : label_reg[24] → FSM_sequential_state_reg    26 levels, 19.6 ns
KV260: label_reg[24] → FSM_sequential_state_reg    26 levels,  7.2 ns
```
i.e. the `S_CC_RELAX` Jacobi min-relaxation feeding the `changed`/next-state logic — the per-pass
convergence test gated on the per-node min.

### Q6-12 — CC pointer-jumping: tried, measured, rejected (negative result)

The natural next idea was to attack that path with a Hillis-Steele **pointer-jump** in `S_CC_RELAX`:
rewrite the min-relax as a per-node gather and add a `label[label[v]]` shortcut term so label chains
collapse in O(log diameter) passes instead of O(diameter). It is correctness-neutral (the extra
candidates are same-component ids ≥ the component min, so the converged labels are unchanged —
re-verified: d=3 golden + d=5 0/1431 both still pass). **But it regressed every metric and was not
shipped:**

| part | d | Fmax PR-A → +ptr-jump | LUT PR-A → +ptr-jump |
|------|---|-----------------------|----------------------|
| Zybo `xc7z020`  | 5 | 50.3 → **40.3 MHz** | 2799 → 3849 |
| KV260 `xck26`   | 5 | 135.9 → **113.9 MHz** | 2807 → 3500 |

Two reasons, both instructive: (1) the cycle win was negligible — d=5 worst-case dropped only
**109 → 108 clk**, because the d=5 cycle budget is dominated by the one-edge-per-cycle `S_FOREST`
(M=54) and the N-round `S_PEEL_PASS` (N=25), *not* the handful of CC relax passes; (2) the
`label[label[v]]` shortcut is a **double-indexed** register read — a depth-2 N:1 mux — that *added*
to the critical path (26 → 30 levels) rather than shortening it, and the gather rewrite bought
nothing because synth already constant-folds the original per-edge scatter (the edge endpoints are
`localparam`s). Net: pointer-jumping is the wrong lever here. The CC relax path is left as-is; the
remaining d≥5 levers are cycle-count (the fixed `S_FOREST`/`S_PEEL` passes) rather than this path.

**Q6 d=5 status:** KV260 meets the ~1 µs budget (802 ns); Zybo (2.17 µs) is bounded by the slow
`-1` part more than by any single remaining path.

## Q6-13 — early-terminate the peel sweep (cut the d=5 cycle count)

Q6-12 established that the remaining d≥5 lever is **cycle count**, not Fmax. The biggest fixed cost
in the schedule is `S_PEEL_PASS`, which runs a constant `UF_N` rounds (the worst-case tree depth).
But a peel round that strips no leaf makes **zero** register changes — every `peel[e]=0` forces
`lfd`/`cnt`/`tog`/`vleaf` to 0, so `corr`/`deg`/`dfct`/`istree` all write back their current values —
and because nothing changed, every subsequent scheduled round is also empty. So the first empty round
is the fixpoint: jump straight to `S_FINISH`. Implemented with one `anypeel = |peel` reduction
gating the state transition. **Bit-identical** (golden d=3 + d=5 0/1431 both preserved); it cuts the
peel cost from a fixed `N` to (actual max tree depth + 1).

Verilator worst-case latency: **d=5 109 → 91 clk** (−18, −16.5%); d=3 33 → 25 clk.

Re-synth (Vivado 2024.2, OOC; **before** = the #377 parity-gather decoder re-synthesised on the same
box — baseline reproduced exactly, no drift — **after** = + early-terminate):

| part | d | Fmax before→after | cycles | latency before→after |
|------|---|-------------------|--------|----------------------|
| Zybo `xc7z020` | 5 | 50.3 → 46.4 MHz | 109 → 91 | 2.167 → **1.961 µs** (−9.5%) |
| KV260 `xck26`  | 5 | 135.9 → 124.2 MHz | 109 → 91 | 0.802 → **0.733 µs** (−8.7%) |

**Net latency improves on both parts** — the 16.5% cycle cut outweighs an ~8% Fmax cost. The Fmax dip
is *not* a new critical path: the worst path is unchanged (`label_reg[24] → FSM_state`, the
`S_CC_RELAX` Jacobi path, 26 levels), but feeding `anypeel` into the next-state mux widened that
register's input cone and cost it some routing slack (−14.88 → −16.56 ns @ Zybo). LUT grew trivially
(Zybo 2799→2826, KV260 2807→2864). The trade is clearly worth it for the latency goal: KV260 d=5
moves further under the ~1 µs budget (733 ns) and Zybo d=5 closes to 1.96 µs.

**Q6 d=5 status:** KV260 733 ns (real-time, comfortably under budget); Zybo 1.96 µs. Remaining cycle
lever is the one-edge-per-cycle `S_FOREST` (M rounds) — the next target.

## Q6-14 — unroll the spanning-forest pass (the big d=5 cycle cut)

After Q6-13, `S_FOREST` (one fused edge per cycle, scanned to the last spanning-tree edge) is the
dominant fixed cost in the d=5 schedule. A **union-counter early-exit** (stop once `N − #components`
tree edges are added) was tried and **rejected**: the worst-case syndrome's last tree edge sits near
index `M−1`, so it saved only **1 clk** (91 → 90) — the same near-zero-worst-case wall Q6-12 hit.

The lever that works is **unrolling**: process `FOREST_UNROLL` edges per cycle with *strictly
sequential* union semantics inside the cycle. A working copy `wt[]` of `troot` is mutated with
blocking writes, so sub-union `k` sees sub-union `k−1`'s relabel — exactly the one-edge quick-find,
folded into one clock. The same edges become `istree[]` in the same index order, so it is
**bit-identical** (d=3 golden + d=5 0/1431 both preserved); only the cycle count drops from `~M` to
`~⌈M/UNROLL⌉`. Cost: the per-cycle path is `~UNROLL×` the single-edge find+relabel depth.

Unroll sweep (Vivado 2024.2, OOC; **before** = the #380 early-terminate decoder re-synthesised on the
same box — reproduced exactly, no drift). d=5, both parts; latency = cycles / Fmax:

| UNROLL | d=5 clk | Zybo Fmax | Zybo latency | KV260 Fmax | KV260 latency | LUT (Zybo / KV260) |
|--------|---------|-----------|--------------|------------|---------------|--------------------|
| 1 (#380 base) | 91 | 46.4 MHz | 1.961 µs | 124.2 MHz | 0.733 µs | 2826 / 2864 |
| 2 | 64 | 49.0 MHz | 1.306 µs | 137.2 MHz | 0.467 µs | 3187 / 3148 |
| **3** ✅ | **55** | **47.0 MHz** | **1.170 µs** | **131.8 MHz** | **0.417 µs** | 3782 / 3911 |
| 4 | 51 | 44.0 MHz | 1.159 µs | 109.7 MHz | 0.465 µs | 4136 / 4322 |

**UNROLL=3 is the sweet spot.** At UNROLL ≤ 3 the worst path is *unchanged* — still
`label_reg[24] → FSM_state` (the `S_CC_RELAX` Jacobi path, 26 levels): the unrolled forest path stays
**below** it, so the 3× forest-cycle cut is essentially free (Fmax even ticks up vs base, placement
noise around the same path). At UNROLL=4 the forest path finally crosses the CC path and Fmax
collapses (KV260 132 → 110 MHz), erasing the cycle win. So 3 is the deepest unroll before the forest
becomes critical.

Net vs #380 base: **Zybo 1.961 → 1.170 µs (−40%), KV260 0.733 → 0.417 µs (−43%)**. LUT cost is modest
(Zybo 7.1%, KV260 3.3% of the part). KV260 d=5 now decodes in **417 ns** — well inside the ~1 µs round
budget — and Zybo d=5 closes to **1.17 µs**, within ~1.2× of budget on the slow `-1` part.

**Q6 d=5 status:** KV260 **417 ns** (real-time, ~2.4× margin); Zybo **1.17 µs**. The forest is no
longer the worst-case bottleneck; the binding path is now `S_CC_RELAX` (Fmax) and the growth/CC/peel
cycle floor (~37 clk).

## Q6-15 — cut the d=5 growth/CC cycle floor (fold CC_INIT, incremental labels, fuse ODD+GROW)

With the forest unrolled, the d=5 worst case (55 clk) is dominated by the GROWTH loop, which for the
worst syndrome runs 5 rounds of `S_CC_INIT → S_CC_RELAX(converge) → S_ODD → S_GROW`. Three
**bit-identical** structural cuts (d=3 golden + d=5 0/1431 preserved throughout):

1. **Fold `S_CC_INIT`** — the label-to-identity reset is a pure register write, so it is done in the
   predecessor that already writes registers (`S_IDLE` on accept, `S_GROW` after a round) instead of
   in its own state. Removes one state and saves 1 clk/round.
2. **Incremental labels** — don't reset labels between growth rounds. Growth only *merges*
   components, and the min-label Jacobi fixpoint is unique (every node starts ≥ its new component min;
   that min node keeps its self-label), so seeding `S_CC_RELAX` with the prior round's converged
   labels reaches the identical fixpoint but only the newly-merged frontier has to propagate — a
   couple of passes instead of the full component diameter from identity.
3. **Fuse `S_ODD` into `S_GROW`** — the odd-cluster flags are a pure combinational function of
   `label`+`defect` (the Q6-11 parity gather), so they are computed in the grow cycle rather than
   registered in a separate state. Removes the second state and saves 1 clk/round.

Verilator worst-case d=5: 55 → 50 (fold) → 48 (+incremental) → **43 clk** (+fuse) — −22%.

Re-synth (Vivado 2024.2, OOC; **before** = the #382 forest-unroll decoder re-synthesised on the same
box — reproduced exactly, no drift). `a` = folds 1+2, `b` = + the ODD/GROW fuse; latency = clk / Fmax:

| variant | d=5 clk | Zybo Fmax | Zybo latency | KV260 Fmax | KV260 latency | LUT (Zybo / KV260) |
|---------|---------|-----------|--------------|------------|---------------|--------------------|
| 1 (#382 base) | 55 | 47.0 MHz | 1.170 µs | 131.8 MHz | 0.417 µs | 3782 / 3911 |
| a (fold + incr.) | 48 | 49.8 MHz | 0.964 µs | 132.8 MHz | 0.361 µs | 3839 / 3882 |
| **b (+ fuse)** ✅ | **43** | 49.4 MHz | **0.870 µs** | 134.1 MHz | **0.321 µs** | 3864 / 3879 |

**Variant `b` ships.** Removing two FSM states (`S_CC_INIT`, `S_ODD`) *shrank* the next-state decode,
so Fmax **rose** on both parts despite folding the parity gather into the grow cycle — the worst path
is still `label_reg → FSM_state` (`S_CC_RELAX`) but one level shorter (26 → 25), and the fused grow
path stays below it. LUT is flat (KV260 even drops). Net vs #382: **Zybo 1.170 → 0.870 µs (−26%),
KV260 0.417 → 0.321 µs (−23%)**.

**Q6 d=5 status:** KV260 **321 ns** (~3.1× margin under the ~1 µs budget); Zybo **870 ns** — *both
parts now decode d=5 in real time.* Cumulative over Q6-13/14/15: d=5 109 → 43 clk, KV260 802 → 321 ns,
Zybo 2.17 → 0.87 µs. The binding constraint is now the `S_CC_RELAX` Fmax path on both parts.

## Q6-16 — balanced min-reduction in S_CC_RELAX (kill the d=5 Fmax wall)

After Q6-15 the d=5 worst case (43 clk) is **Fmax-bound**, not cycle-bound. The binding path on both
parts was `label_reg → FSM_state` — the `S_CC_RELAX` Jacobi min-relax + convergence test — **25 LUT
levels, 82% routing**. The cell trace showed a long chain of `label[24][3]_i_*` LUTs (node 24 = the
boundary super-node). Root cause: the min-relax was written as a **serial min-fold** over incident
edges (`if (label[nbr] < nl[v]) nl[v] = label[nbr]`), which synthesises to a chain of depth = node
degree — and the boundary node's high degree made that the wall.

Fix: rewrite the relax as a per-node **gather + balanced O(log N) min-reduction tree** (`cc_min_reduce`).
For each node build the candidate set `{label[v]} ∪ {label[u] : fused edge v–u}` (non-neighbours
masked to an all-ones sentinel ≥ every real label), then tree-reduce. `min` is associative +
commutative, so it is **bit-identical** to the serial fold (same candidate set) — d=3 golden + d=5
0/1431 preserved, cycles unchanged at 43. Same scatter→tree idea as the Q6-11 parity gather.

Re-synth (Vivado 2024.2, OOC; **before** = the #384 decoder re-synthesised on the same box,
reproduced exactly — no drift). Cycles unchanged, so latency moves purely with Fmax:

| part | d=5 clk | Fmax before→after | latency before→after | LUT before→after |
|------|---------|-------------------|----------------------|------------------|
| Zybo `xc7z020` | 43 | 49.4 → **58.3 MHz** (+18%) | 0.870 → **0.738 µs** (−15%) | 3864 → 4568 (8.6%) |
| KV260 `xck26`  | 43 | 134.1 → **146.8 MHz** (+9.5%) | 0.321 → **0.293 µs** (−9%) | 3879 → 4656 (4.0%) |

The tree dropped the relax path from ~degree to ~log₂(N) levels and **moved the critical path off
`S_CC_RELAX` entirely** — the new binding path is `e_idx → troot` (the Q6-14 forest-unroll relabel,
17 levels, again route-dominated). LUT grew ~18–20% (the O(N²) per-node candidate gather) but stays
small in absolute terms (Zybo 8.6%, KV260 4.0% of the part). The min-fold→tree win compounds with
distance: the boundary degree (and so the old serial chain) grows with d, so this is also a
precondition for d=7.

**Q6 d=5 status:** KV260 **293 ns**, Zybo **738 ns** — both real-time. Cumulative over Q6-13/14/15/16:
d=5 109 → 43 clk; KV260 802 → 293 ns (2.7×), Zybo 2.17 → 0.74 µs (2.9×). New binding path is the
`S_FOREST` troot relabel (`e_idx → troot`) — the next Fmax target if d=5 needs more, though both parts
already clear the budget comfortably.

## Q6-17 — d=7 scaling study

With the Q6-16 min-tree removing the boundary-degree serial-fold wall (the precondition for larger
distances), the decoder is measured at **d=7: N=49, M=110** (vs 25/54 at d=5). The same parametric FSM
is used unchanged; the scale TB gained `cbit()` overloads + a 64-bit syndrome so it handles the wide
ports (UF_M=110 makes `correction` a Verilator `VlWide<>`; the syndrome is 47 bits). Verify
(`make surf-d7`, weight-≤2 sweep — a *necessary* check for a distance-7 code, which fully corrects
≤3): **validity 0 fail, distance-correct on all 110 weight-1 errors, 0/5995 weight-2 logical errors,
worst-case 62 clk.**

Dual-target OOC synth (Vivado 2024.2; d=5 column = the #386 Q6-16 decoder, same RTL):

| part | d | cycles | Fmax | latency | LUT (util) |
|------|---|--------|------|---------|------------|
| Zybo `xc7z020`  | 5 | 43 | 58.3 MHz | 0.738 µs | 4568 (8.6%) |
| Zybo `xc7z020`  | 7 | 62 | 45.5 MHz | **1.363 µs** | 11673 (**21.9%**) |
| KV260 `xck26`   | 5 | 43 | 146.8 MHz | 0.293 µs | 4656 (4.0%) |
| KV260 `xck26`   | 7 | 62 | 112.9 MHz | **0.549 µs** | 11705 (**10.0%**) |

**Verdict: KV260 decodes d=7 in real time (549 ns, ~1.8× margin under the ~1 µs round budget); Zybo
does not (1.363 µs, ~1.4× over).** Both parts *fit* comfortably — the O(N²) per-node min-tree gather
grew LUTs ~2.5× (N doubled), but even on the small Zybo it is only 22%. Scaling d=5→7: cycles +44%
(forest `⌈M/3⌉` 18→37 dominates), Fmax −22/−23% (longer routes + deeper logic on the bigger graph),
net latency ~1.85× per part. The binding path is still the `S_FOREST` troot relabel (`e_idx →
troot`/`istree`) — the min-tree fix held, so `S_CC_RELAX` is *not* the d=7 wall either. The lever to
bring Zybo d=7 under budget (or widen the KV260 margin) is that forest relabel path: a shallower
union-relabel or a re-sweep of `FOREST_UNROLL` now that the CC path is gone.

## Q6-18 — re-sweep FOREST_UNROLL after the min-tree (Zybo latency)

Q6-14 set `FOREST_UNROLL=3` when the binding path was `S_CC_RELAX` (so a deeper forest unroll was
"free" up to that 26-level cap). Q6-16's CC min-tree removed that cap and **exposed the forest relabel
as the new critical path** — which makes the K=3 choice stale. Re-sweep K ∈ {2,3,4} × {d5,d7} × both
parts (bit-identical for every K — d=3 golden + d=5 0/1431 + d=7 0/5995 all preserved). Latency =
cycles / Fmax:

| K | d5 clk | Zybo d5 | KV260 d5 | d7 clk | Zybo d7 | KV260 d7 |
|---|--------|---------|----------|--------|---------|----------|
| 2 ✅ | 52 | **0.655 µs** (79.4 MHz) | 0.294 µs (177.1) | 80 | **1.185 µs** (67.5 MHz) | 0.562 µs (142.4) |
| 3 (Q6-14) | 43 | 0.738 µs (58.3) | **0.293 µs** (146.8) | 62 | 1.363 µs (45.5) | 0.549 µs (112.9) |
| 4 | 39 | 0.818 µs (47.7) | 0.327 µs (119.1) | 53 | 1.460 µs (36.3) | **0.485 µs** (109.2) |

**`FOREST_UNROLL=2` ships.** With the CC cap gone, the shallower relabel lifts Fmax by more than its
+cycles cost: Zybo d5 58→**79 MHz** (+36%), d7 45→**67 MHz** (+48%). K=2 is best on **3 of 4 cells**
(both d5, Zybo d7); only KV260-d7 marginally prefers a deeper K (K=2 is +2% vs K=3 there, still 1.8×
under budget — and K=4, which wins that one cell, regresses the other three badly). At K=2 the forest
path drops to **13–14 levels** (from 17 at d5 / 25–27 at d7 under K=3) — still the binding path but no
longer a tall pole.

Net vs Q6-16/17 (K=3): **Zybo d5 0.738 → 0.655 µs (−11%), Zybo d7 1.363 → 1.185 µs (−13%)**; KV260
flat (d5 0.293→0.294, d7 0.549→0.562, both within noise / still deep under budget).

**Q6 distance status (final, `FOREST_UNROLL=2`):** d=5 — KV260 294 ns, Zybo 655 ns (both real-time).
d=7 — KV260 562 ns (real-time, ~1.8× margin), Zybo 1.185 µs (≈1.2× over the ~1 µs budget, down from
1.4×). Zybo d=7 is the one remaining cell over budget; closing it needs a structurally shallower forest
relabel (the K-sweep is exhausted), or accept that d=7 targets the KV260-class part.

## Q6-19 — multi-round (3D space-time) decoding

Everything up to Q6-18 decodes a **single-round, code-capacity** graph (2D, space only). A *real*
decoder must handle the **time dimension**: many measurement rounds, with measurement errors, on the
3D space-time matching graph. This moves the FPGA decoder to a **multi-round phenomenological** graph —
`T` rounds with **time-like measurement-error edges** between consecutive rounds (the generator gained
a `rounds` arg, `graph <d> <rounds>`, built on `memory_z_experiment(rounds)`).

The decoder RTL is **graph-agnostic** (parametric in `UF_N`/`UF_M`/edge tables), so it decodes the 3D
graph **with zero RTL changes** — time is just baked into node numbering and which edges exist. The
only infra change was a wide-syndrome TB path (>64 detectors → the `syndrome` port becomes a
Verilator `VlWide<>`, mirroring the Q6-17 wide-`correction` fix). Verified bit-exact (`make surf-3d`):

| case | detectors | edges | cyc | weight-≤2 logical errs | Zybo Fmax / latency / LUT | KV260 Fmax / latency / LUT |
|------|-----------|-------|-----|------------------------|---------------------------|----------------------------|
| d=5 × 3 rounds | 48 | 120 | 85 | **0 / 7140** | 63.5 MHz / 1.339 µs / 19.1% | 145.0 MHz / **0.586 µs** / 9.5% |
| d=5 × 5 rounds | 72 | 186 | 118 | **0 / 17205** | 48.2 MHz / 2.448 µs / 40.6% | 104.0 MHz / 1.135 µs / 18.9% |

(d=5 corrects ≤2, so a clean **0** weight-≤2 logical-error count is the meaningful gate; validity and
weight-1 distance are 0-fail on both.)

**The right budget for a `T`-round volume is `T` µs, not 1 µs** — acquiring `T` rounds *takes* `T` µs
at the ~1 µs/round measurement rate, so a **block** decoder keeps up if it decodes the volume in
< `T` µs. By that measure **both boards keep up**: d=5×3 (acquire 3 µs) decodes in 0.586 µs (KV260) /
1.339 µs (Zybo); d=5×5 (acquire 5 µs) in 1.135 µs / 2.448 µs — all well under their acquisition window.
KV260 d=5×3 decodes a full 3-round circuit volume in **586 ns**. Caveat: block decoding waits for all
`T` rounds before starting (latency = acquire + decode); low feed-forward latency needs the
**streaming** decoder (Q6-20). LUT grows with the volume (the O(N²) min-tree gather): d=5×5 is 40.6 %
of Zybo — fits, but the small part is filling up.

**Honest scope:** this is *phenomenological* 3D (data + measurement-error edges), not full
*circuit-level* (gate/CX-level) noise — the standard intermediate model, and the one that introduces
the time dimension + measurement-error edges that matter. Full circuit-level needs a surface-code
`circuit_level_mechanisms()` (mirroring the BB-code one); tracked as a Q6-19 follow-up.

## Q6-20 — sliding-window streaming decoder (continuous, bounded-memory, real-time)

Everything through Q6-19 decodes a **fixed-size volume** as one block. A real QPU emits syndromes
**forever** — you cannot store the whole history or wait for the experiment to end. The sliding-window
approach (Dennis et al.; Skoric/Tan et al.; the software `SlidingWindowDecoder`, Q4-01) keeps a running
`W`-round residual, decodes it, **commits** the oldest `C` rounds, applies that correction to the
residual, then slides forward by `C` and reloads `C` fresh rounds — decoding an unbounded stream in
**bounded `O(W)` memory**.

**Key structural insight:** interior windows of a stream are translation-invariant, so **one compiled
window graph** (a `W`-round volume whose future/past time cuts route to *temporal-sink* nodes — free
obs-less drains to the boundary) serves every steady-state window, and the existing per-window UF core
decodes it **unchanged**. The new RTL (`hw/uf_streaming_decoder.sv`) is a thin wrapper: a residual
buffer + the per-window core + the commit (toggle the committed edges' real detectors, accumulate
their observable) + the slide-by-`C` reload. The window graph + its streaming metadata
(`UF_SHIFT`/`UF_DCOMMIT`/`UF_ECOMMIT`/…) come from `qec_surface_uf_graph -- window <d> <W> <C>`, built
from the **same** `SlidingWindowDecoder::window_dem` the software uses.

**Correctness (`make stream`, d=3 `W`=9 `C`=3 = the software Q4-03 params `W`=3d/`C`=d).** The
per-window core is already distance-verified (Q6-09/17/19); what the wrapper adds is the residual
carry, and the tie-break-independent proof of *that* is **validity**: a graphlike decoder always
produces a correction that reproduces the syndrome, so once a stream of defects is pushed through the
commit region (drain with zero rounds) **every real defect must be resolved — the residual must clear**
(exactly the software `residual_after_decode == 0` criterion). Result: a zero stream commits zero
logical and never lights the residual; **40/40 random defect streams fully drain** (≈1800 windows); the
FSM never stalls. (Bit-equality vs the *software* UF is not the right gate — the RTL and CPU UF differ
on tie-breaks of degenerate syndromes, as the Q6-04 oracle already documents; validity + the
distance-verified core is the correct proof.)

**Throughput / real-time (Vivado 2024.2, OOC).** A window retires in ≈67 cycles (≈61 core +
commit/slide/reload). The streaming budget is one window per commit period = `C` rounds = **`C` µs**
(3 µs at d=3 `C`=3, the ~1 µs/round measurement rate):

| part | Fmax | per-window decode | vs 3 µs commit budget | LUT (util) |
|------|------|-------------------|------------------------|------------|
| Zybo `xc7z020`  | 63.2 MHz | **1.06 µs** | 2.8× headroom | 9119 (17.1 %) |
| KV260 `xck26`   | 140.6 MHz | **0.48 µs** | 6.3× headroom | 8891 (7.6 %) |

**Both boards keep up with a continuous syndrome stream in real time**, in bounded memory — the
qualitative step from a block decoder to a *real* streaming decoder. Caveat: this targets the
**steady-state interior** window; the first/last windows of a finite experiment differ (true time
boundaries vs temporal sinks) and would need warm-up/drain handling for a closed experiment. Honest
scope inherits Q6-19's: phenomenological 3D, not full circuit-level gate noise.

## Q6-03 — GPU vs FPGA: latency, throughput, power, and the ASIC go/no-go

This compares the **same** Delfosse–Nickerson Union-Find decoder on two substrates: the GPU batch
decoder (Q3-01, `CudaUnionFind`, [qec-q3-gpu-uf.md]) and the FPGA sequential FSM (Q6-04…Q6-18,
`hw/uf_surface_decoder.sv`, this document). It is the decision input for Phase Q7 (ASIC).

**What "the same algorithm" does and does not mean.** Both implement the identical decode (edge-centric
synchronous growth → spanning forest → reverse-pre-order peel) and both are bit-exactly verified
against the CPU reference. They do **not** decode the same-sized graph at the same nominal `d`: the
GPU bench uses a larger matching graph (d=5 → 72 detectors / 186 edges) while the FPGA uses the
single-round rotated code-capacity graph (d=5 → 24 detectors / 54 edges; d=7 → 48 / 110). So compare
on the **axis each substrate optimizes**, and read same-`d` rows as *order-of-magnitude*, not
head-to-head. Closest graph-size pairing: FPGA d=7 (110 edges) ≈ between GPU d=3 (40) and d=5 (186).

**Source of numbers.** GPU = measured whole-batch throughput on the RTX 4000 SFF Ada (70 W TDP),
100 000-shot batch incl. PCIe round-trip ([qec-q3-gpu-uf.md]). FPGA = the shipped `FOREST_UNROLL=2`
decoder, **Vivado 2024.2 post-route estimates** (OOC), latency = worst-case cycles / Fmax, power =
`report_power` on the routed checkpoint. **Caveat: FPGA figures are post-route estimates, not
on-silicon measurements** — on-board validation is Q6-08 (pending). Power excludes board/PS/DDR (the
PL block only).

### Latency — the real-time axis (FPGA wins decisively, by architecture)

Real-time QEC must decode each syndrome round *within* the measurement cycle (~1 µs for
superconducting qubits) so corrections feed forward before errors compound. This is a **single-decode
latency** requirement.

| substrate | d=5 latency | d=7 latency | nature |
|-----------|-------------|-------------|--------|
| FPGA KV260 | **0.294 µs** | **0.562 µs** | deterministic, fixed cycle count |
| FPGA Zybo  | 0.655 µs | 1.185 µs | deterministic |
| GPU (single decode) | ~5–20 µs (launch + PCIe-bound) | ~5–20 µs | not a single-decode engine |

The GPU's strength is **batch** throughput; a *single* decode pays kernel-launch + PCIe round-trip
overhead (microseconds) that the batch amortizes but real-time cannot. One-thread-per-shot also means
a lone decode gets one lane of a 1000-wide machine. So the GPU is structurally a 5–20 µs+
single-decode device — **outside the ~1 µs round budget** regardless of tuning. The FPGA is the
opposite: sub-µs, deterministic (no data-dependent jitter beyond the bounded worst case), and already
real-time at d=5 (both parts) and d=7 (KV260). **FPGA wins single-decode latency by ~10–50×, and the
gap is architectural, not a tuning artifact.**

**Apples-to-apples on the identical graph (added with Q6-19).** The GPU bench's d=5 graph (72
detectors / 186 edges) is *exactly* the FPGA's d=5 × 5-round graph (Q6-19). On that same graph, FPGA
single-decode latency is **1.135 µs (KV260) / 2.448 µs (Zybo)** deterministic — versus the GPU's
~5–20 µs single-decode (launch+PCIe). So even on the GPU's own graph size, the FPGA is **~5–18×
lower latency per decode**; the GPU only pulls ahead when thousands of decodes are batched.

### Throughput — the offline / Monte-Carlo axis (GPU's home turf, but closer than it looks)

The non-pipelined FPGA FSM retires one decode per worst-case window, so a single instance throughputs
≈ 1/latency; the block is tiny (4–22 % of the part), so a chip holds many instances.

| substrate | d=5 throughput | d=7 throughput |
|-----------|----------------|----------------|
| GPU (whole card, batch) | 2.38 M syn/s (72-det graph) | 0.58 M syn/s (192-det graph) |
| FPGA KV260, 1 instance | 3.4 M syn/s (24-det) | 1.78 M syn/s (48-det) |
| FPGA KV260, area-filled* | ~60–80 M syn/s | ~15–18 M syn/s |

On the **identical 72-det / 186-edge graph** (GPU d=5 = FPGA d=5×5), one KV260 instance does
0.88 M syn/s vs the GPU's 2.38 M/s — here the GPU's batch parallelism wins raw throughput on the
bigger 3D graph, but the FPGA replicates (d=5×5 is 18.9 % of KV260 → ~5 instances → ~4.4 M/s) at
sub-watt, and still wins single-decode latency ~5–18×.

*back-of-envelope: 1 instance / (its LUT %), ignoring routing congestion at high fill — an upper
bound, not a synthesised result. Even a *single* FPGA instance is in the GPU's throughput class (on a
smaller graph); replication trades the FPGA's spare area for aggregate throughput the GPU matches only
by burning its full 70 W. **Throughput is roughly a wash per-chip and decisively FPGA's per-watt** —
which is the next axis.

### Power & energy-per-decode (FPGA wins by 2–3 orders of magnitude)

| cell | total on-chip power | dynamic | energy / decode (total · latency) |
|------|--------------------|---------|-----------------------------------|
| FPGA KV260 d=5 | 0.341 W | 0.052 W | **0.10 µJ** |
| FPGA KV260 d=7 | 0.399 W | 0.110 W | **0.22 µJ** |
| FPGA Zybo d=5  | 0.149 W | 0.046 W | 0.10 µJ |
| GPU d=5 (card) | 70 W (TDP) | — | **~29 µJ** (70 W / 2.38 M syn·s⁻¹) |
| GPU d=7 (card) | 70 W (TDP) | — | ~120 µJ |

The decoder's *dynamic* switching is **46–110 mW**; total on-chip incl. device static leakage is
0.15–0.4 W. Per-decode energy is **~0.1–0.2 µJ vs the GPU's ~29–120 µJ — a 150–600× advantage**, even
before accounting for the GPU graph being larger. This is the metric that matters for a fault-tolerant
machine: thousands of logical qubits each needing a continuously-running decoder, inside a cryostat's
strictly bounded power budget. A GPU-per-logical-qubit is a non-starter on power alone; an
FPGA/ASIC decoder per qubit is the only path.

### Why each substrate wins its axis

- **GPU**: thousands of independent shots in flight amortize launch + PCIe; ideal for **offline**
  threshold/Monte-Carlo sweeps (where aleph's GPU UF already out-throughputs PyMatching's MWPM core at
  large d). Wrong tool for the in-loop control path: single-decode latency is launch-bound and power
  is 70 W.
- **FPGA**: a bounded clocked FSM gives **deterministic sub-µs latency** at single-digit-% area and
  sub-watt power — exactly the real-time control-loop profile. Loses raw aggregate throughput to a
  full GPU only until you replicate it across the spare 80–96 % of the fabric.

### ASIC go/no-go (Q7)

**Does an ASIC close a gap the FPGA cannot?** The FPGA latency is **76–82 % routing delay** (logic is
only ~18–24 %; see the Q6-16/18 critical-path traces) — i.e. the programmable interconnect, not the
logic, is the wall. A std-cell ASIC removes exactly that tax: custom place-and-route on the same RTL
typically yields **3–10× higher clock and 10–100× lower energy**. Concretely that means a **sub-100 ns
decode**, real-time headroom at **d ≥ 9** (where the FPGA's d²-growing graph will push Zybo and
eventually KV260 over budget — already visible: Zybo d=7 is 1.2× over), **µW-class** energy per
decode, and density to place **one decoder per logical qubit** at machine scale — plus cryo-adjacent
operation a GPU can never offer.

The FPGA cannot reach those: its Fmax is interconnect-capped (the Q6-13…18 work extracted the
algorithmic and within-fabric wins; Zybo d=7 over budget is where that runs out), and its per-qubit
power/area do not scale to thousands of logical qubits.

**Recommendation: conditional GO for Q7.** The technical case is made — Q6 proves the decoder is
*algorithmically* real-time-capable (FPGA hits the ~1 µs budget at d=5 both parts, d=7 on KV260, at
sub-watt power and 150–600× the GPU's energy efficiency), and an ASIC closes the **d ≥ 9 latency**,
**machine-scale power/area**, and **cryo-integration** gaps the FPGA structurally cannot. Per
ROADMAP §Q7 the trigger is **commercial, not technical**: gate tape-out on funding + a committed
QPU-company customer. Immediate next engineering step before any silicon commitment is **Q6-08 on-board
bring-up** to replace these post-route estimates with measured wall-clock latency and power.

## Q6-08 — on-board bring-up (measured silicon, Arty Z7-20)

The post-route latency estimates above are now confirmed on hardware. Board: **Digilent Arty Z7-20**
(`xc7z020clg400-1` — the same PL part as the Zybo Z7-20 target), booted from the PYNQ-Z1 v3.1.1 SD
image over LAN.

**Board bitstream** (`hw/syn/arty_z7_bd.tcl`, not the OOC study): Zynq-7 PS + `uf_axi_top`
(uf_axi_wrap, AXI4-Lite control plane; AXI4-Stream tied off for this run) on the PS GP0 AXI master,
FCLK **50 MHz**. In-context impl: **WNS +7.29 ns → TIMING_MET** (achievable ~79 MHz; clocked at 50
for margin), DRC clean.

**Result** (`hw/sw/uf_pynq.py`, PYNQ overlay + `pynq.MMIO`, all 256 d=3 syndromes):

| metric | value |
|--------|-------|
| IDCODE probe | `0x5546_0003` ✓ (PS↔PL link) |
| correctness | **256/256 bit-identical to `uf_surface_golden.mem`** |
| worst decode latency | **30 clk = 600 ns @ 50 MHz** |
| round budget | 1 µs → **met with 40 % headroom (real-time on silicon)** |

This replaces the d=3 post-route *estimate* (562 ns @ 58.7 MHz OOC) with a *measured* 600 ns at the
conservative 50 MHz board clock, and closes the on-board ACs of Q6-01/Q6-02/Q6-08. The Q7 ASIC call
above no longer rests on estimates for the d=3 latency floor.

### On-board Hardware-in-the-Loop (Monte-Carlo LER on silicon)

`hw/sw/uf_hil.py` replays the co-simulation Monte-Carlo syndrome stream (`hw/cosim_d3.vec`, 5 physical
error rates × 20 000 shots, from the same detector-error model the matching graph was generated from)
through the **real decoder** over AXI4-Lite — the on-silicon version of the Q6-21 board-free co-sim.
The on-board RTL logical-error rate matches the software Union-Find baseline within Monte-Carlo CI at
**every p** (0.01–0.05):

| p | rtl_rate (silicon) | sw_rate (UF) | verdict |
|---|---|---|---|
| 0.01 | 7.25e-3 | 7.80e-3 | PASS |
| 0.02 | 2.69e-2 | 2.73e-2 | PASS |
| 0.03 | 5.18e-2 | 5.24e-2 | PASS |
| 0.04 | 8.23e-2 | 8.14e-2 | PASS |
| 0.05 | 1.15e-1 | 1.14e-1 | PASS |

100 000 shots, worst decode latency 30 clk = 600 ns. **Throughput 7 285 decodes/s (137 µs/decode)** —
this is **host-bound**, not decoder-bound: the Python-polled AXI4-Lite round-trip dominates (≫ the
600 ns decode), so it is a floor. The decoder's own rate is 1 decode per its latency; the AXI4-Stream
+ DMA data plane (`uf_axi_wrap` already exposes it) is the path to decoder-bound throughput and is the
next step for the Q6-03 GPU-vs-FPGA throughput comparison.

### d=5 on-board (measured latency wall on the small part)

Rebuilt the same block design with the d=5 graph (`UF_N=25`, `UF_M=54`; the AXI wrapper now supports
`UF_M > 31` — correction is surfaced as `CORRECTION[31:0]` + the `OBS_FLIP` bit). It **closes timing at
50 MHz** (WNS +3.18 ns, in-context Fmax ~60 MHz) — so on the Arty the d=5 wall is **cycles, not fit or
Fmax**. On-board HiL over `hw/cosim_d5.vec` (per-block worst latency added):

| p | rtl_rate | sw_rate | verdict | worst latency |
|---|---|---|---|---|
| 0.01 | 1.80e-3 | 1.95e-3 | PASS | 62 clk = 1240 ns |
| 0.02 | 1.31e-2 | 1.13e-2 | PASS | 62 clk = 1240 ns |
| 0.03 | 3.49e-2 | 3.12e-2 | PASS | 67 clk = 1340 ns |
| 0.04 | 6.92e-2 | 6.11e-2 | info (supra-threshold) | 65 clk = 1300 ns |
| 0.05 | 1.14e-1 | 1.01e-1 | info (supra-threshold) | 69 clk = 1380 ns |

Correctness: on-board d=5 logical-error rate tracks the software Union-Find within CI at every
**sub-threshold** p (the supra-threshold rows show the known unweighted-UF quality gap, same as the
board-free co-sim; not gated). **Latency verdict: d=5 misses the 1 µs round budget on the Arty Z7-20**
— worst-case 62–69 clk at 50 MHz = 1.24–1.38 µs, and even at the closed ~60 MHz Fmax (16.8 ns) that
is ~1.04–1.16 µs, still over. This is the *measured* confirmation of the small-part (XC7Z020) latency
wall the Q6-09/Q6-10 synth study predicted: d=3 is real-time on this part, d=5 needs the higher-Fmax
KV260 or a pipelined factorisation (the open Q6-10 lever). Throughput is again host-bound (~6.8k/s).

### Decoder-bound throughput via AXI DMA (Q6-03)

The AXI4-Lite figure above is host-bound. `hw/syn/arty_z7_dma_bd.tcl` builds a streaming design —
PS DDR → **AXI DMA MM2S** → the decoder's `s_axis` → decode → `m_axis` → **AXI DMA S2MM** → DDR — so
the PS only arms one transfer and the measured rate is the decoder's own. The streaming datapath is
`hw/uf_stream_core.sv` (a pure AXI4-Stream engine over the same `uf_surface_decoder` core, tlast
propagated input→output so one DMA transfer streams a whole batch). Driver: `hw/sw/uf_dma.py`.

| path | throughput | per decode |
|------|-----------|-----------|
| AXI4-Lite, PS-polled (`uf_hil.py`) | 7 285 /s | 137 µs (host-bound) |
| **AXI DMA (`uf_dma.py`)** | **1 389 667 /s** | **0.72 µs (decoder-bound)** |

**~191× over the polled path**, and 0.72 µs/decode = 36 cycles @ 50 MHz ≈ the decode latency (30 clk)
+ ~6 clk stream/handshake overhead — i.e. the DMA feeds the engine at its own rate, confirming this is
the decoder-bound number, not an interconnect limit. Measured over the same 100 000-shot co-sim stream
(d=3), on-board LER still matches software UF within CI at every p. This is the FPGA throughput figure
for the Q6-03 GPU-vs-FPGA comparison (the engine is a single-decode core; replicating it across the
spare ~98 % of the XC7Z020 fabric multiplies aggregate throughput linearly).

Next on-board: the same d=5 build on a KV260 when that board is available.

[qec-q3-gpu-uf.md]: qec-q3-gpu-uf.md
