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

## Q6-03 — GPU vs FPGA

> Pending board bring-up (Q6-08) for measured on-board latency/throughput/power vs the Q3 GPU decoder.
