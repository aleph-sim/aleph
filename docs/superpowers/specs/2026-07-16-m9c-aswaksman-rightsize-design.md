# M9c Step 5 (Q7-04) — arbitrary-size (AS-Waksman) right-sizing of the m_cm / e_cm fabrics

**Status:** design (2026-07-16). Follows M9c Step 4 (addr→ROM, PR #465 `d3bf1de`). Part of the
post-Beneš area campaign (`docs/perf/qec-q7-fixed-bp.md` § M9c).

## Honest framing (read first)

An OOC probe (2026-07-16, standalone `bp_benes_mcm_wr` on xck26) showed the **padding waste is already
largely reclaimed by the synthesizer**: the N=1024 fabric is **213,248 LUT standalone** but only
**75,985 LUT in-context** (2.8×), because Vivado constant-propagates the tied padding inputs
(288/1024 active) and prunes unread outputs (800/1024). So explicit right-sizing's **LUT** upside over
the current in-context number is **small — single-digit %**, and the campaign math already shows a
single-KV260 **fit is unreachable** (irreducible runtime fabrics ≈108.8 % + non-fabric floor ≈68 %).

**This lever is therefore justified primarily by BRAM, not LUT.** The Beneš control ROMs are the
BRAM hog (tile count scales with ROM *width* = switch count, not total bits). BRAM is at **114.2 %**
after Step 4 — close. The two widest control ROMs are `BP_ROM_BENES_MCMWR` (9728 bits, N=1024) and
`BP_ROM_BENES_ECMRD` (8704 bits, N=512×2 ports). Right-sizing their networks to the real N (800 / 400)
via the switch-optimal **AS-Waksman** construction narrows those ROMs → fewer BRAM tiles, plausibly
bringing BRAM **toward or under 100 %** (half of the two-constraint fit), with a small LUT bonus. The
goal is **maximum honest compression on BRAM**, not a full KV260 fit.

## Goal

Replace the power-of-two-padded Beneš fabrics for m_cm write (1024) and e_cm read (512×2) with
**AS-Waksman networks sized to the real permutation dimensions** — N=800 (m_cm: nhb=800) and N=400
(e_cm: neb=400) — routing the same per-var-group partial injections bit-exactly. Keep the addr path
(now a ROM) and everything else unchanged.

## Background — current fabrics

Per-group, time-multiplexed through one physical network each (control from per-group ROM):
- **m_cm write** `u_benes_wr`: N=`BP_BENES_MCM_M`=1024 (padded from nhb=800), W=1+BWC+MSG_BITS=13,
  PIPE=`BENES_PIPE_MCM`=4. Real map: nvb=288 taps → 800 half-banks. LUT (in-context) 75,985; control
  ROM `BP_ROM_BENES_MCMWR` 9728 bits × GV.
- **e_cm read** `u_benes_rd0/rd1`: N=`BP_BENES_ECM_M`=512 (padded from neb=400), W=MSG_BITS=8,
  PIPE=`BENES_PIPE_ECM`=3, two ports. Real map: 288 taps → 400 banks. LUT 51,456; control ROM
  `BP_ROM_BENES_ECMRD` 8704 bits × GV (2 ports packed).

Both are driven by control synthesized from `benes_group_matchings` via `benes_control` (power-of-two
Beneš) in the emitter (`qec_q7_bp_graph.rs`), and realised by `bp_benes_ecm_read` / `bp_benes_mcm_wr`
(`hw/bp_benes.sv`, recursive power-of-two `bp_benes_block`).

## Approach — AS-Waksman (arbitrary-size Waksman)

Beauquier & Darrot, "On Arbitrary Size Waksman Networks" (2002): a rearrangeable network for **any**
N (not just 2^k), switch count `⌈N·log2 N⌉ − N + 1` (switch-optimal, ≈ information-theoretic lower
bound). Recursive construction:
- N=1: pass-through (no switch). N=2: one switch.
- N≥3: an input stage of `⌊N/2⌋` switches on pairs (2i, 2i+1); the last element passes straight when
  N is odd. Two subnetworks: upper of size `⌈N/2⌉`, lower of size `⌊N/2⌋`. An output stage of
  `⌈N/2⌉ − 1` switches. One fixed bypass (the Waksman optimisation: the first input switch is fixed to
  bar, saving one switch per recursion) — placement per the paper to preserve rearrangeability.

Routing (control synthesis) is the looping algorithm generalised to the odd/even split: pick an
unrouted output, route it and its input-switch partner through alternating subnetworks, close loops.
Same structure as Beneš's `route` but with the arbitrary-size split and the fixed bypass.

### Why this and not the alternatives

- **Power-of-two Beneš (current):** pads to 1024/512 — the switch/ROM-width waste this lever removes.
- **Concentrator + small Beneš + spray (exploit K=288 « N):** larger potential LUT cut but the probe
  shows LUT is not the binding win, and the concentrator/spray reintroduces the mux-trap risk that
  killed the serial gather (`docs/perf/qec-q7-fixed-bp.md` § Step 3). Rejected for risk/ROI.
- **AS-Waksman** is the switch-optimal *drop-in* for the existing "route a full permutation" contract
  — it changes only the network size/topology, not the surrounding schedule, so the existing
  bit-exact co-sim gate applies unchanged.

## Design

### 1. Routing library `crates/aleph-qec/src/aswaksman.rs`

Mirror `benes.rs`'s structure and its "route + apply share one decomposition" discipline (they agree
by construction, proven by a round-trip oracle):
- `aswaksman_switch_count(n) -> usize` — `⌈n·log2 n⌉ − n + 1` for n≥2, 0 for n≤1.
- `aswaksman_control(perm: &[usize]) -> Vec<bool>` — control bits for a full permutation on `0..n`
  (n arbitrary ≥1), a flat column-major-ish layout the RTL mirrors.
- `aswaksman_apply(ctrl, input) -> Vec<usize>` — fabric simulation sharing the recursion.
- Reuse `complete_partial` (already in `benes.rs`, pads a partial injection to a full permutation on
  `0..m` — works for arbitrary m, no power-of-two assumption; verify and relax the `assert!(m.is_power_of_two())`).
- Tests: round-trip `aswaksman_apply(aswaksman_control(p), identity) == p` for random permutations at
  many N including odd (3,5,7,17,400,800), swap/reverse/identity unit cases, duplicate-target panic.

### 2. Emitter `crates/aleph-qec/examples/qec_q7_bp_graph.rs`

- Add localparams `BP_ASW_MCM_N=800`, `BP_ASW_ECM_N=400` (the real nhb/neb, not padded) and their
  switch/column layout params the RTL needs.
- For m_cm: `complete_partial(&dest_mcm, 800)` → `aswaksman_control` → pack into `BP_ROM_BENES_MCMWR`
  (now `aswaksman_switch_count(800)`-bit rows, not 9728).
- For e_cm: per port `complete_partial(&dest_ecm[p], 400)` → `aswaksman_control` → `BP_ROM_BENES_ECMRD`
  (per-port `aswaksman_switch_count(400)` bits, packed port0-low/port1-high as today).
- Keep the gen-time guard: `aswaksman_apply` reproduces the routed matching, `assert`, both bankings.
- `nvb` (288) taps still index the low lanes; `complete_partial` pads inputs 288→N as today.

### 3. RTL `hw/bp_benes.sv` (+ core wiring)

- Add `bp_asw_block` (recursive arbitrary-size core) + `bp_asw_mcm_wr` / `bp_asw_ecm_read` tops,
  mirroring the existing `bp_benes_*` modules' port shape (`clk`, `din[N][W]`, `ctrl`, `dout[N][W]`)
  and the ctrl-pipelined-with-data timing contract (fresh (din,ctrl)/cycle, `dout@t+PIPE`). The
  recursion handles odd N (straight-through last lane) and the fixed bypass switch.
- `hw/bp_relay_banked_bram_m.sv`: swap `bp_benes_mcm_wr` (N=1024) → `bp_asw_mcm_wr` (N=800) and
  `bp_benes_ecm_read` ×2 (N=512) → `bp_asw_ecm_read` ×2 (N=400). Widths/PIPE unchanged. The
  din-padding loops change bound 1024→800 / 512→400. Latency (PIPE) unchanged → schedule untouched,
  bit-exact.

### 4. Standalone fabric unit test `hw/tb_bp_asw.cpp`

Mirror `tb_bp_benes.cpp`: an independent C++ port of `aswaksman_apply` as the oracle; stream random
(din,ctrl) at N=400 and N=800, assert `dout@t+PIPE == oracle`. Non-tautological (independent re-impl).

### Known design risk — PIPE uniformity under odd N

The existing `bp_benes_block` gets its "exactly PIPE cycles for every lane" property for free because
every node at recursion depth `d` operates on the same block size `M/2^d`, so all paths cross the same
column count. AS-Waksman's arbitrary-size split does **not** guarantee this: N=800 recurses
800→400→200→100→50→**25→13/12**, and odd blocks (25→13+12) make sibling subnetworks differ in depth,
so raw path lengths across the fabric are **non-uniform**. The ctrl-pipelined-with-data timing
contract (`ctrl_pipe[stage(c)]`, `stage(c)=floor(c·PIPE/COLS)`) assumes a single global column count.

**Mitigation (decide in the routing-lib + RTL tasks):** pad each recursion level to a uniform column
budget by inserting straight-through delay stages on the shorter branch (balance path lengths to the
max sibling depth), so the fabric is again depth-uniform and the existing PIPE placement + timing
contract carry over verbatim. This costs a few balancing FFs (cheap — BRAM/LUT are the constraints,
FFs are at 43.6 %), and keeps the co-sim gate and latency invariant intact. If balancing proves to
change the effective switch count enough to erode the BRAM win, that is a stop signal to surface — the
standalone fabric probe (verification step 3) measures it before the full-core integration.

## Verification

1. **Rust round-trip** (library tests) — arbitrary N incl. odd, up to 800.
2. **Gen-time guard** — emitted ROMs proved against `aswaksman_apply`, both bankings, `panic` on
   mismatch.
3. **Standalone fabric co-sim** (`tb_bp_asw.cpp`) — fabric == independent C++ oracle, N=400/800.
4. **Full-core co-sim** — bit-exact **40/40 vs `FixedRelayBp`** at both bankings (`make bpbankedbramm`),
   worst-case latency **unchanged** (PIPE identical).
5. **OOC synth** on xck26 — report the LUT and (the point of this lever) **BRAM tile** delta vs Step-4
   (206,931 LUT / 164.5 tiles). Success = BRAM materially down (target ≤ ~100 %); LUT expected small.

## Out of scope

- Concentrator/spray K«N exploitation (mux-trap risk, LUT not the binding constraint).
- The addr path (already a ROM, Step 4).
- Real-time throughput / #455 — unchanged; this is area only, and a full KV260 fit remains unreachable
  (documented). This lever pursues the BRAM constraint and maximum honest compression, not a fit.

## Files

- `crates/aleph-qec/src/aswaksman.rs` (new) + `lib.rs` export.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (emit AS-Waksman ROMs + guard).
- `hw/bp_benes.sv` (add `bp_asw_*` modules) — or a new `hw/bp_asw.sv`.
- `hw/bp_relay_banked_bram_m.sv` (swap the two fabric instantiations).
- `hw/tb_bp_asw.cpp` (new) + `hw/Makefile` target.
- `docs/perf/qec-q7-fixed-bp.md` (§ Step-5 result).
