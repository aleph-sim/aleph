# Q7-04 M9b — emitter streamgraph/streamvectors + BRAM-ified core + streaming RTL + bit-exact co-sim

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** AC-2 of issue #455 — a streaming schedule (window advance + commit) on the banked
relay-BP core, bit-exact to the windowed software golden in Verilator co-sim, with the core
BRAM-ified so the W=6 window graph fits the KV260.

**Architecture:** Three layers. (1) Software: a hardware-schedule golden `HwSlidingWindowBp`
(ONE baked interior window graph for every slot, zero-pad past stream end) + emitter modes
`streamgraph` (window header + streaming metadata) and `streamvectors` (golden round-streams +
per-slot decisions). (2) Core: `bp_relay_banked_bram.sv`, a sibling of the M8 core with the
O(E) literal-constant fabrics moved into sync-read BRAM ROMs (+1 pipeline re-lag, M8 recipe),
decision-equal to the LUT core. (3) Streaming shell: `bp_streaming_decoder.sv`
(WARM/RUN/WAIT/COMMIT/SLIDE/RELOAD FSM, lift of `uf_streaming_decoder.sv`) +
`bp_stream_win_core.sv` AXI-Stream front-end (lift of `uf_stream_win_core.sv`), gated by a
bit-exact Verilator co-sim vs `streamvectors`.

**Tech Stack:** Rust (aleph-qec), SystemVerilog, Verilator (Mac 5.050 / EPYC 5.032), Vivado
2024.2 batch OOC on `openwebgui.splynx.com`, part `xck26-sfvc784-2LV-c`.

## Global Constraints

- Operating point (M9a sweep verdict, do not re-litigate): **W=6, C=2, residual-only seam**,
  gross code memory-X, rounds=12, p=0.003 for header/vectors generation.
- Frozen decoder constants everywhere: `MSG_BITS=8, FRAC_BITS=3` (Q5.3), `LEGS=6, ITERS=10`,
  `GAMMA=(-0.3,0.9)`, `SEED=0x5E1A_4B9C`, `BANK_SOLVE_SEED=7`.
- Correctness first: every RTL change gates on bit-exact/decision-equal co-sim before synth.
- No `unsafe`; no `unwrap()` in library code (tests OK); clippy `-D warnings`; `cargo fmt`.
- Local test gates are **crate-scoped**: `cargo test -p aleph-qec --release`. Do NOT run
  `cargo test --workspace` locally — that is CI's job.
- **No git worktrees.** Branch `q7-04-m9b-stream-rtl` directly in `/Users/ex/GitHub/aleph`.
- One issue one PR: PR title `[Q7-04] M9b: ...`, body says "Part 2 of 3 for #455 — do not
  auto-close" (no `Closes #455`).
- EPYC box (`ssh root@195.154.249.85`): cargo needs
  `PATH=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH`; verify idle
  (`uptime`, `pgrep -af "cargo bench|bencher run|Runner.Worker"`) before any timing/LER run;
  verilator there is 5.032 and needs `-Wno-LATCH`.
- Vivado box (`ssh root@openwebgui.splynx.com`): `source /tools/Xilinx/Vivado/2024.2/settings64.sh`;
  its git clone is STALE — stage files via rsync into `/root/kv260synth/<dir>`; launch batch
  runs detached (`nohup bash -c '…' &`) and poll `^RESULT` lines in the log.
- Header include-guard stays `BB_GROSS_TANNER_SVH` (house pattern: generated headers are staged
  as `bb_gross_tanner.svh` in scratch build dirs; both checked-in headers share the guard).
- Emitter CLI follows the `wingraph` arg order: `streamgraph rounds p W C bankW bankV`
  (documented deviation from the spec's `bankW bankV W C`).

## Design decisions locked here (deviations from the design spec, with rationale)

1. **Uniform-graph hardware schedule.** The M9a golden compiles a per-slot exact window DEM;
   left-boundary (s=0), right-boundary (windows touching the final-readout slice), and tail
   windows all have *different* graphs. One baked RTL header cannot replay that bit-exactly.
   M9b therefore defines the hardware contract as: **every slot decodes on the single interior
   window graph** (the strict-eq translation-invariant DEM from M9a), the residual frame is
   local (W·DPR bits), the reload region zero-fills past stream end, and every slot uses the
   baked commit rule (var has an in-window detector at relative round < C). Slot count
   ⌈num_slices/C⌉ is unchanged (rounds=12, C=2 → 7 slots), matching the pinned tail contract.
   The commit regions tile [0, C·num_slots) ⊇ [0, num_slices), so every var is committed by the
   tiling and no commit-all special case is needed. The software `HwSlidingWindowBp` implements
   exactly this and is the co-sim golden; its LER delta vs the M9a exact-schedule golden is
   measured on the same shots and documented (expected small; if it is large, that is a finding
   for the results doc, and the escalation is a second baked "final-window" graph — NOT built
   unless data demands it).
2. **No BP_VAR_DET array.** A committed var's residual toggle set (its in-window detectors) is
   already in the header CSR: var v's edges are `BP_VAR_OFF[v] .. BP_VAR_OFF[v+1]` rows of the
   var-sorted edge list, and each edge's detector is `BP_EDGE_CHK[e]`. The RTL commit path
   derives toggles from the existing tables; only `BP_VAR_COMMIT` (1 bit/var) is new.
3. **BRAM-ified core is a sibling file** (`bp_relay_banked_bram.sv`), not a parameter fork of
   the M8 core — the repo's established variant pattern (`bp_relay_fast.sv`,
   `bp_relay_bram*.sv`). The M8 `bp_relay_banked.sv` stays byte-untouched; equality of decode
   decisions between the two IS the correctness gate. The streaming shell instantiates the BRAM
   sibling.
4. **SAT parity fabric stays LUT** (constant-index wire taps + XOR trees are cheap); ROM
   conversion covers the write-scatter/address/select/λ/γ/obs fabrics. Revisit only if the OOC
   probe still overflows.
5. **Commit/residual-toggle fabric is combinational** (UF pattern, one S_COMMIT cycle).
   Fallback if the probe says it's too big: serialize over var groups (GV cycles) — documented
   trade, not built up front.

## File map

- Modify: `crates/aleph-qec/src/relay_window.rs` — add `WindowTrace`, `HwSlidingWindowBp`.
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — add `streamgraph`, `streamvectors`.
- Modify: `crates/aleph-qec/examples/qec_q7_stream_sweep.rs` — add `--hw` comparison mode.
- Create: `hw/bp_relay_banked_bram.sv` (sibling of `bp_relay_banked.sv`).
- Create: `hw/bp_streaming_decoder.sv`, `hw/bp_stream_win_core.sv`, `hw/bp_stream_win.v`.
- Create: `hw/tb_bp_stream.cpp`, `hw/tb_bp_stream_axi.cpp`.
- Commit generated: `hw/bb_stream_tanner.svh`, `hw/bp_stream_vectors.txt`.
- Modify: `hw/Makefile` (targets `bpbram`, `bpbram-lint`, `bpstream`, `bpstream-lint`,
  `bpstream-axi`), `hw/syn/ooc_banked.tcl` (optional top-module tclarg).
- Modify: `docs/perf/qec-q7-fixed-bp.md` (§ M9b), `docs/qec/BACKLOG.md` (AC-2 tick).

---

### Task 1: Hardware-schedule golden `HwSlidingWindowBp`

**Files:**
- Modify: `crates/aleph-qec/src/relay_window.rs`
- Modify: `crates/aleph-qec/examples/qec_q7_stream_sweep.rs`

**Interfaces:**
- Consumes: `SlidingWindowBp::new / window_dem / compile_window` (relay_window.rs),
  `FixedRelayBp::with_budget / decode_fixed_soft / hw_view` (fixed_bp.rs),
  `BBCode::gross().circuit_level_dem(rounds, noise)`,
  `BBMemoryExperiment::detector_rounds()`.
- Produces (Tasks 2, 5, 6 rely on these exact names):

```rust
/// Per-slot decision of the hardware schedule — the unit the RTL must reproduce bit-exactly.
pub struct WindowTrace {
    /// Per window-var: 1 iff the decoder chose the var AND its commit bit is set.
    pub committed: Vec<u8>,
    /// XOR of BP_OBS_MASK over committed vars — this slot's contribution to the logical.
    pub obs: u64,
    /// The base decoder's syndrome-valid flag for this slot.
    pub valid: bool,
    /// True iff the commit region [0, C*dpr) of the frame is all-zero AFTER this slot's
    /// commit toggle — i.e. the rounds about to slide off are drained. A converged decode
    /// always drains them (every var with a det in the commit region has its commit bit
    /// set by construction), so this is the non-vacuous per-slot drain observable; it maps
    /// to the RTL result word's residual_empty bit.
    pub commit_clean: bool,
}

pub struct HwSlidingWindowBp { /* private */ }
impl HwSlidingWindowBp {
    pub fn new(dem: DetectorErrorModel, detector_round: Vec<usize>,
               window: usize, commit: usize) -> Self;
    /// The ONE interior window graph (same offset formula as the emitter:
    /// s0 = ((rounds - window)/2).max(1), rounds = num_slices - 1).
    pub fn window_export(&self) -> &WindowBpExport;
    pub fn dpr(&self) -> usize;                       // detectors per round (72 for gross)
    pub fn commit_mask(&self) -> &[bool];             // baked BP_VAR_COMMIT, len = window vars
    pub fn num_slots(&self) -> usize;                 // ceil(num_slices / commit)
    pub fn decode_stream(&self, syn: &Syndrome) -> (Correction, StreamStats);
    pub fn decode_stream_trace(&self, syn: &Syndrome)
        -> (Correction, StreamStats, Vec<WindowTrace>);
}
```

**Semantics (normative, mirrors the RTL FSM exactly):**

```rust
// construction:
//   assert uniform dpr: detector_round is round-major, num_slices = max+1,
//     every round has the same count (assert, like the UF emitter does)
//   s0 = ((num_slices - 1 - window) / 2).max(1); assert s0 + window <= num_slices - 1
//   export = compile_window(s0, s0 + window)     // the interior graph
//   decoder = FixedRelayBp::with_budget(&export.dem, LEGS, ITERS_PER_LEG, GAMMA, SEED, 8, 3)
//   commit_mask[v] = export.dem.errors[v].dets.iter()
//                        .any(|&d| d / dpr < commit)   // local det round < C
// decode_stream_trace:
//   let wlen = window * dpr;
//   let mut frame = vec![false; wlen];            // local residual frame
//   // warm: load slices 0..window (zero for slices >= num_slices)
//   for k in 0..num_slots {
//       let syn_w = Syndrome from frame;           // frame holds slices [kC, kC+W)
//       let soft = decoder.decode_fixed_soft(&syn_w);
//       let mut committed = vec![0u8; nvars]; let mut obs = 0u64;
//       for v in 0..nvars where soft.ehat[v] == 1 && commit_mask[v] {
//           committed[v] = 1; obs ^= obs_mask[v];
//           for &d in &export.dem.errors[v].dets { frame[d] ^= true; }  // local toggle
//       }
//       trace.push(WindowTrace { committed, obs, valid: soft.converged });
//       // slide by C rounds, reload C fresh slices (zero past stream end):
//       frame.copy_within(commit * dpr .., 0);
//       for r in 0..commit { load slice (k+1)*C + (window-commit) + r  or zeros }
//   }
//   per-slot, after the commit toggle and BEFORE the slide:
//       commit_clean = frame[0 .. commit*dpr) all zero
//   StreamStats { windows: num_slots, nonconverged: count(!valid),
//                 residual: cumulative popcount of frame[0 .. commit*dpr) sampled after
//                           each slot's commit — lit bits DISCARDED by the slide, summed
//                           over all slots. (Measuring the frame after the final slide is
//                           always 0 by construction — the frame is then entirely
//                           zero-padded rounds — so that is NOT the metric.) }
//   Correction: obs XOR accumulated across slots (same construction as
//               SlidingWindowBp::decode_stream uses for its Correction)
```

- [ ] **Step 1: failing tests first.** Add to `relay_window.rs` `mod tests`:

```rust
#[test]
fn hw_interior_graph_matches_window_dem() {
    let (dem, dr) = gross_stream(12, 0.003);           // reuse the module's test helper
    let sw = SlidingWindowBp::new(dem.clone(), dr.clone(), 6, 2);
    let hw = HwSlidingWindowBp::new(dem, dr, 6, 2);
    let s0 = ((12 - 6) / 2).max(1);
    assert_eq!(hw.window_export().dem, sw.window_dem(s0, s0 + 6).dem);
}

#[test]
fn hw_slot_count_and_determinism() {
    // rounds=12 => num_slices=13, C=2 => 7 slots
    let hw = hw_gross(12, 0.003, 6, 2);
    let syn = sample_one_shot(&hw, 0xD00D);
    let (c1, s1, t1) = hw.decode_stream_trace(&syn);
    let (c2, s2, t2) = hw.decode_stream_trace(&syn);
    assert_eq!(t1.len(), 7);
    assert_eq!(s1.windows, 7);
    assert_eq!((c1, s1, t1.iter().map(|t| (t.obs, t.valid)).collect::<Vec<_>>()),
               (c2, s2, t2.iter().map(|t| (t.obs, t.valid)).collect::<Vec<_>>()));
}

#[test]
fn hw_trace_aggregates_to_stream_decode() {
    let hw = hw_gross(12, 0.003, 6, 2);
    for seed in 0..20u64 {
        let syn = sample_one_shot(&hw, seed);
        let (corr, stats) = hw.decode_stream(&syn);
        let (corr_t, stats_t, trace) = hw.decode_stream_trace(&syn);
        assert_eq!(corr, corr_t);
        assert_eq!(stats.residual, stats_t.residual);
        let obs_xor = trace.iter().fold(0u64, |a, t| a ^ t.obs);
        assert_eq!(obs_from(&corr), obs_xor);
    }
}

#[test]
fn hw_converged_stream_drains_residual() {
    // mirror of M9a's converged_stream_drains_residual, on the HW schedule
    let hw = hw_gross(12, 0.003, 6, 2);
    let mut checked = 0;
    for seed in 0..200u64 {
        let syn = sample_one_shot(&hw, seed);
        let (_c, stats) = hw.decode_stream(&syn);
        if stats.nonconverged == 0 { assert_eq!(stats.residual, 0); checked += 1; }
    }
    assert!(checked > 0);
}

#[test]
#[ignore] // slow sanity: HW-schedule LER within 2x of exact-schedule LER, same shots
fn hw_ler_close_to_exact_schedule() {
    // n=2000 shots, p=0.003, rounds=12, W=6 C=2; count logical errors of
    // HwSlidingWindowBp vs SlidingWindowBp on identical sampled shots.
    // assert hw_errors <= 2 * exact_errors + 5   (loose; real number goes in the doc)
}
```

Adapt helper names to what the module's existing tests actually use (there are existing
helpers for building the gross stream DEM and sampling shots — reuse them; do not invent a
second sampling path).

- [ ] **Step 2: run, verify the new tests fail to compile / fail.**
  `cargo test -p aleph-qec --release relay_window` — expect compile errors (types missing).
- [ ] **Step 3: implement `WindowTrace` + `HwSlidingWindowBp`** per the semantics block above.
  Reuse `compile_window` and the module's frozen constants; keep everything a pure function of
  (stream, config). Module doc: add a "hardware schedule" section stating the uniform-graph +
  zero-pad + baked-commit contract and that `streamvectors`/the RTL co-sim gate key on THIS
  struct, while `SlidingWindowBp` remains the exact-schedule LER reference.
- [ ] **Step 4: run tests until green.**
  `cargo test -p aleph-qec --release relay_window` then the ignored one once:
  `cargo test -p aleph-qec --release relay_window -- --ignored hw_ler_close_to_exact_schedule`.
- [ ] **Step 5: extend `qec_q7_stream_sweep.rs`** with a `--hw` flag: same shot set decoded by
  batch `FixedRelayBp`, `SlidingWindowBp`, and `HwSlidingWindowBp`; print a per-p LER table.
  Run locally at n=2000 to smoke it, and on the EPYC (idle check first, PATH prefix) at
  n=20000, p ∈ {0.001, 0.003, 0.005}, rounds=12 — paste the table into the PR notes; Task 8
  copies it into the results doc.
- [ ] **Step 6: `cargo fmt` + `cargo clippy -p aleph-qec --all-targets -- -D warnings`; commit**
  `git commit -m "[Q7-04] M9b: HwSlidingWindowBp — uniform-graph hardware-schedule golden"`.

### Task 2: emitter `streamgraph` + `streamvectors`

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs`
- Commit generated: `hw/bb_stream_tanner.svh`, `hw/bp_stream_vectors.txt`

**Interfaces:**
- Consumes: `HwSlidingWindowBp` (Task 1), existing `print_graph`, `solve_banking`,
  `print_banking`, `verify_banking`, `FixedRelayBp::with_budget/hw_view`.
- Produces: header macros consumed by Tasks 3/5/6 RTL — appended after the banking tables,
  inside the include guard, wrapped in `/* verilator lint_off UNUSEDPARAM */` like the UF
  window emitter:

```systemverilog
localparam int BP_DPR      = 72;               // detectors per round
localparam int BP_WIN_W    = 6;                // window rounds
localparam int BP_WIN_C    = 2;                // commit rounds per slide
localparam int BP_LOAD_LO  = 288;              // (BP_WIN_W-BP_WIN_C)*BP_DPR, reload region base
localparam int BP_SHIFT [BP_C]  = '{...};      // det l -> l - BP_WIN_C*BP_DPR, sentinel BP_C if dropped
localparam bit BP_VAR_COMMIT [BP_N] = '{...};  // 1 iff var has an in-window det at rel round < C
```

  and the vector file `hw/bp_stream_vectors.txt`:

```
# qec_q7_bp_graph streamvectors <rounds> <p> <W> <C> <n> <seed>  (+ regen command comment)
T SLICES DPR SLOTS BP_N BP_OBS
r <DPR bits '0'/'1'>          <- SLICES round lines per trial, round 0 first, det 0 first
w <slot> <BP_N bits committed> <BP_OBS bits obs> <0|1 vflag> <0|1 commit_clean>   <- SLOTS lines per trial
```

- [ ] **Step 1: `streamgraph` mode.** CLI `streamgraph rounds p W C bankW bankV` (wingraph arg
  order). Implementation: build `HwSlidingWindowBp`, take `window_export()`, run the existing
  `print_graph` + `solve_banking` + `print_banking` on a `FixedRelayBp::with_budget` of the
  window DEM (exactly what `emit_win_graph` does), then emit the streaming metadata block
  above, then `` `endif``. Derivations:
  - `BP_DPR` = `hw.dpr()`; assert `BP_C == BP_WIN_W * BP_DPR` (every window detector is a check).
  - `BP_SHIFT[l]` = `l - C*DPR` for `l >= C*DPR`, else sentinel `BP_C` (emit via the UF
    `round_start` pattern, not the closed form, so a non-uniform future code fails loudly).
  - `BP_VAR_COMMIT[v]` = `hw.commit_mask()[v]`.
  - Inline asserts (house style, like `verify_banking`): commit tiling — every det round of
    every var is < `C * ceil(num_slices/C)`; every var has >= 1 in-window det; SHIFT is
    injective on kept dets.
  - Provenance comment: full regen command line.
- [ ] **Step 2: `streamvectors` mode.** CLI `streamvectors rounds p W C n seed`. Sample `n`
  multi-round shots (same `sample_shots` path the other vector modes use), run
  `hw.decode_stream_trace` per shot, write the format above. Round lines are the RAW stream
  detector bits (the RTL XORs commits into its own frame; do NOT pre-toggle).
- [ ] **Step 3: generate + commit the artifacts.**

```bash
cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- streamgraph 12 0.003 6 2 8 24 > hw/bb_stream_tanner.svh
cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- streamvectors 12 0.003 6 2 40 7 > hw/bp_stream_vectors.txt
```

  (8/24 banking provisional; Task 4's probe may drop it to 4/12 — regen + recommit there.)
- [ ] **Step 4: gates.** `cargo fmt`, `cargo clippy -p aleph-qec --all-targets -- -D warnings`,
  `cargo test -p aleph-qec --release`. Sanity-grep the header: `BP_WIN_W`, `BP_VAR_COMMIT`
  present; `grep -c` the vectors file: `T`-line SLOTS=7, 40 trials × (13 r-lines + 7 w-lines).
- [ ] **Step 5: commit** `"[Q7-04] M9b: emitter streamgraph/streamvectors — window header + commit metadata + HW-schedule golden vectors"`.

### Task 3: BRAM-ified core sibling `bp_relay_banked_bram.sv`

**Files:**
- Create: `hw/bp_relay_banked_bram.sv` (from a copy of `hw/bp_relay_banked.sv`)
- Modify: `hw/Makefile` (add `bpbram-lint`, `bpbram`)

**Interfaces:**
- Consumes: `bb_gross_tanner.svh` macro set (unchanged), `check_minsum.sv`, `var_update.sv`.
- Produces: module `bp_relay_banked_bram` with the **identical port list** to
  `bp_relay_banked` (`clk, rst_n, in_valid, early_exit, syndrome_in[BP_C], busy, out_valid,
  corr_out[BP_N], obs_flip[BP_OBS-1:0], valid_flag, latency_cycles[31:0]`) and identical
  decode decisions (different latency). Tasks 5–7 instantiate THIS module.

**Conversion recipe** (the M9a probe showed the 169 % LUT is the O(E) constant-mux fabric;
the M8 register-plane move is the re-lag template — "values bit-exact, only cycle counts
grow"):

| # | site (line anchors in `bp_relay_banked.sv`) | today | becomes |
|---|---|---|---|
| R1 | m_cm write scatter comb `:457-479` (keyed `wg`) + `bp_mvm_cell` write decode `:213-216` | per-group constants from `vedge_at`/`BP_EDGE_HB/ROW`/`BP_LAMBDA` unrolled per group | `SCATTER_ROM[GV]`, sync read; row packs per-slot/lane {hb, row, λ-seed}; cells take the decoded control word as input ports instead of deriving internally |
| R2 | e_cm read-address comb `:484-504` (keyed `pc`) + `bp_ecm_cell` decode `:178` | `BP_EDGE_EB/ROW/EPORT` unrolled per group | `ECM_ADDR_ROM[GC]`, sync read; row packs {eb, row, eport} per (j,k) lane |
| R3 | CHK gather HB selects `:572-589` | constant `BP_EDGE_HB[e]` taps per (j,k) | `CHK_SEL_ROM[GC]` feeding the (unchanged, still-LUT) gather muxes |
| R4 | VAR gather `:627-655` | `BP_LAMBDA[v]`, `BP_GAMMA[l*BP_N+v]` (6:1 leg mux), `BP_EDGE_EB/EPORT` selects | `LAMBDA_ROM[GV]` (data), `GAMMA_ROM[BP_LEGS*GV]` addressed by {leg, group} (data), `VAR_SEL_ROM[GV]` (selects) |
| R5 | SAT parity `:731-741`, `:806-834` | constant-index `ehat` taps + XOR trees | **KEEP as-is** (wire taps are cheap) |
| R6 | S_EMIT obs `:842-864` | `BP_OBS_MASK[v]` unrolled | `OBS_ROM[GV]` row = V×BP_OBS bits, +1 lag on the accumulate |

ROM style: `logic [W-1:0] rom [DEPTH]` initialized from the existing localparam arrays in an
`initial`/generate copy, sync read `q <= rom[addr]`, `(* rom_style = "block" *)`. Depths are
GC/GV (tiny), widths are wide — Vivado packs them into parallel BRAM36 columns; URAM
(`ram_style = "ultra"`) is the overflow lever (verify URAM ROM init works on Vivado 2024.2 in
the Task 4 probe before relying on it).

**Re-lag table (M8 recipe, +1 everywhere a ROM feeds a launch or scatter):**

| schedule constant | M8 value | BRAM value |
|---|---|---|
| S_CHECK phase end | `pc == GC+3` | `pc == GC+4` |
| e_cm scatter gate / group | `pc>=4`, `wg=pc-4` | `pc>=5`, `wg=pc-5` |
| S_VAR phase end | `pc == GV+2` | `pc == GV+3` |
| m_vm/m_cm scatter gate / group | `pc>=3`, `wg=pc-3` | `pc>=4`, `wg=pc-4` |
| `ehat`/`ehat_w` update | `pc-3` | `pc-4` |
| submodule `en` delay | `en_chk_r`/`en_var_r` (1 stage) | +1 stage (`en_chk_rr`/`en_var_rr`) |
| S_INIT | `pc = 0..GV-1`, write at `pc` | `pc = 0..GV`, ROM read at `pc`, write at `pc-1` |
| S_EMIT | write at `pc` | ROM read at `pc`, accumulate/write at `pc-1`, tail +1 |
| SAT finalize / early-exit | finalize `pc==GC-1` | unchanged (R5 kept LUT) — re-verify against waves |

The m_vm read-row(pc)/write-row(pc−4) disjointness argument gets MORE slack; update the
comment. The `` `ifndef SYNTHESIS`` elaboration guards (`:268-371`) MUST survive verbatim —
they are the emitter-consistency safety net.

- [ ] **Step 1: copy + rename.** `cp hw/bp_relay_banked.sv hw/bp_relay_banked_bram.sv`; rename
  module and the file-top comment (state the sibling relationship + this plan's recipe).
  **Do not touch `bp_relay_banked.sv`.** Note `$unit`-scope helpers (`chk_at/var_at/...` at
  `:89-114`) will collide if both siblings are compiled together — guard them with
  `` `ifndef BP_BANKED_HELPERS`` / define, or suffix the copies `_bq` in the sibling; pick one
  and keep the Makefile targets single-module-per-build (house pattern already builds one top
  per scratch dir).
- [ ] **Step 2: lint target first.** Makefile `bpbram-lint`: mirror `bpbanked-lint`
  (`--lint-only -Wall`, plus `-Wno-LATCH` when run on the EPYC). Run it; fix elaboration.
- [ ] **Step 3: co-sim equality gate.** Makefile `bpbram`: mirror `bpbanked` (`Makefile:268-277`)
  — regen `circgraph 1 0.003 $W $V` header into the scratch dir, build
  `check_minsum.sv var_update.sv bp_relay_banked_bram.sv` with `tb_bp_banked.cpp` **unchanged**
  via `--prefix Vbp_relay_banked` (Verilator `--prefix` keeps the TB's model class name), run
  against `bp_dec_vectors.txt`/`bp_circ_vectors.txt` exactly as `bpbanked` does, at BOTH 16/48
  and 8/24. Gate: 40/40 decision-equal (corr, obs, vflag) at both bankings; latency grows by
  roughly +(GC+GV)·legs·iters/… cycles — record before/after cycle counts (expect ≈ +100–200
  on rounds=1; anything ×2 means a schedule bug).
- [ ] **Step 4: run `bpbram` + `bpbram-lint` on the Mac; commit**
  `"[Q7-04] M9b: bp_relay_banked_bram — E-fabric to sync-read BRAM ROMs, +1 re-lag, decision-equal to M8 core"`.

### Task 4: OOC fit probe (the gate before any FSM work)

**Files:**
- Modify: `hw/syn/ooc_banked.tcl` — add optional 3rd tclarg `top` (default `bp_relay_banked`).
- Possibly regen: `hw/bb_stream_tanner.svh` (if 8/24 fails and 4/12 fits).

**Interfaces:**
- Consumes: `bp_relay_banked_bram.sv` (Task 3), `bb_stream_tanner.svh` (Task 2).
- Produces: a RESULT line per config recorded in the PR + Task 8 doc; the FINAL banking choice
  for the committed stream header.

- [ ] **Step 1:** add the `top` tclarg to `ooc_banked.tcl` (keep default behavior identical).
- [ ] **Step 2:** stage on the Vivado box (rsync, stale clone — never rely on its git):
  `/root/kv260synth/m9b_bram_w6_824/` ← `check_minsum.sv var_update.sv bp_relay_banked_bram.sv
  ooc_banked.tcl` + `bb_stream_tanner.svh` **copied as `bb_gross_tanner.svh`**.
- [ ] **Step 3:** run detached at 5 ns:
  `nohup bash -c 'source /tools/Xilinx/Vivado/2024.2/settings64.sh && vivado -mode batch -source ooc_banked.tcl -tclargs 5.0 m9b_bram_w6_824 bp_relay_banked_bram' > synth.log 2>&1 &`
  Poll `grep ^RESULT synth.log`. Also probe the M9a baseline sanity: the SAME staged dir minus
  BRAM core (LUT core) is already known: 169 % — no need to re-run.
- [ ] **Step 4: evaluate the gate.** PASS = CLB LUT ≤ ~95 % (≈111 k), RAMB36 ≤ 144, URAM ≤ 64,
  Fmax ≥ 140 MHz (margin over the 133.332 PS grid). Check `util_hier.rpt` to confirm the ROMs
  actually mapped to block RAM (RAMB > 0; if RAMB=0 the `rom_style` didn't take — fix
  attributes/coding style before concluding anything).
- [ ] **Step 5: fallback ladder, in order, each documented with its RESULT line:**
  (a) 4/12 banking (regen header, restage); (b) ROMify R5 SAT + serialize the commit fabric;
  (c) W=4, C=2 (regen everything; the sweep table prices the LER cost 2.2–5.6× — this is the
  last resort and needs an explicit note in the results doc). Stop at the first PASS; recommit
  `hw/bb_stream_tanner.svh` + `hw/bp_stream_vectors.txt` at the final (W, C, banking) if it
  changed from Task 2's provisional 8/24.
- [ ] **Step 6: commit** the tcl change + any header regen:
  `"[Q7-04] M9b: OOC fit probe — BRAM core + W=6 window header on KV260 (RESULT: <numbers>)"`.

### Task 5: streaming FSM `bp_streaming_decoder.sv` + golden co-sim

**Files:**
- Create: `hw/bp_streaming_decoder.sv`, `hw/tb_bp_stream.cpp`
- Modify: `hw/Makefile` (`bpstream-lint`, `bpstream`)

**Interfaces:**
- Consumes: `bp_relay_banked_bram` ports (Task 3), header macros `BP_DPR, BP_WIN_W, BP_WIN_C,
  BP_LOAD_LO, BP_SHIFT, BP_VAR_COMMIT, BP_OBS_MASK, BP_VAR_OFF, BP_EDGE_CHK` (+ base CSR)
  (Task 2), `hw/bp_stream_vectors.txt` (Task 2).
- Produces: module used verbatim by Task 6:

```systemverilog
module bp_streaming_decoder (
    input  logic clk, rst_n,
    input  logic early_exit,
    input  logic in_valid,                  // one round per accepted handshake
    input  logic [BP_DPR-1:0] in_round,
    input  logic in_last,                   // asserted with the stream's final round
    output logic in_ready,
    output logic out_valid,                 // one pulse per slot
    output logic [BP_OBS-1:0] out_obs,      // this slot's committed-obs XOR
    output logic out_vflag,
    output logic out_last,                  // pulses with the final slot's out_valid
    output logic out_commit_clean,          // commit region drained after this slot's commit
    output logic commit_corr [BP_N],        // per-slot committed vars (TB gate; unconnected in AXI wrap)
    output logic [15:0] last_latency
);
```

**FSM (lift of `uf_streaming_decoder.sv`, states S_WARM → S_RUN → S_WAIT → S_COMMIT → S_SLIDE
→ S_RELOAD → S_RUN):**

- Residual frame `res [BP_WIN_W*BP_DPR-1:0]`; `core_syn` = unpack of `res` into
  `syndrome_in[BP_C]` (assert `BP_C == BP_WIN_W*BP_DPR` at elab).
- Round/slot bookkeeping: `slices_seen` counts accepted rounds until `in_last`;
  `slots_total = ceil(slices_seen / BP_WIN_C)` latched at `in_last`; `slots_done` counts
  emitted slots. **After `in_last`, reload/warm cursor zero-fills instead of consuming input**
  (the internal zero-pad drain — this is the ⌈num_slices/C⌉ tail contract).
- S_WARM: fill `res[lptr +: BP_DPR] <= in_round`, W rounds (or zero-pad past `in_last`).
- S_RUN: pulse core `in_valid` 1 cycle. S_WAIT: wait `out_valid`, latch `latency_cycles[15:0]`
  (saturate) and `valid_flag`.
- S_COMMIT (1 cycle, combinational fabric): `committed[v] = corr_out[v] & BP_VAR_COMMIT[v]`;
  `out_obs` = XOR-reduce `BP_OBS_MASK[v][BP_OBS-1:0]` over committed vars; residual toggle:
  for each committed v, for each edge `e in BP_VAR_OFF[v]..BP_VAR_OFF[v+1]`,
  `tog[BP_EDGE_CHK[e]] ^= 1`; `res <= res ^ tog`; drive `commit_corr <= committed`; pulse
  `out_valid` with `out_vflag = valid_flag`, `out_last = (slots_done+1 == slots_total)`,
  and `out_commit_clean = ((res ^ tog)[BP_WIN_C*BP_DPR-1:0] == '0)` — the golden's
  per-slot `WindowTrace.commit_clean` (drain observable over the rounds about to slide off).
- S_SLIDE: `res <= res >> (BP_WIN_C*BP_DPR)` — equivalently the `BP_SHIFT` map; use the shift
  map form so a future non-uniform layout still works — `lptr <= BP_LOAD_LO`.
- S_RELOAD: load `BP_WIN_C` rounds into `[BP_LOAD_LO, BP_C)` (zeros past `in_last`) → S_RUN;
  when `slots_done == slots_total` → S_WARM (frame done).
- `in_ready = (state inside {S_WARM, S_RELOAD}) && !seen_last`.

- [ ] **Step 1:** write the module; `make bpstream-lint` (mirror of `bpbanked-lint` with the
  stream header staged as `bb_gross_tanner.svh` in `_bpstreambuild/`).
- [ ] **Step 2: TB.** `tb_bp_stream.cpp` (adapt `tb_bp_banked.cpp`'s vector parser + latency
  histogram to the `bp_stream_vectors.txt` format): per trial, drive SLICES rounds
  (handshake, `in_last` on the final one), collect SLOTS results, compare
  `{commit_corr, out_obs, out_vflag, out_commit_clean}` **bit-exact** per slot, all 40 trials, both
  `early_exit=0` and `early_exit=1` runs (golden is schedule-independent: same decisions).
  Guard: 32 M cycles/trial.
- [ ] **Step 3:** Makefile `bpstream` target (scratch dir `_bpstreambuild`, stream header copy,
  `--prefix` trick not needed here — fresh TB). Run: expect **40/40 ×2 modes**. Debug until
  green; the golden is truth (any mismatch is an RTL bug or a Task 1/2 contract bug — if the
  contract is wrong, fix it in Rust FIRST, regen vectors, then re-gate).

**AMENDMENT (execution finding):** "the golden is schedule-independent — identical decisions
both modes" was WRONG: the core's early-exit takes the FIRST syndrome-valid leg, the software
golden keeps the BEST-KEPT decision over all legs; they differ whenever first-valid ≠ best-kept
(25/280 slots at the op point). The house pattern (M6–M8 `circvectorsearly`) applies: each mode
gets its OWN golden. `HwSlidingWindowBp` gains `with_early_exit(bool)` (passthrough to
`FixedRelayBp::with_early_exit`, trace unchanged otherwise); emitter gains `streamvectorsearly`
(same CLI as `streamvectors`) emitting `hw/bp_stream_vectors_early.txt` (committed); the TB
gates early_exit=0 against `bp_stream_vectors.txt` and early_exit=1 against the early file,
both **40/40 bit-exact**.
- [ ] **Step 4: commit** `"[Q7-04] M9b: bp_streaming_decoder — W/C sliding FSM, bit-exact 40/40 vs HW-schedule golden"`.

### Task 6: AXI-Stream front-end + robustness co-sim

**Files:**
- Create: `hw/bp_stream_win_core.sv`, `hw/bp_stream_win.v`, `hw/tb_bp_stream_axi.cpp`
- Modify: `hw/Makefile` (`bpstream-axi`)

**Interfaces:**
- Consumes: `bp_streaming_decoder` (Task 5), UF template `uf_stream_win_core.sv` +
  `uf_stream_win.v` + `tb_uf_stream_win.cpp`.
- Produces: `bp_stream_win_core` (s_axis/m_axis 32-bit + tlast + `early_exit_i` input) and the
  Verilog-2001 BD-top passthrough `bp_stream_win.v` — the M9c build units.

**Contract:**
- Input: one round = **3 MM2S beats**: beat0 = round bits [31:0], beat1 = [63:32], beat2 =
  {24'b0, bits[71:64]}; `in_last` ← tlast (on the round completed by the tlast beat).
- Output: **one 32-bit S2MM word per slot**:
  `[31:20] = out_obs[11:0]`, `[19] = vflag`, `[18] = commit_clean`, `[17:16] = 2'b00`,
  `[15:0] = latency` (saturating). tlast on the frame's final slot (`out_last`).
- 1-deep result slot: `s_axis_tready = dec_ready & ~out_full & ~frame_rst` (UF pattern).
- Per-frame re-arm: when the tlast-tagged result word is consumed, pulse `frame_rst` for one
  cycle (`dec_rst_n = aresetn & ~frame_rst`) — the Q6-20 mid-stream-resume fix, port it
  verbatim.

- [ ] **Step 1:** write `bp_stream_win_core.sv` + `bp_stream_win.v` (blind 32-bit passthrough,
  same reason as UF: Vivado BD module-reference tops must be Verilog). `bpstream-axi` Makefile
  target greps `BP_DPR/BP_WIN_W/BP_WIN_C/BP_OBS` from the generated header into `-CFLAGS -D`
  (the `stream-axi` pattern at `Makefile:374-387`).
- [ ] **Step 2: TB gates** (`tb_bp_stream_axi.cpp`, mirror `tb_uf_stream_win.cpp` §§):
  1. zero stream → all slots obs=0, vflag=1, commit_clean=1 on every word, exact slot
     count ⌈13/2⌉=7, tlast on word 7 only;
  2. golden equality — 40 vector trials, output words' obs/vflag fields bit-equal to
     `bp_stream_vectors.txt` (latency field: assert > 0, not golden-compared);
  3. back-pressure invariance — splitmix64-random `m_axis_tready`, byte-identical word
     sequence vs full-speed;
  4. frame independence — 3 frames back-to-back, no external reset, per-frame counts + gate 2
     still hold.
- [ ] **Step 3:** run `make bpstream-axi` → all four gates green, both early-exit modes.
- [ ] **Step 4: commit** `"[Q7-04] M9b: bp_stream_win_core AXI front-end — bit-exact + back-pressure + frame-independence co-sim"`.

### Task 7: OOC probe of the full streaming top

**Files:** none new (staging only; possible small RTL fixes if timing/fit regress).

- [ ] **Step 1:** stage `/root/kv260synth/m9b_stream_top/`: `check_minsum.sv var_update.sv
  bp_relay_banked_bram.sv bp_streaming_decoder.sv bp_stream_win_core.sv` + final stream header
  as `bb_gross_tanner.svh`; run `ooc_banked.tcl -tclargs 5.0 m9b_stream_top bp_stream_win_core`
  (top tclarg from Task 4).
- [ ] **Step 2:** gate: LUT ≤ ~95 %, RAMB/URAM in budget, Fmax ≥ 140 MHz. If the commit fabric
  or residual mux breaks timing/fit: serialize the commit over var groups (design decision 5
  fallback) and re-gate co-sim (Task 5/6 gates rerun) before re-probing.
- [ ] **Step 3:** record the RESULT + `util_hier.rpt` breakdown for the doc; commit any fixes.

### Task 8: docs + backlog

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md` (new § M9b), `docs/qec/BACKLOG.md` (Q7-04 AC-2).

- [ ] **Step 1:** results-doc § M9b: what shipped; the uniform-graph HW-schedule contract and
  WHY (one baked header), zero-pad drain, no-BP_VAR_DET derivation; HW-vs-exact-schedule LER
  table (Task 1 EPYC run); BRAM-ification: site table + re-lag + before/after OOC fit
  (169 %/138 % → final numbers) + final banking; co-sim gates (40/40 ×2 modes, back-pressure,
  frame independence, zero-stream); cycle counts per window at the final config and the honest
  µs/window estimate vs the C µs real-time budget (the M9c measurement is the verdict — do not
  overclaim); deviations from the design spec (arg order, BP_VAR_DET, uniform schedule) each
  with one-line rationale.
- [ ] **Step 2:** tick AC-2 in `docs/qec/BACKLOG.md` § Q7-04 with a one-line annotation
  (PR number + "bit-exact 40/40, W=6 C=2 @ <banking>").
- [ ] **Step 3:** commit `"[Q7-04] M9b: results doc + AC-2 tick"`.

### Task 9: PR

- [ ] **Step 1:** full local gate sweep: `cargo fmt --check`,
  `cargo clippy -p aleph-qec --all-targets -- -D warnings`,
  `cargo test -p aleph-qec --release`, `make -C hw bpbram bpstream bpstream-axi` + lints.
- [ ] **Step 2:** push branch, open PR `[Q7-04] M9b: streaming relay-BP RTL — BRAM-ified core +
  window FSM + AXI front-end, bit-exact co-sim`. Body: "Part 2 of 3 for #455 (M9b of
  `docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md`). **Do not auto-close
  #455** — M9c (silicon) follows." + approach summary + all gate evidence (co-sim counts, OOC
  RESULT lines, LER table, cycle counts) + deviations.
- [ ] **Step 3:** two review passes (fresh-eyes self-review, then adversarial review of the
  diff), fix, wait for CI green, squash-merge.

## Self-review checklist (run after writing, before executing)

- Spec coverage: AC-2 (streaming schedule bit-exact in co-sim) → Tasks 5/6; emitter →
  Task 2; fit lever → Tasks 3/4; M9a memory's "M9b inputs" (BRAM-ify, tail contract,
  streamgraph metadata) all have tasks. ✓
- The known contract risks are pinned as tests BEFORE RTL: interior-graph strict-eq,
  slot count, drain, trace aggregation. ✓
- Type/name consistency: `BP_WIN_W/BP_WIN_C/BP_DPR/BP_LOAD_LO/BP_SHIFT/BP_VAR_COMMIT`,
  `HwSlidingWindowBp::{window_export, commit_mask, dpr, num_slots, decode_stream,
  decode_stream_trace}`, `WindowTrace{committed, obs, valid}`, module names
  `bp_relay_banked_bram`, `bp_streaming_decoder`, `bp_stream_win_core` — used identically
  across tasks. ✓
