# M9c Step 5 — AS-Waksman right-sizing: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the power-of-two-padded Beneš fabrics for m_cm write (N=1024) and e_cm read (N=512×2) with switch-optimal **AS-Waksman** networks sized to the real dimensions (N=800 / N=400), bit-exact, to narrow the control ROMs (BRAM lever; BRAM at 114.2 % after Step 4).

**Architecture:** A new arbitrary-size routing library (`aswaksman.rs`, mirroring `benes.rs`'s "route + apply share one recursion, proven by round-trip oracle" discipline) feeds AS-Waksman control ROMs from the emitter; new depth-balanced RTL fabrics (`bp_asw.sv`) realise them with the same ctrl-pipelined-with-data timing contract as `bp_benes.sv`, so the schedule and latency are untouched and the existing 40/40 co-sim gate applies.

**Tech Stack:** Rust (`aleph-qec` lib + `qec_q7_bp_graph` example), SystemVerilog (`hw/bp_asw.sv`, `hw/bp_relay_banked_bram_m.sv`), Verilator (`tb_bp_asw.cpp`, `make bpbankedbramm`), Vivado OOC synth (xck26) on EPYC `root@195.154.249.85`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-m9c-aswaksman-rightsize-design.md`.
- **Bit-exact 40/40 vs `FixedRelayBp`** at both bankings (`8 24`, `16 48`) via `make bpbankedbramm`, and worst-case latency **unchanged** (8/24→4475, 16/48→2810) — the fabrics keep the same PIPE, so latency must not move.
- **Routing correctness is gated by a round-trip oracle** (`aswaksman_apply(aswaksman_control(p), identity)` realises `p`) — `route` and `apply` MUST share one recursion so they agree by construction, exactly as `benes.rs` does. No hand-mirrored second wiring formula.
- AS-Waksman switch count ≈ `n·log2 n − n + 1` (n≥2), computed exactly by `aswaksman_switch_count` (deterministic — use whatever it returns, ~6900–7200 for N=800, ~3100–3300/port for N=400); the invariant that matters is it is **materially < the Beneš 9728 (N=1024) / 4352 (N=512)** it replaces.
- Arbitrary N including ODD sub-blocks (800→…→25→13/12): the recursion and the fabric must handle odd N.
- Only the m_cm write + e_cm read fabrics change. The addr path (ROM, Step 4), the write/read *data* semantics, PIPE values (`BENES_PIPE_MCM`=4, `BENES_PIPE_ECM`=3), and everything else stay.
- Rust: `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check` clean; no `unwrap`/`expect` in lib outside tests.
- Branch `q7-04-m9c-aswaksman` (created off `origin/main`). Commit per task.

---

## File Structure

- `crates/aleph-qec/src/aswaksman.rs` (new) — arbitrary-size Waksman routing: `aswaksman_switch_count`, `aswaksman_columns`/layout, `aswaksman_control`, `aswaksman_apply`, shared recursion. Export via `lib.rs`.
- `crates/aleph-qec/src/benes.rs` — relax `complete_partial`'s `assert!(m.is_power_of_two())` to accept arbitrary `m` (it already fills unused outputs ascending; no power-of-two dependence in the body). Confirm and adjust.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — emit AS-Waksman `BP_ROM_BENES_MCMWR`/`BP_ROM_BENES_ECMRD` at N=800/400 + layout localparams; guard against `aswaksman_apply`.
- `hw/bp_asw.sv` (new) — `bp_asw_switch` (reuse `bp_benes_switch` if identical), `bp_asw_block` (arbitrary-size, depth-balanced), `bp_asw_mcm_wr`, `bp_asw_ecm_read`.
- `hw/tb_bp_asw.cpp` (new) + `hw/Makefile` target `bpasw` — standalone fabric vs independent C++ oracle.
- `hw/bp_relay_banked_bram_m.sv` — swap the two Beneš fabric instantiations for `bp_asw_*`; adjust din-padding bounds 1024→800 / 512→400.
- `docs/perf/qec-q7-fixed-bp.md` — § Step-5 result.

---

## Task 1: `aswaksman.rs` routing library (control + apply, round-trip oracle)

**This is the algorithm-heavy task. The round-trip test is the ground truth — derive the recursion with TDD against it.** Reference: Beauquier & Darrot, "On Arbitrary Size Waksman Networks" (2002); the power-of-two special case must match `benes.rs`'s looping-algorithm structure.

**Files:**
- Create: `crates/aleph-qec/src/aswaksman.rs`
- Modify: `crates/aleph-qec/src/lib.rs` (add `mod aswaksman;` + `pub use`), `crates/aleph-qec/src/benes.rs` (`complete_partial` power-of-two relax)

**Interfaces:**
- Produces: `aswaksman_switch_count(n: usize) -> usize`; `aswaksman_control(perm: &[usize]) -> Vec<bool>` (flat layout, documented, that the RTL mirrors); `aswaksman_apply(ctrl: &[bool], input: &[usize]) -> Vec<usize>`; a layout descriptor the emitter/RTL need (e.g. `aswaksman_columns(n)` or an explicit per-level switch/column map). Consumes `complete_partial` (arbitrary m).

- [ ] **Step 1: Write the round-trip oracle test FIRST (this is the correctness spec)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn check(perm: &[usize]) {
        let ctrl = aswaksman_control(perm);
        assert_eq!(ctrl.len(), aswaksman_switch_count(perm.len()));
        let ident: Vec<usize> = (0..perm.len()).collect();
        let routed = aswaksman_apply(&ctrl, &ident);
        for (i, &o) in perm.iter().enumerate() {
            assert_eq!(routed[o], i, "perm {perm:?}: output {o} got {} not {i}", routed[o]);
        }
    }

    #[test] fn identity_small() { for n in 1..=17 { check(&(0..n).collect::<Vec<_>>()); } }
    #[test] fn reverse_small() { for n in 1..=17 { check(&(0..n).rev().collect::<Vec<_>>()); } }
    #[test] fn odd_sizes() { for &n in &[3usize,5,7,9,25,400,800] {
        // a fixed non-trivial perm: rotate by 1
        let p: Vec<usize> = (0..n).map(|i| (i+1)%n).collect(); check(&p);
    }}
    #[test] fn random_permutations() {
        // deterministic LCG (no rand dep, matches benes.rs style)
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut rng = || { s ^= s<<13; s ^= s>>7; s ^= s<<17; s };
        for &n in &[2usize,3,5,8,13,25,64,400,512,800] {
            for _ in 0..200 {
                let mut p: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() { let j = (rng() as usize) % (i+1); p.swap(i,j); }
                check(&p);
            }
        }
    }
    #[test] #[should_panic] fn duplicate_target_rejected() {
        // if the lib validates injectivity of a partial before completion; else drop this test
        let _ = aswaksman_control(&[0,0,2]);
    }
}
```

- [ ] **Step 2: Run the tests — verify they fail to compile (functions absent)**

Run: `cargo test -p aleph-qec --lib aswaksman 2>&1 | tail`
Expected: FAIL — `aswaksman_switch_count`/`_control`/`_apply` not found.

- [ ] **Step 3: Implement `aswaksman_switch_count` + the layout**

```rust
//! Arbitrary-size (AS-)Waksman rearrangeable permutation-network routing.
//! Switch-optimal generalisation of Beneš to any N (Beauquier & Darrot 2002). `route`
//! (control synthesis) and `apply` (fabric sim) share ONE recursion so they agree by
//! construction — same discipline and round-trip oracle as benes.rs.

/// Switches in an AS-Waksman network on `n` inputs: `ceil(n*log2 n) - n + 1` (0 for n<=1).
pub fn aswaksman_switch_count(n: usize) -> usize {
    if n <= 1 { return 0; }
    // recursion: floor(n/2) input switches + (ceil(n/2)-1) output switches + subnets,
    // with the Waksman fixed-bypass already folded into the closed form below.
    let mut total = 0usize;
    fn rec(n: usize, total: &mut usize) {
        if n <= 1 { return; }
        if n == 2 { *total += 1; return; }
        *total += (n / 2) + (n.div_ceil(2) - 1); // input + output stage switches
        rec(n.div_ceil(2), total); // upper
        rec(n / 2, total);         // lower
    }
    rec(n, &mut total);
    total
}
```
(The closed form `⌈n·log2 n⌉−n+1` is the invariant to assert against `rec` for a few n; keep `rec` as the source of truth for the flat control layout so control offsets and switch count agree.)

- [ ] **Step 4: Implement `aswaksman_control` + `aswaksman_apply` sharing the recursion**

Derive the AS-Waksman recursion (input stage of `n/2` switches on pairs, straight-through last lane when `n` odd; upper subnet size `n.div_ceil(2)`, lower `n/2`; output stage of `n.div_ceil(2)-1` switches; the Waksman fixed bypass on the designated switch). Use a **flat `Vec<bool>` control indexed by a running switch counter** in recursion order (input switches of this block, then recurse upper, then lower, then output switches) — the SAME order the RTL `bp_asw_block` will instantiate, so offsets agree by construction. The 2-colouring/looping logic is the `benes.rs::route` cycle-walk generalised: an odd block has one unpaired input/output that is pinned to a fixed subnet. Make `aswaksman_apply` walk the identical recursion reading the same counter order. **The `check` round-trip test is the gate — iterate until `random_permutations` passes for all N incl. odd/large.**

- [ ] **Step 5: Run tests to green**

Run: `cargo test -p aleph-qec --lib aswaksman 2>&1 | tail`
Expected: PASS (identity/reverse/odd/random/should_panic all green).

- [ ] **Step 6: Relax `complete_partial` for arbitrary m + export**

In `benes.rs`, change `pub fn complete_partial(dest: &[Option<usize>], m: usize)`'s `assert!(m.is_power_of_two())` to just `assert_eq!(dest.len(), m)` (the body fills unused outputs ascending — no power-of-two need). Add a test `complete_partial(&[Some(2),None,Some(0)], 3)` → `[2,1,0]`. In `lib.rs`: `mod aswaksman;` + `pub use aswaksman::{aswaksman_apply, aswaksman_control, aswaksman_switch_count};`.

- [ ] **Step 7: clippy + fmt + commit**

```bash
cargo clippy -p aleph-qec --all-targets -- -D warnings 2>&1 | tail -3
cargo fmt -p aleph-qec
git add crates/aleph-qec/src/aswaksman.rs crates/aleph-qec/src/lib.rs crates/aleph-qec/src/benes.rs
git commit -m "[Q7-04] M9c Step 5a: AS-Waksman routing lib (arbitrary N, round-trip oracle)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Emitter — emit AS-Waksman m_cm(800)/e_cm(400) control ROMs + guard

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (the Beneš emit block, ~1478–1567, post-Step-4)

**Interfaces:**
- Consumes: `aswaksman_control`, `aswaksman_switch_count`, `aswaksman_apply` (Task 1); `complete_partial` (arbitrary m); `benes_group_matchings` (existing); `nhb`(=800), `neb`(=400), `nvb`, `gv`, `msg` (in scope).
- Produces: `localparam BP_ASW_MCM_N=nhb`, `BP_ASW_ECM_N=neb`, switch-count localparams; ROMs `BP_ROM_BENES_MCMWR` (now `aswaksman_switch_count(nhb)`-bit rows) and `BP_ROM_BENES_ECMRD` (per-port `aswaksman_switch_count(neb)` bits, port0-low/port1-high). Removes the old `BP_BENES_MCM_M/COLS` and `BP_BENES_ECM_M/COLS` Beneš-specific sizing where they only fed these two nets (keep any still used by the ECM read PIPE math — verify).

- [ ] **Step 1: Swap m_cm control synth to AS-Waksman**
Replace `let full_mcm = complete_partial(&dest_mcm, mcm_m); let ctrl_mcm = benes_control(&full_mcm);` with `complete_partial(&dest_mcm, nhb)` → `aswaksman_control`. Row width becomes `aswaksman_switch_count(nhb)`.

- [ ] **Step 2: Swap e_cm read control synth to AS-Waksman (per port)**
For each port: `complete_partial(dest, neb)` → `aswaksman_control(&invert(&full))` (read is the inverse perm, as today). Pack port0-low/port1-high with the new per-port width `aswaksman_switch_count(neb)`.

- [ ] **Step 3: Emit the new sizing localparams**
```rust
println!("localparam int BP_ASW_MCM_N   = {nhb};");
println!("localparam int BP_ASW_MCM_SW  = {};", aswaksman_switch_count(nhb));
println!("localparam int BP_ASW_ECM_N   = {neb};");
println!("localparam int BP_ASW_ECM_SW  = {};", aswaksman_switch_count(neb));
```
(Keep `BP_BENES_ECM_PORTS=2`.)

- [ ] **Step 4: Update the gen-time guard to `aswaksman_apply`**
Where the current guard runs `benes_apply(benes_control(&full), ident)` for m_cm and e_cm, switch to `aswaksman_apply(aswaksman_control(&full), ident)` and assert it reproduces `dest`'s matching bank-for-bank, both bankings. (The addr/readrow guard from Step 4 is untouched.)

- [ ] **Step 5: Build + run emitter both bankings; guard passes, ROMs resized**
```bash
cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 8 24  > /tmp/g824.svh
cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > /tmp/g1648.svh
grep -E "BP_ASW_MCM_SW|BP_ASW_ECM_SW|BP_ROM_BENES_MCMWR \[" /tmp/g1648.svh | head
```
Expected: no panic (guard passes) at both; `BP_ASW_MCM_SW` ≈ 6900–7200 and `BP_ASW_ECM_SW` ≈ 3100–3300 (whatever `aswaksman_switch_count` returns, both materially < 9727 / 4352); MCMWR row width narrows from `[9727:0]` to `[BP_ASW_MCM_SW-1:0]`.

- [ ] **Step 6: clippy + fmt + commit** (`cargo clippy -p aleph-qec --examples -- -D warnings`; commit message `[Q7-04] M9c Step 5b: emit AS-Waksman m_cm/e_cm control ROMs + guard`).

---

## Task 3: RTL fabric `hw/bp_asw.sv` + standalone TB

**Files:**
- Create: `hw/bp_asw.sv`, `hw/tb_bp_asw.cpp`
- Modify: `hw/Makefile` (add `bpasw` target mirroring `bpbenes`)

**Interfaces:**
- Produces modules: `bp_asw_mcm_wr #(N,W,PIPE)`, `bp_asw_ecm_read #(N,W,PIPE)` with ports `clk`, `din[N][W]`, `ctrl[ASW_SW(N)-1:0]`, `dout[N][W]` — same shape/timing contract as `bp_benes_*` (fresh (din,ctrl)/cycle, `dout@t+PIPE`). `ASW_SW(N)` = `aswaksman_switch_count` as an SV function.

- [ ] **Step 1: Implement `bp_asw_block` (arbitrary-size, depth-balanced)**
Recursive module: `n/2` input switches (last lane straight if odd), upper subnet `n.div_ceil(2)`, lower `n/2`, `n.div_ceil(2)-1` output switches, control read in the SAME recursion/counter order as `aswaksman_control`. **Depth-balance (spec risk):** pad the shallower sibling subnet with straight-through register stages so every lane crosses the same number of registered boundaries — reuse `bp_benes.sv`'s uniform-PIPE placement rule (`stage(c)=floor(c*PIPE/COLS_TOTAL)`) computed on the balanced column count. Cite the balancing in a comment.

- [ ] **Step 2: Write `tb_bp_asw.cpp` — independent C++ oracle**
Port `aswaksman_apply` to C++ independently (do NOT translate the SV — re-derive from the algorithm, as `tb_bp_benes.cpp` did for Beneš). Stream random (din,ctrl) at N=400 and N=800, assert `dout@t+PIPE == oracle`, 10000 cases each.

- [ ] **Step 3: Add `bpasw` Makefile target + run**
```bash
cd hw && make bpasw 2>&1 | tail -20
```
Expected: builds; both N=400 and N=800 report 10000/10000 match.

- [ ] **Step 4: Commit** (`[Q7-04] M9c Step 5c: bp_asw.sv arbitrary-size fabric + standalone TB`).

---

## Task 4: Wire AS-Waksman fabrics into the core; full co-sim 40/40

**Files:**
- Modify: `hw/bp_relay_banked_bram_m.sv`

- [ ] **Step 1: Baseline** — the current committed core is at 40/40, latency 4475/2810 (Step 4). (Do not re-measure live — Task 2 already changed the emitter; use these as the invariant.)

- [ ] **Step 2: Swap the m_cm write fabric**
`bp_benes_mcm_wr #(.N(BP_BENES_MCM_M=1024),...)` → `bp_asw_mcm_wr #(.N(BP_ASW_MCM_N=800), .W(1+BWC+MSG_BITS), .PIPE(BENES_PIPE_MCM))`. Change the `mcm_wr_din` padding loop bound and the `we_mcm/wa_mcm/wd_mcm` output loop to `NHB`(=800) domain; control ROM `benes_mcmwr_q` width follows `BP_ASW_MCM_SW`.

- [ ] **Step 3: Swap the two e_cm read fabrics**
`bp_benes_ecm_read #(.N(512),...)` ×2 → `bp_asw_ecm_read #(.N(BP_ASW_ECM_N=400),...)` ×2; din padding 512→400; control slices use `BP_ASW_ECM_SW` per port.

- [ ] **Step 4: Lint** `make bpbankedbramm-lint` — fix width/unused from the resize.

- [ ] **Step 5: Full co-sim gate**
```bash
cd hw && make bpbankedbramm 2>&1 | grep -iE "40/40|worst|== W|FAIL" | tee /tmp/asw_cosim.txt
```
Expected: **40/40 both bankings**, worst latency **4475 (8/24) / 2810 (16/48)** unchanged. If latency moves, the fabric PIPE/balancing is off — fix in Task 3's balancing, not here.

- [ ] **Step 6: Commit** (`[Q7-04] M9c Step 5d: wire AS-Waksman fabrics into core (40/40, latency identical)`).

---

## Task 5: OOC synth (LUT + BRAM) + perf-doc note

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md`

- [ ] **Step 1: Regenerate at-rest 16/48 header; push branch**
```bash
cd hw && cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > bb_gross_tanner.svh
cd .. && git add hw/bb_gross_tanner.svh && git commit -m "regen at-rest 16/48 header (AS-Waksman)" ; git push -u origin q7-04-m9c-aswaksman
```

- [ ] **Step 2: OOC synth on EPYC** (`root@195.154.249.85`, `/tools/Xilinx/Vivado/2024.2/settings64.sh`): scp `bp_relay_banked_bram_m.sv`, `bp_asw.sv`, `bp_benes.sv`, `check_minsum.sv`, `var_update.sv`, `bb_gross_tanner.svh` to `/data/kv260fit`, run `vivado -mode batch -source ooc_serial.tcl -tclargs 5.0 m9c_step5 bp_relay_banked_bram_m`, read `util_banked.rpt`.
Expected vs Step-4 (206,931 LUT / 164.5 BRAM tiles): **BRAM tiles materially down** (the point — target toward ≤144/100 %); LUT flat-to-slightly-down.

- [ ] **Step 3: Write `### M9c Step 5` note** — LUT + BRAM before/after, whether BRAM crossed under 100 % (half the two-constraint fit), the honest LUT-stays-over verdict, Fmax. Cite spec + plan.

- [ ] **Step 4: Commit** (`[Q7-04] M9c Step 5e: AS-Waksman synth result — BRAM <measured>`).

---

## Self-Review

**Spec coverage:** routing lib + `complete_partial` relax (T1) ✓ spec §1; emitter (T2) ✓ §2; RTL fabric + balancing for the odd-N PIPE risk (T3) ✓ §3 + risk section; core wiring (T4) ✓ §3; standalone TB (T3) ✓ §4; synth+BRAM verdict (T5) ✓ §Verification/goal. The odd-N depth-balancing risk is addressed in T3 Step 1 with the standalone probe (T3 Step 3) measuring it before core integration, per the spec's mitigation.

**Placeholder scan:** Task 1 Steps 4 gives algorithm structure + the round-trip test as the executable spec rather than verbatim final code — this is deliberate for the derive-the-recursion task (the test is complete and correct; the implementer iterates against it, TDD). All commands, test code, and interface signatures are concrete. No TBD/TODO.

**Type consistency:** `aswaksman_switch_count`/`_control`/`_apply` names consistent T1→T2→T3; `BP_ASW_MCM_N/SW`, `BP_ASW_ECM_N/SW` produced in T2, consumed in T4; fabric module names `bp_asw_mcm_wr`/`bp_asw_ecm_read` consistent T3→T4; N=800 (nhb)/N=400 (neb), switch counts 7201/3201 consistent throughout.
