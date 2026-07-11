# M8 — squeeze the KV260 (pipeline the banked core, rehabilitate 16/48) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `bp_relay_banked`'s KV260 latency from 32.8 µs worst / 2.5 µs early-exit mean to ~17–22 µs / ~1.5 µs by pipelining the measured critical path (L1: bank-output registers, L3: 3-stage `check_minsum`) and re-probing the LUT-bound 16/48 config (L4) at higher PS-grid FCLKs with impl-strategy escalation (L2) — bit-exact in VALUES throughout (cycle counts change and are re-recorded).

**Architecture:** Two pure re-timings: a 1-cycle register plane on every bank read output (+ the sbit/lam/gam/present launch context registered alongside), and a register plane inside `check_minsum`'s tournament tree after level 3, behind a `STAGES` parameter (default 2 — `bp_relay_unroll_pipe`/`bp_unroll_skeleton` untouched). FSM scatter lags move: CHK `pc−2 → pc−4` (read reg + 3-stage minsum), VAR `pc−2 → pc−3` (read reg + 2-stage var_update). Then the M7 probe/build/silicon pipeline re-runs with a fixed DSP counter.

**Tech Stack:** SystemVerilog (Verilator 5.050 Mac / 5.032 EPYC), Vivado 2024.2 on `openwebgui.splynx.com`, `FixedRelayBp` golden + `tb_bp_banked.cpp`/`tb_bp_axi_banked.cpp`, PYNQ runner on KV260 (`192.168.88.174`).

## Global Constraints

- **Track:** commits `[Q7-02] M8: …`; PR `Advances #322` (NOT Closes). Branch `q7-02-m8-kv260-squeeze` (created off origin/main; spec `cf91c99` on it). No worktrees.
- **Correctness bar (spec §5):** bit-exact in VALUES to the unchanged golden (`bp_circ_vectors.txt`, `circvectors 1 0.003 40 2024`); latency counts change and are recorded anew. Any corr_out/obs_flip/valid_flag diff = bug.
- **Expected new cycle counts (model ±5 %, exact numbers come from the TB):** per-iter = (GC+4) + (GV+3) + 2; totals ≈ **8/24 → ~3 875**, **12/36 → ~2 775**, **16/48 → ~2 220** (old: 3 570 / 2 460 / 1 905).
- **Verilator prefix:** `PATH="$HOME/.cargo/bin:$PATH" make -C hw <target>`. EPYC co-sim box `root@195.154.249.85` (verilator 5.032, add `-Wno-LATCH` there only — 5.032 false-positive). Vivado ops per M7: detached nohup + `</dev/null`, poll `^RESULT`, PGID-scoped kills, rsync staging (box git stale).
- **PS-grid FCLK candidates:** 125 (1500/12), 115.384 (/13), 107.142 (/14), 100 (/15); M7 shipped 75 (/20).
- **Header regen:** `cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 <W> <V> > hw/bb_gross_tanner.svh`; at-rest header stays (12,36) unless 16/48 wins the build (then commit the winner).

---

## File Structure

- `hw/check_minsum.sv` — **modify.** `STAGES` parameter (2 default / 3), register plane after tree level 3.
- `hw/tb_check_minsum.cpp` — **modify.** Latency-parametric wait; run both STAGES.
- `hw/bp_relay_banked.sv` — **modify.** L1 register plane + FSM lag/drain updates; instantiates `check_minsum #(.STAGES(3))`.
- `hw/Makefile` — **modify.** `checkminsum` runs both stages; no other target semantics change.
- `hw/syn/ooc_banked.tcl` — **modify.** DSP counted as `REF_NAME == DSP48E2`.
- `hw/syn/kv260_bp_circ_banked_bd.tcl` — **modify.** 5th tclarg `strategy` (default `Vivado Implementation Defaults`, escalation `Performance_Explore`).
- `docs/perf/qec-q7-fixed-bp.md` — **modify.** M7 correction note + M8 section.

---

### Task 1: `check_minsum` STAGES parameter + dual-stage TB

**Files:**
- Modify: `hw/check_minsum.sv`
- Modify: `hw/tb_check_minsum.cpp`
- Modify: `hw/Makefile` (target `checkminsum`)

**Interfaces:**
- Produces: `module check_minsum #(parameter int MW = 8, parameter int DEG = 25, parameter int STAGES = 2) (input logic clk, en, sbit, input logic signed [MW-1:0] m_in [DEG], input logic present [DEG], output logic signed [MW-1:0] e_out [DEG]);` — result valid **STAGES+... exactly STAGES+0? contract: e_out valid `STAGES` clocks after `en`** (STAGES=2 today: en → stage1 regs → stage2 regs = 2 clocks; STAGES=3 adds one mid-tree plane = 3 clocks).
- Consumers: `bp_relay_unroll_pipe.sv:112` and `bp_unroll_skeleton.sv:90` instantiate WITHOUT `.STAGES` → default 2, byte-identical behavior (regression-gated).

- [ ] **Step 1: Extend the TB.** In `hw/tb_check_minsum.cpp`, replace the hardcoded 2-cycle wait with `LATENCY` from a compile define (`#ifndef LATENCY / #define LATENCY 2 / #endif`), and print it in the PASS line: `PASS: 10000/10000 check_minsum (LATENCY=%d) outputs bit-identical...`. The C++ reference model is untouched (values don't depend on STAGES).

- [ ] **Step 2: Makefile — run both stages.** Rewrite the `checkminsum` target to build twice (model on the `bpunrollpipe` dual-build pattern):
```make
checkminsum:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph > hw/bb_gross_tanner.svh
	rm -rf obj_chkmin2 obj_chkmin3
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_chkmin2 --top-module check_minsum \
		-GSTAGES=2 -CFLAGS -DLATENCY=2 check_minsum.sv tb_check_minsum.cpp -o sim_checkminsum2
	./obj_chkmin2/sim_checkminsum2
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_chkmin3 --top-module check_minsum \
		-GSTAGES=3 -CFLAGS -DLATENCY=3 check_minsum.sv tb_check_minsum.cpp -o sim_checkminsum3
	./obj_chkmin3/sim_checkminsum3
```
(Check the existing target first: if it builds in a `_build` dir or passes vectors, keep that shape and add only the second build+run leg + the `-G/-D` pairs.)

- [ ] **Step 3: Run to see the STAGES=3 leg fail** (parameter doesn't exist yet).
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw checkminsum`
Expected: leg 1 passes (LATENCY=2), leg 2 fails at verilation (`STAGES` unknown parameter).

- [ ] **Step 4: Implement STAGES in `hw/check_minsum.sv`.** The tournament tree currently reduces all `NLVL = $clog2(NLEAF)` levels combinationally into stage-1 registers. Add:
```systemverilog
parameter int STAGES = 2   // 2 = M7-compatible; 3 = extra register plane after tree level SPLIT_LVL
localparam int SPLIT_LVL = 3;             // levels 1..3 before the plane, 4..NLVL after
```
Generate-split the reduction: compute `lvl[SPLIT_LVL]` nodes combinationally; if `STAGES == 3`, register that node array (+ register `sbit`/`m_in`/`present` pass-throughs one extra cycle so stage-2 sees aligned context); continue levels `SPLIT_LVL+1..NLVL` combinationally into the existing stage-1 registers. If `STAGES == 2`, wire straight through (no plane) — the generated logic must be IDENTICAL to today's (guard with `if (STAGES == 3) begin : gplane ... end else begin : gpass ... end`). Values are unchanged by construction (registers only); the neg XOR fold splits the same way (fold present-sign terms of leaves under each side of the plane consistently — simplest: register the partial XOR alongside the node plane).

- [ ] **Step 5: Run both legs green.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw checkminsum`
Expected: `PASS: 10000/10000 ... (LATENCY=2)` and `PASS: 10000/10000 ... (LATENCY=3)`.

- [ ] **Step 6: Regression — the 2-stage consumers are untouched.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpunrollpipe unrollskel`
Expected: NGROUP=4 + NGROUP=2 both `PASS: 40 ...` (identical latencies to M7: 728/484 cyc), skeleton lints clean.

- [ ] **Step 7: Commit.**
```bash
git add hw/check_minsum.sv hw/tb_check_minsum.cpp hw/Makefile
git commit -m "[Q7-02] M8: check_minsum STAGES param (2 default / 3) — mid-tree register plane, bit-exact both"
```

---

### Task 2: L1 bank-output registers + FSM lags in `bp_relay_banked`

**Files:**
- Modify: `hw/bp_relay_banked.sv`

**Interfaces:**
- Consumes: `check_minsum #(.STAGES(3))` (Task 1). `var_update` unchanged (2-stage).
- Produces: same module ports; NEW timing contract — CHK scatter lag `pc−4`, VAR scatter lag `pc−3`, phase lengths CHK = GC+4, VAR = GV+3 cycles.

- [ ] **Step 1: Register plane.** Register every bank read output and the launch context, one plane, all `always_ff @(posedge clk)` unconditional:
  - `qmcm_r[b] <= qmcm[b]`, `qa_ecm_r[b] <= qa_ecm[b]`, `qb_ecm_r[b] <= qb_ecm[b]`, `qmvm_r[b] <= qmvm[b]`;
  - the per-slot launch context currently consumed combinationally at `pc` — `sbit_i`, `present_i` masks, `lam_i`, `gam_i` (and the VAR-side present) — register alongside (i.e., the gather combs still index ROMs by `pc`, their outputs latch into `*_r`, and the submodule inputs read the `_r` plane). Bank read ADDRESSES stay driven by `pc` (unregistered).
  - Submodule `en`: delay one cycle — `en_chk_r <= (state == S_CHECK) && (pc < GC)` then `.en(en_chk_r)` (same for VAR). Add matching `en` fencing so the first registered operands (garbage at pc=0) are never consumed.
- [ ] **Step 2: FSM lag arithmetic.** In S_CHECK: scatter group `pc−4` (`if (pc >= 4)`), phase end at `pc == GC+3`; in S_VAR: scatter `pc−3` (`if (pc >= 3)`), end `pc == GV+2`; S_SATF unchanged (reads ehat flops directly — verify it does NOT read banks; if its parity uses the same gather ROMs on ehat only, no reg needed). S_INIT unchanged (write-only). Keep the SAT-overlap launch gating at `pc < GC` and its finalize at `pc == GC−1` (launch-side, unaffected by drains). The elaboration guards are unchanged (banking maps untouched).
- [ ] **Step 3: Lint.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpbanked-lint`
Expected: clean (-Wall).
- [ ] **Step 4: Co-sim 8/24 on the Mac** (manual leg of `bpbanked`, as in M7):
```bash
cd .. && cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
cd .. && cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 8 24 > hw/bb_gross_tanner.svh
cd hw && rm -rf obj_banked && verilator --cc --exe --build -j 4 -Wall --Mdir obj_banked --top-module bp_relay_banked \
  check_minsum.sv var_update.sv bp_relay_banked.sv tb_bp_banked.cpp -o sim_bpbanked
./obj_banked/sim_bpbanked bp_circ_vectors.txt
```
Expected: `PASS: 40 full decodes bit-identical...; worst latency = ~3 875` (model ±5 %; whatever the TB prints becomes the recorded number — investigate any >10 % deviation as a scheduling bug before accepting).
- [ ] **Step 5: AXI gate.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpaxibanked`
Expected: `PASS: 40 circuit-level decodes ... over wide AXI4-Lite; worst latency = ~2 775` (12/36 header).
- [ ] **Step 6: Commit.**
```bash
git add hw/bp_relay_banked.sv
git commit -m "[Q7-02] M8: bank-output register plane + STAGES=3 minsum — CHK lag pc-4, VAR pc-3, bit-exact values, <n> cyc at 8/24"
```

---

### Task 3: EPYC tri-config co-sim (closes the geometry gate)

**Files:** none new (EPYC runner script pattern from M7).

- [ ] **Step 1: Stage to EPYC** (`/root/m7cosim` was cleaned for CI disk — recreate):
```bash
ssh root@195.154.249.85 'mkdir -p /root/m8cosim'
for wv in "8 24" "12 36" "16 48"; do set -f; set -- $wv 2>/dev/null || true; done   # zsh: use explicit calls instead
# generate the three headers explicitly (zsh does not word-split — M7 lesson):
PATH="$HOME/.cargo/bin:$PATH" cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 8 24  > /tmp/bank_w8v24.svh
PATH="$HOME/.cargo/bin:$PATH" cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 12 36 > /tmp/bank_w12v36.svh
PATH="$HOME/.cargo/bin:$PATH" cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > /tmp/bank_w16v48.svh
rsync -az hw/check_minsum.sv hw/var_update.sv hw/bp_relay_banked.sv hw/tb_bp_banked.cpp hw/bp_circ_vectors.txt /tmp/bank_w*.svh root@195.154.249.85:/root/m8cosim/
```
- [ ] **Step 2: Run script** (same shape as M7's run.sh; verilator flags `-Wall -Wno-LATCH`, per-config build dir, `bank_$cfg.svh → bb_gross_tanner.svh`), detached + watcher on `COSIM_PASS|COSIM_FAIL|VERILATE_FAIL|ALL_CONFIGS_DONE`.
Expected: `COSIM_PASS` ×3; record the three worst-latency numbers (≈3 875 / 2 775 / 2 220).
- [ ] **Step 3: Ledger the numbers; delete /root/m8cosim afterwards** (CI disk hygiene — M7 lesson).

---

### Task 4: OOC probes with fixed DSP count (both configs)

**Files:**
- Modify: `hw/syn/ooc_banked.tcl`

- [ ] **Step 1: Fix the DSP counter.** Replace `set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]` with `set dsp [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]`. Commit message must name the 9× sub-primitive inflation.
- [ ] **Step 2: Stage + launch probes** for 12/36 and 16/48 (dirs `/root/kv260synth/m8_w12v36`, `m8_w16v48`; each gets check_minsum.sv var_update.sv bp_relay_banked.sv, its header as bb_gross_tanner.svh, syn/ooc_banked.tcl), `-tclargs 5.0 <label>`, detached, watcher on `^RESULT`.
- [ ] **Step 3: Record.** Expected shape: LUT ≈ M7 +2–4 k (the register planes), DSP 124/164 real, WNS materially better than −5.8 (target ≥ −3 @5 ns → Fmax ≥ ~115). If WNS is still ≤ −5: pull `timing_banked.rpt` critical path and STOP — report to the coordinator (the next lever is a design question, not a retry).
- [ ] **Step 4: Commit the tcl fix** (`[Q7-02] M8: ooc_banked.tcl — count DSP48E2 primitives (old DSP* matched sub-primitives 9x)`).

---

### Task 5: Board builds with strategy escalation

**Files:**
- Modify: `hw/syn/kv260_bp_circ_banked_bd.tcl`

- [ ] **Step 1: Strategy tclarg.** 5th arg `strategy` (default empty = Vivado defaults): when non-empty, `set_property strategy $strategy [get_runs impl_1]` before `launch_runs`. Keep `FLATTEN_HIERARCHY none` (load-bearing).
- [ ] **Step 2: Build ladder** (each build detached + watched; stop at the first TIMING_MET):
  1. 16/48 @ FCLK from Task-4 Fmax rounded DOWN to the PS grid (125/115/107/100), default strategy.
  2. same, `Performance_Explore`.
  3. FCLK one grid step down, default → explore.
  4. Fallback: 12/36, same ladder.
Header staging: the winning config's header as `bb_gross_tanner.svh` in `/root/kv260synth/hw`. Wide wrap + top + runner are unchanged from M7 (IDCODE `0x4250_0003`).
- [ ] **Step 3: Commit** (`[Q7-02] M8: board build — <cfg> TIMING_MET @ <f> MHz (<strategy>)`), and if 16/48 won, regen + commit the at-rest header at (16,48) in the same commit.

---

### Task 6: Silicon (both modes)

**Files:** none (M7 runner).

- [ ] **Step 1: Transfer** `.bit`/`.hwh` (rsync box→Mac, base64→KV260 per M7; vectors/runner already on the board — re-send `bp_circ_vectors.txt` anyway to be safe).
- [ ] **Step 2: Run** `bp_circ_kv260.py <bit> bp_circ_vectors.txt --clk <f>e6 --idcode 0x42500003`.
Expected: `40/40` both modes; full-schedule cycles == Task-3's number for the built config; worst-case µs = cyc/f (target band 17–22 µs); early-exit stats recorded.
- [ ] **Step 3: Ledger the numbers.**

---

### Task 7: Docs (M7 correction + M8 section) + PR + merge

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md`

- [ ] **Step 1: M7 correction note.** Insert immediately after the M7 OOC-sweep table: a short "**Correction (M8):**" paragraph — DSP column was 9× inflated (`DSP*` matched DSP48E2 sub-primitives); real 82/124/164 (6.6/10/13 %); "16/48 no fit (DSP)" was wrong — 16/48 is LUT-bound (90 %) and was rehabilitated in M8. Do NOT silently edit the M7 numbers in place — the correction must be visible.
- [ ] **Step 2: M8 section.** Levers (L1–L4 with the lag arithmetic), new cycle counts (all three configs), OOC + board ladder results (incl. failed rungs — honest), silicon numbers both modes, updated ladder (6.72 ms → 32.8 µs (M7) → <X> µs (M8)), remaining levers (deeper var pipeline, bigger part).
- [ ] **Step 3: Final whole-branch review** (SDD flow), fold housekeeping, then:
```bash
git push -u origin q7-02-m8-kv260-squeeze
gh pr create --base main --title "[Q7-02] M8: KV260 squeeze — <X> us worst / <Y> us early-exit (<n>x M7)" --body "…Advances #322…"
```
Merge on green (real gates: macos×2 + linux stable + clippy + rustfmt + python; beta flake → rerun; EPYC disk watch — clean `_work/aleph/aleph/target` if 5s-fails reappear). `gh pr merge --squash --delete-branch`.

---

## Self-Review

**Spec coverage:** §1 corrections → Tasks 4 (tcl) + 7 (doc note); §3 L1/L3 → Tasks 1–2; L4 → Tasks 3–5 (16/48 through every gate); L2 → Task 5; §5 gates → Tasks 1 (dual-stage TB + unroll regression), 2 (lint, 8/24, AXI), 3 (tri-config EPYC); §6 deliverables → all; §7 fallback → Task 5 ladder rung 4. Covered.
**Placeholders:** `<n>/<X>/<f>/<cfg>` are measurement outputs (Tasks 2–6), by design. The Task-1 Step-2 Makefile block says "check the existing target first" — that is an instruction to preserve repo shape, not a gap.
**Type consistency:** `STAGES` name used in Tasks 1→2; lag constants (CHK pc−4 / GC+4, VAR pc−3 / GV+3) match the Global-Constraints cycle model ((GC+4)+(GV+3)+2 per iter); TB define `LATENCY` paired with `-GSTAGES` in the same step.
**Granularity note:** Task 2 is the delicate one (FSM lags); its Step-4 co-sim is the net — expect edit-run iterations inside that step, as in M7.
