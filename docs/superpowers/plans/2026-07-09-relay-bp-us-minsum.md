# µs relay-BP: synthesis-friendly full-unroll min-sum — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A register-resident, hierarchically-modular, shallow-pipelined full-unroll relay-BP core that *synthesizes* at circuit scale and decodes the gross BB code in ~1–3 µs on KV260 silicon.

**Architecture:** Replace `bp_relay_fast`'s flat inline unroll with stamped submodules — `check_minsum` ×144 + `var_update` ×864, wired at compile-time-constant edge indices, each internally 2-stage pipelined — so Vivado synthesizes one small module and stamps copies (`-flatten_hierarchy none`) instead of choking on a single ~300k-cell combinational cloud. A **Step-0 OOC fit gate** decides go/no-go before the full core is built.

**Tech Stack:** SystemVerilog (Verilator 5.050 on the Mac for co-sim, Vivado 2024.2 on `openwebgui.splynx.com` for synth), the existing `FixedRelayBp` Rust golden + `tb_bp_relay.cpp` harness, PYNQ on KV260.

## Global Constraints

- **Track / issue:** all commits `[Q7-02] M7: …`; PR body `Advances #322` (umbrella issue — NOT `Closes`). Trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Branch:** `q7-02-m7-us-minsum` (already created off `origin/main`; the design spec `b9568ae` is on it). No worktrees.
- **Verilator-first (CLAUDE §Testing / handoff §4):** every RTL change verified in Verilator on the Mac before Vivado. `cargo`/`verilator` prefix: `PATH="$HOME/.cargo/bin:$PATH" make -C hw <target>`.
- **Graph constants:** `BP_N=864 BP_C=144 BP_E=2952 BP_OBS=12 BP_CHK_DEG=25 BP_VAR_DEG=? MSG_BITS=? FRAC_BITS=3 BP_LEGS=6 BP_ITERS=10` — all from the generated `hw/bb_gross_tanner.svh` (circuit header, gitignored; regenerate with `cargo run -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 > hw/bb_gross_tanner.svh`).
- **Correctness bar:** bit-exact to `FixedRelayBp` wherever the change is pure re-timing; the Monte-Carlo LER oracle is used **only** if Step-0 forces narrower `MSG_BITS`.
- **Golden vectors:** `hw/bp_circ_vectors.txt` (gitignored; `qec_q7_bp_graph -- circvectors 1 0.003 40 2024`). 40 shots, all converge.
- **Synth ops (handoff §3):** launch Vivado fully detached (`nohup bash -c '… >log 2>&1' >/dev/null 2>&1 &`); poll the `^RESULT`/`ALL_*_DONE` line, never `pgrep`; scoped-kill your own run by PGID (`kill -9 -<pgid>`), never `pkill -x vivado` (shared box). Stage RTL via `rsync` to `/root/kv260synth/hw` (box git is stale).

---

## File Structure

- `hw/check_minsum.sv` — **new.** One check's min-sum (min1/min2/argmin → `e_cv = ±(exmin − exmin>>3)`), 2-stage pipelined, degree `BP_CHK_DEG` + present-mask. One responsibility: the check-update kernel.
- `hw/var_update.sv` — **new.** One variable's update (`total = λ + Σe_cv`; per-edge constant-γ blend + saturate; `ehat` bit), 2-stage pipelined, degree `BP_VAR_DEG` + present-mask.
- `hw/bp_relay_unroll_pipe.sv` — **new.** Top core: message flop arrays + 144 `check_minsum` + 864 `var_update` stamped via `generate` at constant edge indices + the 6×10 FSM with SAT-overlap. Same top ports as `bp_relay_fast`.
- `hw/tb_bp_unroll_pipe.cpp` — **new** (or reuse `tb_bp_relay.cpp` via a `-DUNROLLPIPE` path). Drives the full core over `bp_circ_vectors.txt`, asserts bit-exact to golden + reports latency.
- `hw/syn/ooc_fit_gate.tcl` — **new.** Step-0 OOC synth of the modular slice at `xck26`, `-flatten_hierarchy none`, prints `RESULT LUT/FF/DSP/BRAM + area-opt completed`.
- `hw/Makefile` — **modify.** Add `bpunrollpipe` target (regen header+vectors, build+run Verilator).
- `hw/syn/kv260_bp_circ_bd.tcl` — **reuse** (M6); a Task-6 variant swaps the core into the wide wrap.
- `docs/perf/qec-q7-fixed-bp.md` — **modify.** M7 section.

---

### Task 1: `check_minsum` submodule + standalone bit-exact test

**Files:**
- Create: `hw/check_minsum.sv`
- Create/extend test: `hw/tb_check_minsum.cpp`
- Modify: `hw/Makefile` (target `checkminsum`)

**Interfaces:**
- Produces: `module check_minsum #(parameter int MW=8, parameter int DEG=25) (input logic clk, en, sbit, input logic signed [MW-1:0] m_in [DEG], input logic present [DEG], output logic signed [MW-1:0] e_out [DEG]);` — 2-cycle latency (result valid 2 clocks after `en`).

- [ ] **Step 1: Write the failing standalone test.** In `hw/tb_check_minsum.cpp`, generate random `m_in[DEG]`/`present[]`/`sbit`, drive one `en` pulse, wait 2 clocks, and compare `e_out` against a C++ reference that reimplements exactly the S_CHECK inner loop of `hw/bp_relay_fast.sv:121-145` (min1/min2/argmin, `excl`, `exmin`, `mag = exmin - (exmin>>3)`, sign). Assert bit-identical over 10 000 random cases.

- [ ] **Step 2: Run, verify it fails** (module absent).
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw checkminsum`
Expected: build error `Cannot find file containing module: check_minsum`.

- [ ] **Step 3: Implement `hw/check_minsum.sv`.** Stage-1 (`always_ff @(posedge clk) if(en)`): compute+register `min1,min2,argmin,neg` over the `DEG` slots gated by `present[k]` (copy the reduction from `bp_relay_fast.sv:125-134`). Stage-2 (`always_ff`): for each slot, `excl=(m<0)?~neg:neg; exmin=(k==argmin)?min2:min1; if(exmin==INF)exmin=0; mag=exmin-(exmin>>3); e_out[k]<= excl? -mag : mag;` (register `m_in`/`present` into stage-2 alongside). `localparam MW-wide INF='1;`.

- [ ] **Step 4: Run, verify pass.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw checkminsum`
Expected: `PASS: 10000/10000 check_minsum outputs bit-identical to bp_relay_fast reference`.

- [ ] **Step 5: Commit.**
```bash
git add hw/check_minsum.sv hw/tb_check_minsum.cpp hw/Makefile
git commit -m "[Q7-02] M7: check_minsum submodule (2-stage, bit-exact to bp_relay_fast S_CHECK)"
```

---

### Task 2: `var_update` submodule + standalone bit-exact test

**Files:**
- Create: `hw/var_update.sv`
- Create: `hw/tb_var_update.cpp`
- Modify: `hw/Makefile` (target `varupdate`)

**Interfaces:**
- Produces: `module var_update #(parameter int MW=8, parameter int WACC=16, parameter int FRAC=3, parameter int DEG=?, parameter int MAXMAG=?) (input logic clk, en, input logic signed [MW-1:0] lam, input logic signed [MW-1:0] gam, input logic signed [MW-1:0] e_in [DEG], input logic present [DEG], output logic signed [MW-1:0] m_out [DEG], output logic ehat_bit);` — 2-cycle latency. `gam` is the per-var γ for the current leg (top selects it from `BP_GAMMA` by `leg`).

- [ ] **Step 1: Write the failing test.** `hw/tb_var_update.cpp`: random `e_in`/`lam`/`gam`/`present`, compare `m_out`/`ehat_bit` to a C++ reference reimplementing `bp_relay_fast.sv:157-182` exactly (`total=λ+Σe_cv`; `newbit=total[WACC-1]`; per edge `computed=total-ev; num=omg*computed+g*old; blend=num>>>FRAC; clamp ±MAXMAG`). 10 000 cases, bit-identical.

- [ ] **Step 2: Run, verify fail.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw varupdate`
Expected: `Cannot find file containing module: var_update`.

- [ ] **Step 3: Implement `hw/var_update.sv`.** Stage-1: register `total = signed'(WACC'(lam)) + Σ_{present} WACC'(e_in[k])` and `ehat_bit_s1 = total[WACC-1]`; register `e_in`,`present`,`lam`,`gam`. Stage-2: `omg = (1<<FRAC) - gam`; per slot `computed=total-WACC'(e_in[k]); num=omg*computed + gam*WACC'(...old...)` — **note:** the "old" `m_vc` is `blend`'s feedback; for the standalone module pass `old` in as part of `e_in`? No — re-read: in `bp_relay_fast` `old = m_vc[e]` (the message being updated). So `var_update` also needs `m_in[DEG]` (the current `m_vc` of its edges). **Amend interface:** add `input logic signed [MW-1:0] m_in [DEG]`. Compute `num=omg*computed+gam*WACC'(m_in[k])`, `blend=num>>>FRAC`, clamp, `m_out[k]<=blend[MW-1:0]`.

- [ ] **Step 4: Run, verify pass.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw varupdate`
Expected: `PASS: 10000/10000 var_update outputs bit-identical to bp_relay_fast reference`.

- [ ] **Step 5: Commit.**
```bash
git add hw/var_update.sv hw/tb_var_update.cpp hw/Makefile
git commit -m "[Q7-02] M7: var_update submodule (2-stage, bit-exact to bp_relay_fast S_VAR)"
```

---

### Task 3: Step-0 OOC fit gate — GO/NO-GO (the de-risk)

**Files:**
- Create: `hw/bp_unroll_skeleton.sv` (all 144 + 864 instances constant-wired, trivial passthrough FSM — enough for OOC to place real logic)
- Create: `hw/syn/ooc_fit_gate.tcl`

**Interfaces:**
- Consumes: `check_minsum`, `var_update` (Tasks 1–2), `bb_gross_tanner.svh` constants.

- [ ] **Step 1: Write `hw/bp_unroll_skeleton.sv`.** `generate` 144 `check_minsum` bound via `BP_CHECK_OFF`/`BP_CHECK_EDGES` constant indices into `m_vc[BP_E]`, and 864 `var_update` via `BP_VAR_OFF` into `e_cv[BP_E]`; drive all `en` from one input, expose a reduced OR of outputs as a single output pin (so nothing is optimized away). `(* dont_touch *)` on `m_vc`/`e_cv`.

- [ ] **Step 2: Write `hw/syn/ooc_fit_gate.tcl`** (model on `hw/syn/ooc.tcl`, M6): `read_verilog -sv` the 3 files, `synth_design -top bp_unroll_skeleton -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy none -include_dirs [pwd]`, then `report_utilization`, FF count, and `puts "RESULT LUT=.. FF=.. DSP=.. RAMB=.. area_opt=COMPLETED"`.

- [ ] **Step 3: Verilator-elaborate the skeleton** (catch wiring errors cheaply before Vivado).
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw unrollskel` (a lint/elab-only target).
Expected: elaborates clean (no `UNDRIVEN`/index errors).

- [ ] **Step 4: Stage + OOC synth on openwebgui.**
```bash
rsync -az hw/check_minsum.sv hw/var_update.sv hw/bp_unroll_skeleton.sv hw/bb_circuit_tanner.svh hw/syn/ooc_fit_gate.tcl root@openwebgui.splynx.com:/root/kv260synth/hw/  # + copy circuit header as bb_gross_tanner.svh
ssh root@openwebgui.splynx.com 'cd /root/kv260synth/hw && source /tools/Xilinx/Vivado/2024.2/settings64.sh && nohup bash -c "vivado -mode batch -source syn/ooc_fit_gate.tcl > /root/kv260synth/fitgate.log 2>&1" >/dev/null 2>&1 & echo LAUNCHED'
```
Then poll `^RESULT` in `fitgate.log`.

- [ ] **Step 5: Decision + record.** Read `RESULT`. **GO** if area-opt COMPLETED, LUT < ~105k (90%), FF plausible, DSP ≤ 1248. **Marginal** → narrow `MSG_BITS` (regen header with a narrower fixed-point; requires the LER oracle in Task 5) and re-run. **NO-GO** (stalls/OOMs/overflows even narrowed) → stop here, write the honest finding into `docs/perf/qec-q7-fixed-bp.md`, and fall back to spec §7 (banked-BRAM ~10 µs, separate plan). Commit the gate artifacts + a one-paragraph verdict either way.
```bash
git add hw/bp_unroll_skeleton.sv hw/syn/ooc_fit_gate.tcl
git commit -m "[Q7-02] M7: Step-0 OOC fit gate — <GO|NO-GO>, LUT=<n> (<pct>%), area-opt <completed|stalled>"
```

> **Tasks 4–8 run only on GO.** On NO-GO the milestone ends at the documented finding + fallback plan.

---

### Task 4: `bp_relay_unroll_pipe.sv` — full core FSM

**Files:**
- Create: `hw/bp_relay_unroll_pipe.sv`

**Interfaces:**
- Consumes: `check_minsum`, `var_update`.
- Produces: `module bp_relay_unroll_pipe (clk, rst_n, in_valid, syndrome_in[BP_C], busy, out_valid, corr_out[BP_N], obs_flip[BP_OBS], valid_flag, latency_cycles[15:0])` — identical ports to `bp_relay_fast` (drop-in for `bp_axi_wrap_wide`; note `latency_cycles` is 16-bit there, wide-wrap zero-extends).

- [ ] **Step 1: Write the core.** Message flop arrays `m_vc/e_cv/ehat/s_reg` (`(* dont_touch *)`). `generate` 144 `check_minsum` (constant-wired) and 864 `var_update`; the per-var `gam` input is muxed from `BP_GAMMA[leg*BP_N+v]` by the runtime `leg` (6-way constant mux — cheap). FSM `S_IDLE→S_CHECK→S_VAR→…→S_SATF→S_EMIT→S_DONE` mirroring `bp_relay_fast`, but each phase asserts the submodules' `en` and waits **2 cycles** for the pipelined result before latching it back into the message arrays and advancing `iter`/`leg`. Keep the SAT-overlap + `sat_pending` + `run_sat` logic from `bp_relay_fast.sv:65-87,147,197`. Synchronous reset (Synth 8-7137).

- [ ] **Step 2: Verilator-elaborate.**
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw unrollpipe-lint`
Expected: elaborates clean.

- [ ] **Step 3: Commit.**
```bash
git add hw/bp_relay_unroll_pipe.sv hw/Makefile
git commit -m "[Q7-02] M7: bp_relay_unroll_pipe full core (stamped submodules + 6x10 FSM)"
```

---

### Task 5: Full-core co-sim vs golden + latency

**Files:**
- Create: `hw/tb_bp_unroll_pipe.cpp` (model on `hw/tb_bp_relay.cpp`)
- Modify: `hw/Makefile` (target `bpunrollpipe`)

**Interfaces:**
- Consumes: `bp_relay_unroll_pipe`, `bp_circ_vectors.txt`.

- [ ] **Step 1: Write the co-sim.** Regen header+vectors, then for each of the 40 vectors: pulse `in_valid` with `syndrome_in`, run until `out_valid`, compare `corr_out`/`obs_flip`/`valid_flag` to golden. Assert 40/40. Print latency distribution (cycles). Add a pipeline-flush invariance line (back-to-back decodes give identical results).

- [ ] **Step 2: Run, verify fail** first (before the core is correct — expect mismatches while wiring is debugged).
Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpunrollpipe`
Expected: initially FAIL; iterate on Task-4 core until →

- [ ] **Step 3: Iterate to PASS.**
Expected: `PASS: 40 full decodes bit-identical to the fixed-point golden; worst latency = <~250> cycles`.
- **If Step-0 narrowed `MSG_BITS`:** bit-exact will not hold — instead add `hw/../ Monte-Carlo LER` (decode ~10 000 circuit-noise shots through the RTL vs `FixedRelayBp`, assert LER within 5σ via `aleph_oracle::assert_distribution_close`) and gate on that.

- [ ] **Step 4: Commit.**
```bash
git add hw/tb_bp_unroll_pipe.cpp hw/Makefile
git commit -m "[Q7-02] M7: full-core co-sim — 40/40 bit-exact, worst <n> cycles"
```

---

### Task 6: KV260 board build with the new core

**Files:**
- Create: `hw/syn/kv260_bp_circ_upipe_bd.tcl` (copy `kv260_bp_circ_bd.tcl`, add `bp_relay_unroll_pipe.sv`, and either (a) a wide-wrap variant instantiating the new core, or (b) confirm the new core's ports let `bp_axi_wrap_wide` bind it directly — it has no `early_exit`; add a tie-off or a wrap variant).

- [ ] **Step 1: Resolve the wrap↔core port delta.** `bp_axi_wrap_wide` drives `.early_exit(early_mode)` and reads a 32-bit `latency_cycles`; the new core has no `early_exit` and 16-bit `latency_cycles`. Either add an (ignored) `early_exit` input + widen `latency_cycles` to 32 in the new core for a clean drop-in, **or** make a `bp_axi_wrap_wide_upipe.sv`. Prefer widening the core's port for zero wrap change. Re-run Task-5 co-sim after the port tweak.

- [ ] **Step 2: Full board build on openwebgui** (reuse the M6 flow; FCLK 100 MHz first, then push toward OOC Fmax).
```bash
rsync -az hw/*.sv root@openwebgui.splynx.com:/root/kv260synth/hw/
ssh root@openwebgui.splynx.com 'cd /root/kv260synth/hw && source …/settings64.sh && nohup bash -c "vivado -mode batch -source syn/kv260_bp_circ_upipe_bd.tcl -tclargs /root/kv260synth/build/upipe /root/kv260synth/out_upipe 100 > /root/kv260synth/upipe_build.log 2>&1" >/dev/null 2>&1 & echo LAUNCHED'
```
Poll `^RESULT`. Expected: `TIMING_MET`, `.bit`+`.hwh` produced.

- [ ] **Step 3: Commit.**
```bash
git add hw/syn/kv260_bp_circ_upipe_bd.tcl hw/bp_relay_unroll_pipe.sv
git commit -m "[Q7-02] M7: KV260 board build for bp_relay_unroll_pipe (TIMING_MET @ <f> MHz)"
```

---

### Task 7: On-silicon decode + latency

**Files:** none (reuse `hw/sw/bp_circ_kv260.py`).

- [ ] **Step 1: Transfer overlay to KV260** (base64-stream — scp is flaky on this link; §M6): `.bit`/`.hwh` + `bp_circ_vectors.txt` + `bp_circ_kv260.py`/`bp_circ_pynq.py` into `/tmp`.

- [ ] **Step 2: Run.**
```bash
ssh root@192.168.88.174 'cd /tmp && XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 bp_circ_kv260.py bp_kv260_circ_upipe.bit bp_circ_vectors.txt --clk 100e6'
```
Expected: `IDCODE ok (0x42500002)`, `40/40 … match golden`, `latency cycles ~250 → ~2.5 µs`. (Set `--clk` to the built FCLK.)

- [ ] **Step 3: Record the silicon numbers** (full-schedule µs + early-exit avg) for the doc.

---

### Task 8: Document M7 + PR

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md` (append `# Q7-02 M7 —` section)

- [ ] **Step 1: Write the M7 doc section.** Fit (Step-0 LUT/DSP), Fmax, silicon latency (both modes), the µs result (or the honest NO-GO + fallback), and the ladder update (6.72 ms M6 → <µs> M7).

- [ ] **Step 2: Commit + push + PR.**
```bash
git add docs/perf/qec-q7-fixed-bp.md
git commit -m "[Q7-02] M7: docs — µs relay-BP on KV260 (<n> µs), fit + silicon"
git push -u origin q7-02-m7-us-minsum
gh pr create --base main --title "[Q7-02] M7: µs-class circuit-level relay-BP on KV260 (<n> µs)" --body "…Advances #322"
```

- [ ] **Step 3: Merge on green** (real gates: macos + linux-stable + clippy + rustfmt + python; beta disk-full is a flake → rerun). `gh pr merge --squash --delete-branch`.

---

## Self-Review

**Spec coverage:** §3 architecture → Tasks 1,2,4. §4 fit gate → Task 3 (go/no-go, gates 4–8). §5 correctness (bit-exact + LER lever) → Tasks 1,2,5. §5 on-silicon → Task 7. §6 latency/fit → Tasks 3,5,7. §7 fallback B → Task 3 NO-GO branch. §8 deliverables → all. Covered.

**Placeholder scan:** the `BP_VAR_DEG`/`MSG_BITS`/`MAXMAG` `=?` marks are deliberate — they are read from the generated header at build time; each is resolved by the `bb_gross_tanner.svh` `localparam`, not left for the engineer to invent. No behavioral TODOs.

**Type consistency:** `check_minsum(clk,en,sbit,m_in[DEG],present[DEG],e_out[DEG])` and `var_update(clk,en,lam,gam,e_in[DEG],present[DEG],m_in[DEG],m_out[DEG],ehat_bit)` — the `m_in` addition to `var_update` (Task 2 Step 3) is reflected in the Task-4 wiring (old-message feedback). Core ports match `bp_relay_fast` (Task 4) and the wrap port delta is resolved in Task 6 Step 1.

**Note on granularity:** Tasks 4–5 are RTL-heavy and inherently iterative (timing/wiring debug) — the "write core / elaborate / co-sim to green" cycle is the test loop; expect multiple edit-run passes within Step 3 of Task 5, which is normal for the FSM assembly and not a plan gap.

---

## AMENDMENT 2026-07-09 — fit-gate pivot (full-unroll → modular partial-unroll)

**Task 3 (Step-0 fit gate) result:** the modular full-unroll skeleton (144 `check_minsum` + 864
`var_update`, `-flatten_hierarchy none`) **synthesized in ~3 min / 4.7 GB peak** — confirming
modularization defeats the flat-cloud area-opt wall (the core M7 hypothesis). **But it does NOT fit:**
CLB LUT **452 802 = 386 %** of the KV260's 117 120 (CARRY8 276 %; FF 66 % ok; DSP/BRAM 0). Full spatial
unroll is ~3.9× too big → true 1–3 µs at full unroll is physically impossible on this part.

**Decision (user): push to ~3 µs via MODULAR PARTIAL-UNROLL + narrow fixed-point.** Process a fraction
`1/G` of checks+vars per cycle through stamped groups (modularization also tames the mux that stalled
`partial_fast`), and narrow the fixed-point (`FixedRelayBp` is `(msg_bits,frac_bits)`-parameterizable →
regenerate a matching golden, so the RTL stays **bit-exact at the re-derived width**; the width's LER is
sanity-checked separately vs Aer/8-bit). Fit math: `453k × (1/G) × (width/8) + mux`. Target `G≈4`,
`msg_bits≈5–6` → ~70–90k LUT (fits), ~4–5× the full-unroll cycle count; push FCLK ~150–200 MHz toward 3 µs.

**Revised tasks (supersede Tasks 4–5 above):**

### Task 4′: `bp_relay_unroll_pipe.sv` — modular PARTIAL-unroll core, param `(G, MW)`
- Parameters: `NGROUP` (checks/vars processed per cycle = `BP_C/NGROUP` etc.), `MW`/`WACC`/`FRAC` from header.
- `generate` `CHK_UNROLL = ceil(BP_C/NGROUP)` `check_minsum` slots + `VAR_UNROLL = ceil(BP_N/NGROUP)`
  `var_update` slots. Each slot, for the active group `grp`, gathers its check/var's messages at
  **compile-time-constant** edge indices (the `bp_relay_fast`/`partial_fast` gather idiom — mux the operands
  by `grp == g`, never runtime-index the arrays), feeds the submodule, scatters the result back.
- FSM: per iteration, sweep `grp = 0..NGROUP-1` through S_CHECK then S_VAR (2-stage submodule pipeline;
  overlap groups where the reg→reg paths are independent). SAT-overlap as in `bp_relay_fast`. 6×10 schedule.
- Same top ports as `bp_relay_fast` (drop-in). Synchronous reset.
- **Fit-tune loop (in-session, openwebgui):** OOC-synth the core at `(G,MW)`; if LUT > ~105k, raise `G`
  or narrow `MW`; if it fits with margin, lower `G` (fewer cycles). Land the smallest-cycle config that
  fits ≤ ~105k LUT and closes timing at the target FCLK.

### Task 5′: bit-exact co-sim at the re-derived precision
- Add `msg_bits`/`frac_bits` CLI args to `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (emitter currently
  hardcodes `MSG_BITS=8,FRAC_BITS=3` at lines 21-22; `FixedRelayBp::with_budget` already takes them) so
  `circgraph`/`circvectors` emit at the chosen width. Regenerate header + `bp_circ_vectors.txt` at the tuned
  width. Co-sim `bp_relay_unroll_pipe` vs that golden → 40/40 bit-exact + latency. **Plus** a one-time LER
  sanity: confirm the chosen width's `FixedRelayBp` LER is within band vs the 8-bit / Aer reference
  (reuse the existing message-width sweep in `crates/aleph-qec` if present, else a short Monte-Carlo).

---

## AMENDMENT 2 (2026-07-10) — banked-store pivot (β-split check-major LUTRAM banking)

**Supersedes Tasks 4′/5′ and re-scopes Tasks 6–8.** Spec: AMENDMENT 2 of
`docs/superpowers/specs/2026-07-09-relay-bp-us-minsum-design.md`. The partial-unroll core
(`bp_relay_unroll_pipe`, Tasks 4′) is kept on the branch as the FSM/correctness reference but is NOT the
ship vehicle (area-opt stalls on its gather muxes). The ship vehicle is `bp_relay_banked.sv`.

**Architecture recap (see spec A2.1–A2.2).** Emitter solves offline: check→(group g, slot j), var→(group
h, slot i), β(e)∈{0,1}; banks are `m_cm[(j,k,β)]` (rows = g), `e_cm[(j,k)]` (rows = g), `m_vm[(i,d)]`
(rows = h), all LUTRAM. CHK phase reads m_cm row `pc` broadcast through per-lane 2:1 β-muxes and writes
e_cm row `pc−2` hardwired; VAR phase reads e_cm via ROM-driven ≤GV:1 muxes + m_vm mux-free, writes m_cm
(≤1 write/half-bank/cycle, ≤GC:1 src mux) + m_vm mux-free. Verified feasible at (W,V) =
(8,24)/(12,36)/(16,48) → ~3 946/2 836/2 281 cyc/decode.

**Key numbers:** `GC = ⌈BP_C/W⌉`, `GV = ⌈BP_N/V⌉`. At (8,24): GC=18, GV=36.

---

### Task 9: Emitter — offline banking solve + header tables

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (`emit_circ_graph` + new `solve_banking`)

**Interfaces:**
- Consumes: the existing in-memory CSR (`var_off`, `check_off`, `check_edges` vectors already built by
  `emit_circ_graph`).
- Produces (new header tables, appended after the existing ones — existing lines stay byte-identical):
  `BP_BANK_W`, `BP_BANK_V`, `BP_GC`, `BP_GV` (ints); `BP_CHK_AT [BP_GC*BP_BANK_W]` (check id at flat
  `g*W+j`, −1 = empty); `BP_VAR_AT [BP_GV*BP_BANK_V]` (var id at flat `h*V+i`, −1 = empty);
  `BP_EDGE_CHK [BP_E]`, `BP_EDGE_POS [BP_E]` (edge → its check, its in-check logical position);
  `BP_EDGE_BETA [BP_E]` (0/1).
- CLI: `circgraph <rounds> <p> [bankW] [bankV]` — `args.get(4)/get(5)`, defaults 8/24.

- [ ] **Step 1: Implement `solve_banking(w, v, ...) -> Banking`.** Deterministic (xorshift64 seeded 7,
  no new crates). Port of the verified Python (`scratchpad/nopi_grouping.py` logic):
  1. *Slot assignment:* iterate checks by descending degree; for each, pick slot `j` (capacity GC)
     minimizing the count of `(k, var)` pairs already seen in `j` (tie: lower occupancy, then rng).
  2. *Var grouping (cap 2):* bank of edge = `(slot_of_check, pos_in_check)`. Iterate vars by descending
     degree; place into the emptiest group `h` (capacity V) where every bank stays ≤2 counting the var's
     edges; if none, eviction-repair loop (random group, evict a conflicting var, re-queue; ≤200 000
     iterations, panic past that with a "banking solve failed — see spec A2.2 König fallback" message).
  3. *β assignment:* per (group h, bank) pair with 2 edges → β = 0 and 1 by edge-id order; singletons β=0.
  4. *Exact verify (always on):* assert per check group all `(j,k)` distinct; per var group all
     `(j,k,β)` half-banks distinct AND per `(j,k)` ≤2; every check/var appears exactly once in
     `BP_CHK_AT`/`BP_VAR_AT`. Panic on any violation.

- [ ] **Step 2: Emit the new tables in `emit_circ_graph`** after the existing prints, same
  `localparam int NAME [SIZE] = '{...};` style. Run and eyeball:
  `cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 8 24 | tail -8`
  Expected: the six new tables present; solve verify passed (no panic).

- [ ] **Step 3: Byte-identity guard for the old tables.** BEFORE committing: regen with the
  pre-change emitter, then with the new one, and require additions only:
  ```bash
  git stash                                                              # park the emitter edit
  cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 > /tmp/hdr_before.svh
  git stash pop
  cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 8 24 > hw/bb_gross_tanner.svh
  diff /tmp/hdr_before.svh hw/bb_gross_tanner.svh | grep -c '^<'         # expect 0 (nothing removed/changed)
  ```

- [ ] **Step 4: Feasibility sweep** (all three probe configs solve):
  `for wv in "8 24" "12 36" "16 48"; do cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 $wv > /dev/null && echo "OK $wv"; done`
  Expected: `OK` ×3.

- [ ] **Step 5: `cargo fmt` + clippy the touched crate, then commit.**
  ```bash
  cargo fmt && cargo clippy -p aleph-qec --all-targets -- -D warnings
  git add crates/aleph-qec/examples/qec_q7_bp_graph.rs
  git commit -m "[Q7-02] M7: emitter banking solve (slot assign + cap-2 var grouping + beta split)"
  ```

---

### Task 10: `hw/bp_relay_banked.sv` — the banked core

**Files:**
- Create: `hw/bp_relay_banked.sv`
- Modify: `hw/Makefile` (target `bpbanked-lint`)

**Interfaces:**
- Consumes: `check_minsum #(MW, DEG=BP_CHK_DEG)` and `var_update #(MW, WACC=16, FRAC, DEG=BP_VAR_DEG,
  MAXMAG)` (Tasks 1–2, unchanged); header tables from Task 9.
- Produces: `module bp_relay_banked (input logic clk, rst_n, in_valid, early_exit, input logic
  syndrome_in [BP_C], output logic busy, out_valid, output logic corr_out [BP_N], output logic
  [BP_OBS-1:0] obs_flip, output logic valid_flag, output logic [31:0] latency_cycles);` — **drop-in for
  `bp_axi_wrap_wide`'s instance contract** (same ports as `bp_relay_bram_dp`). W/V/GC/GV come from the
  header (`BP_BANK_W` …), NOT module parameters — header and RTL cannot desync.

**Structure (follow `bp_relay_unroll_pipe.sv` for everything not named here — FSM states
S_IDLE/S_INIT/S_CHECK/S_VAR/S_SATF/S_EMIT/S_DONE, launch/scatter `pc` pipeline, SAT-overlap in S_CHECK,
`sat_pending`, best-kept commit, `found`, sync reset; plus `early_exit`: in the S_CHECK-entry SAT
finalize, `if (early_exit && found)` → S_EMIT, the `bp_relay_bram_dp` semantics):**

- [ ] **Step 1: Elaboration helpers + ROM localparams.** All `function automatic int` over header tables:
  ```systemverilog
  function automatic int chk_at(int g, int j);  return BP_CHK_AT[g*BP_BANK_W+j]; endfunction
  function automatic int var_at(int h, int i);  return BP_VAR_AT[h*BP_BANK_V+i]; endfunction
  function automatic int edge_at(int g, int j, int k);          // -1 if no check / k >= deg
  function automatic int vedge_at(int h, int i, int d);         // -1 if no var / d >= deg
  function automatic int grp_of_chk(int c);  // scan BP_CHK_AT; slot_of_chk likewise
  function automatic int bank_of_edge(int e); return slot_of_chk(BP_EDGE_CHK[e])*BP_CHK_DEG + BP_EDGE_POS[e]; endfunction
  ```
  Port-assignment for the ≤2 e_cm readers per bank per VAR cycle: readers of bank `b` in var-group `h`,
  ordered by `(i,d)` — first gets port A, second port B (elab function `ecm_port(h,i,d)` returns 0/1).

- [ ] **Step 2: Memories — LUTRAM inference idiom.** Per-bank `generate` blocks, ONE sync write port +
  ADDRESSED async reads only (never constant-tap a mem cell — that kills LUTRAM inference and rebuilds
  the flop-array mux wall):
  ```systemverilog
  for (genvar b = 0; b < 2*BP_BANK_W*BP_CHK_DEG; b++) begin : gmcm   // m_cm half-banks
    logic signed [MSG_BITS-1:0] mem [BP_GC];
    always_ff @(posedge clk) if (we[b]) mem[waddr[b]] <= wdata[b];
    assign q[b] = mem[raddr[b]];
  end
  ```
  Same shape for `e_cm` (single banks, TWO read addresses `raddr_a/raddr_b` → two `assign` reads) and
  `m_vm` (read `raddr = pc`, write `waddr = pc-2`). Consumers mux the banks' **output wires** `q[...]`
  by group-indexed ROMs (localparam int arrays) — never index generate instances at runtime.

- [ ] **Step 3: CHK phase dataflow.** Launch (`pc < GC`): every m_cm half-bank `raddr = pc`; lane (j,k):
  `m_in[j][k] = BETA_LANE[j][k][pc] ? q[hb1(j,k)] : q[hb0(j,k)]` (BETA_LANE a localparam bit ROM),
  `present = (edge_at(pc,j,k) >= 0)`, `sbit` = GC:1 mux over `s_reg[chk_at(pc,j)]`. Scatter (`pc >= 2`):
  `e_cm[(j,k)]`: `we = present(pc-2,j,k)`, `waddr = pc-2`, `wdata = chk_e_out[j][k]`. SAT-overlap: parity
  lane = `ehat[BP_EDGE_VAR[edge_at(pc,j,k)]]` — GC:1 1-bit muxes, folded exactly like
  `bp_relay_unroll_pipe.sv:211-235`.

- [ ] **Step 4: VAR phase dataflow.** Launch: e_cm read addrs per bank from ROMs (`ROWA[b][pc]`,
  `ROWB[b][pc]`); operand (i,d): `e_in[i][d] = ECM_PORT[i][d][pc] ? qb[ECM_BANK[i][d][pc]] :
  qa[ECM_BANK[i][d][pc]]` — a GV-deep case-mux on `pc` with all-constant leaves; `m_in[i][d] =
  m_vm_q[(i,d)]` (raddr=pc); `lam/gam` GV:1 ROM muxes (`BP_LAMBDA[var_at(pc,i)]`,
  `BP_GAMMA[leg*BP_N + var_at(pc,i)]`). Scatter (`pc-2`): `m_vm[(i,d)]` write mux-free; m_cm half-bank
  `b`: `we/waddr/wsrc` from per-half-bank ROMs indexed by `pc-2` (`wdata = var_m_out[WSRC_I[b][pc-2]][WSRC_D[b][pc-2]]`,
  ≤GC:1 source mux over output WIRES); `ehat[var_at(pc-2,i)] <= var_ehat_out[i]` + `ehat_w` accumulate
  (as `bp_relay_unroll_pipe.sv:263-278`).

- [ ] **Step 5: S_INIT.** `m_vc = λ` through the VAR write path: GV cycles, `m_vm[(i,d)] <=
  BP_LAMBDA[var_at(pc,i)]` + the same m_cm scatter ROMs with λ data (2-cycle lag not needed — direct,
  no pipeline). `s_reg`/`ehat` init as in unroll_pipe S_IDLE.

- [ ] **Step 6: S_EMIT / S_DONE.** V vars/cycle from `ehat`/`best_e` via `var_at(pc,i)` (mirror
  `bp_relay_unroll_pipe.sv:328-353`), `latency_cycles` 32-bit.

- [ ] **Step 7: Lint target + elaborate clean.** Makefile:
  ```make
  bpbanked-lint:
  	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph 1 0.003 8 24 > hw/bb_gross_tanner.svh
  	$(VERILATOR) --lint-only -Wall --top-module bp_relay_banked \
  		check_minsum.sv var_update.sv bp_relay_banked.sv
  ```
  Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpbanked-lint` → elaborates clean.

- [ ] **Step 8: Commit.**
  ```bash
  git add hw/bp_relay_banked.sv hw/Makefile
  git commit -m "[Q7-02] M7: bp_relay_banked core (beta-split check-major LUTRAM banks)"
  ```

---

### Task 11: Full-core co-sim — bit-exact + (W,V)-invariance + latency

**Files:**
- Create: `hw/tb_bp_banked.cpp` (copy `hw/tb_bp_unroll_pipe.cpp`, adjust: `early_exit` port driven 0,
  32-bit `latency_cycles`)
- Modify: `hw/Makefile` (target `bpbanked`)

**Interfaces:**
- Consumes: `bp_relay_banked`, `bp_circ_vectors.txt` golden (UNCHANGED — spec A2.3).

- [ ] **Step 1: Makefile target** (model on `bpunrollpipe`; W/V via header regen, not -G):
  ```make
  bpbanked:
  	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
  	@for wv in "8 24" "12 36" "16 48"; do \
  		set -- $$wv; echo "== W=$$1 V=$$2 =="; \
  		(cd .. && $(CARGO) qec_q7_bp_graph -- circgraph 1 0.003 $$1 $$2 > hw/bb_gross_tanner.svh); \
  		rm -rf obj_banked; \
  		$(VERILATOR) --cc --exe --build -Wall --Mdir obj_banked --top-module bp_relay_banked \
  			check_minsum.sv var_update.sv bp_relay_banked.sv tb_bp_banked.cpp -o sim_bpbanked || exit 1; \
  		./obj_banked/sim_bpbanked bp_circ_vectors.txt || exit 1; \
  	done
  ```

- [ ] **Step 2: Run, iterate the Task-10 core to green.**
  Run: `PATH="$HOME/.cargo/bin:$PATH" make -C hw bpbanked`
  Expected (final): three sections, each `PASS: 40 full decodes bit-identical to the fixed-point golden`
  + latency print. Sanity: worst latency ≈ 3 946 / 2 836 / 2 281 cycles (±10% — the model is close, the
  RTL number is the truth). All three PASSING **is** the (W,V)-invariance check. Also run
  `make -C hw bpunrollpipe checkminsum varupdate bpbramdp` once — prior cores still green with the
  fattened header.

- [ ] **Step 3: Commit.**
  ```bash
  git add hw/tb_bp_banked.cpp hw/Makefile
  git commit -m "[Q7-02] M7: banked co-sim — 40/40 bit-exact at W/V=8/24,12/36,16/48; worst <n> cyc"
  ```

---

### Task 12: OOC fit/Fmax probes ×3 → pick (W,V)

**Files:**
- Create: `hw/syn/ooc_banked.tcl` (model on `hw/syn/ooc_core.tcl`; reads `-tclargs <period_ns>`; synth
  `-top bp_relay_banked -mode out_of_context -flatten_hierarchy none -part xck26-sfvc784-2LV-c`; prints
  one line `RESULT W=.. V=.. LUT=.. (pct) FF=.. LUTRAM=.. BRAM=.. DSP=.. WNS=.. area_opt=COMPLETED`).

- [ ] **Step 1: Stage three config dirs on openwebgui** (box git is stale — rsync only):
  ```bash
  for wv in "8 24" "12 36" "16 48"; do set -- $wv; d=bank_w$1v$2; \
    PATH="$HOME/.cargo/bin:$PATH" cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 $1 $2 > /tmp/$d.svh; \
    ssh root@openwebgui.splynx.com "mkdir -p /root/kv260synth/$d/syn"; \
    rsync -az hw/check_minsum.sv hw/var_update.sv hw/bp_relay_banked.sv root@openwebgui.splynx.com:/root/kv260synth/$d/; \
    rsync -az /tmp/$d.svh root@openwebgui.splynx.com:/root/kv260synth/$d/bb_gross_tanner.svh; \
    rsync -az hw/syn/ooc_banked.tcl root@openwebgui.splynx.com:/root/kv260synth/$d/syn/; done
  ```

- [ ] **Step 2: Launch three detached OOC runs at 5.0 ns** (200 MHz probe; handoff §5 ops — detached
  nohup, poll `^RESULT`, never pgrep; PGID scoped-kill if needed):
  ```bash
  for d in bank_w8v24 bank_w12v36 bank_w16v48; do \
    ssh root@openwebgui.splynx.com "cd /root/kv260synth/$d && source /tools/Xilinx/Vivado/2024.2/settings64.sh && nohup bash -c 'vivado -mode batch -source syn/ooc_banked.tcl -tclargs 5.0 > /root/kv260synth/$d.log 2>&1' >/dev/null 2>&1 & echo LAUNCHED $d"; done
  ```
  Poll (bounded foreground loop): `until` all three logs have `^RESULT` or an error signature
  (`ERROR:|Abnormal|Killed`); also treat >40 min of area-opt silence as the stall signature (kill by
  PGID, record NO-GO for that config).

- [ ] **Step 3: Decide + record.** Pick the LARGEST (W,V) with `area_opt=COMPLETED`, LUT ≤ ~105k, and
  WNS ≥ −1.0 ns at 5.0 (else rerun that config at 10.0 to set the board FCLK at 100). Tie-break: fewest
  cycles (Task-11 numbers). Commit the tcl + a one-paragraph verdict in the commit body:
  ```bash
  git add hw/syn/ooc_banked.tcl
  git commit -m "[Q7-02] M7: OOC probes — picked W=<w> V=<v> (LUT=<n> <pct>%, WNS=<x> @5ns)"
  ```

---

### Task 13: KV260 board build

**Files:**
- Create: `hw/bp_axi_wrap_banked.sv` (copy `hw/bp_axi_wrap_wide.sv`; instance `bp_relay_banked u_dec`
  — port list is contract-identical; `localparam logic [31:0] IDCODE = 32'h4250_0003;` and update the
  header comment)
- Create: `hw/syn/kv260_bp_circ_banked_bd.tcl` (copy `hw/syn/kv260_bp_circ_bd.tcl`; swap in
  `bp_relay_banked.sv` + `bp_axi_wrap_banked.sv` + submodules; keep `-tclargs <build> <out> <fclk>`),
  and the matching top if the bd names one (`bp_axi_top_wide.v` pattern — read the tcl first and mirror
  exactly what it reads).
- Modify: `hw/sw/bp_circ_kv260.py` — accept `--idcode 0x42500003` (default stays the old constant).

- [ ] **Step 1: Wire + lint** (`make -C hw bpbanked-lint` still green; verilator-lint the wrap too:
  add it to the lint target's file list).

- [ ] **Step 2: Full board build on openwebgui** at the Task-12 FCLK (100 or 200):
  ```bash
  rsync -az hw/*.sv hw/*.v root@openwebgui.splynx.com:/root/kv260synth/hw/
  rsync -az /tmp/bank_w<w>v<v>.svh root@openwebgui.splynx.com:/root/kv260synth/hw/bb_gross_tanner.svh
  rsync -az hw/syn/kv260_bp_circ_banked_bd.tcl root@openwebgui.splynx.com:/root/kv260synth/hw/syn/
  ssh root@openwebgui.splynx.com 'cd /root/kv260synth/hw && source /tools/Xilinx/Vivado/2024.2/settings64.sh && nohup bash -c "vivado -mode batch -source syn/kv260_bp_circ_banked_bd.tcl -tclargs /root/kv260synth/build_banked /root/kv260synth/out_banked <fclk> > /root/kv260synth/banked_build.log 2>&1" >/dev/null 2>&1 & echo LAUNCHED'
  ```
  Poll `^RESULT`. Expected: `TIMING_MET`, `.bit` + `.hwh` in `out_banked/`. If timing fails at 200,
  rebuild at 150 then 100 — report the honest FCLK.

- [ ] **Step 3: Commit.**
  ```bash
  git add hw/bp_axi_wrap_banked.sv hw/syn/kv260_bp_circ_banked_bd.tcl hw/sw/bp_circ_kv260.py
  git commit -m "[Q7-02] M7: KV260 board build for bp_relay_banked (TIMING_MET @ <f> MHz)"
  ```

---

### Task 14: Silicon decode + latency

**Files:** none new (reuse `hw/sw/bp_circ_kv260.py`, `hw/sw/bp_circ_pynq.py`).

- [ ] **Step 1: Transfer via base64 stream** (scp flaky on this link):
  `for f in out_banked.bit out_banked.hwh bp_circ_vectors.txt bp_circ_kv260.py bp_circ_pynq.py; do base64 < $f | ssh root@192.168.88.174 "base64 -d > /tmp/$f"; done`
  (bit/hwh pulled from openwebgui first via rsync.)

- [ ] **Step 2: Run both modes** (PYNQ Overlay bypass is already inside the runner):
  ```bash
  ssh root@192.168.88.174 'cd /tmp && XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 bp_circ_kv260.py out_banked.bit bp_circ_vectors.txt --clk <f>e6 --idcode 0x42500003'
  ```
  Expected: `IDCODE ok (0x42500003)`, `40/40 ... match golden`, full-schedule latency ≈ the Task-11
  cycle count → **µs number**; then the early-exit run for the average-case number.

- [ ] **Step 3: Record** cycles + µs (both modes) for Task 15. No commit (no files changed).

---

### Task 15: Document M7 + PR

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md` (append `# Q7-02 M7` section)

- [ ] **Step 1: Write the M7 section.** The full design-space story (it is the milestone's real
  finding): full-unroll fit-gate 453k LUT = 386% (synthesizes in ~3 min once modular — the compute wall
  was RTL structure); partial-unroll correct but area-opt stalls on gather muxes; banked path — the
  solve, the three probes (LUT/WNS table), chosen (W,V), silicon latency both modes, ladder
  6.72 ms (M6) → <X> µs (M7, ~<n>×), honest scope (1–3 µs needs a bigger part).

- [ ] **Step 2: Commit, push, PR.**
  ```bash
  git add docs/perf/qec-q7-fixed-bp.md
  git commit -m "[Q7-02] M7: docs — banked relay-BP on KV260 (<X> us, <n>x over M6)"
  git push -u origin q7-02-m7-us-minsum
  gh pr create --base main --title "[Q7-02] M7: us-class banked relay-BP on KV260 (<X> us)" --body "…Advances #322…"
  ```
  (PR body: summary, co-sim 40/40 ×3 configs, OOC table, silicon numbers, `Advances #322` — NOT Closes.)

- [ ] **Step 3: Merge on green** (real gates: macos + linux-stable + clippy + rustfmt + python; beta
  disk-full = flake → rerun). `gh pr merge --squash --delete-branch`.

---

## Amendment-2 Self-Review

**Spec coverage:** A2.1 stores/dataflow → Task 10 Steps 2–6; A2.2 solve+feasibility → Task 9 (König fallback = documented panic path, YAGNI); A2.3 bit-exact/no-regen → Task 11 (golden untouched,
vectors regenerated byte-identical by the same command); A2.4 emitter/probes/board/docs → Tasks 9/12/13/15.
**Types:** core port list matches `bp_axi_wrap_wide`'s instance contract (verified against
`bp_axi_wrap_wide.sv:73-77`); `BP_*` table names consistent across Tasks 9→10; `ecm_port/ECM_PORT`,
`BETA_LANE`, `ROWA/ROWB`, `WSRC_I/WSRC_D` named once and reused. **Placeholders:** `<w>/<v>/<f>/<X>`
are measurement outputs by design, resolved at Tasks 12–14. **Granularity:** Tasks 10–11 are the
iterative RTL loop (as Amendment-1 noted, normal for FSM assembly).
