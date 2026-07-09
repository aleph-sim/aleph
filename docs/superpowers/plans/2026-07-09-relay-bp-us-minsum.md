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
