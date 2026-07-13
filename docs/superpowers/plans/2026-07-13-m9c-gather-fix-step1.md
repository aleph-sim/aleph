# M9c gather-crossbar fix — Step 1 (m_cm 2:1 beta-split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the m_cm CHK-read runtime crossbar (`qmcm[chk_hbsel_r[idx]]`, ~235k muxes) into a 2:1 beta-select in `hw/bp_relay_banked_bram_m.sv`, bit-exact, then measure the residual mux/LUT on the EPYC synth to decide whether Step 2 (Beneš) is needed.

**Architecture:** `chk_hbsel_r[idx]` is provably `idx*2 + beta` (half-bank `HB = eb*2+beta`, `eb = tap idx = j*CHK_DEG+k`). Rewriting the read as `beta ? qmcm[idx*2+1] : qmcm[idx*2]` makes the tap base a compile-time constant so Vivado folds it to a 2:1 mux instead of an NHB(=800)-way crossbar. Behavior-preserving refactor: same two candidate banks, same select bit → identical read. An elaboration guard proves the `HB>>1 == idx` invariant at time-0.

**Tech Stack:** SystemVerilog (Verilator 5 co-sim), Rust emitter (`qec_q7_bp_graph`), Vivado 2024.2 OOC synth on EPYC.

## Global Constraints

- **Bit-exact:** 40/40 decision-equal vs `FixedRelayBp` at **both** bankings (8/24 and 16/48); worst-case latency unchanged (2206 cyc @16/48, 3871 @8/24). This is the gate — no synth before Verilator is green.
- **No emitter change in Step 1** — `beta` is derived in RTL from `chk_hbsel_r[idx][0]`; the ROM tables stay byte-identical.
- **Preserve the `-flatten_hierarchy none` module boundaries** (stamped cells are load-bearing for Vivado area-opt).
- **Bench host:** EPYC `root@195.154.249.85`, Vivado at `/tools/Xilinx` (source `settings64.sh`), **serial `set_param general.maxThreads 1`** (parallel_synth_helper deadlocks), 123 GB RAM + 128 GB `/data/swapfile`.

---

### Task 1: m_cm CHK read → 2:1 beta-split (RTL, bit-exact refactor)

**Files:**
- Modify: `hw/bp_relay_banked_bram_m.sv` — read site (~L834) + `rom_contract` guard loop (~L510)
- Test (existing regression gate): `hw/Makefile` target `bpbankedbramm` (Verilator 40/40, both bankings)

**Interfaces:**
- Consumes: `qmcm[NHB]` (m_cm half-bank read outputs), `chk_hbsel_r[NEB]` (registered ROM tap→half-bank id), `chk_epres_r[NEB]` (tap present mask), `BP_EDGE_HB[e]` (emitter constant `= eb*2+beta`), `edge_at_bqm(g,j,k)`.
- Produces: `m_in_j[k]` (gathered per-tap m_cm message) — unchanged type/semantics, only the read expression changes.

- [ ] **Step 1: Capture the green baseline (both bankings pass BEFORE the change)**

Run from repo root:
```bash
make -C hw bpbankedbramm
```
Expected: two blocks `== W=8 V=24 ==` and `== W=16 V=48 ==`, each ending in the co-sim's `40/40` PASS line (exit 0). If this is not green on an unmodified tree, STOP — the environment is broken, not the change.

- [ ] **Step 2: Rewrite the read site (2:1 beta-split)**

In `hw/bp_relay_banked_bram_m.sv`, the CHK gather `always_comb` (~L831-837), replace the read line:
```systemverilog
          m_in_j[k]    = chk_epres_r[idx] ? qmcm[chk_hbsel_r[idx]] : '0;
```
with:
```systemverilog
          // M9c: 2:1 beta-split. chk_hbsel_r[idx] == idx*2 + beta (HB = eb*2+beta, eb = idx), so the
          // tap base idx*2 is a compile-time constant and only bit0 (beta) is runtime -> Vivado folds
          // this to a 2:1 mux instead of an NHB-way crossbar. Invariant enforced in rom_contract below.
          m_in_j[k]    = chk_epres_r[idx]
                       ? (chk_hbsel_r[idx][0] ? qmcm[idx*2 + 1] : qmcm[idx*2])
                       : '0;
```
(`idx = j*BP_CHK_DEG + k`; the `for (int k...)` loop unrolls so `idx*2` is constant per tap. `idx*2+1 < NHB = 2*NEB` always in range since `idx < NEB`.)

- [ ] **Step 3: Add the invariant guard (time-0 proof that HB>>1 == tap id)**

In the `rom_contract` `initial` block, inside the `if (e >= 0) begin ... end` that sets `x_hbsel` (~L510-514), add the check:
```systemverilog
          if (e >= 0) begin
            x_epres[j*BP_CHK_DEG + k]                = 1'b1;
            x_hbsel[(j*BP_CHK_DEG + k)*HBW +: HBW]   = HBW'(BP_EDGE_HB[e]);
            // M9c 2:1 beta-split invariant: HB(e) >> 1 must equal the tap id, else qmcm[idx*2+beta] is wrong.
            if ((BP_EDGE_HB[e] >> 1) != (j*BP_CHK_DEG + k)) begin
              $display("bp_relay_banked_bram_m BETA-SPLIT FAIL: g=%0d j=%0d k=%0d HB=%0d expected base %0d",
                       g, j, k, BP_EDGE_HB[e], j*BP_CHK_DEG + k); fails = fails + 1;
            end
          end
```

- [ ] **Step 4: Re-run the co-sim — must stay 40/40 at both bankings, no guard FAIL**

Run:
```bash
make -C hw bpbankedbramm
```
Expected: both `W=8 V=24` and `W=16 V=48` blocks still print `40/40` PASS; **no** `BETA-SPLIT FAIL` or `ROM-CONTRACT FAIL` line; exit 0. (Behavior-preserving → identical decisions and latency.)

- [ ] **Step 5: Lint clean**

Run:
```bash
make -C hw bpbankedbramm-lint
```
Expected: Verilator `--lint-only -Wall` exits 0 (no new warnings from the changed lines).

- [ ] **Step 6: Confirm the streaming wrapper still builds/gates (the core it wraps changed)**

Run:
```bash
make -C hw bpstream
```
Expected: the `bpstream` co-sim stays green (40×7×2 bit-exact). NOTE: `bpstream` instantiates the **flat** `bp_relay_banked_bram.sv`, not the modular `_m`; if so, this step is a no-op sanity check and passes unchanged — record which core it uses. If it wraps `_m`, it must stay green.

- [ ] **Step 7: Commit**

```bash
git add hw/bp_relay_banked_bram_m.sv
git commit -m "[Q7-04] M9c step 1: m_cm CHK read 2:1 beta-split (kill ~235k-mux crossbar)

qmcm[chk_hbsel_r[idx]] was a fake NHB(=800)-way crossbar: chk_hbsel_r[idx] is
provably idx*2+beta, so only bit0 varies. Rewrite as beta ? qmcm[idx*2+1] :
qmcm[idx*2] (tap base constant -> 2:1 mux). rom_contract guard now asserts
HB>>1==tap id. Bit-exact: 40/40 both bankings, latency unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 2: Measure the residual on EPYC (decides Step 2)

**Files:**
- Uses: `hw/bp_relay_banked_bram_m.sv` (Task 1 output), `hw/check_minsum.sv`, `hw/var_update.sv`, generated `hw/bb_gross_tanner.svh` (16/48 gross)
- Bench: EPYC `/data/kv260fit/` (Vivado OOC), `ooc_serial.tcl`

**Interfaces:**
- Consumes: the committed Task-1 core.
- Produces: a residual **mux count + placed LUT/LUTRAM/BRAM/Fmax** number that determines whether Step 2 (Beneš) runs. Recorded in `docs/perf/qec-q7-fixed-bp.md` § M9c.

- [ ] **Step 1: Generate the 16/48 gross header and stage the fit inputs**

```bash
cd /Users/ex/GitHub/aleph
cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > hw/bb_gross_tanner.svh
scp hw/bp_relay_banked_bram_m.sv hw/check_minsum.sv hw/var_update.sv hw/bb_gross_tanner.svh \
    root@195.154.249.85:/data/kv260fit/
```
Expected: 4 files copied. (The `.svh` matches the shipped M8 16/48 banking; the co-sim already proved this header + core are 40/40.)

- [ ] **Step 2: Launch the serial OOC synth (detached) and capture the RTL mux stats**

```bash
ssh root@195.154.249.85 'cd /data/kv260fit && source /tools/Xilinx/Vivado/2024.2/settings64.sh && \
  mv -f fit.log fit_prestep1.log 2>/dev/null; \
  nohup vivado -mode batch -source ooc_serial.tcl -tclargs 5.0 m9c_step1 bp_relay_banked_bram_m \
    > fit.log 2>&1 & echo $! > fit.pid; echo "PID $(cat fit.pid)"'
```
Expected: a PID printed, `fit.log` growing. (Serial `maxThreads 1` per Global Constraints.)

- [ ] **Step 3: Extract the pre-map mux count (fast signal, before waiting for full synth)**

Once the log passes "RTL Component Statistics" (minutes in):
```bash
ssh root@195.154.249.85 'grep -iE "Muxes :=" /data/kv260fit/fit.log | awk -F":=" "{s+=\$2} END{print \"total muxes ~\", s}"'
```
Expected: total muxes **≪ 1.17M** (the ~235k 8-bit CHK muxes gone; residual ≈ the e_cm/scatter permutations). This alone tells us if Step 2 is needed.

- [ ] **Step 4: Let synth reach RESULT (or OOM) and record placed numbers**

Poll `fit.pid` (`kill -0`), watch for `^RESULT` in `fit.log` or `util_banked.rpt`. On completion:
```bash
ssh root@195.154.249.85 'grep "^RESULT" /data/kv260fit/fit.log; \
  grep -iE "CLB LUTs|LUT as Memory|Block RAM Tile|DSPs" /data/kv260fit/util_banked.rpt'
```
Expected outcomes:
- **Fits (≤~80% LUT, RESULT prints Fmax):** Step 1 alone solved it → skip Step 2, document + go to streaming re-fit.
- **Still OOM / >100% LUT:** residual permutations (e_cm gather + scatters) dominate → **write the Step-2 (Beneš) plan** using the measured residual, per the spec.

- [ ] **Step 5: Record the number in the perf doc + decide**

Append the measured mux/LUT/Fmax (or OOM point) to `docs/perf/qec-q7-fixed-bp.md` § M9c, state the Step-1 delta vs the 1.17M baseline, and the go/no-go on Step 2. Commit that doc change.

---

## Follow-up (contingent): Step 2 — Beneš permutation for e_cm gather + scatters

Not planned in detail here **by design** — the spec sequences it after Task 2's measured residual so the network sizing and any pipeline stage are chosen against real numbers, not guessed. When Task 2 shows Step 2 is needed, write `docs/superpowers/plans/YYYY-MM-DD-m9c-gather-fix-step2.md` covering: per-group Beneš routing in `qec_q7_bp_graph.rs` (`solve_banking`/`print_rom_rows`), control-bit ROM emission with a `rom_contract` guard, the RTL network for sites 2–4, and the same Verilator-then-EPYC gate.

## Self-review notes

- Spec coverage: Task 1 = spec Step 1; Task 2 = spec "measure then decide"; Step 2 = spec Step 2 (contingent). Verification plan (Verilator gate before synth, incremental EPYC synth) reflected in Task 1 Steps 4–6 and Task 2. Bit-exactness argument (same banks/bit) encoded in the guard (Task 1 Step 3).
- No placeholders: every code/command step has literal content.
- Type consistency: `qmcm`, `chk_hbsel_r`, `chk_epres_r`, `m_in_j`, `BP_EDGE_HB`, `HBW`, `NEB`/`NHB` used as defined in `bp_relay_banked_bram_m.sv`.
