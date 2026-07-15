# M9c Step 3 — serial-gather relay-BP core: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new sibling relay-BP core (`hw/bp_relay_serial.sv`) that replaces the Beneš parallel gather (~159k LUT, 2.05× over KV260) with a **memory-based serial gather** — P BRAM banks read P/cycle over `ceil(N/P)` steps into a counter-indexed buffer feeding the *unchanged* `check_minsum`/`var_update` — so the core **fits `xck26` ≤ ~80 % LUT**, bit-exact to `FixedRelayBp`.

**Architecture:** Every message-routing plane is made mux-free: the ROM feeds **BRAM address ports** (a memory read, not a register-array index), and the **bank-select and buffer position are the cycle counter** (compile-time-sequenced constants). An offline solver assigns each edge a fixed storage slot `(physical bank, intra-address)` and a per-group read schedule `(step, port)` that is **conflict-free** (≤ 1 read per physical bank per step), verified by a gen-time guard. `P` (banks read/cycle) is the area↔throughput knob, pinned from a synth probe; kept a `localparam`.

**Tech Stack:** Rust (`aleph-qec` lib + `qec_q7_bp_graph` emitter), SystemVerilog (Verilator 5 co-sim + Vivado 2024.2 OOC synth on EPYC).

**Spec:** `docs/superpowers/specs/2026-07-15-m9c-serial-gather-design.md`.
**Predecessor:** Step 2 (Beneš) — `docs/superpowers/plans/2026-07-14-m9c-gather-fix-step2.md`, verdict in `docs/perf/qec-q7-fixed-bp.md § M9c` (9.3× LUT cut, still 2.05× over). The Step-2 Beneš core (`bp_relay_banked_bram_m.sv`, `bp_benes.sv`) is **KEPT** as the parallel/over-budget sibling; this plan adds a serial sibling, it does not delete Step 2.

## Global Constraints

- **Bit-exact:** 40/40 decision-equal vs `FixedRelayBp` at **both** bankings (8/24, 16/48). Gate = a new `make -C hw bpserial` co-sim (mirrors `bpbankedbramm`, same golden `bp_circ_vectors.txt`). No synth before green.
- **Latency is ALLOWED to grow** (serialization is the point) — record it, never fix it. This is the one hard difference from Step 2's latency-exact constraint.
- **`check_minsum` and `var_update` are UNCHANGED** — the serial gather feeds them the same parallel operand set. Do not edit those modules.
- **No ROM-value-as-register-index anywhere.** ROM values feed BRAM address ports only; bank-select and buffer position are the step/port counters.
- **Emitter conflict-free assignment is gen-time guarded** — a `rom_contract`-style time-0 `$fatal` on any `(bank, step)` collision or tap-order mismatch (house `verify_banking` pattern).
- **Config (16/48 gross, the fit config):** `NEB=400, NHB=800, NVB=288, MSG_BITS`, groups `GC`/`GV`, `BP_CHK_DEG=25, BP_VAR_DEG=6, BP_LEGS=6, BP_ITERS=10`. RTL stays parameterized (8/24 must also pass).
- **Bench host:** EPYC `root@195.154.249.85`, Vivado `/tools/Xilinx/Vivado/2024.2` (`source settings64.sh`), serial `set_param general.maxThreads 1`, stage dir `/data/kv260fit/`, `ooc_serial.tcl`. Synth ~25 min (Step-2 rate), so P can be probed across a few values.
- **KV260 budget:** 117 120 CLB LUT, 144 `RAMB36` (288 `RAMB18`), 234 240 FF, 1 248 DSP.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-qec/src/serial_gather.rs` (**create**) | Pure solver: fold the per-edge logical banking onto P physical banks; assign storage `(bank, addr)` + per-group read schedule `(step, port)` conflict-free; produce the address/schedule tables. Fully unit-tested. |
| `crates/aleph-qec/src/lib.rs` (**modify**) | export the solver fns. |
| `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (**modify**) | new emit mode `serialgraph`: call the solver, emit `BP_ROM_SG_*` address ROMs + `BP_SG_P/STEPS/...` localparams + gen-time guard. |
| `hw/bp_serial_mem.sv` (**create**) | `bp_serial_mem #(P, DEPTH, W)` — P true-dual-port BRAM banks; read/write by (port, addr). |
| `hw/bp_gather_seq.sv` (**create**) | `bp_gather_seq` sequencer FSM + `bp_gather_buf` in-order buffer. |
| `hw/tb_bp_serial_mem.cpp`, `hw/tb_bp_gather_seq.cpp` (**create**) | standalone module TBs. |
| `hw/bp_relay_serial.sv` (**create**) | the new core: serial mem + sequencer + buffer feeding unchanged `check_minsum`/`var_update`; serial-step FSM. |
| `hw/tb_bp_serial.cpp` (**create**) | full-core co-sim (fork of `tb_bp_banked.cpp`, same golden). |
| `hw/Makefile` (**modify**) | targets `bpsgmem`, `bpgatherseq`, `bpserial`, `bpserial-lint`. |
| `docs/perf/qec-q7-fixed-bp.md` (**modify**) | § M9c Step-3 fit verdict. |

---

### Task 1: Serial-gather solver (`crates/aleph-qec/src/serial_gather.rs`) — pure, TDD

**Files:** Create `crates/aleph-qec/src/serial_gather.rs`; modify `lib.rs`; inline `#[cfg(test)]`.

**Interfaces:**
- Consumes (from the caller, extracted from `Banking`): per message-class, the per-group list of edges with their `(logical_bank, row)` — for e_cm: `edge_eb`/`edge_row` grouped by var-group; for m_cm: `edge_hb`/`edge_row`; for m_vm: the `(i,d)` slots.
- Produces:
  - `pub struct SerialLayout { p: usize, steps: usize, bank_of: Vec<usize>, addr_of: Vec<usize>, sched: Vec<Vec<(usize,usize)>> }` where `bank_of[e]`/`addr_of[e]` = edge e's fixed storage; `sched[g]` lists, per group g, the `(step, port)` at which each tap is read, in **tap order** (so `buffer[step*P+port]` = that tap).
  - `pub fn plan_serial(edges: &[(usize,usize)], groups: &[Vec<usize>], p: usize) -> SerialLayout` — assign storage + a conflict-free per-group schedule; panics if infeasible (should not happen for `p ≤` physical-bank count with balanced folding).
  - `pub fn verify_layout(layout: &SerialLayout, groups: &[Vec<usize>]) -> Result<(), String>` — the checker used by the emitter guard and tests: (a) every group's reads have distinct `(step,port)` and hit distinct banks per step, (b) `bank_of[e]`/`addr_of[e]` distinct per physical bank (no storage aliasing across different edges mapped to the same slot), (c) tap-order preserved.

- [ ] **Step 1: Write the failing test.** Assert `plan_serial` on a small synthetic graph (e.g. 12 edges, 3 groups, p=2) yields a layout `verify_layout` accepts; and a stress test over 100 random graphs (deterministic xorshift RNG) at several `p` — every layout passes `verify_layout`, `steps == ceil(max_group_edges/p)`, and no `(step,port)` collision. Include a case that would collide under a naive round-robin to prove the solver actually resolves conflicts.
- [ ] **Step 2:** `cargo test -p aleph-qec --lib serial_gather` → FAIL (unimplemented).
- [ ] **Step 3: Implement.** Storage: fold logical banks onto `p` physical banks by balanced assignment (round-robin logical→physical so each group spreads across banks); `addr_of` = a per-physical-bank running offset that never aliases two edges. Schedule: per group, colour the group's edges by physical bank, then greedily pack into `ceil(n/p)` steps with ≤1 per bank per step (a bipartite edge-colouring / first-fit by bank). Emit in tap order. Keep it deterministic (sort by edge id; no RNG in the solver).
- [ ] **Step 4:** `cargo test -p aleph-qec --lib serial_gather` → PASS.
- [ ] **Step 5:** `cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt --check` → clean.
- [ ] **Step 6: Commit** `[Q7-04] M9c step 3.1: serial-gather conflict-free slot solver (TDD)`.

---

### Task 2: Emitter — emit address ROMs + gen-time guard (`qec_q7_bp_graph.rs`)

**Files:** Modify `crates/aleph-qec/examples/qec_q7_bp_graph.rs`.

**Interfaces:**
- Consumes: `Banking`, `aleph_qec::{plan_serial, verify_layout}` (crate-root re-export — verify path in `lib.rs`, as Step-2 did NOT use `aleph_qec::benes::`).
- Produces: a new emit mode `serialgraph` that emits, per message class, `BP_ROM_SG_ECM_ADDR`/`_MCM_ADDR`/`_MVM_ADDR` (per-group per-step per-port intra-bank addresses, packed rows, house `emit_rom_table`) + localparams `BP_SG_P`, `BP_SG_STEPS_ECM/MCM/MVM`, storage-depth params; and runs `verify_layout` at emit time (`assert!`/`$fatal` twin) so a bad schedule fails generation, both bankings.

- [ ] **Step 1:** Add a `verify_layout` call in a new emit path; run `cargo run --release --example qec_q7_bp_graph -- serialgraph 1 0.003 16 48 > /dev/null` → currently fails (mode absent). This is the guard test.
- [ ] **Step 2:** Confirm FAIL.
- [ ] **Step 3: Implement** `emit_serial_graph(...)`: build the per-class `edges`/`groups`, call `plan_serial` with a `P` arg (default from CLI, e.g. `serialgraph <rounds> <p> <bank_w> <bank_v>`), emit the address ROMs + localparams (mirror `print_rom_rows`/`emit_rom_table`), and `verify_layout` → panic on error.
- [ ] **Step 4:** `cargo run --release --example qec_q7_bp_graph -- serialgraph 1 8 0.003 16 48 | grep -c BP_ROM_SG` → ≥3; and `8 24` exits 0. No panic (guard passes both bankings).
- [ ] **Step 5:** clippy + fmt clean.
- [ ] **Step 6: Commit** `[Q7-04] M9c step 3.2: emit serial-gather address ROMs + gen-time guard`.

---

### Task 3: `bp_serial_mem.sv` — P-banked message store + standalone TB

**Files:** Create `hw/bp_serial_mem.sv`, `hw/tb_bp_serial_mem.cpp`; modify `hw/Makefile` (`bpsgmem`).

**Interfaces:** `bp_serial_mem #(parameter int P, DEPTH, W)( input clk, input logic [P-1:0] we, input logic [P-1:0][$clog2(DEPTH)-1:0] waddr/raddr, input logic [P-1:0][W-1:0] wdata, output logic [P-1:0][W-1:0] rdata )` — P independent true-dual-port BRAM banks (`(* ram_style="block" *)`), one write + one read port each per cycle, addressed independently. Consumers: `bp_gather_seq`.

- [ ] **Step 1:** `tb_bp_serial_mem.cpp` — write random `(bank, addr, data)` triples (distinct addrs/bank), then read them back on the read port; assert bit-exact; ≥10000 ops. Mirror `tb_var_update.cpp` structure.
- [ ] **Step 2:** Add `bpsgmem` Make target; run → FAIL (module absent).
- [ ] **Step 3:** Implement `bp_serial_mem` (generate P BRAM banks).
- [ ] **Step 4:** `make -C hw bpsgmem` → PASS; BRAM inference confirmed (no LUTRAM warning for the intended `block` style).
- [ ] **Step 5:** `verilator --lint-only -Wall --top-module bp_serial_mem -GP=8 -GDEPTH=64 -GW=6 bp_serial_mem.sv` → exit 0.
- [ ] **Step 6: Commit** `[Q7-04] M9c step 3.3: P-banked serial message store + unit test`.

---

### Task 4: `bp_gather_seq.sv` + `bp_gather_buf` — sequencer + in-order buffer + TB

**Files:** Create `hw/bp_gather_seq.sv`, `hw/tb_bp_gather_seq.cpp`; modify `hw/Makefile` (`bpgatherseq`).

**Interfaces:** `bp_gather_seq #(P, STEPS, N, W)( input clk, start, input <addr ROM stream for this group>, output busy, done, output logic [N-1:0][W-1:0] gathered )` — on `start`, iterate `step=0..STEPS-1`: drive the P read addresses (from the group's ROM row) into `bp_serial_mem`, latch the P results into `buffer[step*P +: P]` (counter index), assert `done` after `STEPS`. `bp_gather_buf` = the `[N][W]` register file written at `step*P+port`. Scatter mode: reverse (drain buffer → mem writes).

- [ ] **Step 1:** `tb_bp_gather_seq.cpp` — instantiate `bp_gather_seq` + a `bp_serial_mem` preloaded with known data; drive a random conflict-free schedule (generated in C++, mirroring `verify_layout`); assert `gathered[tap]` == the value stored for that tap; assert `done` at exactly `STEPS`. ≥10000 groups.
- [ ] **Step 2:** Add `bpgatherseq` target; run → FAIL.
- [ ] **Step 3:** Implement the sequencer + buffer. The buffer write index is `step*P+port` (constant per (step,port) via generate/counter) — assert in a comment this is the mux-free plane.
- [ ] **Step 4:** `make -C hw bpgatherseq` → PASS (bit-exact gather, `done` at STEPS).
- [ ] **Step 5:** lint clean.
- [ ] **Step 6: Commit** `[Q7-04] M9c step 3.4: serial gather/scatter sequencer + in-order buffer + unit test`.

---

### Task 5: New core `bp_relay_serial.sv` — integrate; co-sim bit-exact

**Files:** Create `hw/bp_relay_serial.sv`, `hw/tb_bp_serial.cpp`; modify `hw/Makefile` (`bpserial`, `bpserial-lint`).

**Interfaces:**
- Consumes: `bp_serial_mem`, `bp_gather_seq`/`buf`, the `BP_ROM_SG_*` ROMs, unchanged `check_minsum`/`var_update`.
- Produces: a core with the same external contract as `bp_relay_banked_bram_m` (same AXI-ish load/decode/readout), bit-exact decisions, latency grown.

- [ ] **Step 1: Baseline** the golden gate: `make -C hw bpbankedbramm` still 40/40 (sanity the env). Record.
- [ ] **Step 2: Build the core.** Fork `bp_relay_banked_bram_m.sv`; replace the 400/800/288 per-bank cells + the three Beneš gather/scatter sites with: `bp_serial_mem` stores (e_cm/m_cm/m_vm) + `bp_gather_seq` gather before each `check_minsum`/`var_update` launch + scatter after; extend the FSM with the inner `step` loop (each `S_CHECK`/`S_VAR` group now spans `STEPS` gather + compute + `STEPS` scatter). Keep `check_minsum`/`var_update` instantiations byte-identical. Feed the serial ROMs via registered address twins.
- [ ] **Step 3:** `tb_bp_serial.cpp` = fork of `tb_bp_banked.cpp` (same `bp_circ_vectors.txt` golden, prints `worst latency`). Add `bpserial` target (both bankings, like `bpbankedbramm`) generating `serialgraph` headers.
- [ ] **Step 4:** `make -C hw bpserial` → **40/40 both bankings** (the primary gate). Iterate the step-loop alignment against the co-sim until decisions match. Record `worst latency` (expected to grow — that is fine).
- [ ] **Step 5:** `make -C hw bpserial-lint` → exit 0.
- [ ] **Step 6: Commit** `[Q7-04] M9c step 3.5: serial-gather core bit-exact (latency grown, records the number)`.

---

### Task 6: EPYC synth — pin P by probe, then fit verdict

**Files:** stage sources to EPYC; modify `docs/perf/qec-q7-fixed-bp.md`.

**Interfaces:** Produces placed LUT/BRAM/DSP/Fmax for `bp_relay_serial` at 2–3 values of `P`, and the pinned `P`.

- [ ] **Step 1: Probe P.** For `P ∈ {4, 8, 16}` (regenerate the `serialgraph` header at each), stage + synth (`ooc_serial.tcl`, `bp_relay_serial`, `xck26`, serial). ~25 min each. Record LUT %, BRAM %, Fmax, and `STEPS` (→ latency) per P.
- [ ] **Step 2: Pin P** = the largest P that fits **≤ ~80 % LUT and ≤ ~90 % BRAM** (best throughput within the fit). Set the `BP_SG_P` default; re-run `make -C hw bpserial` to confirm 40/40 at the pinned P; re-synth once to confirm the placed numbers.
- [ ] **Step 3: Record** the verdict in `docs/perf/qec-q7-fixed-bp.md § M9c` Step-3: the P-sweep table, the pinned-P placed util/Fmax, the latency (cycles + µs at Fmax), and the fit go/no-go. Note throughput honestly (median µs vs 1 µs, per the fit-first decision).
- [ ] **Step 4: Commit** the doc.

---

### Task 7: Streaming re-fit note + branch finish

**Files:** `docs/perf/qec-q7-fixed-bp.md`; branch housekeeping.

- [ ] **Step 1:** If the rounds=1 serial core fits, note whether the **W=6 streaming wrapper** re-fits on it (the streaming core reuses this datapath) → AC-3 buildable on-silicon (fit); if the wrapper needs its own pass, record it as the next item.
- [ ] **Step 2:** Update `.superpowers/sdd/progress.md` + the perf-doc summary: Step-3 outcome, the kept Step-2 sibling, and AC-3 status (fit-met / throughput-documented).
- [ ] **Step 3:** Use superpowers:finishing-a-development-branch to decide merge/PR for the whole `q7-04-m9c-gather-fix` branch (Steps 1–3).

---

## Self-Review

**Spec coverage:** memory-based serial gather (Tasks 3–5) ✅; P-banks + sequencer + in-order buffer (Tasks 3, 4) ✅; conflict-free `(bank,step)` solver + gen-time guard (Tasks 1, 2) ✅; unchanged `check_minsum`/`var_update` (Task 5 constraint) ✅; P = max-that-fits via probe (Task 6) ✅; bit-exact + latency-grown gate (Global Constraints, Task 5) ✅; fit verdict + throughput honesty (Task 6) ✅; keep Step-2 sibling (file structure) ✅; streaming re-fit / AC-3 (Task 7) ✅.

**Placeholder scan:** No TBD/TODO. `P` is a deliberate probe-pinned localparam (Task 6), not a placeholder. RTL micro-arch details in Tasks 3–5 are gated by concrete oracles (module TBs + co-sim 40/40), the same convergence-against-oracle pattern that Step 2 used successfully.

**Type consistency:** `plan_serial`/`verify_layout`/`SerialLayout` used identically in Tasks 1→2. RTL names `bp_serial_mem`/`bp_gather_seq`/`bp_gather_buf`, params `P`/`STEPS`/`N`/`W`, ROMs `BP_ROM_SG_*`, localparams `BP_SG_P`/`BP_SG_STEPS_*` consistent Tasks 2→6. Gate `make -C hw bpserial` = 40/40 both bankings everywhere.

**Risks (carried from spec):** (1) **BRAM pressure** — the P-bank consolidation must fit ≤~90 % BRAM while relieving LUT (Step 2 already at 77 %); the P-probe (Task 6) is where this is measured and is the main fit risk. (2) **Serial-write pipelining** (scatter overlap) — Task 5 decides via the co-sim latency gate; start simple (separate phase) and overlap only if latency is unacceptable. (3) **Solver feasibility** — Task 1's `verify_layout` + Task 2's gen-time guard prove conflict-freedom before any RTL.
