# M9c Step 4 — e_cm read-addr fabric → BRAM ROM: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Execution corrections (post-merge, for the record):** (1) Task 2 Step 4's delay chain must be `BENES_PIPE_ECM` deep (=3), NOT `BENES_PIPE_ECM-1` — the ROM sync-read lands on the same cycle the old fabric's `din` did, so reaching `t+PIPE` needs the full PIPE stages (co-sim caught this: 39/40 -> 40/40; matches the file's own `benes_ecmrd_q_d` idiom). (2) The baseline worst-case latency pairing is **8/24 -> 4475, 16/48 -> 2810** (perf-doc: 2206/3871 + 604 uniform), the reverse of what an earlier draft stated. (3) BRAM was already over budget at 151.7%% (Block RAM Tiles, not the mis-stated 77%% RAMB18 count) — addr->ROM measured -54 tiles, so it helped both constraints.

**Goal:** Replace the static e_cm read-address Beneš fabric (`u_benes_ad0/ad1`, 31,474 LUT) in `bp_relay_banked_bram_m.sv` with a sync-read BRAM data ROM of the resolved read rows, keeping the core bit-exact.

**Architecture:** The addr fabric routes only per-group ROM constants (`{var_epres, var_erow}` under ROM control), so its output `ra_ecm/rb_ecm` is a pure function of the var-group. The emitter precomputes that lookup into `BP_ROM_ECM_READROW`; the RTL reads it and latency-matches to the old `BENES_PIPE_ECM` depth so every downstream schedule offset is untouched.

**Tech Stack:** Rust (emitter example `qec_q7_bp_graph`), SystemVerilog (`bp_relay_banked_bram_m.sv`, `bp_benes.sv`), Verilator co-sim (`hw/Makefile` target `bpbankedbramm`, TB `tb_bp_banked.cpp`), Vivado OOC synth on the EPYC host for the area probe.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-m9c-ecm-addr-rom-design.md`.
- Bit-exactness gate: **40/40 decision-equal vs `FixedRelayBp` golden** at BOTH bankings (`8 24`, `16 48`) via `make bpbankedbramm`. Non-negotiable.
- **Worst-case decode latency must stay identical** to the current committed core (the latency-match is the whole point). Read the number the co-sim prints on `main` before changing anything; it must not move.
- Gen-time guard must prove the emitted `BP_ROM_ECM_READROW` against the trusted `benes_apply` addr-permutation oracle (no independent re-derivation), `$fatal`/`panic` on mismatch — mirrors the existing `verify_banking` guard discipline.
- The read (`rd0/rd1`) and write (`u_benes_wr`) Beneš fabrics are **not touched**.
- Rust: edition 2021, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` clean. No `unwrap`/`expect` outside tests (the emitter example may use `expect` in gen-time guards, matching the existing `verify_banking` style).
- Commit after each task; branch `q7-04-m9c-ecm-addr-rom` (already created off `origin/main`).

---

## File Structure

- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — **modify** `emit_rom` (~1478–1567): build+emit `BP_ROM_ECM_READROW` + `BP_ECM_READROW_W`; drop `benes_ecmaddr` / `BP_ROM_BENES_ECMADDR`; add the readrow gen-time guard next to the existing Beneš guard.
- `hw/bp_relay_banked_bram_m.sv` — **modify**: delete `bp_rom_benes_ecmaddr_bqm` (~252–260) + its instance (~771), `benes_ecmaddr_q` (~676), `ecm_ad_din`/`u_benes_ad0`/`u_benes_ad1` + the `ra_ecm/rb_ecm` select (~830–846); add `bp_rom_ecm_readrow_bqm` + instance + latency-match regs + unpack.
- `docs/perf/qec-q7-fixed-bp.md` — **modify**: add § M9c Step-4 result note after the OOC probe.

---

## Task 1: Emitter — emit `BP_ROM_ECM_READROW`, guard it, drop the addr control ROM

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (`emit_rom`, ~1478–1567)

**Interfaces:**
- Consumes: `benes_group_matchings(view, b, g, var_deg, ecm_m, mcm_m) -> ([Vec<Option<usize>>;2], Vec<Option<usize>>)` (existing, ~924); `Banking` fields `var_at`, `edge_eb`, `edge_eport`, `edge_row` (existing); `FixedHwView::var_off` (existing); `RomRow::new(bits)` + `RomRow::set(lo, width, val)` (existing); `benes_control`, `complete_partial`, `benes_apply` (existing, from `aleph_qec`); `emit_rom_table(name, depth_expr, rows)` (existing, ~1306).
- Produces (into `bb_gross_tanner.svh`): `localparam int BP_ECM_READROW_W = NEB*2*BWC;` and ROM `BP_ROM_ECM_READROW [BP_GV]`, each row `NEB*2*BWC` bits, bank `b` field at `(b*2+port)*BWC +: BWC`. Removes `BP_ROM_BENES_ECMADDR` and (if now unused) `BP_BENES_ECM_*ADDR*` — but KEEP `BP_BENES_ECM_COLS/M/PORTS` (still used by `BP_ROM_BENES_ECMRD`).

- [ ] **Step 1: Read the current addr-ROM emit and the var-row builder to confirm field widths**

Read `crates/aleph-qec/examples/qec_q7_bp_graph.rs` lines 1354–1370 (find `neb`, `bwc`, `gv`, `var_deg`) and 1434–1440 (the `r_erow.set(s*bwc, bwc, b.edge_row[e] as u64)` write). Confirm: `neb = w * chk_deg`, `bwc = clog2(gc)`, tap `s = i*var_deg + d`, and `edge_row[e]` is the `bwc`-bit read row. These are the exact values the readrow table must reproduce.

- [ ] **Step 2: Build the readrow rows (insert before the `benes_ecmaddr`/`benes_ecmrd`/`benes_mcmwr` loop, ~1511)**

```rust
// M9c Step 4 (Q7-04): resolved e_cm read-address ROM. The addr Beneš fabric routed the static
// per-group {var_epres, var_erow} payload to e_cm banks; its output ra_ecm/rb_ecm is therefore a
// pure function of the group. Precompute it here: for var group g, bank b, port p, the read row is
// edge_row of the (unique, <=1-per-bank-per-port) tap that lands on (b, p), else 0. Consumes NO
// runtime data -> replaces the 512x2 network (`bp_benes_ecm_addr` x2) with this data ROM.
// Bit-exact note: the old fabric emitted 0 both for an absent tap AND for a present tap whose row
// is 0 (`valid ? row : '0`, and check-group 0 has row 0) -> storing bwc row bits (0 in both cases)
// reproduces it exactly; no valid bit is needed.
let mut benes_readrow = Vec::with_capacity(gv);
for g in 0..gv {
    let mut r_readrow = RomRow::new(neb * 2 * bwc);
    for i in 0..vcap {
        let var = b.var_at[g * vcap + i];
        if var < 0 {
            continue;
        }
        let var = var as usize;
        let deg = (view.var_off[var + 1] - view.var_off[var]) as usize;
        for d in 0..var_deg.min(deg) {
            let e = view.var_off[var] as usize + d;
            let bank = b.edge_eb[e] as usize; // 0..neb
            let port = b.edge_eport[e] as usize; // 0/1
            let row = b.edge_row[e] as u64; // bwc bits
            r_readrow.set((bank * 2 + port) * bwc, bwc, row);
        }
    }
    benes_readrow.push(r_readrow);
}
```

- [ ] **Step 3: Add the gen-time guard (right after Step 2's loop)**

Cross-check every emitted readrow field against the addr permutation applied through the trusted `benes_apply` oracle, so the ROM is proved against the same fabric semantics it replaces.

```rust
// Gen-time guard: reproduce the readrow table via the addr Beneš permutation (complete_partial +
// benes_control + benes_apply) and assert bank-for-bank equality. This ties the data ROM to the
// SAME trusted permutation the pre-Step-4 fabric realised (benes.rs round-trip oracle), not a
// hand-mirrored map. Panics loudly on any divergence, both bankings.
// Small LSB-first reader over RomRow's `bits: Vec<bool>` (RomRow exposes `set` but no `get`).
let read_field = |row: &RomRow, lo: usize, width: usize| -> u64 {
    let mut v = 0u64;
    for i in 0..width {
        if row.bits[lo + i] {
            v |= 1u64 << i;
        }
    }
    v
};
for g in 0..gv {
    let (dest_ecm, _) = benes_group_matchings(*view, b, g, var_deg, ecm_m, mcm_m);
    // Per-tap payload = row (0 when absent), matching the store.
    let mut erow_of_tap = vec![0u64; ecm_m];
    for i in 0..vcap {
        let var = b.var_at[g * vcap + i];
        if var < 0 {
            continue;
        }
        let var = var as usize;
        let deg = (view.var_off[var + 1] - view.var_off[var]) as usize;
        for d in 0..var_deg.min(deg) {
            let e = view.var_off[var] as usize + d;
            erow_of_tap[i * var_deg + d] = b.edge_row[e] as u64;
        }
    }
    for (port, dest) in dest_ecm.iter().enumerate() {
        let full = complete_partial(dest, ecm_m);
        let routed = benes_apply(&benes_control(&full), &(0..ecm_m).collect::<Vec<_>>());
        // routed[out] = the input tap landing at output bank `out`.
        for bank in 0..neb {
            let tap = routed[bank];
            let expect = if dest[tap].is_some() { erow_of_tap[tap] } else { 0 };
            let got = read_field(&benes_readrow[g], (bank * 2 + port) * bwc, bwc);
            assert_eq!(
                got, expect,
                "BP_ROM_ECM_READROW guard: group {g} bank {bank} port {port} \
                 got {got} want {expect}"
            );
        }
    }
}
```

(If `RomRow` has no `get(lo,width)` reader, add a small one mirroring `set`, or read `.bits`. Check the `RomRow` definition ~near `emit_rom_table` first; use whatever accessor exists.)

- [ ] **Step 4: Delete the addr control-ROM production**

Remove the `benes_ecmaddr` vector, its `pack_bits(&ctrl_addr)` push inside the `for g in 0..gv` loop (~1521–1531), the `ctrl_addr`/`benes_control(&full)` addr lines, and the `emit_rom_table("BP_ROM_BENES_ECMADDR", "BP_GV", &benes_ecmaddr);` call (~1564). Keep `benes_ecmrd` and `benes_mcmwr` untouched. (The `invert`/`complete_partial` for the READ ctrl stay.)

- [ ] **Step 5: Emit the new localparam + ROM table (near the other `emit_rom_table` calls, ~1560)**

```rust
println!();
println!("localparam int BP_ECM_READROW_W = {};", neb * 2 * bwc);
emit_rom_table("BP_ROM_ECM_READROW", "BP_GV", &benes_readrow);
```

- [ ] **Step 6: Build + run the emitter at both bankings; guard must pass, header must swap**

```bash
cd /Users/ex/GitHub/aleph
cargo build --example qec_q7_bp_graph 2>&1 | tail -3
cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 8 24  > /tmp/g824.svh
cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > /tmp/g1648.svh
grep -c "BP_ROM_ECM_READROW"    /tmp/g824.svh /tmp/g1648.svh
grep -c "BP_ROM_BENES_ECMADDR"  /tmp/g824.svh /tmp/g1648.svh
```
Expected: emitter runs with **no panic** (guard passes) at both; `BP_ROM_ECM_READROW` count ≥ 1; `BP_ROM_BENES_ECMADDR` count **0**.

- [ ] **Step 7: clippy + fmt, then commit**

```bash
cargo clippy -p aleph-qec --examples -- -D warnings 2>&1 | tail -3
cargo fmt -p aleph-qec
git add crates/aleph-qec/examples/qec_q7_bp_graph.rs
git commit -m "[Q7-04] M9c Step 4a: emit BP_ROM_ECM_READROW, guard vs benes_apply, drop addr ctrl ROM

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: RTL — swap addr fabric for the readrow ROM, latency-matched; co-sim 40/40

**Files:**
- Modify: `hw/bp_relay_banked_bram_m.sv`

**Interfaces:**
- Consumes: `BP_ROM_ECM_READROW`, `BP_ECM_READROW_W`, `BP_GV`, `BQM_NEBW` (=NEB), `BQM_BWCW` (=BWC), `BQM_AWV`, `var_rd`, `BENES_PIPE_ECM` (all existing/new-from-Task-1).
- Produces: `ra_ecm[b]`, `rb_ecm[b]` (`BWC`-bit read rows per e_cm bank `b`), arriving at the SAME cycle as the removed fabric's outputs.

- [ ] **Step 1: Record the current worst-case latency (baseline to preserve)**

```bash
cd /Users/ex/GitHub/aleph/hw && make bpbankedbramm 2>&1 | grep -iE "worst|latency|40/40|PASS|== W" | tee /tmp/latency_before.txt
```
Expected: `40/40` at both `W=8 V=24` and `W=16 V=48`. Note the printed worst/mean latency — Task 2 must reproduce it exactly.

- [ ] **Step 2: Add the readrow ROM module (beside `bp_rom_benes_ecmaddr_bqm`, ~252)**

Replace the `bp_rom_benes_ecmaddr_bqm` module with:

```systemverilog
module bp_rom_ecm_readrow_bqm (
    input  logic clk,
    input  logic [BQM_AWV-1:0] addr,
    output logic [BP_ECM_READROW_W-1:0] q);
  (* rom_style = "block" *) logic [BP_ECM_READROW_W-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_ECM_READROW[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
```

- [ ] **Step 3: Swap the instance (was `u_rom_benes_ecmaddr`, ~771)**

Remove `benes_ecmaddr_q` (~676) and `u_rom_benes_ecmaddr` (~771). Add:

```systemverilog
  logic [BP_ECM_READROW_W-1:0] ecm_readrow_q;
  bp_rom_ecm_readrow_bqm u_rom_ecm_readrow (.clk(clk), .addr(BQM_AWV'(var_rd)), .q(ecm_readrow_q));
```

- [ ] **Step 4: Delete the addr fabric block (~818–846) and replace with latency-matched unpack**

Delete `ecm_ad_din`, `u_benes_ad0`, `u_benes_ad1`, and the `ra_ecm/rb_ecm` select reading `ad0_dout/ad1_dout`. Replace with (ROM sync-read is 1 cycle; add `BENES_PIPE_ECM-1` delay stages so total latency == old `PIPE=BENES_PIPE_ECM`):

```systemverilog
  // M9c Step 4: e_cm read rows come straight from BP_ROM_ECM_READROW (the old ad0/ad1 Beneš fabric
  // computed a static per-group permutation of ROM constants). Latency-match the removed PIPE=
  // BENES_PIPE_ECM fabric: sync ROM read is 1 cycle, so delay the unpacked rows BENES_PIPE_ECM-1
  // more -> ra_ecm/rb_ecm land on the identical cycle they did before, and every downstream offset
  // (benes_ecmrd_q_d depth, var-operand twins, S_VAR +3, BENES_ECM_LAT) is untouched.
  logic [BQM_BWCW-1:0] ra_ecm_rom [BQM_NEBW];
  logic [BQM_BWCW-1:0] rb_ecm_rom [BQM_NEBW];
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm_rom[b] = ecm_readrow_q[(b*2 + 0)*BQM_BWCW +: BQM_BWCW];
      rb_ecm_rom[b] = ecm_readrow_q[(b*2 + 1)*BQM_BWCW +: BQM_BWCW];
    end
  end
  // BENES_PIPE_ECM-1 pipeline stages (the ROM's own read is stage 0 of BENES_PIPE_ECM).
  logic [BQM_BWCW-1:0] ra_ecm_d [BENES_PIPE_ECM-1][BQM_NEBW];
  logic [BQM_BWCW-1:0] rb_ecm_d [BENES_PIPE_ECM-1][BQM_NEBW];
  always_ff @(posedge clk) begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm_d[0][b] <= ra_ecm_rom[b];
      rb_ecm_d[0][b] <= rb_ecm_rom[b];
    end
    for (int s = 1; s < BENES_PIPE_ECM-1; s++)
      for (int b = 0; b < NEB; b++) begin
        ra_ecm_d[s][b] <= ra_ecm_d[s-1][b];
        rb_ecm_d[s][b] <= rb_ecm_d[s-1][b];
      end
  end
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = ra_ecm_d[BENES_PIPE_ECM-2][b];
      rb_ecm[b] = rb_ecm_d[BENES_PIPE_ECM-2][b];
    end
  end
```

(If `BENES_PIPE_ECM == 1` in some config, this array is empty — guard with a generate/`if`; the shipped config is `BENES_PIPE_ECM = 3`, so the array is size 2. Confirm the value near its localparam before finalizing; keep the pipeline depth exactly `BENES_PIPE_ECM`.)

- [ ] **Step 5: Lint**

```bash
cd /Users/ex/GitHub/aleph/hw && make bpbankedbramm-lint 2>&1 | tail -20
```
Expected: no errors (warnings clean under `-Wall`). Fix any UNUSED/WIDTH issues (e.g. remove now-dead `ad0_dout/ad1_dout` decls, `BENES_PIPE_MCM` unaffected).

- [ ] **Step 6: Full co-sim — the 40/40 gate + latency invariance**

```bash
cd /Users/ex/GitHub/aleph/hw && make bpbankedbramm 2>&1 | grep -iE "worst|latency|40/40|PASS|FAIL|== W" | tee /tmp/latency_after.txt
diff /tmp/latency_before.txt /tmp/latency_after.txt && echo "LATENCY IDENTICAL"
```
Expected: `40/40` at both bankings AND `LATENCY IDENTICAL` (worst/mean unchanged). If latency moved, the pipeline depth is off by one — re-check the `BENES_PIPE_ECM` stage count in Step 4.

- [ ] **Step 7: Commit**

```bash
cd /Users/ex/GitHub/aleph
git add hw/bp_relay_banked_bram_m.sv
git commit -m "[Q7-04] M9c Step 4b: e_cm addr fabric -> BRAM readrow ROM (latency-matched, 40/40)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: OOC synth area probe + perf-doc result note

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md` (add § M9c Step-4 note)

**Interfaces:**
- Consumes: EPYC Vivado host `ssh root@195.154.249.85` (per memory `aleph-bench-server`); target part `xck26-sfvc784-2LV-c` (KV260).

- [ ] **Step 1: Regenerate the at-rest header and push the branch for the host to pull**

```bash
cd /Users/ex/GitHub/aleph/hw && cargo run --quiet --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > bb_gross_tanner.svh
cd /Users/ex/GitHub/aleph && git add hw/bb_gross_tanner.svh 2>/dev/null; git status --short
git push -u origin q7-04-m9c-ecm-addr-rom
```
(Only commit `bb_gross_tanner.svh` if it is a tracked file that changed; `make` leaves it at the 16/48 at-rest state by design.)

- [ ] **Step 2: OOC-synth the modified core on the EPYC host (like the Step-3 ~8-min probe)**

On `ssh root@195.154.249.85`: pull the branch, run an OOC `synth_design -part xck26-sfvc784-2LV-c -mode out_of_context -top bp_relay_banked_bram_m` over `check_minsum.sv var_update.sv bp_benes.sv bp_relay_banked_bram_m.sv` with the 16/48 header, and report `report_utilization` CLB LUTs. Compare against the Step-2 baseline 239,750.
Expected: **~208,000 LUT (~178 %)**, a ~31k drop; the `u_benes_ad0/ad1` cells gone from the hierarchy; no new critical-path warning on the e_cm operand path.

- [ ] **Step 3: Write the result note in `docs/perf/qec-q7-fixed-bp.md`**

Add a `### M9c Step 4 — e_cm read-addr fabric → BRAM ROM` subsection after the Step-3 section: the before/after LUT (239,750 → measured), the % (204.7 % → measured), the BRAM delta (old `BP_ROM_BENES_ECMADDR` removed, `BP_ROM_ECM_READROW` added — net figure from `report_utilization`), the 40/40 + latency-identical co-sim result, and the honest framing (still NO-FIT; next lever = partial-perm on the 76k m_cm write net). Cite the spec + plan paths.

- [ ] **Step 4: Commit**

```bash
cd /Users/ex/GitHub/aleph
git add docs/perf/qec-q7-fixed-bp.md
git commit -m "[Q7-04] M9c Step 4c: perf-doc note — addr->ROM measured LUT drop

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** emitter readrow + guard (Task 1) ✓ spec §1; RTL swap + latency-match (Task 2) ✓ spec §2; ROM sizing/BRAM (Task 2 module + Task 3 measurement) ✓ spec §3; verification — gen-time guard (T1), 40/40 co-sim + latency invariance (T2), OOC probe (T3) ✓ spec §Verification. The spec's "Rust unit test (round-trip)" is realized as the **gen-time guard** (T1 Step 3) instead of a separate library test, following the codebase's `verify_banking` pattern (the emitter logic lives in an example binary, not a library) — equivalent oracle, same `benes_apply` round-trip.

**Placeholder scan:** no TBD/TODO; every code step shows real code. Two explicit "confirm the exact value before finalizing" notes (RomRow accessor in T1S3, `BENES_PIPE_ECM` depth in T2S4) are verification instructions, not placeholders — the code is complete for the shipped config (`BENES_PIPE_ECM=3`).

**Type consistency:** `neb`/`bwc`/`gv`/`var_deg`/`b.v` match the emitter's existing names (verified at lines 934–947, 1424–1440); RTL `BQM_NEBW`/`BQM_BWCW`/`BQM_AWV`/`BENES_PIPE_ECM`/`NEB` match `bp_relay_banked_bram_m.sv` localparams (verified 64–171); `BP_ROM_ECM_READROW`/`BP_ECM_READROW_W` are produced by T1 and consumed by T2 with matching name and `NEB*2*BWC` width.
