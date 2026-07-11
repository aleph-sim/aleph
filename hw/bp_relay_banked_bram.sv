// Q7-04 M9b — BRAM-ified sibling of the K-BANKED relay-BP core (`bp_relay_banked_bram`).
//
// SIBLING RELATIONSHIP (do not edit the M8 original `bp_relay_banked.sv`):
//   This is a byte-for-byte DECISION-EQUAL twin of `bp_relay_banked` whose per-group CONSTANT decode
//   fabrics — the O(E) constant-mux clouds that the M9a probe fingered as the 169% LUT on the W=6 window
//   graph, while 144 BRAM36 + 64 URAM sit idle — are relocated into SYNCHRONOUS-READ ROMs so Vivado infers
//   them into block RAM instead of LUT mux fabric. The ROMs are initialised (time-0 `initial` copy) from
//   the SAME header-derived expressions (`edge_at`/`vedge_at`/`BP_EDGE_HB`/`BP_EDGE_EB`/`BP_EDGE_ROW`/
//   `BP_EDGE_EPORT`/`BP_LAMBDA`/`BP_GAMMA`/`BP_OBS_MASK`) the M8 combs used, so the STORED VALUES are
//   provably identical — only WHEN they are read changes (a sync ROM read = the combinational decode plus
//   one output register). Every decode DECISION (corr, obs, vflag) is bit-exact to the M8 core; only the
//   latency grows, by exactly the M8 register-plane recipe applied ONCE MORE: +1 everywhere a ROM feeds a
//   launch or a scatter.
//
// RE-LAG (the M8 recipe, +1; a sync-ROM read inserts one register stage in the constant-decode path):
//   * a bank-read + gather register plane already existed in M8 (STAGES=3 CHK + 2-stage VAR); this sibling
//     inserts ONE MORE register plane — the sync ROM read of the SELECT/ADDRESS words — AHEAD of the M8
//     plane, and re-aligns the message-memory reads (m_cm/m_vm read address registered by 1; the e_cm read
//     address IS the ROM's registered output feeding the async-read cell) so DATA and SELECT still meet.
//   * submodule `en` is delayed TWO cycles (en_chk_rr / en_var_rr) instead of one, matching the extra plane.
//   * schedule constants, all bumped +1 from M8:
//       S_INIT   : pc = 0..GV,   ROM read at pc,           write at pc-1
//       S_CHECK  : phase end pc==GC+4;  e_cm scatter gate pc>=5, write-group pc-5
//       S_VAR    : phase end pc==GV+3;  m_cm/m_vm scatter gate pc>=4, write-group pc-4; ehat/ehat_w at pc-4
//       S_EMIT   : ROM read at pc, accumulate/write at pc-1, phase end pc==GV (tail +1)
//   * R5 (the overlapped-SAT parity + finalize) stays LUT wire taps — cheap constant-index `ehat` taps and
//     XOR trees — UNCHANGED, still finalising at pc==GC-1 (re-verified against the co-sim waves). The
//     `early_exit` path (first syndrome-valid decision -> S_EMIT) is likewise structurally unchanged; the
//     golden gate drives early_exit=0.
//   The m_vm read-row(pc-1)/write-row(pc-4) disjointness argument keeps the SAME 3-cycle gap as M8
//     (which reads row pc and writes row pc-3): both cursors shift by one, the gap is unchanged.
//
// CELLS: `bp_ecm_cell_bq` / `bp_mvm_cell_bq` become PURE PORT-DRIVEN memories (like the M7 `bp_mcm_cell`) —
//   their write decode no longer calls `edge_at`/`vedge_at` internally (that self-scan is exactly the LUT
//   scanner cloud this milestone removes); the TOP drives their write enable/address from the ROMs. Cell
//   BOUNDARIES are preserved (the stamped-module structure is load-bearing for `-flatten_hierarchy none`).
//
// $unit-SCOPE NAMES are suffixed `_bq` (BB_*_bq geometry, *_bq helper functions) and the stamped cell
//   modules `*_bq`, so this sibling is collision-safe in a future multi-top compilation unit alongside the
//   M8 original (the Makefile targets build ONE top per scratch dir, so a collision cannot occur in the
//   gate, but the suffix hardens it for free). The `ifndef SYNTHESIS` elaboration guards survive verbatim
//   (they still scan the original localparam arrays; only the one `vedge_at`->`vedge_at_bq` call and the
//   $display module name are renamed — the guard LOGIC is byte-identical).

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

// ================================================================ $unit-scope geometry (shared by cells)
localparam int BB_GC_bq   = BP_GC;                    // number of check groups (m_cm / e_cm rows)
localparam int BB_GV_bq   = BP_GV;                    // number of var groups   (m_vm rows)
localparam int BB_BWC_bq  = $clog2(BP_GC);            // m_cm / e_cm row address width
localparam int BB_BWV_bq  = $clog2(BP_GV);            // m_vm row address width

/* verilator lint_off UNUSEDSIGNAL */
// ============================================================ $unit elaboration helpers over header tables
// Same bodies as the M8 core's `chk_at/var_at/...`, suffixed `_bq` so both siblings can share one $unit.
// Called only with compile-time-constant args (genvar / cell params / group index gated by pc==g), and in
// the ROM-fill `initial` blocks with loop-runtime args (procedural, evaluated once at time-0 — cheap, like
// the elaboration guards below — NOT elaboration-time constant functions, which OOM-kill Verilator).
function automatic int chk_at_bq(input int g, input int j);
  return BP_CHK_AT[g * BP_BANK_W + j];
endfunction
function automatic int var_at_bq(input int h, input int i);
  return BP_VAR_AT[h * BP_BANK_V + i];
endfunction
function automatic int chk_deg_bq(input int c);
  return BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c];
endfunction
function automatic int var_deg_bq(input int v);
  return BP_VAR_OFF[v + 1] - BP_VAR_OFF[v];
endfunction
function automatic int edge_at_bq(input int g, input int j, input int k);
  automatic int c = chk_at_bq(g, j);
  if (c < 0) return -1;
  if (k >= chk_deg_bq(c)) return -1;
  return BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];
endfunction
function automatic int vedge_at_bq(input int h, input int i, input int d);
  automatic int v = var_at_bq(h, i);
  if (v < 0) return -1;
  if (d >= var_deg_bq(v)) return -1;
  return BP_VAR_OFF[v] + d;
endfunction
/* verilator lint_on UNUSEDSIGNAL */

// ============================================================================ STAMPED BANK CELL MODULES
// PURE PORT-DRIVEN memories (write enable/address/data all from the TOP's ROM-fed scatter). One instance
// per (half-bank | bank), so `-flatten_hierarchy none` keeps each memory as an independent area-opt unit.
/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDPARAM */

// --------------------------------------------------------------- m_cm half-bank: 1 sync write, 1 async read
module bp_mcm_cell_bq #(
    parameter int B = 0                                  // half-bank id (identity only; wiring via ports)
) (
    input  logic                       clk,
    input  logic                       we,               // from the top's ROM-fed m_cm write scatter
    input  logic [BB_BWC_bq-1:0]       wa,               // write row (= edge's CHECK group, from ROM)
    input  logic signed [MSG_BITS-1:0] wd,               // write data (lambda in S_INIT, var m_out in S_VAR)
    input  logic [BB_BWC_bq-1:0]       ra,               // read row (= registered pc)
    output logic signed [MSG_BITS-1:0] q
);
  logic signed [MSG_BITS-1:0] mem [BB_GC_bq];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign q = mem[ra];
endmodule

// --------------------------------------------------------------- e_cm bank: 1 sync write, 2 async reads
// PORT-DRIVEN (M9b): write enable/row now come from the TOP (ECM_WR_ROM), not an internal `edge_at` scan.
module bp_ecm_cell_bq #(
    parameter int B = 0                                  // bank id = check-slot j * CHK_DEG + lane k
) (
    input  logic                       clk,
    input  logic                       we,               // ECM_WR_ROM present bit & S_CHECK scatter gate
    input  logic [BB_BWC_bq-1:0]       wa,               // write chk-group row (= pc-5, shared)
    input  logic signed [MSG_BITS-1:0] wd,               // this bank's chk lane message (chk_e_out[JJ][KK])
    input  logic [BB_BWC_bq-1:0]       ra,               // port-A read row (from ROM-fed e_cm read-addr)
    input  logic [BB_BWC_bq-1:0]       rb,               // port-B read row
    output logic signed [MSG_BITS-1:0] qa,
    output logic signed [MSG_BITS-1:0] qb
);
  logic signed [MSG_BITS-1:0] mem [BB_GC_bq];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign qa = mem[ra];
  assign qb = mem[rb];
endmodule

// --------------------------------------------------------------- m_vm bank: 1 sync write, 1 async read
// PORT-DRIVEN (M9b): write enable/row/data all from the TOP (SCATTER_ROM present + lambda), not a scan.
module bp_mvm_cell_bq #(
    parameter int B = 0                                  // bank id = var-slot i * VAR_DEG + edge d
) (
    input  logic                       clk,
    input  logic                       we,               // SCATTER_ROM present bit & (init | var) gate
    input  logic [BB_BWV_bq-1:0]       wa,               // write var-group row (S_INIT: pc-1, S_VAR: pc-4)
    input  logic signed [MSG_BITS-1:0] wd,               // lambda (S_INIT) or var_m_out[II][DD] (S_VAR)
    input  logic [BB_BWV_bq-1:0]       rg,               // read row (= registered pc)
    output logic signed [MSG_BITS-1:0] q
);
  logic signed [MSG_BITS-1:0] mem [BB_GV_bq];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign q = mem[rg];
endmodule
/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on DECLFILENAME */

// ========================================================================================= TOP CORE
module bp_relay_banked_bram (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                early_exit,               // stop at the first syndrome-valid decision
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],
    output logic [BP_OBS-1:0]   obs_flip,
    output logic                valid_flag,
    output logic [31:0]         latency_cycles
);
  // ------------------------------------------------------------------------------- sizes / geometry
  localparam int WACC = 16;                          // matches bp_relay_fast.sv / var_update WACC
  localparam int WW   = $clog2(BP_N + 1);
  localparam int W    = BP_BANK_W;                   // check slots stamped (checks per check-group)
  localparam int V    = BP_BANK_V;                   // var slots stamped (vars per var-group)
  localparam int GC   = BP_GC;                       // number of check groups
  localparam int GV   = BP_GV;                       // number of var groups
  localparam int NHB  = 2 * BP_BANK_W * BP_CHK_DEG;  // m_cm half-banks
  localparam int NEB  = BP_BANK_W * BP_CHK_DEG;      // e_cm banks   (= W * CHK_DEG (j,k) lanes)
  localparam int NVB  = BP_BANK_V * BP_VAR_DEG;      // m_vm banks   (= V * VAR_DEG (i,d) slots)
  localparam int BWC  = $clog2(BP_GC);               // m_cm / e_cm row address width
  localparam int BWV  = $clog2(BP_GV);               // m_vm row address width
  localparam int HBW  = $clog2(NHB);                 // half-bank index width (ROM-stored m_cm tap/scatter)
  localparam int EBW  = $clog2(NEB);                 // e_cm bank index width (ROM-stored operand select)
  localparam int CW   = $clog2(BP_C);                // check index width (ROM-stored sbit source)
  localparam int VW   = $clog2(BP_N);                // var index width (ROM-stored obs / corr target)

  /* verilator lint_off UNUSEDSIGNAL */
  // ================================================================== elaboration guards (verbatim from M8)
  // Enforce the offline (Task-9 emitter) split invariants at time-0. Byte-identical to `bp_relay_banked`
  // except the one `vedge_at`->`vedge_at_bq` call and the $display module name; the guard LOGIC is unchanged.
`ifndef SYNTHESIS
  initial begin : elab_guards
    automatic int fails = 0;
    // (d1) BP_CHK_GRP/BP_CHK_SLOT invert BP_CHK_AT: a fresh scan finds check c at (g_scan,j_scan).
    for (int c = 0; c < BP_C; c++) begin
      automatic int g_scan = -1;
      automatic int j_scan = -1;
      for (int g = 0; g < BP_GC; g++)
        for (int j = 0; j < BP_BANK_W; j++)
          if (BP_CHK_AT[g * BP_BANK_W + j] == c) begin
            g_scan = g;
            j_scan = j;
          end
      if (BP_CHK_GRP[c] != g_scan || BP_CHK_SLOT[c] != j_scan) begin
        $display("bp_relay_banked_bram GUARD(d1) FAIL: check %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
                 c, BP_CHK_GRP[c], BP_CHK_SLOT[c], g_scan, j_scan);
        fails = fails + 1;
      end
    end
    // (d2) BP_VAR_GRP/BP_VAR_SLOT invert BP_VAR_AT.
    for (int v = 0; v < BP_N; v++) begin
      automatic int h_scan = -1;
      automatic int i_scan = -1;
      for (int h = 0; h < BP_GV; h++)
        for (int i = 0; i < BP_BANK_V; i++)
          if (BP_VAR_AT[h * BP_BANK_V + i] == v) begin
            h_scan = h;
            i_scan = i;
          end
      if (BP_VAR_GRP[v] != h_scan || BP_VAR_SLOT[v] != i_scan) begin
        $display("bp_relay_banked_bram GUARD(d2) FAIL: var %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
                 v, BP_VAR_GRP[v], BP_VAR_SLOT[v], h_scan, i_scan);
        fails = fails + 1;
      end
    end
    // (d3) BP_EDGE_EB/HB/ROW recompute from the (d1-validated) chk grp/slot tables.
    for (int e = 0; e < BP_E; e++) begin
      automatic int c   = BP_EDGE_CHK[e];
      automatic int eb  = BP_CHK_SLOT[c] * BP_CHK_DEG + BP_EDGE_POS[e];
      automatic int hb  = 2 * eb + BP_EDGE_BETA[e];
      automatic int row = BP_CHK_GRP[c];
      if (BP_EDGE_EB[e] != eb || BP_EDGE_HB[e] != hb || BP_EDGE_ROW[e] != row) begin
        $display("bp_relay_banked_bram GUARD(d3) FAIL: edge %0d EB/HB/ROW (%0d,%0d,%0d) != recompute (%0d,%0d,%0d)",
                 e, BP_EDGE_EB[e], BP_EDGE_HB[e], BP_EDGE_ROW[e], eb, hb, row);
        fails = fails + 1;
      end
    end
    // (a)/(b)/EPORT: per var-group, accumulate writers-per-half-bank and readers-per-e_cm-bank in ONE pass
    // over the group's present edges (via the scan-validated BP_EDGE_HB/EB), then scan the counters. Counting
    // in a single (i,d) pass (rather than re-scanning all edges for each bank) keeps Verilator's constant-
    // unroll O(GV*V*VAR_DEG) instead of O(GV*(NHB+NEB)*V*VAR_DEG) — the latter symbolically explodes cvt.
    for (int h = 0; h < GV; h++) begin
      automatic int wcnt [NHB];
      automatic int rcnt [NEB];
      for (int b = 0; b < NHB; b++) wcnt[b] = 0;
      for (int b = 0; b < NEB; b++) rcnt[b] = 0;
      for (int i = 0; i < V; i++)
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int e = vedge_at_bq(h, i, d);
          if (e >= 0) begin
            automatic int hb = BP_EDGE_HB[e];
            automatic int eb = BP_EDGE_EB[e];
            // EPORT is the count of same-e_cm-bank readers of this group seen BEFORE e in (i,d) order.
            if (BP_EDGE_EPORT[e] != rcnt[eb]) begin
              $display("bp_relay_banked_bram GUARD(eport) FAIL: var-group %0d edge %0d EPORT=%0d != readers-so-far %0d",
                       h, e, BP_EDGE_EPORT[e], rcnt[eb]);
              fails = fails + 1;
            end
            wcnt[hb] = wcnt[hb] + 1;
            rcnt[eb] = rcnt[eb] + 1;
          end
        end
      // (a) <=1 writer per (var-group, m_cm half-bank) — the single m_cm write port per half-bank.
      for (int b = 0; b < NHB; b++)
        if (wcnt[b] > 1) begin
          $display("bp_relay_banked_bram GUARD(a) FAIL: var-group %0d m_cm half-bank %0d has %0d writers (>1)",
                   h, b, wcnt[b]);
          fails = fails + 1;
        end
      // (b) <=2 readers per (var-group, e_cm bank) — the two async read ports per e_cm bank.
      for (int b = 0; b < NEB; b++)
        if (rcnt[b] > 2) begin
          $display("bp_relay_banked_bram GUARD(b) FAIL: var-group %0d e_cm bank %0d has %0d readers (>2)",
                   h, b, rcnt[b]);
          fails = fails + 1;
        end
    end
    // (c) BP_EDGE_POS[e] is e's position in its check's CSR row (edge_at_bq / BP_EDGE_HB tap correctness).
    for (int e = 0; e < BP_E; e++) begin
      automatic int c   = BP_EDGE_CHK[e];
      automatic int idx = BP_CHECK_OFF[c] + BP_EDGE_POS[e];
      if (idx >= BP_CHECK_OFF[c + 1] || BP_CHECK_EDGES[idx] != e) begin
        $display("bp_relay_banked_bram GUARD(c) FAIL: edge %0d (check %0d) EDGE_POS=%0d does not match CSR row",
                 e, c, BP_EDGE_POS[e]);
        fails = fails + 1;
      end
    end
    if (fails != 0)
      $fatal(1, "bp_relay_banked_bram: %0d elaboration-guard violation(s) — header/emitter split is unsafe", fails);
    else
      $display("bp_relay_banked_bram: elaboration guards (a/b/c/d) PASS (GV=%0d NHB=%0d NEB=%0d BP_E=%0d)",
               GV, NHB, NEB, BP_E);
  end
`endif

  // =============================================================================== FSM state / registers
  typedef enum logic [2:0] {
    S_IDLE, S_INIT, S_CHECK, S_VAR, S_SATF, S_EMIT, S_DONE
  } state_t;
  state_t state;

  (* dont_touch = "true" *) logic s_reg  [BP_C];
  (* dont_touch = "true" *) logic ehat   [BP_N];
  (* dont_touch = "true" *) logic best_e [BP_N];
  logic [WW-1:0]     ehat_w, best_w;
  logic              found, all_sat, sat_pending;
  logic [BP_OBS-1:0] obs_acc;

  int          leg, iter, pc;                        // pc = phase/group cursor
  logic [31:0] lat;

  assign busy           = (state != S_IDLE);
  assign latency_cycles = lat;

  // submodule launch enables (only while groups remain to start)
  logic en_chk, en_var;
  assign en_chk = (state == S_CHECK) && (pc < GC);
  assign en_var = (state == S_VAR)   && (pc < GV);

  // M9b: TWO en-delay stages (was one). The ROM-read plane (+1) precedes the M8 gather plane (+1); the
  // submodule `en` is delayed by both so it captures only once BOTH planes hold a real group's operands.
  logic en_chk_r, en_var_r, en_chk_rr, en_var_rr;
  always_ff @(posedge clk) begin
    en_chk_r  <= en_chk;
    en_var_r  <= en_var;
    en_chk_rr <= en_chk_r;
    en_var_rr <= en_var_r;
  end

  // registered submodule outputs (STAGES clocks after their group launch)
  logic signed [MSG_BITS-1:0] chk_e_out    [W][BP_CHK_DEG];
  logic signed [MSG_BITS-1:0] var_m_out    [V][BP_VAR_DEG];
  logic                       var_ehat_out [V];

  // bank async-read output wires (from the stamped cells)
  logic signed [MSG_BITS-1:0] qmcm   [NHB];
  logic signed [MSG_BITS-1:0] qa_ecm [NEB];
  logic signed [MSG_BITS-1:0] qb_ecm [NEB];
  logic signed [MSG_BITS-1:0] qmvm   [NVB];

  // registered message-memory read addresses (= pc, delayed 1 to meet the sync ROM select)
  logic [BWC-1:0]             mcm_ra_r;                 // m_cm read row
  logic [BWV-1:0]             mvm_ra_r;                 // m_vm read row
  logic [BWC-1:0]             ra_ecm [NEB];             // per-bank e_cm port-A read row (from ROM)
  logic [BWC-1:0]             rb_ecm [NEB];             // per-bank e_cm port-B read row

  // m_cm write drivers (ROM-fed scatter; each half-bank <=1 writer per write-group — guard(a))
  logic                       we_mcm [NHB];
  logic [BWC-1:0]             wa_mcm [NHB];
  logic signed [MSG_BITS-1:0] wd_mcm [NHB];

  // e_cm / m_vm write drivers (ROM-fed; shared write row, per-bank enable)
  logic                       we_ecm [NEB];
  logic [BWC-1:0]             wa_ecm;                   // shared e_cm write row (= pc-5)
  logic                       we_mvm [NVB];
  logic signed [MSG_BITS-1:0] wd_mvm [NVB];
  logic [BWV-1:0]             wa_mvm;                   // shared m_vm write row (S_INIT: pc-1, S_VAR: pc-4)

  // flattened submodule-output buses feeding the cells' write-data ports (thin glue)
  logic signed [MSG_BITS-1:0] chk_e_flat [NEB];
  logic signed [MSG_BITS-1:0] var_m_flat [NVB];

  // ===================================================================== BRAM ROMs (constant decode fabric)
  // Filled at time-0 from the SAME header expressions the M8 combs used; sync-read at the group cursor.
  //
  // ROM SHAPE (Task-4 probe rework): every ROM is a SINGLE unpacked dimension (depth = group count) with a
  // PACKED row word — Vivado's BRAM inference requires exactly this shape; the earlier two-dimensional
  // unpacked form ([GC][W] etc.) silently ignored `rom_style="block"` and decomposed into per-element
  // registers + a read network, reproducing the register+mux explosion this core exists to remove (the
  // rounds=1 graph then stalled synthesis at 47 GB RSS). Packing layout, uniform for every ROM: slot t's
  // field of width FW occupies row bits [t*FW +: FW] (slot = check-slot j, (j,k) lane, var-slot i, or (i,d)
  // slot as noted per ROM). Fields are NOT merged across ROMs: one ROM per field keeps the +: slicing
  // trivial and each ROM independently BRAM-inferable (disclosed choice; same total bits either way).
  // R3 — CHK gather selects (m_cm half-bank tap + syndrome-bit source), per check-group; slot = j or (j,k).
  (* rom_style = "block" *) logic [W*CW-1:0]        chk_sbit_idx_rom  [GC];
  (* rom_style = "block" *) logic [W-1:0]           chk_sbit_pres_rom [GC];
  (* rom_style = "block" *) logic [NEB*HBW-1:0]     chk_hbsel_rom     [GC];
  (* rom_style = "block" *) logic [NEB-1:0]         chk_epres_rom     [GC];
  // R4 + R2(read) — VAR gather (lambda / gamma / e_cm operand bank+port+row), per var-group[, leg];
  // slot = i or (i,d). var_gam_rom row address = leg*GV + group.
  (* rom_style = "block" *) logic [V-1:0]           var_pres_rom  [GV];
  (* rom_style = "block" *) logic [V*MSG_BITS-1:0]  var_lam_rom   [GV];
  (* rom_style = "block" *) logic [V*MSG_BITS-1:0]  var_gam_rom   [BP_LEGS*GV];
  (* rom_style = "block" *) logic [NVB-1:0]         var_epres_rom [GV];
  (* rom_style = "block" *) logic [NVB*EBW-1:0]     var_ebsel_rom [GV];
  (* rom_style = "block" *) logic [NVB-1:0]         var_eport_rom [GV];   // 1 => port B (EPORT==1)
  (* rom_style = "block" *) logic [NVB*BWC-1:0]     var_erow_rom  [GV];
  // R1 — m_cm scatter (+ shared by the m_vm write): half-bank, check-group row, lambda-seed, present;
  // per var-group, slot = (i,d).
  (* rom_style = "block" *) logic [NVB-1:0]          scat_pres_rom [GV];
  (* rom_style = "block" *) logic [NVB*HBW-1:0]      scat_hb_rom   [GV];
  (* rom_style = "block" *) logic [NVB*BWC-1:0]      scat_row_rom  [GV];
  (* rom_style = "block" *) logic [NVB*MSG_BITS-1:0] scat_lam_rom  [GV];
  // R2(write) — e_cm write present mask, per check-group; slot = (j,k).
  (* rom_style = "block" *) logic [NEB-1:0]          ecm_wpres_rom [GC];
  // R6 — S_EMIT observable (per-var present / index / obs-mask), per var-group; slot = i.
  (* rom_style = "block" *) logic [V-1:0]            obs_pres_rom [GV];
  (* rom_style = "block" *) logic [V*VW-1:0]         obs_var_rom  [GV];
  (* rom_style = "block" *) logic [V*BP_OBS-1:0]     obs_mask_rom [GV];

  // -------------------------------------------------------------------------- ROM fills (time-0 initial)
  // PURE DIRECT ARRAY INDEXING of the header localparams — no helper-function calls (Task-4 probe / house
  // 12c lesson: Vivado materializes function evaluation in initial-block ROM fills pathologically). The
  // `_bq` helper bodies (chk_at/var_at/edge_at/vedge_at) are inlined below; the empty-slot / short-degree
  // guards are nested `if`s so no out-of-range header index is ever formed.
  initial begin : fill_chk
    for (int g = 0; g < GC; g++)
      for (int j = 0; j < W; j++) begin
        automatic int c = BP_CHK_AT[g * BP_BANK_W + j];              // chk_at_bq, inlined
        chk_sbit_pres_rom[g][j]         = (c >= 0);
        chk_sbit_idx_rom[g][j*CW +: CW] = (c >= 0) ? CW'(c) : '0;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          automatic int e;
          e = -1;                                                    // edge_at_bq, inlined
          if (c >= 0)
            if (k < BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c]) e = BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];
          chk_epres_rom[g][j*BP_CHK_DEG + k]                 = (e >= 0);
          chk_hbsel_rom[g][(j*BP_CHK_DEG + k)*HBW +: HBW]    = (e >= 0) ? HBW'(BP_EDGE_HB[e]) : '0;
        end
      end
  end
  initial begin : fill_var
    for (int g = 0; g < GV; g++)
      for (int i = 0; i < V; i++) begin
        automatic int v = BP_VAR_AT[g * BP_BANK_V + i];              // var_at_bq, inlined
        var_pres_rom[g][i]                          = (v >= 0);
        var_lam_rom[g][i*MSG_BITS +: MSG_BITS]      = (v >= 0) ? BP_LAMBDA[v][MSG_BITS-1:0] : '0;
        for (int l = 0; l < BP_LEGS; l++)
          var_gam_rom[l*GV + g][i*MSG_BITS +: MSG_BITS] =
              (v >= 0) ? BP_GAMMA[l*BP_N + v][MSG_BITS-1:0] : '0;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int e;
          e = -1;                                                    // vedge_at_bq, inlined
          if (v >= 0)
            if (d < BP_VAR_OFF[v + 1] - BP_VAR_OFF[v]) e = BP_VAR_OFF[v] + d;
          var_epres_rom[g][i*BP_VAR_DEG + d]              = (e >= 0);
          var_ebsel_rom[g][(i*BP_VAR_DEG + d)*EBW +: EBW] = (e >= 0) ? EBW'(BP_EDGE_EB[e]) : '0;
          var_eport_rom[g][i*BP_VAR_DEG + d]              = (e >= 0) ? (BP_EDGE_EPORT[e] == 1) : 1'b0;
          var_erow_rom[g][(i*BP_VAR_DEG + d)*BWC +: BWC]  = (e >= 0) ? BWC'(BP_EDGE_ROW[e]) : '0;
        end
      end
  end
  initial begin : fill_scat
    for (int g = 0; g < GV; g++)
      for (int i = 0; i < V; i++) begin
        automatic int v = BP_VAR_AT[g * BP_BANK_V + i];              // var_at_bq, inlined
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int e;
          e = -1;                                                    // vedge_at_bq, inlined
          if (v >= 0)
            if (d < BP_VAR_OFF[v + 1] - BP_VAR_OFF[v]) e = BP_VAR_OFF[v] + d;
          scat_pres_rom[g][i*BP_VAR_DEG + d]              = (e >= 0);
          scat_hb_rom[g][(i*BP_VAR_DEG + d)*HBW +: HBW]   = (e >= 0) ? HBW'(BP_EDGE_HB[e]) : '0;
          scat_row_rom[g][(i*BP_VAR_DEG + d)*BWC +: BWC]  = (e >= 0) ? BWC'(BP_EDGE_ROW[e]) : '0;
          // m_cm S_INIT seed and m_vm S_INIT seed are the SAME lambda: BP_LAMBDA[BP_EDGE_VAR[e]] ==
          // BP_LAMBDA[var_at(g,i)] for e = vedge_at(g,i,d). One ROM serves both scatters.
          scat_lam_rom[g][(i*BP_VAR_DEG + d)*MSG_BITS +: MSG_BITS] =
              (e >= 0) ? BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0] : '0;
        end
      end
  end
  initial begin : fill_ecmw
    for (int g = 0; g < GC; g++)
      for (int j = 0; j < W; j++) begin
        automatic int c = BP_CHK_AT[g * BP_BANK_W + j];              // chk_at_bq, inlined
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          automatic int e;
          e = -1;                                                    // edge_at_bq, inlined
          if (c >= 0)
            if (k < BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c]) e = BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];
          ecm_wpres_rom[g][j*BP_CHK_DEG + k] = (e >= 0);
        end
      end
  end
  initial begin : fill_obs
    for (int g = 0; g < GV; g++)
      for (int i = 0; i < V; i++) begin
        automatic int v = BP_VAR_AT[g * BP_BANK_V + i];              // var_at_bq, inlined
        obs_pres_rom[g][i]                        = (v >= 0);
        obs_var_rom[g][i*VW +: VW]                = (v >= 0) ? VW'(v) : '0;
        obs_mask_rom[g][i*BP_OBS +: BP_OBS]       = (v >= 0) ? BP_OBS_MASK[v][BP_OBS-1:0] : '0;
      end
  end

  // -------------------------------------------------------------------------- registered ROM row words
  // ONE sync read per ROM per cycle; the `_q` row register IS the (single) ROM output register — the same
  // pipeline stage the earlier per-field registered copies formed, no extra depth.
  logic [W*CW-1:0]         chk_sbit_idx_q;
  logic [W-1:0]            chk_sbit_pres_q;
  logic [NEB*HBW-1:0]      chk_hbsel_q;
  logic [NEB-1:0]          chk_epres_q;
  logic [V-1:0]            var_pres_q;
  logic [V*MSG_BITS-1:0]   var_lam_q;
  logic [V*MSG_BITS-1:0]   var_gam_q;
  logic [NVB-1:0]          var_epres_q;
  logic [NVB*EBW-1:0]      var_ebsel_q;
  logic [NVB-1:0]          var_eport_q;
  logic [NVB*BWC-1:0]      var_erow_q;
  logic [NVB-1:0]          scat_pres_q;
  logic [NVB*HBW-1:0]      scat_hb_q;
  logic [NVB*BWC-1:0]      scat_row_q;
  logic [NVB*MSG_BITS-1:0] scat_lam_q;
  logic [NEB-1:0]          ecm_wpres_q;
  logic [V-1:0]            obs_pres_q;
  logic [V*VW-1:0]         obs_var_q;
  logic [V*BP_OBS-1:0]     obs_mask_q;

  // combinational per-slot field slices of the registered rows (the pre-rework consumer names, unchanged
  // downstream: gathers, scatters, S_EMIT all read these exactly as before)
  logic [CW-1:0]        chk_sbit_idx_r  [W];
  logic                 chk_sbit_pres_r [W];
  logic [HBW-1:0]       chk_hbsel_r     [NEB];
  logic                 chk_epres_r     [NEB];
  logic                 var_pres_r  [V];
  logic [MSG_BITS-1:0]  var_lam_r   [V];
  logic [MSG_BITS-1:0]  var_gam_r   [V];
  logic                 var_epres_r [NVB];
  logic [EBW-1:0]       var_ebsel_r [NVB];
  logic                 var_eport_r [NVB];
  logic [BWC-1:0]       var_erow_r  [NVB];
  logic                 scat_pres_r [NVB];
  logic [HBW-1:0]       scat_hb_r   [NVB];
  logic [BWC-1:0]       scat_row_r  [NVB];
  logic [MSG_BITS-1:0]  scat_lam_r  [NVB];
  logic                 ecm_wpres_r [NEB];
  logic                 obs_pres_r  [V];
  logic [VW-1:0]        obs_var_r   [V];
  logic [BP_OBS-1:0]    obs_mask_r  [V];
  always_comb begin
    for (int j = 0; j < W; j++) begin
      chk_sbit_idx_r[j]  = chk_sbit_idx_q[j*CW +: CW];
      chk_sbit_pres_r[j] = chk_sbit_pres_q[j];
    end
    for (int b = 0; b < NEB; b++) begin
      chk_hbsel_r[b] = chk_hbsel_q[b*HBW +: HBW];
      chk_epres_r[b] = chk_epres_q[b];
      ecm_wpres_r[b] = ecm_wpres_q[b];
    end
    for (int i = 0; i < V; i++) begin
      var_pres_r[i] = var_pres_q[i];
      var_lam_r[i]  = var_lam_q[i*MSG_BITS +: MSG_BITS];
      var_gam_r[i]  = var_gam_q[i*MSG_BITS +: MSG_BITS];
      obs_pres_r[i] = obs_pres_q[i];
      obs_var_r[i]  = obs_var_q[i*VW +: VW];
      obs_mask_r[i] = obs_mask_q[i*BP_OBS +: BP_OBS];
    end
    for (int b = 0; b < NVB; b++) begin
      var_epres_r[b] = var_epres_q[b];
      var_ebsel_r[b] = var_ebsel_q[b*EBW +: EBW];
      var_eport_r[b] = var_eport_q[b];
      var_erow_r[b]  = var_erow_q[b*BWC +: BWC];
      scat_pres_r[b] = scat_pres_q[b];
      scat_hb_r[b]   = scat_hb_q[b*HBW +: HBW];
      scat_row_r[b]  = scat_row_q[b*BWC +: BWC];
      scat_lam_r[b]  = scat_lam_q[b*MSG_BITS +: MSG_BITS];
    end
  end

  // ------------------------------------------------------------------- shared comb: read/write cursors
  // Gather ROMs are addressed by the M8 GATHER cursor (pc); their registered output aligns with the
  // +1-delayed launch, and the message-read address is registered by 1 to meet it. Scatter ROMs are
  // addressed by the M8 WRITE cursor (INIT:pc / VAR:pc-3); their registered output aligns with the
  // +1-bumped write cursor (INIT:pc-1 / VAR:pc-4). All addresses clamped in-range (out-of-phase reads
  // are gated unused downstream).
  int chk_rd, var_rd, obs_rd, gam_rd, scat_rd_i, scat_rd, ecmw_rd_i, ecmw_rd;
  int wa_ecm_i, wa_mvm_i;
  logic ecm_we_gate, mvm_we_gate, mcm_we_gate, scat_is_init;
  always_comb begin
    chk_rd    = (pc >= 0 && pc < GC) ? pc : 0;
    var_rd    = (pc >= 0 && pc < GV) ? pc : 0;
    obs_rd    = (pc >= 0 && pc < GV) ? pc : 0;
    gam_rd    = leg * GV + var_rd;                       // leg in [0,LEGS), var_rd in [0,GV)
    scat_rd_i = (state == S_INIT) ? pc : (pc - 3);
    scat_rd   = (scat_rd_i >= 0 && scat_rd_i < GV) ? scat_rd_i : 0;
    ecmw_rd_i = pc - 4;
    ecmw_rd   = (ecmw_rd_i >= 0 && ecmw_rd_i < GC) ? ecmw_rd_i : 0;

    wa_ecm_i     = pc - 5;
    wa_ecm       = (wa_ecm_i >= 0 && wa_ecm_i < GC) ? BWC'(wa_ecm_i) : '0;
    wa_mvm_i     = (state == S_INIT) ? (pc - 1) : (pc - 4);
    wa_mvm       = (wa_mvm_i >= 0 && wa_mvm_i < GV) ? BWV'(wa_mvm_i) : '0;
    scat_is_init = (state == S_INIT);
    ecm_we_gate  = (state == S_CHECK) && (pc >= 5);
    mvm_we_gate  = ((state == S_INIT) && (pc >= 1)) || ((state == S_VAR) && (pc >= 4));
    mcm_we_gate  = ((state == S_INIT) && (pc >= 1)) || ((state == S_VAR) && (pc >= 4));
  end

  // ------------------------------------------------------------------- sync ROM reads + read-addr registers
  // ONE whole-row sync read per ROM per cycle (the BRAM-inference template); per-slot fields are sliced
  // combinationally from the `_q` row registers above.
  always_ff @(posedge clk) begin
    chk_sbit_idx_q  <= chk_sbit_idx_rom[chk_rd];
    chk_sbit_pres_q <= chk_sbit_pres_rom[chk_rd];
    chk_hbsel_q     <= chk_hbsel_rom[chk_rd];
    chk_epres_q     <= chk_epres_rom[chk_rd];
    ecm_wpres_q     <= ecm_wpres_rom[ecmw_rd];
    var_pres_q      <= var_pres_rom[var_rd];
    var_lam_q       <= var_lam_rom[var_rd];
    var_gam_q       <= var_gam_rom[gam_rd];
    var_epres_q     <= var_epres_rom[var_rd];
    var_ebsel_q     <= var_ebsel_rom[var_rd];
    var_eport_q     <= var_eport_rom[var_rd];
    var_erow_q      <= var_erow_rom[var_rd];
    scat_pres_q     <= scat_pres_rom[scat_rd];
    scat_hb_q       <= scat_hb_rom[scat_rd];
    scat_row_q      <= scat_row_rom[scat_rd];
    scat_lam_q      <= scat_lam_rom[scat_rd];
    obs_pres_q      <= obs_pres_rom[obs_rd];
    obs_var_q       <= obs_var_rom[obs_rd];
    obs_mask_q      <= obs_mask_rom[obs_rd];
    mcm_ra_r <= BWC'(chk_rd);
    mvm_ra_r <= BWV'(var_rd);
  end

  // ------------------------------------------------------------------- comb: m_cm write scatter (from ROM)
  // For each present (i,d) of the registered scatter group: drive its half-bank's single write port. Row and
  // half-bank from the ROM; data = lambda-seed (S_INIT) or the var-update output (S_VAR). Guard(a) => <=1
  // writer per half-bank per group, so no scatter conflict.
  always_comb begin
    for (int b = 0; b < NHB; b++) begin
      we_mcm[b] = 1'b0;
      wa_mcm[b] = '0;
      wd_mcm[b] = '0;
    end
    if (mcm_we_gate) begin
      for (int i = 0; i < V; i++)
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int idx = i * BP_VAR_DEG + d;
          if (scat_pres_r[idx]) begin
            we_mcm[scat_hb_r[idx]] = 1'b1;
            wa_mcm[scat_hb_r[idx]] = scat_row_r[idx];
            wd_mcm[scat_hb_r[idx]] = scat_is_init ? signed'(scat_lam_r[idx]) : var_m_out[i][d];
          end
        end
    end
  end

  // ------------------------------------------------------------------- comb: e_cm read addresses (from ROM)
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = '0;
      rb_ecm[b] = '0;
    end
    if (state == S_VAR) begin
      for (int i = 0; i < V; i++)
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int idx = i * BP_VAR_DEG + d;
          if (var_epres_r[idx]) begin
            if (var_eport_r[idx]) rb_ecm[var_ebsel_r[idx]] = var_erow_r[idx];
            else                  ra_ecm[var_ebsel_r[idx]] = var_erow_r[idx];
          end
        end
    end
  end

  // ------------------------------------------------------------------- comb: e_cm / m_vm write enables+data
  always_comb begin
    for (int b = 0; b < NEB; b++) we_ecm[b] = ecm_wpres_r[b] & ecm_we_gate;
    for (int b = 0; b < NVB; b++) begin
      we_mvm[b] = scat_pres_r[b] & mvm_we_gate;
      wd_mvm[b] = scat_is_init ? signed'(scat_lam_r[b]) : var_m_flat[b];
    end
  end

  // ------------------------------------------------------------------- thin glue: submodule outputs -> buses
  generate
    for (genvar j = 0; j < W; j++) begin : gceflat
      for (genvar k = 0; k < BP_CHK_DEG; k++) begin : gceflat_k
        assign chk_e_flat[j * BP_CHK_DEG + k] = chk_e_out[j][k];
      end
    end
    for (genvar i = 0; i < V; i++) begin : gvmflat
      for (genvar d = 0; d < BP_VAR_DEG; d++) begin : gvmflat_d
        assign var_m_flat[i * BP_VAR_DEG + d] = var_m_out[i][d];
      end
    end
  endgenerate

  // ===================================================================== m_cm half-bank cells
  generate
    for (genvar b = 0; b < NHB; b++) begin : gmcm
      bp_mcm_cell_bq #(.B(b)) u_mcm (
          .clk(clk), .we(we_mcm[b]), .wa(wa_mcm[b]), .wd(wd_mcm[b]), .ra(mcm_ra_r), .q(qmcm[b])
      );
    end
  endgenerate

  // ===================================================================== e_cm bank cells
  generate
    for (genvar b = 0; b < NEB; b++) begin : gecm
      bp_ecm_cell_bq #(.B(b)) u_ecm (
          .clk(clk), .we(we_ecm[b]), .wa(wa_ecm), .wd(chk_e_flat[b]),
          .ra(ra_ecm[b]), .rb(rb_ecm[b]), .qa(qa_ecm[b]), .qb(qb_ecm[b])
      );
    end
  endgenerate

  // ===================================================================== m_vm bank cells
  generate
    for (genvar b = 0; b < NVB; b++) begin : gmvm
      bp_mvm_cell_bq #(.B(b)) u_mvm (
          .clk(clk), .we(we_mvm[b]), .wa(wa_mvm), .wd(wd_mvm[b]), .rg(mvm_ra_r), .q(qmvm[b])
      );
    end
  endgenerate

  // ===================================================================== W check_minsum slots
  generate
    for (genvar j = 0; j < W; j++) begin : gchk
      logic                       sbit_j;
      logic signed [MSG_BITS-1:0] m_in_j    [BP_CHK_DEG];
      logic                       present_j [BP_CHK_DEG];
      // gather from the REGISTERED CHK select ROM (group = pc-1) tapping the REGISTERED-address m_cm reads.
      always_comb begin
        sbit_j = chk_sbit_pres_r[j] ? s_reg[chk_sbit_idx_r[j]] : 1'b0;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          automatic int idx = j * BP_CHK_DEG + k;
          m_in_j[k]    = chk_epres_r[idx] ? qmcm[chk_hbsel_r[idx]] : '0;
          present_j[k] = chk_epres_r[idx];
        end
      end
      // M8 gather register plane (retained): latch the gathered launch context one cycle. Free-running;
      // en_chk_rr fences the pre-launch garbage snapshots out.
      logic                       sbit_jr;
      logic signed [MSG_BITS-1:0] m_in_jr    [BP_CHK_DEG];
      logic                       present_jr [BP_CHK_DEG];
      always_ff @(posedge clk) begin
        sbit_jr <= sbit_j;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          m_in_jr[k]    <= m_in_j[k];
          present_jr[k] <= present_j[k];
        end
      end
      check_minsum #(
          .MW    (MSG_BITS),
          .DEG   (BP_CHK_DEG),
          .STAGES(3)
      ) u_chk (
          .clk    (clk),
          .en     (en_chk_rr),
          .sbit   (sbit_jr),
          .m_in   (m_in_jr),
          .present(present_jr),
          .e_out  (chk_e_out[j])
      );
    end
  endgenerate

  // ===================================================================== V var_update slots
  generate
    for (genvar i = 0; i < V; i++) begin : gvar
      logic signed [MSG_BITS-1:0] lam_i, gam_i;
      logic signed [MSG_BITS-1:0] e_in_i    [BP_VAR_DEG];
      logic signed [MSG_BITS-1:0] m_in_i    [BP_VAR_DEG];
      logic                       present_i [BP_VAR_DEG];
      // gather from the REGISTERED VAR select ROM (group = pc-1): e_cm operand (port-selected, ROM-addressed
      // async read) + the "old" m_vc from m_vm (registered-address read) + lambda/gamma from their ROMs.
      always_comb begin
        lam_i = var_pres_r[i] ? signed'(var_lam_r[i]) : '0;
        gam_i = var_pres_r[i] ? signed'(var_gam_r[i]) : '0;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int idx = i * BP_VAR_DEG + d;
          if (var_epres_r[idx]) begin
            e_in_i[d]    = var_eport_r[idx] ? qb_ecm[var_ebsel_r[idx]] : qa_ecm[var_ebsel_r[idx]];
            m_in_i[d]    = qmvm[idx];
            present_i[d] = 1'b1;
          end else begin
            e_in_i[d]    = '0;
            m_in_i[d]    = '0;
            present_i[d] = 1'b0;
          end
        end
      end
      // M8 gather register plane (retained).
      logic signed [MSG_BITS-1:0] lam_ir, gam_ir;
      logic signed [MSG_BITS-1:0] e_in_ir    [BP_VAR_DEG];
      logic signed [MSG_BITS-1:0] m_in_ir    [BP_VAR_DEG];
      logic                       present_ir [BP_VAR_DEG];
      always_ff @(posedge clk) begin
        lam_ir <= lam_i;
        gam_ir <= gam_i;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          e_in_ir[d]    <= e_in_i[d];
          m_in_ir[d]    <= m_in_i[d];
          present_ir[d] <= present_i[d];
        end
      end
      var_update #(
          .MW    (MSG_BITS),
          .WACC  (WACC),
          .FRAC  (FRAC_BITS),
          .DEG   (BP_VAR_DEG),
          .MAXMAG(MAX_MAG)
      ) u_var (
          .clk     (clk),
          .en      (en_var_rr),
          .lam     (lam_ir),
          .gam     (gam_ir),
          .e_in    (e_in_ir),
          .m_in    (m_in_ir),
          .present (present_ir),
          .m_out   (var_m_out[i]),
          .ehat_bit(var_ehat_out[i])
      );
    end
  endgenerate

  // ===================================================================================== control FSM
  always_ff @(posedge clk) begin                     // synchronous reset (Synth 8-7137)
    if (!rst_n) begin
      state      <= S_IDLE;
      out_valid  <= 1'b0;
      valid_flag <= 1'b0;
      lat        <= '0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        // ----------------------------------------------------------------- accept syndrome + init flops
        S_IDLE: begin
          if (in_valid) begin
            for (int c = 0; c < BP_C; c++) s_reg[c] <= syndrome_in[c];
            for (int v = 0; v < BP_N; v++) ehat[v]  <= 1'b0;
            found       <= 1'b0;
            best_w      <= '1;
            ehat_w      <= '0;
            all_sat     <= 1'b1;
            sat_pending <= 1'b0;
            leg <= '0; iter <= '0; pc <= '0;
            lat <= '0;
            state <= S_INIT;
          end
        end

        // ----------------------------------- seed m_cm/m_vm with lambda (M9b: pc=0..GV, write at pc-1)
        S_INIT: begin
          if (pc == GV) begin pc <= '0; state <= S_CHECK; end
          else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch check group `pc-1` + scatter group pc-5 (M9b)  || S_SAT
        S_CHECK: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          // R5 (UNCHANGED): overlapped SAT parity of the PREVIOUS decision on the launched group's checks;
          // launch-aligned at pc<GC, finalises at pc==GC-1 (LUT wire taps / XOR trees, not ROM-ified).
          if (pc < GC && sat_pending) begin
            for (int j = 0; j < W; j++)
              for (int g = 0; g < GC; g++)
                if (chk_at_bq(g, j) >= 0 && pc == g) begin
                  p = s_reg[chk_at_bq(g, j)];
                  for (int k = 0; k < BP_CHK_DEG; k++) begin
                    automatic int e = edge_at_bq(g, j, k);
                    if (e >= 0) p = p ^ ehat[BP_EDGE_VAR[e]];
                  end
                  if (p != 1'b0) grp_sat = 1'b0;
                end
            if (!grp_sat) all_sat <= 1'b0;
            if (pc == GC - 1) begin
              final_sat = all_sat & grp_sat;
              if (final_sat) begin
                found <= 1'b1;
                if (ehat_w < best_w) begin
                  best_w <= ehat_w;
                  for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
                end
              end
            end
          end
          // e_cm scatter of group pc-5 (M9b lag) handled by the ROM-fed per-bank bp_ecm_cell_bq write.
          if (early_exit && final_sat) begin
            pc <= '0; state <= S_EMIT;
          end else if (pc == GC + 4) begin              // M9b: was GC+3 (extra ROM-read plane +1)
            pc      <= '0;
            all_sat <= 1'b1;
            state   <= S_VAR;
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch var group `pc-1` + scatter group pc-4 (M9b)
        S_VAR: begin
          automatic logic wterm [V];
          automatic int   wsum;
          wsum = 0;
          if (pc == 0) ehat_w <= '0;
          if (pc >= 4) begin                          // M9b: var scatter lag pc-4 (was pc-3)
            for (int i = 0; i < V; i++) wterm[i] = 1'b0;
            for (int i = 0; i < V; i++)
              for (int g = 0; g < GV; g++)
                if (var_at_bq(g, i) >= 0 && (pc - 4) == g) begin
                  automatic int v = var_at_bq(g, i);
                  ehat[v] <= var_ehat_out[i];
                  wterm[i] = var_ehat_out[i];
                end
            for (int i = 0; i < V; i++) wsum = wsum + (wterm[i] ? 1 : 0);
            ehat_w <= ehat_w + WW'(wsum);
          end
          // m_vm / m_cm writes of group pc-4 (M9b lag) handled by the bp_mvm_cell_bq + m_cm scatter comb.
          if (pc == GV + 3) begin                       // M9b: was GV+2 (extra ROM-read plane +1)
            pc          <= '0;
            sat_pending <= 1'b1;
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_SATF;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ----------------------------- trailing SAT for the final decision (R5, UNCHANGED)
        S_SATF: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          for (int j = 0; j < W; j++)
            for (int g = 0; g < GC; g++)
              if (chk_at_bq(g, j) >= 0 && pc == g) begin
                p = s_reg[chk_at_bq(g, j)];
                for (int k = 0; k < BP_CHK_DEG; k++) begin
                  automatic int e = edge_at_bq(g, j, k);
                  if (e >= 0) p = p ^ ehat[BP_EDGE_VAR[e]];
                end
                if (p != 1'b0) grp_sat = 1'b0;
              end
          if (!grp_sat) all_sat <= 1'b0;
          if (pc == GC - 1) begin
            final_sat = all_sat & grp_sat;
            if (final_sat) begin
              found <= 1'b1;
              if (ehat_w < best_w) begin
                best_w <= ehat_w;
                for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
              end
            end
            pc    <= '0;
            state <= S_EMIT;
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat -> obs (M9b)
        // ROM read at pc; accumulate/write the group's slots at pc-1 (obs_*_r hold group pc-1). Per-slot
        // observable term (own constant-folded group in the ROM) then a pure XOR reduction (associative).
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] base;
          automatic logic [BP_OBS-1:0] term [V];
          automatic logic [BP_OBS-1:0] acc;
          base = (pc == 1) ? {BP_OBS{1'b0}} : obs_acc;   // group (pc-1)==0 <=> pc==1: fresh accumulation
          for (int i = 0; i < V; i++) term[i] = {BP_OBS{1'b0}};
          if (pc >= 1) begin
            for (int i = 0; i < V; i++)
              if (obs_pres_r[i]) begin
                automatic int   v  = int'(obs_var_r[i]);
                automatic logic bb = found ? best_e[v] : ehat[v];
                corr_out[v] <= bb;
                if (bb) term[i] = obs_mask_r[i];
              end
            acc = base;
            for (int i = 0; i < V; i++) acc = acc ^ term[i];
            obs_acc <= acc;
          end
          if (pc == GV) begin pc <= '0; state <= S_DONE; end
          else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        S_DONE: begin
          obs_flip   <= obs_acc;
          valid_flag <= found;
          out_valid  <= 1'b1;
          state      <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
