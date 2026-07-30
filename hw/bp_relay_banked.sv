// Q7-02 M7 — K-BANKED relay-BP decoder core (`bp_relay_banked`): beta-split, check-major LUTRAM banks,
// with the message-bank fabric restructured into HIERARCHICALLY-MODULAR STAMPED CELLS (area-opt rescue).
//
// WHY THIS FILE IS SHAPED AS MANY SMALL MODULES (the M7 synthesis post-mortem):
//   The functionally-proven FLAT version of this core (40/40 bit-exact co-sim vs the fixed-point golden at
//   8/24, 12/36, 16/48; worst latency 3570/2460/1905) STALLED Vivado OOC synthesis: all three configs sat
//   >3.5 h single-threaded in "Cross Boundary and Area Optimization" without completing — one flat top-level
//   bank/mux/ROM cloud is the pathological input for that pass. The empirically-proven fix this milestone
//   found (the 1008-instance `bp_unroll_skeleton` cleared the SAME phase in ~3 min where flat cores
//   OOM'd/stalled) is HIERARCHICAL MODULARIZATION: with `synth_design -flatten_hierarchy none`, Vivado
//   optimizes many small stamped module instances independently and cheaply, but chokes on the equivalent
//   flattened cloud. So the LUTRAM message-bank fabric — the 2*W*CHK_DEG m_cm half-banks, the W*CHK_DEG
//   e_cm banks, and the V*VAR_DEG m_vm banks (744 memories at 8/24) — is decomposed into small
//   `parameter int`-stamped cell modules (`bp_mcm_cell`, `bp_ecm_cell`, `bp_mvm_cell`), one instance per
//   memory. This is a PURE-STRUCTURAL refactor of the flat core: identical schedule, quantisation, memory
//   map, and keep-lowest-weight-valid decision — the worst-case latency is bit-for-bit unchanged (3570
//   cycles at 8/24). No behavior moved; only module boundaries were drawn.
//
// SCOPE OF THE MODULARIZATION (a stated, forced deviation from a "5-cell / cell-owns-its-ROM" plan):
//   The per-edge SCATTER/GATHER RESOLUTION (which var-slot writes which half-bank; which e_cm bank each var
//   operand reads at which port) is an O(GV*V*VAR_DEG) sweep over the header graph tables. Pushing it INTO
//   each cell — whether as elaboration-time `localparam` ROMs (constant functions) or as per-cell RTL
//   scans — is infeasible on the Verilator 5.050 co-sim box this gate runs on: constant-function evaluation
//   over the large header arrays (BP_CHECK_EDGES/BP_VAR_AT/... , thousands of elements) is pathologically
//   memory-hungry (even four small reverse-lookup tables cost ~22 s; the full per-cell-ROM set OOM-kills the
//   process at ~35 s on a 16 GB host), and replicating the O(GV*V*VAR_DEG) sweep as RTL across 744 cells
//   likewise blows past memory. So the scatter/read-address RESOLUTION stays in the TOP as the SAME shared
//   combs the flat core used (constant-indexed — every bank id folds to a compile-time constant; the cursor
//   only selects the row/group), and each cell is the MEMORY plus its own mux-free (e_cm/m_vm) or top-fed
//   (m_cm) write port. The 744 memories ARE the LUTRAM fabric the area-opt pass chokes on, and they are now
//   independent hierarchy units. The gathers feeding check_minsum/var_update are pure constant-tap selects
//   of the cells' async-read OUTPUT wires and remain in the TOP (they were never the mux wall).
//
// It still STAMPS the SAME two unit-verified submodules (`check_minsum`, `var_update`, Tasks 1-2). The
// difference from `bp_relay_fast.sv` is WHERE the per-edge messages live: not in flop arrays (which rebuild
// the runtime-index register-file mux wall that stalled Vivado area-opt), but in many small single-write-
// port LUTRAM banks, one message per (bank,row) addressed by a compile-time group/slot map baked into the
// header (Task 9). Every runtime index touches ONLY: (a) the banks' async read OUTPUT wires, tapped at
// compile-time-constant bank ids in the gathers, and (b) small ehat/s_reg flop arrays in the TOP.
//
// MEMORY MAP (spec A2.1):
//   * m_cm (v->c messages, read by the CHK phase): 2*W*CHK_DEG half-banks x GC rows (`bp_mcm_cell`). Half-
//     bank of edge e = BP_EDGE_HB[e] (= EB*2 + beta, EB = slot*CHK_DEG + pos); row = BP_EDGE_ROW[e] (the
//     edge's check group). One sync write port; async read at row = pc. (M7: literal tables, was a scan.)
//   * e_cm (c->v messages, written by CHK, read by VAR): W*CHK_DEG banks x GC rows (`bp_ecm_cell`), same
//     (j,k) map, no beta split, TWO async read address ports (readers ordered by (i,d): first->A, second->B).
//   * m_vm (the "old" v->c message the var-update blends against): V*VAR_DEG banks x GV rows (`bp_mvm_cell`).
//     Bank = (var-slot i, edge d); read row = pc, write row = pc-3 (M8; disjoint in the software pipeline).
//
// M8 PIPELINE DEEPENING (this milestone): a 1-cycle BANK-READ + LAUNCH-CONTEXT register plane is inserted
// between the (unregistered, pc-addressed) bank async reads / gather muxes and the submodule input ports,
// and `check_minsum` runs at STAGES=3 (one extra mid-tree register plane). Net per-launch latency grows by
// 2 for CHK (reg plane +1, STAGES 2->3 +1) and by 1 for VAR (reg plane +1; var_update stays 2-stage). So
// the SOFTWARE-PIPELINE lags lengthen: CHK scatter of group pc-4 (was pc-2), phase pc=0..GC+3 (was GC+1);
// VAR scatter of group pc-3 (was pc-2), phase pc=0..GV+2 (was GV+1). Bank read ADDRESSES stay pc-driven;
// only the read DATA + the gathered per-slot launch context (sbit/present/m_in on CHK; lam/gam/e_in/m_in/
// present on VAR) latch through the plane, and the submodule `en` is delayed one cycle (en_chk_r/en_var_r)
// so the plane's pc=0 garbage snapshot is never consumed. Values are BIT-EXACT to M7; only cycle counts
// grow (worst latency 3570 -> 3750 at 8/24: +60*(2+1) over 60 CHK + 60 VAR phases).
//
// FSM mirrors `bp_relay_unroll_pipe.sv` (S_CHECK/S_VAR 2-deep software pipeline over the submodules' 2-cycle
// latency; SAT folded into the S_CHECK launches; best-kept commit; obs reduction; sync reset) with the
// banked additions: (1) an S_INIT state seeding m_cm/m_vm with lambda; (2) an `early_exit` input (first
// syndrome-valid decision jumps to S_EMIT); (3) a 32-bit `latency_cycles` output. W/V/GC/GV come from the
// header, never module parameters, so header and RTL cannot desync.
//
// Q7-08 ASIC REGFILE RESTRUCTURING (`BP_RF_REGFILE`, opt-in define; default = the DFF baseline above):
//   The SKY130 ORFS probe showed the flat-DFF message fabric is ROUTE-INFEASIBLE (GRT met2 overflow at 30%
//   AND 20% core utilization — see docs/perf/qec-q7-asic-sky130-probe.md): every 8-bit message row carries
//   its own private write-decoder AND read-mux wiring, and the resulting met2 demand exceeds the fabric at
//   any sane density. `BP_RF_REGFILE` restructures ONLY the storage inside the cells, per access pattern:
//   * m_vm: the VAR_DEG per-(i,d) banks of each var slot collapse into ONE wide byte-masked array
//     (`bp_mvm_rf`, mem[GV] rows of VAR_DEG*MSG_BITS). All lanes ALREADY share the global write/read
//     cursors (wg_var/mvm_ra); the per-cell present gate `vedge_at(h,II,DD)>=0` moves up into a per-lane
//     write mask. One row decoder per slot instead of VAR_DEG private ones — area-neutral, routing win.
//   * m_cm / e_cm: stay per-(half-)bank cells (m_cm's write row is per-cell from the scatter; e_cm has two
//     fabric-driven read rows — nothing to share), but storage becomes a TRANSPARENT-LOW LATCH array
//     (2.54x denser than DFF in sky130_fd_sc_hd: 15.1 vs 38.4 um^2/bit, measured on the 864-bit slice).
//     Write-pulse discipline (latch opens during clk-low; wd/wa must be settled) is a known physical-design
//     caveat, acceptable for area/routability evidence; reads and writes of these arrays happen in
//     DIFFERENT FSM states, so the half-cycle-early latch write is invisible to the schedule (bit-exact).
//   Values are BIT-EXACT to the DFF baseline in both styles; the co-sim gate runs both.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

// ================================================================ $unit-scope geometry (shared by cells)
// Hoisted out of `bp_relay_banked` so the sibling cell modules below can size their ports/memories against
// the same header-derived geometry.
localparam int BB_GC   = BP_GC;                       // number of check groups (m_cm / e_cm rows)
localparam int BB_GV   = BP_GV;                       // number of var groups   (m_vm rows)
// Q7-02 Task B0 Option A: row-address widths are FLOORED AT 1 bit. `$clog2(1) == 0`, and a zero-width
// vector (`logic [-1:0]`) is illegal SV — which is what made every single-group geometry unelaboratable,
// including the full-parallel 144/864 target (GC = GV = 1). The stored address is then always 0, so the
// floor costs one always-zero bit at GC/GV = 1 and is a no-op for every multi-group geometry. Mirrored
// by `row_addr_width` in the emitter (crates/aleph-qec/examples/qec_q7_bp_graph.rs), which sizes the
// matching BP_ROM_* row fields.
localparam int BB_BWC  = (BP_GC <= 1) ? 1 : $clog2(BP_GC);   // m_cm / e_cm row address width
localparam int BB_BWV  = (BP_GV <= 1) ? 1 : $clog2(BP_GV);   // m_vm row address width

/* verilator lint_off UNUSEDSIGNAL */
// ============================================================ $unit elaboration helpers over header tables
// Hoisted to compilation-unit scope so the sibling cell modules can call them (functions defined INSIDE
// bp_relay_banked are not visible to siblings). Called only with compile-time-constant arguments (genvar /
// cell parameters, or a group index gated to a constant by `pc == g` / `wg == h`), so every use constant-
// folds into a fixed bank id / row / constant — never a runtime index into the graph tables. These are the
// SAME bodies the flat core used inline; keeping them as ordinary RTL helper functions (NOT elaboration-time
// constant functions building localparams) is deliberate — the latter OOM-kills Verilator over the large
// header arrays (see the SCOPE note in the file header).
function automatic int chk_at(input int g, input int j);
  return BP_CHK_AT[g * BP_BANK_W + j];
endfunction
function automatic int var_at(input int h, input int i);
  return BP_VAR_AT[h * BP_BANK_V + i];
endfunction
function automatic int chk_deg(input int c);
  return BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c];
endfunction
function automatic int var_deg(input int v);
  return BP_VAR_OFF[v + 1] - BP_VAR_OFF[v];
endfunction
// edge index at (check-group g, slot j, position k), or -1 if empty slot / k >= that check's degree.
function automatic int edge_at(input int g, input int j, input int k);
  automatic int c = chk_at(g, j);
  if (c < 0) return -1;
  if (k >= chk_deg(c)) return -1;
  return BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];       // edge whose EDGE_POS == k (verified in TB step)
endfunction
// edge index at (var-group h, slot i, edge d), or -1 if empty slot / d >= that var's degree.
function automatic int vedge_at(input int h, input int i, input int d);
  automatic int v = var_at(h, i);
  if (v < 0) return -1;
  if (d >= var_deg(v)) return -1;
  return BP_VAR_OFF[v] + d;                         // var edges are variable-major contiguous
endfunction
// M7 Vivado-folding rescue: the former scan-loop resolvers `grp_of_chk`/`slot_of_chk` (and the helpers
// built on them — `ecm_bank`/`hb_of_edge`/`ecm_port`) have been REPLACED by direct literal-table indexing
// of the emitter-baked inverses `BP_CHK_GRP`/`BP_CHK_SLOT`/`BP_EDGE_EB`/`BP_EDGE_HB`/`BP_EDGE_ROW`/
// `BP_EDGE_EPORT`. Verilator const-folded the old scans, but Vivado materialised each call site as a
// hardware scanner + mux cloud (~386k LUT / 747 DSP / 1445 CARRY8 top-level whale). The `ifndef SYNTHESIS`
// elaboration guard below re-derives grp/slot/EB/HB/ROW/EPORT by SCANNING (simulation-only) and asserts
// the literal tables match — so the guard's protection is preserved with zero scans on synthesised paths.
/* verilator lint_on UNUSEDSIGNAL */

// ============================================================================ STAMPED BANK CELL MODULES
// One instance per (half-bank | bank), so `-flatten_hierarchy none` keeps each memory as an independent
// area-opt hierarchy unit instead of one flat cloud. m_cm is a thin memory driven by the top's shared
// scatter (its write-select is the constant-indexed sweep that cannot be per-cell-ized on this co-sim box —
// see the file header); e_cm and m_vm carry their own MUX-FREE write ports (a single writer keyed on the
// bank's own compile-time (j,k)/(i,d) coordinates, cheap in RTL). DECLFILENAME is fenced: multiple modules
// share this file so Vivado optimizes the bank fabric per cell.
/* verilator lint_off DECLFILENAME */

// --------------------------------------------------------------- m_cm half-bank: 1 sync write, 1 async read
// B is identity-only (this cell is a pure port-driven memory — its write-select is the top's shared
// scatter), so its parameter is intentionally unread; fence UNUSEDPARAM for it alone.
/* verilator lint_off UNUSEDPARAM */
module bp_mcm_cell #(
    parameter int B = 0                                  // half-bank id (identity only; wiring via ports)
) (
    input  logic                       clk,
    input  logic                       we,               // from the top's shared m_cm write scatter
    input  logic [BB_BWC-1:0]          wa,               // write row (= edge's CHECK group)
    input  logic signed [MSG_BITS-1:0] wd,               // write data (lambda in S_INIT, var m_out in S_VAR)
    input  logic [BB_BWC-1:0]          ra,               // read row (= pc, clamped)
    output logic signed [MSG_BITS-1:0] q
);
  logic signed [MSG_BITS-1:0] mem [BB_GC];
`ifdef BP_RF_REGFILE
  // Q7-08: transparent-low latch rows (sky130 dlxtp-class, 2.54x denser than DFF). we/wa/wd come from the
  // top's scatter comb (registers + registered submodule outputs), settled well before the clk-low pulse.
  // Blocking `=` (not `<=`): Verilator executes always_latch combinationally and would silently convert
  // the non-blocking form anyway (COMBDLY); with a single writer per row the two are equivalent here.
  always_latch
    for (int i = 0; i < BB_GC; i++)
      if (!clk && we && wa == BB_BWC'(i)) mem[i] = wd;
`else
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
`endif
  assign q = mem[ra];
endmodule
/* verilator lint_on UNUSEDPARAM */

// --------------------------------------------------------------- e_cm bank: 1 sync write, 2 async reads
module bp_ecm_cell #(
    parameter int B = 0                                  // bank id = check-slot j * CHK_DEG + lane k
) (
    input  logic                       clk,
    input  logic                       wr_en,            // S_CHECK && pc>=4 (M8: scatter of group pc-4)
    input  logic [BB_BWC-1:0]          wg,               // write chk-group cursor (M8: = pc-4)
    input  logic signed [MSG_BITS-1:0] wd,               // this bank's chk lane message (chk_e_out[JJ][KK])
    input  logic [BB_BWC-1:0]          ra,               // port-A read row (from top's e_cm read-addr comb)
    input  logic [BB_BWC-1:0]          rb,               // port-B read row
    output logic signed [MSG_BITS-1:0] qa,
    output logic signed [MSG_BITS-1:0] qb
);
  localparam int JJ = B / BP_CHK_DEG;                    // check slot j
  localparam int KK = B % BP_CHK_DEG;                    // lane / position k
  logic signed [MSG_BITS-1:0] mem [BB_GC];
  logic                       we_b;
  logic [BB_BWC-1:0]          wa_b;
  // CHK scatter: lane (JJ,KK) of group pc-4 (M8 lag) writes its OWN bank — a single writer, no index mux
  // (JJ/KK are compile-time constants, so edge_at(g,JJ,KK) folds per group).
  always_comb begin
    we_b = 1'b0;
    wa_b = '0;
    if (wr_en)
      for (int g = 0; g < BB_GC; g++)
        if (wg == BB_BWC'(g) && edge_at(g, JJ, KK) >= 0) begin
          we_b = 1'b1;
          wa_b = BB_BWC'(g);
        end
  end
`ifdef BP_RF_REGFILE
  // Q7-08: latch-array storage (same rationale/discipline as bp_mcm_cell; we_b/wa_b are combs of the
  // registered wg cursor, wd is a registered check_minsum output — all settled before the clk-low pulse).
  // Blocking `=` for the same COMBDLY reason as bp_mcm_cell.
  always_latch
    for (int i = 0; i < BB_GC; i++)
      if (!clk && we_b && wa_b == BB_BWC'(i)) mem[i] = wd;
`else
  always_ff @(posedge clk) if (we_b) mem[wa_b] <= wd;
`endif
  assign qa = mem[ra];
  assign qb = mem[rb];
endmodule

// --------------------------------------------------------------- m_vm bank: 1 sync write, 1 async read
module bp_mvm_cell #(
    parameter int B = 0                                  // bank id = var-slot i * VAR_DEG + edge d
) (
    input  logic                       clk,
    input  logic                       wr_init,          // S_INIT (lambda seed)
    input  logic                       wr_var,           // S_VAR && pc>=3 (M8: scatter of group pc-3)
    input  logic [BB_BWV-1:0]          wg,               // write var-group cursor (S_INIT: pc, S_VAR: pc-3)
    input  logic signed [MSG_BITS-1:0] wd,               // this bank's var slot message (var_m_out[II][DD])
    input  logic [BB_BWV-1:0]          rg,               // read row (= pc, clamped)
    output logic signed [MSG_BITS-1:0] q
);
  localparam int II = B / BP_VAR_DEG;                    // var slot i
  localparam int DD = B % BP_VAR_DEG;                    // edge d
  logic signed [MSG_BITS-1:0] mem [BB_GV];
  logic                       we_b;
  logic [BB_BWV-1:0]          wa_b;
  logic signed [MSG_BITS-1:0] wd_b;
  // written by its own var slot (mux-free): row = write group wg; data = var-update output (or lambda).
  always_comb begin
    we_b = 1'b0;
    wa_b = '0;
    wd_b = wd;
    if (wr_init || wr_var)
      for (int h = 0; h < BB_GV; h++)
        if (wg == BB_BWV'(h) && vedge_at(h, II, DD) >= 0) begin
          we_b = 1'b1;
          wa_b = BB_BWV'(h);
          if (wr_init) wd_b = signed'(BP_LAMBDA[var_at(h, II)][MSG_BITS-1:0]);
        end
  end
  always_ff @(posedge clk) if (we_b) mem[wa_b] <= wd_b;
  assign q = mem[rg];
endmodule

`ifdef BP_RF_REGFILE
// ------------------------------------------- m_vm wide slot bank (Q7-08): VAR_DEG lanes, shared 1W1R rows
// Replaces the VAR_DEG per-(i,d) `bp_mvm_cell` banks of var slot II with ONE wide byte-masked array. The
// lanes ALREADY share the write cursor (wg) and read cursor (rg) every cycle — the only per-(i,d) term in
// the old cells was the `vedge_at(h,II,DD)>=0` present gate, which moves up into the per-lane write mask.
// One row decoder per slot instead of VAR_DEG private ones: area-neutral, the win is write/read wiring.
module bp_mvm_rf #(
    parameter int II = 0                                 // var slot i (lanes d = 0..VAR_DEG-1)
) (
    input  logic                       clk,
    input  logic                       wr_init,          // S_INIT (lambda seed)
    input  logic                       wr_var,           // S_VAR && pc>=3 (M8: scatter of group pc-3)
    input  logic [BB_BWV-1:0]          wg,               // write var-group cursor (S_INIT: pc, S_VAR: pc-3)
    input  logic signed [MSG_BITS-1:0] wd [BP_VAR_DEG],  // this slot's var messages (var_m_out[II])
    input  logic [BB_BWV-1:0]          rg,               // read row (= pc, clamped)
    output logic signed [MSG_BITS-1:0] q  [BP_VAR_DEG]
);
  logic [BP_VAR_DEG*MSG_BITS-1:0] mem [BB_GV];
  logic [BP_VAR_DEG-1:0]          we_lane;
  logic [BP_VAR_DEG*MSG_BITS-1:0] wd_row;
  // per-lane write mask: lane d of row h is written iff the slot has a present edge d in group h (the old
  // per-cell enable, hoisted). Data = var-update output, or lambda during the S_INIT seed.
  always_comb begin
    we_lane = '0;
    for (int d = 0; d < BP_VAR_DEG; d++) wd_row[d*MSG_BITS+:MSG_BITS] = wd[d];
    if (wr_init || wr_var)
      for (int h = 0; h < BB_GV; h++)
        if (wg == BB_BWV'(h))
          for (int d = 0; d < BP_VAR_DEG; d++)
            if (vedge_at(h, II, d) >= 0) begin
              we_lane[d] = 1'b1;
              if (wr_init) wd_row[d*MSG_BITS+:MSG_BITS] = BP_LAMBDA[var_at(h, II)][MSG_BITS-1:0];
            end
  end
  // we_lane is only set when wg matched a row index < BB_GV, so mem[wg] is always in range when written.
  always_ff @(posedge clk)
    for (int d = 0; d < BP_VAR_DEG; d++)
      if (we_lane[d]) mem[wg][d*MSG_BITS+:MSG_BITS] <= wd_row[d*MSG_BITS+:MSG_BITS];
  for (genvar d = 0; d < BP_VAR_DEG; d++) begin : gq
    assign q[d] = signed'(mem[rg][d*MSG_BITS+:MSG_BITS]);
  end
endmodule
`endif
/* verilator lint_on DECLFILENAME */

// ========================================================================================= TOP CORE
module bp_relay_banked (
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
  localparam int NEB  = BP_BANK_W * BP_CHK_DEG;      // e_cm banks
  localparam int NVB  = BP_BANK_V * BP_VAR_DEG;      // m_vm banks
  localparam int BWC  = BB_BWC;                      // m_cm / e_cm row address width (1-bit floor)
  localparam int BWV  = BB_BWV;                      // m_vm row address width (1-bit floor)

  /* verilator lint_off UNUSEDSIGNAL */
  // ================================================================== elaboration guards (Task-10 review)
  // The banked datapath silently depends on offline (Task-9 emitter) guarantees. If a future emitter split
  // ever violates one, the RTL corrupts messages with NO other symptom: (a) a second writer on a half-bank's
  // single write port would last-write-win; (b) a third reader of an e_cm bank has no port and is dropped;
  // (c) a wrong BP_EDGE_POS mis-taps the m_cm half-bank in the CHK gather. Recompute and enforce all of these
  // at elaboration (time-0 `initial`, constant-folded over the header tables — no runtime hardware; an initial
  // block of system tasks synthesises to nothing). Fail LOUDLY on any violation.
  //
  // M7 addition (d): the synthesised paths now index the emitter-baked literal resolution tables
  // (BP_CHK_GRP/SLOT, BP_EDGE_EB/HB/ROW/EPORT) instead of scanning BP_CHK_AT at elaboration. To keep the
  // guard's protection, this block RE-DERIVES those maps by scanning (scans are fine here — simulation-only,
  // never synthesised) and asserts every literal table matches the scan. So a bad emitter table now fails the
  // co-sim gate LOUDLY rather than silently mis-routing a bank. (a)/(b)/EPORT then reuse the scan-validated
  // BP_EDGE_HB/EB, and EPORT is validated as "readers-so-far of the edge's e_cm bank in (i,d) order".
  // Fenced from synthesis: Vivado's handling of $fatal-in-initial is not a documented guarantee, and the
  // guard's job is done in the Verilator co-sim gate that always precedes any synth run of this core.
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
        $display("bp_relay_banked GUARD(d1) FAIL: check %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
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
        $display("bp_relay_banked GUARD(d2) FAIL: var %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
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
        $display("bp_relay_banked GUARD(d3) FAIL: edge %0d EB/HB/ROW (%0d,%0d,%0d) != recompute (%0d,%0d,%0d)",
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
          automatic int e = vedge_at(h, i, d);
          if (e >= 0) begin
            automatic int hb = BP_EDGE_HB[e];
            automatic int eb = BP_EDGE_EB[e];
            // EPORT is the count of same-e_cm-bank readers of this group seen BEFORE e in (i,d) order.
            if (BP_EDGE_EPORT[e] != rcnt[eb]) begin
              $display("bp_relay_banked GUARD(eport) FAIL: var-group %0d edge %0d EPORT=%0d != readers-so-far %0d",
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
          $display("bp_relay_banked GUARD(a) FAIL: var-group %0d m_cm half-bank %0d has %0d writers (>1)",
                   h, b, wcnt[b]);
          fails = fails + 1;
        end
      // (b) <=2 readers per (var-group, e_cm bank) — the two async read ports per e_cm bank.
      for (int b = 0; b < NEB; b++)
        if (rcnt[b] > 2) begin
          $display("bp_relay_banked GUARD(b) FAIL: var-group %0d e_cm bank %0d has %0d readers (>2)",
                   h, b, rcnt[b]);
          fails = fails + 1;
        end
    end
    // (c) BP_EDGE_POS[e] is e's position in its check's CSR row (edge_at / BP_EDGE_HB tap correctness).
    for (int e = 0; e < BP_E; e++) begin
      automatic int c   = BP_EDGE_CHK[e];
      automatic int idx = BP_CHECK_OFF[c] + BP_EDGE_POS[e];
      if (idx >= BP_CHECK_OFF[c + 1] || BP_CHECK_EDGES[idx] != e) begin
        $display("bp_relay_banked GUARD(c) FAIL: edge %0d (check %0d) EDGE_POS=%0d does not match CSR row",
                 e, c, BP_EDGE_POS[e]);
        fails = fails + 1;
      end
    end
    if (fails != 0)
      $fatal(1, "bp_relay_banked: %0d elaboration-guard violation(s) — header/emitter split is unsafe", fails);
    else
      $display("bp_relay_banked: elaboration guards (a/b/c/d) PASS (GV=%0d NHB=%0d NEB=%0d BP_E=%0d)",
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
  int          wg;                                   // write group (comb): pc in S_INIT, pc-3 in S_VAR (M8)
  logic [31:0] lat;

  assign busy           = (state != S_IDLE);
  assign latency_cycles = lat;

  // submodule launch enables (only while groups remain to start)
  logic en_chk, en_var;
  assign en_chk = (state == S_CHECK) && (pc < GC);
  assign en_var = (state == S_VAR)   && (pc < GV);

  // M8: the bank-read + launch-context register plane (below, in the gchk/gvar slots) delays each launch's
  // operands by exactly one cycle, so the submodule `en` is delayed one cycle to stay aligned. This is also
  // the fence that keeps the plane's first (pc=0) garbage snapshot from being consumed: en_chk/en_var are 0
  // outside their phase (and at the phase's first cycle the registered copy still holds that 0), so the
  // submodule only captures once the plane holds a real group's operands. Free-running, mirroring the plane.
  logic en_chk_r, en_var_r;
  always_ff @(posedge clk) begin
    en_chk_r <= en_chk;
    en_var_r <= en_var;
  end

  // registered submodule outputs (2 clocks after their group launch)
  logic signed [MSG_BITS-1:0] chk_e_out    [W][BP_CHK_DEG];
  logic signed [MSG_BITS-1:0] var_m_out    [V][BP_VAR_DEG];
  logic                       var_ehat_out [V];

  // bank async-read output wires (produced by the stamped cells, tapped by CONSTANT bank id in the gathers)
  logic signed [MSG_BITS-1:0] qmcm   [NHB];
  logic signed [MSG_BITS-1:0] qa_ecm [NEB];
  logic signed [MSG_BITS-1:0] qb_ecm [NEB];
  logic signed [MSG_BITS-1:0] qmvm   [NVB];
  logic [BWC-1:0]             mcm_ra;                 // uniform m_cm read row (= pc, clamped)
  logic [BWV-1:0]             mvm_ra;                 // uniform m_vm read row (= pc, clamped)
  logic [BWC-1:0]             ra_ecm [NEB];           // per-bank e_cm port-A read row
  logic [BWC-1:0]             rb_ecm [NEB];           // per-bank e_cm port-B read row

  // m_cm write drivers (shared scatter; each half-bank has <=1 writer per write-group — guard(a))
  logic                       we_mcm [NHB];
  logic [BWC-1:0]             wa_mcm [NHB];
  logic signed [MSG_BITS-1:0] wd_mcm [NHB];

  // flattened submodule-output buses feeding the cells' write-data ports (thin glue)
  logic signed [MSG_BITS-1:0] chk_e_flat [NEB];      // chk_e_out[j][k]  -> e_cm cell wd
  logic signed [MSG_BITS-1:0] var_m_flat [NVB];      // var_m_out[i][d]  -> m_vm cell wd

  // sized write cursors / phase gates for the e_cm and m_vm cells (m_cm uses the shared scatter above)
  logic [BWV-1:0] wg_var;                            // var-group write cursor (S_INIT: pc, S_VAR: pc-3)
  logic [BWC-1:0] wg_chk;                            // chk-group write cursor (S_CHECK: pc-4)
  logic           mvm_wr_init, mvm_wr_var, ecm_wr_en;

  // ------------------------------------------------------------------- shared comb: cursors / addresses
  // M8 lags: reg plane (+1) deepens both phases; STAGES=3 CHK (+1 more) -> scatter group pc-4; var_update
  // stays 2-stage -> scatter group pc-3. Read ADDRESSES stay pc-driven; only the data/context is registered.
  always_comb begin
    wg          = (state == S_INIT) ? pc : (pc - 3);  // int; the shared m_cm scatter relies on <0 not matching
    mcm_ra      = (pc >= 0 && pc < GC) ? BWC'(pc) : '0;    // clamp: out-of-phase reads are unused
    mvm_ra      = (pc >= 0 && pc < GV) ? BWV'(pc) : '0;
    wg_var      = (state == S_INIT) ? BWV'(pc) : BWV'(pc - 3);
    wg_chk      = BWC'(pc - 4);
    mvm_wr_init = (state == S_INIT);
    mvm_wr_var  = (state == S_VAR)   && (pc >= 3);     // pc>=3 gate (M8): pc-3 in [0,GV-1], no spurious wrap
    ecm_wr_en   = (state == S_CHECK) && (pc >= 4);     // pc>=4 gate (M8): STAGES=3 CHK + 1-cyc reg plane
  end

  // ------------------------------------------------------------------- shared comb: m_cm write scatter
  // VAR (or S_INIT) scatters group `wg`: for each present edge of each var slot, drive its half-bank's
  // single write port. Row = the edge's CHECK group; data = the var-update output (or lambda in S_INIT).
  // This O(GV*V*VAR_DEG) constant-indexed sweep is the resolution that cannot be per-cell-ized on the
  // co-sim box; it stays SHARED here, exactly as the flat core (bank ids fold to constants, cursor picks row).
  always_comb begin
    for (int b = 0; b < NHB; b++) begin
      we_mcm[b] = 1'b0;
      wa_mcm[b] = '0;
      wd_mcm[b] = '0;
    end
    if (state == S_INIT || state == S_VAR) begin
      for (int h = 0; h < GV; h++)
        if (wg == h) begin
          for (int i = 0; i < V; i++)
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(h, i, d);
              if (e >= 0) begin
                automatic int hb = BP_EDGE_HB[e];             // literal half-bank (was hb_of_edge scan)
                we_mcm[hb] = 1'b1;
                wa_mcm[hb] = BWC'(BP_EDGE_ROW[e]);            // literal row (was grp_of_chk scan)
                if (state == S_INIT) wd_mcm[hb] = signed'(BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0]);
                else                 wd_mcm[hb] = var_m_out[i][d];
              end
            end
        end
    end
  end

  // ------------------------------------------------------------------- shared comb: e_cm read addresses
  // VAR launch group `pc`: for each present edge operand, route its bank's port-A/B read row. Same O(...)
  // constant-indexed sweep, kept shared for the same reason.
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = '0;
      rb_ecm[b] = '0;
    end
    if (state == S_VAR) begin
      for (int h = 0; h < GV; h++)
        if (pc == h) begin
          for (int i = 0; i < V; i++)
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(h, i, d);
              if (e >= 0) begin
                automatic int bank = BP_EDGE_EB[e];          // literal e_cm bank (was ecm_bank scan)
                automatic int row  = BP_EDGE_ROW[e];         // literal row (was grp_of_chk scan)
                if (BP_EDGE_EPORT[e] == 0) ra_ecm[bank] = BWC'(row);  // literal port (was ecm_port scan)
                else                       rb_ecm[bank] = BWC'(row);
              end
            end
        end
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
      bp_mcm_cell #(.B(b)) u_mcm (
          .clk(clk),
          .we (we_mcm[b]),
          .wa (wa_mcm[b]),
          .wd (wd_mcm[b]),
          .ra (mcm_ra),
          .q  (qmcm[b])
      );
    end
  endgenerate

  // ===================================================================== e_cm bank cells
  generate
    for (genvar b = 0; b < NEB; b++) begin : gecm
      bp_ecm_cell #(.B(b)) u_ecm (
          .clk  (clk),
          .wr_en(ecm_wr_en),
          .wg   (wg_chk),
          .wd   (chk_e_flat[b]),
          .ra   (ra_ecm[b]),
          .rb   (rb_ecm[b]),
          .qa   (qa_ecm[b]),
          .qb   (qb_ecm[b])
      );
    end
  endgenerate

  // ===================================================================== m_vm bank cells
`ifdef BP_RF_REGFILE
  // Q7-08: one wide byte-masked array per var slot (see bp_mvm_rf) instead of VAR_DEG per-(i,d) banks.
  // qmvm keeps its flat (i*VAR_DEG+d) indexing so the gathers are untouched.
  generate
    for (genvar i = 0; i < V; i++) begin : gmvm
      logic signed [MSG_BITS-1:0] q_i [BP_VAR_DEG];
      bp_mvm_rf #(.II(i)) u_mvm (
          .clk    (clk),
          .wr_init(mvm_wr_init),
          .wr_var (mvm_wr_var),
          .wg     (wg_var),
          .wd     (var_m_out[i]),
          .rg     (mvm_ra),
          .q      (q_i)
      );
      for (genvar d = 0; d < BP_VAR_DEG; d++) begin : gq
        assign qmvm[i * BP_VAR_DEG + d] = q_i[d];
      end
    end
  endgenerate
`else
  generate
    for (genvar b = 0; b < NVB; b++) begin : gmvm
      bp_mvm_cell #(.B(b)) u_mvm (
          .clk    (clk),
          .wr_init(mvm_wr_init),
          .wr_var (mvm_wr_var),
          .wg     (wg_var),
          .wd     (var_m_flat[b]),
          .rg     (mvm_ra),
          .q      (qmvm[b])
      );
    end
  endgenerate
`endif

  // ===================================================================== W check_minsum slots
  generate
    for (genvar j = 0; j < W; j++) begin : gchk
      logic                       sbit_j;
      logic signed [MSG_BITS-1:0] m_in_j    [BP_CHK_DEG];
      logic                       present_j [BP_CHK_DEG];
      // gather group `pc`'s check for slot j from m_cm at CONSTANT half-bank taps (beta folds to constant).
      always_comb begin
        sbit_j = 1'b0;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          m_in_j[k]    = '0;
          present_j[k] = 1'b0;
        end
        for (int g = 0; g < GC; g++)
          if (chk_at(g, j) >= 0 && pc == g) begin
            sbit_j = s_reg[chk_at(g, j)];
            for (int k = 0; k < BP_CHK_DEG; k++) begin
              automatic int e = edge_at(g, j, k);
              if (e >= 0) begin
                m_in_j[k]    = qmcm[BP_EDGE_HB[e]];   // literal half-bank tap (compile-time constant)
                present_j[k] = 1'b1;
              end
            end
          end
      end
      // M8 register plane: latch this slot's gathered launch context (bank-read magnitudes + syndrome bit +
      // present mask) one cycle, so the async-LUTRAM read + gather mux is no longer in series with the
      // check_minsum stage-1 reduction (Fmax). Free-running (unconditional, like check_minsum's stage 2);
      // en_chk_r fences the pc=0 garbage snapshot out. Registers the gather OUTPUTS (pc-select stays at pc).
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
          .STAGES(3)                                   // M8: extra mid-tree register plane for Fmax
      ) u_chk (
          .clk    (clk),
          .en     (en_chk_r),
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
      // gather group `pc`'s var for slot i: e_cm operands (port-selected) + the "old" m_vc from m_vm.
      always_comb begin
        lam_i = '0;
        gam_i = '0;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          e_in_i[d]    = '0;
          m_in_i[d]    = '0;
          present_i[d] = 1'b0;
        end
        for (int g = 0; g < GV; g++)
          if (var_at(g, i) >= 0 && pc == g) begin
            automatic int v = var_at(g, i);
            lam_i = signed'(BP_LAMBDA[v][MSG_BITS-1:0]);
            // Constant-fold the BP_GAMMA index over leg: `leg` is a runtime 32-bit int, and Vivado does
            // not range-bound it — `BP_GAMMA[leg*BP_N+v]` elaborates to full 32-bit index arithmetic (a
            // DSP multiply!) plus a 5184:1 ROM mux PER SITE (~864 sites = the ~386k-LUT whale the OOC
            // probes caught). Folding over l makes every index a compile-time constant -> a 6:1 mux.
            for (int l = 0; l < BP_LEGS; l++)
              if (leg == l) gam_i = signed'(BP_GAMMA[l * BP_N + v][MSG_BITS-1:0]);
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(g, i, d);
              if (e >= 0) begin
                automatic int bank = BP_EDGE_EB[e];          // literal e_cm bank (was ecm_bank scan)
                e_in_i[d]    = (BP_EDGE_EPORT[e] == 1) ? qb_ecm[bank] : qa_ecm[bank];  // literal port
                m_in_i[d]    = qmvm[i * BP_VAR_DEG + d];
                present_i[d] = 1'b1;
              end
            end
          end
      end
      // M8 register plane (same rationale as the CHK slots): latch the gathered var launch context (e_cm
      // operands, old m_vc, lambda, gamma, present mask) one cycle. Free-running; en_var_r fences pc=0.
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
          .en      (en_var_r),
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
            sat_pending <= 1'b0;                       // no decision to SAT before the first S_VAR
            leg <= '0; iter <= '0; pc <= '0;
            lat <= '0;
            state <= S_INIT;                           // banked messages must be lambda-seeded a group/cyc
          end
        end

        // ----------------------------------- seed m_cm/m_vm with lambda, one var-group/cycle (direct)
        S_INIT: begin
          if (pc == GV - 1) begin pc <= '0; state <= S_CHECK; end
          else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch check group `pc` + scatter `pc-4` (M8)  ||  overlapped S_SAT
        S_CHECK: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          // overlapped SAT: parity of the PREVIOUS decision (ehat) on the LAUNCHED group's checks
          if (pc < GC && sat_pending) begin
            for (int j = 0; j < W; j++)
              for (int g = 0; g < GC; g++)
                if (chk_at(g, j) >= 0 && pc == g) begin
                  p = s_reg[chk_at(g, j)];
                  for (int k = 0; k < BP_CHK_DEG; k++) begin
                    automatic int e = edge_at(g, j, k);
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
          // e_cm scatter of group pc-4 (M8 lag) handled by the per-bank bp_ecm_cell write.
          // advance cursor; early_exit takes the first syndrome-valid decision straight to S_EMIT.
          // (SAT is launch-aligned at pc<GC and finalises at pc==GC-1 — unaffected by the longer drain tail;
          //  the tail pc=GC..GC+3 only completes the deepened CHK scatter pipeline, no SAT runs there.)
          if (early_exit && final_sat) begin
            pc <= '0; state <= S_EMIT;
          end else if (pc == GC + 3) begin              // M8: was GC+1 (reg plane +1, STAGES 2->3 +1)
            pc      <= '0;
            all_sat <= 1'b1;
            state   <= S_VAR;
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch var group `pc` + scatter `pc-3` (M8)
        S_VAR: begin
          automatic logic wterm [V];                  // per-slot decision bit (present slot of group pc-3)
          automatic int   wsum;
          wsum = 0;
          if (pc == 0) ehat_w <= '0;                  // fresh decision-weight accumulation
          if (pc >= 3) begin                          // M8: var scatter lag pc-3 (was pc-2)
            for (int i = 0; i < V; i++) wterm[i] = 1'b0;
            for (int i = 0; i < V; i++)
              for (int g = 0; g < GV; g++)
                if (var_at(g, i) >= 0 && (pc - 3) == g) begin
                  automatic int v = var_at(g, i);
                  ehat[v] <= var_ehat_out[i];
                  wterm[i] = var_ehat_out[i];         // hoist the group mux out of the accumulation
                end
            // balanced add reduction over the V 1-bit terms (integer add associative -> identical wsum)
            for (int i = 0; i < V; i++) wsum = wsum + (wterm[i] ? 1 : 0);
            ehat_w <= ehat_w + WW'(wsum);
          end
          // m_vm / m_cm writes of group pc-3 (M8 lag) handled by the bp_mvm_cell + m_cm scatter comb.
          // End at pc==GV+2 so the last scatter (group GV-1 at pc=GV+2) — and its ehat[]/ehat_w update —
          // completes before the handoff; the next S_CHECK's launch-time SAT then reads a fully-written ehat.
          if (pc == GV + 2) begin                       // M8: was GV+1 (reg plane +1; var_update 2-stage)
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

        // ----------------------------- trailing SAT for the final decision (no following S_CHECK)
        S_SATF: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          for (int j = 0; j < W; j++)
            for (int g = 0; g < GC; g++)
              if (chk_at(g, j) >= 0 && pc == g) begin
                p = s_reg[chk_at(g, j)];
                for (int k = 0; k < BP_CHK_DEG; k++) begin
                  automatic int e = edge_at(g, j, k);
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

        // ----------------------------------------------------------------- reduce chosen ehat -> obs
        // TIMING RESCUE (137-level `pc_reg -> obs_acc_reg` path): the former serial fold interleaved the
        // GV:1 group mux with the observable XOR inside one V-deep double loop, which Vivado could not
        // rebalance. Split it: compute an independent per-slot term (its own constant-folded group mux),
        // then XOR-reduce the term array. XOR is associative/commutative -> bit-exact regardless of order.
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] base;
          automatic logic [BP_OBS-1:0] term [V];
          automatic logic [BP_OBS-1:0] acc;
          automatic logic              bb;
          base = (pc == 0) ? {BP_OBS{1'b0}} : obs_acc;
          bb   = 1'b0;
          for (int i = 0; i < V; i++) term[i] = {BP_OBS{1'b0}};
          for (int i = 0; i < V; i++)
            for (int g = 0; g < GV; g++)
              if (var_at(g, i) >= 0 && pc == g) begin
                automatic int v = var_at(g, i);
                bb = found ? best_e[v] : ehat[v];
                corr_out[v] <= bb;
                if (bb) term[i] = BP_OBS_MASK[v][BP_OBS-1:0];
              end
          acc = base;                              // pure XOR reduction over the per-slot term array
          for (int i = 0; i < V; i++) acc = acc ^ term[i];
          obs_acc <= acc;
          if (pc == GV - 1) begin pc <= '0; state <= S_DONE; end
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
