// Q7-04 M9b — MODULAR BRAM-ified sibling (`bp_relay_banked_bram_m`) of `bp_relay_banked_bram`.
//
// PROBE-RUN-5 VARIANT (12b house lesson): identical decode/schedule to the flat literal core
// `bp_relay_banked_bram.sv` (which stays intact as the A/B control), but every ROM + its sync-read row
// register is wrapped in a DEDICATED STAMPED MODULE (`bp_rom_*_bqm`, 14 of them) so
// `-flatten_hierarchy none` hands Vivado small repeated hierarchy units instead of one flat top — the
// M7 post-mortem's fix (the 1008-instance skeleton cleared in ~3 min the same phases where flat tops
// stalled). Each ROM cell reads its own BP_ROM_* literal localparam from the $unit header include and
// exposes addr -> registered row (the SAME pipeline stage the flat core's _q register formed — zero
// schedule change, decisions bit-exact, latency unchanged 2206/3871). Scatter/routing demuxes stay in
// the TOP for now (ROM-cells-only scope keeps this diff mechanical; they are the next candidates if
// the A/B says hierarchy is the lever). $unit names are suffixed `_bqm` for multi-top safety.
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
//       S_INIT   : pc = 0..GV+4 (GV..GV+3 = BENES_PIPE_MCM write-scatter drain), ROM read at pc, write at pc-1
//       S_CHECK  : phase end pc==GC+4;  e_cm scatter gate pc>=5, write-group pc-5
//       S_VAR    : phase end pc==GV+13 (M9c: site-3 GV+9 + site-4 drain +4);  m_cm/m_vm scatter gate pc>=10, write-group pc-10; ehat/ehat_w at pc-10
//       S_EMIT   : constant-folded per M8 (pc=0..GV-1) — REVERTED from the ROM form (probe run 4: the obs
//                  ROMs runtime-indexed BP_N-scale register arrays; see the ROM CONSUMER RULE below)
//   * R5 (the overlapped-SAT parity + finalize) stays LUT wire taps — cheap constant-index `ehat` taps and
//     XOR trees — UNCHANGED, still finalising at pc==GC-1 (re-verified against the co-sim waves). The
//     `early_exit` path (first syndrome-valid decision -> S_EMIT) is likewise structurally unchanged; the
//     golden gate drives early_exit=0.
//   The m_vm read-row(pc-1)/write-row(pc-10) disjointness argument keeps a WIDER (9-cycle) gap than the M8
//     baseline (which reads row pc and writes row pc-3): the write cursor shifted further (M9c site-3/4
//     lag) while the read cursor did not, so the gap only grows and disjointness still holds.
//
// CELLS: `bp_ecm_cell_bqm` / `bp_mvm_cell_bqm` become PURE PORT-DRIVEN memories (like the M7 `bp_mcm_cell`) —
//   their write decode no longer calls `edge_at`/`vedge_at` internally (that self-scan is exactly the LUT
//   scanner cloud this milestone removes); the TOP drives their write enable/address from the ROMs. Cell
//   BOUNDARIES are preserved (the stamped-module structure is load-bearing for `-flatten_hierarchy none`).
//
// $unit-SCOPE NAMES are suffixed `_bqm` (BB_*_bqm geometry, *_bqm helper functions) and the stamped cell
//   modules `*_bqm`, so this sibling is collision-safe in a future multi-top compilation unit alongside the
//   M8 original (the Makefile targets build ONE top per scratch dir, so a collision cannot occur in the
//   gate, but the suffix hardens it for free). The `ifndef SYNTHESIS` elaboration guards survive verbatim
//   (they still scan the original localparam arrays; only the one `vedge_at`->`vedge_at_bqm` call and the
//   $display module name are renamed — the guard LOGIC is byte-identical).

`timescale 1ns / 1ps
// Q7-02 B0 Option A: the header's BP_ROM_* literal block is `ifdef-gated so cores that never read it
// (bp_relay_banked, every non-banked core) do not have to parse rows that cross Verilator's 65536-bit
// number limit at the single-group geometries. This core DOES read it — opt in before the `include.
`define BP_BRAM_ROMS
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

// ================================================================ $unit-scope geometry (shared by cells)
localparam int BB_GC_bqm   = BP_GC;                    // number of check groups (m_cm / e_cm rows)
localparam int BB_GV_bqm   = BP_GV;                    // number of var groups   (m_vm rows)
localparam int BB_BWC_bqm  = $clog2(BP_GC);            // m_cm / e_cm row address width
localparam int BB_BWV_bqm  = $clog2(BP_GV);            // m_vm row address width

/* verilator lint_off UNUSEDSIGNAL */
// ============================================================ $unit elaboration helpers over header tables
// Same bodies as the M8 core's `chk_at/var_at/...`, suffixed `_bqm` so both siblings can share one $unit.
// Called only with compile-time-constant args (genvar / cell params / group index gated by pc==g), and in
// the ROM-fill `initial` blocks with loop-runtime args (procedural, evaluated once at time-0 — cheap, like
// the elaboration guards below — NOT elaboration-time constant functions, which OOM-kill Verilator).
function automatic int chk_at_bqm(input int g, input int j);
  return BP_CHK_AT[g * BP_BANK_W + j];
endfunction
function automatic int var_at_bqm(input int h, input int i);
  return BP_VAR_AT[h * BP_BANK_V + i];
endfunction
function automatic int chk_deg_bqm(input int c);
  return BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c];
endfunction
function automatic int var_deg_bqm(input int v);
  return BP_VAR_OFF[v + 1] - BP_VAR_OFF[v];
endfunction
function automatic int edge_at_bqm(input int g, input int j, input int k);
  automatic int c = chk_at_bqm(g, j);
  if (c < 0) return -1;
  if (k >= chk_deg_bqm(c)) return -1;
  return BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];
endfunction
function automatic int vedge_at_bqm(input int h, input int i, input int d);
  automatic int v = var_at_bqm(h, i);
  if (v < 0) return -1;
  if (d >= var_deg_bqm(v)) return -1;
  return BP_VAR_OFF[v] + d;
endfunction
/* verilator lint_on UNUSEDSIGNAL */

// ============================================================================ STAMPED BANK CELL MODULES
// PURE PORT-DRIVEN memories (write enable/address/data all from the TOP's ROM-fed scatter). One instance
// per (half-bank | bank), so `-flatten_hierarchy none` keeps each memory as an independent area-opt unit.
/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDPARAM */

// --------------------------------------------------------------- m_cm half-bank: 1 sync write, 1 async read
module bp_mcm_cell_bqm #(
    parameter int B = 0                                  // half-bank id (identity only; wiring via ports)
) (
    input  logic                       clk,
    input  logic                       we,               // from the top's ROM-fed m_cm write scatter
    input  logic [BB_BWC_bqm-1:0]       wa,               // write row (= edge's CHECK group, from ROM)
    input  logic signed [MSG_BITS-1:0] wd,               // write data (lambda in S_INIT, var m_out in S_VAR)
    input  logic [BB_BWC_bqm-1:0]       ra,               // read row (= registered pc)
    output logic signed [MSG_BITS-1:0] q
);
  logic signed [MSG_BITS-1:0] mem [BB_GC_bqm];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign q = mem[ra];
endmodule

// --------------------------------------------------------------- e_cm bank: 1 sync write, 2 async reads
// PORT-DRIVEN (M9b): write enable/row now come from the TOP (ECM_WR_ROM), not an internal `edge_at` scan.
module bp_ecm_cell_bqm #(
    parameter int B = 0                                  // bank id = check-slot j * CHK_DEG + lane k
) (
    input  logic                       clk,
    input  logic                       we,               // ECM_WR_ROM present bit & S_CHECK scatter gate
    input  logic [BB_BWC_bqm-1:0]       wa,               // write chk-group row (= pc-5, shared)
    input  logic signed [MSG_BITS-1:0] wd,               // this bank's chk lane message (chk_e_out[JJ][KK])
    input  logic [BB_BWC_bqm-1:0]       ra,               // port-A read row (from ROM-fed e_cm read-addr)
    input  logic [BB_BWC_bqm-1:0]       rb,               // port-B read row
    output logic signed [MSG_BITS-1:0] qa,
    output logic signed [MSG_BITS-1:0] qb
);
  logic signed [MSG_BITS-1:0] mem [BB_GC_bqm];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign qa = mem[ra];
  assign qb = mem[rb];
endmodule

// --------------------------------------------------------------- m_vm bank: 1 sync write, 1 async read
// PORT-DRIVEN (M9b): write enable/row/data all from the TOP (SCATTER_ROM present + lambda), not a scan.
module bp_mvm_cell_bqm #(
    parameter int B = 0                                  // bank id = var-slot i * VAR_DEG + edge d
) (
    input  logic                       clk,
    input  logic                       we,               // SCATTER_ROM present bit & (init | var) gate
    input  logic [BB_BWV_bqm-1:0]       wa,               // write var-group row (S_INIT: pc-1, S_VAR: pc-10)
    input  logic signed [MSG_BITS-1:0] wd,               // lambda (S_INIT) or var_m_out[II][DD] (S_VAR)
    input  logic [BB_BWV_bqm-1:0]       rg,               // read row (= registered pc)
    output logic signed [MSG_BITS-1:0] q
);
  logic signed [MSG_BITS-1:0] mem [BB_GV_bqm];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  assign q = mem[rg];
endmodule
/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on DECLFILENAME */

// ============================================================ $unit geometry for the stamped ROM cells
// Row widths / depths / address widths of the 14 BP_ROM_* tables, derived from the header exactly like
// the TOP's own localparams (same $clog2 expressions — the packing contract's widths).
localparam int BQM_NEBW = BP_BANK_W * BP_CHK_DEG;                  // e_cm banks   = (j,k) lanes
localparam int BQM_NVBW = BP_BANK_V * BP_VAR_DEG;                  // m_vm banks   = (i,d) slots
localparam int BQM_HBW  = $clog2(2 * BQM_NEBW);                    // half-bank index width
localparam int BQM_BWCW = $clog2(BP_GC);                           // m_cm/e_cm row address width
localparam int BQM_AWC  = $clog2(BP_GC);                           // GC-depth ROM address width
localparam int BQM_AWV  = $clog2(BP_GV);                           // GV-depth ROM address width
localparam int BQM_AWG  = $clog2(BP_LEGS * BP_GV);                 // gamma ROM address width

// ============================================================ STAMPED ROM CELLS (one module per ROM)
// 12b lesson: a synthesis hierarchy boundary around every ROM + its output register, so Vivado's
// area-opt/closure passes see 14 small units instead of one flat ROM cloud. Each cell is a pure
// literal-copy ROM (zero elaboration compute, same as the flat core's fill_roms) + ONE sync read.
/* verilator lint_off DECLFILENAME */
module bp_rom_chk_hbsel_bqm (
    input  logic clk, input logic [BQM_AWC-1:0] addr, output logic [BQM_NEBW*BQM_HBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NEBW*BQM_HBW-1:0] rom [BP_GC];
  initial for (int i = 0; i < BP_GC; i++) rom[i] = BP_ROM_CHK_HBSEL[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_chk_epres_bqm (
    input  logic clk, input logic [BQM_AWC-1:0] addr, output logic [BQM_NEBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NEBW-1:0] rom [BP_GC];
  initial for (int i = 0; i < BP_GC; i++) rom[i] = BP_ROM_CHK_EPRES[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_ecm_wpres_bqm (
    input  logic clk, input logic [BQM_AWC-1:0] addr, output logic [BQM_NEBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NEBW-1:0] rom [BP_GC];
  initial for (int i = 0; i < BP_GC; i++) rom[i] = BP_ROM_ECM_WPRES[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_pres_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BP_BANK_V-1:0] q);
  (* rom_style = "block" *) logic [BP_BANK_V-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_VAR_PRES[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_lam_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BP_BANK_V*MSG_BITS-1:0] q);
  (* rom_style = "block" *) logic [BP_BANK_V*MSG_BITS-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_VAR_LAM[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_gam_bqm (
    input  logic clk, input logic [BQM_AWG-1:0] addr, output logic [BP_BANK_V*MSG_BITS-1:0] q);
  (* rom_style = "block" *) logic [BP_BANK_V*MSG_BITS-1:0] rom [BP_LEGS*BP_GV];
  initial for (int i = 0; i < BP_LEGS*BP_GV; i++) rom[i] = BP_ROM_VAR_GAM[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_epres_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_VAR_EPRES[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_eport_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_VAR_EPORT[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_var_erow_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW*BQM_BWCW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW*BQM_BWCW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_VAR_EROW[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_scat_pres_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_SCAT_PRES[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
// M9c site 2: registered per-group Beneš read-gather control ROM (both port halves in one row).
// Row width = PORTS * COLS * (M/2) (never hard-coded: adapts to the per-banking M). Addressed by the
// same var read-group cursor as the other var-gather ROMs so its output aligns to the pc-1 launch and
// is held stable for that group's single-cycle window (satisfies the fabric's combinational ctrl-hold).
// M9c Step 5d: row width is now BP_ASW_ECM_SW (AS-Waksman switch count at the REAL neb bank count)
// per port, not the old power-of-two-padded Beneš COLS*(M/2) width -- the emitter narrowed the ROM
// content to match (Task 2); this cell's declared width follows.
module bp_rom_benes_ecmrd_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr,
    output logic [BP_BENES_ECM_PORTS*BP_ASW_ECM_SW-1:0] q);
  (* rom_style = "block" *)
  logic [BP_BENES_ECM_PORTS*BP_ASW_ECM_SW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_BENES_ECMRD[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
// M9c Step 4b: e_cm read-ADDRESS scatter is no longer a runtime Beneš route -- it is a pure function of
// the group, precomputed at gen time into BP_ROM_ECM_READROW (see the Rust emitter guard). This ROM holds
// the resolved read row per bank/port directly (row bit layout: bank `bk` port `p` at (bk*2+p)*BWC +: BWC,
// 0 when no tap lands there), replacing `bp_rom_benes_ecmaddr_bqm` + the `u_benes_ad0/ad1` fabric below.
module bp_rom_ecm_readrow_bqm (
    input  logic clk,
    input  logic [BQM_AWV-1:0] addr,
    output logic [BP_ECM_READROW_W-1:0] q);
  (* rom_style = "block" *) logic [BP_ECM_READROW_W-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_ECM_READROW[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
// M9c site 4: registered per-group Beneš m_cm write-scatter control ROM (single network — hb=eb*2+beta
// is injective, no port split). Row width = COLS*(M/2). Addressed by the scatter cursor (like scat_*).
// M9c Step 5d: row width is now BP_ASW_MCM_SW (AS-Waksman switch count at the REAL nhb half-bank
// count), not the old power-of-two-padded Beneš COLS*(M/2) width.
module bp_rom_benes_mcmwr_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr,
    output logic [BP_ASW_MCM_SW-1:0] q);
  (* rom_style = "block" *)
  logic [BP_ASW_MCM_SW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_BENES_MCMWR[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_scat_row_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW*BQM_BWCW-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW*BQM_BWCW-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_SCAT_ROW[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
module bp_rom_scat_lam_bqm (
    input  logic clk, input logic [BQM_AWV-1:0] addr, output logic [BQM_NVBW*MSG_BITS-1:0] q);
  (* rom_style = "block" *) logic [BQM_NVBW*MSG_BITS-1:0] rom [BP_GV];
  initial for (int i = 0; i < BP_GV; i++) rom[i] = BP_ROM_SCAT_LAM[i];
  always_ff @(posedge clk) q <= rom[addr];
endmodule
/* verilator lint_on DECLFILENAME */

// ========================================================================================= TOP CORE
module bp_relay_banked_bram_m (
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
  // M9c Beneš pipeline depths. BENES_PIPE_ECM registers each e_cm fabric (site-2 read, site-3 addr) into
  // <=3 timing stages; the two are in series on the e_cm operand path (addr -> async read -> read-gather)
  // so the total e_cm latency is BENES_ECM_LAT = 2*BENES_PIPE_ECM. BENES_PIPE_MCM registers the site-4
  // m_cm write-scatter fabric (M9c Step 5d: AS-Waksman, single N=BP_ASW_MCM_N network) into <=4 timing
  // stages.
  localparam int BENES_PIPE_ECM = 3;
  localparam int BENES_ECM_LAT  = 2 * BENES_PIPE_ECM;
  localparam int BENES_PIPE_MCM = 4;

  /* verilator lint_off UNUSEDSIGNAL */
  // ================================================================== elaboration guards (verbatim from M8)
  // Enforce the offline (Task-9 emitter) split invariants at time-0. Byte-identical to `bp_relay_banked`
  // except the one `vedge_at`->`vedge_at_bqm` call and the $display module name; the guard LOGIC is unchanged.
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
        $display("bp_relay_banked_bram_m GUARD(d1) FAIL: check %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
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
        $display("bp_relay_banked_bram_m GUARD(d2) FAIL: var %0d table grp/slot (%0d,%0d) != scan (%0d,%0d)",
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
        $display("bp_relay_banked_bram_m GUARD(d3) FAIL: edge %0d EB/HB/ROW (%0d,%0d,%0d) != recompute (%0d,%0d,%0d)",
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
          automatic int e = vedge_at_bqm(h, i, d);
          if (e >= 0) begin
            automatic int hb = BP_EDGE_HB[e];
            automatic int eb = BP_EDGE_EB[e];
            // EPORT is the count of same-e_cm-bank readers of this group seen BEFORE e in (i,d) order.
            if (BP_EDGE_EPORT[e] != rcnt[eb]) begin
              $display("bp_relay_banked_bram_m GUARD(eport) FAIL: var-group %0d edge %0d EPORT=%0d != readers-so-far %0d",
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
          $display("bp_relay_banked_bram_m GUARD(a) FAIL: var-group %0d m_cm half-bank %0d has %0d writers (>1)",
                   h, b, wcnt[b]);
          fails = fails + 1;
        end
      // (b) <=2 readers per (var-group, e_cm bank) — the two async read ports per e_cm bank.
      for (int b = 0; b < NEB; b++)
        if (rcnt[b] > 2) begin
          $display("bp_relay_banked_bram_m GUARD(b) FAIL: var-group %0d e_cm bank %0d has %0d readers (>2)",
                   h, b, rcnt[b]);
          fails = fails + 1;
        end
    end
    // (c) BP_EDGE_POS[e] is e's position in its check's CSR row (edge_at_bqm / BP_EDGE_HB tap correctness).
    for (int e = 0; e < BP_E; e++) begin
      automatic int c   = BP_EDGE_CHK[e];
      automatic int idx = BP_CHECK_OFF[c] + BP_EDGE_POS[e];
      if (idx >= BP_CHECK_OFF[c + 1] || BP_CHECK_EDGES[idx] != e) begin
        $display("bp_relay_banked_bram_m GUARD(c) FAIL: edge %0d (check %0d) EDGE_POS=%0d does not match CSR row",
                 e, c, BP_EDGE_POS[e]);
        fails = fails + 1;
      end
    end
    if (fails != 0)
      $fatal(1, "bp_relay_banked_bram_m: %0d elaboration-guard violation(s) — header/emitter split is unsafe", fails);
    else
      $display("bp_relay_banked_bram_m: elaboration guards (a/b/c/d) PASS (GV=%0d NHB=%0d NEB=%0d BP_E=%0d)",
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
  // M9c: the var launch is delayed BENES_PIPE_ECM extra cycles (PIPE=3 Beneš e_in + aligned operands),
  // so the var enable gets 3 more stages (en_var_r5) to fire var_update on the same cycle its now-3-late
  // gather-plane operands land. The check path is untouched (en_chk_rr).
  logic en_var_r3, en_var_r4, en_var_r5;
  // M9c site 3: the addr scatter adds a SECOND PIPE=3 Beneš in series ahead of the site-2 read fabric
  // (addr -> async e_cm read -> read-gather), so e_in emerges BENES_ECM_LAT(=6) cycles late instead of 3.
  // en_var gains 3 more stages (en_var_r8) so var_update fires when its now-6-late operands land.
  logic en_var_r6, en_var_r7, en_var_r8;
  always_ff @(posedge clk) begin
    en_chk_r  <= en_chk;
    en_var_r  <= en_var;
    en_chk_rr <= en_chk_r;
    en_var_rr <= en_var_r;
    en_var_r3 <= en_var_rr;
    en_var_r4 <= en_var_r3;
    en_var_r5 <= en_var_r4;
    en_var_r6 <= en_var_r5;
    en_var_r7 <= en_var_r6;
    en_var_r8 <= en_var_r7;
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
  logic [BWV-1:0]             wa_mvm;                   // shared m_vm write row (S_INIT: pc-1, S_VAR: pc-10)

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
  //
  // ROM CONSUMER RULE (Task-4 probe run 4): a ROM output may feed (a) DATA paths (lambda/gamma/lam-seed,
  // submodule operands) and (b) ADDRESSES / WRITE-ENABLES of the actual inferred memories (the LUTRAM
  // m_cm/e_cm/m_vm bank cells) or selects over their OUTPUT buses (NHB/NEB-way muxes) — it must NEVER be
  // a runtime index into a register array or wire bundle of BP_C/BP_N scale: Vivado does not fold
  // runtime-indexed accesses into large register arrays (M9a ops lesson), it port-splits and explodes
  // closure. The former CHK sbit-select ROM (s_reg[BP_C] index) and S_EMIT obs ROMs (ehat/best_e[BP_N]
  // reads + corr_out[BP_N] write decode) violated this and are REVERTED to the LUT-core constant-folded
  // `pc==g` form — those are C/N-scale sites that were cheap in the LUT core; the 169%-LUT whale was the
  // E-scale edge fabric, which is exactly what stays ROM-ified below.
  // R3 (CHK gather half-bank taps), R4+R2(read) (VAR gather), R1 (m_cm/m_vm scatter), R2(write)
  // (e_cm write masks): the ROMs themselves live in the 14 STAMPED `bp_rom_*_bqm` cells above (12b
  // modularization — one hierarchy unit per ROM + its output register); instances below. R6 (S_EMIT
  // observable) is NOT a ROM: constant-folded per the consumer rule above.

  // ------------------------------------------------ ROM packing-contract guard (emitter <-> RTL, sim-only)
  // Recomputes EVERY BP_ROM_* row from the graph tables using the OLD fill expressions (the pre-literal
  // computed fills, via the `_bqm` helpers) and asserts bit-equality with the emitter's literals. Any
  // packing drift — field width, slot stride, bit order, or content — fails the co-sim gate LOUDLY at
  // time-0 instead of silently mis-routing a bank. Fenced from synthesis like the elaboration guards.
`ifndef SYNTHESIS
  initial begin : rom_contract
    automatic int fails = 0;
    for (int g = 0; g < GC; g++) begin
      automatic logic [NEB*HBW-1:0] x_hbsel;
      automatic logic [NEB-1:0]     x_epres;
      x_hbsel = '0; x_epres = '0;
      for (int j = 0; j < W; j++) begin
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          automatic int e = edge_at_bqm(g, j, k);
          if (e >= 0) begin
            x_epres[j*BP_CHK_DEG + k]                = 1'b1;
            x_hbsel[(j*BP_CHK_DEG + k)*HBW +: HBW]   = HBW'(BP_EDGE_HB[e]);
            // M9c 2:1 beta-split invariant: HB(e) >> 1 must equal the tap id, else qmcm[idx*2+beta] is wrong.
            if ((BP_EDGE_HB[e] >> 1) != (j*BP_CHK_DEG + k)) begin
              $display("bp_relay_banked_bram_m BETA-SPLIT FAIL: g=%0d j=%0d k=%0d HB=%0d expected base %0d",
                       g, j, k, BP_EDGE_HB[e], j*BP_CHK_DEG + k); fails = fails + 1;
            end
          end
        end
      end
      if (BP_ROM_CHK_HBSEL[g] !== x_hbsel) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: CHK_HBSEL row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_CHK_EPRES[g] !== x_epres) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: CHK_EPRES row %0d", g); fails = fails + 1;
      end
      // ecm_wpres is the same "lane (j,k) has a real edge" mask as chk_epres (different read address).
      if (BP_ROM_ECM_WPRES[g] !== x_epres) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: ECM_WPRES row %0d", g); fails = fails + 1;
      end
    end
    for (int g = 0; g < GV; g++) begin
      automatic logic [V-1:0]            x_pres;
      automatic logic [V*MSG_BITS-1:0]   x_lam;
      automatic logic [NVB-1:0]          x_epres;
      automatic logic [NVB*EBW-1:0]      x_ebsel;
      automatic logic [NVB-1:0]          x_eport;
      automatic logic [NVB*BWC-1:0]      x_erow;
      automatic logic [NVB-1:0]          x_spres;
      automatic logic [NVB*HBW-1:0]      x_shb;
      automatic logic [NVB*BWC-1:0]      x_srow;
      automatic logic [NVB*MSG_BITS-1:0] x_slam;
      x_pres = '0; x_lam = '0; x_epres = '0; x_ebsel = '0; x_eport = '0; x_erow = '0;
      x_spres = '0; x_shb = '0; x_srow = '0; x_slam = '0;
      for (int i = 0; i < V; i++) begin
        automatic int v = var_at_bqm(g, i);
        if (v >= 0) begin
          x_pres[i]                       = 1'b1;
          x_lam[i*MSG_BITS +: MSG_BITS]   = BP_LAMBDA[v][MSG_BITS-1:0];
        end
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int e = vedge_at_bqm(g, i, d);
          automatic int s = i*BP_VAR_DEG + d;
          if (e >= 0) begin
            x_epres[s]              = 1'b1;
            x_ebsel[s*EBW +: EBW]   = EBW'(BP_EDGE_EB[e]);
            x_eport[s]              = (BP_EDGE_EPORT[e] == 1);
            x_erow[s*BWC +: BWC]    = BWC'(BP_EDGE_ROW[e]);
            x_spres[s]              = 1'b1;
            x_shb[s*HBW +: HBW]     = HBW'(BP_EDGE_HB[e]);
            x_srow[s*BWC +: BWC]    = BWC'(BP_EDGE_ROW[e]);
            x_slam[s*MSG_BITS +: MSG_BITS] = BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0];
          end
        end
      end
      if (BP_ROM_VAR_PRES[g]  !== x_pres)  begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_PRES row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_VAR_LAM[g]   !== x_lam)   begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_LAM row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_VAR_EPRES[g] !== x_epres) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_EPRES row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_VAR_EBSEL[g] !== x_ebsel) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_EBSEL row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_VAR_EPORT[g] !== x_eport) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_EPORT row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_VAR_EROW[g]  !== x_erow)  begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_EROW row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_SCAT_PRES[g] !== x_spres) begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: SCAT_PRES row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_SCAT_HB[g]   !== x_shb)   begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: SCAT_HB row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_SCAT_ROW[g]  !== x_srow)  begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: SCAT_ROW row %0d", g); fails = fails + 1;
      end
      if (BP_ROM_SCAT_LAM[g]  !== x_slam)  begin
        $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: SCAT_LAM row %0d", g); fails = fails + 1;
      end
      for (int l = 0; l < BP_LEGS; l++) begin
        automatic logic [V*MSG_BITS-1:0] x_gam;
        x_gam = '0;
        for (int i = 0; i < V; i++) begin
          automatic int v = var_at_bqm(g, i);
          if (v >= 0) x_gam[i*MSG_BITS +: MSG_BITS] = BP_GAMMA[l*BP_N + v][MSG_BITS-1:0];
        end
        if (BP_ROM_VAR_GAM[l*GV + g] !== x_gam) begin
          $display("bp_relay_banked_bram_m ROM-CONTRACT FAIL: VAR_GAM row (leg %0d, group %0d)", l, g);
          fails = fails + 1;
        end
      end
    end
    if (fails != 0)
      $fatal(1, "bp_relay_banked_bram_m: %0d ROM packing-contract violation(s) — emitter literal block and RTL layout disagree", fails);
    else
      $display("bp_relay_banked_bram_m: ROM packing contract PASS (14 ROMs, GC=%0d GV=%0d)", GC, GV);
  end
`endif

  // -------------------------------------------------------------------- ROM cell row-word outputs
  // Driven by the stamped `bp_rom_*_bqm` cells below; each cell's registered read IS the (single) ROM
  // output register — same pipeline stage as the flat core's `_q` registers, no extra depth.
  logic [NEB*HBW-1:0]      chk_hbsel_q;
  logic [NEB-1:0]          chk_epres_q;
  logic [V-1:0]            var_pres_q;
  logic [V*MSG_BITS-1:0]   var_lam_q;
  logic [V*MSG_BITS-1:0]   var_gam_q;
  logic [NVB-1:0]          var_epres_q;
  logic [NVB-1:0]          var_eport_q;
  logic [NVB*BWC-1:0]      var_erow_q;
  logic [NVB-1:0]          scat_pres_q;
  logic [NVB*BWC-1:0]      scat_row_q;
  logic [NVB*MSG_BITS-1:0] scat_lam_q;
  logic [NEB-1:0]          ecm_wpres_q;
  // M9c site 2: registered read-gather control (both port halves), pc-1-aligned like the ROMs above.
  // M9c Step 5d: per-port ctrl width is now BP_ASW_ECM_SW (AS-Waksman, real neb count), not the old
  // power-of-two-padded Beneš COLS*(M/2) width.
  localparam int BENES_ECM_CTRLW = BP_ASW_ECM_SW;   // per-port ctrl width
  logic [BP_BENES_ECM_PORTS*BENES_ECM_CTRLW-1:0] benes_ecmrd_q;
  // M9c Step 4b: registered e_cm resolved read-row data (both banks/ports in one row); replaces the
  // site-3 Beneš addr-scatter control `benes_ecmaddr_q` (now a pure data ROM, see bp_rom_ecm_readrow_bqm).
  logic [BP_ECM_READROW_W-1:0] ecm_readrow_q;
  // M9c site 4: registered m_cm write-scatter control (single network).
  // M9c Step 5d: ctrl width is now BP_ASW_MCM_SW (AS-Waksman, real nhb count), not the old
  // power-of-two-padded Beneš COLS*(M/2) width.
  localparam int BENES_MCM_CTRLW = BP_ASW_MCM_SW;
  logic [BENES_MCM_CTRLW-1:0] benes_mcmwr_q;

  // combinational per-slot field slices of the registered rows (the pre-rework consumer names, unchanged
  // downstream: gathers, scatters, S_EMIT all read these exactly as before)
  logic [HBW-1:0]       chk_hbsel_r     [NEB];
  logic                 chk_epres_r     [NEB];
  logic                 var_pres_r  [V];
  logic [MSG_BITS-1:0]  var_lam_r   [V];
  logic [MSG_BITS-1:0]  var_gam_r   [V];
  logic                 var_epres_r [NVB];
  logic                 var_eport_r [NVB];
  logic [BWC-1:0]       var_erow_r  [NVB];
  logic                 scat_pres_r [NVB];
  logic [BWC-1:0]       scat_row_r  [NVB];
  logic [MSG_BITS-1:0]  scat_lam_r  [NVB];
  logic                 ecm_wpres_r [NEB];
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      chk_hbsel_r[b] = chk_hbsel_q[b*HBW +: HBW];
      chk_epres_r[b] = chk_epres_q[b];
      ecm_wpres_r[b] = ecm_wpres_q[b];
    end
    for (int i = 0; i < V; i++) begin
      var_pres_r[i] = var_pres_q[i];
      var_lam_r[i]  = var_lam_q[i*MSG_BITS +: MSG_BITS];
      var_gam_r[i]  = var_gam_q[i*MSG_BITS +: MSG_BITS];
    end
    for (int b = 0; b < NVB; b++) begin
      var_epres_r[b] = var_epres_q[b];
      var_eport_r[b] = var_eport_q[b];
      var_erow_r[b]  = var_erow_q[b*BWC +: BWC];
      scat_pres_r[b] = scat_pres_q[b];
      scat_row_r[b]  = scat_row_q[b*BWC +: BWC];
      scat_lam_r[b]  = scat_lam_q[b*MSG_BITS +: MSG_BITS];
    end
  end

  // ------------------------------------------------------------------- shared comb: read/write cursors
  // Gather ROMs are addressed by the M8 GATHER cursor (pc); their registered output aligns with the
  // +1-delayed launch, and the message-read address is registered by 1 to meet it. Scatter ROMs are
  // addressed by the WRITE cursor (INIT:pc / VAR:pc-9, M9c site-3 lag); their registered output aligns
  // with the +1-bumped write cursor (INIT:pc-1 / VAR:pc-10). All addresses clamped in-range (out-of-phase reads
  // are gated unused downstream).
  int chk_rd, var_rd, gam_rd, scat_rd_i, scat_rd, ecmw_rd_i, ecmw_rd;
  int wa_ecm_i, wa_mvm_i;
  logic ecm_we_gate, mvm_we_gate, mcm_we_gate, scat_is_init;
  always_comb begin
    chk_rd    = (pc >= 0 && pc < GC) ? pc : 0;
    var_rd    = (pc >= 0 && pc < GV) ? pc : 0;
    gam_rd    = leg * GV + var_rd;                       // leg in [0,LEGS), var_rd in [0,GV)
    // M9c: S_VAR scatter cursor. Site-2 read fabric (+3) and site-3 addr fabric (+3) put var_m_out on the
    // e_cm path 6 cycles late (BENES_ECM_LAT), so the S_VAR scatter cursor is pc-9 (M9b pc-3 + 6). S_INIT
    // seeding and S_CHECK e_cm writes ride the un-delayed check path and keep their baseline offsets.
    scat_rd_i = (state == S_INIT) ? pc : (pc - 9);
    scat_rd   = (scat_rd_i >= 0 && scat_rd_i < GV) ? scat_rd_i : 0;
    ecmw_rd_i = pc - 4;
    ecmw_rd   = (ecmw_rd_i >= 0 && ecmw_rd_i < GC) ? ecmw_rd_i : 0;

    wa_ecm_i     = pc - 5;
    wa_ecm       = (wa_ecm_i >= 0 && wa_ecm_i < GC) ? BWC'(wa_ecm_i) : '0;
    wa_mvm_i     = (state == S_INIT) ? (pc - 1) : (pc - 10);
    wa_mvm       = (wa_mvm_i >= 0 && wa_mvm_i < GV) ? BWV'(wa_mvm_i) : '0;
    scat_is_init = (state == S_INIT);
    ecm_we_gate  = (state == S_CHECK) && (pc >= 5);
    // M9c site 4: the m_cm/m_vm write windows are now explicitly UPPER-bounded to the last real scatter
    // group. Site 4 extends S_INIT/S_VAR by BENES_PIPE_MCM so the write-scatter fabric fully drains inside
    // the phase; during those extra drain cycles the din-side gate must be low (scat_rd clamps to group 0,
    // whose scat_pres would else re-issue a spurious write). m_vm is a direct (non-fabric) write, so its
    // gate is the din window; m_cm uses the PIPE-delayed mcm_we_gate_d for the actual write-enable.
    mvm_we_gate  = ((state == S_INIT) && (pc >= 1) && (pc <= GV)) ||
                   ((state == S_VAR)  && (pc >= 10) && (pc <= GV + 9));
    mcm_we_gate  = ((state == S_INIT) && (pc >= 1) && (pc <= GV)) ||
                   ((state == S_VAR)  && (pc >= 10) && (pc <= GV + 9));
  end

  // ------------------------------------------------------- stamped ROM cells (sync read inside each cell)
  // ONE whole-row sync read per ROM per cycle, INSIDE its own stamped cell (the cell's q register IS the
  // flat core's `_q` row register — same pipeline stage, zero schedule change). Addresses are the same
  // clamped cursors the flat core used, sized to each cell's address width.
  bp_rom_chk_hbsel_bqm  u_rom_chk_hbsel  (.clk(clk), .addr(BQM_AWC'(chk_rd)),  .q(chk_hbsel_q));
  bp_rom_chk_epres_bqm  u_rom_chk_epres  (.clk(clk), .addr(BQM_AWC'(chk_rd)),  .q(chk_epres_q));
  bp_rom_ecm_wpres_bqm  u_rom_ecm_wpres  (.clk(clk), .addr(BQM_AWC'(ecmw_rd)), .q(ecm_wpres_q));
  bp_rom_var_pres_bqm   u_rom_var_pres   (.clk(clk), .addr(BQM_AWV'(var_rd)),  .q(var_pres_q));
  bp_rom_var_lam_bqm    u_rom_var_lam    (.clk(clk), .addr(BQM_AWV'(var_rd)),  .q(var_lam_q));
  bp_rom_var_gam_bqm    u_rom_var_gam    (.clk(clk), .addr(BQM_AWG'(gam_rd)),  .q(var_gam_q));
  bp_rom_var_epres_bqm  u_rom_var_epres  (.clk(clk), .addr(BQM_AWV'(var_rd)),  .q(var_epres_q));
  bp_rom_var_eport_bqm  u_rom_var_eport  (.clk(clk), .addr(BQM_AWV'(var_rd)),  .q(var_eport_q));
  bp_rom_var_erow_bqm   u_rom_var_erow   (.clk(clk), .addr(BQM_AWV'(var_rd)),  .q(var_erow_q));
  bp_rom_scat_pres_bqm  u_rom_scat_pres  (.clk(clk), .addr(BQM_AWV'(scat_rd)), .q(scat_pres_q));
  bp_rom_scat_row_bqm   u_rom_scat_row   (.clk(clk), .addr(BQM_AWV'(scat_rd)), .q(scat_row_q));
  bp_rom_scat_lam_bqm   u_rom_scat_lam   (.clk(clk), .addr(BQM_AWV'(scat_rd)), .q(scat_lam_q));
  bp_rom_benes_ecmrd_bqm u_rom_benes_ecmrd (.clk(clk), .addr(BQM_AWV'(var_rd)), .q(benes_ecmrd_q));
  bp_rom_ecm_readrow_bqm u_rom_ecm_readrow (.clk(clk), .addr(BQM_AWV'(var_rd)), .q(ecm_readrow_q));
  bp_rom_benes_mcmwr_bqm   u_rom_benes_mcmwr   (.clk(clk), .addr(BQM_AWV'(scat_rd)), .q(benes_mcmwr_q));

  always_ff @(posedge clk) begin
    mcm_ra_r <= BWC'(chk_rd);
    mvm_ra_r <= BWV'(var_rd);
  end

  // ===================================================================== M9c site 4: m_cm write-scatter fabric
  // M9c Step 5d: swapped from the power-of-two Beneš network to a SINGLE AS-Waksman network sized to the
  // REAL half-bank count (N=BP_ASW_MCM_N; no port split -- hb = eb*2+beta is injective). Each scatter slot
  // s carries {valid=scat_pres_r[s], row=scat_row_r[s], data=(scat_is_init?scat_lam_r[s]:var_m_flat[s])};
  // the ROM control routes slot s to its half-bank scat_hb_r[s]. At output half-bank b: we_mcm[b]=valid &
  // mcm_we_gate_d, wa_mcm[b]=row, wd_mcm[b]=data. Guard(a) => <=1 valid writer per half-bank per group, so
  // no scatter conflict.
  //
  // PIPE=BENES_PIPE_MCM (4, Fmax-safe: the AS-Waksman route in <=4 registered stages). The scatter din
  // is captured at the same cycle the old combinational write fired (scat_*_r + var_m_flat already aligned
  // by the S_VAR scat cursor), so the m_cm write now LANDS BENES_PIPE_MCM cycles later. The write enable is
  // therefore the PIPE-delayed gate mcm_we_gate_d (a value snapshot -- it fires for exactly the same groups,
  // 4 cycles on); the drain of the last groups' writes spills into the first BENES_PIPE_MCM cycles of the
  // next phase (harmless: those half-banks are not re-read that early -- verified bit-exact by co-sim).
  // M9c Step 5d: N is now BP_ASW_MCM_N (= NHB, the real half-bank count) -- the AS-Waksman fabric is
  // sized exactly to the domain, not the old power-of-two-padded BP_BENES_MCM_M(1024). din padding
  // (lanes [NVB..BP_ASW_MCM_N) filled with 0) and the output consumers (we/wa/wd_mcm[0..NHB)) already
  // matched the NHB(=800) domain before this swap, so only the din array bound moves 1024->800.
  logic [BP_ASW_MCM_N-1:0][1+BWC+MSG_BITS-1:0] mcm_wr_din, mcm_wr_dout;
  always_comb begin
    for (int s = 0; s < BP_ASW_MCM_N; s++)
      mcm_wr_din[s] = (s < NVB)
          ? {scat_pres_r[s], scat_row_r[s],
             unsigned'(scat_is_init ? signed'(scat_lam_r[s]) : var_m_flat[s])}
          : '0;
  end
  bp_asw_mcm_wr #(.N(BP_ASW_MCM_N), .W(1+BWC+MSG_BITS), .PIPE(BENES_PIPE_MCM)) u_asw_wr (
      .clk(clk), .din(mcm_wr_din), .ctrl(benes_mcmwr_q), .dout(mcm_wr_dout));
  // PIPE-delayed write-enable gate: aligns the fabric's now-4-late write with its output valid bit.
  logic mcm_we_gate_pipe [BENES_PIPE_MCM];
  logic mcm_we_gate_d;
  always_ff @(posedge clk) begin
    mcm_we_gate_pipe[0] <= mcm_we_gate;
    for (int s = 1; s < BENES_PIPE_MCM; s++) mcm_we_gate_pipe[s] <= mcm_we_gate_pipe[s-1];
  end
  assign mcm_we_gate_d = mcm_we_gate_pipe[BENES_PIPE_MCM-1];
  always_comb begin
    for (int b = 0; b < NHB; b++) begin
      we_mcm[b] = mcm_we_gate_d & mcm_wr_dout[b][BWC+MSG_BITS];       // valid bit
      wa_mcm[b] = mcm_wr_dout[b][MSG_BITS +: BWC];                    // row
      wd_mcm[b] = signed'(mcm_wr_dout[b][MSG_BITS-1:0]);             // data
    end
  end

  // ===================================================================== M9c site 3: e_cm read-ADDR (readrow ROM)
  // M9c Step 4b: e_cm read rows come straight from BP_ROM_ECM_READROW (the old ad0/ad1 Beneš fabric
  // computed a static per-group permutation of ROM constants). Latency-match the removed fabric: the
  // old `bp_benes_ecm_addr #(.PIPE(BENES_PIPE_ECM))` produced dout BENES_PIPE_ECM cycles after its din
  // was valid, and its din (ecm_ad_din, driven combinationally off var_epres_r/var_erow_r) was valid on
  // the SAME cycle `benes_ecmaddr_q`/`var_epres_q` land (both direct 1-cycle sync ROM reads off the same
  // var_rd address). `ecm_readrow_q` is likewise a direct 1-cycle sync read off var_rd, so it lands on
  // that identical cycle too -- meaning we need exactly BENES_PIPE_ECM registered stages after the comb
  // unpack (not BENES_PIPE_ECM-1) for ra_ecm/rb_ecm to land on the same cycle the old fabric's outputs
  // did. (Empirically confirmed: BENES_PIPE_ECM-1 stages under-delays by exactly one cycle and corrupts
  // decode data, caught by the co-sim 40/40 gate -- see task-2 report.) Every downstream offset
  // (benes_ecmrd_q_d depth, var-operand twins, S_VAR +3, BENES_ECM_LAT) is untouched.
  logic [BQM_BWCW-1:0] ra_ecm_rom [BQM_NEBW];
  logic [BQM_BWCW-1:0] rb_ecm_rom [BQM_NEBW];
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm_rom[b] = ecm_readrow_q[(b*2 + 0)*BQM_BWCW +: BQM_BWCW];
      rb_ecm_rom[b] = ecm_readrow_q[(b*2 + 1)*BQM_BWCW +: BQM_BWCW];
    end
  end
  // BENES_PIPE_ECM pipeline stages (the ROM's own sync read lands on the same cycle the old fabric's
  // combinational din did, so the full PIPE depth -- not PIPE-1 -- has to follow the comb unpack).
  logic [BQM_BWCW-1:0] ra_ecm_d [BENES_PIPE_ECM][BQM_NEBW];
  logic [BQM_BWCW-1:0] rb_ecm_d [BENES_PIPE_ECM][BQM_NEBW];
  always_ff @(posedge clk) begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm_d[0][b] <= ra_ecm_rom[b];
      rb_ecm_d[0][b] <= rb_ecm_rom[b];
    end
    for (int s = 1; s < BENES_PIPE_ECM; s++)
      for (int b = 0; b < NEB; b++) begin
        ra_ecm_d[s][b] <= ra_ecm_d[s-1][b];
        rb_ecm_d[s][b] <= rb_ecm_d[s-1][b];
      end
  end
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = ra_ecm_d[BENES_PIPE_ECM-1][b];
      rb_ecm[b] = rb_ecm_d[BENES_PIPE_ECM-1][b];
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
      bp_mcm_cell_bqm #(.B(b)) u_mcm (
          .clk(clk), .we(we_mcm[b]), .wa(wa_mcm[b]), .wd(wd_mcm[b]), .ra(mcm_ra_r), .q(qmcm[b])
      );
    end
  endgenerate

  // ===================================================================== e_cm bank cells
  generate
    for (genvar b = 0; b < NEB; b++) begin : gecm
      bp_ecm_cell_bqm #(.B(b)) u_ecm (
          .clk(clk), .we(we_ecm[b]), .wa(wa_ecm), .wd(chk_e_flat[b]),
          .ra(ra_ecm[b]), .rb(rb_ecm[b]), .qa(qa_ecm[b]), .qb(qb_ecm[b])
      );
    end
  endgenerate

  // ===================================================================== m_vm bank cells
  generate
    for (genvar b = 0; b < NVB; b++) begin : gmvm
      bp_mvm_cell_bqm #(.B(b)) u_mvm (
          .clk(clk), .we(we_mvm[b]), .wa(wa_mvm), .wd(wd_mvm[b]), .rg(mvm_ra_r), .q(qmvm[b])
      );
    end
  endgenerate

  // ===================================================================== W check_minsum slots
  generate
    for (genvar j = 0; j < W; j++) begin : gchk
      logic                       sbit_sel;
      logic                       sbit_j;
      logic signed [MSG_BITS-1:0] m_in_j    [BP_CHK_DEG];
      logic                       present_j [BP_CHK_DEG];
      // syndrome-bit select: LUT-core CONSTANT-FOLDED form (probe run 4 revert) — `chk_at_bqm(g,j)` folds
      // per group under `pc == g`, so s_reg[BP_C] is never runtime-indexed (Vivado port-splits/explodes on
      // runtime indices into large register arrays; the SAT parity below uses the same shape). The fold
      // has 0 ROM latency, so ONE register stage (sbit_sel -> sbit_j) re-aligns it with the ROM-fed
      // gather data (group pc-1) — the global schedule is unchanged.
      always_comb begin
        sbit_sel = 1'b0;
        for (int g = 0; g < GC; g++)
          if (chk_at_bqm(g, j) >= 0 && pc == g) sbit_sel = s_reg[chk_at_bqm(g, j)];
      end
      always_ff @(posedge clk) sbit_j <= sbit_sel;    // ROM-latency twin stage (aligns with *_r ROM rows)
      // gather from the REGISTERED CHK select ROM (group = pc-1) tapping the REGISTERED-address m_cm reads.
      always_comb begin
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          automatic int idx = j * BP_CHK_DEG + k;
          // M9c: 2:1 beta-split. chk_hbsel_r[idx] == idx*2 + beta (HB = eb*2+beta, eb = idx), so the
          // tap base idx*2 is a compile-time constant and only bit0 (beta) is runtime -> Vivado folds
          // this to a 2:1 mux instead of an NHB-way crossbar. Invariant enforced in rom_contract below.
          m_in_j[k]    = chk_epres_r[idx]
                       ? (chk_hbsel_r[idx][0] ? qmcm[idx*2 + 1] : qmcm[idx*2])
                       : '0;
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

  // ===================================================================== M9c site 2: e_cm read-gather fabric
  // M9c Step 5d: swapped from the power-of-two Beneš network to a ROM-configured AS-Waksman permutation
  // fabric (N=BP_ASW_ECM_N, the real neb bank count) per e_cm port (a/b): rd0 routes qa_ecm, rd1 routes
  // qb_ecm, both driven by the registered per-group control ROM (low/high halves). dout[idx] delivers the
  // operand for var-edge slot idx = i*BP_VAR_DEG+d exactly where the old crossbar tapped
  // qX_ecm[var_ebsel_r[idx]]; the leaf below eport-selects between the two ports, unchanged.
  //
  // PIPE=3 (Fmax-safe). The fabric pipelines ctrl in LOCKSTEP with data (bp_asw.sv, same TIMING CONTRACT
  // as bp_benes.sv commit 1ec6d54): apply (din_t, ctrl_t) at cycle t -> dout at t+PIPE ==
  // aswaksman_apply(ctrl_t, din_t), and a FRESH pair may be applied every cycle (II=1). So the per-cycle
  // registered ROM output benes_ecmrd_q feeds ctrl directly with no hold requirement; the obsolete PIPE=0
  // "ctrl stable one cycle" concern is gone. The 3 registered column-boundaries break the
  // asw_cols(BP_ASW_ECM_N)-deep combinational route into <=3 stages for timing closure (same balanced
  // column-budget guarantee as Beneš -- see bp_asw.sv's DEPTH-BALANCING banner). The e_in operand now
  // emerges 3 cycles later,
  // so EVERY co-launched var_update operand is delayed 3 cycles (var_epres/eport/pres/lam/gam, qmvm) and
  // the S_VAR launch/consume schedule is shifted +3 (see FSM), keeping every var_update input aligned to
  // the same group and the decisions bit-exact.
  // M9c Step 5d: N is now BP_ASW_ECM_N (= NEB, the real e_cm bank count) -- the din padding bound moves
  // 512->400 (the `b < NEB` guard is now always true since BP_ASW_ECM_N==NEB, kept for clarity/safety).
  logic [BP_ASW_ECM_N-1:0][MSG_BITS-1:0] rd0_din, rd0_dout, rd1_din, rd1_dout;
  always_comb begin
    for (int b = 0; b < BP_ASW_ECM_N; b++) begin
      rd0_din[b] = (b < NEB) ? unsigned'(qa_ecm[b]) : '0;
      rd1_din[b] = (b < NEB) ? unsigned'(qb_ecm[b]) : '0;
    end
  end
  // M9c site 3: the addr fabric delays ra_ecm/rb_ecm (hence the async-read qa_ecm/qb_ecm feeding this
  // fabric's din) by BENES_PIPE_ECM. Delay the site-2 read-gather ctrl ROM by the same amount so ctrl and
  // din stay group-aligned at the read fabric input.
  logic [BP_BENES_ECM_PORTS*BENES_ECM_CTRLW-1:0] benes_ecmrd_q_d [BENES_PIPE_ECM];
  always_ff @(posedge clk) begin
    benes_ecmrd_q_d[0] <= benes_ecmrd_q;
    for (int s = 1; s < BENES_PIPE_ECM; s++) benes_ecmrd_q_d[s] <= benes_ecmrd_q_d[s-1];
  end
  bp_asw_ecm_read #(.N(BP_ASW_ECM_N), .W(MSG_BITS), .PIPE(BENES_PIPE_ECM)) u_asw_rd0 (
      .clk(clk), .din(rd0_din),
      .ctrl(benes_ecmrd_q_d[BENES_PIPE_ECM-1][0*BENES_ECM_CTRLW +: BENES_ECM_CTRLW]), .dout(rd0_dout));
  bp_asw_ecm_read #(.N(BP_ASW_ECM_N), .W(MSG_BITS), .PIPE(BENES_PIPE_ECM)) u_asw_rd1 (
      .clk(clk), .din(rd1_din),
      .ctrl(benes_ecmrd_q_d[BENES_PIPE_ECM-1][1*BENES_ECM_CTRLW +: BENES_ECM_CTRLW]), .dout(rd1_dout));

  // ------------------------------------------------------- M9c: PIPE-deep var-operand alignment delay
  // The Beneš e_in output is delayed BENES_ECM_LAT cycles vs the group-pc-1 `_r` stage that feeds the addr
  // fabric din/ctrl (site-3 addr PIPE=3 -> async e_cm read -> site-2 read PIPE=3). Every OTHER var_update
  // operand co-launched with e_in (the epres gate, the eport leaf select, the m_vc read qmvm, and
  // lam/gam/pres) is registered through an identical BENES_ECM_LAT-deep shift so all inputs of a given var
  // group arrive on the SAME cycle -> the gather comb, gather register plane and var_update below are
  // byte-identical to baseline, only fed 6 cycles later. Free-running (like the M8 gather plane); the
  // en_var_r8 fence and the +6 S_VAR schedule keep pre-launch snapshots out and re-time the launch/consume
  // window. Deepest stage index = BENES_ECM_LAT-1.
  logic                       var_epres_d [BENES_ECM_LAT][NVB];
  logic                       var_eport_d [BENES_ECM_LAT][NVB];
  logic signed [MSG_BITS-1:0] qmvm_d      [BENES_ECM_LAT][NVB];
  logic                       var_pres_d  [BENES_ECM_LAT][V];
  logic [MSG_BITS-1:0]        var_lam_d   [BENES_ECM_LAT][V];
  logic [MSG_BITS-1:0]        var_gam_d   [BENES_ECM_LAT][V];
  always_ff @(posedge clk) begin
    for (int b = 0; b < NVB; b++) begin
      var_epres_d[0][b] <= var_epres_r[b];
      var_eport_d[0][b] <= var_eport_r[b];
      qmvm_d[0][b]      <= qmvm[b];
    end
    for (int i = 0; i < V; i++) begin
      var_pres_d[0][i] <= var_pres_r[i];
      var_lam_d[0][i]  <= var_lam_r[i];
      var_gam_d[0][i]  <= var_gam_r[i];
    end
    for (int s = 1; s < BENES_ECM_LAT; s++) begin
      for (int b = 0; b < NVB; b++) begin
        var_epres_d[s][b] <= var_epres_d[s-1][b];
        var_eport_d[s][b] <= var_eport_d[s-1][b];
        qmvm_d[s][b]      <= qmvm_d[s-1][b];
      end
      for (int i = 0; i < V; i++) begin
        var_pres_d[s][i] <= var_pres_d[s-1][i];
        var_lam_d[s][i]  <= var_lam_d[s-1][i];
        var_gam_d[s][i]  <= var_gam_d[s-1][i];
      end
    end
  end

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
        // M9c: all operands read from the BENES_ECM_LAT-delayed twins so they align with the e_in output
        // (rd*_dout, itself BENES_ECM_LAT=6 cycles behind the group-pc-1 addr-fabric din/ctrl feed).
        lam_i = var_pres_d[BENES_ECM_LAT-1][i] ? signed'(var_lam_d[BENES_ECM_LAT-1][i]) : '0;
        gam_i = var_pres_d[BENES_ECM_LAT-1][i] ? signed'(var_gam_d[BENES_ECM_LAT-1][i]) : '0;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int idx = i * BP_VAR_DEG + d;
          if (var_epres_d[BENES_ECM_LAT-1][idx]) begin
            // M9c: e_cm operand from the Beneš read-gather (dout[idx]), eport-selected between ports a/b.
            e_in_i[d]    = var_eport_d[BENES_ECM_LAT-1][idx] ? signed'(rd1_dout[idx]) : signed'(rd0_dout[idx]);
            m_in_i[d]    = qmvm_d[BENES_ECM_LAT-1][idx];
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
          .en      (en_var_r8),
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

        // ----------------------------------- seed m_cm/m_vm with lambda (pc=0..GV+4, M9c: GV..GV+3 drain
        // BENES_PIPE_MCM; write at pc-1)
        S_INIT: begin
          // M9c site 4: extend by BENES_PIPE_MCM (GV -> GV+4) so the m_cm seed write-scatter fabric fully
          // drains before S_CHECK reads m_cm (the write now lands PIPE=4 cycles after its din cycle).
          if (pc == GV + 4) begin pc <= '0; state <= S_CHECK; end
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
                if (chk_at_bqm(g, j) >= 0 && pc == g) begin
                  p = s_reg[chk_at_bqm(g, j)];
                  for (int k = 0; k < BP_CHK_DEG; k++) begin
                    automatic int e = edge_at_bqm(g, j, k);
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
          // e_cm scatter of group pc-5 (M9b lag) handled by the ROM-fed per-bank bp_ecm_cell_bqm write.
          if (early_exit && final_sat) begin
            pc <= '0; state <= S_EMIT;
          end else if (pc == GC + 4) begin              // M9b: was GC+3 (extra ROM-read plane +1)
            pc      <= '0;
            all_sat <= 1'b1;
            state   <= S_VAR;
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch var group `pc-1` + scatter group pc-10 (M9c)
        S_VAR: begin
          automatic logic wterm [V];
          automatic int   wsum;
          wsum = 0;
          if (pc == 0) ehat_w <= '0;
          if (pc >= 10) begin                         // M9c: var scatter lag pc-10 (M9b pc-4 + BENES_ECM_LAT-ish)
            for (int i = 0; i < V; i++) wterm[i] = 1'b0;
            for (int i = 0; i < V; i++)
              for (int g = 0; g < GV; g++)
                if (var_at_bqm(g, i) >= 0 && (pc - 10) == g) begin
                  automatic int v = var_at_bqm(g, i);
                  ehat[v] <= var_ehat_out[i];
                  wterm[i] = var_ehat_out[i];
                end
            for (int i = 0; i < V; i++) wsum = wsum + (wterm[i] ? 1 : 0);
            ehat_w <= ehat_w + WW'(wsum);
          end
          // m_vm / m_cm writes of group pc-10 (M9c lag) handled by the bp_mvm_cell_bqm + m_cm scatter fabric.
          // M9c site 4: extend by BENES_PIPE_MCM (GV+9 -> GV+13) so the m_cm write-scatter fabric drains
          // fully before the next S_CHECK reads m_cm.
          if (pc == GV + 13) begin                      // M9c: site-3 GV+9 + site-4 fabric drain +4
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
              if (chk_at_bqm(g, j) >= 0 && pc == g) begin
                p = s_reg[chk_at_bqm(g, j)];
                for (int k = 0; k < BP_CHK_DEG; k++) begin
                  automatic int e = edge_at_bqm(g, j, k);
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
        // LUT-core CONSTANT-FOLDED form (probe run 4 revert): the ROM form runtime-indexed ehat/best_e
        // [BP_N] and write-decoded corr_out[BP_N] — the closure whale. `var_at_bqm(g,i)` folds per group
        // under `pc == g` (M8's 12e split: per-slot term with its own constant-folded group mux, then a
        // pure XOR reduction — associative, bit-exact). Phase is pc=0..GV-1 again (the +1 ROM tail is
        // gone), so the decode is ONE cycle shorter than the ROM-form core; decisions are unchanged.
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
              if (var_at_bqm(g, i) >= 0 && pc == g) begin
                automatic int v = var_at_bqm(g, i);
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
