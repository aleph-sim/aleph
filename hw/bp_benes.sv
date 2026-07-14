// Q7-04 M9c Step 2.3 — parameterized Beneš rearrangeable permutation-network fabrics.
//
// `bp_benes_switch` is the shared 2x2 crossbar primitive. `bp_benes_block` is the recursive
// routing core that mirrors `benes_apply`/`apply_block` in crates/aleph-qec/src/benes.rs
// EXACTLY: same column-major control-bit addressing `ctrl[col*(M/2)+switch]` (M = global network
// width, i.e. the Rust `stride` = M/2 argument that stays constant across the whole recursion),
// same recursive up/lo block split (n inputs -> n/2 switches -> two n/2-wide sub-blocks), same
// bar/cross convention (ctrl bit false = straight, true = cross). This structural 1:1
// correspondence — same (col0,row0,stride) bookkeeping the Rust `route`/`apply_block` share — is
// what lets the emitter's control ROMs (`benes_control`, gen-time-guarded against `benes_apply`)
// drive this fabric correctly with no independent re-derivation of the wiring formula.
//
// PIPE registers: recursion depth d deterministically fixes TWO global column indices for every
// node at that depth, regardless of which branch of the tree it is (down-recursion always halves
// N uniformly, so every node at depth d operates on the same block size M/2^d and therefore the
// same (COL0=d, OUT_COL=2*log2(M)-2-d)) — the "down" column where this block's own switches read
// the input, and the "up" column where its own switches assemble the final output from its two
// children. Because a column boundary's register decision (`floor((c+1)*PIPE/COLS_TOTAL) !=
// floor(c*PIPE/COLS_TOTAL)`, i.e. partition the COLS_TOTAL columns into PIPE even groups) depends
// only on (c, PIPE, COLS_TOTAL) — all constant across the whole recursion — every node at a given
// depth makes the identical register decision, so every column boundary is registered (or not)
// UNIFORMLY across the full width. Consequently every input->output path crosses exactly PIPE
// registered boundaries: latency is exactly PIPE cycles for every lane, not just on average.
//
// Ref: Beneš (1965); Lee, "On the Rearrangeability of 2(log2 N)-1 Stage Permutation Networks",
// IEEE ToC C-34(5), 1985 (looping algorithm) — see benes.rs module doc for the full citation.
//
// TIMING CONTRACT (Step 2.3b): `ctrl` now travels WITH its own `din` through the pipeline, so a
// FRESH `(din, ctrl)` pair may be applied every cycle (initiation interval = 1) -- there is no
// hold requirement. Feeding `(din_t, ctrl_t)` at cycle `t` produces `dout` at cycle `t+PIPE` equal
// to `benes_apply(ctrl_t, din_t)`; overlapping in-flight cases never corrupt each other, because
// each switch column reads a delayed COPY of `ctrl` rather than the live port. Concretely: the top
// -level site modules (`bp_benes_ecm_read`/`_addr`, `bp_benes_mcm_wr`) register the whole `ctrl`
// vector through a PIPE-deep shift chain (`ctrl_pipe[0..PIPE]`, `ctrl_pipe[0]` = the live port,
// `ctrl_pipe[k]` = that same vector delayed by `k` cycles) and pass the WHOLE chain down through
// the recursion unchanged. Column `c` (0-indexed in the flattened COLS_TOTAL column order) reads
// `ctrl_pipe[stage(c)]`, where `stage(c) = floor(c*PIPE/COLS_TOTAL)` is EXACTLY the number of
// data-pipeline registers upstream of that column -- the same telescoping sum that the `REG_AFTER*`
// register-placement decisions below are built from, so `ctrl` and `din` always arrive at a given
// column having crossed the identical number of registered boundaries. This replaces the old
// "hold ctrl stable for >= PIPE cycles" contract, which the QEC core's control-ROM (whose output
// changes every cycle -- a 1-cycle group window) could not meet. `PIPE=0` is unaffected: it stays
// fully combinational, `dout` at cycle `t` for `din_t`/`ctrl_t` applied at cycle `t`.

`ifndef BP_BENES_SV
`define BP_BENES_SV

// File holds four modules (switch leaf + recursive core + three site-specific tops), none named
// `bp_benes` -- DECLFILENAME is fenced for the whole file, mirroring the existing convention in
// bp_relay_banked.sv / bp_relay_banked_bram.sv / bp_relay_banked_bram_m.sv.
/* verilator lint_off DECLFILENAME */

// 2x2 crossbar switch: sel=0 -> straight (a_out=a_in, b_out=b_in), sel=1 -> cross (a_out=b_in,
// b_out=a_in). Pure combinational. Matches benes.rs::apply_block's n==2 base case exactly:
// `if ctrl { [input[1], input[0]] } else { [input[0], input[1]] }`.
module bp_benes_switch #(
    parameter int W = 1
) (
    input  logic         sel,
    input  logic [W-1:0] a_in,
    input  logic [W-1:0] b_in,
    output logic [W-1:0] a_out,
    output logic [W-1:0] b_out
);
  assign a_out = sel ? b_in : a_in;
  assign b_out = sel ? a_in : b_in;
endmodule

// Recursive Beneš routing block. N = this block's width (elements, power of 2, >=2); M = the
// GLOBAL network width (constant across the whole recursion -> ctrl stride = M/2, the Rust
// `stride`); PIPE = the GLOBAL total pipeline depth (also constant across the recursion); COL0 /
// ROW0 place this block's switches in the shared, flattened, column-major `ctrl` vector exactly
// as `route`/`apply_block`'s (col0, row0, stride) recursion does — see benes.rs `route` /
// `apply_block` for the reference recursion this mirrors 1:1.
module bp_benes_block #(
    parameter int N,
    parameter int W,
    parameter int M,
    parameter int PIPE,
    parameter int COL0,
    parameter int ROW0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    // Leaf (N==2) instances whose own column falls on a non-registered boundary never use `clk`
    // locally (they have no children to forward it to) — intentional; every OTHER instance
    // forwards `clk` to its two children, and the module as a whole always registers exactly
    // PIPE column boundaries somewhere in the tree.
    input  logic clk,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [N-1:0][W-1:0]              din,
    // ctrl_pipe[k] = the top-level fabric's `ctrl` input delayed by exactly k clock cycles
    // (k=0..PIPE; ctrl_pipe[0] is the live, unregistered port). Built ONCE by the site-specific
    // top module (see e.g. `bp_benes_ecm_read`) and threaded down through the recursion
    // unmodified. Column `c` (this block's own COL0 for input switches, OUT_COL for output
    // switches) reads `ctrl_pipe[stage(c)]` where `stage(c) = floor(c*PIPE/COLS_TOTAL)` is the
    // number of data-pipeline register boundaries upstream of that column -- see file banner.
    input  logic [(2*$clog2(M)-1)*(M/2)-1:0] ctrl_pipe [0:PIPE],
    output logic [N-1:0][W-1:0]              dout
);
  localparam int STRIDE     = M / 2;
  localparam int COLS_TOTAL = 2 * $clog2(M) - 1;

  generate
    if (N == 2) begin : g_base
      localparam bit REG_AFTER =
          (((COL0 + 1) * PIPE) / COLS_TOTAL) != ((COL0 * PIPE) / COLS_TOTAL);
      localparam int CTRL_STAGE = (COL0 * PIPE) / COLS_TOTAL;

      logic [1:0][W-1:0] dout_c;
      bp_benes_switch #(.W(W)) u_sw (
          .sel  (ctrl_pipe[CTRL_STAGE][COL0*STRIDE + ROW0]),
          .a_in (din[0]),
          .b_in (din[1]),
          .a_out(dout_c[0]),
          .b_out(dout_c[1])
      );

      if (REG_AFTER) begin : g_reg
        always_ff @(posedge clk) dout <= dout_c;
      end else begin : g_comb
        assign dout = dout_c;
      end
    end else begin : g_rec
      localparam int HALF    = N / 2;
      localparam int COLS_N  = 2 * $clog2(N) - 1;
      localparam int OUT_COL = COL0 + COLS_N - 1;
      localparam bit REG_AFTER_IN =
          (((COL0 + 1) * PIPE) / COLS_TOTAL) != ((COL0 * PIPE) / COLS_TOTAL);
      localparam bit REG_AFTER_OUT =
          (((OUT_COL + 1) * PIPE) / COLS_TOTAL) != ((OUT_COL * PIPE) / COLS_TOTAL);
      localparam int CTRL_STAGE_IN  = (COL0 * PIPE) / COLS_TOTAL;
      localparam int CTRL_STAGE_OUT = (OUT_COL * PIPE) / COLS_TOTAL;

      logic [HALF-1:0][W-1:0] upper_in_c, lower_in_c;
      logic [HALF-1:0][W-1:0] upper_in_r, lower_in_r;
      logic [HALF-1:0][W-1:0] upper_out, lower_out;
      logic [N-1:0][W-1:0]    dout_c;

      genvar isw;
      for (isw = 0; isw < HALF; isw++) begin : g_in_sw
        bp_benes_switch #(.W(W)) u_sw_in (
            .sel  (ctrl_pipe[CTRL_STAGE_IN][COL0*STRIDE + ROW0 + isw]),
            .a_in (din[2*isw]),
            .b_in (din[2*isw+1]),
            .a_out(upper_in_c[isw]),
            .b_out(lower_in_c[isw])
        );
      end

      if (REG_AFTER_IN) begin : g_reg_in
        always_ff @(posedge clk) begin
          upper_in_r <= upper_in_c;
          lower_in_r <= lower_in_c;
        end
      end else begin : g_comb_in
        assign upper_in_r = upper_in_c;
        assign lower_in_r = lower_in_c;
      end

      bp_benes_block #(
          .N(HALF), .W(W), .M(M), .PIPE(PIPE), .COL0(COL0 + 1), .ROW0(ROW0)
      ) u_up (
          .clk      (clk),
          .din      (upper_in_r),
          .ctrl_pipe(ctrl_pipe),
          .dout     (upper_out)
      );
      bp_benes_block #(
          .N(HALF), .W(W), .M(M), .PIPE(PIPE), .COL0(COL0 + 1), .ROW0(ROW0 + HALF / 2)
      ) u_lo (
          .clk      (clk),
          .din      (lower_in_r),
          .ctrl_pipe(ctrl_pipe),
          .dout     (lower_out)
      );

      genvar osw;
      for (osw = 0; osw < HALF; osw++) begin : g_out_sw
        bp_benes_switch #(.W(W)) u_sw_out (
            .sel  (ctrl_pipe[CTRL_STAGE_OUT][OUT_COL*STRIDE + ROW0 + osw]),
            .a_in (upper_out[osw]),
            .b_in (lower_out[osw]),
            .a_out(dout_c[2*osw]),
            .b_out(dout_c[2*osw+1])
        );
      end

      if (REG_AFTER_OUT) begin : g_reg_out
        always_ff @(posedge clk) dout <= dout_c;
      end else begin : g_comb_out
        assign dout = dout_c;
      end
    end
  endgenerate
endmodule

// ---------------------------------------------------------------------------------------------
// Site-specific top-level fabrics. Each roots one `bp_benes_block` at (M=N, COL0=0, ROW0=0) --
// i.e. the whole network is one top-level block spanning all COLS = 2*$clog2(N)-1 columns. Kept
// as three separate modules (not one generic fabric) per the M9c design decision: e_cm
// read-gather, e_cm addr-gather, and m_cm write-scatter are wired at different sites in the core
// by Tasks 4/5 even though they share this leaf cell (`bp_benes_switch`) and routing core
// (`bp_benes_block`). Widths (N/W/PIPE) are per-site parameters set by the caller; payload width
// W is generic — a permutation fabric doesn't care what it's routing.
// ---------------------------------------------------------------------------------------------

module bp_benes_ecm_read #(
    parameter int N,
    parameter int W,
    parameter int PIPE
) (
    input  logic                             clk,
    input  logic [N-1:0][W-1:0]              din,
    // TIMING CONTRACT (Step 2.3b): `ctrl` travels WITH its own `din` -- a fresh (din,ctrl) pair
    // may be applied every cycle, dout at t+PIPE == benes_apply(ctrl_t,din_t). No hold
    // requirement (see file banner). Registered internally into `ctrl_pipe` below.
    input  logic [(2*$clog2(N)-1)*(N/2)-1:0] ctrl,
    output logic [N-1:0][W-1:0]              dout
);
  localparam int CTRL_W = (2 * $clog2(N) - 1) * (N / 2);

  // ctrl_pipe[k] = `ctrl` delayed by k cycles (k=0..PIPE); ctrl_pipe[0] is the live port. Shared,
  // unmodified, by the whole `bp_benes_block` recursion below -- see file banner / block's
  // `ctrl_pipe` port comment for how each switch column picks its own delayed copy.
  logic [CTRL_W-1:0] ctrl_pipe[0:PIPE];
  assign ctrl_pipe[0] = ctrl;
  genvar k;
  generate
    for (k = 1; k <= PIPE; k++) begin : g_ctrl_pipe
      always_ff @(posedge clk) ctrl_pipe[k] <= ctrl_pipe[k-1];
    end
  endgenerate

  bp_benes_block #(.N(N), .W(W), .M(N), .PIPE(PIPE), .COL0(0), .ROW0(0)) u_core (
      .clk (clk), .din(din), .ctrl_pipe(ctrl_pipe), .dout(dout)
  );
endmodule

module bp_benes_ecm_addr #(
    parameter int N,
    parameter int W,
    parameter int PIPE
) (
    input  logic                             clk,
    input  logic [N-1:0][W-1:0]              din,
    // TIMING CONTRACT (Step 2.3b): `ctrl` travels WITH its own `din` -- a fresh (din,ctrl) pair
    // may be applied every cycle, dout at t+PIPE == benes_apply(ctrl_t,din_t). No hold
    // requirement (see file banner). Registered internally into `ctrl_pipe` below.
    input  logic [(2*$clog2(N)-1)*(N/2)-1:0] ctrl,
    output logic [N-1:0][W-1:0]              dout
);
  localparam int CTRL_W = (2 * $clog2(N) - 1) * (N / 2);

  logic [CTRL_W-1:0] ctrl_pipe[0:PIPE];
  assign ctrl_pipe[0] = ctrl;
  genvar k;
  generate
    for (k = 1; k <= PIPE; k++) begin : g_ctrl_pipe
      always_ff @(posedge clk) ctrl_pipe[k] <= ctrl_pipe[k-1];
    end
  endgenerate

  bp_benes_block #(.N(N), .W(W), .M(N), .PIPE(PIPE), .COL0(0), .ROW0(0)) u_core (
      .clk (clk), .din(din), .ctrl_pipe(ctrl_pipe), .dout(dout)
  );
endmodule

module bp_benes_mcm_wr #(
    parameter int N,
    parameter int W,
    parameter int PIPE
) (
    input  logic                             clk,
    input  logic [N-1:0][W-1:0]              din,
    // TIMING CONTRACT (Step 2.3b): `ctrl` travels WITH its own `din` -- a fresh (din,ctrl) pair
    // may be applied every cycle, dout at t+PIPE == benes_apply(ctrl_t,din_t). No hold
    // requirement (see file banner). Registered internally into `ctrl_pipe` below.
    input  logic [(2*$clog2(N)-1)*(N/2)-1:0] ctrl,
    output logic [N-1:0][W-1:0]              dout
);
  localparam int CTRL_W = (2 * $clog2(N) - 1) * (N / 2);

  logic [CTRL_W-1:0] ctrl_pipe[0:PIPE];
  assign ctrl_pipe[0] = ctrl;
  genvar k;
  generate
    for (k = 1; k <= PIPE; k++) begin : g_ctrl_pipe
      always_ff @(posedge clk) ctrl_pipe[k] <= ctrl_pipe[k-1];
    end
  endgenerate

  bp_benes_block #(.N(N), .W(W), .M(N), .PIPE(PIPE), .COL0(0), .ROW0(0)) u_core (
      .clk (clk), .din(din), .ctrl_pipe(ctrl_pipe), .dout(dout)
  );
endmodule

/* verilator lint_on DECLFILENAME */

`endif
