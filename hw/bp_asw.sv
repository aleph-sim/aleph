// Q7-04 M9c Step 5c — parameterized ARBITRARY-SIZE (AS-)Waksman rearrangeable permutation-network
// fabrics. Sibling of `bp_benes.sv`, but for ANY input count N (including odd), not just powers of
// two. Two site-specific tops -- `bp_asw_ecm_read` (e_cm read gather, N=400) and `bp_asw_mcm_wr`
// (m_cm write scatter, N=800) -- share the `bp_asw_switch` 2x2 leaf cell and the recursive
// `bp_asw_block` routing core.
//
// STRUCTURAL CONTRACT (hard): `bp_asw_block` mirrors `route`/`apply_block` in
// crates/aleph-qec/src/aswaksman.rs EXACTLY. For a block of `n` inputs:
//   * an input stage of floor(n/2) switches on pairs (2i, 2i+1); if n is ODD the last input n-1
//     bypasses straight into the UPPER subnet at position ceil(n/2)-1;
//   * an upper subnet of size ceil(n/2), a lower subnet of size floor(n/2);
//   * an output stage of ceil(n/2)-1 switches on output pairs (2j, 2j+1); the last output(s) are
//     HARDWIRED (Waksman's removed switch): out[sw_out] = upper subnet's last output; for EVEN n
//     also out[sw_out+1] = lower subnet's last output. (sw_out = 2*(ceil(n/2)-1).)
// The flat `ctrl` vector is indexed by a RUNNING switch counter in recursion order:
//     [ input switches of this block ] [ entire upper subnet ] [ entire lower subnet ]
//     [ output switches of this block ]
// `BASE` is this block's first switch index in that flat vector; the child bases are threaded with
// `asw_sw_count(m_up)` / `asw_sw_count(m_lo)` exactly as `route` threads its running offset. This
// 1:1 correspondence is what lets the emitter's `aswaksman_control` ROM drive this fabric with no
// independent re-derivation of the wiring. bar (ctrl=0) = straight, cross (ctrl=1) = swap, matching
// `bp_asw_switch` / `apply_block`.
//
// DEPTH-BALANCING (the spec's key design risk). Beneš gets "exactly PIPE cycles for every lane" for
// free because every recursion node at a given depth has the SAME block size, so all root->leaf
// paths cross the same column count. AS-Waksman's arbitrary/odd split does NOT: a block's upper
// subnet (ceil(n/2)) and lower subnet (floor(n/2)) can have DIFFERENT natural recursion depths
// (e.g. n=5 -> upper=3 [3 cols] vs lower=2 [1 col]), so raw path lengths are non-uniform and the
// single-global-column-count timing contract (stage(c)=floor(c*PIPE/COLS_TOTAL)) would place a
// lane's PIPE registers inconsistently. We RESTORE uniformity by allocating BOTH sibling subnets
// the SAME column budget: define the balanced column width
//     COLS(n) = 0 (n<=1), 1 (n==2), 2 + COLS(ceil(n/2)) (n>=3).
// Because COLS is monotonic and ceil(n/2) >= floor(n/2), COLS(ceil(n/2)) >= COLS(floor(n/2)), so a
// block of balanced width COLS(n) spends 1 input column + COLS(ceil(n/2)) middle columns +
// 1 output column, and BOTH children are handed the middle budget COLS(n)-2 (= COLS(ceil(n/2))) via
// the `COLBUDGET` parameter. The upper child fills it exactly; the SHALLOWER lower child pads the
// difference COLS(ceil(n/2)) - COLS(floor(n/2)) with straight-through delay columns absorbed
// RECURSIVELY inside its own subtree (a size-1 leaf becomes a pure delay line; a size-2 leaf is a
// switch followed by delay columns). Padding consumes no `ctrl` bits, so the running-counter layout
// is untouched -- it costs only a few flops (fine: FFs at 43.6%, BRAM/LUT are the tight resources).
// Net effect: every root->output path crosses exactly COLS(N) columns, hence exactly COLS(N)
// registered-or-not boundaries, PIPE of which are registered by the SAME uniform placement rule
// `bp_benes.sv` uses -> latency is exactly PIPE for every lane, not just on average.
//
// Ref: Beauquier & Darrot, "On Arbitrary Size Waksman Networks", 2002; Waksman, "A Permutation
// Network", J. ACM 15(1), 1968 -- see the aswaksman.rs module doc for the full citation.
//
// TIMING CONTRACT (identical to bp_benes.sv Step 2.3b): `ctrl` travels WITH its own `din` through
// the pipeline, so a FRESH (din,ctrl) pair may be applied every cycle (initiation interval = 1),
// dout at cycle t+PIPE == aswaksman_apply(ctrl_t, din_t). The site tops register the whole `ctrl`
// vector through a PIPE-deep shift chain `ctrl_pipe[0:PIPE]` (ctrl_pipe[0] = live port, ctrl_pipe[k]
// = ctrl delayed k cycles) and thread the WHOLE chain unchanged through the recursion. A switch at
// global column c reads ctrl_pipe[stage(c)] with stage(c)=floor(c*PIPE/COLS_TOTAL) = the number of
// data-pipeline registers upstream of c, so ctrl and din always arrive at a column having crossed
// the identical number of registered boundaries. PIPE=0 stays fully combinational.

`ifndef BP_ASW_SV
`define BP_ASW_SV

// File holds a switch leaf + the recursive core + two site tops, none named `bp_asw` -- DECLFILENAME
// fenced for the whole file, mirroring bp_benes.sv / bp_relay_banked*.sv.
/* verilator lint_off DECLFILENAME */

// ---------------------------------------------------------------------------------------------
// Compile-time helpers. Both are the same recursion the routing uses, so the RTL's switch layout
// and column budget agree with aswaksman.rs by construction.
// ---------------------------------------------------------------------------------------------

// Number of 2x2 switches in an AS-Waksman network on `n` inputs -- MUST match aswaksman.rs's
// `aswaksman_switch_count` exactly (closed form ceil(n*log2 n) - n + 1; n=400->3089, n=512->4097,
// n=800->6977). floor(n/2) input + (ceil(n/2)-1) output switches (= n-1 at this level) + both subnets.
function automatic int asw_sw_count(int n);
  if (n <= 1) return 0;
  return (n - 1) + asw_sw_count((n + 1) / 2) + asw_sw_count(n / 2);
endfunction

// Balanced column width COLS(n) (see DEPTH-BALANCING in the file banner). n==2 is special because it
// has NO output-switch column (out_count = ceil(2/2)-1 = 0): just the single input switch.
function automatic int asw_cols(int n);
  if (n <= 1) return 0;
  if (n == 2) return 1;
  return 2 + asw_cols((n + 1) / 2);
endfunction

// 2x2 crossbar: sel=0 -> straight (a_out=a_in, b_out=b_in), sel=1 -> cross. Pure combinational.
// Matches apply_block's bar/cross convention exactly.
module bp_asw_switch #(
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

// ---------------------------------------------------------------------------------------------
// Recursive AS-Waksman routing block.
//   N        = this block's width (>=1).
//   NTOP     = GLOBAL network width (constant across the whole recursion) -> ctrl vector width
//              asw_sw_count(NTOP) and COLS_TOTAL = asw_cols(NTOP).
//   PIPE     = GLOBAL pipeline depth (constant across the recursion).
//   COL0     = global column index of this block's INPUT switch stage.
//   BASE     = flat `ctrl` offset of this block's first (input) switch, in running-counter order.
//   COLBUDGET= columns allotted to this block (>= asw_cols(N); the excess is depth-balancing pad).
// This block spans global columns [COL0, COL0+COLBUDGET): input stage at COL0, output stage at
// COL0+COLBUDGET-1, both children at COL0+1 with budget COLBUDGET-2. See file banner.
// ---------------------------------------------------------------------------------------------
module bp_asw_block #(
    parameter int N,
    parameter int NTOP,
    parameter int W,
    parameter int PIPE,
    parameter int COL0,
    parameter int BASE,
    parameter int COLBUDGET
) (
    /* verilator lint_off UNUSEDSIGNAL */
    // Pure-delay / non-registered leaf instances never use `clk` locally; every other instance
    // forwards it to children / its own registers. Likewise a pure-delay (N<=1) leaf -- the
    // terminus of a depth-balancing pad -- has no switches, so it never reads `ctrl_pipe`. Both are
    // intentional (clk mirrors bp_benes.sv; the ctrl_pipe case is new to the arbitrary-size fabric).
    input logic clk,
    input  logic [N-1:0][W-1:0]                din,
    // ctrl_pipe[k] = the top fabric's `ctrl` delayed by k cycles (k=0..PIPE); ctrl_pipe[0] = live
    // port. Threaded unchanged through the recursion; column c reads ctrl_pipe[floor(c*PIPE/COLS_TOTAL)].
    input  logic [asw_sw_count(NTOP)-1:0]      ctrl_pipe [0:PIPE],
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [N-1:0][W-1:0]                dout
);
  localparam int COLS_TOTAL = asw_cols(NTOP);

  generate
    if (N <= 1) begin : g_wire
      // Pure delay line of COLBUDGET registered-or-not boundaries on the single lane -- the leaf a
      // depth-balancing pad terminates in (see file banner). No switches, no ctrl bits.
      logic [W-1:0] chain[0:COLBUDGET];
      assign chain[0] = din[0];
      genvar c;
      for (c = 0; c < COLBUDGET; c++) begin : g_dl
        localparam int GC = COL0 + c;
        if ((((GC + 1) * PIPE) / COLS_TOTAL) != ((GC * PIPE) / COLS_TOTAL)) begin : g_r
          always_ff @(posedge clk) chain[c+1] <= chain[c];
        end else begin : g_c
          assign chain[c+1] = chain[c];
        end
      end
      assign dout[0] = chain[COLBUDGET];

    end else if (N == 2) begin : g_leaf2
      // Single input switch at column COL0 (ctrl index BASE; out_count=0 -> no output switch), then
      // COLBUDGET boundaries of delay to fill this leaf's (possibly padded) budget.
      localparam int CS = (COL0 * PIPE) / COLS_TOTAL;
      logic [1:0][W-1:0] sw;
      bp_asw_switch #(.W(W)) u_sw (
          .sel  (ctrl_pipe[CS][BASE]),
          .a_in (din[0]),
          .b_in (din[1]),
          .a_out(sw[0]),
          .b_out(sw[1])
      );
      logic [1:0][W-1:0] chain[0:COLBUDGET];
      assign chain[0] = sw;
      genvar c;
      for (c = 0; c < COLBUDGET; c++) begin : g_dl
        localparam int GC = COL0 + c;
        if ((((GC + 1) * PIPE) / COLS_TOTAL) != ((GC * PIPE) / COLS_TOTAL)) begin : g_r
          always_ff @(posedge clk) chain[c+1] <= chain[c];
        end else begin : g_c
          assign chain[c+1] = chain[c];
        end
      end
      assign dout = chain[COLBUDGET];

    end else begin : g_rec
      localparam int IN_CNT   = N / 2;              // floor(n/2) input switches
      localparam int OUT_CNT  = (N + 1) / 2 - 1;    // ceil(n/2)-1 output switches
      localparam int M_UP     = (N + 1) / 2;        // upper subnet size
      localparam int M_LO     = N / 2;              // lower subnet size
      localparam int HAS_BYP  = N % 2;              // odd n -> input n-1 bypasses to upper
      localparam int SW_OUT   = 2 * OUT_CNT;        // outputs 0..SW_OUT switched; rest hardwired
      localparam int BASE_UP  = BASE + IN_CNT;
      localparam int BASE_LO  = BASE_UP + asw_sw_count(M_UP);
      localparam int BASE_OUT = BASE_LO + asw_sw_count(M_LO);
      localparam int CHILDB   = COLBUDGET - 2;      // balanced budget handed to BOTH children
      localparam int OUT_COL  = COL0 + COLBUDGET - 1;
      localparam int CS_IN    = (COL0 * PIPE) / COLS_TOTAL;
      localparam int CS_OUT   = (OUT_COL * PIPE) / COLS_TOTAL;
      localparam bit REG_IN   =
          (((COL0 + 1) * PIPE) / COLS_TOTAL) != ((COL0 * PIPE) / COLS_TOTAL);
      localparam bit REG_OUT  =
          (((OUT_COL + 1) * PIPE) / COLS_TOTAL) != ((OUT_COL * PIPE) / COLS_TOTAL);

      logic [M_UP-1:0][W-1:0] upper_in_c, upper_in_r, upper_out;
      logic [M_LO-1:0][W-1:0] lower_in_c, lower_in_r, lower_out;
      logic [N-1:0][W-1:0]    dout_c;

      // Input stage: switch i splits (din[2i], din[2i+1]) into (upper_in[i], lower_in[i]).
      genvar isw;
      for (isw = 0; isw < IN_CNT; isw++) begin : g_in_sw
        bp_asw_switch #(.W(W)) u_in (
            .sel  (ctrl_pipe[CS_IN][BASE + isw]),
            .a_in (din[2*isw]),
            .b_in (din[2*isw+1]),
            .a_out(upper_in_c[isw]),
            .b_out(lower_in_c[isw])
        );
      end
      // Odd n: last input bypasses straight into the upper subnet's last input position.
      if (HAS_BYP != 0) begin : g_byp
        assign upper_in_c[M_UP-1] = din[N-1];
      end

      if (REG_IN) begin : g_reg_in
        always_ff @(posedge clk) begin
          upper_in_r <= upper_in_c;
          lower_in_r <= lower_in_c;
        end
      end else begin : g_comb_in
        assign upper_in_r = upper_in_c;
        assign lower_in_r = lower_in_c;
      end

      // Both children start at COL0+1 with the SAME balanced budget CHILDB (depth-balancing): the
      // upper child (M_UP) fills it exactly; the lower child (M_LO) pads the slack inside its subtree.
      bp_asw_block #(
          .N(M_UP), .NTOP(NTOP), .W(W), .PIPE(PIPE),
          .COL0(COL0 + 1), .BASE(BASE_UP), .COLBUDGET(CHILDB)
      ) u_up (
          .clk(clk), .din(upper_in_r), .ctrl_pipe(ctrl_pipe), .dout(upper_out)
      );
      bp_asw_block #(
          .N(M_LO), .NTOP(NTOP), .W(W), .PIPE(PIPE),
          .COL0(COL0 + 1), .BASE(BASE_LO), .COLBUDGET(CHILDB)
      ) u_lo (
          .clk(clk), .din(lower_in_r), .ctrl_pipe(ctrl_pipe), .dout(lower_out)
      );

      // Output stage: switch j combines (upper_out[j], lower_out[j]) into (dout[2j], dout[2j+1]);
      // then the hardwired last output(s) -- Waksman's removed switch.
      genvar osw;
      for (osw = 0; osw < OUT_CNT; osw++) begin : g_out_sw
        bp_asw_switch #(.W(W)) u_out (
            .sel  (ctrl_pipe[CS_OUT][BASE_OUT + osw]),
            .a_in (upper_out[osw]),
            .b_in (lower_out[osw]),
            .a_out(dout_c[2*osw]),
            .b_out(dout_c[2*osw+1])
        );
      end
      assign dout_c[SW_OUT] = upper_out[M_UP-1];
      if (HAS_BYP == 0) begin : g_hw_lo
        assign dout_c[SW_OUT+1] = lower_out[M_LO-1];
      end

      if (REG_OUT) begin : g_reg_out
        always_ff @(posedge clk) dout <= dout_c;
      end else begin : g_comb_out
        assign dout = dout_c;
      end
    end
  endgenerate
endmodule

// ---------------------------------------------------------------------------------------------
// Site-specific top-level fabrics. Each roots one `bp_asw_block` at (NTOP=N, COL0=0, BASE=0,
// COLBUDGET=asw_cols(N)) -- the whole network is one top-level block spanning all asw_cols(N)
// columns. Kept as two separate modules (not one generic fabric) per the M9c design decision: the
// e_cm read-gather (N=400) and m_cm write-scatter (N=800) are wired at different sites in the core
// even though they share this leaf cell + routing core. Payload width W is generic.
// ---------------------------------------------------------------------------------------------

module bp_asw_ecm_read #(
    parameter int N,
    parameter int W,
    parameter int PIPE
) (
    input  logic                          clk,
    input  logic [N-1:0][W-1:0]           din,
    // TIMING CONTRACT: `ctrl` travels WITH its own `din`; a fresh (din,ctrl) pair may be applied
    // every cycle, dout at t+PIPE == aswaksman_apply(ctrl_t, din_t). Registered into ctrl_pipe below.
    input  logic [asw_sw_count(N)-1:0]    ctrl,
    output logic [N-1:0][W-1:0]           dout
);
  localparam int CTRL_W = asw_sw_count(N);
  logic [CTRL_W-1:0] ctrl_pipe[0:PIPE];
  assign ctrl_pipe[0] = ctrl;
  genvar k;
  generate
    for (k = 1; k <= PIPE; k++) begin : g_ctrl_pipe
      always_ff @(posedge clk) ctrl_pipe[k] <= ctrl_pipe[k-1];
    end
  endgenerate
  bp_asw_block #(
      .N(N), .NTOP(N), .W(W), .PIPE(PIPE), .COL0(0), .BASE(0), .COLBUDGET(asw_cols(N))
  ) u_core (
      .clk(clk), .din(din), .ctrl_pipe(ctrl_pipe), .dout(dout)
  );
endmodule

module bp_asw_mcm_wr #(
    parameter int N,
    parameter int W,
    parameter int PIPE
) (
    input  logic                          clk,
    input  logic [N-1:0][W-1:0]           din,
    // TIMING CONTRACT: see bp_asw_ecm_read.
    input  logic [asw_sw_count(N)-1:0]    ctrl,
    output logic [N-1:0][W-1:0]           dout
);
  localparam int CTRL_W = asw_sw_count(N);
  logic [CTRL_W-1:0] ctrl_pipe[0:PIPE];
  assign ctrl_pipe[0] = ctrl;
  genvar k;
  generate
    for (k = 1; k <= PIPE; k++) begin : g_ctrl_pipe
      always_ff @(posedge clk) ctrl_pipe[k] <= ctrl_pipe[k-1];
    end
  endgenerate
  bp_asw_block #(
      .N(N), .NTOP(N), .W(W), .PIPE(PIPE), .COL0(0), .BASE(0), .COLBUDGET(asw_cols(N))
  ) u_core (
      .clk(clk), .din(din), .ctrl_pipe(ctrl_pipe), .dout(dout)
  );
endmodule

/* verilator lint_on DECLFILENAME */

`endif
