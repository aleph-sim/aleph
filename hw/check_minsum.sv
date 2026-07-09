// Q7-02 M7 — standalone min-sum check-node submodule (`check_minsum`), 2-cycle pipelined, generic MW/DEG.
//
// Bit-exact twin of ONE check's inner loop from `bp_relay_fast.sv`'s S_CHECK state (lines 121-145):
// min-sum with exclusive minimum via the running (min1,min2,argmin) trick, alpha=7/8 multiply-free
// scaling (`mag = exmin - (exmin>>3)`), and a per-edge sign from the check's syndrome bit XOR'd with
// each real edge's own message sign. `present[k]` masks real vs unused slots exactly like the source
// loop's `if (lo+k<hi)` guard — DEG is the graph's max check degree (BP_CHK_DEG), one module variant
// handles every check regardless of its real degree.
//
// This is the FIRST of the two small per-check/per-var submodules the M7 hierarchically-modular
// relay-BP core stamps out many times (one per check node); pipelining it (rather than leaving it one
// combinational cloud like `bp_check_update.sv`) is what makes that stamping synthesis-friendly:
//   stage 1 (posedge, gated by `en`): reduce over the DEG (present-masked) slots -> registered
//            min1/min2/argmin/neg; also register `m_in`/`present` themselves so stage 2 sees the
//            SAME snapshot the reduction ran over.
//   stage 2 (posedge, unconditional — free-running off the stage-1 registers): per real slot,
//            exclude this edge's own contribution and emit the signed check->variable message.
//            Unused slots emit 0 (no inferred latch, deterministic pad; bp_relay_fast never writes
//            e_cv for a non-real slot at all, so 0 there is a don't-care downstream — picked here
//            for a fully-defined, bit-reproducible output over the WHOLE e_out array).
//
// `INF` (all-ones of width MW) is the "no message yet" sentinel exactly as in bp_relay_fast /
// bp_check_update; a degree-1 (or all-unused) check would leave the excluded min at INF, treated as 0
// (no constraint) — same as `bp_relay_fast.sv:141`.

`timescale 1ns / 1ps

module check_minsum #(
    parameter int MW  = 8,
    parameter int DEG = 25
) (
    input  logic                 clk,
    input  logic                 en,             // pulse to start; result valid 2 clocks later
    input  logic                 sbit,           // this check's syndrome bit
    input  logic signed [MW-1:0] m_in    [DEG],
    input  logic                 present [DEG],  // 1 = real edge, 0 = unused slot (deg < DEG)
    output logic signed [MW-1:0] e_out   [DEG]
);
  localparam logic [MW-1:0] INF = '1;

  // ------------------------------------------------------------------- stage 1: reduction registers
  logic [MW-1:0] min1_r, min2_r;
  int            argmin_r;
  logic          neg_r;
  logic signed [MW-1:0] m_in_r    [DEG];
  logic                 present_r [DEG];

  always_ff @(posedge clk) begin
    if (en) begin
      automatic logic                 neg;
      automatic logic [MW-1:0]        min1, min2, a;
      automatic int                   argmin;
      automatic logic signed [MW-1:0] m;

      neg    = sbit;
      min1   = INF;
      min2   = INF;
      argmin = -1;

      for (int k = 0; k < DEG; k++) begin
        if (present[k]) begin
          m = m_in[k];
          if (m < 0) neg = ~neg;
          a = m[MW-1] ? unsigned'(-m) : unsigned'(m);
          if (a < min1) begin
            min2   = min1;
            min1   = a;
            argmin = k;
          end else if (a < min2) begin
            min2 = a;
          end
        end
      end

      min1_r   <= min1;
      min2_r   <= min2;
      argmin_r <= argmin;
      neg_r    <= neg;
      for (int k = 0; k < DEG; k++) begin
        m_in_r[k]    <= m_in[k];
        present_r[k] <= present[k];
      end
    end
  end

  // ------------------------------------------------------------------- stage 2: emit (free-running)
  always_ff @(posedge clk) begin
    for (int k = 0; k < DEG; k++) begin
      if (present_r[k]) begin
        automatic logic signed [MW-1:0] m;
        automatic logic                 excl;
        automatic logic [MW-1:0]        exmin, mag;

        m     = m_in_r[k];
        excl  = (m < 0) ? ~neg_r : neg_r;
        exmin = (k == argmin_r) ? min2_r : min1_r;
        if (exmin == INF) exmin = '0;
        mag = exmin - (exmin >> 3);           // alpha = 7/8, multiply-free
        e_out[k] <= excl ? -$signed(mag) : $signed(mag);
      end else begin
        e_out[k] <= '0;
      end
    end
  end
endmodule
