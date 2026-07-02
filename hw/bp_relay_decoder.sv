// Q7-02 M2 — fixed-point relay-BP decoder for the gross BB code [[144,12,12]], SYNTHESIZABLE
// SEQUENTIAL FORM.
//
// The full relay-BP decode of `FixedRelayBp` (crates/aleph-qec/src/fixed_bp.rs) time-multiplexed into
// a clocked FSM, one *bounded* pass per cycle (like the Q6-04 UF rewrite): a cycle touches exactly one
// check (≤ BP_CHK_DEG edges) or one variable (≤ BP_VAR_DEG edges), so the per-cycle combinational
// depth is bounded and independent of the graph size. The outer loops (legs × iterations) advance in
// time across FSM states.
//
// Per iteration:
//   S_CHECK  (BP_C cycles): e_cv ← min-sum(m_vc, syndrome)          — one check / cycle
//   S_VAR    (BP_N cycles): m_vc,ehat ← var-update-with-memory(e_cv) — one variable / cycle
//   S_SAT    (BP_C cycles): all_sat ← (H·ehat == s); record lowest-weight-valid ehat
// looped over BP_LEGS × BP_ITERS with the per-leg disorder pattern γ; then
//   S_EMIT   (BP_N cycles): reduce the chosen ehat → observable flips
//   S_DONE   (1 cycle):     pulse out_valid.
//
// Equivalence: identical schedule, quantisation, and keep-lowest-weight-valid rule as the Rust golden
// — check-update (multiply-free α=7/8), memory blend `(1−γ)·computed + γ·m_old` (the one multiply),
// truncating arithmetic-shift rounding. Verified in Verilator (`tb_bp_relay.cpp`) bit-for-bit against
// `FixedRelayBp::decode_fixed_ehat` on empty / single-error / random low-weight syndromes.
//
// M2 is correctness-first: BP_C+BP_N+BP_C cycles/iteration × BP_LEGS·BP_ITERS with no early exit
// (relay-BP keeps the best across ALL legs). That cycle count is the honest latency M3 measures and
// M4 attacks (unrolling checks/variables per cycle, exactly as UF's FOREST_UNROLL did).

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_decoder (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],       // chosen error pattern (one bit / variable)
    output logic [BP_OBS-1:0]   obs_flip,                 // predicted observable flips
    output logic                valid_flag,               // a syndrome-valid decision was found
    output logic [15:0]         latency_cycles
);
  // Magnitudes are ≤ MAX_MAG (< 2^(MSG_BITS-1)); the all-ones word (> every magnitude) is the +∞
  // sentinel for the running minima, matching the Rust golden's i32::MAX sentinel.
  localparam logic [MSG_BITS-1:0] INF = '1;
  localparam int WACC = 32;             // wide accumulator/product (M2: generous; M4 right-sizes)
  localparam int WW   = $clog2(BP_N + 1);

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SAT, S_EMIT, S_DONE } state_t;
  state_t state;

  // Loop cursors are plain `int` working vars (only their low bits index the tables); this keeps the
  // width arithmetic clean under -Wall. Same idiom the UF decoder uses for its peel cursors.
  /* verilator lint_off UNUSEDSIGNAL */

  logic                       s_reg  [BP_C];
  logic signed [MSG_BITS-1:0] m_vc   [BP_E];   // variable→check
  logic signed [MSG_BITS-1:0] e_cv   [BP_E];   // check→variable
  logic                       ehat   [BP_N];   // current hard decision
  logic                       best_e [BP_N];   // lowest-weight syndrome-valid decision seen
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter, idx;
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      out_valid  <= 1'b0;
      valid_flag <= 1'b0;
      lat        <= '0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        // ----------------------------------------------------------------- accept + init
        S_IDLE: begin
          if (in_valid) begin
            for (int c = 0; c < BP_C; c++) s_reg[c] <= syndrome_in[c];
            // Init M_{v→c} = λ_v (quantised); messages relay across legs.
            for (int e = 0; e < BP_E; e++)
              m_vc[e] <= signed'(BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0]);
            for (int v = 0; v < BP_N; v++) ehat[v] <= 1'b0;
            found  <= 1'b0;
            best_w <= '1;
            ehat_w <= '0;
            leg <= '0; iter <= '0; idx <= '0;
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ----------------------------------------------------------------- check → variable (min-sum)
        S_CHECK: begin
          int lo, hi, argmin, e;
          logic neg, excl;
          logic [MSG_BITS-1:0] min1, min2, a, exmin, mag;
          logic signed [MSG_BITS-1:0] m;
          lo = BP_CHECK_OFF[idx];
          hi = BP_CHECK_OFF[idx + 1];
          neg = s_reg[idx];
          min1 = INF; min2 = INF; argmin = -1;
          // pass 1: sign, two smallest magnitudes, argmin
          for (int k = 0; k < BP_CHK_DEG; k++) begin
            if (lo + k < hi) begin
              e = BP_CHECK_EDGES[lo + k];
              m = m_vc[e];
              if (m < 0) neg = ~neg;
              a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);
              if (a < min1) begin min2 = min1; min1 = a; argmin = e; end
              else if (a < min2) begin min2 = a; end
            end
          end
          // pass 2: exclude each edge's own contribution
          for (int k = 0; k < BP_CHK_DEG; k++) begin
            if (lo + k < hi) begin
              e = BP_CHECK_EDGES[lo + k];
              m = m_vc[e];
              excl  = (m < 0) ? ~neg : neg;
              exmin = (e == argmin) ? min2 : min1;
              if (exmin == INF) exmin = '0;
              mag = exmin - (exmin >> 3);          // α = 7/8, multiply-free
              e_cv[e] <= excl ? -$signed(mag) : $signed(mag);
            end
          end
          if (idx == BP_C - 1) begin idx <= '0; state <= S_VAR; end
          else idx <= idx + 1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- variable → check + memory
        S_VAR: begin
          int lo, hi, e;
          logic newbit;
          logic signed [WACC-1:0] total, g, omg, ev, old, computed, num, blend;
          lo = BP_VAR_OFF[idx];
          hi = BP_VAR_OFF[idx + 1];
          total = signed'(WACC'(BP_LAMBDA[idx]));
          for (int k = 0; k < BP_VAR_DEG; k++)
            if (lo + k < hi) total = total + signed'(WACC'(e_cv[lo + k]));
          newbit = total[WACC-1];                  // total < 0 ⇒ decision 1
          ehat[idx] <= newbit;
          ehat_w <= (idx == 0 ? WW'(0) : ehat_w) + WW'(newbit ? 1'b1 : 1'b0);
          g   = signed'(WACC'(BP_GAMMA[leg * BP_N + idx]));
          omg = signed'(WACC'(1 << FRAC_BITS)) - g;     // (1−γ) in 2^F units
          for (int k = 0; k < BP_VAR_DEG; k++) begin
            if (lo + k < hi) begin
              e = lo + k;                          // edges are variable-major (contiguous)
              ev  = signed'(WACC'(e_cv[e]));
              old = signed'(WACC'(m_vc[e]));
              computed = total - ev;
              num   = omg * computed + g * old;
              blend = num >>> FRAC_BITS;           // truncating (floor) rounding
              if (blend > signed'(WACC'(MAX_MAG)))       blend = signed'(WACC'(MAX_MAG));
              else if (blend < -signed'(WACC'(MAX_MAG))) blend = -signed'(WACC'(MAX_MAG));
              m_vc[e] <= blend[MSG_BITS-1:0];
            end
          end
          if (idx == BP_N - 1) begin idx <= '0; all_sat <= 1'b1; state <= S_SAT; end
          else idx <= idx + 1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- H·ehat == s ? keep best
        S_SAT: begin
          int lo, hi;
          logic p, final_sat;
          lo = BP_CHECK_OFF[idx];
          hi = BP_CHECK_OFF[idx + 1];
          p = s_reg[idx];
          for (int k = 0; k < BP_CHK_DEG; k++)
            if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
          if (p != 1'b0) all_sat <= 1'b0;
          if (idx == BP_C - 1) begin
            final_sat = all_sat & (p == 1'b0);   // include this last check combinationally
            if (final_sat) begin
              found <= 1'b1;
              if (ehat_w < best_w) begin
                best_w <= ehat_w;
                for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
              end
            end
            // advance iteration / leg
            idx <= '0;
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_EMIT;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else idx <= idx + 1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          logic b;
          logic [BP_OBS-1:0] msk;
          b   = found ? best_e[idx] : ehat[idx];
          msk = BP_OBS_MASK[idx][BP_OBS-1:0];
          corr_out[idx] <= b;
          obs_acc <= (idx == 0 ? {BP_OBS{1'b0}} : obs_acc) ^ (b ? msk : {BP_OBS{1'b0}});
          if (idx == BP_N - 1) state <= S_DONE;
          else idx <= idx + 1;
          lat <= lat + 16'd1;
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
