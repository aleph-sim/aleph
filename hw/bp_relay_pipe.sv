// Q7-02 M5-followup — MIN-SUM-PIPELINED spatially-unrolled relay-BP decoder for the gross BB code.
//
// Identical datapath to `bp_relay_unrolled.sv` (M4/M5) except the per-check min-sum (S_CHECK) is split
// into two pipeline stages. The M5 Fmax study found the wall is the S_CHECK min-sum path
// `m_vc_reg → e_cv_reg` (route-dominated ~55% at ~80% util, ~96 MHz on KV260), NOT the S_VAR blend.
// That path is one cycle of: read m_vc[e] → abs → 6-way exclusive-minimum tournament (min1,min2,argmin)
// → per-edge select exmin → α=7/8 shift → sign → write e_cv[e]. The deep part is the reduction (pass 1);
// the output (pass 2) is shallow. So we register between them:
//
//   S_CHK1 (1 cyc): for all 72 checks, compute {sign-parity neg, two smallest |m|, argmin slot} and
//                   REGISTER them into per-check pipeline regs pc_*.  (the deep tournament)
//   S_CHK2 (1 cyc): for all 72 checks, for each edge, read the registered minima + m_vc[e] and emit
//                   e_cv[e] = ±(exmin - exmin>>3).                    (shallow: one select + shift)
//
// Cost: +1 cycle per BP iteration (schedule 6×10 → 60 iters, now 4 cyc/iter → 241 cyc vs M5's 181).
// So this is a NET latency win only if the shorter S_CHK1/S_CHK2 stages raise Fmax by more than the
// 241/181 = 1.33× cycle penalty (break-even Fmax ≈ 128 MHz). Measured OOC on KV260 (synth_bp.tcl).
//
// Bit-exactness with M4/M5 (hence `FixedRelayBp`) is structural: m_vc is not written between S_CHK1 and
// S_CHK2 (only S_VAR writes it), so re-reading m_vc[e] in pass 2 sees the same values pass 1 reduced;
// argmin is carried as the LOCAL slot k (strict-< first-winner, same tie-break as the global-edge form).
// Verified in Verilator (`tb_bp_relay.cpp` with -DPIPE) bit-for-bit on the same vectors M4/M5 passed.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_pipe (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],
    output logic [BP_OBS-1:0]   obs_flip,
    output logic                valid_flag,
    output logic [15:0]         latency_cycles
);
  localparam logic [MSG_BITS-1:0] INF = '1;
  localparam int WACC = 16;                          // right-sized blend accumulator (M5 step 2a)
  localparam int WW   = $clog2(BP_N + 1);
  localparam int AW   = $clog2(BP_CHK_DEG);          // bits to index a check's edge slot (argmin)

  typedef enum logic [2:0] { S_IDLE, S_CHK1, S_CHK2, S_VAR, S_SAT, S_EMIT, S_DONE } state_t;
  state_t state;

  // dont_touch: same const-prop fold hazard as M4 (constant indices let synth chase the 100-iter
  // feedback to ehat≡0 and delete the datapath). Anchoring the state registers blocks the fold.
  /* verilator lint_off UNUSEDSIGNAL */
  (* dont_touch = "true" *) logic                       s_reg  [BP_C];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc   [BP_E];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv   [BP_E];
  (* dont_touch = "true" *) logic                       ehat   [BP_N];
  (* dont_touch = "true" *) logic                       best_e [BP_N];
  // per-check min-sum pipeline registers (written by S_CHK1, read by S_CHK2)
  (* dont_touch = "true" *) logic                 pc_neg  [BP_C];
  (* dont_touch = "true" *) logic [MSG_BITS-1:0]  pc_min1 [BP_C];
  (* dont_touch = "true" *) logic [MSG_BITS-1:0]  pc_min2 [BP_C];
  (* dont_touch = "true" *) logic [AW-1:0]        pc_arg  [BP_C];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter;
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  always_ff @(posedge clk) begin                     // synchronous reset (M4 Synth 8-7137 note)
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
            for (int e = 0; e < BP_E; e++)
              m_vc[e] <= signed'(BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0]);
            for (int v = 0; v < BP_N; v++) ehat[v] <= 1'b0;
            found  <= 1'b0;
            best_w <= '1;
            ehat_w <= '0;
            leg <= '0; iter <= '0;
            lat <= '0;
            state <= S_CHK1;
          end
        end

        // -------------------------------------------------- min-sum PASS 1: reduction → pipeline regs
        // 72 parallel tournaments; register {sign parity, two smallest magnitudes, argmin slot} per check.
        S_CHK1: begin
          automatic int lo, hi;
          automatic logic neg;
          automatic logic [MSG_BITS-1:0] min1, min2, a;
          automatic logic signed [MSG_BITS-1:0] m;
          automatic logic [AW-1:0] arg;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            neg = s_reg[c];
            min1 = INF; min2 = INF; arg = '0;
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) begin
                m = m_vc[BP_CHECK_EDGES[lo + k]];
                if (m < 0) neg = ~neg;
                a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);
                if (a < min1) begin min2 = min1; min1 = a; arg = AW'(k); end
                else if (a < min2) begin min2 = a; end
              end
            pc_neg[c]  <= neg;
            pc_min1[c] <= min1;
            pc_min2[c] <= min2;
            pc_arg[c]  <= arg;
          end
          state <= S_CHK2;
          lat <= lat + 16'd1;
        end

        // -------------------------------------------------- min-sum PASS 2: output (shallow) → e_cv
        S_CHK2: begin
          automatic int lo, hi;
          automatic logic excl;
          automatic logic [MSG_BITS-1:0] exmin, mag;
          automatic logic signed [MSG_BITS-1:0] m;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) begin
                m     = m_vc[BP_CHECK_EDGES[lo + k]];
                excl  = (m < 0) ? ~pc_neg[c] : pc_neg[c];
                exmin = (AW'(k) == pc_arg[c]) ? pc_min2[c] : pc_min1[c];
                if (exmin == INF) exmin = '0;
                mag   = exmin - (exmin >> 3);          // α = 7/8, multiply-free
                e_cv[BP_CHECK_EDGES[lo + k]] <= excl ? -$signed(mag) : $signed(mag);
              end
          end
          state <= S_VAR;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- ALL variables → check + memory
        S_VAR: begin
          automatic int lo, hi, e, wsum;
          automatic logic newbit;
          automatic logic signed [WACC-1:0] total, g, omg, ev, old, computed, num, blend;
          wsum = 0;
          for (int v = 0; v < BP_N; v++) begin
            lo = BP_VAR_OFF[v];
            hi = BP_VAR_OFF[v + 1];
            total = signed'(WACC'(BP_LAMBDA[v]));
            for (int k = 0; k < BP_VAR_DEG; k++)
              if (lo + k < hi) total = total + signed'(WACC'(e_cv[lo + k]));
            newbit = total[WACC-1];
            ehat[v] <= newbit;
            wsum = wsum + (newbit ? 1 : 0);
            g   = signed'(WACC'(BP_GAMMA[leg * BP_N + v]));
            omg = signed'(WACC'(1 << FRAC_BITS)) - g;
            for (int k = 0; k < BP_VAR_DEG; k++)
              if (lo + k < hi) begin
                e = lo + k;
                ev  = signed'(WACC'(e_cv[e]));
                old = signed'(WACC'(m_vc[e]));
                computed = total - ev;
                num   = omg * computed + g * old;
                blend = num >>> FRAC_BITS;
                if (blend > signed'(WACC'(MAX_MAG)))       blend = signed'(WACC'(MAX_MAG));
                else if (blend < -signed'(WACC'(MAX_MAG))) blend = -signed'(WACC'(MAX_MAG));
                m_vc[e] <= blend[MSG_BITS-1:0];
              end
          end
          ehat_w <= WW'(wsum);
          state  <= S_SAT;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- H·ehat == s ? keep best
        S_SAT: begin
          automatic int lo, hi;
          automatic logic p, sat;
          sat = 1'b1;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            p = s_reg[c];
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
            if (p != 1'b0) sat = 1'b0;
          end
          if (sat) begin
            found <= 1'b1;
            if (ehat_w < best_w) begin
              best_w <= ehat_w;
              for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
            end
          end
          if (iter == BP_ITERS - 1) begin
            iter <= '0;
            if (leg == BP_LEGS - 1) state <= S_EMIT;
            else begin leg <= leg + 1; state <= S_CHK1; end
          end else begin
            iter <= iter + 1;
            state <= S_CHK1;
          end
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;
          automatic logic b;
          acc = '0;
          for (int v = 0; v < BP_N; v++) begin
            b = found ? best_e[v] : ehat[v];
            corr_out[v] <= b;
            if (b) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
          end
          obs_acc <= acc;
          state <= S_DONE;
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
