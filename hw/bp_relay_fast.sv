// Q7-02 M5-followup — SAT-OVERLAPPED spatially-unrolled relay-BP decoder for the gross BB code.
//
// Identical datapath to `bp_relay_unrolled.sv` (M4/M5) except the syndrome-validity check (S_SAT:
// `H·ehat == s` + keep-lowest-weight-valid) no longer costs its own cycle in the BP loop. M4/M5 ran
// S_CHECK → S_VAR → S_SAT = 3 cyc/iter. But S_SAT reads only `ehat` (from the just-finished S_VAR),
// while the NEXT S_CHECK reads only `m_vc` (also from that S_VAR) — the two are independent
// register→register clouds. So S_SAT is folded to run IN PARALLEL with the next iteration's S_CHECK:
//
//   S_CHECK (1 cyc): e_cv ← min-sum(m_vc, s)         ‖  [if a decision is pending] evaluate S_SAT on
//                                                        the previous iteration's ehat, update best
//   S_VAR   (1 cyc): m_vc, ehat ← var-update-with-memory;  mark a decision pending
//
// 2 cyc/iter → schedule 6×10 = 60 iters → ~123 cycles (vs M5's 181) at the SAME per-cycle critical path
// (the S_SAT parity XOR is shallower than the S_CHECK min-sum wall, so running them concurrently keeps
// Fmax at the min-sum bound). A trailing S_SATF cycle evaluates the last iteration's ehat (it has no
// following S_CHECK) before S_EMIT consumes `best`. This is a pure CYCLE-COUNT lever — no Fmax gamble,
// unlike the min-sum pipeline (`bp_relay_pipe.sv`, +1 cyc, needs Fmax >1.33×).
//
// Bit-exactness with M4/M5 (hence `FixedRelayBp`): S_SAT still evaluates exactly the 60 registered ehat
// values in the same order (a `sat_pending` guard skips the all-zero init ehat, which M4 also never
// evaluated), and `best`/`found` are the sole consumers of the check, read only at S_EMIT after S_SATF
// has committed the last one. `ehat`/`ehat_w` are not overwritten between the S_VAR that writes them and
// the next-cycle S_CHECK that reads them for the check. Verified in Verilator (`-DFAST`) 65/65.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_fast (
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
  localparam int WACC = 16;
  localparam int WW   = $clog2(BP_N + 1);

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SATF, S_EMIT, S_DONE } state_t;
  state_t state;

  /* verilator lint_off UNUSEDSIGNAL */
  (* dont_touch = "true" *) logic                       s_reg  [BP_C];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc   [BP_E];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv   [BP_E];
  (* dont_touch = "true" *) logic                       ehat   [BP_N];
  (* dont_touch = "true" *) logic                       best_e [BP_N];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, sat_pending;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter;
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  // Combinational S_SAT: parity of `ehat` against the syndrome, and the keep-lowest-weight-valid
  // bookkeeping. Factored into a task so it can fire in two places (parallel with S_CHECK, and in the
  // trailing S_SATF) with identical logic. Reads registers, writes registers — its own reg→reg path.
  task automatic run_sat;
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
  endtask

  always_ff @(posedge clk) begin                     // synchronous reset (M4 Synth 8-7137 note)
    if (!rst_n) begin
      state       <= S_IDLE;
      out_valid   <= 1'b0;
      valid_flag  <= 1'b0;
      lat         <= '0;
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
            sat_pending <= 1'b0;                       // no decision to check before the first S_VAR
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ---------------------------------------------- ALL checks → variable (min-sum) ‖ overlapped S_SAT
        S_CHECK: begin
          automatic int lo, hi, argmin, e;
          automatic logic neg, excl;
          automatic logic [MSG_BITS-1:0] min1, min2, a, exmin, mag;
          automatic logic signed [MSG_BITS-1:0] m;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            neg = s_reg[c];
            min1 = INF; min2 = INF; argmin = -1;
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) begin
                e = BP_CHECK_EDGES[lo + k];
                m = m_vc[e];
                if (m < 0) neg = ~neg;
                a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);
                if (a < min1) begin min2 = min1; min1 = a; argmin = e; end
                else if (a < min2) begin min2 = a; end
              end
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) begin
                e = BP_CHECK_EDGES[lo + k];
                m = m_vc[e];
                excl  = (m < 0) ? ~neg : neg;
                exmin = (e == argmin) ? min2 : min1;
                if (exmin == INF) exmin = '0;
                mag = exmin - (exmin >> 3);            // α = 7/8, multiply-free
                e_cv[e] <= excl ? -$signed(mag) : $signed(mag);
              end
          end
          // overlapped S_SAT on the previous iteration's decision (parallel reg→reg path)
          if (sat_pending) run_sat();
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
          sat_pending <= 1'b1;                          // a fresh decision is now available to check
          // advance iteration / leg; the SAT for THIS ehat runs next cycle (in S_CHECK, or S_SATF if last)
          if (iter == BP_ITERS - 1) begin
            iter <= '0;
            if (leg == BP_LEGS - 1) state <= S_SATF;
            else begin leg <= leg + 1; state <= S_CHECK; end
          end else begin
            iter <= iter + 1;
            state <= S_CHECK;
          end
          lat <= lat + 16'd1;
        end

        // ----------------------------------- trailing S_SAT for the final ehat (no following S_CHECK)
        S_SATF: begin
          run_sat();
          state <= S_EMIT;
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
