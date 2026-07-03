// Q7-02 M5-followup — SAT-OVERLAPPED partially-unrolled relay-BP decoder (the Arty-fit fast variant).
//
// Same synth-friendly gather→compute→scatter partial unroll as `bp_relay_partial.sv` (the decoder on the
// Arty, #442), but with the SAT-overlap cycle lever from `bp_relay_fast.sv` (#443) folded in. The partial
// spends three grouped phases per BP iteration — S_CHECK (G_CHK cycles), S_VAR (G_VAR cycles),
// S_SAT (G_CHK cycles). The syndrome-validity check S_SAT reads only `ehat` (from the just-finished
// S_VAR) while the next iteration's S_CHECK reads only `m_vc` — independent register→register clouds over
// the SAME `grp` group cursor. So S_SAT is folded to run IN PARALLEL with the next S_CHECK, group for
// group: during S_CHECK group `grp` we also XOR-check the previous iteration's decision on that group's
// checks and accumulate `all_sat`, finalising best/found when the sweep completes. A trailing S_SATF does
// the last iteration's decision (no following S_CHECK).
//
// Cost: G_CHK + G_VAR cycles/iteration instead of 2·G_CHK + G_VAR — for the 12/24 Arty build (G_CHK =
// G_VAR = 6) that is 12 cyc/iter not 18, i.e. **~1.5× fewer cycles** (≈ 720 vs 1086 at 6×10), at the same
// Arty part (the SAT parity XOR is far shallower than the min-sum, so the per-cycle path is unchanged; the
// 25 MHz Arty build has ample timing margin regardless). Pure cycle-count lever, no Fmax gamble.
//
// Bit-exactness with `bp_relay_partial` (hence `FixedRelayBp`): S_SAT still evaluates exactly the same
// per-iteration ehat values in the same order (a `sat_pending` guard skips the all-zero init decision,
// which the base decoder also never checks); `best`/`found` are read only at S_EMIT after S_SATF commits
// the last one; `ehat`/`ehat_w` are not overwritten between the S_VAR that writes them and the S_CHECK
// group sweep that reads them. Verified in Verilator (`-DPARTIALFAST`) bit-for-bit vs the golden.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_partial_fast #(
    parameter int CHK_UNROLL = 12,
    parameter int VAR_UNROLL = 24
) (
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
  localparam int WACC  = 16;
  localparam int WW    = $clog2(BP_N + 1);
  localparam int G_CHK = (BP_C + CHK_UNROLL - 1) / CHK_UNROLL;
  localparam int G_VAR = (BP_N + VAR_UNROLL - 1) / VAR_UNROLL;
  localparam int GW    = (G_CHK > G_VAR) ? $clog2(G_CHK + 1) : $clog2(G_VAR + 1);

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SATF, S_EMIT, S_DONE } state_t;
  state_t state;

  /* verilator lint_off UNUSEDSIGNAL */
  logic                       s_reg  [BP_C];
  logic signed [MSG_BITS-1:0] m_vc   [BP_E];
  logic signed [MSG_BITS-1:0] e_cv   [BP_E];
  logic                       ehat   [BP_N];
  logic                       best_e [BP_N];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat, sat_pending;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter;
  logic [GW-1:0] grp;
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  always_ff @(posedge clk) begin        // synchronous reset (see M4 note on Synth 8-7137)
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
            all_sat <= 1'b1;
            sat_pending <= 1'b0;         // no decision to check before the first S_VAR
            leg <= '0; iter <= '0; grp <= '0;
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ------------------------------- CHK_UNROLL checks / cycle (min-sum)  ‖  overlapped S_SAT group
        S_CHECK: begin
          automatic logic sbit;
          automatic logic signed [MSG_BITS-1:0] mm  [BP_CHK_DEG];
          automatic logic signed [MSG_BITS-1:0] ee  [BP_CHK_DEG];
          automatic logic present [BP_CHK_DEG];
          automatic int argk;
          automatic logic neg, excl;
          automatic logic [MSG_BITS-1:0] min1, min2, a, exmin, mag;
          automatic logic signed [MSG_BITS-1:0] mv;
          automatic logic grp_sat, final_sat, p;
          for (int i = 0; i < CHK_UNROLL; i++) begin
            // ---- gather this slot's check inputs by grp, at CONSTANT edge indices ----
            sbit = 1'b0;
            for (int k = 0; k < BP_CHK_DEG; k++) begin mm[k] = '0; present[k] = 1'b0; end
            for (int g = 0; g < G_CHK; g++)
              if (g * CHK_UNROLL + i < BP_C && grp == GW'(g)) begin
                automatic int c  = g * CHK_UNROLL + i;
                automatic int lo = BP_CHECK_OFF[c];
                automatic int hi = BP_CHECK_OFF[c + 1];
                sbit = s_reg[c];
                for (int k = 0; k < BP_CHK_DEG; k++)
                  if (lo + k < hi) begin
                    mm[k]      = m_vc[BP_CHECK_EDGES[lo + k]];
                    present[k] = 1'b1;
                  end
              end
            // ---- compute: min-sum (two-pass exclusive-minimum) ----
            neg = sbit; min1 = INF; min2 = INF; argk = -1;
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (present[k]) begin
                mv = mm[k];
                if (mv < 0) neg = ~neg;
                a = mv[MSG_BITS-1] ? unsigned'(-mv) : unsigned'(mv);
                if (a < min1) begin min2 = min1; min1 = a; argk = k; end
                else if (a < min2) begin min2 = a; end
              end
            for (int k = 0; k < BP_CHK_DEG; k++) begin
              ee[k] = '0;
              if (present[k]) begin
                mv    = mm[k];
                excl  = (mv < 0) ? ~neg : neg;
                exmin = (k == argk) ? min2 : min1;
                if (exmin == INF) exmin = '0;
                mag   = exmin - (exmin >> 3);
                ee[k] = excl ? -$signed(mag) : $signed(mag);
              end
            end
            // ---- scatter e_cv back at the same CONSTANT edges, by grp ----
            for (int g = 0; g < G_CHK; g++)
              if (g * CHK_UNROLL + i < BP_C && grp == GW'(g)) begin
                automatic int lo = BP_CHECK_OFF[g * CHK_UNROLL + i];
                automatic int hi = BP_CHECK_OFF[g * CHK_UNROLL + i + 1];
                for (int k = 0; k < BP_CHK_DEG; k++)
                  if (lo + k < hi) e_cv[BP_CHECK_EDGES[lo + k]] <= ee[k];
              end
          end

          // ---- overlapped S_SAT: parity of the PREVIOUS ehat on this group's checks ----
          grp_sat = 1'b1;
          if (sat_pending)
            for (int i = 0; i < CHK_UNROLL; i++)
              for (int g = 0; g < G_CHK; g++)
                if (g * CHK_UNROLL + i < BP_C && grp == GW'(g)) begin
                  automatic int c  = g * CHK_UNROLL + i;
                  automatic int lo = BP_CHECK_OFF[c];
                  automatic int hi = BP_CHECK_OFF[c + 1];
                  p = s_reg[c];
                  for (int k = 0; k < BP_CHK_DEG; k++)
                    if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
                  if (p != 1'b0) grp_sat = 1'b0;
                end
          if (sat_pending && !grp_sat) all_sat <= 1'b0;

          // ---- advance group; on the last group finalise the overlapped SAT + step to S_VAR ----
          if (grp == GW'(G_CHK - 1)) begin
            if (sat_pending) begin
              final_sat = all_sat & grp_sat;
              if (final_sat) begin
                found <= 1'b1;
                if (ehat_w < best_w) begin
                  best_w <= ehat_w;
                  for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
                end
              end
            end
            all_sat <= 1'b1;                 // reset accumulator for the next iteration's SAT sweep
            grp <= '0; state <= S_VAR;
          end else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- VAR_UNROLL variables / cycle
        S_VAR: begin
          automatic int wsum;
          automatic logic newbit;
          automatic logic signed [WACC-1:0] total, g, omg, ev, old, computed, num, blend;
          automatic logic signed [MSG_BITS-1:0] ecv_k [BP_VAR_DEG];
          automatic logic signed [MSG_BITS-1:0] mvc_k [BP_VAR_DEG];
          automatic logic present [BP_VAR_DEG];
          automatic int lam, gam;
          wsum = 0;
          for (int i = 0; i < VAR_UNROLL; i++) begin
            lam = 0; gam = 0;
            for (int k = 0; k < BP_VAR_DEG; k++) begin ecv_k[k] = '0; mvc_k[k] = '0; present[k] = 1'b0; end
            for (int gg = 0; gg < G_VAR; gg++)
              if (gg * VAR_UNROLL + i < BP_N && grp == GW'(gg)) begin
                automatic int v  = gg * VAR_UNROLL + i;
                automatic int lo = BP_VAR_OFF[v];
                automatic int hi = BP_VAR_OFF[v + 1];
                lam = BP_LAMBDA[v];
                gam = BP_GAMMA[leg * BP_N + v];
                for (int k = 0; k < BP_VAR_DEG; k++)
                  if (lo + k < hi) begin
                    ecv_k[k]   = e_cv[lo + k];
                    mvc_k[k]   = m_vc[lo + k];
                    present[k] = 1'b1;
                  end
              end
            total = signed'(WACC'(lam));
            for (int k = 0; k < BP_VAR_DEG; k++)
              if (present[k]) total = total + signed'(WACC'(ecv_k[k]));
            newbit = total[WACC-1];
            wsum   = wsum + (newbit ? 1 : 0);
            g   = signed'(WACC'(gam));
            omg = signed'(WACC'(1 << FRAC_BITS)) - g;
            for (int gg = 0; gg < G_VAR; gg++)
              if (gg * VAR_UNROLL + i < BP_N && grp == GW'(gg)) begin
                automatic int v  = gg * VAR_UNROLL + i;
                automatic int lo = BP_VAR_OFF[v];
                automatic int hi = BP_VAR_OFF[v + 1];
                ehat[v] <= newbit;
                for (int k = 0; k < BP_VAR_DEG; k++)
                  if (lo + k < hi) begin
                    ev  = signed'(WACC'(ecv_k[k]));
                    old = signed'(WACC'(mvc_k[k]));
                    computed = total - ev;
                    num   = omg * computed + g * old;
                    blend = num >>> FRAC_BITS;
                    if (blend > signed'(WACC'(MAX_MAG)))       blend = signed'(WACC'(MAX_MAG));
                    else if (blend < -signed'(WACC'(MAX_MAG))) blend = -signed'(WACC'(MAX_MAG));
                    m_vc[lo + k] <= blend[MSG_BITS-1:0];
                  end
              end
          end
          ehat_w <= (grp == '0 ? WW'(0) : ehat_w) + WW'(wsum);
          if (grp == GW'(G_VAR - 1)) begin
            grp <= '0;
            sat_pending <= 1'b1;              // a fresh decision is now available to check
            // advance iteration / leg; the SAT for THIS ehat runs next cycle (in S_CHECK, or S_SATF if last)
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_SATF;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------- trailing S_SAT for the final ehat (no following S_CHECK)
        S_SATF: begin
          automatic logic p, grp_sat, final_sat;
          grp_sat = 1'b1;
          for (int i = 0; i < CHK_UNROLL; i++)
            for (int g = 0; g < G_CHK; g++)
              if (g * CHK_UNROLL + i < BP_C && grp == GW'(g)) begin
                automatic int c  = g * CHK_UNROLL + i;
                automatic int lo = BP_CHECK_OFF[c];
                automatic int hi = BP_CHECK_OFF[c + 1];
                p = s_reg[c];
                for (int k = 0; k < BP_CHK_DEG; k++)
                  if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
                if (p != 1'b0) grp_sat = 1'b0;
              end
          if (!grp_sat) all_sat <= 1'b0;
          if (grp == GW'(G_CHK - 1)) begin
            final_sat = all_sat & grp_sat;
            if (final_sat) begin
              found <= 1'b1;
              if (ehat_w < best_w) begin
                best_w <= ehat_w;
                for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
              end
            end
            grp <= '0; state <= S_EMIT;
          end else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;
          automatic logic b;
          acc = (grp == '0) ? {BP_OBS{1'b0}} : obs_acc;
          for (int i = 0; i < VAR_UNROLL; i++)
            for (int gg = 0; gg < G_VAR; gg++)
              if (gg * VAR_UNROLL + i < BP_N && grp == GW'(gg)) begin
                automatic int v = gg * VAR_UNROLL + i;
                b = found ? best_e[v] : ehat[v];
                corr_out[v] <= b;
                if (b) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
              end
          obs_acc <= acc;
          if (grp == GW'(G_VAR - 1)) begin grp <= '0; state <= S_DONE; end
          else grp <= grp + 1'b1;
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
