// Q7-02 M5-followup — PARTIALLY-UNROLLED fixed-point relay-BP decoder for the gross BB code.
//
// M4 (`bp_relay_unrolled.sv`) processes ALL 72 checks / 144 variables per cycle. That is fast (181
// cycles at 6×10) but big: 94 k LUT = 80% of the KV260 and 172% of a xc7z020 (Zybo/Arty) — it does
// NOT fit the small parts, and its 80%-util routing congestion is what caps Fmax (the M5 timing report
// put the binding path on the S_CHECK min-sum, 55% routing). M2 (`bp_relay_decoder.sv`) is the other
// extreme: ONE node/cycle (fits anything, but 17 424 cycles at 6×10 and a big runtime cursor mux).
//
// This is the middle: process `CHK_UNROLL` checks and `VAR_UNROLL` variables per cycle, stepping a
// group cursor `grp` across `G_CHK = ⌈BP_C/CHK_UNROLL⌉` / `G_VAR = ⌈BP_N/VAR_UNROLL⌉` groups per phase.
// Area scales ~`CHK_UNROLL/BP_C`, so a modest factor fits the xc7z020 while staying far faster than the
// fully-sequential M2. It also relieves the M4 congestion the M5 report blamed for the Fmax wall.
//
// Why this does NOT re-introduce the M3 cursor wall: each unrolled slot `i` only ever handles checks
// `{i, i+CHK_UNROLL, …}`, so its input mux is `(BP_C/CHK_UNROLL):1` — shallow — not the single 72:1
// select + 432-way demux that made M3's per-node cursor 44 logic levels. It is the same spatial unroll
// as M4, just `CHK_UNROLL`-wide instead of `BP_C`-wide, with a group cursor selecting which slice.
//
// Bit-exactness with M4 (hence the M0 golden `FixedRelayBp`) is structural: within each phase the nodes
// are independent — every Tanner edge belongs to exactly one check (check-major CSR) and one variable
// (var-major CSR) — and the groups PARTITION the nodes, so splitting M4's one-cycle phase into `G`
// grouped cycles touches disjoint e_cv / m_vc / ehat entries and changes only timing. The cross-group
// reductions M4 did combinationally (weight, all-satisfied, obs) accumulate across the group cycles
// instead (the M2 pattern). Verified in Verilator (`tb_bp_relay.cpp -DPARTIAL`) bit-for-bit vs the same
// golden at any (CHK_UNROLL, VAR_UNROLL).

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_partial #(
    // Nodes processed per cycle. Full unroll = (BP_C, BP_N); sequential = (1, 1). Defaults divide
    // 72/144 evenly (6 groups each) — a ~6× area cut vs M4, comfortably inside a xc7z020.
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
  localparam int WACC  = 16;                               // right-sized blend accumulator (M5 2a)
  localparam int WW    = $clog2(BP_N + 1);
  localparam int G_CHK = (BP_C + CHK_UNROLL - 1) / CHK_UNROLL;
  localparam int G_VAR = (BP_N + VAR_UNROLL - 1) / VAR_UNROLL;
  localparam int GW    = (G_CHK > G_VAR) ? $clog2(G_CHK + 1) : $clog2(G_VAR + 1);

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SAT, S_EMIT, S_DONE } state_t;
  state_t state;

  /* verilator lint_off UNUSEDSIGNAL */
  (* dont_touch = "true" *) logic                       s_reg  [BP_C];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc   [BP_E];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv   [BP_E];
  (* dont_touch = "true" *) logic                       ehat   [BP_N];
  (* dont_touch = "true" *) logic                       best_e [BP_N];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter;
  logic [GW-1:0] grp;                     // group cursor within a phase
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
            leg <= '0; iter <= '0; grp <= '0;
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ----------------------------------------------------------------- CHK_UNROLL checks / cycle
        S_CHECK: begin
          automatic int lo, hi, argmin, e, c;
          automatic logic neg, excl;
          automatic logic [MSG_BITS-1:0] min1, min2, a, exmin, mag;
          automatic logic signed [MSG_BITS-1:0] m;
          for (int i = 0; i < CHK_UNROLL; i++) begin
            c = grp * CHK_UNROLL + i;                 // runtime group → (BP_C/CHK_UNROLL):1 slot mux
            if (c < BP_C) begin
              lo = BP_CHECK_OFF[c];
              hi = BP_CHECK_OFF[c + 1];
              neg = s_reg[c];
              min1 = INF; min2 = INF; argmin = -1;
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
              for (int k = 0; k < BP_CHK_DEG; k++) begin
                if (lo + k < hi) begin
                  e = BP_CHECK_EDGES[lo + k];
                  m = m_vc[e];
                  excl  = (m < 0) ? ~neg : neg;
                  exmin = (e == argmin) ? min2 : min1;
                  if (exmin == INF) exmin = '0;
                  mag = exmin - (exmin >> 3);
                  e_cv[e] <= excl ? -$signed(mag) : $signed(mag);
                end
              end
            end
          end
          if (grp == GW'(G_CHK - 1)) begin grp <= '0; state <= S_VAR; end
          else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- VAR_UNROLL variables / cycle
        S_VAR: begin
          automatic int lo, hi, e, v, wsum;
          automatic logic newbit;
          automatic logic signed [WACC-1:0] total, g, omg, ev, old, computed, num, blend;
          wsum = 0;
          for (int i = 0; i < VAR_UNROLL; i++) begin
            v = grp * VAR_UNROLL + i;
            if (v < BP_N) begin
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
              for (int k = 0; k < BP_VAR_DEG; k++) begin
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
            end
          end
          // Hamming weight accumulates across the var groups (M4 did it in one cycle).
          ehat_w <= (grp == '0 ? WW'(0) : ehat_w) + WW'(wsum);
          if (grp == GW'(G_VAR - 1)) begin grp <= '0; all_sat <= 1'b1; state <= S_SAT; end
          else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- H·ehat == s ? keep best
        S_SAT: begin
          automatic int lo, hi, c;
          automatic logic p, grp_sat, final_sat;
          grp_sat = 1'b1;
          for (int i = 0; i < CHK_UNROLL; i++) begin
            c = grp * CHK_UNROLL + i;
            if (c < BP_C) begin
              lo = BP_CHECK_OFF[c];
              hi = BP_CHECK_OFF[c + 1];
              p = s_reg[c];
              for (int k = 0; k < BP_CHK_DEG; k++)
                if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
              if (p != 1'b0) grp_sat = 1'b0;
            end
          end
          if (!grp_sat) all_sat <= 1'b0;
          if (grp == GW'(G_CHK - 1)) begin
            final_sat = all_sat & grp_sat;           // include this last group combinationally
            if (final_sat) begin
              found <= 1'b1;
              if (ehat_w < best_w) begin
                best_w <= ehat_w;
                for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
              end
            end
            grp <= '0;
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_EMIT;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else grp <= grp + 1'b1;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;
          automatic logic b;
          automatic int v;
          acc = (grp == '0) ? {BP_OBS{1'b0}} : obs_acc;
          for (int i = 0; i < VAR_UNROLL; i++) begin
            v = grp * VAR_UNROLL + i;
            if (v < BP_N) begin
              b = found ? best_e[v] : ehat[v];
              corr_out[v] <= b;
              if (b) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
            end
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
