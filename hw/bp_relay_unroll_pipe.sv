// Q7-02 M7 — MODULAR PARTIAL-UNROLL relay-BP decoder core (hierarchically-modular, pipelined slots).
// SUPERSEDED by `bp_relay_banked.sv` (its runtime-`grp` gather muxes stall Vivado area-opt at circuit
// scale) — kept as the G-invariance / FSM reference, NOT a ship vehicle.
//
// The KV260 fit-gate (`bp_unroll_skeleton.sv` OOC) proved a FULL modular unroll — one `check_minsum`
// per check (144x) + one `var_update` per variable (864x) — is 3.9x too big for the part. This core
// keeps the SAME two unit-verified submodules (Tasks 1-2) but STAMPS ONLY 1/NGROUP of them and
// time-multiplexes each slot across NGROUP groups per BP phase, so instantiated logic volume drops ~NGROUP x
// while the decode is bit-identical to `bp_relay_fast.sv` (grouping is a pure scheduling change, not an
// algorithm change — see the group-gather idiom in `bp_relay_partial_fast.sv`).
//
// STRUCTURE (per phase, CHK_UNROLL check_minsum slots / VAR_UNROLL var_update slots):
//   - Each slot `i` gathers its active group `grp`'s check/variable operands at COMPILE-TIME-CONSTANT
//     edge indices (`c = g*CHK_UNROLL+i`, `v = g*VAR_UNROLL+i`), muxed by `grp == g` — never a runtime
//     index into the big `m_vc`/`e_cv`/`s_reg`/`ehat` flop arrays (the partial_fast rule).
//   - The gathered operands feed the pipelined SUBMODULE instance (2-cycle latency: pulse `en`, result
//     2 clocks later). This modular structure is exactly what the M7 fit-gate proved synthesizes, vs the
//     inline-math partial core that stalled Vivado synth.
//   - The FSM sweeps a phase cursor `pc` = 0..NGROUP+1: it LAUNCHES group `pc` (pulses `en`, gathers)
//     when `pc < NGROUP`, and SCATTERS the registered outputs of group `pc-2` (2 cycles behind) when
//     `pc >= 2` — a 2-deep software pipeline over the submodules' 2-cycle latency, so each phase costs
//     NGROUP+2 cycles. Groups are disjoint in edges, so the overlapped launch(read)/scatter(write) touch
//     independent register clouds (S_CHECK writes only e_cv while reading m_vc; S_VAR's group pc reads
//     m_vc[group pc] while group pc-2's write hits the disjoint m_vc[group pc-2]).
//
// SAT-overlap + best-kept + 6x10 schedule are carried verbatim from `bp_relay_fast.sv`: the syndrome
// parity S_SAT reads only `ehat`/`s_reg` (no submodule), so it runs group-by-group in parallel with the
// next S_CHECK's launches; a trailing S_SATF sweeps the final iteration's decision.
//
// The decode is INDEPENDENT of NGROUP (grouping only changes cycle count) — verified bit-for-bit vs the
// `bp_relay_fast` golden at NGROUP=2 and NGROUP=4 in `tb_bp_unroll_pipe.cpp`.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_unroll_pipe #(
    parameter int NGROUP = 4
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
  localparam int WACC       = 16;                              // matches bp_relay_fast.sv's WACC
  localparam int WW         = $clog2(BP_N + 1);
  localparam int CHK_UNROLL = (BP_C + NGROUP - 1) / NGROUP;    // check slots stamped (ceil)
  localparam int VAR_UNROLL = (BP_N + NGROUP - 1) / NGROUP;    // var slots stamped (ceil)

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SATF, S_EMIT, S_DONE } state_t;
  state_t state;

  /* verilator lint_off UNUSEDSIGNAL */
  (* dont_touch = "true" *) logic                       s_reg  [BP_C];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc   [BP_E];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv   [BP_E];
  (* dont_touch = "true" *) logic                       ehat   [BP_N];
  (* dont_touch = "true" *) logic                       best_e [BP_N];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat, sat_pending;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter, pc;                                 // pc = phase/group cursor (0..NGROUP+1)
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  // submodule enables: launch the slots only while there are groups left to start (pc < NGROUP)
  logic en_chk, en_var;
  assign en_chk = (state == S_CHECK) && (pc < NGROUP);
  assign en_var = (state == S_VAR)   && (pc < NGROUP);

  // registered submodule outputs (2 clocks after the group's launch)
  logic signed [MSG_BITS-1:0] chk_e_out    [CHK_UNROLL][BP_CHK_DEG];
  logic signed [MSG_BITS-1:0] var_m_out    [VAR_UNROLL][BP_VAR_DEG];
  logic                       var_ehat_out [VAR_UNROLL];

  // --------------------------------------------------------------------- CHK_UNROLL check_minsum slots
  generate
    for (genvar i = 0; i < CHK_UNROLL; i++) begin : gchk
      logic                       sbit_i;
      logic signed [MSG_BITS-1:0] m_in_i    [BP_CHK_DEG];
      logic                       present_i [BP_CHK_DEG];

      // gather this slot's active-group check inputs by `pc`, at CONSTANT edge indices (partial_fast idiom)
      always_comb begin
        sbit_i = 1'b0;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          m_in_i[k]    = '0;
          present_i[k] = 1'b0;
        end
        for (int g = 0; g < NGROUP; g++)
          if (g * CHK_UNROLL + i < BP_C && pc == g) begin
            automatic int c  = g * CHK_UNROLL + i;
            automatic int lo = BP_CHECK_OFF[c];
            automatic int hi = BP_CHECK_OFF[c + 1];
            sbit_i = s_reg[c];
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) begin
                m_in_i[k]    = m_vc[BP_CHECK_EDGES[lo + k]];
                present_i[k] = 1'b1;
              end
          end
      end

      check_minsum #(
          .MW (MSG_BITS),
          .DEG(BP_CHK_DEG)
      ) u_chk (
          .clk    (clk),
          .en     (en_chk),
          .sbit   (sbit_i),
          .m_in   (m_in_i),
          .present(present_i),
          .e_out  (chk_e_out[i])
      );
    end
  endgenerate

  // ---------------------------------------------------------------------- VAR_UNROLL var_update slots
  generate
    for (genvar i = 0; i < VAR_UNROLL; i++) begin : gvar
      logic signed [MSG_BITS-1:0] lam_i, gam_i;
      logic signed [MSG_BITS-1:0] e_in_i    [BP_VAR_DEG];
      logic signed [MSG_BITS-1:0] m_in_i    [BP_VAR_DEG];
      logic                       present_i [BP_VAR_DEG];

      // gather this slot's active-group variable inputs by `pc`, at CONSTANT edge indices
      always_comb begin
        lam_i = '0;
        gam_i = '0;
        for (int k = 0; k < BP_VAR_DEG; k++) begin
          e_in_i[k]    = '0;
          m_in_i[k]    = '0;
          present_i[k] = 1'b0;
        end
        for (int g = 0; g < NGROUP; g++)
          if (g * VAR_UNROLL + i < BP_N && pc == g) begin
            automatic int v  = g * VAR_UNROLL + i;
            automatic int lo = BP_VAR_OFF[v];
            automatic int hi = BP_VAR_OFF[v + 1];
            lam_i = BP_LAMBDA[v][MSG_BITS-1:0];
            gam_i = BP_GAMMA[leg * BP_N + v][MSG_BITS-1:0];    // this var's gamma for the current leg
            for (int k = 0; k < BP_VAR_DEG; k++)
              if (lo + k < hi) begin
                e_in_i[k]    = e_cv[lo + k];                   // the check's message (this iter's S_CHECK)
                m_in_i[k]    = m_vc[lo + k];                   // this edge's CURRENT m_vc (the "old" blend)
                present_i[k] = 1'b1;
              end
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
          .en      (en_var),
          .lam     (lam_i),
          .gam     (gam_i),
          .e_in    (e_in_i),
          .m_in    (m_in_i),
          .present (present_i),
          .m_out   (var_m_out[i]),
          .ehat_bit(var_ehat_out[i])
      );
    end
  endgenerate

  // ------------------------------------------------------------------------------------ control FSM
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
            found       <= 1'b0;
            best_w      <= '1;
            ehat_w      <= '0;
            all_sat     <= 1'b1;
            sat_pending <= 1'b0;                       // no decision to check before the first S_VAR
            leg <= '0; iter <= '0; pc <= '0;
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ------------------------------ launch group `pc` (min-sum) + scatter group `pc-2`  ‖  S_SAT
        S_CHECK: begin
          automatic logic grp_sat, final_sat, p;
          // ---- overlapped S_SAT: parity of the PREVIOUS ehat on the LAUNCHED group's checks ----
          if (pc < NGROUP && sat_pending) begin
            grp_sat = 1'b1;
            for (int i = 0; i < CHK_UNROLL; i++)
              for (int g = 0; g < NGROUP; g++)
                if (g * CHK_UNROLL + i < BP_C && pc == g) begin
                  automatic int c  = g * CHK_UNROLL + i;
                  automatic int lo = BP_CHECK_OFF[c];
                  automatic int hi = BP_CHECK_OFF[c + 1];
                  p = s_reg[c];
                  for (int k = 0; k < BP_CHK_DEG; k++)
                    if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
                  if (p != 1'b0) grp_sat = 1'b0;
                end
            if (!grp_sat) all_sat <= 1'b0;
            if (pc == NGROUP - 1) begin               // last launched group finalises the SAT sweep
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

          // ---- scatter group `pc-2`'s check outputs back to e_cv at the same CONSTANT edges ----
          if (pc >= 2) begin
            automatic int sg = pc - 2;
            for (int i = 0; i < CHK_UNROLL; i++)
              for (int g = 0; g < NGROUP; g++)
                if (g * CHK_UNROLL + i < BP_C && sg == g) begin
                  automatic int lo = BP_CHECK_OFF[g * CHK_UNROLL + i];
                  automatic int hi = BP_CHECK_OFF[g * CHK_UNROLL + i + 1];
                  for (int k = 0; k < BP_CHK_DEG; k++)
                    if (lo + k < hi) e_cv[BP_CHECK_EDGES[lo + k]] <= chk_e_out[i][k];
                end
          end

          // ---- advance the phase cursor; on drain, reset the SAT accumulator and step to S_VAR ----
          if (pc == NGROUP + 1) begin
            pc      <= '0;
            all_sat <= 1'b1;
            state   <= S_VAR;
          end else pc <= pc + 1;
          lat <= lat + 16'd1;
        end

        // ------------------------------ launch group `pc` (var-update) + scatter group `pc-2`
        S_VAR: begin
          if (pc == 0) ehat_w <= '0;                  // start the fresh decision-weight accumulation

          if (pc >= 2) begin                          // scatter group `pc-2`'s var outputs
            automatic int sg = pc - 2;
            automatic int wsum = 0;
            for (int i = 0; i < VAR_UNROLL; i++)
              for (int g = 0; g < NGROUP; g++)
                if (g * VAR_UNROLL + i < BP_N && sg == g) begin
                  automatic int v  = g * VAR_UNROLL + i;
                  automatic int lo = BP_VAR_OFF[v];
                  automatic int hi = BP_VAR_OFF[v + 1];
                  ehat[v] <= var_ehat_out[i];
                  wsum = wsum + (var_ehat_out[i] ? 1 : 0);
                  for (int k = 0; k < BP_VAR_DEG; k++)
                    if (lo + k < hi) m_vc[lo + k] <= var_m_out[i][k];
                end
            ehat_w <= ehat_w + WW'(wsum);
          end

          if (pc == NGROUP + 1) begin
            pc          <= '0;
            sat_pending <= 1'b1;                       // a fresh decision is now available to check
            // advance iteration / leg; the SAT for THIS ehat runs next (in S_CHECK, or S_SATF if last)
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_SATF;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else pc <= pc + 1;
          lat <= lat + 16'd1;
        end

        // ----------------------------- trailing S_SAT for the final ehat (no following S_CHECK)
        S_SATF: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat = 1'b1;
          for (int i = 0; i < CHK_UNROLL; i++)
            for (int g = 0; g < NGROUP; g++)
              if (g * CHK_UNROLL + i < BP_C && pc == g) begin
                automatic int c  = g * CHK_UNROLL + i;
                automatic int lo = BP_CHECK_OFF[c];
                automatic int hi = BP_CHECK_OFF[c + 1];
                p = s_reg[c];
                for (int k = 0; k < BP_CHK_DEG; k++)
                  if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
                if (p != 1'b0) grp_sat = 1'b0;
              end
          if (!grp_sat) all_sat <= 1'b0;
          if (pc == NGROUP - 1) begin
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
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat -> obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;
          automatic logic b;
          acc = (pc == 0) ? {BP_OBS{1'b0}} : obs_acc;
          for (int i = 0; i < VAR_UNROLL; i++)
            for (int g = 0; g < NGROUP; g++)
              if (g * VAR_UNROLL + i < BP_N && pc == g) begin
                automatic int v = g * VAR_UNROLL + i;
                b = found ? best_e[v] : ehat[v];
                corr_out[v] <= b;
                if (b) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
              end
          obs_acc <= acc;
          if (pc == NGROUP - 1) begin
            pc    <= '0;
            state <= S_DONE;
          end else pc <= pc + 1;
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
