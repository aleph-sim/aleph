// Q7-02 M7 — Step-0 OOC "fit gate": full modular unroll SKELETON for the circuit-level gross BB
// Tanner graph (BP_N=864 vars / BP_C=144 checks / BP_E=2952 edges, from `bb_gross_tanner.svh`
// regenerated here via `circgraph 1 0.003` — NOT the code-capacity graph the header's own filename
// historically held; see hw/Makefile's `bpcirc`/`bpbram*` targets for the same "circuit graph under the
// bb_gross_tanner.svh name" convention used by this M7 skeleton).
//
// PURPOSE: this is NOT a functionally-correct relay-BP decoder. It exists solely to answer one
// synthesis question cheaply (OOC on the KV260 part) before committing to the hierarchically-modular
// M7 design: does instantiating ALL 144 `check_minsum` + ALL 864 `var_update` submodules (Tasks 1-2),
// wired at the SAME CSR-constant edge indices `bp_relay_fast.sv`'s S_CHECK/S_VAR loops use (see the
// citations below), fit the device at all? So every wiring decision here mirrors bp_relay_fast.sv
// exactly EXCEPT that the per-node arithmetic clouds are now separate pipelined instances rather than
// one big combinational loop body, and the "iterate BP_LEGS x BP_ITERS times" schedule is entirely
// absent (this skeleton only wires ONE pass of check-update -> var-update, at leg 0) — a fit-gate cares
// about instantiated-logic VOLUME, not iteration count (iteration is time-multiplexed reuse of the SAME
// instances in the real M7 design, so it does not add area).
//
// WIRING (mirrors bp_relay_fast.sv `S_CHECK`/`S_VAR`, lines 121-181):
//   - check c, local slot k (k < BP_CHECK_OFF[c+1]-BP_CHECK_OFF[c]): its m_in comes from
//     `m_vc[BP_CHECK_EDGES[BP_CHECK_OFF[c]+k]]` — check_minsum's own `present[k]` masks k >= that
//     check's real degree (bp_relay_fast.sv:126-127). Its e_out[k] is written back to
//     `e_cv[BP_CHECK_EDGES[BP_CHECK_OFF[c]+k]]` (bp_relay_fast.sv:135-143).
//   - variable v, local slot k (k < BP_VAR_OFF[v+1]-BP_VAR_OFF[v]): unlike checks, a variable's edges
//     are the CONTIGUOUS range [BP_VAR_OFF[v], BP_VAR_OFF[v+1]) directly — edge e = BP_VAR_OFF[v]+k with
//     NO indirection table (bp_relay_fast.sv:162-179 indexes `e_cv[e]`/`m_vc[e]` with `e = lo+k`
//     directly; `BP_EDGE_VAR` is the redundant inverse map, unused here since we build the CSR the same
//     direction bp_relay_fast does). e_in[k] = e_cv[e] (the check's message), m_in[k] = m_vc[e] (this
//     var's OWN previous message — the "old" blend input; see var_update.sv's port doc), and its
//     m_out[k] is registered back into `m_vc[e]` (bp_relay_fast.sv:179). `lam = BP_LAMBDA[v]`,
//     `gam = BP_GAMMA[0*BP_N+v]` (leg 0 constant — this skeleton fixes the leg since a fit gate does not
//     care which leg's gammas are loaded, only that a representative 8-bit constant multiplier feeds
//     every var_update instance, per the task's leg-0-is-fine note).
//
// `m_vc`/`e_cv` are `dont_touch` REGISTERED arrays exactly as in bp_relay_fast.sv (its own dont_touch
// state, lines 50-53) — an EXTRA commit register stage beyond each submodule's own internal 2-cycle
// pipeline, so OOC synthesis sees the same "shared message register file + per-node compute pipeline"
// shape the real M7 design will have, not just floating submodule outputs. Every edge of `e_cv`/`m_vc`
// is driven by EXACTLY ONE generated `always_ff` (checks and vars partition BP_E disjointly — verified
// offline: `BP_CHECK_EDGES` is a permutation of 0..BP_E-1, and `BP_VAR_OFF` partitions 0..BP_E-1 into
// contiguous per-variable ranges), so there is no multi-driver conflict.
//
// ANTI-OPTIMIZATION: `en`/`syndrome_in` are primary inputs (OOC boundary — Vivado cannot constant-fold
// through them), `dont_touch` pins the message arrays, and every check's `e_out` + every var's
// `m_out`/`ehat_bit` additionally feed a single XOR-reduction registered into `out_bit` so nothing
// upstream of the reduction can be trimmed as unobservable, even independent of the dont_touch arrays.

`timescale 1ns / 1ps

/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_unroll_skeleton (
    input  logic clk,
    input  logic en,                     // single enable, fans out to every check_minsum/var_update
    input  logic syndrome_in [BP_C],      // per-check syndrome bit (drives check_minsum's `sbit`)
    output logic out_bit                  // registered XOR-reduction of every instance's outputs
);
  localparam int WACC = 16;               // matches bp_relay_fast.sv's WACC (not in the generated header)

  // ------------------------------------------------------------------- shared message state (dont_touch)
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc [BP_E];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv [BP_E];

  // per-instance output taps (top-level arrays so the reduction below can walk them with plain loops)
  logic signed [MSG_BITS-1:0] chk_e_out [BP_C][BP_CHK_DEG];
  logic signed [MSG_BITS-1:0] var_m_out [BP_N][BP_VAR_DEG];
  logic                       var_ehat  [BP_N];

  // ------------------------------------------------------------------------------- 144x check_minsum
  generate
    for (genvar c = 0; c < BP_C; c++) begin : g_chk
      localparam int LO   = BP_CHECK_OFF[c];
      localparam int HI   = BP_CHECK_OFF[c + 1];
      localparam int DEGC = HI - LO;

      logic signed [MSG_BITS-1:0] m_in_c    [BP_CHK_DEG];
      logic                       present_c [BP_CHK_DEG];

      for (genvar k = 0; k < BP_CHK_DEG; k++) begin : g_slot
        if (k < DEGC) begin : g_real
          assign m_in_c[k]    = m_vc[BP_CHECK_EDGES[LO+k]];
          assign present_c[k] = 1'b1;
        end else begin : g_pad
          assign m_in_c[k]    = '0;
          assign present_c[k] = 1'b0;
        end
      end

      check_minsum #(
          .MW (MSG_BITS),
          .DEG(BP_CHK_DEG)
      ) u_chk (
          .clk    (clk),
          .en     (en),
          .sbit   (syndrome_in[c]),
          .m_in   (m_in_c),
          .present(present_c),
          .e_out  (chk_e_out[c])
      );

      for (genvar k = 0; k < BP_CHK_DEG; k++) begin : g_wr
        if (k < DEGC) begin : g_real
          always_ff @(posedge clk) e_cv[BP_CHECK_EDGES[LO+k]] <= chk_e_out[c][k];
        end
      end
    end
  endgenerate

  // -------------------------------------------------------------------------------- 864x var_update
  generate
    for (genvar v = 0; v < BP_N; v++) begin : g_var
      localparam int LO   = BP_VAR_OFF[v];
      localparam int HI   = BP_VAR_OFF[v + 1];
      localparam int DEGV = HI - LO;

      logic signed [MSG_BITS-1:0] e_in_v    [BP_VAR_DEG];
      logic signed [MSG_BITS-1:0] m_in_v    [BP_VAR_DEG];
      logic                       present_v [BP_VAR_DEG];

      for (genvar k = 0; k < BP_VAR_DEG; k++) begin : g_slot
        if (k < DEGV) begin : g_real
          assign e_in_v[k]    = e_cv[LO+k];   // e = lo+k directly — bp_relay_fast.sv:172
          assign m_in_v[k]    = m_vc[LO+k];   // "old" message — bp_relay_fast.sv:173
          assign present_v[k] = 1'b1;
        end else begin : g_pad
          assign e_in_v[k]    = '0;
          assign m_in_v[k]    = '0;
          assign present_v[k] = 1'b0;
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
          .en      (en),
          .lam     (BP_LAMBDA[v][MSG_BITS-1:0]),
          .gam     (BP_GAMMA[v][MSG_BITS-1:0]),   // leg 0 == BP_GAMMA[0*BP_N+v]
          .e_in    (e_in_v),
          .m_in    (m_in_v),
          .present (present_v),
          .m_out   (var_m_out[v]),
          .ehat_bit(var_ehat[v])
      );

      for (genvar k = 0; k < BP_VAR_DEG; k++) begin : g_wr
        if (k < DEGV) begin : g_real
          always_ff @(posedge clk) m_vc[LO+k] <= var_m_out[v][k];
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------------- anti-optimization reduction
  logic acc_comb;
  always_comb begin
    acc_comb = 1'b0;
    for (int c = 0; c < BP_C; c++)
      for (int k = 0; k < BP_CHK_DEG; k++) acc_comb = acc_comb ^ (^chk_e_out[c][k]);
    for (int v = 0; v < BP_N; v++) begin
      for (int k = 0; k < BP_VAR_DEG; k++) acc_comb = acc_comb ^ (^var_m_out[v][k]);
      acc_comb = acc_comb ^ var_ehat[v];
    end
  end

  logic out_bit_r;
  always_ff @(posedge clk) out_bit_r <= acc_comb;
  assign out_bit = out_bit_r;

endmodule
