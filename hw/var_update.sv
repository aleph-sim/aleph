// Q7-02 M7 — standalone variable-update submodule (`var_update`), 2-cycle pipelined, generic MW/DEG.
//
// Bit-exact twin of ONE variable's inner loop from `bp_relay_fast.sv`'s S_VAR state (lines 153-182):
// `total = lambda + sum_present e_cv`; `ehat_bit = total`'s sign bit (`total[WACC-1]`); per real edge
// the blended message `num = omg*computed + gam*old` (`omg = (1<<FRAC)-gam`, `computed = total -
// e_cv[edge]`, `old` = the edge's OWN current m_vc — the pre-image being replaced) is right-shifted by
// FRAC and clamped to +-MAXMAG. `present[k]` masks real vs unused slots exactly like the source loop's
// `if (lo+k<hi)` guard — DEG is the graph's max variable degree (BP_VAR_DEG), one module variant
// handles every variable regardless of its real degree.
//
// All the S_VAR intermediate variables (`total,g,omg,ev,old,computed,num,blend`) are declared at
// WACC(=16) bits, matching bp_relay_fast EXACTLY — including its narrow (WACC-wide, not double-wide)
// multiply-accumulate for `num`. For typical operating magnitudes (MW=8, DEG<=~6) `total`/`computed`
// stay well inside +-32767, but `omg*computed` alone can already exceed it, so the product genuinely
// wraps mod 2^16 before the FRAC shift and clamp. This module intentionally reproduces that (it's
// bp_relay_fast's actual synthesized behavior) rather than "fixing" it, so it stays bit-exact with the
// reference decoder; see `tb_var_update.cpp`'s C++ reference for the matching 16-bit truncation.
//
// This is the SECOND of the two small per-check/per-var submodules the M7 hierarchically-modular
// relay-BP core stamps out many times (one per variable node, 864x on the circuit graph); pipelining
// it (rather than leaving it one combinational cloud) is what makes that stamping synthesis-friendly:
//   stage 1 (posedge, gated by `en`): reduce `total = lam + sum of present e_in slots` -> registered
//            `total_r`; also register the sign bit as `ehat_bit_s1`, and register `gam`/`e_in`/`m_in`/
//            `present` themselves so stage 2 sees the SAME snapshot the reduction ran over.
//   stage 2 (posedge, free-running off the stage-1 registers): per real slot, exclude this edge's own
//            contribution, blend with the old message via the per-var gamma, clamp, and emit; also
//            re-registers `ehat_bit_s1` -> `ehat_bit` so both outputs become valid on the SAME edge
//            (2 clocks after `en`), matching `check_minsum`'s convention. Unused slots emit 0 (no
//            inferred latch, deterministic pad — bp_relay_fast never writes m_vc for a non-real slot
//            at all, so 0 there is a don't-care downstream — picked here for a fully-defined,
//            bit-reproducible output over the WHOLE m_out array).

`timescale 1ns / 1ps

module var_update #(
    parameter int MW     = 8,
    parameter int WACC   = 16,
    parameter int FRAC   = 3,
    parameter int DEG    = 6,
    parameter int MAXMAG = 127
) (
    input  logic                 clk,
    input  logic                 en,             // pulse to start; result valid 2 clocks later
    input  logic signed [MW-1:0] lam,            // this var's lambda (BP_LAMBDA[v])
    input  logic signed [MW-1:0] gam,            // this var's gamma for the current leg
    input  logic signed [MW-1:0] e_in    [DEG],  // this var's edges' e_cv
    input  logic signed [MW-1:0] m_in    [DEG],  // this var's edges' current m_vc (the "old")
    input  logic                 present [DEG],  // 1 = real edge, 0 = unused slot (deg < DEG)
    output logic signed [MW-1:0] m_out   [DEG],  // updated m_vc per edge
    output logic                 ehat_bit
);

  // ------------------------------------------------------------------- stage 1: reduction registers
  logic signed [WACC-1:0] total_r;
  logic                   ehat_bit_s1;
  logic signed [MW-1:0]   gam_r;
  logic signed [MW-1:0]   e_in_r    [DEG];
  logic signed [MW-1:0]   m_in_r    [DEG];
  logic                   present_r [DEG];

  always_ff @(posedge clk) begin
    if (en) begin
      automatic logic signed [WACC-1:0] total;

      total = signed'(WACC'(lam));
      for (int k = 0; k < DEG; k++)
        if (present[k]) total = total + signed'(WACC'(e_in[k]));

      total_r     <= total;
      ehat_bit_s1 <= total[WACC-1];
      gam_r       <= gam;
      for (int k = 0; k < DEG; k++) begin
        e_in_r[k]    <= e_in[k];
        m_in_r[k]    <= m_in[k];
        present_r[k] <= present[k];
      end
    end
  end

  // ------------------------------------------------------------------- stage 2: emit (free-running)
  always_ff @(posedge clk) begin
    automatic logic signed [WACC-1:0] g, omg;

    ehat_bit <= ehat_bit_s1;

    g   = signed'(WACC'(gam_r));
    omg = signed'(WACC'(1 << FRAC)) - g;

    for (int k = 0; k < DEG; k++) begin
      if (present_r[k]) begin
        automatic logic signed [WACC-1:0] ev, old, computed, num, blend;

        ev       = signed'(WACC'(e_in_r[k]));
        old      = signed'(WACC'(m_in_r[k]));
        computed = total_r - ev;
        num      = omg * computed + g * old;
        blend    = num >>> FRAC;
        if (blend > signed'(WACC'(MAXMAG))) blend = signed'(WACC'(MAXMAG));
        else if (blend < -signed'(WACC'(MAXMAG))) blend = -signed'(WACC'(MAXMAG));
        m_out[k] <= blend[MW-1:0];
      end else begin
        m_out[k] <= '0;
      end
    end
  end
endmodule
