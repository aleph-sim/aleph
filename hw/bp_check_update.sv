// Q7-02 M1 — min-sum check→variable update for the fixed-point relay-BP decoder, COMBINATIONAL FORM.
//
// One half-iteration of belief propagation: given the current variable→check messages `m_vc` (one
// signed MSG_BITS word per Tanner edge) and the syndrome `s_in` (one bit per check), compute the
// check→variable messages `e_cv` by the normalised min-sum rule
//
//     E_{c→v} = (-1)^{s_c} · α · (∏_{v'≠v} sign M_{v'→c}) · min_{v'≠v} |M_{v'→c}|,   α = 7/8.
//
// This mirrors `FixedRelayBp::check_update` (crates/aleph-qec/src/fixed_bp.rs) bit-for-bit:
//   * exclusive-minimum via the two-smallest-magnitudes (min1,min2)+argmin trick — the excluded min
//     is min2 for the argmin edge, else min1;
//   * α = 7/8 is `mag - (mag >> 3)` (multiply-free);
//   * the sign excluding this edge is the check parity XOR the syndrome bit XOR this edge's own sign.
//
// Like the Q6-02 combinational UF draft, this is a single `always_comb` cloud — correct but not yet
// timing-closable; M2 sequentialises it into a clocked FSM. The graph is baked in via the generated
// `bb_gross_tanner.svh` (`BP_CHECK_OFF` / `BP_CHECK_EDGES`).
//
// Verified in Verilator (`tb_bp_check.cpp`) against `bp_check_vectors.txt` — the fixed-point Rust
// golden — bit-for-bit on all 432 output messages over 256 random input vectors.

`timescale 1ns / 1ps
// The shared graph header also carries the params M2's full FSM needs (priors, γ, var-major CSR);
// this check-update module uses only the check-major subset, so silence "unused" for the rest.
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_check_update (
    input  logic signed [MSG_BITS-1:0] m_vc [BP_E],   // variable→check messages
    input  logic                       s_in [BP_C],   // syndrome bit per check
    output logic signed [MSG_BITS-1:0] e_cv [BP_E]    // check→variable messages
);
  // Magnitudes are in [0, MAX_MAG] (≤ 127 at Q5.3), so an unsigned MSG_BITS word holds them and the
  // all-ones sentinel (255) exceeds every real magnitude — it acts as +∞ for the running minima.
  localparam logic [MSG_BITS-1:0] INF = '1;

  always_comb begin
    // Default every output so no edge index is ever left unassigned (no inferred latch).
    for (int i = 0; i < BP_E; i++) e_cv[i] = '0;

    for (int c = 0; c < BP_C; c++) begin
      logic                  neg;     // running product sign (true ⇒ negative)
      logic [MSG_BITS-1:0]   min1;    // smallest magnitude
      logic [MSG_BITS-1:0]   min2;    // second-smallest magnitude
      int                    argmin;  // edge holding the smallest magnitude
      int                    lo, hi;

      lo  = BP_CHECK_OFF[c];
      hi  = BP_CHECK_OFF[c + 1];
      neg = s_in[c];
      min1 = INF;
      min2 = INF;
      argmin = -1;

      // Pass 1: overall sign, two smallest magnitudes, and the argmin edge.
      for (int k = lo; k < hi; k++) begin
        int e;
        logic signed [MSG_BITS-1:0] m;
        logic [MSG_BITS-1:0] a;
        e = BP_CHECK_EDGES[k];
        m = m_vc[e];
        if (m < 0) neg = ~neg;
        a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);   // |m|, no -128 (saturated to ±MAX_MAG)
        if (a < min1) begin
          min2 = min1;
          min1 = a;
          argmin = e;
        end else if (a < min2) begin
          min2 = a;
        end
      end

      // Pass 2: exclude each edge's own contribution.
      for (int k = lo; k < hi; k++) begin
        int e;
        logic signed [MSG_BITS-1:0] m;
        logic excl_neg;
        logic [MSG_BITS-1:0] exmin, mag;
        e = BP_CHECK_EDGES[k];
        m = m_vc[e];
        excl_neg = (m < 0) ? ~neg : neg;
        exmin = (e == argmin) ? min2 : min1;
        // Degree-1 check would leave the excluded min at the sentinel; treat as 0 (no constraint).
        if (exmin == INF) exmin = '0;
        // α = 7/8 (multiply-free). exmin ≤ MAX_MAG (127) ⇒ mag ≤ 111 < MAX_MAG, so no clamp needed.
        mag = exmin - (exmin >> 3);
        // mag < 128 ⇒ its sign bit is 0, so $signed(mag) is +mag and both branches stay in MSG_BITS.
        e_cv[e] = excl_neg ? -$signed(mag) : $signed(mag);
      end
    end
  end
endmodule
