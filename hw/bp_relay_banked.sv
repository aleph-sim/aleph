// Q7-02 M7 — K-BANKED relay-BP decoder core (`bp_relay_banked`): beta-split, check-major LUTRAM banks.
//
// Same schedule / quantisation / keep-lowest-weight-valid decision as `bp_relay_fast.sv` and the modular
// partial-unroll `bp_relay_unroll_pipe.sv`, and it STAMPS the SAME two unit-verified submodules
// (`check_minsum`, `var_update`, Tasks 1-2). The difference is WHERE the per-edge messages live: not in
// flop arrays (which rebuild the runtime-index register-file mux wall that stalled Vivado area-opt), but
// in many small single-write-port LUTRAM banks, one message per (bank,row) addressed by a compile-time
// group/slot map baked into the header (Task 9). Every runtime index touches ONLY: (a) the banks' async
// read OUTPUT wires, tapped at compile-time-constant bank ids, and (b) small ehat/s_reg flop arrays. No
// runtime index ever reaches a bank memory CELL or the graph tables by a computed edge index — that is
// the LUTRAM-inference rule the M7 synthesis post-mortems require.
//
// MEMORY MAP (spec A2.1):
//   * m_cm (v->c messages, read by the CHK phase): 2*W*CHK_DEG half-banks x GC rows. Bank of edge e =
//     slot_of_chk(EDGE_CHK[e])*CHK_DEG + EDGE_POS[e]; half = EDGE_BETA[e]; row = grp_of_chk(EDGE_CHK[e]).
//     One sync write port; async read at row = pc (uniform across banks — every CHK lane of group pc
//     reads its own bank at row pc). Beta is compile-time-constant per lane/group, so the 2:1 beta-select
//     folds into a constant bank tap.
//   * e_cm (c->v messages, written by CHK, read by VAR): W*CHK_DEG banks x GC rows, same (j,k) map, no
//     beta split, TWO async read address ports (the guaranteed <=2 readers/bank/var-group; readers of a
//     bank in a var-group are ordered by (i,d): first -> port A, second -> port B).
//   * m_vm (the "old" v->c message the var-update blends against): V*VAR_DEG banks x GV rows. Bank =
//     (var-slot i, edge d); read row = pc, write row = pc-2 (disjoint groups in the software pipeline).
//
// FSM mirrors `bp_relay_unroll_pipe.sv` (S_CHECK/S_VAR launch-group-pc / scatter-group-(pc-2) 2-deep
// software pipeline over the submodules' 2-cycle latency; SAT folded into the S_CHECK launches;
// sat_pending; best-kept commit; obs reduction; sync reset) with three additions the banked layout forces
// or the drop-in contract needs: (1) an S_INIT state that seeds m_cm/m_vm with lambda through the write
// ports (the unroll core could flop-init m_vc in S_IDLE; banked messages must be written a group/cycle);
// (2) an `early_exit` input — at the S_CHECK-entry SAT finalize, `early_exit && found` jumps to S_EMIT
// (the `bp_relay_bram_dp` S_SAT2 semantics: found the moment a decision satisfies the syndrome => first
// valid); (3) a 32-bit `latency_cycles` output. W/V/GC/GV come from the header, never module parameters,
// so header and RTL cannot desync.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_banked (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                early_exit,               // stop at the first syndrome-valid decision
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],
    output logic [BP_OBS-1:0]   obs_flip,
    output logic                valid_flag,
    output logic [31:0]         latency_cycles
);
  // ------------------------------------------------------------------------------- sizes / geometry
  localparam int WACC = 16;                          // matches bp_relay_fast.sv / var_update WACC
  localparam int WW   = $clog2(BP_N + 1);
  localparam int W    = BP_BANK_W;                   // check slots stamped (checks per check-group)
  localparam int V    = BP_BANK_V;                   // var slots stamped (vars per var-group)
  localparam int GC   = BP_GC;                       // number of check groups
  localparam int GV   = BP_GV;                       // number of var groups
  localparam int NHB  = 2 * BP_BANK_W * BP_CHK_DEG;  // m_cm half-banks
  localparam int NEB  = BP_BANK_W * BP_CHK_DEG;      // e_cm banks
  localparam int NVB  = BP_BANK_V * BP_VAR_DEG;      // m_vm banks
  localparam int BWC  = $clog2(BP_GC);               // m_cm / e_cm row address width
  localparam int BWV  = $clog2(BP_GV);               // m_vm row address width

  /* verilator lint_off UNUSEDSIGNAL */
  // ============================================================ elaboration helpers over header tables
  // All are called only with compile-time-constant arguments (genvar / unrolled-loop indices, or a
  // group index gated to a constant by `pc == g`), so Verilator constant-folds every use into a fixed
  // bank id / row / constant — never a runtime index into the graph tables.
  function automatic int chk_at(input int g, input int j);
    return BP_CHK_AT[g * BP_BANK_W + j];
  endfunction
  function automatic int var_at(input int h, input int i);
    return BP_VAR_AT[h * BP_BANK_V + i];
  endfunction
  function automatic int chk_deg(input int c);
    return BP_CHECK_OFF[c + 1] - BP_CHECK_OFF[c];
  endfunction
  function automatic int var_deg(input int v);
    return BP_VAR_OFF[v + 1] - BP_VAR_OFF[v];
  endfunction
  function automatic int grp_of_chk(input int c);
    for (int g = 0; g < BP_GC; g++)
      for (int j = 0; j < BP_BANK_W; j++)
        if (BP_CHK_AT[g * BP_BANK_W + j] == c) return g;
    return -1;
  endfunction
  function automatic int slot_of_chk(input int c);
    for (int g = 0; g < BP_GC; g++)
      for (int j = 0; j < BP_BANK_W; j++)
        if (BP_CHK_AT[g * BP_BANK_W + j] == c) return j;
    return -1;
  endfunction
  // edge index at (check-group g, slot j, position k), or -1 if empty slot / k >= that check's degree.
  function automatic int edge_at(input int g, input int j, input int k);
    automatic int c = chk_at(g, j);
    if (c < 0) return -1;
    if (k >= chk_deg(c)) return -1;
    return BP_CHECK_EDGES[BP_CHECK_OFF[c] + k];       // edge whose EDGE_POS == k (verified in TB step)
  endfunction
  // edge index at (var-group h, slot i, edge d), or -1 if empty slot / d >= that var's degree.
  function automatic int vedge_at(input int h, input int i, input int d);
    automatic int v = var_at(h, i);
    if (v < 0) return -1;
    if (d >= var_deg(v)) return -1;
    return BP_VAR_OFF[v] + d;                         // var edges are variable-major contiguous
  endfunction
  // e_cm bank of edge e (also the un-split m_cm bank before the beta half).
  function automatic int ecm_bank(input int e);
    return slot_of_chk(BP_EDGE_CHK[e]) * BP_CHK_DEG + BP_EDGE_POS[e];
  endfunction
  // m_cm half-bank of edge e = 2 * (bank) + beta.
  function automatic int hb_of_edge(input int e);
    return 2 * (slot_of_chk(BP_EDGE_CHK[e]) * BP_CHK_DEG + BP_EDGE_POS[e]) + BP_EDGE_BETA[e];
  endfunction
  // port of the (i,d) reader of its e_cm bank within var-group h: readers ordered by (i,d), first -> A(0),
  // second -> B(1). Guaranteed <=2 readers/bank/var-group by the offline (Task 9) assignment.
  function automatic int ecm_port(input int h, input int i, input int d);
    automatic int bank = ecm_bank(vedge_at(h, i, d));
    automatic int cnt  = 0;
    for (int ii = 0; ii < BP_BANK_V; ii++)
      for (int dd = 0; dd < BP_VAR_DEG; dd++) begin
        if (ii == i && dd == d) return cnt;
        begin
          automatic int e = vedge_at(h, ii, dd);
          if (e >= 0 && ecm_bank(e) == bank) cnt = cnt + 1;
        end
      end
    return cnt;
  endfunction

  // ================================================================== elaboration guards (Task-10 review)
  // The banked datapath silently depends on three offline (Task-9 emitter) guarantees. If a future emitter
  // split ever violates one, the RTL corrupts messages with NO other symptom: (a) a second writer on a
  // half-bank's single write port would last-write-win; (b) a third reader of an e_cm bank has no port and
  // is dropped; (c) a wrong BP_EDGE_POS mis-taps the m_cm half-bank in the CHK gather. Recompute and enforce
  // all three at elaboration (time-0 `initial`, constant-folded over the header tables — no runtime hardware;
  // an initial block of system tasks synthesises to nothing). Fail LOUDLY on any violation.
  initial begin : elab_guards
    automatic int fails = 0;
    // (a)/(b): per var-group, accumulate writers-per-half-bank and readers-per-e_cm-bank in ONE pass
    // over the group's present edges, then scan the counters. Counting in a single (i,d) pass (rather
    // than re-scanning all edges for each bank) keeps Verilator's constant-unroll of this initial block
    // O(GV*V*VAR_DEG) instead of O(GV*(NHB+NEB)*V*VAR_DEG) — the latter symbolically explodes cvt.
    for (int h = 0; h < GV; h++) begin
      automatic int wcnt [NHB];
      automatic int rcnt [NEB];
      for (int b = 0; b < NHB; b++) wcnt[b] = 0;
      for (int b = 0; b < NEB; b++) rcnt[b] = 0;
      for (int i = 0; i < V; i++)
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          automatic int e = vedge_at(h, i, d);
          if (e >= 0) begin
            automatic int hb = hb_of_edge(e);
            automatic int eb = ecm_bank(e);
            wcnt[hb] = wcnt[hb] + 1;
            rcnt[eb] = rcnt[eb] + 1;
          end
        end
      // (a) <=1 writer per (var-group, m_cm half-bank) — the single m_cm write port per half-bank.
      for (int b = 0; b < NHB; b++)
        if (wcnt[b] > 1) begin
          $display("bp_relay_banked GUARD(a) FAIL: var-group %0d m_cm half-bank %0d has %0d writers (>1)",
                   h, b, wcnt[b]);
          fails = fails + 1;
        end
      // (b) <=2 readers per (var-group, e_cm bank) — the two async read ports per e_cm bank.
      for (int b = 0; b < NEB; b++)
        if (rcnt[b] > 2) begin
          $display("bp_relay_banked GUARD(b) FAIL: var-group %0d e_cm bank %0d has %0d readers (>2)",
                   h, b, rcnt[b]);
          fails = fails + 1;
        end
    end
    // (c) BP_EDGE_POS[e] is e's position in its check's CSR row (edge_at / hb_of_edge tap correctness).
    for (int e = 0; e < BP_E; e++) begin
      automatic int c   = BP_EDGE_CHK[e];
      automatic int idx = BP_CHECK_OFF[c] + BP_EDGE_POS[e];
      if (idx >= BP_CHECK_OFF[c + 1] || BP_CHECK_EDGES[idx] != e) begin
        $display("bp_relay_banked GUARD(c) FAIL: edge %0d (check %0d) EDGE_POS=%0d does not match CSR row",
                 e, c, BP_EDGE_POS[e]);
        fails = fails + 1;
      end
    end
    if (fails != 0)
      $fatal(1, "bp_relay_banked: %0d elaboration-guard violation(s) — header/emitter split is unsafe", fails);
    else
      $display("bp_relay_banked: elaboration guards (a/b/c) PASS (GV=%0d NHB=%0d NEB=%0d BP_E=%0d)",
               GV, NHB, NEB, BP_E);
  end

  // =============================================================================== FSM state / registers
  typedef enum logic [2:0] {
    S_IDLE, S_INIT, S_CHECK, S_VAR, S_SATF, S_EMIT, S_DONE
  } state_t;
  state_t state;

  (* dont_touch = "true" *) logic s_reg  [BP_C];
  (* dont_touch = "true" *) logic ehat   [BP_N];
  (* dont_touch = "true" *) logic best_e [BP_N];
  logic [WW-1:0]     ehat_w, best_w;
  logic              found, all_sat, sat_pending;
  logic [BP_OBS-1:0] obs_acc;

  int          leg, iter, pc;                        // pc = phase/group cursor
  int          wg;                                   // write group (comb): pc in S_INIT, pc-2 in S_VAR
  logic [31:0] lat;

  assign busy           = (state != S_IDLE);
  assign latency_cycles = lat;

  // submodule launch enables (only while groups remain to start)
  logic en_chk, en_var;
  assign en_chk = (state == S_CHECK) && (pc < GC);
  assign en_var = (state == S_VAR)   && (pc < GV);

  // registered submodule outputs (2 clocks after their group launch)
  logic signed [MSG_BITS-1:0] chk_e_out    [W][BP_CHK_DEG];
  logic signed [MSG_BITS-1:0] var_m_out    [V][BP_VAR_DEG];
  logic                       var_ehat_out [V];

  // bank async-read output wires (tapped by CONSTANT bank id in the gathers) + shared read addresses
  logic signed [MSG_BITS-1:0] qmcm   [NHB];
  logic signed [MSG_BITS-1:0] qa_ecm [NEB];
  logic signed [MSG_BITS-1:0] qb_ecm [NEB];
  logic signed [MSG_BITS-1:0] qmvm   [NVB];
  logic [BWC-1:0]             mcm_ra;                 // uniform m_cm read row (= pc, clamped)
  logic [BWV-1:0]             mvm_ra;                 // uniform m_vm read row (= pc, clamped)
  logic [BWC-1:0]             ra_ecm [NEB];           // per-bank e_cm port-A read row
  logic [BWC-1:0]             rb_ecm [NEB];           // per-bank e_cm port-B read row

  // m_cm write drivers (one shared always_comb; each half-bank has <=1 writer per write-group)
  logic                       we_mcm [NHB];
  logic [BWC-1:0]             wa_mcm [NHB];
  logic signed [MSG_BITS-1:0] wd_mcm [NHB];

  // ------------------------------------------------------------------- shared comb: cursors / addresses
  always_comb begin
    wg     = (state == S_INIT) ? pc : (pc - 2);
    mcm_ra = (pc >= 0 && pc < GC) ? BWC'(pc) : '0;    // clamp: out-of-phase reads are unused
    mvm_ra = (pc >= 0 && pc < GV) ? BWV'(pc) : '0;
  end

  // ------------------------------------------------------------------- shared comb: m_cm write scatter
  // VAR (or S_INIT) scatters group `wg`: for each present edge of each var slot, drive its half-bank's
  // single write port. Row = the edge's CHECK group; data = the var-update output (or lambda in S_INIT).
  always_comb begin
    for (int b = 0; b < NHB; b++) begin
      we_mcm[b] = 1'b0;
      wa_mcm[b] = '0;
      wd_mcm[b] = '0;
    end
    if (state == S_INIT || state == S_VAR) begin
      for (int h = 0; h < GV; h++)
        if (wg == h) begin
          for (int i = 0; i < V; i++)
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(h, i, d);
              if (e >= 0) begin
                automatic int hb = hb_of_edge(e);
                we_mcm[hb] = 1'b1;
                wa_mcm[hb] = BWC'(grp_of_chk(BP_EDGE_CHK[e]));
                if (state == S_INIT) wd_mcm[hb] = signed'(BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0]);
                else                 wd_mcm[hb] = var_m_out[i][d];
              end
            end
        end
    end
  end

  // ------------------------------------------------------------------- shared comb: e_cm read addresses
  // VAR launch group `pc`: for each present edge operand, route its bank's port-A/B read row.
  always_comb begin
    for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = '0;
      rb_ecm[b] = '0;
    end
    if (state == S_VAR) begin
      for (int h = 0; h < GV; h++)
        if (pc == h) begin
          for (int i = 0; i < V; i++)
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(h, i, d);
              if (e >= 0) begin
                automatic int bank = ecm_bank(e);
                automatic int row  = grp_of_chk(BP_EDGE_CHK[e]);
                if (ecm_port(h, i, d) == 0) ra_ecm[bank] = BWC'(row);
                else                        rb_ecm[bank] = BWC'(row);
              end
            end
        end
    end
  end

  // ===================================================================== m_cm half-banks (LUTRAM idiom)
  generate
    for (genvar b = 0; b < NHB; b++) begin : gmcm
      logic signed [MSG_BITS-1:0] mem [GC];
      always_ff @(posedge clk) if (we_mcm[b]) mem[wa_mcm[b]] <= wd_mcm[b];
      assign qmcm[b] = mem[mcm_ra];
    end
  endgenerate

  // ===================================================================== e_cm banks (LUTRAM, 2 read ports)
  generate
    for (genvar b = 0; b < NEB; b++) begin : gecm
      localparam int JJ = b / BP_CHK_DEG;             // check slot j
      localparam int KK = b % BP_CHK_DEG;             // lane / position k
      logic signed [MSG_BITS-1:0] mem [GC];
      logic                       we_b;
      logic [BWC-1:0]             wa_b;
      logic signed [MSG_BITS-1:0] wd_b;
      // CHK scatter: lane (JJ,KK) of group pc-2 writes its own bank (single writer, no mux).
      always_comb begin
        we_b = 1'b0;
        wa_b = '0;
        wd_b = chk_e_out[JJ][KK];
        if (state == S_CHECK && pc >= 2)
          for (int g = 0; g < GC; g++)
            if ((pc - 2) == g && edge_at(g, JJ, KK) >= 0) begin
              we_b = 1'b1;
              wa_b = BWC'(g);
            end
      end
      always_ff @(posedge clk) if (we_b) mem[wa_b] <= wd_b;
      assign qa_ecm[b] = mem[ra_ecm[b]];
      assign qb_ecm[b] = mem[rb_ecm[b]];
    end
  endgenerate

  // ===================================================================== m_vm banks (LUTRAM idiom)
  generate
    for (genvar b = 0; b < NVB; b++) begin : gmvm
      localparam int II = b / BP_VAR_DEG;             // var slot i
      localparam int DD = b % BP_VAR_DEG;             // edge d
      logic signed [MSG_BITS-1:0] mem [GV];
      logic                       we_b;
      logic [BWV-1:0]             wa_b;
      logic signed [MSG_BITS-1:0] wd_b;
      // written by its own var slot (mux-free): row = write group wg; data = var-update output (or lambda).
      always_comb begin
        we_b = 1'b0;
        wa_b = '0;
        wd_b = var_m_out[II][DD];
        if (state == S_INIT || state == S_VAR)
          for (int h = 0; h < GV; h++)
            if (wg == h && vedge_at(h, II, DD) >= 0) begin
              we_b = 1'b1;
              wa_b = BWV'(h);
              if (state == S_INIT) wd_b = signed'(BP_LAMBDA[var_at(h, II)][MSG_BITS-1:0]);
            end
      end
      always_ff @(posedge clk) if (we_b) mem[wa_b] <= wd_b;
      assign qmvm[b] = mem[mvm_ra];
    end
  endgenerate

  // ===================================================================== W check_minsum slots
  generate
    for (genvar j = 0; j < W; j++) begin : gchk
      logic                       sbit_j;
      logic signed [MSG_BITS-1:0] m_in_j    [BP_CHK_DEG];
      logic                       present_j [BP_CHK_DEG];
      // gather group `pc`'s check for slot j from m_cm at CONSTANT half-bank taps (beta folds to constant).
      always_comb begin
        sbit_j = 1'b0;
        for (int k = 0; k < BP_CHK_DEG; k++) begin
          m_in_j[k]    = '0;
          present_j[k] = 1'b0;
        end
        for (int g = 0; g < GC; g++)
          if (chk_at(g, j) >= 0 && pc == g) begin
            sbit_j = s_reg[chk_at(g, j)];
            for (int k = 0; k < BP_CHK_DEG; k++) begin
              automatic int e = edge_at(g, j, k);
              if (e >= 0) begin
                m_in_j[k]    = qmcm[hb_of_edge(e)];   // hb_of_edge(e) is a compile-time constant here
                present_j[k] = 1'b1;
              end
            end
          end
      end
      check_minsum #(
          .MW (MSG_BITS),
          .DEG(BP_CHK_DEG)
      ) u_chk (
          .clk    (clk),
          .en     (en_chk),
          .sbit   (sbit_j),
          .m_in   (m_in_j),
          .present(present_j),
          .e_out  (chk_e_out[j])
      );
    end
  endgenerate

  // ===================================================================== V var_update slots
  generate
    for (genvar i = 0; i < V; i++) begin : gvar
      logic signed [MSG_BITS-1:0] lam_i, gam_i;
      logic signed [MSG_BITS-1:0] e_in_i    [BP_VAR_DEG];
      logic signed [MSG_BITS-1:0] m_in_i    [BP_VAR_DEG];
      logic                       present_i [BP_VAR_DEG];
      // gather group `pc`'s var for slot i: e_cm operands (port-selected) + the "old" m_vc from m_vm.
      always_comb begin
        lam_i = '0;
        gam_i = '0;
        for (int d = 0; d < BP_VAR_DEG; d++) begin
          e_in_i[d]    = '0;
          m_in_i[d]    = '0;
          present_i[d] = 1'b0;
        end
        for (int g = 0; g < GV; g++)
          if (var_at(g, i) >= 0 && pc == g) begin
            automatic int v = var_at(g, i);
            lam_i = signed'(BP_LAMBDA[v][MSG_BITS-1:0]);
            gam_i = signed'(BP_GAMMA[leg * BP_N + v][MSG_BITS-1:0]);
            for (int d = 0; d < BP_VAR_DEG; d++) begin
              automatic int e = vedge_at(g, i, d);
              if (e >= 0) begin
                automatic int bank = ecm_bank(e);
                e_in_i[d]    = (ecm_port(g, i, d) == 1) ? qb_ecm[bank] : qa_ecm[bank];
                m_in_i[d]    = qmvm[i * BP_VAR_DEG + d];
                present_i[d] = 1'b1;
              end
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

  // ===================================================================================== control FSM
  always_ff @(posedge clk) begin                     // synchronous reset (Synth 8-7137)
    if (!rst_n) begin
      state      <= S_IDLE;
      out_valid  <= 1'b0;
      valid_flag <= 1'b0;
      lat        <= '0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        // ----------------------------------------------------------------- accept syndrome + init flops
        S_IDLE: begin
          if (in_valid) begin
            for (int c = 0; c < BP_C; c++) s_reg[c] <= syndrome_in[c];
            for (int v = 0; v < BP_N; v++) ehat[v]  <= 1'b0;
            found       <= 1'b0;
            best_w      <= '1;
            ehat_w      <= '0;
            all_sat     <= 1'b1;
            sat_pending <= 1'b0;                       // no decision to SAT before the first S_VAR
            leg <= '0; iter <= '0; pc <= '0;
            lat <= '0;
            state <= S_INIT;                           // banked messages must be lambda-seeded a group/cyc
          end
        end

        // ----------------------------------- seed m_cm/m_vm with lambda, one var-group/cycle (direct)
        S_INIT: begin
          if (pc == GV - 1) begin pc <= '0; state <= S_CHECK; end
          else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch check group `pc` + scatter `pc-2`  ||  overlapped S_SAT
        S_CHECK: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          // overlapped SAT: parity of the PREVIOUS decision (ehat) on the LAUNCHED group's checks
          if (pc < GC && sat_pending) begin
            for (int j = 0; j < W; j++)
              for (int g = 0; g < GC; g++)
                if (chk_at(g, j) >= 0 && pc == g) begin
                  p = s_reg[chk_at(g, j)];
                  for (int k = 0; k < BP_CHK_DEG; k++) begin
                    automatic int e = edge_at(g, j, k);
                    if (e >= 0) p = p ^ ehat[BP_EDGE_VAR[e]];
                  end
                  if (p != 1'b0) grp_sat = 1'b0;
                end
            if (!grp_sat) all_sat <= 1'b0;
            if (pc == GC - 1) begin
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
          // e_cm scatter of group pc-2 handled by the per-bank gecm write comb.
          // advance cursor; early_exit takes the first syndrome-valid decision straight to S_EMIT.
          if (early_exit && final_sat) begin
            pc <= '0; state <= S_EMIT;
          end else if (pc == GC + 1) begin
            pc      <= '0;
            all_sat <= 1'b1;
            state   <= S_VAR;
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ------------------------------ launch var group `pc` + scatter `pc-2`
        S_VAR: begin
          automatic int wsum;
          wsum = 0;
          if (pc == 0) ehat_w <= '0;                  // fresh decision-weight accumulation
          if (pc >= 2) begin
            for (int i = 0; i < V; i++)
              for (int g = 0; g < GV; g++)
                if (var_at(g, i) >= 0 && (pc - 2) == g) begin
                  automatic int v = var_at(g, i);
                  ehat[v] <= var_ehat_out[i];
                  wsum = wsum + (var_ehat_out[i] ? 1 : 0);
                end
            ehat_w <= ehat_w + WW'(wsum);
          end
          // m_vm / m_cm writes of group pc-2 handled by the gmvm write comb + m_cm scatter comb.
          if (pc == GV + 1) begin
            pc          <= '0;
            sat_pending <= 1'b1;
            if (iter == BP_ITERS - 1) begin
              iter <= '0;
              if (leg == BP_LEGS - 1) state <= S_SATF;
              else begin leg <= leg + 1; state <= S_CHECK; end
            end else begin
              iter <= iter + 1;
              state <= S_CHECK;
            end
          end else pc <= pc + 1;
          lat <= lat + 32'd1;
        end

        // ----------------------------- trailing SAT for the final decision (no following S_CHECK)
        S_SATF: begin
          automatic logic grp_sat, final_sat, p;
          grp_sat   = 1'b1;
          final_sat = 1'b0;
          p         = 1'b0;
          for (int j = 0; j < W; j++)
            for (int g = 0; g < GC; g++)
              if (chk_at(g, j) >= 0 && pc == g) begin
                p = s_reg[chk_at(g, j)];
                for (int k = 0; k < BP_CHK_DEG; k++) begin
                  automatic int e = edge_at(g, j, k);
                  if (e >= 0) p = p ^ ehat[BP_EDGE_VAR[e]];
                end
                if (p != 1'b0) grp_sat = 1'b0;
              end
          if (!grp_sat) all_sat <= 1'b0;
          if (pc == GC - 1) begin
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
          lat <= lat + 32'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat -> obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;
          automatic logic              bb;
          acc = (pc == 0) ? {BP_OBS{1'b0}} : obs_acc;
          bb  = 1'b0;
          for (int i = 0; i < V; i++)
            for (int g = 0; g < GV; g++)
              if (var_at(g, i) >= 0 && pc == g) begin
                automatic int v = var_at(g, i);
                bb = found ? best_e[v] : ehat[v];
                corr_out[v] <= bb;
                if (bb) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
              end
          obs_acc <= acc;
          if (pc == GV - 1) begin pc <= '0; state <= S_DONE; end
          else pc <= pc + 1;
          lat <= lat + 32'd1;
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
