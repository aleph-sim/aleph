// Q6-04 — surface-code Union-Find decoder, SYNTHESIZABLE SEQUENTIAL FORM.
//
// Same Delfosse-Nickerson algorithm as the Q6-02 combinational draft (growth -> spanning forest ->
// peeling on the fixed d=3 matching graph in `uf_surface_graph.svh`), but time-multiplexed into a
// clocked FSM so the synthesised critical path is bounded. The Q6-02 version did the entire decode
// inside one `always_comb` — a triple-nested fixpoint (gi x p x e) unrolled into a single giant
// combinational cloud that cannot close timing on real silicon. Here every cycle performs exactly
// ONE bounded pass (over the M edges or N nodes); the outer loops (growth rounds, CC relaxation,
// peel sweeps) advance in *time* across FSM states. Per-cycle combinational depth is now O(M),
// independent of the iteration counts.
//
// Equivalence: the support fixpoint, spanning forest, and peel are computed by the identical update
// rules in the identical order, so the result is bit-for-bit the Q6-02 output. The CC relabel uses a
// parallel min-relaxation (Jacobi) iterated to a fixpoint, which yields the same connected-component
// min-labels the combinational version produced. Locked by a 256-row golden table (`tb_uf_surface`
// compares every syndrome against `uf_surface_golden.mem`, snapshotted from the Q6-02 RTL).
//
// Handshake: pulse `in_valid` with `syndrome`; the core asserts `busy`, runs for `latency_cycles`
// clocks, then pulses `out_valid` for one cycle with `correction`/`obs_flip` valid.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_surface_decoder (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    input  logic [UF_N-2:0]   syndrome,    // detectors 0..N-2 (boundary node N-1 has no detector)
    output logic              busy,
    output logic              out_valid,
    output logic [UF_M-1:0]   correction,
    output logic              obs_flip,
    output logic [15:0]       latency_cycles
);
  localparam int IDXW  = $clog2(UF_N);       // node-index width (N=9 -> 4)
  localparam int EIDXW = $clog2(UF_M + 1);   // edge-counter width (M=18 -> 5)

  typedef enum logic [3:0] {
    S_IDLE, S_CC_INIT, S_CC_RELAX, S_ODD, S_GROW,
    S_FOREST, S_PEEL_INIT, S_PEEL_PASS, S_FINISH
  } state_t;

  state_t            state;

  // `u`/`v` peel cursors are `int` working vars; only their low (node-index) bits are used.
  /* verilator lint_off UNUSEDSIGNAL */

  // working state (all registered; one bounded pass mutates it per cycle)
  logic              defect [UF_N];   // latched syndrome (boundary node = 0)
  logic              dfct   [UF_N];   // peel working defect copy
  logic [1:0]        support[UF_M];   // 0,1,2 ; 2 = fused
  logic [IDXW-1:0]   label  [UF_N];   // connected-component label (min node id)
  logic [IDXW-1:0]   troot  [UF_N];   // spanning-tree union-find parent
  logic              istree [UF_M];   // edge is in the spanning forest
  logic [4:0]        deg    [UF_N];   // node degree in the forest
  logic              oddc   [UF_N];   // per-node: in an odd (non-neutral) cluster
  logic              anyodd;
  logic [UF_M-1:0]   corr;
  logic [IDXW-1:0]   grow_cnt;        // growth rounds done (caps at UF_N, as in Q6-02)
  logic [EIDXW-1:0]  e_idx;           // forest edge cursor
  logic [IDXW-1:0]   pit;             // peel sweep counter
  logic [15:0]       lat;             // cycles since in_valid accepted

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= S_IDLE;
      out_valid      <= 1'b0;
      busy           <= 1'b0;
      correction     <= '0;
      obs_flip       <= 1'b0;
      latency_cycles <= '0;
      lat            <= '0;
      anyodd         <= 1'b0;
      grow_cnt       <= '0;
      e_idx          <= '0;
      pit            <= '0;
      corr           <= '0;
      for (int i = 0; i < UF_N; i++) begin
        defect[i] <= 1'b0; dfct[i] <= 1'b0; label[i] <= '0; troot[i] <= '0;
        deg[i] <= '0; oddc[i] <= 1'b0;
      end
      for (int e = 0; e < UF_M; e++) begin support[e] <= 2'd0; istree[e] <= 1'b0; end
    end else begin
      out_valid <= 1'b0;                 // default: out_valid is a 1-cycle pulse
      if (state != S_IDLE) lat <= lat + 16'd1;

      unique case (state)
        // ---- accept a syndrome ----
        S_IDLE: begin
          if (in_valid) begin
            for (int i = 0; i < UF_N; i++)
              defect[i] <= (i < UF_N - 1) ? syndrome[i] : 1'b0;
            for (int e = 0; e < UF_M; e++) support[e] <= 2'd0;
            grow_cnt <= '0;
            lat      <= 16'd1;
            busy     <= 1'b1;
            state    <= S_CC_INIT;
          end
        end

        // ---- GROWTH: recompute connected components, then grow odd clusters ----
        S_CC_INIT: begin
          for (int i = 0; i < UF_N; i++) label[i] <= IDXW'(i);
          state <= S_CC_RELAX;
        end

        // one parallel (Jacobi) min-relaxation pass over fused edges; iterate until stable.
        S_CC_RELAX: begin
          automatic logic [IDXW-1:0] nl [UF_N];
          automatic logic            changed;
          for (int i = 0; i < UF_N; i++) nl[i] = label[i];
          for (int e = 0; e < UF_M; e++)
            if (support[e] == 2'd2) begin
              if (label[UF_EB[e]] < nl[UF_EA[e]]) nl[UF_EA[e]] = label[UF_EB[e]];
              if (label[UF_EA[e]] < nl[UF_EB[e]]) nl[UF_EB[e]] = label[UF_EA[e]];
            end
          changed = 1'b0;
          for (int i = 0; i < UF_N; i++) begin
            if (nl[i] != label[i]) changed = 1'b1;
            label[i] <= nl[i];
          end
          if (!changed) state <= S_ODD;
        end

        // per-cluster defect parity + boundary flag -> odd (non-neutral) nodes.
        S_ODD: begin
          automatic logic par  [UF_N];
          automatic logic hasb [UF_N];
          automatic logic any;
          for (int v = 0; v < UF_N; v++) begin par[v] = 1'b0; hasb[v] = 1'b0; end
          for (int i = 0; i < UF_N; i++) par[label[i]] = par[label[i]] ^ defect[i];
          hasb[label[UF_BOUNDARY]] = 1'b1;
          any = 1'b0;
          for (int i = 0; i < UF_N; i++) begin
            oddc[i] <= par[label[i]] & ~hasb[label[i]];
            any = any | (par[label[i]] & ~hasb[label[i]]);
          end
          anyodd <= any;
          state  <= S_GROW;
        end

        // grow incident edges of odd clusters by one half-step (cap support at 2); else enter forest.
        S_GROW: begin
          if (anyodd && grow_cnt < IDXW'(UF_N)) begin
            for (int e = 0; e < UF_M; e++) begin
              automatic logic [2:0] s;
              s = {1'b0, support[e]} + {2'b0, oddc[UF_EA[e]]} + {2'b0, oddc[UF_EB[e]]};
              support[e] <= (s > 3'd2) ? 2'd2 : s[1:0];
            end
            grow_cnt <= grow_cnt + 1'b1;
            state    <= S_CC_INIT;
          end else begin
            for (int i = 0; i < UF_N; i++) troot[i] <= IDXW'(i);
            for (int e = 0; e < UF_M; e++) istree[e] <= 1'b0;
            e_idx <= '0;
            state <= S_FOREST;
          end
        end

        // ---- SPANNING FOREST: one fused edge per cycle (root-walk find + union) ----
        S_FOREST: begin
          if (support[e_idx] == 2'd2) begin
            automatic logic [IDXW-1:0] ra, rb;
            ra = IDXW'(UF_EA[e_idx]);
            for (int k = 0; k < UF_N; k++) ra = troot[ra];
            rb = IDXW'(UF_EB[e_idx]);
            for (int k = 0; k < UF_N; k++) rb = troot[rb];
            if (ra != rb) begin
              troot[ra]      <= rb;
              istree[e_idx]  <= 1'b1;
            end
          end
          if (e_idx == EIDXW'(UF_M - 1)) state <= S_PEEL_INIT;
          else                          e_idx <= e_idx + 1'b1;
        end

        // ---- PEELING: init degrees, then N leaf-strip sweeps ----
        S_PEEL_INIT: begin
          automatic logic [4:0] d [UF_N];
          for (int i = 0; i < UF_N; i++) d[i] = 5'd0;
          for (int e = 0; e < UF_M; e++)
            if (istree[e]) begin d[UF_EA[e]] = d[UF_EA[e]] + 5'd1; d[UF_EB[e]] = d[UF_EB[e]] + 5'd1; end
          for (int i = 0; i < UF_N; i++) begin deg[i] <= d[i]; dfct[i] <= defect[i]; end
          corr  <= '0;
          pit   <= '0;
          state <= S_PEEL_PASS;
        end

        // one full leaf-strip sweep over all edges (loop-carried, identical to the Q6-02 inner body).
        S_PEEL_PASS: begin
          automatic logic [4:0]      d  [UF_N];
          automatic logic            df [UF_N];
          automatic logic            it [UF_M];
          automatic logic [UF_M-1:0] cr;
          for (int i = 0; i < UF_N; i++) begin d[i] = deg[i]; df[i] = dfct[i]; end
          for (int e = 0; e < UF_M; e++) it[e] = istree[e];
          cr = corr;
          for (int e = 0; e < UF_M; e++)
            if (it[e]) begin
              automatic int u, v;
              u = -1; v = -1;
              if (d[UF_EA[e]] == 5'd1 && UF_EA[e] != UF_BOUNDARY) begin u = UF_EA[e]; v = UF_EB[e]; end
              else if (d[UF_EB[e]] == 5'd1 && UF_EB[e] != UF_BOUNDARY) begin u = UF_EB[e]; v = UF_EA[e]; end
              if (u != -1) begin
                if (df[u]) begin cr[e] = 1'b1; df[v] = df[v] ^ 1'b1; end
                it[e] = 1'b0;
                d[u]  = 5'd0;
                d[v]  = d[v] - 5'd1;
              end
            end
          for (int i = 0; i < UF_N; i++) begin deg[i] <= d[i]; dfct[i] <= df[i]; end
          for (int e = 0; e < UF_M; e++) istree[e] <= it[e];
          corr <= cr;
          if (pit == IDXW'(UF_N - 1)) state <= S_FINISH;
          else                        pit   <= pit + 1'b1;
        end

        // ---- logical flip = parity of the logical flag over the correction edges ----
        S_FINISH: begin
          automatic logic ob;
          ob = 1'b0;
          for (int e = 0; e < UF_M; e++) if (corr[e] & UF_ELOG[e]) ob = ob ^ 1'b1;
          correction     <= corr;
          obs_flip       <= ob;
          out_valid      <= 1'b1;
          busy           <= 1'b0;
          latency_cycles <= lat;
          state          <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
