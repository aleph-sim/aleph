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
  // Q6-14: spanning-forest edges processed per cycle. The union loop is unrolled FOREST_UNROLL times
  // with strictly sequential semantics (each sub-union sees the previous one's relabel), so the set
  // and order of `istree[]` edges is IDENTICAL to the one-edge-per-cycle form — only the cycle count
  // drops from ~M to ~ceil(M/FOREST_UNROLL). Cost: the per-cycle combinational path is ~UNROLL× the
  // single-edge find+relabel depth. 2 is the measured sweet spot: Q6-14 picked 3 when the binding path
  // was S_CC_RELAX (so a deeper forest was "free" up to that cap), but Q6-16's CC min-tree removed
  // that cap and EXPOSED the forest relabel as the new critical path — so the shallower 2 now wins on
  // Fmax (Zybo d5 58->79 MHz) by more than its +cycles cost, at d=5 AND d=7. Re-swept K∈{2,3,4} ×
  // {d5,d7} × both parts; 2 is best on 3 of 4 cells (only d7/KV260 marginally prefers a deeper K).
  // See docs/perf/qec-q6-fpga.md §Q6-18.
  localparam int FOREST_UNROLL = 2;

  // Q6-15: S_CC_INIT folded away — the label-to-identity reset is now done in the predecessor states
  // (S_IDLE on accept, S_GROW after a grow round), which already write registers, so it costs no
  // extra cycle. Saves one clock per growth round (G+1 total). Bit-identical: S_CC_RELAX still reads
  // label=identity + the current support on entry, exactly as it did via the dedicated init state.
  typedef enum logic [3:0] {
    S_IDLE, S_CC_RELAX, S_GROW,
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
  logic [UF_M-1:0]   corr;
  logic [IDXW-1:0]   grow_cnt;        // growth rounds done (caps at UF_N, as in Q6-02)
  logic [EIDXW-1:0]  e_idx;           // forest edge cursor
  logic [IDXW-1:0]   pit;             // peel sweep counter
  logic [15:0]       lat;             // cycles since in_valid accepted

  // Q6-16: balanced pairwise min-reduction over UF_N label candidates. Used by S_CC_RELAX to compute
  // each node's relaxed label as min(self, fused-neighbour labels). The Q6-04 form folded the min
  // SERIALLY across incident edges (`if (label[nbr] < nl[v]) nl[v] = label[nbr]`), which synthesises
  // to a chain of depth = node degree — and the boundary super-node's high degree made that the
  // binding d=5 critical path (25 LUT levels, `label_reg -> FSM_state`). min is associative +
  // commutative, so reducing the same candidate set as an explicit O(log N) tree is bit-identical but
  // shrinks the path to ~log2(N) levels. (Same scatter->tree idea as the Q6-11 parity gather.)
  function automatic logic [IDXW-1:0] cc_min_reduce(input logic [IDXW-1:0] a [UF_N]);
    logic [IDXW-1:0] r [UF_N];
    int n;
    for (int i = 0; i < UF_N; i++) r[i] = a[i];
    n = UF_N;
    while (n > 1) begin
      for (int i = 0; i < n / 2; i++)
        r[i] = (r[2*i + 1] < r[2*i]) ? r[2*i + 1] : r[2*i];
      if (n[0]) r[n/2] = r[n - 1];   // odd count: carry the unpaired element up a level
      n = (n + 1) / 2;
    end
    return r[0];
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= S_IDLE;
      out_valid      <= 1'b0;
      busy           <= 1'b0;
      correction     <= '0;
      obs_flip       <= 1'b0;
      latency_cycles <= '0;
      lat            <= '0;
      grow_cnt       <= '0;
      e_idx          <= '0;
      pit            <= '0;
      corr           <= '0;
      for (int i = 0; i < UF_N; i++) begin
        defect[i] <= 1'b0; dfct[i] <= 1'b0; label[i] <= '0; troot[i] <= '0;
        deg[i] <= '0;
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
            for (int i = 0; i < UF_N; i++) label[i] <= IDXW'(i);  // Q6-15: folded CC_INIT
            grow_cnt <= '0;
            lat      <= 16'd1;
            busy     <= 1'b1;
            state    <= S_CC_RELAX;
          end
        end

        // ---- GROWTH: connected components (label reset is folded into S_IDLE/S_GROW), grow odd ----
        // one parallel (Jacobi) min-relaxation pass over fused edges; iterate until stable.
        S_CC_RELAX: begin
          automatic logic [IDXW-1:0] nl [UF_N];
          automatic logic            changed;
          // Q6-16: per-node GATHER then balanced min-reduce (see cc_min_reduce). For each node v,
          // build the candidate set { label[v] } ∪ { label[u] : fused edge v–u }, with non-neighbours
          // masked to the all-ones sentinel (>= every real label, since UF_N-1 <= 2^IDXW-1), then
          // tree-reduce. Bit-identical to the old serial min-fold (same set, min is assoc/comm) but
          // the per-node path is O(log N) deep instead of O(degree).
          for (int v = 0; v < UF_N; v++) begin
            automatic logic [IDXW-1:0] cand [UF_N];
            for (int u = 0; u < UF_N; u++) cand[u] = '1;   // sentinel = +inf
            cand[v] = label[v];                            // self
            for (int e = 0; e < UF_M; e++)
              if (support[e] == 2'd2) begin
                if (UF_EA[e] == v) cand[UF_EB[e]] = label[UF_EB[e]];
                if (UF_EB[e] == v) cand[UF_EA[e]] = label[UF_EA[e]];
              end
            nl[v] = cc_min_reduce(cand);
          end
          changed = 1'b0;
          for (int i = 0; i < UF_N; i++) begin
            if (nl[i] != label[i]) changed = 1'b1;
            label[i] <= nl[i];
          end
          if (!changed) state <= S_GROW;
        end

        // Q6-15: fused ODD+GROW. The odd-cluster flags are a pure combinational function of label +
        // defect (the Q6-11 parity GATHER), so there is no need to register them and spend a separate
        // S_ODD cycle — compute them here and grow in the same cycle. Saves one clock per growth round.
        // Bit-identical: `oddc_c` is exactly what S_ODD wrote to the `oddc` register, the grow update
        // and the anyodd/cap test are unchanged. The parity gather (O(log N) tree) + the shallow grow
        // adder stay below the binding S_CC_RELAX critical path.
        S_GROW: begin
          automatic logic par    [UF_N];
          automatic logic hasb   [UF_N];
          automatic logic oddc_c [UF_N];
          automatic logic any;
          for (int v = 0; v < UF_N; v++) begin
            automatic logic p;
            p = 1'b0;
            for (int i = 0; i < UF_N; i++)
              if (label[i] == IDXW'(v)) p = p ^ defect[i];
            par[v]  = p;
            hasb[v] = (label[UF_BOUNDARY] == IDXW'(v));
          end
          any = 1'b0;
          for (int i = 0; i < UF_N; i++) begin
            oddc_c[i] = par[label[i]] & ~hasb[label[i]];
            any = any | oddc_c[i];
          end
          if (any && grow_cnt < IDXW'(UF_N)) begin
            for (int e = 0; e < UF_M; e++) begin
              automatic logic [2:0] s;
              s = {1'b0, support[e]} + {2'b0, oddc_c[UF_EA[e]]} + {2'b0, oddc_c[UF_EB[e]]};
              support[e] <= (s > 3'd2) ? 2'd2 : s[1:0];
            end
            // Q6-15: do NOT reset labels here — keep this round's converged labels as the next
            // round's CC_RELAX seed. Growth only MERGES components, and the min-label Jacobi fixpoint
            // is unique (every node starts >= its new component min, and that min node keeps its
            // self-label), so the converged result is identical — but starting from the prior labels
            // only the newly-merged frontier has to propagate, so CC_RELAX converges in a couple of
            // passes instead of re-running the full component diameter from identity each round.
            grow_cnt <= grow_cnt + 1'b1;
            state    <= S_CC_RELAX;
          end else begin
            for (int i = 0; i < UF_N; i++) troot[i] <= IDXW'(i);
            for (int e = 0; e < UF_M; e++) istree[e] <= 1'b0;
            e_idx <= '0;
            state <= S_FOREST;
          end
        end

        // ---- SPANNING FOREST: one fused edge per cycle, QUICK-FIND (flat forest) ----
        // Q6-10: the Q6-02/Q6-04 form did a lazy quick-union — find walked `troot` N times per
        // endpoint (`for k in 0..N: ra = troot[ra]`), an N-deep serial chain of index-muxed troot
        // lookups. That chain was the binding d>=5 critical path (78 LUT levels @ d=5, the #373
        // finding) and the reason Fmax collapsed with distance. Switch to QUICK-FIND: maintain the
        // invariant that `troot[x]` is ALWAYS the direct component root (flat forest). Then find is
        // a single depth-1 read, and union eagerly relabels every member of the absorbed root in
        // parallel — N independent compare-muxes, no serial dependency. This is BIT-IDENTICAL to the
        // lazy form: edges are still processed in index order, the union test is still ra!=rb, and
        // the surviving root is still rb, so `istree[]` is unchanged (d=3 stays golden-bit-exact;
        // d=5 weight-<=2 quality identical). Only the per-cycle depth drops from O(N) to O(log N)
        // (the troot/UF_EA index muxes), which is what frees Fmax at d>=5.
        // Q6-14: process FOREST_UNROLL edges per cycle. `wt` is a working copy of `troot` mutated
        // with BLOCKING writes, so sub-union k sees the relabel from sub-union k-1 — exactly the
        // sequential quick-find of the one-edge form, just folded into one cycle. Same edges become
        // `istree[]`, in the same index order => d=3 golden + d=5 0/1431 both bit-preserved.
        S_FOREST: begin
          automatic logic [IDXW-1:0] wt [UF_N];   // working troot for this cycle's sequential unions
          for (int i = 0; i < UF_N; i++) wt[i] = troot[i];
          for (int k = 0; k < FOREST_UNROLL; k++) begin
            automatic int unsigned ei;
            ei = UF_M;                             // sentinel: out-of-range => skip
            if (int'(e_idx) + k < UF_M) ei = int'(e_idx) + k;
            if (ei < UF_M && support[ei] == 2'd2) begin
              automatic logic [IDXW-1:0] ra, rb;
              ra = wt[UF_EA[ei]];                  // depth-1 find on the in-cycle working forest
              rb = wt[UF_EB[ei]];
              if (ra != rb) begin
                for (int i = 0; i < UF_N; i++)
                  if (wt[i] == ra) wt[i] = rb;     // blocking: visible to the next sub-union
                istree[ei] <= 1'b1;
              end
            end
          end
          for (int i = 0; i < UF_N; i++) troot[i] <= wt[i];
          if (int'(e_idx) + FOREST_UNROLL >= UF_M) state <= S_PEEL_INIT;
          else                                     e_idx <= e_idx + EIDXW'(FOREST_UNROLL);
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

        // one PARALLEL leaf-strip round: peel ALL current non-boundary leaves at once. The correction
        // for a tree edge is the defect-parity of its leaf-side subtree, which is independent of peel
        // order, so this is bit-equivalent to the Q6-02 loop-carried sweep (golden-locked) — but each
        // round is bounded-depth: the per-node updates are associative count/XOR reductions over
        // incident edges (Vivado balances them into O(log M) trees), with no M-long dependency chain.
        // That removes the O(M) critical path that collapsed Fmax with distance (Q6-09).
        S_PEEL_PASS: begin
          automatic logic            peel [UF_M];   // edge peels this round
          automatic logic [IDXW-1:0] lf   [UF_M];   // its leaf endpoint
          automatic logic [IDXW-1:0] nb   [UF_M];   // its neighbour (non-leaf) endpoint
          automatic logic            lfd  [UF_M];   // defect routed off the leaf onto the neighbour
          automatic logic            anypeel;       // any leaf stripped this round?
          // 1. classify each tree edge — all reads from registers, no inter-edge dependency.
          for (int e = 0; e < UF_M; e++) begin
            peel[e] = 1'b0; lf[e] = '0; nb[e] = '0; lfd[e] = 1'b0;
            if (istree[e]) begin
              if (deg[UF_EA[e]] == 5'd1 && UF_EA[e] != UF_BOUNDARY) begin
                peel[e] = 1'b1; lf[e] = IDXW'(UF_EA[e]); nb[e] = IDXW'(UF_EB[e]);
              end else if (deg[UF_EB[e]] == 5'd1 && UF_EB[e] != UF_BOUNDARY) begin
                peel[e] = 1'b1; lf[e] = IDXW'(UF_EB[e]); nb[e] = IDXW'(UF_EA[e]);
              end
              lfd[e] = peel[e] & dfct[lf[e]];
            end
          end
          // 2. correction + tree removal — per-edge, independent.
          for (int e = 0; e < UF_M; e++) begin
            if (lfd[e]) corr[e] <= 1'b1;
            istree[e] <= istree[e] & ~peel[e];
          end
          // 3. per-node update via associative reductions over incident peeling edges.
          for (int vv = 0; vv < UF_N; vv++) begin
            automatic logic [4:0] cnt;     // # peeling edges whose neighbour is vv
            automatic logic       tog;     // XOR of defects routed into vv
            automatic logic       vleaf;   // vv was peeled as a leaf this round
            cnt = 5'd0; tog = 1'b0; vleaf = 1'b0;
            for (int e = 0; e < UF_M; e++)
              if (peel[e]) begin
                if (nb[e] == IDXW'(vv)) begin cnt = cnt + 5'd1; tog = tog ^ lfd[e]; end
                if (lf[e] == IDXW'(vv)) vleaf = 1'b1;
              end
            deg[vv]  <= vleaf ? 5'd0 : (deg[vv] - cnt);
            dfct[vv] <= dfct[vv] ^ tog;
          end
          // Q6-13: early terminate. S_PEEL_PASS is scheduled for a fixed UF_N rounds (worst-case
          // tree depth), but a round that strips no leaf makes ZERO register changes (every peel[e]=0
          // => lfd/cnt/tog/vleaf all 0 => corr/deg/dfct/istree write back their current values), and
          // since nothing changed the next round would also peel nothing. So once a round is empty the
          // remaining scheduled rounds are provably no-ops: jump to S_FINISH. This is BIT-IDENTICAL to
          // running all UF_N rounds (golden d=3 + d=5 weight-<=2 quality unchanged); it just cuts the
          // peel cost from a fixed N to (actual max tree depth + 1), typically << N at d>=5.
          anypeel = 1'b0;
          for (int e = 0; e < UF_M; e++) anypeel = anypeel | peel[e];
          if (!anypeel || pit == IDXW'(UF_N - 1)) state <= S_FINISH;
          else                                    pit   <= pit + 1'b1;
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
