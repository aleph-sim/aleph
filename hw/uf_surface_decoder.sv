// Q6-02 (sim) — surface-code Union-Find decoder (2-D matching graph).
//
// The real Union-Find decoder (Delfosse-Nickerson) on the d=3 rotated surface-code memory-Z matching
// graph (generated into `uf_surface_graph.svh`): N nodes (detectors + one boundary node), M edges.
// Three stages, all on the fixed graph:
//
//   1. GROWTH   — defects (lit detectors) seed odd clusters; each iteration grows every edge incident
//                 to an odd cluster by one half-step; an edge fuses at support 2; clusters are merged
//                 (connected components over fused edges) and a cluster is neutral once it has even
//                 defect parity OR contains the boundary. Repeat until no odd clusters.
//   2. TREE     — a spanning forest of the fused edges (a tiny union-find keeps it acyclic).
//   3. PEEL     — leaf-strip the forest: a leaf that is a defect adds its edge to the correction and
//                 toggles its neighbour's defect; the boundary node is never peeled (it absorbs
//                 routed defects). The correction reproduces the syndrome.
//
// Output: the M-bit correction (edges flipped) and the predicted logical flip (XOR of the logical
// flag over the correction edges). This is a genuine UF datapath, not a lookup table — verified
// bit-exactly in simulation against the Rust unweighted `UnionFindDecoder` (`tb_uf_surface.cpp`).
//
// The decode is combinational (small fixed graph) and registered to a 1-cycle valid/valid handshake;
// pipelining the growth iterations for timing on the KV260 is a later step.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_surface_decoder (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    input  logic [UF_N-2:0]   syndrome,    // detectors 0..N-2 (boundary node N-1 has no detector)
    output logic              out_valid,
    output logic [UF_M-1:0]   correction,
    output logic              obs_flip
);
  // Decode `syn` on the fixed graph, returning {obs_flip, correction[M-1:0]}.
  // (Node indices are `int` working variables; their upper bits are intentionally unused.)
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic [UF_M:0] uf_decode(input logic [UF_N-2:0] syn);
    logic               defect  [UF_N];
    int                 support [UF_M];   // 0,1,2 ; 2 = fused
    int                 label   [UF_N];   // connected-component label (min node id)
    logic               par     [UF_N];   // per-label defect parity (indexed by label root)
    logic               hasb    [UF_N];   // per-label: contains the boundary node
    logic               odd     [UF_N];   // per-node: in an odd (non-neutral) cluster
    int                 troot   [UF_N];   // spanning-tree union-find parent
    logic               istree  [UF_M];   // edge is in the spanning forest
    int                 deg     [UF_N];   // node degree in the forest
    logic               dfct    [UF_N];   // working defect copy for peeling
    logic [UF_M-1:0]    corr;
    logic               obs;
    int                 m, ra, rb, u, v;
    logic               anyodd;

    // --- init ---
    for (int i = 0; i < UF_N; i++) defect[i] = (i < UF_N - 1) ? syn[i] : 1'b0;
    for (int e = 0; e < UF_M; e++) support[e] = 0;

    // --- 1. GROWTH ---  (<= N iterations always suffice to neutralise on this graph)
    for (int gi = 0; gi < UF_N; gi++) begin
      // connected components over fused edges, by min-label propagation to fixpoint.
      for (int i = 0; i < UF_N; i++) label[i] = i;
      for (int p = 0; p < UF_N; p++)
        for (int e = 0; e < UF_M; e++)
          if (support[e] == 2) begin
            m = (label[UF_EA[e]] < label[UF_EB[e]]) ? label[UF_EA[e]] : label[UF_EB[e]];
            label[UF_EA[e]] = m;
            label[UF_EB[e]] = m;
          end
      // per-cluster parity / boundary, stored at the label root.
      for (int i = 0; i < UF_N; i++) begin par[i] = 1'b0; hasb[i] = 1'b0; end
      for (int i = 0; i < UF_N; i++) par[label[i]] = par[label[i]] ^ defect[i];
      hasb[label[UF_BOUNDARY]] = 1'b1;
      // odd (non-neutral) clusters, and grow their incident edges.
      anyodd = 1'b0;
      for (int i = 0; i < UF_N; i++) begin
        odd[i] = par[label[i]] & ~hasb[label[i]];
        anyodd = anyodd | odd[i];
      end
      if (anyodd)
        for (int e = 0; e < UF_M; e++) begin
          int inc;
          inc = (odd[UF_EA[e]] ? 1 : 0) + (odd[UF_EB[e]] ? 1 : 0);
          support[e] = (support[e] + inc > 2) ? 2 : support[e] + inc;
        end
    end

    // --- 2. SPANNING FOREST over fused edges (tree union-find) ---
    for (int i = 0; i < UF_N; i++) troot[i] = i;
    for (int e = 0; e < UF_M; e++) istree[e] = 1'b0;
    for (int e = 0; e < UF_M; e++)
      if (support[e] == 2) begin
        ra = UF_EA[e];
        for (int k = 0; k < UF_N; k++) if (troot[ra] != ra) ra = troot[ra];
        rb = UF_EB[e];
        for (int k = 0; k < UF_N; k++) if (troot[rb] != rb) rb = troot[rb];
        if (ra != rb) begin
          troot[ra]  = rb;
          istree[e]  = 1'b1;
        end
      end

    // --- 3. PEELING ---
    for (int i = 0; i < UF_N; i++) begin deg[i] = 0; dfct[i] = defect[i]; end
    for (int e = 0; e < UF_M; e++)
      if (istree[e]) begin deg[UF_EA[e]]++; deg[UF_EB[e]]++; end
    corr = '0;
    for (int pit = 0; pit < UF_N; pit++)
      for (int e = 0; e < UF_M; e++)
        if (istree[e]) begin
          // peel a non-boundary leaf endpoint of this edge.
          u = -1;
          v = -1;
          if (deg[UF_EA[e]] == 1 && UF_EA[e] != UF_BOUNDARY) begin u = UF_EA[e]; v = UF_EB[e]; end
          else if (deg[UF_EB[e]] == 1 && UF_EB[e] != UF_BOUNDARY) begin u = UF_EB[e]; v = UF_EA[e]; end
          if (u != -1) begin
            if (dfct[u]) begin corr[e] = 1'b1; dfct[v] = dfct[v] ^ 1'b1; end
            istree[e] = 1'b0;
            deg[u]    = 0;
            deg[v]    = deg[v] - 1;
          end
        end

    // logical flip = parity of the logical flag over the correction edges.
    obs = 1'b0;
    for (int e = 0; e < UF_M; e++) if (corr[e] & UF_ELOG[e]) obs = obs ^ 1'b1;

    uf_decode = {obs, corr};
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  logic [UF_M:0] decoded;
  always_comb decoded = uf_decode(syndrome);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid  <= 1'b0;
      correction <= '0;
      obs_flip   <= 1'b0;
    end else begin
      out_valid  <= in_valid;
      correction <= in_valid ? decoded[UF_M-1:0] : '0;
      obs_flip   <= in_valid ? decoded[UF_M] : 1'b0;
    end
  end
endmodule
