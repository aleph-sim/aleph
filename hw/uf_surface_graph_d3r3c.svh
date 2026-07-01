// d=3 rotated surface-code memory-Z (3 round(s)) circuit-level matching graph — GENERATED, do not edit.
// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- graph-circuit 3 3 > hw/uf_surface_graph_d3.svh
`ifndef UF_SURFACE_GRAPH_SVH
`define UF_SURFACE_GRAPH_SVH
localparam int UF_N = 17;
localparam int UF_M = 49;
localparam int UF_BOUNDARY = 16;
localparam int UF_EA   [UF_M] = '{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 15};
localparam int UF_EB   [UF_M] = '{2, 4, 16, 2, 3, 5, 16, 4, 5, 6, 16, 5, 7, 16, 6, 8, 16, 6, 7, 9, 16, 8, 9, 10, 16, 9, 11, 16, 10, 12, 16, 10, 11, 13, 16, 12, 13, 14, 16, 13, 15, 16, 14, 16, 14, 15, 16, 16, 16};
localparam bit UF_ELOG [UF_M] = '{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
`endif
