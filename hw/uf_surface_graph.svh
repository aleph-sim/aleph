// d=3 rotated surface-code memory-Z (1 round(s)) phenomenological matching graph — GENERATED, do not edit.
// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- graph 3 1 > hw/uf_surface_graph_d3.svh
`ifndef UF_SURFACE_GRAPH_SVH
`define UF_SURFACE_GRAPH_SVH
localparam int UF_N = 9;
localparam int UF_M = 18;
localparam int UF_BOUNDARY = 8;
localparam int UF_EA   [UF_M] = '{0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7};
localparam int UF_EB   [UF_M] = '{2, 4, 8, 2, 3, 5, 8, 6, 8, 7, 8, 6, 8, 6, 7, 8, 8, 8};
localparam bit UF_ELOG [UF_M] = '{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
`endif
