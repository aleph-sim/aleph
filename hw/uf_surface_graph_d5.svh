// d=5 rotated surface-code memory-Z (1 round(s)) phenomenological matching graph — GENERATED, do not edit.
// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- graph 5 1 > hw/uf_surface_graph_d5.svh
`ifndef UF_SURFACE_GRAPH_SVH
`define UF_SURFACE_GRAPH_SVH
localparam int UF_N = 25;
localparam int UF_M = 54;
localparam int UF_BOUNDARY = 24;
localparam int UF_EA   [UF_M] = '{0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 17, 18, 19, 19, 20, 20, 21, 22, 23};
localparam int UF_EB   [UF_M] = '{3, 12, 24, 3, 4, 13, 24, 4, 5, 14, 24, 6, 7, 15, 7, 8, 16, 8, 17, 9, 18, 9, 10, 19, 10, 11, 20, 21, 24, 22, 24, 23, 24, 15, 24, 15, 16, 24, 16, 17, 24, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 24};
localparam bit UF_ELOG [UF_M] = '{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
`endif
