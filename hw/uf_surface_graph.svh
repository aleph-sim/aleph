// d=3 rotated surface-code memory-Z (1 round) matching graph — GENERATED, do not edit.
// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- graph > hw/uf_surface_graph.svh
localparam int UF_N = 9;
localparam int UF_M = 18;
localparam int UF_BOUNDARY = 8;
localparam int UF_EA   [UF_M] = '{0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7};
localparam int UF_EB   [UF_M] = '{2, 4, 8, 2, 3, 5, 8, 6, 8, 7, 8, 6, 8, 6, 7, 8, 8, 8};
localparam bit UF_ELOG [UF_M] = '{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
