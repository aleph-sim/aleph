// P2 appliance v2-large — build-time selection for the F2 CL.
`ifndef CL_ALEPH_BP_DEFINES
`define CL_ALEPH_BP_DEFINES

  // The HDK's top.sv instantiates `CL_NAME.
  `define CL_NAME cl_aleph_bp

  // Decoder clock from the CL's own MMCM: 100 MHz clk_hbm_ref * 15 / ALEPH_CORE_DIV_F.
  // Default 125 MHz (12.0). Build with +define+ALEPH_150 for 150 MHz (10.0); the 64/192 core met 150.4 MHz
  // post-route out of context (docs/perf/q7-02-fullparallel-fpga.md §9), so 150 has almost no margin.
  `ifdef ALEPH_150
    `define ALEPH_CORE_DIV_F 10.0
    `define ALEPH_CORE_KHZ   150000
  `else
    `define ALEPH_CORE_DIV_F 12.0
    `define ALEPH_CORE_KHZ   125000
  `endif

`endif
