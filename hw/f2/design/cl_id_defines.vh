// P2 appliance v2-large — PCIe identity of the AFI (AWS F2 requires these in every CL).
`ifndef CL_ID_DEFINES
`define CL_ID_DEFINES

  // [31:16] device ID, [15:0] vendor ID. With Amazon's vendor ID 0x1D0F the device ID must be 0xF000-0xF0FF.
  `define CL_SH_ID0 32'hF0A1_1D0F

  // [31:16] subsystem ID, [15:0] subsystem vendor ID. Neither may be zero.
  `define CL_SH_ID1 32'hA1E9_1D51

`endif
