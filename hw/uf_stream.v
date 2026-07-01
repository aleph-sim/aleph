// Q6-03 (throughput) — Verilog-2001 board top for the streaming DMA build.
//
// Thin passthrough over the SystemVerilog `uf_stream_core` (Vivado forbids a SV file as the top of a
// block-design module reference; the AXI4-Stream ports are fixed 32-bit so this top is distance-
// independent). Instantiated in the block design between the AXI DMA MM2S (syndromes) and S2MM
// (results) channels. See `uf_stream_core.sv`.

`timescale 1ns / 1ps

module uf_stream (
  input  wire        aclk,
  input  wire        aresetn,

  input  wire [31:0] s_axis_tdata,
  input  wire        s_axis_tvalid,
  output wire        s_axis_tready,
  input  wire        s_axis_tlast,

  output wire [31:0] m_axis_tdata,
  output wire        m_axis_tvalid,
  input  wire        m_axis_tready,
  output wire        m_axis_tlast
);

  uf_stream_core u_core (
    .aclk          (aclk),
    .aresetn       (aresetn),
    .s_axis_tdata  (s_axis_tdata),
    .s_axis_tvalid (s_axis_tvalid),
    .s_axis_tready (s_axis_tready),
    .s_axis_tlast  (s_axis_tlast),
    .m_axis_tdata  (m_axis_tdata),
    .m_axis_tvalid (m_axis_tvalid),
    .m_axis_tready (m_axis_tready),
    .m_axis_tlast  (m_axis_tlast)
  );

endmodule
