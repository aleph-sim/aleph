// Q6-03 (throughput scaling, out-of-order) — Verilog-2001 board top for the reorder-buffer array.
//
// Thin passthrough over `uf_stream_array_ooo_core` (Vivado forbids a SV top for a BD module
// reference; fixed 32-bit AXIS, distance-independent). K = number of parallel decoder engines; set it
// on the BD cell via CONFIG.K. See `uf_stream_array_ooo_core.sv`.

`timescale 1ns / 1ps

module uf_stream_array_ooo #(
  parameter K = 8
)(
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

  uf_stream_array_ooo_core #(.K(K)) u_core (
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
