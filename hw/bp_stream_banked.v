// Q7-06 (AC-1) — Verilog-2001 board top for the AXI4-Stream BATCH banked-decoder build.
//
// Thin passthrough over the SystemVerilog `bp_stream_banked_core` (Vivado forbids a SV file as the top of
// a block-design module reference; the AXI4-Stream ports are fixed 32-bit so this top is graph-independent,
// mirroring `bp_stream_win.v` / `uf_stream_win.v`). `early_exit` is a board-top port tied at BD level
// (constant or a debug switch); it is not part of the AXI-DMA path. See `bp_stream_banked_core.sv`.

`timescale 1ns / 1ps

module bp_stream_banked (
  input  wire        aclk,
  input  wire        aresetn,
  input  wire        early_exit,

  input  wire [31:0] s_axis_tdata,
  input  wire        s_axis_tvalid,
  output wire        s_axis_tready,
  input  wire        s_axis_tlast,

  output wire [31:0] m_axis_tdata,
  output wire        m_axis_tvalid,
  input  wire        m_axis_tready,
  output wire        m_axis_tlast
);

  bp_stream_banked_core u_core (
    .aclk          (aclk),
    .aresetn       (aresetn),
    .early_exit_i  (early_exit),
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
