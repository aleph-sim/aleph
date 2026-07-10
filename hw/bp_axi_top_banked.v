// Q7-02 M5-followup — Verilog module-ref top for the AXI4-Lite wrapper for the M7 banked decoder.
//
// Thin structural top so Vivado's block-design module reference elaborates (must be Verilog, not SV; the
// submodules bp_axi_wrap_banked / bp_relay_decoder stay SystemVerilog). Same shape as bp_axi_top.v.

`timescale 1ns / 1ps

module bp_axi_top_banked #(
  parameter C_ADDR_W = 8
)(
  input  wire                 aclk,
  input  wire                 aresetn,

  input  wire [C_ADDR_W-1:0]  s_axil_awaddr,
  input  wire                 s_axil_awvalid,
  output wire                 s_axil_awready,
  input  wire [31:0]          s_axil_wdata,
  input  wire [3:0]           s_axil_wstrb,
  input  wire                 s_axil_wvalid,
  output wire                 s_axil_wready,
  output wire [1:0]           s_axil_bresp,
  output wire                 s_axil_bvalid,
  input  wire                 s_axil_bready,
  input  wire [C_ADDR_W-1:0]  s_axil_araddr,
  input  wire                 s_axil_arvalid,
  output wire                 s_axil_arready,
  output wire [31:0]          s_axil_rdata,
  output wire [1:0]           s_axil_rresp,
  output wire                 s_axil_rvalid,
  input  wire                 s_axil_rready
);

  bp_axi_wrap_banked #(.C_ADDR_W(C_ADDR_W)) u_wrap (
    .aclk           (aclk),
    .aresetn        (aresetn),
    .s_axil_awaddr  (s_axil_awaddr),
    .s_axil_awvalid (s_axil_awvalid),
    .s_axil_awready (s_axil_awready),
    .s_axil_wdata   (s_axil_wdata),
    .s_axil_wstrb   (s_axil_wstrb),
    .s_axil_wvalid  (s_axil_wvalid),
    .s_axil_wready  (s_axil_wready),
    .s_axil_bresp   (s_axil_bresp),
    .s_axil_bvalid  (s_axil_bvalid),
    .s_axil_bready  (s_axil_bready),
    .s_axil_araddr  (s_axil_araddr),
    .s_axil_arvalid (s_axil_arvalid),
    .s_axil_arready (s_axil_arready),
    .s_axil_rdata   (s_axil_rdata),
    .s_axil_rresp   (s_axil_rresp),
    .s_axil_rvalid  (s_axil_rvalid),
    .s_axil_rready  (s_axil_rready)
  );

endmodule
