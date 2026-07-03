// Q7-02 board build — top level for the Arty Z7-20 block design (partial relay-BP decoder).
//
// Instantiates `bp_axi_wrap` exposing the AXI4-Lite control plane to the PS. Plain Verilog-2001 on
// purpose: Vivado forbids a SystemVerilog file as the TOP of a block-design module reference. The
// submodules (bp_axi_wrap, bp_relay_partial) stay SystemVerilog; only this thin structural top must be
// Verilog. Logic-free wrapper — nothing about the verified decoder or wrapper changes.

`timescale 1ns / 1ps

module bp_axi_top #(
  parameter C_ADDR_W = 6
)(
  input  wire                 aclk,
  input  wire                 aresetn,

  // ---- AXI4-Lite slave (control plane) ----
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

  bp_axi_wrap #(.C_ADDR_W(C_ADDR_W)) u_wrap (
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
