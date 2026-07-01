// Q6-08 (board build) — top level for the Arty Z7-20 block design.
//
// Instantiates `uf_axi_wrap` exposing ONLY the AXI4-Lite control plane to the PS; the AXI4-Stream
// data plane is tied off here (first on-board bring-up drives the decoder over AXI4-Lite, exactly
// as `hw/sw/uf_pynq.py` does). Keeping the tie-off in RTL — rather than as dangling interface pins
// in the block design — makes the BD self-contained and deterministic for a headless tcl build.
//
// Plain Verilog-2001 on purpose: Vivado forbids a SystemVerilog file as the TOP file of a block-
// design module reference. The submodules (uf_axi_wrap, uf_surface_decoder) stay SystemVerilog;
// only this thin structural top must be Verilog. Logic-free wrapper — nothing about the verified
// decoder or wrapper changes.

`timescale 1ns / 1ps

module uf_axi_top #(
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

  // AXI4-Stream tie-off: no ingress traffic, egress ready held high so the wrapper never stalls.
  wire        s_axis_tready_unused;
  wire [31:0] m_axis_tdata_unused;
  wire        m_axis_tvalid_unused;
  wire        m_axis_tlast_unused;

  uf_axi_wrap #(.C_ADDR_W(C_ADDR_W)) u_wrap (
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
    .s_axil_rready  (s_axil_rready),

    // AXI4-Stream tied off for the AXI4-Lite-only bring-up.
    .s_axis_tdata   (32'b0),
    .s_axis_tvalid  (1'b0),
    .s_axis_tready  (s_axis_tready_unused),
    .m_axis_tdata   (m_axis_tdata_unused),
    .m_axis_tvalid  (m_axis_tvalid_unused),
    .m_axis_tready  (1'b1),
    .m_axis_tlast   (m_axis_tlast_unused)
  );

endmodule
