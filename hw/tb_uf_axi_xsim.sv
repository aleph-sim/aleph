// Q6-07 — xsim testbench for the AXI PS<->PL wrapper (uf_axi_wrap).
//
// Exercises BOTH planes against the frozen golden table over all 256 syndromes:
//   * AXI4-Lite: write SYNDROME, pulse CTRL.START, poll STATUS.DONE, read CORRECTION + OBS_FLIP.
//   * AXI4-Stream: push syndrome on s_axis, capture {obs,correction} on m_axis.
// Plus an IDCODE read sanity check. A pass certifies the wrapper presents the decoder correctly to
// the PS without changing its results.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module tb_uf_axi_xsim;
  localparam int SYN_W = UF_N - 1;
  localparam int NSYN  = 1 << SYN_W;
  localparam logic [5:0] A_CTRL=6'h00, A_STATUS=6'h04, A_SYND=6'h08, A_CORR=6'h0C,
                         A_LAT=6'h10, A_ID=6'h14;

  logic aclk = 1'b0, aresetn = 1'b0;
  always #5 aclk = ~aclk;

  // AXI-Lite
  logic [5:0]  awaddr;  logic awvalid, awready;
  logic [31:0] wdata;   logic [3:0] wstrb; logic wvalid, wready;
  logic [1:0]  bresp;   logic bvalid, bready;
  logic [5:0]  araddr;  logic arvalid, arready;
  logic [31:0] rdata;   logic [1:0] rresp; logic rvalid, rready;
  // AXI-Stream
  logic [31:0] s_tdata; logic s_tvalid, s_tready;
  logic [31:0] m_tdata; logic m_tvalid, m_tready; logic m_tlast;

  logic [UF_M:0] golden [0:NSYN-1];

  uf_axi_wrap dut (
    .aclk, .aresetn,
    .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
    .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
    .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
    .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
    .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),
    .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
    .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready), .m_axis_tlast(m_tlast)
  );

  int lite_fail = 0, axis_fail = 0;

  task automatic axil_write(input logic [5:0] a, input logic [31:0] d);
    @(posedge aclk); awaddr <= a; awvalid <= 1'b1; wdata <= d; wstrb <= 4'hF; wvalid <= 1'b1;
    forever begin @(posedge aclk); if (awready && wready) break; end
    awvalid <= 1'b0; wvalid <= 1'b0;
    bready <= 1'b1;                              // only now accept the write response
    forever begin @(posedge aclk); if (bvalid) break; end
    bready <= 1'b0;
  endtask

  task automatic axil_read(input logic [5:0] a, output logic [31:0] d);
    @(posedge aclk); araddr <= a; arvalid <= 1'b1;
    forever begin @(posedge aclk); if (arready) break; end
    arvalid <= 1'b0;
    rready <= 1'b1;                              // only now accept read data
    forever begin @(posedge aclk); if (rvalid) break; end
    d = rdata;
    rready <= 1'b0;
  endtask

  task automatic axis_decode(input int s, output logic [31:0] outd);
    @(posedge aclk); s_tdata <= s; s_tvalid <= 1'b1;
    forever begin @(posedge aclk); if (s_tready) break; end
    s_tvalid <= 1'b0;
    forever begin @(posedge aclk); if (m_tvalid) break; end   // m_tready held high
    outd = m_tdata;
  endtask

  logic [31:0] tmp, status, corr;
  initial begin
    $readmemh("uf_surface_golden.mem", golden);
    awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0; s_tvalid=0; m_tready=1'b1;
    awaddr=0; wdata=0; wstrb=0; araddr=0; s_tdata=0;
    repeat (20) @(posedge aclk);
    aresetn <= 1'b1;
    repeat (2) @(posedge aclk);

    // IDCODE sanity
    axil_read(A_ID, tmp);
    if (tmp !== 32'h5546_0003) begin $display("IDCODE FAIL: got %h", tmp); lite_fail++; end

    // ---- AXI4-Lite path over all syndromes ----
    for (int s = 0; s < NSYN; s++) begin
      axil_write(A_SYND, s);
      axil_write(A_CTRL, 32'h1);                 // START
      do axil_read(A_STATUS, status); while (!status[1]);   // poll DONE
      axil_read(A_CORR, corr);
      // {obs, correction} == golden
      if ({status[2], corr[UF_M-1:0]} !== golden[s]) begin
        $display("LITE FAIL s=%0d: {obs,corr}=%h golden=%h", s, {status[2], corr[UF_M-1:0]}, golden[s]);
        if (++lite_fail > 10) break;
      end
    end

    // ---- AXI4-Stream path over all syndromes ----
    for (int s = 0; s < NSYN; s++) begin
      axis_decode(s, tmp);
      if (tmp[UF_M:0] !== golden[s]) begin
        $display("AXIS FAIL s=%0d: {obs,corr}=%h golden=%h", s, tmp[UF_M:0], golden[s]);
        if (++axis_fail > 10) break;
      end
    end

    $display("axi: syndromes=%0d  lite_fail=%0d  axis_fail=%0d", NSYN, lite_fail, axis_fail);
    if (lite_fail || axis_fail) begin $display("RESULT: FAIL"); $fatal(1, "AXI wrapper sign-off failed"); end
    else $display("RESULT: PASS (all %0d syndromes match golden via AXI-Lite + AXI-Stream)", NSYN);
    $finish;
  end

  // global timeout guard
  initial begin
    #5_000_000;
    $display("RESULT: FAIL (timeout)");
    $fatal(1, "timeout");
  end
endmodule
