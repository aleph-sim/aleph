// P2 appliance v2-large — AWS F2 Custom Logic top: banked relay-BP decoder behind OCL BAR0.
//
// Everything except OCL (registers, per-word path) and PCIS (batched path) is tied off: no DDR (sh_ddr with
// DDR_PRESENT=0, which the shell requires to be instantiated), no HBM, no PCIM, no SDA, no interrupts. The decoder lives in `aleph_bp_ocl`; this
// file only adapts it to the shell's port list (`cl_ports.vh`, AWS F2 HDK).

module cl_aleph_bp #(
    parameter EN_DDR = 0,
    parameter EN_HBM = 0
) (
    `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "cl_aleph_bp_defines.vh"

  // ---------------------------------------------------------------- decoder on OCL
  logic rst_main_n_q;
  always_ff @(posedge clk_main_a0) rst_main_n_q <= rst_main_n;

  aleph_bp_ocl #(
      .CORE_DIV_F(`ALEPH_CORE_DIV_F),
      .CORE_KHZ  (`ALEPH_CORE_KHZ)
  ) u_aleph (
      .clk_main  (clk_main_a0),
      .rst_main_n(rst_main_n_q),
      .clk_ref   (clk_hbm_ref),
      .s_awaddr  (ocl_cl_awaddr),
      .s_awvalid (ocl_cl_awvalid),
      .s_awready (cl_ocl_awready),
      .s_wdata   (ocl_cl_wdata),
      .s_wstrb   (ocl_cl_wstrb),
      .s_wvalid  (ocl_cl_wvalid),
      .s_wready  (cl_ocl_wready),
      .s_bresp   (cl_ocl_bresp),
      .s_bvalid  (cl_ocl_bvalid),
      .s_bready  (ocl_cl_bready),
      .s_araddr  (ocl_cl_araddr),
      .s_arvalid (ocl_cl_arvalid),
      .s_arready (cl_ocl_arready),
      .s_rdata   (cl_ocl_rdata),
      .s_rresp   (cl_ocl_rresp),
      .s_rvalid  (cl_ocl_rvalid),
      .s_rready  (ocl_cl_rready),
      // PCIS: burst type, cache/prot/qos/lock and user bits carry nothing this slave needs
      .p_awaddr  (sh_cl_dma_pcis_awaddr),
      .p_awid    (sh_cl_dma_pcis_awid),
      .p_awsize  (sh_cl_dma_pcis_awsize),
      .p_awvalid (sh_cl_dma_pcis_awvalid),
      .p_awready (cl_sh_dma_pcis_awready),
      .p_wdata   (sh_cl_dma_pcis_wdata),
      .p_wstrb   (sh_cl_dma_pcis_wstrb),
      .p_wlast   (sh_cl_dma_pcis_wlast),
      .p_wvalid  (sh_cl_dma_pcis_wvalid),
      .p_wready  (cl_sh_dma_pcis_wready),
      .p_bid     (cl_sh_dma_pcis_bid),
      .p_bresp   (cl_sh_dma_pcis_bresp),
      .p_bvalid  (cl_sh_dma_pcis_bvalid),
      .p_bready  (sh_cl_dma_pcis_bready),
      .p_araddr  (sh_cl_dma_pcis_araddr),
      .p_arid    (sh_cl_dma_pcis_arid),
      .p_arlen   (sh_cl_dma_pcis_arlen),
      .p_arsize  (sh_cl_dma_pcis_arsize),
      .p_arvalid (sh_cl_dma_pcis_arvalid),
      .p_arready (cl_sh_dma_pcis_arready),
      .p_rid     (cl_sh_dma_pcis_rid),
      .p_rdata   (cl_sh_dma_pcis_rdata),
      .p_rresp   (cl_sh_dma_pcis_rresp),
      .p_rlast   (cl_sh_dma_pcis_rlast),
      .p_rvalid  (cl_sh_dma_pcis_rvalid),
      .p_rready  (sh_cl_dma_pcis_rready)
  );
  assign cl_sh_dma_pcis_ruser = '0;

  // ---------------------------------------------------------------- globals
  always_comb begin
    cl_sh_flr_done    = 1'b1;
    cl_sh_status0     = '0;
    cl_sh_status1     = '0;
    cl_sh_status2     = '0;
    cl_sh_id0         = `CL_SH_ID0;
    cl_sh_id1         = `CL_SH_ID1;
    cl_sh_status_vled = '0;
    cl_sh_dma_wr_full = '0;
    cl_sh_dma_rd_full = '0;
  end

  // ---------------------------------------------------------------- PCIM (unused master: idle)
  always_comb begin
    cl_sh_pcim_awaddr  = '0; cl_sh_pcim_awsize  = '0; cl_sh_pcim_awburst = '0; cl_sh_pcim_awvalid = '0;
    cl_sh_pcim_awid    = '0; cl_sh_pcim_awlen   = '0; cl_sh_pcim_awcache = '0; cl_sh_pcim_awlock  = '0;
    cl_sh_pcim_awprot  = '0; cl_sh_pcim_awqos   = '0; cl_sh_pcim_awuser  = '0;
    cl_sh_pcim_wdata   = '0; cl_sh_pcim_wstrb   = '0; cl_sh_pcim_wlast   = '0; cl_sh_pcim_wvalid  = '0;
    cl_sh_pcim_wid     = '0; cl_sh_pcim_wuser   = '0;
    cl_sh_pcim_araddr  = '0; cl_sh_pcim_arsize  = '0; cl_sh_pcim_arburst = '0; cl_sh_pcim_arvalid = '0;
    cl_sh_pcim_arid    = '0; cl_sh_pcim_arlen   = '0; cl_sh_pcim_arcache = '0; cl_sh_pcim_arlock  = '0;
    cl_sh_pcim_arprot  = '0; cl_sh_pcim_arqos   = '0; cl_sh_pcim_aruser  = '0;
    cl_sh_pcim_bready  = '0;
    cl_sh_pcim_rready  = '0;
  end

  // ---------------------------------------------------------------- SDA (unused slave)
  always_comb begin
    cl_sda_awready = '0; cl_sda_wready = '0; cl_sda_bresp = '0; cl_sda_bvalid = '0;
    cl_sda_arready = '0; cl_sda_rdata  = '0; cl_sda_rresp = '0; cl_sda_rvalid = '0;
  end

  // ---------------------------------------------------------------- DDR: required instance, not present
  sh_ddr #(.DDR_PRESENT(EN_DDR)) SH_DDR (
      .clk(clk_main_a0), .rst_n(), .stat_clk(clk_main_a0), .stat_rst_n(),
      .CLK_DIMM_DP(CLK_DIMM_DP), .CLK_DIMM_DN(CLK_DIMM_DN), .M_ACT_N(M_ACT_N), .M_MA(M_MA), .M_BA(M_BA),
      .M_BG(M_BG), .M_CKE(M_CKE), .M_ODT(M_ODT), .M_CS_N(M_CS_N), .M_CLK_DN(M_CLK_DN), .M_CLK_DP(M_CLK_DP),
      .M_PAR(M_PAR), .M_DQ(M_DQ), .M_ECC(M_ECC), .M_DQS_DP(M_DQS_DP), .M_DQS_DN(M_DQS_DN),
      .cl_RST_DIMM_N(RST_DIMM_N),
      .cl_sh_ddr_axi_awid(), .cl_sh_ddr_axi_awaddr(), .cl_sh_ddr_axi_awlen(), .cl_sh_ddr_axi_awsize(),
      .cl_sh_ddr_axi_awvalid(), .cl_sh_ddr_axi_awburst(), .cl_sh_ddr_axi_awuser(), .cl_sh_ddr_axi_awready(),
      .cl_sh_ddr_axi_wdata(), .cl_sh_ddr_axi_wstrb(), .cl_sh_ddr_axi_wlast(), .cl_sh_ddr_axi_wvalid(),
      .cl_sh_ddr_axi_wready(), .cl_sh_ddr_axi_bid(), .cl_sh_ddr_axi_bresp(), .cl_sh_ddr_axi_bvalid(),
      .cl_sh_ddr_axi_bready(), .cl_sh_ddr_axi_arid(), .cl_sh_ddr_axi_araddr(), .cl_sh_ddr_axi_arlen(),
      .cl_sh_ddr_axi_arsize(), .cl_sh_ddr_axi_arvalid(), .cl_sh_ddr_axi_arburst(), .cl_sh_ddr_axi_aruser(),
      .cl_sh_ddr_axi_arready(), .cl_sh_ddr_axi_rid(), .cl_sh_ddr_axi_rdata(), .cl_sh_ddr_axi_rresp(),
      .cl_sh_ddr_axi_rlast(), .cl_sh_ddr_axi_rvalid(), .cl_sh_ddr_axi_rready(),
      .sh_ddr_stat_bus_addr(), .sh_ddr_stat_bus_wdata(), .sh_ddr_stat_bus_wr(), .sh_ddr_stat_bus_rd(),
      .sh_ddr_stat_bus_ack(), .sh_ddr_stat_bus_rdata(), .ddr_sh_stat_int(), .sh_cl_ddr_is_ready()
  );
  always_comb begin
    cl_sh_ddr_stat_ack   = '0;
    cl_sh_ddr_stat_rdata = '0;
    cl_sh_ddr_stat_int   = '0;
  end

  // ---------------------------------------------------------------- interrupts, JTAG, HBM monitor, PCIe
  always_comb begin
    cl_sh_apppf_irq_req = '0;
    tdo                 = '0;
    hbm_apb_paddr_0 = '0; hbm_apb_pprot_0 = '0; hbm_apb_psel_0   = '0; hbm_apb_penable_0 = '0;
    hbm_apb_pwrite_0 = '0; hbm_apb_pwdata_0 = '0; hbm_apb_pstrb_0 = '0; hbm_apb_pready_0 = '0;
    hbm_apb_prdata_0 = '0; hbm_apb_pslverr_0 = '0;
    hbm_apb_paddr_1 = '0; hbm_apb_pprot_1 = '0; hbm_apb_psel_1   = '0; hbm_apb_penable_1 = '0;
    hbm_apb_pwrite_1 = '0; hbm_apb_pwdata_1 = '0; hbm_apb_pstrb_1 = '0; hbm_apb_pready_1 = '0;
    hbm_apb_prdata_1 = '0; hbm_apb_pslverr_1 = '0;
    PCIE_EP_TXP = '0; PCIE_EP_TXN = '0;
    PCIE_RP_PERSTN = '0; PCIE_RP_TXP = '0; PCIE_RP_TXN = '0;
  end

endmodule
