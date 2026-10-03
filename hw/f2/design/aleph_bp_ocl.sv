// P2 appliance v2-large — AXI4-Lite (AWS F2 OCL BAR0) front end for the banked relay-BP stream core.
//
// F2's small shell exposes no DMA to the CL, so the host reaches the decoder through 32-bit OCL
// pokes/peeks. This module turns those into the same AXI4-Stream the KV260/ZCU104 images feed from an
// AXI-DMA, so the decoder core (`bp_stream_banked_core`) is byte-for-byte the one co-simulated there:
//
//   host --OCL AXI-Lite (clk_main_a0, 250 MHz)--> PUSH --xpm_fifo_async--> s_axis --+
//                                                                                   | core (clk_core)
//   host <--OCL AXI-Lite-- POP <--xpm_fifo_async-- m_axis <-------------------------+
//
// In PCIS mode (CTRL.pcis_mode) the same two FIFOs are fed and drained by `aleph_pcis_ring` instead: the
// host writes whole experiments into BAR4 slots and reads results back from a ring there, so a batch
// costs about one 64-byte store and one eighth of a 64-byte read per experiment instead of five OCL
// stores and polled OCL reads.
//
// clk_core comes from a local MMCM off the shell's fixed 100 MHz clk_hbm_ref, so its frequency is fixed
// in the bitstream: no clock recipe, no runtime clock API, and no way for the host side to program a
// different frequency from the one timing closed at. (That is exactly the PYNQ trap the ZCU104 build hit,
// `docs/perf/p2-appliance-v2.md` §3.) The host can still measure it: CORE_CYC and MAIN_CYC are free-running
// counters in the two domains, and their ratio times 250 MHz is the clock actually running.
//
// Decode latency is the core's own cycle count (result word bits [15:0]), so PCIe and the OCL round
// trip never enter the latency number.
//
// Register map (byte offsets in OCL BAR0, all 32-bit). Every write must carry all four byte strobes; a
// write with any other strobe pattern changes nothing and sets STATUS.err_strb (a 32-bit store that the
// host split into pieces must never turn into PUSH beats). Reads have no side effects (all F2 BARs are
// prefetchable, `AWS_Fpga_Pcie_Memory_Map.md`), so popping a result is an explicit write.
//   0x00 MAGIC     RO  0xA1E9_B0F2
//   0x04 GEOM      RO  {BP_BANK_W[15:0], BP_BANK_V[15:0]}
//   0x08 STATUS    RO  [0] in_full [1] in_empty [2] out_full [3] out_empty [4] mmcm_locked
//                      [5] core_in_reset [6] in_rst_busy [7] out_rst_busy
//                      sticky until CTRL.clear_err / soft reset: [8] err_strb (partial-strobe write)
//                      [9] err_push (PUSH dropped: input FIFO full, or in PCIS mode)
//                      [10] err_pcis (PCIS beat refused, see PCIS_DROP)
//                      live: [11] ring_clearing [12] pcis_mode
//   0x0C CTRL      RW  [0] soft_reset (1 = hold core, FIFOs, counters and the PCIS path in reset)
//                      [1] early_exit [2] pcis_mode [3] clear_err (write-1 pulse, reads 0)
//   0x10 PUSH      WO  one 32-bit syndrome beat into the input FIFO (OCL mode only)
//   0x14 POP       RO  peek at the oldest result: {obs[31:20], valid_flag[19], 0[18], seq[17:16],
//                      latency[15:0]}, or 0xFFFF_FFFF if there is none. seq = RESULTS[1:0]. A result can
//                      never read as all-ones: bit 18 is always 0.
//   0x18 CORE_CYC  RO  free-running clk_core counter (gray-synchronised into clk_main_a0)
//   0x1C MAIN_CYC  RO  free-running clk_main_a0 counter
//   0x20 CORE_KHZ  RO  nominal clk_core in kHz, as built (the MMCM divide in this bitstream)
//   0x24 POP_ACK   WO  [1:0] seq: drop the peeked result if seq matches (a repeated or stale ack is a
//                      no-op, so an ack can never pop twice). OCL mode only.
//   0x28 PUSH_WORDS RO words written into the decoder input FIFO since reset (either path)
//   0x2C RESULTS   RO  results taken out of the decoder since reset (either path)
//   0x30 PCIS_ACC  RO  experiments committed through PCIS since reset
//   0x34 PCIS_DROP RO  PCIS write beats refused since reset (not in PCIS mode, ring-window write, queue full)
//   0x38 VERSION   RO  2 (the first public image has no such register and reads 0xDEAD_BEEF here)
//   0x3C SLOTS     RO  PCIS experiment slots (aleph_pcis_ring.sv describes the BAR4 protocol)

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module aleph_bp_ocl #(
    // MMCM: 100 MHz * CLKFBOUT_MULT_F / CORE_DIV_F. VCO = 1500 MHz (inside the -2 range 800-1600).
    // 12.0 -> 125 MHz, 10.0 -> 150 MHz.
    parameter real CORE_DIV_F = 12.0,
    parameter int  CORE_KHZ   = 125000,
    parameter int  PCIS_SLOTS = 128
) (
    input  logic        clk_main,      // clk_main_a0, 250 MHz, OCL domain
    input  logic        rst_main_n,    // sync to clk_main
    input  logic        clk_ref,       // clk_hbm_ref, 100 MHz, MMCM input

    input  logic [31:0] s_awaddr,
    input  logic        s_awvalid,
    output logic        s_awready,
    input  logic [31:0] s_wdata,
    input  logic [3:0]  s_wstrb,
    input  logic        s_wvalid,
    output logic        s_wready,
    output logic [1:0]  s_bresp,
    output logic        s_bvalid,
    input  logic        s_bready,
    input  logic [31:0] s_araddr,
    input  logic        s_arvalid,
    output logic        s_arready,
    output logic [31:0] s_rdata,
    output logic [1:0]  s_rresp,
    output logic        s_rvalid,
    input  logic        s_rready,

    // PCIS (BAR4) AXI4 slave, clk_main domain; see aleph_pcis_ring
    input  logic [63:0]  p_awaddr,
    input  logic [15:0]  p_awid,
    input  logic [2:0]   p_awsize,
    input  logic         p_awvalid,
    output logic         p_awready,
    input  logic [511:0] p_wdata,
    input  logic [63:0]  p_wstrb,
    input  logic         p_wlast,
    input  logic         p_wvalid,
    output logic         p_wready,
    output logic [15:0]  p_bid,
    output logic [1:0]   p_bresp,
    output logic         p_bvalid,
    input  logic         p_bready,
    input  logic [63:0]  p_araddr,
    input  logic [15:0]  p_arid,
    input  logic [7:0]   p_arlen,
    input  logic [2:0]   p_arsize,
    input  logic         p_arvalid,
    output logic         p_arready,
    output logic [15:0]  p_rid,
    output logic [511:0] p_rdata,
    output logic [1:0]   p_rresp,
    output logic         p_rlast,
    output logic         p_rvalid,
    input  logic         p_rready
);
  localparam logic [31:0] MAGIC   = 32'hA1E9_B0F2;
  localparam logic [31:0] VERSION = 32'd2;

  // ---------------------------------------------------------------- core clock
  logic clk_core, mmcm_fb, mmcm_fb_bufg, clk_core_unbuf, mmcm_locked;

  MMCME4_BASE #(
      .CLKIN1_PERIOD   (10.0),
      .DIVCLK_DIVIDE   (1),
      .CLKFBOUT_MULT_F (15.0),
      .CLKOUT0_DIVIDE_F(CORE_DIV_F)
  ) u_mmcm (
      .CLKIN1   (clk_ref),
      .CLKFBIN  (mmcm_fb_bufg),
      .CLKFBOUT (mmcm_fb),
      .CLKFBOUTB(),
      .CLKOUT0  (clk_core_unbuf),
      .CLKOUT0B (), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(),
      .CLKOUT4  (), .CLKOUT5(), .CLKOUT6(),
      .LOCKED   (mmcm_locked),
      .PWRDWN   (1'b0),
      .RST      (~rst_main_n)
  );
  BUFG u_bufg_fb   (.I(mmcm_fb),        .O(mmcm_fb_bufg));
  BUFG u_bufg_core (.I(clk_core_unbuf), .O(clk_core));

  // ---------------------------------------------------------------- resets
  logic soft_rst, early_exit, pcis_mode;
  // FIFO resets are active-high and must be synchronous to each FIFO's write clock.
  logic in_fifo_rst = 1'b1;  // starts in reset from configuration, before the first clk_main edge
  always_ff @(posedge clk_main) in_fifo_rst <= ~rst_main_n | soft_rst;

  logic core_rst;  // active-high, synchronous to clk_core
  xpm_cdc_async_rst #(.DEST_SYNC_FF(4), .RST_ACTIVE_HIGH(1)) u_core_rst (
      .src_arst (~rst_main_n | soft_rst | ~mmcm_locked),
      .dest_clk (clk_core),
      .dest_arst(core_rst)
  );

  logic early_exit_core;
  xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(1)) u_ee (
      .src_clk(clk_main), .src_in(early_exit), .dest_clk(clk_core), .dest_out(early_exit_core)
  );

  // ---------------------------------------------------------------- FIFOs
  logic        in_wr_en, in_full, in_wr_rst_busy;
  logic [31:0] in_din;
  logic        in_rd_en, in_empty, in_rd_rst_busy;
  logic [31:0] in_dout;

  logic        out_wr_en, out_full, out_wr_rst_busy;
  logic [31:0] out_din;
  logic        out_rd_en, out_empty, out_rd_rst_busy;
  logic [31:0] out_dout;

  xpm_fifo_async #(
      .FIFO_MEMORY_TYPE("auto"), .FIFO_WRITE_DEPTH(512), .WRITE_DATA_WIDTH(32), .READ_DATA_WIDTH(32),
      .READ_MODE("fwft"), .FIFO_READ_LATENCY(0), .CDC_SYNC_STAGES(3), .USE_ADV_FEATURES("0000"),
      .ECC_MODE("no_ecc"), .RELATED_CLOCKS(0), .SIM_ASSERT_CHK(1)
  ) u_in_fifo (
      .rst(in_fifo_rst), .wr_clk(clk_main), .wr_en(in_wr_en), .din(in_din), .full(in_full),
      .wr_rst_busy(in_wr_rst_busy),
      .rd_clk(clk_core), .rd_en(in_rd_en), .dout(in_dout), .empty(in_empty),
      .rd_rst_busy(in_rd_rst_busy),
      .sleep(1'b0), .injectsbiterr(1'b0), .injectdbiterr(1'b0),
      .overflow(), .underflow(), .prog_full(), .prog_empty(), .wr_data_count(), .rd_data_count(),
      .almost_full(), .almost_empty(), .wr_ack(), .data_valid(), .sbiterr(), .dbiterr()
  );

  xpm_fifo_async #(
      .FIFO_MEMORY_TYPE("auto"), .FIFO_WRITE_DEPTH(512), .WRITE_DATA_WIDTH(32), .READ_DATA_WIDTH(32),
      .READ_MODE("fwft"), .FIFO_READ_LATENCY(0), .CDC_SYNC_STAGES(3), .USE_ADV_FEATURES("0000"),
      .ECC_MODE("no_ecc"), .RELATED_CLOCKS(0), .SIM_ASSERT_CHK(1)
  ) u_out_fifo (
      .rst(core_rst), .wr_clk(clk_core), .wr_en(out_wr_en), .din(out_din), .full(out_full),
      .wr_rst_busy(out_wr_rst_busy),
      .rd_clk(clk_main), .rd_en(out_rd_en), .dout(out_dout), .empty(out_empty),
      .rd_rst_busy(out_rd_rst_busy),
      .sleep(1'b0), .injectsbiterr(1'b0), .injectdbiterr(1'b0),
      .overflow(), .underflow(), .prog_full(), .prog_empty(), .wr_data_count(), .rd_data_count(),
      .almost_full(), .almost_empty(), .wr_ack(), .data_valid(), .sbiterr(), .dbiterr()
  );

  // ---------------------------------------------------------------- decoder core (clk_core)
  logic        s_axis_tready, m_axis_tvalid, m_axis_tlast;
  logic [31:0] m_axis_tdata;

  wire in_valid = ~in_empty & ~in_rd_rst_busy;
  assign in_rd_en  = in_valid & s_axis_tready;
  assign out_din   = m_axis_tdata;
  assign out_wr_en = m_axis_tvalid & ~out_full & ~out_wr_rst_busy;

  bp_stream_banked_core u_core (
      .aclk         (clk_core),
      .aresetn      (~core_rst),
      .early_exit_i (early_exit_core),
      .s_axis_tdata (in_dout),
      .s_axis_tvalid(in_valid),
      .s_axis_tready(s_axis_tready),
      .s_axis_tlast (1'b0),
      .m_axis_tdata (m_axis_tdata),
      .m_axis_tvalid(m_axis_tvalid),
      .m_axis_tready(~out_full & ~out_wr_rst_busy),
      .m_axis_tlast (m_axis_tlast)
  );

  // ---------------------------------------------------------------- clock measurement
  // free-running; the start value is irrelevant on hardware but keeps simulation out of X
  logic [31:0] core_cyc = '0, main_cyc = '0, core_cyc_main;
  always_ff @(posedge clk_core) core_cyc <= core_cyc + 32'd1;
  always_ff @(posedge clk_main) main_cyc <= main_cyc + 32'd1;
  xpm_cdc_gray #(.DEST_SYNC_FF(3), .WIDTH(32), .REG_OUTPUT(1)) u_cyc_sync (
      .src_clk(clk_core), .src_in_bin(core_cyc), .dest_clk(clk_main), .dest_out_bin(core_cyc_main)
  );

  logic core_in_reset_main;
  xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) u_rst_mon (
      .src_clk(clk_core), .src_in(core_rst), .dest_clk(clk_main), .dest_out(core_in_reset_main)
  );
  logic locked_main;
  xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) u_lock_mon (
      .src_clk(clk_ref), .src_in(mmcm_locked), .dest_clk(clk_main), .dest_out(locked_main)
  );

  // ---------------------------------------------------------------- PCIS batch path (clk_main)
  logic        p_in_wr_en, p_out_rd_en, p_commit, p_drop, ring_clearing;
  logic [31:0] p_in_din;
  wire         in_can_write = ~in_full & ~in_wr_rst_busy;
  wire         pop_ok       = ~out_empty & ~out_rd_rst_busy;

  aleph_pcis_ring #(.NS((BP_C + 31) / 32), .SLOTS(PCIS_SLOTS)) u_pcis (
      .clk(clk_main), .rst(in_fifo_rst), .enable(pcis_mode),
      .awaddr(p_awaddr), .awid(p_awid), .awsize(p_awsize), .awvalid(p_awvalid), .awready(p_awready),
      .wdata(p_wdata), .wstrb(p_wstrb), .wlast(p_wlast), .wvalid(p_wvalid), .wready(p_wready),
      .bid(p_bid), .bresp(p_bresp), .bvalid(p_bvalid), .bready(p_bready),
      .araddr(p_araddr), .arid(p_arid), .arlen(p_arlen), .arsize(p_arsize), .arvalid(p_arvalid),
      .arready(p_arready), .rid(p_rid), .rdata(p_rdata), .rresp(p_rresp), .rlast(p_rlast),
      .rvalid(p_rvalid), .rready(p_rready),
      .in_wr_en(p_in_wr_en), .in_din(p_in_din), .in_can_write(in_can_write),
      .out_valid(pop_ok), .out_dout(out_dout), .out_rd_en(p_out_rd_en),
      .commit_pulse(p_commit), .drop_pulse(p_drop), .clearing(ring_clearing)
  );

  // ---------------------------------------------------------------- AXI4-Lite slave (clk_main)
  // Write: wait until both address and data are present, take them together, answer OKAY. Only a
  // full-strobe write acts.
  logic [7:0] wr_off;
  assign wr_off = s_awaddr[7:0];
  wire  wr_go   = s_awvalid & s_wvalid & ~s_bvalid;
  wire  wr_full = wr_go & (s_wstrb == 4'hF);
  assign s_awready = wr_go;
  assign s_wready  = wr_go;
  assign s_bresp   = 2'b00;

  logic [31:0] push_words, results, pcis_acc, pcis_drop;
  logic        err_strb, err_push, err_pcis;

  wire ocl_push  = wr_full & (wr_off == 8'h10);
  wire ocl_push_ok = ocl_push & ~pcis_mode & in_can_write;
  wire ocl_ack   = wr_full & (wr_off == 8'h24) & ~pcis_mode & pop_ok & (s_wdata[1:0] == results[1:0]);

  assign in_din    = pcis_mode ? p_in_din : s_wdata;
  assign in_wr_en  = pcis_mode ? p_in_wr_en : ocl_push_ok;
  assign out_rd_en = pcis_mode ? p_out_rd_en : ocl_ack;

  always_ff @(posedge clk_main) begin
    if (!rst_main_n) begin
      s_bvalid   <= 1'b0;
      soft_rst   <= 1'b0;
      early_exit <= 1'b0;
      pcis_mode  <= 1'b0;
    end else begin
      if (wr_go) begin
        s_bvalid <= 1'b1;
        if (wr_full && wr_off == 8'h0C) begin
          soft_rst   <= s_wdata[0];
          early_exit <= s_wdata[1];
          pcis_mode  <= s_wdata[2];
        end
      end else if (s_bready) begin
        s_bvalid <= 1'b0;
      end
    end
  end

  // counters and sticky errors: cleared by soft reset (in_fifo_rst), errors also by CTRL.clear_err
  wire clear_err = wr_full & (wr_off == 8'h0C) & s_wdata[3];
  always_ff @(posedge clk_main) begin
    if (in_fifo_rst) begin
      push_words <= '0; results <= '0; pcis_acc <= '0; pcis_drop <= '0;
      err_strb <= 1'b0; err_push <= 1'b0; err_pcis <= 1'b0;
    end else begin
      if (in_wr_en)  push_words <= push_words + 1'b1;
      if (out_rd_en) results    <= results + 1'b1;
      if (p_commit)  pcis_acc   <= pcis_acc + 1'b1;
      if (p_drop)    pcis_drop  <= pcis_drop + 1'b1;
      if (clear_err) begin
        err_strb <= 1'b0; err_push <= 1'b0; err_pcis <= 1'b0;
      end else begin
        if (wr_go && s_wstrb != 4'hF)  err_strb <= 1'b1;
        if (ocl_push && !ocl_push_ok)  err_push <= 1'b1;
        if (p_drop)                    err_pcis <= 1'b1;
      end
    end
  end

  // Read: one outstanding transaction, no side effects.
  logic [7:0] rd_off;
  assign rd_off    = s_araddr[7:0];
  wire  rd_go      = s_arvalid & ~s_rvalid;
  assign s_arready = rd_go;
  assign s_rresp   = 2'b00;

  always_ff @(posedge clk_main) begin
    if (!rst_main_n) begin
      s_rvalid <= 1'b0;
      s_rdata  <= 32'd0;
    end else if (rd_go) begin
      s_rvalid <= 1'b1;
      unique case (rd_off)
        8'h00:   s_rdata <= MAGIC;
        8'h04:   s_rdata <= {16'(BP_BANK_W), 16'(BP_BANK_V)};
        8'h08:   s_rdata <= {19'd0, pcis_mode, ring_clearing, err_pcis, err_push, err_strb,
                             out_rd_rst_busy, in_wr_rst_busy, core_in_reset_main, locked_main,
                             out_empty, out_full, in_empty, in_full};
        8'h0C:   s_rdata <= {29'd0, pcis_mode, early_exit, soft_rst};
        8'h14:   s_rdata <= pop_ok ? {out_dout[31:18], results[1:0], out_dout[15:0]} : 32'hFFFF_FFFF;
        8'h18:   s_rdata <= core_cyc_main;
        8'h1C:   s_rdata <= main_cyc;
        8'h20:   s_rdata <= 32'(CORE_KHZ);
        8'h28:   s_rdata <= push_words;
        8'h2C:   s_rdata <= results;
        8'h30:   s_rdata <= pcis_acc;
        8'h34:   s_rdata <= pcis_drop;
        8'h38:   s_rdata <= VERSION;
        8'h3C:   s_rdata <= 32'(PCIS_SLOTS);
        default: s_rdata <= 32'hDEAD_BEEF;
      endcase
    end else if (s_rready) begin
      s_rvalid <= 1'b0;
    end
  end

  // m_axis_tlast and the unused upper address bits are deliberately ignored.
  /* verilator lint_off UNUSEDSIGNAL */
  wire unused = m_axis_tlast | (|s_awaddr[31:8]) | (|s_araddr[31:8]) | in_rd_rst_busy;
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
