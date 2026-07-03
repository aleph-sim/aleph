// Q7-02 board bring-up — PS<->PL AXI4-Lite wrapper for the partially-unrolled relay-BP decoder.
//
// Exposes `bp_relay_partial` (the Arty-fit 12/24 partial-unroll decoder, #441) to the Zynq PS over an
// AXI4-Lite control plane. This is the code-capacity gross-BB decoder — one syndrome in, one correction
// out — so, unlike the UF throughput path (Q6, AXI-Stream+DMA), the value here is CORRECTNESS (bit-exact
// vs the golden) + per-decode LATENCY, not decode rate. AXI4-Lite is the right (and simplest) interface:
// same bring-up shape as `uf_axi_wrap.sv`/`uf_pynq.py`, no DMA framing.
//
// The gross code is wider than one 32-bit word, so the register map spreads the ports across words:
//   syndrome  (BP_C  = 72 bits) -> 3 write words (SYND0..2)
//   correction(BP_N  = 144 bits)-> 5 read  words (CORR0..4)
//   obs_flip  (BP_OBS= 12 bits) -> 1 read  word
//
// Register map (AXI4-Lite, 32-bit data, byte addresses):
//   0x00 CTRL     [W]  bit0 START (self-clearing): latch SYND* and run one decode
//   0x04 STATUS   [R]  bit0 BUSY, bit1 DONE (sticky, cleared on next START), bit2 VALID (=valid_flag)
//   0x08 SYND0    [RW] syndrome[31:0]
//   0x0C SYND1    [RW] syndrome[63:32]
//   0x10 SYND2    [RW] syndrome[71:64]  (low 8 bits used)
//   0x14 CORR0    [R]  correction[31:0]
//   0x18 CORR1    [R]  correction[63:32]
//   0x1C CORR2    [R]  correction[95:64]
//   0x20 CORR3    [R]  correction[127:96]
//   0x24 CORR4    [R]  correction[143:128] (low 16 bits used)
//   0x28 OBS      [R]  obs_flip[11:0]
//   0x2C LATENCY  [R]  last decode latency in cycles
//   0x30 IDCODE   [R]  0x4250_0001 constant ('BP', v1) for PS<->PL bring-up sanity
//
// All sequential state is in one always_ff (no multi-driver); the decoder core is instantiated below and
// left unchanged. Synchronous-in-AXI reset convention (active-low aresetn).

`timescale 1ns / 1ps
`include "bb_gross_tanner.svh"

module bp_axi_wrap #(
  parameter int C_ADDR_W = 6                    // 64-byte register space (words 0x00..0x30)
)(
  input  logic                 aclk,
  input  logic                 aresetn,          // active-low (AXI convention)

  // ---- AXI4-Lite slave (control plane) ----
  input  logic [C_ADDR_W-1:0]  s_axil_awaddr,
  input  logic                 s_axil_awvalid,
  output logic                 s_axil_awready,
  input  logic [31:0]          s_axil_wdata,
  input  logic [3:0]           s_axil_wstrb,
  input  logic                 s_axil_wvalid,
  output logic [1:0]           s_axil_bresp,
  output logic                 s_axil_bvalid,
  input  logic                 s_axil_bready,
  output logic                 s_axil_wready,
  input  logic [C_ADDR_W-1:0]  s_axil_araddr,
  input  logic                 s_axil_arvalid,
  output logic                 s_axil_arready,
  output logic [31:0]          s_axil_rdata,
  output logic [1:0]           s_axil_rresp,
  output logic                 s_axil_rvalid,
  input  logic                 s_axil_rready
);
  localparam logic [31:0] IDCODE = 32'h4250_0001;   // 'BP' v1

  // word-address decode (byte addr -> 4-bit word index)
  localparam logic [3:0] A_CTRL  = 4'h0, A_STATUS = 4'h1,
                         A_SYND0 = 4'h2, A_SYND1  = 4'h3, A_SYND2 = 4'h4,
                         A_CORR0 = 4'h5, A_CORR1  = 4'h6, A_CORR2 = 4'h7,
                         A_CORR3 = 4'h8, A_CORR4  = 4'h9,
                         A_OBS   = 4'hA, A_LAT    = 4'hB, A_ID = 4'hC;
  function automatic logic [3:0] word(input logic [C_ADDR_W-1:0] a);
    return a[5:2];
  endfunction

  // ---- decoder core (ports are unpacked bit arrays; the wrapper packs/unpacks) ----
  logic               dec_in_valid, dec_busy, dec_out_valid, dec_vflag;
  logic               dec_syndrome [BP_C];
  logic               dec_corr     [BP_N];
  logic [BP_OBS-1:0]  dec_obs;
  logic [15:0]        dec_lat;

  bp_relay_partial u_dec (
    .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid),
    .syndrome_in(dec_syndrome), .busy(dec_busy), .out_valid(dec_out_valid),
    .corr_out(dec_corr), .obs_flip(dec_obs), .valid_flag(dec_vflag),
    .latency_cycles(dec_lat)
  );

  // ---- registers / state ----
  logic [71:0]        reg_synd;                  // packed syndrome (drives the unpacked core port)
  logic [143:0]       reg_corr;                  // latched correction
  logic [BP_OBS-1:0]  reg_obs;
  logic [15:0]        reg_lat;
  logic               reg_vflag;
  logic               busy, done;

  // packed syndrome register -> unpacked decoder port (constant per-bit fan-out, purely combinational)
  for (genvar i = 0; i < BP_C; i++) assign dec_syndrome[i] = reg_synd[i];

  typedef enum logic [0:0] { O_IDLE, O_RUN } owner_t;
  owner_t ostate;

  // AXI-Lite handshake registers
  logic awready_r, wready_r, bvalid_r, arready_r, rvalid_r;
  logic [31:0] rdata_r;

  // START is a CTRL write with bit0 set, detected the cycle the AW+W beat is accepted
  wire lite_start = awready_r && wready_r && (word(s_axil_awaddr) == A_CTRL) && s_axil_wdata[0];

  // AXI-Lite outputs
  assign s_axil_awready = awready_r;
  assign s_axil_wready  = wready_r;
  assign s_axil_bvalid  = bvalid_r;
  assign s_axil_bresp   = 2'b00;                 // OKAY
  assign s_axil_arready = arready_r;
  assign s_axil_rvalid  = rvalid_r;
  assign s_axil_rresp    = 2'b00;
  assign s_axil_rdata   = rdata_r;

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      ostate <= O_IDLE; dec_in_valid <= 1'b0;
      reg_synd <= '0; reg_corr <= '0; reg_obs <= '0; reg_lat <= '0; reg_vflag <= 1'b0;
      busy <= 1'b0; done <= 1'b0;
      awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
      arready_r <= 1'b0; rvalid_r <= 1'b0; rdata_r <= '0;
    end else begin
      dec_in_valid <= 1'b0;                       // default: in_valid is a 1-cycle pulse

      // ---------------- AXI4-Lite WRITE (combined AW+W beat) ----------------
      if (awready_r) awready_r <= 1'b0;
      if (wready_r)  wready_r  <= 1'b0;
      if (!awready_r && !wready_r && !bvalid_r && s_axil_awvalid && s_axil_wvalid) begin
        awready_r <= 1'b1;
        wready_r  <= 1'b1;
        bvalid_r  <= 1'b1;
        // register write (full 32-bit; wstrb ignored — all regs word-aligned)
        case (word(s_axil_awaddr))
          A_SYND0: reg_synd[31:0]  <= s_axil_wdata;
          A_SYND1: reg_synd[63:32] <= s_axil_wdata;
          A_SYND2: reg_synd[71:64] <= s_axil_wdata[7:0];
          default: ;                              // CTRL.START handled by `lite_start` below
        endcase
      end
      if (bvalid_r && s_axil_bready) bvalid_r <= 1'b0;

      // ---------------- AXI4-Lite READ ----------------
      if (arready_r) arready_r <= 1'b0;
      if (!arready_r && !rvalid_r && s_axil_arvalid) begin
        arready_r <= 1'b1;
        rvalid_r  <= 1'b1;
        case (word(s_axil_araddr))
          A_CTRL:   rdata_r <= 32'h0;
          A_STATUS: rdata_r <= {29'h0, reg_vflag, done, busy};
          A_SYND0:  rdata_r <= reg_synd[31:0];
          A_SYND1:  rdata_r <= reg_synd[63:32];
          A_SYND2:  rdata_r <= {24'h0, reg_synd[71:64]};
          A_CORR0:  rdata_r <= reg_corr[31:0];
          A_CORR1:  rdata_r <= reg_corr[63:32];
          A_CORR2:  rdata_r <= reg_corr[95:64];
          A_CORR3:  rdata_r <= reg_corr[127:96];
          A_CORR4:  rdata_r <= {16'h0, reg_corr[143:128]};
          A_OBS:    rdata_r <= {{(32-BP_OBS){1'b0}}, reg_obs};
          A_LAT:    rdata_r <= {16'h0, reg_lat};
          A_ID:     rdata_r <= IDCODE;
          default:  rdata_r <= 32'hDEAD_BEEF;
        endcase
      end
      if (rvalid_r && s_axil_rready) rvalid_r <= 1'b0;

      // ---------------- decode-owner FSM ----------------
      case (ostate)
        O_IDLE: begin
          if (lite_start) begin
            dec_in_valid <= 1'b1; busy <= 1'b1; done <= 1'b0;
            ostate <= O_RUN;
          end
        end
        O_RUN: begin
          if (dec_out_valid) begin
            for (int v = 0; v < BP_N; v++) reg_corr[v] <= dec_corr[v];
            reg_obs <= dec_obs; reg_lat <= dec_lat; reg_vflag <= dec_vflag;
            busy <= 1'b0; done <= 1'b1;
            ostate <= O_IDLE;
          end
        end
        default: ostate <= O_IDLE;
      endcase
    end
  end
endmodule
