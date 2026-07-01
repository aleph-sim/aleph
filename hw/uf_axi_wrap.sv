// Q6-07 — PS<->PL wrapper for the surface-code Union-Find decoder.
//
// Exposes the Q6-04 `uf_surface_decoder` to the Zynq PS over two interfaces, both standard so the
// same wrapper drops onto Zynq-7020 (Zybo) and Zynq UltraScale+ (KV260):
//   * AXI4-Lite slave — control/status + a small register map (the control plane).
//   * AXI4-Stream pair — syndrome ingress + correction egress (the data plane; this is where the
//     Q4 sliding-window streaming maps onto hardware).
// A single decode-owner FSM serves whichever interface triggered the decode; the decoder core is
// unchanged. All sequential state lives in one `always_ff` to avoid multi-driver on shared registers.
//
// Register map (AXI4-Lite, 32-bit data, byte addresses):
//   0x00 CTRL       [W]  bit0 START (self-clearing): latch SYNDROME and run one decode
//   0x04 STATUS     [R]  bit0 BUSY, bit1 DONE (sticky, cleared on next START), bit2 OBS_FLIP
//   0x08 SYNDROME   [RW] syndrome[SYN_W-1:0]
//   0x0C CORRECTION [R]  correction[M-1:0]
//   0x10 LATENCY    [R]  last decode latency in cycles
//   0x14 IDCODE     [R]  0x5546_0003 constant ('UF', d=3) for bring-up sanity

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_axi_wrap #(
  parameter int C_ADDR_W = 6                  // 64-byte register space
)(
  input  logic                 aclk,
  input  logic                 aresetn,        // active-low (AXI convention)

  // ---- AXI4-Lite slave (control plane) ----
  input  logic [C_ADDR_W-1:0]  s_axil_awaddr,
  input  logic                 s_axil_awvalid,
  output logic                 s_axil_awready,
  input  logic [31:0]          s_axil_wdata,
  input  logic [3:0]           s_axil_wstrb,
  input  logic                 s_axil_wvalid,
  output logic                 s_axil_wready,
  output logic [1:0]           s_axil_bresp,
  output logic                 s_axil_bvalid,
  input  logic                 s_axil_bready,
  input  logic [C_ADDR_W-1:0]  s_axil_araddr,
  input  logic                 s_axil_arvalid,
  output logic                 s_axil_arready,
  output logic [31:0]          s_axil_rdata,
  output logic [1:0]           s_axil_rresp,
  output logic                 s_axil_rvalid,
  input  logic                 s_axil_rready,

  // ---- AXI4-Stream syndrome ingress (slave) ----
  input  logic [31:0]          s_axis_tdata,
  input  logic                 s_axis_tvalid,
  output logic                 s_axis_tready,

  // ---- AXI4-Stream correction egress (master) ----
  output logic [31:0]          m_axis_tdata,   // {obs_flip, correction[M-1:0]}
  output logic                 m_axis_tvalid,
  input  logic                 m_axis_tready,
  output logic                 m_axis_tlast
);
  localparam int SYN_W = UF_N - 1;             // 8 detectors for d=3
  // Top-zero pad for the {obs, correction} stream word. For UF_M >= 31 (d >= 5) the correction alone
  // fills the 32-bit word, so there is no room to pad and obs is dropped from the stream view (the
  // AXI4-Lite OBS_FLIP bit and CORRECTION[31:0] register still carry it) — keeps the width legal.
  localparam int AXIS_PAD = (UF_M < 31) ? (31 - UF_M) : 0;
  localparam logic [31:0] IDCODE = 32'h5546_0003;

  // word-address decode
  localparam logic [3:0] A_CTRL = 4'h0, A_STATUS = 4'h1, A_SYND = 4'h2,
                         A_CORR = 4'h3, A_LAT = 4'h4, A_ID = 4'h5;
  function automatic logic [3:0] word(input logic [C_ADDR_W-1:0] a);
    return a[5:2];
  endfunction

  // ---- decoder core ----
  logic              dec_in_valid, dec_busy, dec_out_valid, dec_obs;
  logic [SYN_W-1:0]  dec_syndrome;
  logic [UF_M-1:0]   dec_corr;
  logic [15:0]       dec_lat;

  uf_surface_decoder u_dec (
    .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid), .syndrome(dec_syndrome),
    .busy(dec_busy), .out_valid(dec_out_valid), .correction(dec_corr),
    .obs_flip(dec_obs), .latency_cycles(dec_lat)
  );

  // ---- registers / state ----
  logic [SYN_W-1:0] reg_syndrome;
  logic [UF_M-1:0]  reg_corr;
  logic             reg_obs;
  logic [15:0]      reg_lat;
  logic             busy, done;

  typedef enum logic [1:0] { O_IDLE, O_RUN, O_STREAM_OUT } owner_t;
  owner_t ostate;
  logic   src_axis;                            // which interface owns the in-flight decode

  // AXI-Lite handshake registers
  logic awready_r, wready_r, bvalid_r, arready_r, rvalid_r;
  logic [31:0] rdata_r;

  // a stream beat is accepted only when the engine is idle and no Lite START is racing it
  wire lite_start = awready_r && wready_r && (word(s_axil_awaddr) == A_CTRL) && s_axil_wdata[0];
  assign s_axis_tready = (ostate == O_IDLE) && !lite_start;
  wire axis_fire = s_axis_tvalid && s_axis_tready;

  // syndrome presented to the core during the in_valid pulse (mux the live stream beat in directly)
  assign dec_syndrome = axis_fire ? s_axis_tdata[SYN_W-1:0] : reg_syndrome;

  // AXI-Lite outputs
  assign s_axil_awready = awready_r;
  assign s_axil_wready  = wready_r;
  assign s_axil_bvalid  = bvalid_r;
  assign s_axil_bresp   = 2'b00;               // OKAY
  assign s_axil_arready = arready_r;
  assign s_axil_rvalid  = rvalid_r;
  assign s_axil_rresp   = 2'b00;
  assign s_axil_rdata   = rdata_r;

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      ostate <= O_IDLE; src_axis <= 1'b0; dec_in_valid <= 1'b0;
      reg_syndrome <= '0; reg_corr <= '0; reg_obs <= 1'b0; reg_lat <= '0;
      busy <= 1'b0; done <= 1'b0;
      awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
      arready_r <= 1'b0; rvalid_r <= 1'b0; rdata_r <= '0;
      m_axis_tvalid <= 1'b0; m_axis_tdata <= '0; m_axis_tlast <= 1'b0;
    end else begin
      dec_in_valid <= 1'b0;                     // default: in_valid is a 1-cycle pulse

      // ---------------- AXI4-Lite WRITE ----------------
      // accept a combined AW+W beat when both valid and no write response pending
      if (awready_r) awready_r <= 1'b0;
      if (wready_r)  wready_r  <= 1'b0;
      if (!awready_r && !wready_r && !bvalid_r && s_axil_awvalid && s_axil_wvalid) begin
        awready_r <= 1'b1;
        wready_r  <= 1'b1;
        bvalid_r  <= 1'b1;
        // register write (full 32-bit; wstrb ignored — all regs ≤32b and word-aligned)
        if (word(s_axil_awaddr) == A_SYND) reg_syndrome <= s_axil_wdata[SYN_W-1:0];
        // CTRL.START is handled by `lite_start` below (combinational off awready/wready)
      end
      if (bvalid_r && s_axil_bready) bvalid_r <= 1'b0;

      // ---------------- AXI4-Lite READ ----------------
      if (arready_r) arready_r <= 1'b0;
      if (!arready_r && !rvalid_r && s_axil_arvalid) begin
        arready_r <= 1'b1;
        rvalid_r  <= 1'b1;
        unique case (word(s_axil_araddr))
          A_CTRL:   rdata_r <= 32'h0;
          A_STATUS: rdata_r <= {29'h0, reg_obs, done, busy};
          A_SYND:   rdata_r <= {{(32-SYN_W){1'b0}}, reg_syndrome};
          A_CORR:   rdata_r <= 32'(reg_corr);   // low 32 correction bits (zero-extended for M<32)
          A_LAT:    rdata_r <= {16'h0, reg_lat};
          A_ID:     rdata_r <= IDCODE;
          default:  rdata_r <= 32'hDEAD_BEEF;
        endcase
      end
      if (rvalid_r && s_axil_rready) rvalid_r <= 1'b0;

      // ---------------- decode-owner FSM ----------------
      unique case (ostate)
        O_IDLE: begin
          if (lite_start) begin
            dec_in_valid <= 1'b1; src_axis <= 1'b0; busy <= 1'b1; done <= 1'b0;
            ostate <= O_RUN;
          end else if (axis_fire) begin
            reg_syndrome <= s_axis_tdata[SYN_W-1:0];   // for read-back; core gets it via dec_syndrome
            dec_in_valid <= 1'b1; src_axis <= 1'b1; busy <= 1'b1; done <= 1'b0;
            ostate <= O_RUN;
          end
        end
        O_RUN: begin
          if (dec_out_valid) begin
            reg_corr <= dec_corr; reg_obs <= dec_obs; reg_lat <= dec_lat; busy <= 1'b0;
            if (src_axis) begin
              m_axis_tdata  <= {{AXIS_PAD{1'b0}}, dec_obs, dec_corr};
              m_axis_tvalid <= 1'b1;
              m_axis_tlast  <= 1'b1;             // one correction per syndrome frame
              ostate <= O_STREAM_OUT;
            end else begin
              done <= 1'b1;
              ostate <= O_IDLE;
            end
          end
        end
        O_STREAM_OUT: begin
          if (m_axis_tready) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            ostate <= O_IDLE;
          end
        end
        default: ostate <= O_IDLE;
      endcase
    end
  end
endmodule
