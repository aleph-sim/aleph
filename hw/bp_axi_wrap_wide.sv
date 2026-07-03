// Q7-02 M5-followup — WIDE PS<->PL AXI4-Lite wrapper for the M2 sequential relay-BP decoder.
//
// The board wrapper `bp_axi_wrap.sv` hard-codes the code-capacity gross sizes (72-bit syndrome / 144-bit
// correction) around the partial decoder. This one is GENERIC in the graph size — it derives the number
// of 32-bit syndrome / correction words from `BP_C` / `BP_N` in the included header — so the SAME wrapper
// serves the code-capacity graph AND the far larger, irregular **circuit-level** graph (rounds=1: 144
// checks / 864 vars → 5 syndrome words / 27 correction words). It wraps the graph-generic **M2 sequential
// decoder** (`bp_relay_decoder`, runtime node cursor), the only variant that handles the circuit graph
// (the unrolled/partial variants bake the graph in and would not fit).
//
// This is the path for the first circuit-level qLDPC decode on the Arty (xc7z020) — a slow but correct
// M2 over AXI4-Lite; value is on-silicon correctness + per-decode latency, not throughput.
//
// Register map (AXI4-Lite, 32-bit data, byte addresses; NS = ceil(BP_C/32), NC = ceil(BP_N/32)):
//   0x00 CTRL     [W]  bit0 START (self-clearing)
//   0x04 STATUS   [R]  bit0 BUSY, bit1 DONE (sticky), bit2 VALID (=valid_flag)
//   0x08 LATENCY  [R]  last decode latency in cycles (32-bit — the circuit M2 decode is ~70k cycles)
//   0x0C OBS      [R]  obs_flip[BP_OBS-1:0]
//   0x10 IDCODE   [R]  0x4250_0002 ('BP', v2 — the wide/M2 wrapper)
//   0x40 SYND0..  [RW] syndrome, NS words (word i = syndrome[i*32 +: 32]); low BP_C bits used
//   0x80 CORR0..  [R]  correction, NC words (word i = correction[i*32 +: 32]); low BP_N bits used

`timescale 1ns / 1ps
`include "bb_gross_tanner.svh"

module bp_axi_wrap_wide #(
  parameter int C_ADDR_W = 8                    // 256-byte register space (control + SYND + CORR words)
)(
  input  logic                 aclk,
  input  logic                 aresetn,

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
  input  logic                 s_axil_rready
);
  localparam int NS = (BP_C + 31) / 32;         // syndrome words
  localparam int NC = (BP_N + 31) / 32;         // correction words
  localparam logic [31:0] IDCODE = 32'h4250_0002;

  // word-address decode (byte addr -> word index a[7:2])
  localparam logic [5:0] A_CTRL = 6'h00, A_STATUS = 6'h01, A_LAT = 6'h02, A_OBS = 6'h03, A_ID = 6'h04;
  localparam logic [5:0] SYND_BASE = 6'h10;     // byte 0x40
  localparam logic [5:0] CORR_BASE = 6'h20;     // byte 0x80
  function automatic logic [5:0] word(input logic [C_ADDR_W-1:0] a);
    return a[7:2];
  endfunction

  // ---- decoder core (M2 sequential) ----
  logic               dec_in_valid, dec_busy, dec_out_valid, dec_vflag;
  logic               dec_syndrome [BP_C];
  logic               dec_corr     [BP_N];
  logic [BP_OBS-1:0]  dec_obs;
  logic [31:0]        dec_lat;

  bp_relay_decoder u_dec (
    .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid),
    .syndrome_in(dec_syndrome), .busy(dec_busy), .out_valid(dec_out_valid),
    .corr_out(dec_corr), .obs_flip(dec_obs), .valid_flag(dec_vflag),
    .latency_cycles(dec_lat)
  );

  // ---- registers / state (SYND/CORR padded up to a 32-bit-word multiple) ----
  logic [NS*32-1:0]   reg_synd;
  logic [NC*32-1:0]   reg_corr;
  logic [BP_OBS-1:0]  reg_obs;
  logic [31:0]        reg_lat;
  logic               reg_vflag, busy, done;

  for (genvar i = 0; i < BP_C; i++) assign dec_syndrome[i] = reg_synd[i];

  typedef enum logic [0:0] { O_IDLE, O_RUN } owner_t;
  owner_t ostate;

  logic awready_r, wready_r, bvalid_r, arready_r, rvalid_r;
  logic [31:0] rdata_r;

  wire lite_start = awready_r && wready_r && (word(s_axil_awaddr) == A_CTRL) && s_axil_wdata[0];

  assign s_axil_awready = awready_r;
  assign s_axil_wready  = wready_r;
  assign s_axil_bvalid  = bvalid_r;
  assign s_axil_bresp   = 2'b00;
  assign s_axil_arready = arready_r;
  assign s_axil_rvalid  = rvalid_r;
  assign s_axil_rresp   = 2'b00;
  assign s_axil_rdata   = rdata_r;

  // Combinational read mux (control regs + the SYND/CORR word regions).
  function automatic logic [31:0] read_word(input logic [5:0] w);
    read_word = 32'hDEAD_BEEF;
    case (w)
      A_CTRL:   read_word = 32'h0;
      A_STATUS: read_word = {29'h0, reg_vflag, done, busy};
      A_LAT:    read_word = reg_lat;
      A_OBS:    read_word = {{(32-BP_OBS){1'b0}}, reg_obs};
      A_ID:     read_word = IDCODE;
      default: begin
        for (int i = 0; i < NS; i++) if (w == SYND_BASE + 6'(i)) read_word = reg_synd[i*32 +: 32];
        for (int i = 0; i < NC; i++) if (w == CORR_BASE + 6'(i)) read_word = reg_corr[i*32 +: 32];
      end
    endcase
  endfunction

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      ostate <= O_IDLE; dec_in_valid <= 1'b0;
      reg_synd <= '0; reg_corr <= '0; reg_obs <= '0; reg_lat <= '0; reg_vflag <= 1'b0;
      busy <= 1'b0; done <= 1'b0;
      awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
      arready_r <= 1'b0; rvalid_r <= 1'b0; rdata_r <= '0;
    end else begin
      dec_in_valid <= 1'b0;

      // ---------------- AXI4-Lite WRITE ----------------
      if (awready_r) awready_r <= 1'b0;
      if (wready_r)  wready_r  <= 1'b0;
      if (!awready_r && !wready_r && !bvalid_r && s_axil_awvalid && s_axil_wvalid) begin
        awready_r <= 1'b1; wready_r <= 1'b1; bvalid_r <= 1'b1;
        for (int i = 0; i < NS; i++)
          if (word(s_axil_awaddr) == SYND_BASE + 6'(i)) reg_synd[i*32 +: 32] <= s_axil_wdata;
      end
      if (bvalid_r && s_axil_bready) bvalid_r <= 1'b0;

      // ---------------- AXI4-Lite READ ----------------
      if (arready_r) arready_r <= 1'b0;
      if (!arready_r && !rvalid_r && s_axil_arvalid) begin
        arready_r <= 1'b1; rvalid_r <= 1'b1;
        rdata_r <= read_word(word(s_axil_araddr));
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
