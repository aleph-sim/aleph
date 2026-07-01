// Q6-03 (throughput) — streaming DMA datapath for the surface-code UF decoder.
//
// A pure AXI4-Stream engine: syndromes arrive on s_axis (one word per beat), each is decoded, and the
// result {obs_flip, correction} leaves on m_axis (one word per beat). No AXI4-Lite / PS-in-the-loop —
// an AXI DMA streams a whole batch from DDR through the decoder and back, so measured throughput is
// decoder-bound (one decode per the core's multi-cycle latency), not host/AXI-Lite-poll-bound.
//
// tlast is propagated input -> output: the DMA's MM2S asserts tlast on the last syndrome of the batch,
// and this engine re-emits it on the corresponding (in-order) result beat, so the S2MM channel receives
// exactly N results in one transfer (a per-beat tlast, as uf_axi_wrap emits, would end S2MM after one).
//
// Single-decode engine (the core isn't internally pipelined): s_axis_tready is asserted only when idle,
// which naturally back-pressures MM2S to the decoder's rate.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_stream_core (
  input  logic        aclk,
  input  logic        aresetn,          // active-low

  // syndrome ingress (AXI4-Stream slave)
  input  logic [31:0] s_axis_tdata,
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,
  input  logic        s_axis_tlast,

  // result egress (AXI4-Stream master): {obs_flip, correction[30:0]}
  output logic [31:0] m_axis_tdata,
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,
  output logic        m_axis_tlast
);
  localparam int SYN_W = UF_N - 1;

  logic              dec_in_valid, dec_busy, dec_out_valid, dec_obs;
  logic [SYN_W-1:0]  dec_syndrome;
  logic [UF_M-1:0]   dec_corr;
  logic [15:0]       dec_lat;

  logic [SYN_W-1:0]  syndrome_reg;
  assign dec_syndrome = syndrome_reg;

  uf_surface_decoder u_dec (
    .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid), .syndrome(dec_syndrome),
    .busy(dec_busy), .out_valid(dec_out_valid), .correction(dec_corr),
    .obs_flip(dec_obs), .latency_cycles(dec_lat)
  );

  typedef enum logic [1:0] { S_IDLE, S_RUN, S_OUT } st_t;
  st_t st;

  logic        frame_last;      // latched s_axis_tlast for the in-flight syndrome
  logic        res_obs;
  logic [30:0] res_corr;

  assign s_axis_tready = (st == S_IDLE);
  assign m_axis_tvalid = (st == S_OUT);
  assign m_axis_tdata  = {res_obs, res_corr};
  assign m_axis_tlast  = frame_last;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      st           <= S_IDLE;
      dec_in_valid <= 1'b0;
      frame_last   <= 1'b0;
      res_obs      <= 1'b0;
      res_corr     <= '0;
    end else begin
      dec_in_valid <= 1'b0;                       // default: one-cycle pulse
      unique case (st)
        S_IDLE: begin
          if (s_axis_tvalid) begin                // tready is high in S_IDLE -> beat accepted
            syndrome_reg <= s_axis_tdata[SYN_W-1:0];
            frame_last   <= s_axis_tlast;
            dec_in_valid <= 1'b1;                 // asserted next cycle (S_RUN) with syndrome_reg stable
            st           <= S_RUN;
          end
        end
        S_RUN: begin
          if (dec_out_valid) begin
            res_obs  <= dec_obs;
            res_corr <= 31'(dec_corr);   // low 31 correction bits (zero-extended for M<31)
            st       <= S_OUT;
          end
        end
        S_OUT: begin
          if (m_axis_tready) st <= S_IDLE;        // result consumed by S2MM
        end
        default: st <= S_IDLE;
      endcase
    end
  end

endmodule
