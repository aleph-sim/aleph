// Q6-03 (throughput) / Q6-19 (3D) — streaming DMA datapath for the surface-code UF decoder.
//
// A pure AXI4-Stream engine: syndromes arrive on s_axis, each is decoded, and the result
// {obs_flip, correction} leaves on m_axis (one word per beat). No AXI4-Lite / PS-in-the-loop —
// an AXI DMA streams a whole batch from DDR through the decoder and back, so measured throughput is
// decoder-bound, not host/AXI-Lite-poll-bound.
//
// The syndrome can be wider than the 32-bit stream word (d5×3 / d7 have SYN_W=48): it arrives over
// NBEATS = ceil(SYN_W/32) input beats, little-endian (beat 0 = syndrome[31:0]), reassembled here.
// So MM2S sends NBEATS*N words and S2MM receives N result words. tlast is propagated from the LAST
// input beat of the batch to its (in-order) result beat so one DMA transfer streams the whole batch
// (a per-beat tlast would end S2MM after one word). For SYN_W<=32 (NBEATS=1) this reduces to the
// original single-beat behaviour.
//
// Single-decode engine (the core isn't internally pipelined): s_axis_tready is asserted only while
// accumulating, which back-pressures MM2S to the decoder's rate.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_stream_core (
  input  logic        aclk,
  input  logic        aresetn,          // active-low

  input  logic [31:0] s_axis_tdata,
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,
  input  logic        s_axis_tlast,

  output logic [31:0] m_axis_tdata,     // {obs_flip, correction[30:0]}
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,
  output logic        m_axis_tlast
);
  localparam int SYN_W  = UF_N - 1;
  localparam int NBEATS = (SYN_W + 31) / 32;                 // stream words per syndrome
  localparam int BCW    = (NBEATS <= 1) ? 1 : $clog2(NBEATS);

  logic              dec_in_valid, dec_busy, dec_out_valid, dec_obs;
  logic [SYN_W-1:0]  dec_syndrome;
  logic [UF_M-1:0]   dec_corr;
  logic [15:0]       dec_lat;

  logic [NBEATS*32-1:0] syn_acc;                             // reassembled syndrome bits
  assign dec_syndrome = syn_acc[SYN_W-1:0];

  uf_surface_decoder u_dec (
    .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid), .syndrome(dec_syndrome),
    .busy(dec_busy), .out_valid(dec_out_valid), .correction(dec_corr),
    .obs_flip(dec_obs), .latency_cycles(dec_lat)
  );

  typedef enum logic [1:0] { S_ACC, S_RUN, S_OUT } st_t;
  st_t st;

  logic [BCW-1:0] beat_cnt;
  logic           frame_last;
  logic           res_obs;
  logic [30:0]    res_corr;

  assign s_axis_tready = (st == S_ACC);
  assign m_axis_tvalid = (st == S_OUT);
  assign m_axis_tdata  = {res_obs, res_corr};
  assign m_axis_tlast  = frame_last;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      st           <= S_ACC;
      dec_in_valid <= 1'b0;
      beat_cnt     <= '0;
      frame_last   <= 1'b0;
      res_obs      <= 1'b0;
      res_corr     <= '0;
    end else begin
      dec_in_valid <= 1'b0;                                  // default: one-cycle pulse
      unique case (st)
        S_ACC: begin
          if (s_axis_tvalid) begin                           // tready high in S_ACC -> beat accepted
            syn_acc[beat_cnt*32 +: 32] <= s_axis_tdata;
            if (beat_cnt == BCW'(NBEATS - 1)) begin          // last beat of this syndrome
              beat_cnt     <= '0;
              frame_last   <= s_axis_tlast;
              dec_in_valid <= 1'b1;                           // fire next cycle with syn_acc settled
              st           <= S_RUN;
            end else begin
              beat_cnt <= beat_cnt + 1'b1;
            end
          end
        end
        S_RUN: begin
          if (dec_out_valid) begin
            res_obs  <= dec_obs;
            res_corr <= dec_corr[30:0];
            st       <= S_OUT;
          end
        end
        S_OUT: begin
          if (m_axis_tready) st <= S_ACC;                    // result consumed by S2MM
        end
        default: st <= S_ACC;
      endcase
    end
  end

endmodule
