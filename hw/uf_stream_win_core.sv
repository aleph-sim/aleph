// Q6-20 (on silicon) — AXI4-Stream front-end for the SLIDING-WINDOW STREAMING decoder.
//
// Q6-19 put the *block* decoder on the Arty over an AXI DMA (`uf_stream_core` -> `uf_surface_decoder`,
// one full syndrome in / one result out). The streaming decoder (`uf_streaming_decoder`, Q6-20) has a
// different shape: it consumes an unbounded stream of measurement ROUNDS one round at a time and emits
// one result per *committed window* (every `C` rounds), decoding in bounded O(W) memory. This wrapper
// carries that round-handshake over the same AXI4-Stream DMA path so the streaming decoder runs on
// silicon with no change to the block design's DMA plumbing.
//
// Framing:
//   * input  — each 32-bit MM2S beat carries one round's detector bits in [UF_DPR-1:0] (UF_DPR<=32 at
//     these distances; the upper bits are ignored). s_axis back-pressures whenever the decoder is not
//     accepting a round (busy mid-window) OR the single result slot is still occupied, so the DMA rate
//     is bounded by the decoder and no window output is ever dropped.
//   * output — one 32-bit S2MM beat per committed window: {out_obs, residual_empty, 0.., latency[15:0]}
//     (bit31 = committed logical parity, matching the block build's obs bit; bit30 = residual-empty /
//     validity-drain flag; low 16 = the window's core decode latency in cycles). tlast is propagated
//     from the last MM2S beat of the batch to the window it completes, so one DMA transfer streams the
//     whole round batch and S2MM receives exactly one word per window.
//
// A real experiment sends R = W + k*C rounds (k windows after warm-up), so the final round lands on a
// window boundary and its tlast tags that last window's result.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_stream_win_core (
  input  logic        aclk,
  input  logic        aresetn,          // active-low

  /* verilator lint_off UNUSEDSIGNAL */ // only [UF_DPR-1:0] of tdata carries a round; upper bits ignored
  input  logic [31:0] s_axis_tdata,     // [UF_DPR-1:0] = this round's detector bits
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,
  input  logic        s_axis_tlast,

  output logic [31:0] m_axis_tdata,     // {obs, residual_empty, ..., latency[15:0]}
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,
  output logic        m_axis_tlast
);
  // ---- sliding-window streaming decoder (round in, one result per committed window out) ----
  logic               dec_in_valid;
  logic [UF_DPR-1:0]  dec_in_round;
  logic               dec_in_ready;
  logic               dec_out_valid, dec_out_obs, dec_res_empty;
  logic [15:0]        dec_last_lat;

  uf_streaming_decoder u_stream (
    .clk(aclk), .rst_n(aresetn),
    .in_valid(dec_in_valid), .in_round(dec_in_round), .in_ready(dec_in_ready),
    .out_valid(dec_out_valid), .out_obs(dec_out_obs),
    .last_latency(dec_last_lat), .residual_empty(dec_res_empty)
  );

  // ---- single result slot (1-deep). We gate the input on it being free, so the decoder cannot retire
  // a new window until S2MM has consumed the previous one — a 1-deep buffer suffices, no overflow. ----
  logic         out_full;
  logic [31:0]  out_word;
  logic         out_last;

  // A round beat is consumed only when the decoder wants one AND the result slot is free.
  assign s_axis_tready = dec_in_ready & ~out_full;
  assign dec_in_valid  = s_axis_tvalid & s_axis_tready;
  /* verilator lint_off UNUSEDSIGNAL */          // only the low UF_DPR detector bits carry a round
  assign dec_in_round  = s_axis_tdata[UF_DPR-1:0];
  /* verilator lint_on UNUSEDSIGNAL */

  assign m_axis_tvalid = out_full;
  assign m_axis_tdata  = out_word;
  assign m_axis_tlast  = out_last;

  // tlast rides the *last* MM2S beat of the batch; latch it and tag the window that beat completes.
  logic pending_last;

  // Async reset, matching the wrapped `uf_streaming_decoder` (avoids a mixed sync/async reset net).
  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      out_full     <= 1'b0;
      out_word     <= '0;
      out_last     <= 1'b0;
      pending_last <= 1'b0;
    end else begin
      // remember the end-of-batch marker on the final accepted round beat
      if (dec_in_valid && s_axis_tlast)
        pending_last <= 1'b1;

      // a committed window arrives: latch its result into the free slot and tag end-of-batch
      if (dec_out_valid) begin
        out_word     <= {dec_out_obs, dec_res_empty, 14'b0, dec_last_lat};
        out_last     <= pending_last;
        out_full     <= 1'b1;
        pending_last <= 1'b0;
      end else if (out_full && m_axis_tready) begin
        // S2MM consumed the result; free the slot
        out_full <= 1'b0;
        out_last <= 1'b0;
      end
    end
  end

endmodule
