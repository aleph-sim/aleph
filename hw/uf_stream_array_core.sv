// Q6-03 (throughput scaling) — K-way replicated streaming decoder array.
//
// Same AXI4-Stream interface as uf_stream_core, but with K decoder cores in parallel to multiply
// aggregate throughput on the ~98 %-free XC7Z020 fabric. One DMA stream in / one out — no block-design
// change, just a different module reference. A round-robin dispatcher hands each incoming syndrome to
// the next engine; a round-robin collector reads results back IN THE SAME ORDER, so output beat i
// corresponds to input beat i (and the batch's tlast, latched with its syndrome, is re-emitted on the
// matching output beat for a clean bulk S2MM). Each engine is single-buffered; s_axis_tready gates on
// the next-to-dispatch engine being idle, which back-pressures the DMA to the array's aggregate rate.
//
// Round-robin (not out-of-order) keeps ordering trivial and correct; a rare heavy syndrome briefly
// head-of-line-blocks its slot, which is fine sub-threshold. Throughput ~= K x single-engine until the
// 1-per-cycle dispatch/collect serialisation binds (K decodes per ~decode-latency cycles).

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_stream_array_core #(
  parameter int K = 8                          // number of parallel decoder engines
)(
  input  logic        aclk,
  input  logic        aresetn,

  input  logic [31:0] s_axis_tdata,
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,
  input  logic        s_axis_tlast,

  output logic [31:0] m_axis_tdata,
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,
  output logic        m_axis_tlast
);
  localparam int SYN_W = UF_N - 1;
  localparam int IW    = (K <= 1) ? 1 : $clog2(K);

  // per-engine wires
  logic [K-1:0]      eng_in_valid, eng_out_valid, eng_obs;
  logic [SYN_W-1:0]  eng_syndrome [K];
  logic [UF_M-1:0]   eng_corr     [K];

  // per-slot state
  typedef enum logic [1:0] { SLOT_IDLE, SLOT_RUN, SLOT_DONE } slot_t;
  slot_t             slot_st      [K];
  logic [30:0]       slot_corr    [K];
  logic              slot_obs     [K];
  logic              slot_last    [K];

  logic [IW-1:0] disp_idx, coll_idx;

  genvar g;
  generate
    for (g = 0; g < K; g++) begin : gen_eng
      uf_surface_decoder u_dec (
        .clk(aclk), .rst_n(aresetn),
        .in_valid(eng_in_valid[g]), .syndrome(eng_syndrome[g]),
        .busy(), .out_valid(eng_out_valid[g]),
        .correction(eng_corr[g]), .obs_flip(eng_obs[g]), .latency_cycles()
      );
    end
  endgenerate

  assign s_axis_tready = (slot_st[disp_idx] == SLOT_IDLE);
  assign m_axis_tvalid = (slot_st[coll_idx] == SLOT_DONE);
  assign m_axis_tdata  = {slot_obs[coll_idx], slot_corr[coll_idx]};
  assign m_axis_tlast  = slot_last[coll_idx];

  integer i;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      eng_in_valid <= '0;
      disp_idx     <= '0;
      coll_idx     <= '0;
      for (i = 0; i < K; i++) slot_st[i] <= SLOT_IDLE;
    end else begin
      eng_in_valid <= '0;                                   // default: one-cycle pulses

      // ---- dispatch: accept a syndrome into the next round-robin engine if it is idle ----
      if (s_axis_tvalid && slot_st[disp_idx] == SLOT_IDLE) begin
        eng_syndrome[disp_idx] <= s_axis_tdata[SYN_W-1:0];
        slot_last[disp_idx]    <= s_axis_tlast;
        eng_in_valid[disp_idx] <= 1'b1;
        slot_st[disp_idx]      <= SLOT_RUN;
        disp_idx               <= (disp_idx == IW'(K - 1)) ? '0 : disp_idx + 1'b1;
      end

      // ---- capture completions on every running engine ----
      for (i = 0; i < K; i++) begin
        if (slot_st[i] == SLOT_RUN && eng_out_valid[i]) begin
          slot_obs[i]  <= eng_obs[i];
          slot_corr[i] <= 31'(eng_corr[i]);
          slot_st[i]   <= SLOT_DONE;
        end
      end

      // ---- collect: emit the next round-robin result when ready ----
      if (m_axis_tvalid && m_axis_tready) begin
        slot_st[coll_idx] <= SLOT_IDLE;
        coll_idx          <= (coll_idx == IW'(K - 1)) ? '0 : coll_idx + 1'b1;
      end
    end
  end

endmodule
