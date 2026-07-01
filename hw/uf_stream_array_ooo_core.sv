// Q6-03 (throughput scaling, out-of-order) — K decoder engines with a reorder buffer.
//
// Improves on the in-order uf_stream_array_core: there, round-robin dispatch/collect stalls on a
// specific slow engine (decode latency varies 30-69 clk), capping scaling at ~71-82 %. Here:
//   * dispatch to ANY free engine (priority pick), each syndrome tagged with a sequence number;
//   * an engine frees the instant it writes its result into the reorder buffer (ROB) — it does NOT
//     wait to be collected, so a slow engine no longer blocks the others;
//   * the collector emits ROB entries strictly in sequence order, so output beat i still corresponds
//     to input beat i (and the batch tlast, latched per slot at dispatch, lands on the last beat) for
//     a clean bulk S2MM.
// Dispatch back-pressures (s_axis_tready=0) only when no engine is free OR the ROB is full — head-of-
// line blocking is confined to the ROB read pointer, which just reorders output, not to the engines.
//
// ROB depth D = next-pow2(4*K) absorbs the completion-order variance; it is a small register file
// (K simultaneous indexed writes on completion, one indexed read at the head).

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module uf_stream_array_ooo_core #(
  parameter int K = 8
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
  localparam int D     = 1 << $clog2(4 * K);       // ROB depth (pow2), >= 4*K
  localparam int AW    = $clog2(D);

  // ---- engines ----
  logic [K-1:0]      eng_in_valid, eng_out_valid, eng_obs;
  logic [SYN_W-1:0]  eng_syndrome [K];
  logic [UF_M-1:0]   eng_corr     [K];
  logic              eng_run      [K];             // 0 = idle, 1 = running
  logic [AW-1:0]     eng_slot     [K];             // ROB slot this engine's result belongs to

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

  // ---- reorder buffer ----
  logic [31:0] rob_data  [D];
  logic        rob_last  [D];
  logic        rob_valid [D];
  logic [AW:0] wr_seq, rd_seq;                      // AW+1 bits (extra bit for full/empty)

  wire [AW:0]     inflight = wr_seq - rd_seq;
  wire            rob_full = (inflight == D[AW:0]);
  wire [AW-1:0]   wr_slot  = wr_seq[AW-1:0];
  wire [AW-1:0]   rd_slot  = rd_seq[AW-1:0];

  // ---- free-engine priority pick ----
  logic          any_free;
  logic [IW-1:0] free_idx;
  always_comb begin
    any_free = 1'b0;
    free_idx = '0;
    for (int i = 0; i < K; i++) begin
      if (!any_free && !eng_run[i]) begin
        any_free = 1'b1;
        free_idx = i[IW-1:0];
      end
    end
  end

  assign s_axis_tready = any_free && !rob_full;
  assign m_axis_tvalid = rob_valid[rd_slot];
  assign m_axis_tdata  = rob_data[rd_slot];
  assign m_axis_tlast  = rob_last[rd_slot];

  integer j;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      eng_in_valid <= '0;
      wr_seq       <= '0;
      rd_seq       <= '0;
      for (j = 0; j < K; j++) eng_run[j]   <= 1'b0;
      for (j = 0; j < D; j++) rob_valid[j] <= 1'b0;
    end else begin
      eng_in_valid <= '0;                            // default: one-cycle pulses

      // ---- dispatch: any free engine + a ROB slot ----
      if (s_axis_tvalid && s_axis_tready) begin
        eng_syndrome[free_idx] <= s_axis_tdata[SYN_W-1:0];
        eng_slot[free_idx]     <= wr_slot;
        eng_in_valid[free_idx] <= 1'b1;
        eng_run[free_idx]      <= 1'b1;
        rob_last[wr_slot]      <= s_axis_tlast;
        wr_seq                 <= wr_seq + 1'b1;
      end

      // ---- completions: write result into its ROB slot, free the engine ----
      for (j = 0; j < K; j++) begin
        if (eng_run[j] && eng_out_valid[j]) begin
          rob_data[eng_slot[j]]  <= {eng_obs[j], 31'(eng_corr[j])};
          rob_valid[eng_slot[j]] <= 1'b1;
          eng_run[j]             <= 1'b0;
        end
      end

      // ---- collect: emit the head ROB entry in sequence order ----
      if (m_axis_tvalid && m_axis_tready) begin
        rob_valid[rd_slot] <= 1'b0;
        rd_seq             <= rd_seq + 1'b1;
      end
    end
  end

endmodule
