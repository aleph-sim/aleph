// Q7-04 M9b (Task 6) — AXI4-Stream front-end for the SLIDING-WINDOW banked-BP streaming decoder
// (`bp_streaming_decoder`, Task 5).
//
// Structural lift of `uf_stream_win_core.sv` (Q6-20 on-silicon AXI wrapper): same 1-deep result slot,
// same tlast latch, same per-frame re-arm (`frame_rst` pulse, `dec_rst_n = aresetn & ~frame_rst` — the
// Q6-20 mid-stream-resume fix) so a follow-on DMA transfer always starts the wrapped decoder fresh in
// warm-up instead of resuming mid-window. Deltas vs the UF wrapper, driven by `bp_streaming_decoder`'s
// distinct handshake:
//   * input framing is 3 MM2S beats per round (BP_DPR=72 > 32, so one round no longer fits a single
//     32-bit beat): beat0 = round bits [31:0], beat1 = [63:32], beat2 = {24'b0, bits[71:64]}. A small
//     beat-assembler register captures beat0/beat1; the completing (3rd) beat combines its own tdata with
//     the two captured beats combinationally, the same cycle, to present the full round to the decoder —
//     no extra latency beyond the 3 beats themselves.
//   * `in_last` fires only on the round completed by a beat carrying tlast (tlast is latched across a
//     round's beats via `round_tlast_acc` in case a producer tags it early; the expected/only case is
//     tlast on the round's final beat).
//   * the decoder emits its OWN `out_last` (unlike the UF core, which had to reconstruct end-of-batch via
//     a `pending_last` latch over `residual_empty`), so the result word's tlast is a direct pass-through.
//   * reset style: `bp_streaming_decoder` uses a SYNCHRONOUS reset (its own comment: "matches the core,
//     Synth 8-7137"). To avoid a mixed sync/async reset net feeding the same decoder instance, this shell
//     uses a SYNCHRONOUS `aresetn` too (the UF shell used async, matching ITS wrapped decoder) — the one
//     deliberate deviation from the UF template's structure, forced by the wrapped core's own reset style.
//
// Output framing: one 32-bit S2MM word per committed slot —
//   [31:20] = out_obs[11:0], [19] = vflag, [18] = commit_clean, [17:16] = 2'b00, [15:0] = latency
//   (already 16-bit saturated by the decoder). tlast rides the frame's final slot (`out_last`).
//
// `commit_corr[BP_N]` (the bare decoder's per-slot committed-variable debug port) is intentionally left
// unconnected here: it is not part of the AXI result-word contract, only {obs, vflag, commit_clean} are.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_stream_win_core (
    input  logic        aclk,
    input  logic        aresetn,        // active-low, SYNCHRONOUS (matches bp_streaming_decoder)
    input  logic        early_exit_i,   // forwarded to the wrapped decoder's per-window core

    /* verilator lint_off UNUSEDSIGNAL */  // only the low BP_DPR bits across 3 beats carry a round; the
                                            // upper bits of the 3rd beat are padding and ignored
    input  logic [31:0] s_axis_tdata,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic         s_axis_tvalid,
    output logic         s_axis_tready,
    input  logic         s_axis_tlast,

    output logic [31:0] m_axis_tdata,
    output logic         m_axis_tvalid,
    input  logic         m_axis_tready,
    output logic         m_axis_tlast
);
`ifndef SYNTHESIS
  initial
    if (BP_DPR != 72)
      $fatal(1, "bp_stream_win_core: framing hardcodes 3 beats of {32,32,8} for BP_DPR=72, got %0d",
             BP_DPR);
`endif

  // ---- input beat assembler: one round = 3 MM2S beats ----
  logic [1:0]  beat_idx;      // 0,1,2 within the current round
  logic [31:0] beat0_r, beat1_r;
  logic        round_tlast_acc;

  logic dec_in_ready;
  wire  beat_accept  = s_axis_tvalid & s_axis_tready;
  wire  round_accept = beat_accept & (beat_idx == 2'd2);
  // Completing beat: combine the two captured beats with THIS cycle's live tdata (low byte only).
  wire [BP_DPR-1:0] round_word = {s_axis_tdata[7:0], beat1_r, beat0_r};

  logic              dec_in_valid;
  logic [BP_DPR-1:0] dec_in_round;
  logic              dec_in_last;
  assign dec_in_valid = round_accept;
  assign dec_in_round = round_word;
  assign dec_in_last  = round_accept & (round_tlast_acc | s_axis_tlast);

  logic              dec_out_valid, dec_out_vflag, dec_out_last, dec_out_commit_clean;
  logic [BP_OBS-1:0] dec_out_obs;
  logic [15:0]       dec_out_lat;

  // Per-FRAME reset: when the tlast-tagged result word drains, pulse the DECODER's reset for one cycle so
  // the next transfer starts fresh in warm-up (S_WARM, empty residual) instead of resuming mid-window —
  // the Q6-20 mid-stream-resume fix, ported verbatim from `uf_stream_win_core`.
  logic frame_rst;
  wire  dec_rst_n = aresetn & ~frame_rst;

  /* verilator lint_off PINCONNECTEMPTY */  // commit_corr is a TB debug port, deliberately unconnected
  bp_streaming_decoder u_dec (
      .clk(aclk), .rst_n(dec_rst_n), .early_exit(early_exit_i),
      .in_valid(dec_in_valid), .in_round(dec_in_round), .in_last(dec_in_last), .in_ready(dec_in_ready),
      .out_valid(dec_out_valid), .out_obs(dec_out_obs), .out_vflag(dec_out_vflag),
      .out_last(dec_out_last), .out_commit_clean(dec_out_commit_clean),
      .commit_corr(),  // TB debug port on the bare decoder; not part of the AXI result-word contract
      .last_latency(dec_out_lat)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // ---- single result slot (1-deep). Input is gated on it being free, so the decoder cannot retire a
  // new slot until S2MM has consumed the previous one — a 1-deep buffer suffices, no overflow. ----
  logic        out_full;
  logic [31:0] out_word;
  logic        out_last;

  // Gate ALL beats (not just the round-completing one) on the decoder + result-slot + re-arm state, the
  // same flat formula as the UF shell: it is safe because `dec_in_valid` only ever fires on the
  // round-completing beat, so beats 0/1 merely accumulate into the assembler.
  assign s_axis_tready = dec_in_ready & ~out_full & ~frame_rst;

  assign m_axis_tvalid = out_full;
  assign m_axis_tdata  = out_word;
  assign m_axis_tlast  = out_last;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      beat_idx        <= 2'd0;
      beat0_r         <= '0;
      beat1_r         <= '0;
      round_tlast_acc <= 1'b0;
      out_full        <= 1'b0;
      out_word        <= '0;
      out_last        <= 1'b0;
      frame_rst       <= 1'b0;
    end else begin
      frame_rst <= 1'b0;  // default: 1-cycle re-arm pulse only

      if (beat_accept) begin
        unique case (beat_idx)
          2'd0: begin beat0_r <= s_axis_tdata; round_tlast_acc <= s_axis_tlast; beat_idx <= 2'd1; end
          2'd1: begin
            beat1_r         <= s_axis_tdata;
            round_tlast_acc <= round_tlast_acc | s_axis_tlast;
            beat_idx        <= 2'd2;
          end
          2'd2: begin round_tlast_acc <= 1'b0; beat_idx <= 2'd0; end
          default: beat_idx <= 2'd0;
        endcase
      end

      // a committed slot arrives: latch its result word into the free slot
      if (dec_out_valid) begin
        out_word <= {dec_out_obs, dec_out_vflag, dec_out_commit_clean, 2'b00, dec_out_lat};
        out_last <= dec_out_last;
        out_full <= 1'b1;
      end else if (out_full && m_axis_tready) begin
        // S2MM consumed the result; free the slot. If this was the frame's last (tlast) slot, re-arm
        // the decoder for the next transfer.
        out_full <= 1'b0;
        if (out_last) frame_rst <= 1'b1;
        out_last <= 1'b0;
      end
    end
  end

endmodule
