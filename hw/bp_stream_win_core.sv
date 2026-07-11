// Q7-04 M9b (Task 6) — AXI4-Stream front-end for the SLIDING-WINDOW banked-BP streaming decoder
// (`bp_streaming_decoder`, Task 5).
//
// Structural lift of `uf_stream_win_core.sv` (Q6-20 on-silicon AXI wrapper): same per-frame re-arm
// (`frame_rst` pulse, `dec_rst_n = aresetn & ~frame_rst` — the Q6-20 mid-stream-resume fix) so a
// follow-on DMA transfer always starts the wrapped decoder fresh in warm-up instead of resuming
// mid-window. Deltas vs the UF wrapper, driven by `bp_streaming_decoder`'s distinct handshake:
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
//   * the UF shell's 1-deep result slot becomes a small result FIFO. The UF streaming decoder cannot
//     retire a window without first consuming input (which the shell gates on the slot being free), so
//     1-deep sufficed there. `bp_streaming_decoder` does NOT have that property: after `in_last` it
//     self-drives through the tail slots with NO input handshake (its S_WARM/S_RELOAD zero-fill branches
//     never consult in_valid, and in_ready is low for the rest of the frame), so S2MM back-pressure
//     cannot stall the drain — a 1-deep slot would be OVERWRITTEN if m_axis_tready stalled across an
//     inter-slot gap of the drain. The FIFO absorbs the whole (bounded) tail; depth derivation below.
//   * reset style: `bp_streaming_decoder` uses a SYNCHRONOUS reset (its own comment: "matches the core,
//     Synth 8-7137"). To avoid a mixed sync/async reset net feeding the same decoder instance, this shell
//     uses a SYNCHRONOUS `aresetn` too (the UF shell used async, matching ITS wrapped decoder) — a
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
  initial begin
    if (BP_DPR != 72)
      $fatal(1, "bp_stream_win_core: framing hardcodes 3 beats of {32,32,8} for BP_DPR=72, got %0d",
             BP_DPR);
    if (BP_OBS != 12)
      $fatal(1, "bp_stream_win_core: result word hardcodes obs[11:0] at bits [31:20], got BP_OBS=%0d",
             BP_OBS);
  end
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

  // ---- result FIFO ----
  // During STREAMING, the input gate below keeps occupancy <= 1: a round is accepted only while the FIFO
  // is EMPTY (exactly the old 1-deep slot's `~out_full` semantics), and the decoder needs C fresh rounds
  // before it can retire another slot, so back-pressure reaches the decoder through the input path.
  // During the post-`in_last` DRAIN that path does not exist — the decoder zero-fills internally and
  // emits the remaining slots regardless of m_axis_tready — so the FIFO must absorb the whole tail.
  // The tail is bounded: at most ceil(W/C) slots (the W-round residual still unretired when in_last
  // lands), and the drain starts with an EMPTY FIFO (the in_last round can only have been accepted while
  // the FIFO was empty, and the decoder never emits a slot in the same cycle it accepts a round: its
  // out_valid pulse is registered from S_COMMIT, when in_ready is low). ceil(W/C)+1 therefore covers the
  // tail with one slot of margin (4 at W=6, C=2).
  localparam int TAIL_SLOTS = (BP_WIN_W + BP_WIN_C - 1) / BP_WIN_C;  // ceil(W/C) drain-tail slots
  localparam int OUT_DEPTH  = TAIL_SLOTS + 1;
  localparam int OPTR_W     = (OUT_DEPTH > 1) ? $clog2(OUT_DEPTH) : 1;
  localparam int OCNT_W     = $clog2(OUT_DEPTH + 1);

  logic [32:0]       out_q [OUT_DEPTH];  // {tlast, result word}
  logic [OPTR_W-1:0] wptr, rptr;
  logic [OCNT_W-1:0] count;

  wire out_empty = (count == '0);
  wire pop       = m_axis_tvalid & m_axis_tready;

  // Input gate: ANY parked result blocks further rounds (the old 1-deep `~out_full` semantics). This is
  // what confines FIFO occupancy > 1 to the drain, where the TAIL_SLOTS bound above applies.
  assign s_axis_tready = dec_in_ready & out_empty & ~frame_rst;

  assign m_axis_tvalid = ~out_empty;
  assign m_axis_tdata  = out_q[rptr][31:0];
  assign m_axis_tlast  = out_q[rptr][32];

`ifndef SYNTHESIS
  // Tripwire for the depth derivation above: a push while full means the drain-tail bound was violated.
  always_ff @(posedge aclk)
    if (aresetn && dec_out_valid && count == OCNT_W'(OUT_DEPTH))
      $fatal(1, "bp_stream_win_core: result FIFO overflow (depth %0d) — drain-tail bound violated",
             OUT_DEPTH);
`endif

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      beat_idx        <= 2'd0;
      beat0_r         <= '0;
      beat1_r         <= '0;
      round_tlast_acc <= 1'b0;
      wptr            <= '0;
      rptr            <= '0;
      count           <= '0;
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

      // a committed slot arrives: push its result word (cannot overflow, per the tail bound above —
      // guarded by the non-synth tripwire)
      if (dec_out_valid) begin
        out_q[wptr] <= {dec_out_last, dec_out_obs, dec_out_vflag, dec_out_commit_clean, 2'b00,
                        dec_out_lat};
        wptr        <= (wptr == OPTR_W'(OUT_DEPTH - 1)) ? '0 : wptr + 1'b1;
      end

      // S2MM consumed a result: advance the read side. If it was the frame's last (tlast) word, re-arm
      // the decoder for the next transfer.
      if (pop) begin
        if (out_q[rptr][32]) frame_rst <= 1'b1;
        rptr <= (rptr == OPTR_W'(OUT_DEPTH - 1)) ? '0 : rptr + 1'b1;
      end

      unique case ({dec_out_valid, pop})
        2'b10:   count <= count + 1'b1;
        2'b01:   count <= count - 1'b1;
        default: ;  // 2'b00 / 2'b11: occupancy unchanged
      endcase
    end
  end

endmodule
