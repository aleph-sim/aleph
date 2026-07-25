// Q7-06 (AC-1) — AXI4-Stream BATCH front-end for the K-banked relay-BP BLOCK decoder (`bp_relay_banked`).
//
// The M8 KV260 overlay drives the banked core over AXI4-Lite one experiment at a time: per decode the
// host writes NS syndrome words, pulses START, POLLS status in a Python loop, then reads the result — so
// the measured "harness throughput" is dominated by per-experiment Python + MMIO round-trips, NOT the
// hardware decode. Q7-06 AC-1 removes that host overhead: a whole BATCH of independent syndrome→result
// experiments streams through one AXI-DMA transfer (MM2S in, S2MM out), the core decodes them back-to-back
// at hardware speed, and the host pays one `dma.transfer` for the whole batch. This is the ≥100×
// harness-throughput lever (and the batched-duty measurement that tightens the Q7-05 µJ/decode bound).
//
// This is a STRUCTURAL sibling of `bp_stream_win_core` (the sliding-window AXIS shell), but the wrapped
// unit here is a BLOCK decoder, which makes the framing strictly simpler:
//   * NO per-frame decoder reset. `bp_relay_banked` is stateless between decodes (each `in_valid` pulse
//     starts a fresh decode from the presented syndrome), so there is no mid-window resume to guard against
//     and no `frame_rst` pulse. A batch is just N independent decodes; `tlast` only marks the batch end.
//   * NO drain-tail FIFO. The block decoder emits EXACTLY ONE result per accepted syndrome and never
//     self-drives extra slots, so a shallow (depth-2) output FIFO covers S2MM back-pressure; there is no
//     unbounded post-`in_last` tail to absorb.
//   * input framing is NS = ceil(BP_C/32) MM2S beats per experiment (5 at BP_C=144): beat i carries
//     syndrome bits [i*32 +: 32]; the low BP_C bits of the assembled NS*32 word are used, the rest padding.
//   * one-experiment-at-a-time: `s_axis_tready` is asserted only while idle and the output FIFO is empty
//     (the same 1-deep gate `bp_stream_win_core` uses during streaming). The core is not pipelined across
//     experiments, so decode latency dominates anyway — overlapping the 5 input beats with a ~900+ cycle
//     decode buys nothing, and serialising keeps the syndrome register stable for the whole decode.
//
// Output framing: one 32-bit S2MM word per experiment (LER needs only the observable flips, not the
// 864-bit correction), bit-compatible with `bp_stream_win_core`'s result word:
//   [31:20] = obs_flip[11:0], [19] = valid_flag, [18] = 1'b0 (reserved), [17:16] = 2'b00,
//   [15:0]  = latency_cycles (16-bit saturated).
// `tlast` rides the result word of the experiment whose final input beat carried `tlast`.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_stream_banked_core (
    input  logic        aclk,
    input  logic        aresetn,        // active-low, synchronous
    input  logic        early_exit_i,   // forwarded to the wrapped banked core (sticky per decode)

    /* verilator lint_off UNUSEDSIGNAL */  // only the low BP_C bits across NS beats carry a syndrome; the
                                            // upper bits of the final beat are padding and ignored
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
  localparam int NS      = (BP_C + 31) / 32;         // syndrome beats per experiment (5 at BP_C=144)
  localparam int IDX_W   = (NS > 1) ? $clog2(NS) : 1;

`ifndef SYNTHESIS
  initial begin
    if (BP_OBS > 12)
      $fatal(1, "bp_stream_banked_core: result word packs obs at bits [31:20], got BP_OBS=%0d", BP_OBS);
  end
`endif

  // ---- input beat assembler: one experiment = NS beats ----
  logic [IDX_W-1:0]  beat_idx;                        // 0..NS-1 within the current experiment
  logic [NS*32-1:0]  synd_asm;                        // accumulates beats [0..NS-2]; final beat live-combined
  logic              synd_tlast_acc;                  // tlast latched across an experiment's beats

  wire  beat_accept = s_axis_tvalid & s_axis_tready;
  wire  last_beat   = (beat_idx == IDX_W'(NS - 1));
  // Completing beat: combine the captured beats with THIS cycle's live tdata in the top word slot. Only
  // the low BP_C bits feed the decoder; the top (NS*32 - BP_C) bits are padding.
  /* verilator lint_off UNUSEDSIGNAL */
  wire [NS*32-1:0] synd_full = synd_asm | ({{(NS*32-32){1'b0}}, s_axis_tdata} << (32 * (NS - 1)));
  /* verilator lint_on UNUSEDSIGNAL */

  // ---- wrapped banked block decoder ----
  /* verilator lint_off UNUSEDSIGNAL */  // busy is unused: completion is tracked via out_valid
  logic              dec_busy;
  /* verilator lint_on UNUSEDSIGNAL */
  logic              dec_in_valid, dec_out_valid, dec_vflag;
  logic              dec_syndrome [BP_C];
  logic [BP_OBS-1:0] dec_obs;
  logic [31:0]       dec_lat;
  logic [BP_C-1:0]   synd_reg;                         // stable syndrome for the in-flight decode
  for (genvar i = 0; i < BP_C; i++) begin : gen_synd_fanout
    assign dec_syndrome[i] = synd_reg[i];
  end

  /* verilator lint_off PINCONNECTEMPTY */  // corr_out is the 864-bit correction: not part of the LER
                                            // result-word contract (only {obs, vflag, latency} are)
  bp_relay_banked u_dec (
      .clk(aclk), .rst_n(aresetn), .in_valid(dec_in_valid), .early_exit(early_exit_i),
      .syndrome_in(dec_syndrome), .busy(dec_busy), .out_valid(dec_out_valid),
      .corr_out(), .obs_flip(dec_obs), .valid_flag(dec_vflag), .latency_cycles(dec_lat)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // latency saturated to 16 bits for the result word
  wire [15:0] lat_sat = (dec_lat > 32'hFFFF) ? 16'hFFFF : dec_lat[15:0];

  // ---- owner FSM: serialise assemble -> decode -> emit ----
  typedef enum logic [1:0] { S_ACCEPT, S_BUSY } state_t;
  state_t state;

  // per-experiment: tlast that must ride this decode's result word
  logic result_tlast;

  // ---- shallow output FIFO (depth 2) for S2MM back-pressure ----
  localparam int OUT_DEPTH = 2;
  localparam int OPTR_W    = 1;                        // $clog2(2)
  localparam int OCNT_W    = 2;                        // $clog2(3)
  logic [32:0]        out_q [OUT_DEPTH];               // {tlast, result word}
  logic [OPTR_W-1:0]  wptr, rptr;
  logic [OCNT_W-1:0]  count;
  wire out_empty = (count == '0);
  wire out_full  = (count == OCNT_W'(OUT_DEPTH));
  wire pop       = m_axis_tvalid & m_axis_tready;

  // Input gate: accept beats only while idle AND no result parked (1-deep streaming semantics). This
  // confines the FIFO to depth<=1 in steady state; the 2nd slot is margin for a decode completing the
  // same cycle a pop drains slot 0.
  assign s_axis_tready = (state == S_ACCEPT) & out_empty;

  assign m_axis_tvalid = ~out_empty;
  assign m_axis_tdata  = out_q[rptr][31:0];
  assign m_axis_tlast  = out_q[rptr][32];

`ifndef SYNTHESIS
  always_ff @(posedge aclk)
    if (aresetn && dec_out_valid && out_full)
      $fatal(1, "bp_stream_banked_core: result FIFO overflow (depth %0d)", OUT_DEPTH);
`endif

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      beat_idx       <= '0;
      synd_asm       <= '0;
      synd_tlast_acc <= 1'b0;
      synd_reg       <= '0;
      dec_in_valid   <= 1'b0;
      result_tlast   <= 1'b0;
      state          <= S_ACCEPT;
      wptr           <= '0;
      rptr           <= '0;
      count          <= '0;
    end else begin
      dec_in_valid <= 1'b0;  // default: one-cycle pulse

      // ---- input assembly (only in S_ACCEPT; s_axis_tready is low otherwise) ----
      if (beat_accept) begin
        if (last_beat) begin
          // present the completed syndrome to the decoder next cycle; latch its tlast
          synd_reg       <= synd_full[BP_C-1:0];
          result_tlast   <= synd_tlast_acc | s_axis_tlast;
          dec_in_valid   <= 1'b1;
          beat_idx       <= '0;
          synd_asm       <= '0;
          synd_tlast_acc <= 1'b0;
          state          <= S_BUSY;
        end else begin
          synd_asm       <= synd_asm | ({{(NS*32-32){1'b0}}, s_axis_tdata} << (32 * beat_idx));
          synd_tlast_acc <= synd_tlast_acc | s_axis_tlast;
          beat_idx       <= beat_idx + 1'b1;
        end
      end

      // ---- decode completion: push result word to the FIFO, return to accepting ----
      if (state == S_BUSY && dec_out_valid) begin
        out_q[wptr] <= {result_tlast, dec_obs, dec_vflag, 1'b0, 2'b00, lat_sat};
        wptr        <= (wptr == OPTR_W'(OUT_DEPTH - 1)) ? '0 : wptr + 1'b1;
        state       <= S_ACCEPT;
      end

      // ---- S2MM drained a result ----
      if (pop) rptr <= (rptr == OPTR_W'(OUT_DEPTH - 1)) ? '0 : rptr + 1'b1;

      unique case ({(state == S_BUSY) && dec_out_valid, pop})
        2'b10:   count <= count + 1'b1;
        2'b01:   count <= count - 1'b1;
        default: ;  // unchanged
      endcase
    end
  end

endmodule
