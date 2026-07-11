// Q7-04 M9b — SLIDING-WINDOW STREAMING wrapper around the BRAM-ified banked relay-BP core
// (`bp_relay_banked_bram`, Task 3). Decodes an unbounded stream of measurement rounds in bounded O(W)
// state: keep a running residual of W=BP_WIN_W rounds (BP_C = BP_WIN_W*BP_DPR check bits), decode it on
// the per-window core, COMMIT the oldest C=BP_WIN_C rounds' correction, XOR that correction's syndrome
// effect into the residual, slide forward by C, reload C fresh rounds, repeat. Interior windows of a
// stream are translation-invariant so ONE compiled window graph (`bb_stream_tanner.svh`, staged as
// `bb_gross_tanner.svh`) serves every steady-state slot and the banked core decodes it unchanged.
//
// Structural lift of `uf_streaming_decoder.sv` (S_WARM -> S_RUN -> S_WAIT -> S_COMMIT -> S_SLIDE ->
// S_RELOAD -> S_RUN). Deltas vs the UF wrapper: the banked BP core has a distinct handshake (early_exit
// pin, unpacked syndrome_in[BP_C], valid_flag, 32-bit latency_cycles); the commit/obs/toggle fabric is
// driven off the var-CSR (BP_VAR_OFF/BP_EDGE_CHK) + BP_VAR_COMMIT + BP_OBS_MASK rather than an edge list;
// the tail is handled by an internal zero-pad drain (after `in_last`, the warm/reload cursor zero-fills
// instead of consuming input, emitting exactly ceil(slices_seen/C) slots); and the frame is independent
// (after the final slot the FSM returns to S_WARM with NO external reset).
//
// Bit-for-bit equivalent to the software steady-state sliding decode: the per-window decode, the
// "commit every var touching the commit region, toggle its incident checks in the residual" rule, and
// the residual/slide/reload updates are the identical operations on the identical window graph. Verified
// against `bp_stream_vectors.txt` (40 trials, both early_exit modes) in tb_bp_stream.cpp.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_streaming_decoder (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                early_exit,        // forwarded to the per-window core
    // Stream input: present one measurement round of BP_DPR detector bits when `in_ready` is high.
    input  logic                in_valid,
    input  logic [BP_DPR-1:0]   in_round,
    input  logic                in_last,           // asserted with the stream's final round
    output logic                in_ready,          // wrapper can accept a round this cycle
    // Streaming output: one pulse per committed slot with that slot's committed decision.
    output logic                out_valid,
    output logic [BP_OBS-1:0]   out_obs,           // this slot's committed-obs XOR
    output logic                out_vflag,
    output logic                out_last,          // pulses with the final slot's out_valid
    output logic                out_commit_clean,  // commit region drained after this slot's commit
    output logic                commit_corr [BP_N],// per-slot committed vars (TB gate; unconnected in AXI wrap)
    output logic [15:0]         last_latency
);
  // BP_C must equal one full W-round window of detector bits (the residual frame width).
`ifndef SYNTHESIS
  initial
    if (BP_C != BP_WIN_W * BP_DPR)
      $fatal(1, "bp_streaming_decoder: BP_C(%0d) != BP_WIN_W*BP_DPR(%0d)", BP_C, BP_WIN_W * BP_DPR);
`endif

  localparam int RESW = BP_C;                 // residual frame width (= BP_WIN_W*BP_DPR)
  localparam int PTRW = $clog2(BP_C + 1);     // load-cursor width (indexes res[BP_C], value up to BP_C)
  localparam int COMW = BP_WIN_C * BP_DPR;    // commit-region width (the C rounds about to slide off)

  typedef enum logic [2:0] { S_WARM, S_RUN, S_WAIT, S_COMMIT, S_SLIDE, S_RELOAD } state_t;
  state_t state;

  logic [RESW-1:0] res;                       // residual syndrome over the W-round window
  logic [PTRW-1:0] lptr;                       // load cursor (bit offset for the next streamed round)
  logic            seen_last;                  // the stream's final round has been consumed
  logic [15:0]     slices_seen;                // accepted rounds this frame (until in_last)
  logic [15:0]     slots_total;                // ceil(slices_seen / BP_WIN_C), latched at in_last
  logic [15:0]     slots_done;                 // slots emitted this frame
  logic            cap_vflag;                  // core valid_flag latched at core out_valid

  // ---- per-window core ----
  logic            core_iv;
  logic            core_syn [BP_C];
  /* verilator lint_off UNUSEDSIGNAL */         // busy/obs unused: we wait on out_valid + recompute obs
  logic            core_busy;
  logic [BP_OBS-1:0] core_obs;
  /* verilator lint_on UNUSEDSIGNAL */
  logic            core_ov, core_vflag;
  logic            corr_out_core [BP_N];
  logic [31:0]     core_lat;

  always_comb
    for (int c = 0; c < BP_C; c++) core_syn[c] = res[c];

  bp_relay_banked_bram core (
      .clk(clk), .rst_n(rst_n),
      .in_valid(core_iv), .early_exit(early_exit), .syndrome_in(core_syn),
      .busy(core_busy), .out_valid(core_ov),
      .corr_out(corr_out_core), .obs_flip(core_obs),
      .valid_flag(core_vflag), .latency_cycles(core_lat)
  );

  // ---- combinational commit fabric (one S_COMMIT cycle, O(E) XOR fabric) ----
  // committed[v] = corr_out[v] & BP_VAR_COMMIT[v]; the committed-obs XOR folds BP_OBS_MASK over the
  // committed vars; the residual toggle XORs, for each committed var, every check incident on it
  // (var-CSR rows BP_VAR_OFF[v]..BP_VAR_OFF[v+1] of BP_EDGE_CHK) — clearing the defects the commit
  // explains and leaving the retained-region residual to slide forward.
  logic              committed_c [BP_N];
  logic [BP_OBS-1:0] commit_obs;
  logic [RESW-1:0]   commit_tog;
  always_comb begin
    commit_obs = '0;
    commit_tog = '0;
    for (int v = 0; v < BP_N; v++) begin
      committed_c[v] = corr_out_core[v] & BP_VAR_COMMIT[v];
      if (committed_c[v]) begin
        commit_obs ^= BP_OBS_MASK[v][BP_OBS-1:0];
        for (int e = BP_VAR_OFF[v]; e < BP_VAR_OFF[v + 1]; e++)
          commit_tog[BP_EDGE_CHK[e]] ^= 1'b1;
      end
    end
  end

  // ---- slide-by-C via the BP_SHIFT map: each residual bit carries to its post-slide index; committed-
  // and-dropped bits (sentinel == BP_C) fall off; the reload region [BP_LOAD_LO, BP_C) is cleared. ----
  logic [RESW-1:0] res_slid;
  always_comb begin
    res_slid = '0;
    for (int c = 0; c < BP_C; c++)
      if (BP_SHIFT[c] < BP_C && res[c]) res_slid[BP_SHIFT[c]] = 1'b1;
  end

  // Accept input only while a load cursor is open and the stream's tail has not yet arrived.
  assign in_ready = ((state == S_WARM) || (state == S_RELOAD)) && !seen_last;

  // ceil((slices_seen + 1) / BP_WIN_C): slots implied once the in_last round is counted.
  logic [15:0] slots_at_last;
  always_comb slots_at_last = 16'(((int'(slices_seen) + 1) + (BP_WIN_C - 1)) / BP_WIN_C);

  always_ff @(posedge clk) begin              // synchronous reset (matches the core, Synth 8-7137)
    if (!rst_n) begin
      state            <= S_WARM;
      res              <= '0;
      lptr             <= '0;
      seen_last        <= 1'b0;
      slices_seen      <= '0;
      slots_total      <= '0;
      slots_done       <= '0;
      cap_vflag        <= 1'b0;
      core_iv          <= 1'b0;
      out_valid        <= 1'b0;
      out_obs          <= '0;
      out_vflag        <= 1'b0;
      out_last         <= 1'b0;
      out_commit_clean <= 1'b0;
      last_latency     <= '0;
      for (int v = 0; v < BP_N; v++) commit_corr[v] <= 1'b0;
    end else begin
      out_valid <= 1'b0;            // 1-cycle pulse
      core_iv   <= 1'b0;

      unique case (state)
        // Warm-up: load the first W rounds one round per cycle (zero-pad past in_last).
        S_WARM: begin
          if (!seen_last) begin
            if (in_valid) begin
              res[lptr +: BP_DPR] <= in_round;
              slices_seen         <= slices_seen + 16'd1;
              if (in_last) begin
                seen_last   <= 1'b1;
                slots_total <= slots_at_last;
              end
              if (lptr + PTRW'(BP_DPR) >= PTRW'(BP_C)) begin lptr <= '0; state <= S_RUN; end
              else                                          lptr <= lptr + PTRW'(BP_DPR);
            end
          end else begin            // internal zero-pad drain
            res[lptr +: BP_DPR] <= '0;
            if (lptr + PTRW'(BP_DPR) >= PTRW'(BP_C)) begin lptr <= '0; state <= S_RUN; end
            else                                          lptr <= lptr + PTRW'(BP_DPR);
          end
        end

        // Kick the per-window core on the current residual (1-cycle in_valid pulse).
        S_RUN: begin
          core_iv <= 1'b1;
          state   <= S_WAIT;
        end

        // Wait for the core's decision; latch latency (saturated) and validity.
        S_WAIT: if (core_ov) begin
          last_latency <= (core_lat[31:16] != '0) ? 16'hFFFF : core_lat[15:0];
          cap_vflag    <= core_vflag;
          state        <= S_COMMIT;
        end

        // Apply the committed correction to the residual and emit this slot's decision.
        S_COMMIT: begin
          automatic logic [RESW-1:0] res_next = res ^ commit_tog;
          res              <= res_next;
          for (int v = 0; v < BP_N; v++) commit_corr[v] <= committed_c[v];
          out_obs          <= commit_obs;
          out_vflag        <= cap_vflag;
          out_commit_clean <= (res_next[COMW-1:0] == '0);   // commit region drained after this commit
          out_last         <= (slots_done + 16'd1 == slots_total);
          out_valid        <= 1'b1;
          slots_done       <= slots_done + 16'd1;
          state            <= S_SLIDE;
        end

        // Slide forward by C (drop committed rounds, clear the reload region) and open the reload cursor.
        S_SLIDE: begin
          res   <= res_slid;
          lptr  <= PTRW'(BP_LOAD_LO);
          state <= S_RELOAD;
        end

        // Reload the C newest rounds into [BP_LOAD_LO, BP_C) (zeros past in_last), then decode the next
        // slot — or return to S_WARM (frame done, no external reset) once every slot is emitted.
        S_RELOAD: begin
          automatic logic done = (lptr + PTRW'(BP_DPR) >= PTRW'(BP_C));
          if (!seen_last) begin
            if (in_valid) begin
              res[lptr +: BP_DPR] <= in_round;
              slices_seen         <= slices_seen + 16'd1;
              if (in_last) begin
                seen_last   <= 1'b1;
                slots_total <= slots_at_last;
              end
              if (done) begin
                lptr <= '0;
                if (slots_done == slots_total) begin
                  state       <= S_WARM;  res <= '0; seen_last <= 1'b0;
                  slices_seen <= '0;      slots_done <= '0; slots_total <= '0;
                end else state <= S_RUN;
              end else lptr <= lptr + PTRW'(BP_DPR);
            end
          end else begin            // internal zero-pad drain
            res[lptr +: BP_DPR] <= '0;
            if (done) begin
              lptr <= '0;
              if (slots_done == slots_total) begin
                state       <= S_WARM;  res <= '0; seen_last <= 1'b0;
                slices_seen <= '0;      slots_done <= '0; slots_total <= '0;
              end else state <= S_RUN;
            end else lptr <= lptr + PTRW'(BP_DPR);
          end
        end

        default: state <= S_WARM;
      endcase
    end
  end
endmodule
