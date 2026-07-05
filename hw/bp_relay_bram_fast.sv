// Q7-02 M2-BRAM-FAST — pipelined-read edge-serial relay-BP decoder (circuit-level synthesizable).
//
// Functionally identical to `bp_relay_bram` (and thus to the M2 reference `bp_relay_decoder`): same
// schedule, quantisation (Q5.3, multiply-free α=7/8 check-update, one memory-blend multiply), truncating
// arithmetic-shift rounding, and keep-lowest-weight-valid rule. Decodes the SAME graph header and is
// verified bit-for-bit against the SAME golden (`tb_bp_relay.cpp`).
//
// WHY IT EXISTS. `bp_relay_bram` (PR #448) already made the circuit-level graph FIT on xc7z020 by moving
// the m_vc/e_cv message tables into block RAM and walking them edge-serially. But its BRAM read passes
// (S_CHK1 min-sum scan, S_VAR1 e_cv read, S_VAR2 old-m_vc read) each spent **2 cycles/edge**: a `ph`
// sub-phase presented the address on ph==0 and consumed the registered read on ph==1. That doubles the
// read-heavy passes for no reason — the read port can accept a new address every cycle.
//
// FIX (this file). **Pipeline the reads**: drive the read address for edge `p` every cycle while
// consuming the registered read datum for edge `p-1`. Each read pass drops from 2·deg to deg+1 cycles
// (one-cycle fill). Per iteration this is 5·BP_E+3·BP_C+3·BP_N (vs the slow core's 8·BP_E+2·BP_C+BP_N)
// — ~1.39× fewer cycles on the rounds=1 circuit graph (24768→17785 cyc/iter). S_VAR2 is a pipelined
// read-modify-write: read m_vc[lo+p] while writing the blend for edge (p-1); adjacent edges are distinct
// addresses so there is no RAW hazard between the read port and the write port.
//
// Second lever: **register the S_SAT keep-best commit**. The slow core folded, in one cycle at the last
// check of each iteration, the parity scan + `ehat_w < best_w` compare into the 864-wide `best_e`
// register-enable fan-out (the route-dominated critical path #448 named). Here that decision is latched
// (`do_commit`) in S_SAT1 and the wide copy happens the next cycle in a dedicated S_SAT2 state, so the
// 864 enables are driven by a plain flop, not the deep combinational chain. Costs +1 cyc/iteration (60
// total — negligible), buys Fmax.
//
// Arithmetic is byte-for-byte the M2 golden: this is a microarchitecture change, not an algorithm change.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_bram_fast (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],       // chosen error pattern (one bit / variable)
    output logic [BP_OBS-1:0]   obs_flip,                 // predicted observable flips
    output logic                valid_flag,               // a syndrome-valid decision was found
    output logic [31:0]         latency_cycles
);
  localparam logic [MSG_BITS-1:0] INF = '1;   // +∞ sentinel for running minima (> every magnitude)
  localparam int WACC = 32;                   // wide accumulator/product (generous; matches M2 golden)
  localparam int WW   = $clog2(BP_N + 1);     // hard-decision weight counter
  localparam int AW   = $clog2(BP_E);         // edge-table / BRAM address width

  // ---- message BRAMs: m_vc (variable→check), e_cv (check→variable) ----
  logic signed [MSG_BITS-1:0] mvc_mem [BP_E];
  logic signed [MSG_BITS-1:0] ecv_mem [BP_E];
  logic [AW-1:0]              mvc_ra, mvc_wa, ecv_ra, ecv_wa;
  logic signed [MSG_BITS-1:0] mvc_rd, ecv_rd, mvc_wd, ecv_wd;
  logic                       mvc_we, ecv_we;

  always_ff @(posedge clk) begin
    if (mvc_we) mvc_mem[mvc_wa] <= mvc_wd;
    mvc_rd <= mvc_mem[mvc_ra];                 // 1-cycle read latency
    if (ecv_we) ecv_mem[ecv_wa] <= ecv_wd;
    ecv_rd <= ecv_mem[ecv_ra];
  end

  // ---- flop state (1-bit / single-access; cheap even with a runtime cursor) ----
  /* verilator lint_off UNUSEDSIGNAL */
  logic                       s_reg  [BP_C];
  logic                       ehat   [BP_N];   // current hard decision
  logic                       best_e [BP_N];   // lowest-weight syndrome-valid decision seen
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat, do_commit;
  logic [BP_OBS-1:0]          obs_acc;

  // per-check pass-1 min-sum accumulators (finalised before pass-2)
  logic                       neg;
  logic [MSG_BITS-1:0]        min1, min2;
  int                         argmin_pos;
  logic                       edge_sgn [BP_CHK_DEG];      // sign bit of each edge's message

  // per-variable accumulators
  logic signed [WACC-1:0]     total, greg, omgreg;
  logic signed [MSG_BITS-1:0] ecv_loc [BP_VAR_DEG];       // e_cv values for the current variable

  logic                       parity;                     // per-check syndrome-parity (S_SAT)

  // cursors / loop counters.  In the pipelined read passes `p` is the *address* cursor: it runs 0..deg,
  // presenting the read for edge p (when p<deg) and consuming the registered read for edge p-1 (when p>=1).
  int          idx, p, leg, iter, lo, hi, deg;
  logic [31:0] lat;

  typedef enum logic [3:0] {
    S_IDLE, S_INIT, S_CHK0, S_CHK1, S_CHK2, S_VAR0, S_VAR1, S_VAR2,
    S_SAT0, S_SAT1, S_SAT2, S_EMIT, S_DONE
  } state_t;
  state_t state;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  // -------------------------------------------------------------------- BRAM port muxing (comb)
  always_comb begin
    logic                excl;
    logic [MSG_BITS-1:0] exmin, mag;
    logic signed [WACC-1:0] computed, num, blend;

    mvc_ra = '0; mvc_wa = '0; mvc_wd = '0; mvc_we = 1'b0;
    ecv_ra = '0; ecv_wa = '0; ecv_wd = '0; ecv_we = 1'b0;
    excl = 1'b0; exmin = '0; mag = '0;         // defaults: keep the block latch-free for synthesis
    computed = '0; num = '0; blend = '0;

    unique case (state)
      // serial init: m_vc[e] = λ_{v(e)}  (idx doubles as edge cursor e)
      S_INIT: begin
        mvc_we = 1'b1;
        mvc_wa = AW'(idx);
        mvc_wd = signed'(BP_LAMBDA[BP_EDGE_VAR[idx]][MSG_BITS-1:0]);
      end
      // check pass-1: present read of m_vc for edge p (address only; consume happens in seq for p-1)
      S_CHK1: if (p < deg) mvc_ra = AW'(BP_CHECK_EDGES[lo + p]);
      // check pass-2: write e_cv[edge] = excluded-min·sign (all inputs registered → pure write, 1/edge)
      S_CHK2: begin
        excl  = neg ^ edge_sgn[p];
        exmin = (p == argmin_pos) ? min2 : min1;
        if (exmin == INF) exmin = '0;
        mag   = exmin - (exmin >> 3);              // α = 7/8, multiply-free
        ecv_we = 1'b1;
        ecv_wa = AW'(BP_CHECK_EDGES[lo + p]);
        ecv_wd = excl ? -$signed(mag) : $signed(mag);
      end
      // var pass-1: present read of e_cv for edge p (edges are variable-major contiguous)
      S_VAR1: if (p < deg) ecv_ra = AW'(lo + p);
      // var pass-2: pipelined read-modify-write — present read m_vc[lo+p] (p<deg) while writing the
      // blended m_vc for edge (p-1) (p>=1). Distinct addresses ⇒ no read/write-port hazard.
      S_VAR2: begin
        if (p < deg) mvc_ra = AW'(lo + p);
        if (p >= 1) begin
          computed = total - signed'(WACC'(ecv_loc[p-1]));
          num      = omgreg * computed + greg * signed'(WACC'(mvc_rd));
          blend    = num >>> FRAC_BITS;            // truncating (floor) rounding
          if (blend > signed'(WACC'(MAX_MAG)))       blend = signed'(WACC'(MAX_MAG));
          else if (blend < -signed'(WACC'(MAX_MAG))) blend = -signed'(WACC'(MAX_MAG));
          mvc_we = 1'b1;
          mvc_wa = AW'(lo + (p - 1));
          mvc_wd = blend[MSG_BITS-1:0];
        end
      end
      default: ;
    endcase
  end

  // -------------------------------------------------------------------- control FSM (seq)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      out_valid  <= 1'b0;
      valid_flag <= 1'b0;
      lat        <= '0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        // --------------------------------------------------------------- accept + latch syndrome
        S_IDLE: begin
          if (in_valid) begin
            for (int c = 0; c < BP_C; c++) s_reg[c] <= syndrome_in[c];
            for (int v = 0; v < BP_N; v++) ehat[v]  <= 1'b0;
            found <= 1'b0; do_commit <= 1'b0;
            best_w <= '1;
            ehat_w <= '0;
            leg <= '0; iter <= '0; idx <= '0; p <= 0;
            lat <= '0;
            state <= S_INIT;                       // serial m_vc = λ init
          end
        end

        // --------------------------------------------------------------- serial m_vc init
        S_INIT: begin
          if (idx == BP_E - 1) begin idx <= '0; state <= S_CHK0; end
          else idx <= idx + 1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- check setup
        S_CHK0: begin
          lo  <= BP_CHECK_OFF[idx];
          hi  <= BP_CHECK_OFF[idx + 1];
          deg <= BP_CHECK_OFF[idx + 1] - BP_CHECK_OFF[idx];
          neg <= s_reg[idx];
          min1 <= INF; min2 <= INF; argmin_pos <= 0;
          p <= 0;
          state <= S_CHK1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- check pass-1 (pipelined min-sum)
        // p is the address cursor 0..deg: present read of edge p (comb, p<deg), consume rd of edge p-1.
        S_CHK1: begin
          if (p >= 1) begin
            logic signed [MSG_BITS-1:0] m;
            logic [MSG_BITS-1:0] a;
            m = mvc_rd;
            a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);
            neg <= neg ^ m[MSG_BITS-1];
            edge_sgn[p-1] <= m[MSG_BITS-1];
            if (a < min1) begin min2 <= min1; min1 <= a; argmin_pos <= p - 1; end
            else if (a < min2) begin min2 <= a; end
          end
          if (p == deg) begin p <= 0; state <= S_CHK2; end
          else p <= p + 1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- check pass-2 (e_cv write, 1/edge)
        S_CHK2: begin
          if (p == deg - 1) begin
            p <= 0;
            if (idx == BP_C - 1) begin idx <= '0; state <= S_VAR0; end
            else begin idx <= idx + 1; state <= S_CHK0; end
          end else p <= p + 1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- variable setup
        S_VAR0: begin
          lo  <= BP_VAR_OFF[idx];
          hi  <= BP_VAR_OFF[idx + 1];
          deg <= BP_VAR_OFF[idx + 1] - BP_VAR_OFF[idx];
          total  <= signed'(WACC'(BP_LAMBDA[idx]));
          greg   <= signed'(WACC'(BP_GAMMA[leg * BP_N + idx]));
          omgreg <= signed'(WACC'(1 << FRAC_BITS)) - signed'(WACC'(BP_GAMMA[leg * BP_N + idx]));
          p <= 0;
          state <= S_VAR1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- var pass-1 (pipelined ecv accum)
        S_VAR1: begin
          if (p == deg) begin
            logic signed [WACC-1:0] total_final;
            logic newbit;
            ecv_loc[deg-1] <= ecv_rd;
            total_final = total + signed'(WACC'(ecv_rd));
            total <= total_final;
            newbit = total_final[WACC-1];          // total < 0 ⇒ decision 1
            ehat[idx] <= newbit;
            ehat_w <= (idx == 0 ? WW'(0) : ehat_w) + WW'(newbit ? 1'b1 : 1'b0);
            p <= 0; state <= S_VAR2;
          end else begin
            if (p >= 1) begin
              ecv_loc[p-1] <= ecv_rd;
              total <= total + signed'(WACC'(ecv_rd));
            end
            p <= p + 1;
          end
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- var pass-2 (pipelined blend RMW)
        S_VAR2: begin
          if (p == deg) begin                      // comb wrote the last edge's blend this cycle
            p <= 0;
            if (idx == BP_N - 1) begin idx <= '0; all_sat <= 1'b1; state <= S_SAT0; end
            else begin idx <= idx + 1; state <= S_VAR0; end
          end else p <= p + 1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- SAT setup
        S_SAT0: begin
          lo  <= BP_CHECK_OFF[idx];
          hi  <= BP_CHECK_OFF[idx + 1];
          deg <= BP_CHECK_OFF[idx + 1] - BP_CHECK_OFF[idx];
          parity <= s_reg[idx];
          p <= 0;
          state <= S_SAT1;
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- SAT scan (H·ehat == s? decide)
        // reads ehat (flops) — already 1 cyc/edge. At the last check, LATCH the commit decision instead
        // of doing the 864-wide best_e copy inline; the copy runs next cycle in S_SAT2.
        S_SAT1: begin
          logic pnow;
          pnow = parity ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + p]]];
          if (p == deg - 1) begin
            if (idx == BP_C - 1) begin
              logic final_sat;
              final_sat = all_sat & (pnow == 1'b0);  // fold the last check combinationally
              if (final_sat) found <= 1'b1;
              do_commit <= final_sat & (ehat_w < best_w);
              idx <= '0;
              state <= S_SAT2;
            end else begin
              if (pnow != 1'b0) all_sat <= 1'b0;
              idx <= idx + 1;
              state <= S_SAT0;
            end
          end else begin
            parity <= pnow;
            p <= p + 1;
          end
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- registered keep-best commit
        S_SAT2: begin
          if (do_commit) begin
            best_w <= ehat_w;
            for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
          end
          if (iter == BP_ITERS - 1) begin
            iter <= '0;
            if (leg == BP_LEGS - 1) state <= S_EMIT;
            else begin leg <= leg + 1; state <= S_CHK0; end
          end else begin
            iter <= iter + 1;
            state <= S_CHK0;
          end
          lat <= lat + 32'd1;
        end

        // --------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          logic b;
          logic [BP_OBS-1:0] msk;
          b   = found ? best_e[idx] : ehat[idx];
          msk = BP_OBS_MASK[idx][BP_OBS-1:0];
          corr_out[idx] <= b;
          obs_acc <= (idx == 0 ? {BP_OBS{1'b0}} : obs_acc) ^ (b ? msk : {BP_OBS{1'b0}});
          if (idx == BP_N - 1) state <= S_DONE;
          else idx <= idx + 1;
          lat <= lat + 32'd1;
        end

        S_DONE: begin
          obs_flip   <= obs_acc;
          valid_flag <= found;
          out_valid  <= 1'b1;
          state      <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
