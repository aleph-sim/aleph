// Q7-02 M2-BRAM-DP — dual-port, 2-edges/cycle edge-serial relay-BP decoder (circuit-level synthesizable).
//
// Functionally identical to `bp_relay_bram` / `bp_relay_bram_fast` (and the M2 reference
// `bp_relay_decoder`): same schedule, quantisation (Q5.3, multiply-free α=7/8 check-update, one
// memory-blend multiply), truncating arithmetic-shift rounding, and keep-lowest-weight-valid rule.
// Decodes the SAME graph header, verified bit-for-bit against the SAME golden (`tb_bp_relay.cpp`).
//
// WHY IT EXISTS. `bp_relay_bram_fast` (#449) got each edge-serial pass to 1 cyc/edge. To go to 2
// edges/cycle every pass needs two message accesses per cycle. The gross-code edge numbering is
// *variable-major*, so the variable passes (INIT, VAR1, VAR2) touch **contiguous** edge indices while the
// check passes (CHK1 reads m_vc, CHK2 writes e_cv) touch **scattered** ones via `BP_CHECK_EDGES` (the
// Tanner-graph transpose). A single 2-bank (even/odd) split gives conflict-free 2/cycle for the contiguous
// passes but not the scattered ones — two scattered edges can land in the same bank.
//
// FIX (this file). Make each message table **two banks, each a TRUE dual-port BRAM** (2 independent R/W
// ports). Bank = edge&1, row = edge>>1. Then even two same-bank scattered accesses use that bank's two
// ports → *every* pass runs 2 edges/cycle:
//   * INIT / CHK2   — 2 writes/cycle (slot0→portA, slot1→portB of each slot's bank).
//   * CHK1 / VAR1   — 2 pipelined reads/cycle (slot0→portA, slot1→portB); 1-cycle read latency, consume
//                     the pair presented last cycle.
//   * VAR2          — 2 pipelined read-modify-writes/cycle: portA reads the leading pair, portB writes the
//                     lagging pair's blend. Contiguous ⇒ the pair straddles both banks, so per bank portA
//                     (read) and portB (write) never collide and hit distinct rows.
//   * SAT1          — 2 ehat (flop) reads/cycle folded into the running parity.
// 4 BRAM tiles total (140 available). Per iteration 2·(BP_E/2)+… ≈ half of #449 on every pass → ~719k
// cyc/decode (2.07× the original #448's 1 489 896). Arithmetic byte-for-byte the M2 golden.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_bram_dp (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                early_exit,               // stop at the first syndrome-valid decision
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],
    output logic [BP_OBS-1:0]   obs_flip,
    output logic                valid_flag,
    output logic [31:0]         latency_cycles
);
  localparam logic [MSG_BITS-1:0] INF  = '1;
  localparam int WACC = 32;
  localparam int WW   = $clog2(BP_N + 1);
  localparam int AW   = $clog2(BP_E);
  localparam int DEPTH = (BP_E + 1) / 2;        // per-bank depth (even or odd edges)
  localparam int BW   = $clog2(DEPTH);

  // ---- message banks: m_vc = {m0 (even edges), m1 (odd)}, e_cv = {c0, c1}. Each is a TRUE dual-port
  //      BRAM: two ports (a,b), each an independent registered read + optional write.
  logic signed [MSG_BITS-1:0] m0 [DEPTH], m1 [DEPTH], c0 [DEPTH], c1 [DEPTH];
  logic [BW-1:0]              m0_aa, m0_ba, m1_aa, m1_ba, c0_aa, c0_ba, c1_aa, c1_ba;
  logic signed [MSG_BITS-1:0] m0_ad, m0_bd, m1_ad, m1_bd, c0_ad, c0_bd, c1_ad, c1_bd;
  logic                       m0_awe,m0_bwe,m1_awe,m1_bwe,c0_awe,c0_bwe,c1_awe,c1_bwe;
  logic signed [MSG_BITS-1:0] m0_aq, m0_bq, m1_aq, m1_bq, c0_aq, c0_bq, c1_aq, c1_bq;

  // TRUE dual-port inference REQUIRES one write per process (Vivado Synth 8-4767): each of the two ports
  // on a bank gets its own always_ff writing the shared array. Two processes / same array = one TDP BRAM.
  always_ff @(posedge clk) begin if (m0_awe) m0[m0_aa] <= m0_ad;  m0_aq <= m0[m0_aa]; end
  always_ff @(posedge clk) begin if (m0_bwe) m0[m0_ba] <= m0_bd;  m0_bq <= m0[m0_ba]; end
  always_ff @(posedge clk) begin if (m1_awe) m1[m1_aa] <= m1_ad;  m1_aq <= m1[m1_aa]; end
  always_ff @(posedge clk) begin if (m1_bwe) m1[m1_ba] <= m1_bd;  m1_bq <= m1[m1_ba]; end
  always_ff @(posedge clk) begin if (c0_awe) c0[c0_aa] <= c0_ad;  c0_aq <= c0[c0_aa]; end
  always_ff @(posedge clk) begin if (c0_bwe) c0[c0_ba] <= c0_bd;  c0_bq <= c0[c0_ba]; end
  always_ff @(posedge clk) begin if (c1_awe) c1[c1_aa] <= c1_ad;  c1_aq <= c1[c1_aa]; end
  always_ff @(posedge clk) begin if (c1_bwe) c1[c1_ba] <= c1_bd;  c1_bq <= c1[c1_ba]; end

  /* verilator lint_off UNUSEDSIGNAL */
  logic                       s_reg  [BP_C];
  logic                       ehat   [BP_N];
  logic                       best_e [BP_N];
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found, all_sat, do_commit;
  logic [BP_OBS-1:0]          obs_acc;

  logic                       neg;
  logic [MSG_BITS-1:0]        min1, min2;
  int                         argmin_pos;
  logic                       edge_sgn [BP_CHK_DEG];

  logic signed [WACC-1:0]     total, greg, omgreg;
  logic signed [MSG_BITS-1:0] ecv_loc [BP_VAR_DEG];

  logic                       parity;

  int          idx, p, leg, iter, lo, hi, deg;
  logic [31:0] lat;

  typedef enum logic [3:0] {
    S_IDLE, S_INIT, S_CHK0, S_CHK1, S_CHK2, S_VAR0, S_VAR1, S_VAR2,
    S_SAT0, S_SAT1, S_SAT2, S_EMIT, S_DONE
  } state_t;
  state_t state;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  // ---- abs / magnitude helper ----
  function automatic logic [MSG_BITS-1:0] mag_of(input logic signed [MSG_BITS-1:0] x);
    mag_of = x[MSG_BITS-1] ? unsigned'(-x) : unsigned'(x);
  endfunction

  // ==================================================================== BRAM port muxing (comb)
  always_comb begin
    logic [AW-1:0] e0, e1;
    logic [AW-1:0] f0, f1;                       // VAR2 lagging (write) pair
    logic signed [MSG_BITS-1:0] rdv0, rdv1;      // VAR2 old-m_vc for the lagging pair (from portA q's)
    logic signed [WACC-1:0] comp0, num0, bl0, comp1, num1, bl1;
    logic excl; logic [MSG_BITS-1:0] exmin, magv;

    m0_aa='0; m0_ba='0; m1_aa='0; m1_ba='0; c0_aa='0; c0_ba='0; c1_aa='0; c1_ba='0;
    m0_ad='0; m0_bd='0; m1_ad='0; m1_bd='0; c0_ad='0; c0_bd='0; c1_ad='0; c1_bd='0;
    m0_awe=0; m0_bwe=0; m1_awe=0; m1_bwe=0; c0_awe=0; c0_bwe=0; c1_awe=0; c1_bwe=0;
    e0='0; e1='0; f0='0; f1='0; rdv0='0; rdv1='0;
    comp0='0; num0='0; bl0='0; comp1='0; num1='0; bl1='0; excl=0; exmin='0; magv='0;

    unique case (state)
      // --- serial m_vc = λ init, 2 contiguous writes/cycle (slot0→portA, slot1→portB of each bank) ---
      S_INIT: begin
        e0 = AW'(p);
        if (!e0[0]) begin m0_awe=1; m0_aa=e0[AW-1:1]; m0_ad=signed'(BP_LAMBDA[BP_EDGE_VAR[e0]][MSG_BITS-1:0]); end
        else        begin m1_awe=1; m1_aa=e0[AW-1:1]; m1_ad=signed'(BP_LAMBDA[BP_EDGE_VAR[e0]][MSG_BITS-1:0]); end
        if (p + 1 < BP_E) begin
          e1 = AW'(p + 1);
          if (!e1[0]) begin m0_bwe=1; m0_ba=e1[AW-1:1]; m0_bd=signed'(BP_LAMBDA[BP_EDGE_VAR[e1]][MSG_BITS-1:0]); end
          else        begin m1_bwe=1; m1_ba=e1[AW-1:1]; m1_bd=signed'(BP_LAMBDA[BP_EDGE_VAR[e1]][MSG_BITS-1:0]); end
        end
      end

      // --- check pass-1: present 2 scattered m_vc reads/cycle (slot0→portA, slot1→portB) ---
      S_CHK1: if (p < deg) begin
        e0 = AW'(BP_CHECK_EDGES[lo + p]);
        if (!e0[0]) m0_aa = e0[AW-1:1]; else m1_aa = e0[AW-1:1];
        if (p + 1 < deg) begin
          e1 = AW'(BP_CHECK_EDGES[lo + p + 1]);
          if (!e1[0]) m0_ba = e1[AW-1:1]; else m1_ba = e1[AW-1:1];
        end
      end

      // --- check pass-2: write 2 scattered e_cv/cycle = excluded-min·sign ---
      S_CHK2: begin
        e0 = AW'(BP_CHECK_EDGES[lo + p]);
        excl  = neg ^ edge_sgn[p];
        exmin = (p == argmin_pos) ? min2 : min1;
        if (exmin == INF) exmin = '0;
        magv  = exmin - (exmin >> 3);
        if (!e0[0]) begin c0_awe=1; c0_aa=e0[AW-1:1]; c0_ad = excl ? -$signed(magv) : $signed(magv); end
        else        begin c1_awe=1; c1_aa=e0[AW-1:1]; c1_ad = excl ? -$signed(magv) : $signed(magv); end
        if (p + 1 < deg) begin
          e1 = AW'(BP_CHECK_EDGES[lo + p + 1]);
          excl  = neg ^ edge_sgn[p+1];
          exmin = ((p+1) == argmin_pos) ? min2 : min1;
          if (exmin == INF) exmin = '0;
          magv  = exmin - (exmin >> 3);
          if (!e1[0]) begin c0_bwe=1; c0_ba=e1[AW-1:1]; c0_bd = excl ? -$signed(magv) : $signed(magv); end
          else        begin c1_bwe=1; c1_ba=e1[AW-1:1]; c1_bd = excl ? -$signed(magv) : $signed(magv); end
        end
      end

      // --- var pass-1: present 2 contiguous e_cv reads/cycle ---
      S_VAR1: if (p < deg) begin
        e0 = AW'(lo + p);
        if (!e0[0]) c0_aa = e0[AW-1:1]; else c1_aa = e0[AW-1:1];
        if (p + 1 < deg) begin
          e1 = AW'(lo + p + 1);
          if (!e1[0]) c0_ba = e1[AW-1:1]; else c1_ba = e1[AW-1:1];
        end
      end

      // --- var pass-2: pipelined RMW. portA reads leading pair@p; portB writes lagging pair@(p-2) blend
      //     from the portA q's (which now hold pair@(p-2)) + stored ecv_loc. Contiguous ⇒ each bank does
      //     one read (portA) + one write (portB) at distinct rows. ---
      S_VAR2: begin
        if (p < deg) begin                       // read leading pair on portA
          e0 = AW'(lo + p);
          if (!e0[0]) m0_aa = e0[AW-1:1]; else m1_aa = e0[AW-1:1];
          if (p + 1 < deg) begin
            e1 = AW'(lo + p + 1);
            if (!e1[0]) m0_aa = e1[AW-1:1]; else m1_aa = e1[AW-1:1];
          end
        end
        if (p >= 2) begin                        // write lagging pair@(p-2) blend on portB
          f0 = AW'(lo + p - 2);
          rdv0 = f0[0] ? m1_aq : m0_aq;          // old m_vc of f0 (was read into portA q last cycle)
          comp0 = total - signed'(WACC'(ecv_loc[p-2]));
          num0  = omgreg * comp0 + greg * signed'(WACC'(rdv0));
          bl0   = num0 >>> FRAC_BITS;
          if (bl0 > signed'(WACC'(MAX_MAG)))       bl0 = signed'(WACC'(MAX_MAG));
          else if (bl0 < -signed'(WACC'(MAX_MAG))) bl0 = -signed'(WACC'(MAX_MAG));
          if (!f0[0]) begin m0_bwe=1; m0_ba=f0[AW-1:1]; m0_bd=bl0[MSG_BITS-1:0]; end
          else        begin m1_bwe=1; m1_ba=f0[AW-1:1]; m1_bd=bl0[MSG_BITS-1:0]; end
          if (p - 1 < deg) begin
            f1 = AW'(lo + p - 1);
            rdv1 = f1[0] ? m1_aq : m0_aq;
            comp1 = total - signed'(WACC'(ecv_loc[p-1]));
            num1  = omgreg * comp1 + greg * signed'(WACC'(rdv1));
            bl1   = num1 >>> FRAC_BITS;
            if (bl1 > signed'(WACC'(MAX_MAG)))       bl1 = signed'(WACC'(MAX_MAG));
            else if (bl1 < -signed'(WACC'(MAX_MAG))) bl1 = -signed'(WACC'(MAX_MAG));
            if (!f1[0]) begin m0_bwe=1; m0_ba=f1[AW-1:1]; m0_bd=bl1[MSG_BITS-1:0]; end
            else        begin m1_bwe=1; m1_ba=f1[AW-1:1]; m1_bd=bl1[MSG_BITS-1:0]; end
          end
        end
      end
      default: ;
    endcase
  end

  // ==================================================================== control FSM (seq)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state<=S_IDLE; out_valid<=0; valid_flag<=0; lat<=0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        S_IDLE: begin
          if (in_valid) begin
            for (int c=0;c<BP_C;c++) s_reg[c]<=syndrome_in[c];
            for (int v=0;v<BP_N;v++) ehat[v]<=1'b0;
            found<=0; do_commit<=0; best_w<='1; ehat_w<=0;
            leg<=0; iter<=0; idx<=0; p<=0; lat<=0;
            state<=S_INIT;
          end
        end

        // serial λ init, 2 writes/cycle
        S_INIT: begin
          if (p + 2 >= BP_E) begin p<=0; state<=S_CHK0; end
          else p<=p+2;
          lat<=lat+32'd1;
        end

        S_CHK0: begin
          lo<=BP_CHECK_OFF[idx]; hi<=BP_CHECK_OFF[idx+1];
          deg<=BP_CHECK_OFF[idx+1]-BP_CHECK_OFF[idx];
          neg<=s_reg[idx]; min1<=INF; min2<=INF; argmin_pos<=0;
          p<=0; state<=S_CHK1; lat<=lat+32'd1;
        end

        // check pass-1 (pipelined 2-wide min-sum). Consume pair@(p-2) from portA/portB q's.
        S_CHK1: begin
          if (p >= 2) begin
            logic [AW-1:0] f0, f1;
            logic signed [MSG_BITS-1:0] d0, d1;
            logic [MSG_BITS-1:0] a0, a1;
            logic [MSG_BITS-1:0] t1, t2; int targ;   // running top-2 after folding d0
            logic v1;
            f0 = AW'(BP_CHECK_EDGES[lo + p - 2]);
            d0 = f0[0] ? m1_aq : m0_aq;               // slot0 was read via port A
            a0 = mag_of(d0);
            neg <= neg ^ d0[MSG_BITS-1];
            edge_sgn[p-2] <= d0[MSG_BITS-1];
            // fold d0 (pos p-2) into (min1,min2,argmin) -> temp
            if (a0 < min1)      begin t1=a0;   t2=min1; targ=p-2;       end
            else if (a0 < min2) begin t1=min1; t2=a0;   targ=argmin_pos; end
            else                begin t1=min1; t2=min2; targ=argmin_pos; end
            v1 = (p - 1 < deg);
            if (v1) begin
              f1 = AW'(BP_CHECK_EDGES[lo + p - 1]);
              d1 = f1[0] ? m1_bq : m0_bq;             // slot1 was read via port B
              a1 = mag_of(d1);
              neg <= neg ^ d0[MSG_BITS-1] ^ d1[MSG_BITS-1];
              edge_sgn[p-1] <= d1[MSG_BITS-1];
              if (a1 < t1)      begin min1<=a1; min2<=t1; argmin_pos<=p-1; end
              else if (a1 < t2) begin min1<=t1; min2<=a1; argmin_pos<=targ; end
              else              begin min1<=t1; min2<=t2; argmin_pos<=targ; end
            end else begin
              min1<=t1; min2<=t2; argmin_pos<=targ;
            end
          end
          if (p >= deg) begin p<=0; state<=S_CHK2; end
          else p<=p+2;
          lat<=lat+32'd1;
        end

        // check pass-2 (2 writes/cycle, comb drives ports)
        S_CHK2: begin
          if (p + 2 >= deg) begin
            p<=0;
            if (idx == BP_C-1) begin idx<=0; state<=S_VAR0; end
            else begin idx<=idx+1; state<=S_CHK0; end
          end else p<=p+2;
          lat<=lat+32'd1;
        end

        S_VAR0: begin
          lo<=BP_VAR_OFF[idx]; hi<=BP_VAR_OFF[idx+1];
          deg<=BP_VAR_OFF[idx+1]-BP_VAR_OFF[idx];
          total<=signed'(WACC'(BP_LAMBDA[idx]));
          greg<=signed'(WACC'(BP_GAMMA[leg*BP_N+idx]));
          omgreg<=signed'(WACC'(1<<FRAC_BITS))-signed'(WACC'(BP_GAMMA[leg*BP_N+idx]));
          p<=0; state<=S_VAR1; lat<=lat+32'd1;
        end

        // var pass-1 (pipelined 2-wide e_cv accumulate). Consume pair@(p-2).
        S_VAR1: begin
          logic signed [WACC-1:0] tf;
          logic newbit; logic v1;
          logic signed [MSG_BITS-1:0] d0, d1;
          logic [AW-1:0] f0, f1;
          tf = total; v1 = (p - 1 < deg);
          if (p >= 2) begin
            f0 = AW'(lo + p - 2);
            d0 = f0[0] ? c1_aq : c0_aq;
            ecv_loc[p-2] <= d0;
            tf = total + signed'(WACC'(d0));
            if (v1) begin
              f1 = AW'(lo + p - 1);
              d1 = f1[0] ? c1_bq : c0_bq;
              ecv_loc[p-1] <= d1;
              tf = tf + signed'(WACC'(d1));
            end
          end
          if (p >= deg) begin                       // last pair consumed this cycle
            newbit = tf[WACC-1];
            ehat[idx] <= newbit;
            ehat_w <= (idx==0 ? WW'(0) : ehat_w) + WW'(newbit ? 1'b1 : 1'b0);
            total <= tf;
            p<=0; state<=S_VAR2;
          end else begin
            total <= tf;
            p<=p+2;
          end
          lat<=lat+32'd1;
        end

        // var pass-2 (pipelined 2-wide RMW; comb reads leading pair@p on portA, writes lagging pair@(p-2)
        // blend on portB). FSM just advances the cursor; termination writes the final lagging pair.
        S_VAR2: begin
          if (p >= deg) begin
            p<=0;
            if (idx == BP_N-1) begin idx<=0; all_sat<=1'b1; state<=S_SAT0; end
            else begin idx<=idx+1; state<=S_VAR0; end
          end else p<=p+2;
          lat<=lat+32'd1;
        end

        S_SAT0: begin
          lo<=BP_CHECK_OFF[idx]; hi<=BP_CHECK_OFF[idx+1];
          deg<=BP_CHECK_OFF[idx+1]-BP_CHECK_OFF[idx];
          parity<=s_reg[idx]; p<=0; state<=S_SAT1; lat<=lat+32'd1;
        end

        // SAT scan (2 ehat flop-reads/cycle folded into parity). At the last check LATCH the commit.
        S_SAT1: begin
          logic pn;
          pn = parity ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo+p]]];
          if (p + 1 < deg) pn = pn ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo+p+1]]];
          if (p + 2 >= deg) begin                    // last edge(s) of this check folded
            if (idx == BP_C-1) begin
              logic final_sat;
              final_sat = all_sat & (pn == 1'b0);
              if (final_sat) found <= 1'b1;
              do_commit <= final_sat & (ehat_w < best_w);
              idx<=0; state<=S_SAT2;
            end else begin
              if (pn != 1'b0) all_sat <= 1'b0;
              idx<=idx+1; state<=S_SAT0;
            end
          end else begin
            parity <= pn; p<=p+2;
          end
          lat<=lat+32'd1;
        end

        S_SAT2: begin
          if (do_commit) begin
            best_w <= ehat_w;
            for (int v=0;v<BP_N;v++) best_e[v] <= ehat[v];
          end
          // early_exit: `found` is set in S_SAT1 the moment an iteration's decision satisfies the
          // syndrome; with the schedule fixed we'd have exited on the first, so found here ⇒ first valid.
          if (early_exit && found) begin
            state <= S_EMIT;                       // idx already reset to 0 in S_SAT1
          end else if (iter == BP_ITERS-1) begin
            iter<=0;
            if (leg == BP_LEGS-1) state<=S_EMIT;
            else begin leg<=leg+1; state<=S_CHK0; end
          end else begin iter<=iter+1; state<=S_CHK0; end
          lat<=lat+32'd1;
        end

        S_EMIT: begin
          logic b; logic [BP_OBS-1:0] msk;
          b = found ? best_e[idx] : ehat[idx];
          msk = BP_OBS_MASK[idx][BP_OBS-1:0];
          corr_out[idx] <= b;
          obs_acc <= (idx==0 ? {BP_OBS{1'b0}} : obs_acc) ^ (b ? msk : {BP_OBS{1'b0}});
          if (idx == BP_N-1) state<=S_DONE;
          else idx<=idx+1;
          lat<=lat+32'd1;
        end

        S_DONE: begin
          obs_flip<=obs_acc; valid_flag<=found; out_valid<=1'b1; state<=S_IDLE;
        end

        default: state<=S_IDLE;
      endcase
    end
  end
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
