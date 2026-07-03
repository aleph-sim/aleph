// Q7-02 M4 — SPATIALLY-UNROLLED fixed-point relay-BP decoder for the gross BB code [[144,12,12]].
//
// M2 (`bp_relay_decoder.sv`) time-multiplexed the relay-BP schedule with a *runtime* node cursor
// `idx`: one check (S_CHECK) or one variable (S_VAR) per cycle. M3 synth found that cursor to be the
// wall — reading `BP_CHECK_OFF[idx]`/`BP_CHECK_EDGES[off+k]` with a runtime index synthesises to a
// 72-way select feeding the min-sum whose result then demuxes across 432 edges (net fanout 586/626,
// 44 logic levels, 74% routing). Not the multiply — the DSP count was 0 — the *cursor mux*.
//
// M4 deletes the cursor by SPATIAL UNROLLING: every check's 6 edges and every variable's 3 edges are
// baked-in CONSTANTS (the loop bounds `BP_CHECK_OFF[c]` etc. are elaboration-time constants once `c`
// is a genvar-like compile-time loop index), so the address decode collapses to constant wiring. A
// whole BP layer — ALL 72 checks, then ALL 144 variables — evaluates in ONE cycle each, exactly the
// combinational cloud of `bp_check_update.sv` registered. The schedule becomes
//
//   S_CHECK (1 cyc): e_cv  ← min-sum(m_vc, s)          for all 72 checks in parallel
//   S_VAR   (1 cyc): m_vc,ehat ← var-update-with-memory for all 144 variables in parallel
//   S_SAT   (1 cyc): all_sat ← (H·ehat == s); keep lowest-weight-valid ehat   (all 72 checks)
//
// looped BP_LEGS×BP_ITERS = 4×25, then S_EMIT (1 cyc, all vars) → S_DONE. Worst-case latency drops
// from 28 944 to 4·25·3 + 1 accept + 1 emit ≈ 302 cycles — ~96× fewer — AND the per-cycle path is now
// shallow constant-wired arithmetic, so Fmax rises too (the M3 diagnosis).
//
// Bit-exactness with M2 (hence the M0/M1 golden `FixedRelayBp`) is structural, not incidental: within
// each M2 phase the nodes are already independent — every Tanner edge belongs to exactly one check
// (check-major CSR) and exactly one variable (var-major CSR), checks read only `m_vc` (fully written
// by the previous S_CHECK), variables read only `e_cv` (fully written by this S_CHECK) and the *old*
// `m_vc`. So folding M2's 72/144 sequential cycles of a phase into one parallel cycle changes nothing
// but timing. Same arithmetic verbatim: multiply-free α=7/8 (`mag-(mag>>3)`), one relay blend multiply
// `(1−γ)·computed + γ·m_old` with per-var constant γ, truncating arith-shift rounding, ±MAX_MAG clamp,
// keep-lowest-weight-valid across all legs. Verified in Verilator (`tb_bp_relay.cpp` with -DUNROLL)
// bit-for-bit against `FixedRelayBp::decode_fixed_ehat` on the same vectors M2 passed.

`timescale 1ns / 1ps
/* verilator lint_off UNUSEDPARAM */
`include "bb_gross_tanner.svh"
/* verilator lint_on UNUSEDPARAM */

module bp_relay_unrolled (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                in_valid,
    input  logic                syndrome_in [BP_C],
    output logic                busy,
    output logic                out_valid,
    output logic                corr_out    [BP_N],       // chosen error pattern (one bit / variable)
    output logic [BP_OBS-1:0]   obs_flip,                 // predicted observable flips
    output logic                valid_flag,               // a syndrome-valid decision was found
    output logic [15:0]         latency_cycles
);
  // All-ones word (> every real magnitude) is the +∞ sentinel for the running minima — matches the
  // Rust golden's i32::MAX. Magnitudes are ≤ MAX_MAG (127) at Q5.3.
  localparam logic [MSG_BITS-1:0] INF = '1;
  // M5 Fmax lever #1 (free — no cycle cost): right-size the blend accumulator. M4 used WACC=32 (M2's
  // generous width), so every `total`/`computed`/`num` add was a 32-bit CARRY chain (4 CARRY8 each) —
  // a big part of the 25-level / 5-CARRY8 S_VAR critical path. The blend never exceeds ~5 600 in
  // magnitude (|omg·computed + γ·old| ≤ 10·472 + 7·127), which fits signed 16 bits with 5× margin, so
  // 16 bits is bit-exact with 32 (verified in Verilator) at half the CARRY depth.
  localparam int WACC = 16;
  localparam int WW   = $clog2(BP_N + 1);

  typedef enum logic [2:0] { S_IDLE, S_CHECK, S_VAR, S_SAT, S_EMIT, S_DONE } state_t;
  state_t state;

  // `dont_touch` on the message/decision registers is load-bearing. With the spatial unroll every
  // array index is a compile-time constant, which lets Vivado's sequential constant-propagation chase
  // the 100-iteration message feedback to a (wrong) fixpoint and fold the whole datapath to `ehat≡0`,
  // leaving a 158-FF shell (a false ~485 MHz on nothing). Verilator computes the real syndrome-
  // dependent decode (65/65 bit-exact), so this is over-aggressive synth optimization, not an RTL bug
  // — M2 dodged it only because its runtime `idx` mux blocked the const-prop. Anchoring the datapath
  // registers disables that fold without changing behaviour; the reachable logic is still fully
  // optimised. (Q6 UF decoders never needed this: their per-cycle graphs were tiny.)
  /* verilator lint_off UNUSEDSIGNAL */
  (* dont_touch = "true" *) logic                       s_reg  [BP_C];
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] m_vc   [BP_E];   // variable→check
  (* dont_touch = "true" *) logic signed [MSG_BITS-1:0] e_cv   [BP_E];   // check→variable
  (* dont_touch = "true" *) logic                       ehat   [BP_N];   // current hard decision
  (* dont_touch = "true" *) logic                       best_e [BP_N];   // lowest-weight syndrome-valid decision seen
  logic [WW-1:0]              ehat_w, best_w;
  logic                       found;
  logic [BP_OBS-1:0]          obs_acc;

  int          leg, iter;
  logic [15:0] lat;

  assign busy = (state != S_IDLE);
  assign latency_cycles = lat;

  // SYNCHRONOUS reset (not async). M2 used `@(posedge clk or negedge rst_n)`, which makes every
  // datapath FF an async-reset FF whose S_IDLE constant load (m_vc←λ) collides with the async reset —
  // Vivado flags this as "Set and reset with same priority … may cause simulation mismatches"
  // (Synth 8-7137) and resolves it by tying `s_reg`/`m_vc` to a constant. M2 got away with it because
  // it reads `s_reg[idx]` at a RUNTIME index, which blocks constant propagation; M4 reads `s_reg[c]`
  // at CONSTANT (unrolled) indices, so the tie-to-constant const-folds the ENTIRE message datapath to
  // a 158-FF shell (a false 485 MHz on nothing). A plain synchronous reset removes the async set/reset
  // conflict → the FFs load their real data → the datapath survives. It is also the FPGA-preferred
  // reset style. Verified: Verilator still bit-exact 65/65 after the change.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      out_valid  <= 1'b0;
      valid_flag <= 1'b0;
      lat        <= '0;
    end else begin
      out_valid <= 1'b0;
      unique case (state)
        // ----------------------------------------------------------------- accept + init
        S_IDLE: begin
          if (in_valid) begin
            for (int c = 0; c < BP_C; c++) s_reg[c] <= syndrome_in[c];
            // Init M_{v→c} = λ_v (quantised); messages relay across legs.
            for (int e = 0; e < BP_E; e++)
              m_vc[e] <= signed'(BP_LAMBDA[BP_EDGE_VAR[e]][MSG_BITS-1:0]);
            for (int v = 0; v < BP_N; v++) ehat[v] <= 1'b0;
            found  <= 1'b0;
            best_w <= '1;
            ehat_w <= '0;
            leg <= '0; iter <= '0;
            lat <= '0;
            state <= S_CHECK;
          end
        end

        // ----------------------------------------------------------------- ALL checks → variable (min-sum)
        // 72 parallel min-sum units; edges are constant-wired (no idx cursor). Identical two-pass
        // exclusive-minimum as M2/M1, per check.
        S_CHECK: begin
          // `automatic` is load-bearing: these are per-node combinational temporaries, freshly
          // computed each loop iteration. As STATIC vars in an `always_ff` (the default) Vivado infers
          // them as registers and — under the spatial `for` unroll — mis-synthesises the datapath,
          // pruning the whole message array to a shell (68 LUTs / 158 FFs, a false 485 MHz). `automatic`
          // gives each unrolled iteration its own scope → pure wires → the 72 min-sum units survive.
          automatic int lo, hi, argmin, e;
          automatic logic neg, excl;
          automatic logic [MSG_BITS-1:0] min1, min2, a, exmin, mag;
          automatic logic signed [MSG_BITS-1:0] m;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            neg = s_reg[c];
            min1 = INF; min2 = INF; argmin = -1;
            // pass 1: sign, two smallest magnitudes, argmin
            for (int k = 0; k < BP_CHK_DEG; k++) begin
              if (lo + k < hi) begin
                e = BP_CHECK_EDGES[lo + k];
                m = m_vc[e];
                if (m < 0) neg = ~neg;
                a = m[MSG_BITS-1] ? unsigned'(-m) : unsigned'(m);
                if (a < min1) begin min2 = min1; min1 = a; argmin = e; end
                else if (a < min2) begin min2 = a; end
              end
            end
            // pass 2: exclude each edge's own contribution
            for (int k = 0; k < BP_CHK_DEG; k++) begin
              if (lo + k < hi) begin
                e = BP_CHECK_EDGES[lo + k];
                m = m_vc[e];
                excl  = (m < 0) ? ~neg : neg;
                exmin = (e == argmin) ? min2 : min1;
                if (exmin == INF) exmin = '0;
                mag = exmin - (exmin >> 3);          // α = 7/8, multiply-free
                e_cv[e] <= excl ? -$signed(mag) : $signed(mag);
              end
            end
          end
          state <= S_VAR;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- ALL variables → check + memory
        // 144 parallel var-update units, each the one relay blend multiply (per-var constant γ).
        S_VAR: begin
          automatic int lo, hi, e, wsum;      // `automatic`: per-node combinational temps (see S_CHECK)
          automatic logic newbit;
          automatic logic signed [WACC-1:0] total, g, omg, ev, old, computed, num, blend;
          wsum = 0;
          for (int v = 0; v < BP_N; v++) begin
            lo = BP_VAR_OFF[v];
            hi = BP_VAR_OFF[v + 1];
            total = signed'(WACC'(BP_LAMBDA[v]));
            for (int k = 0; k < BP_VAR_DEG; k++)
              if (lo + k < hi) total = total + signed'(WACC'(e_cv[lo + k]));
            newbit = total[WACC-1];                  // total < 0 ⇒ decision 1
            ehat[v] <= newbit;
            wsum = wsum + (newbit ? 1 : 0);
            g   = signed'(WACC'(BP_GAMMA[leg * BP_N + v]));
            omg = signed'(WACC'(1 << FRAC_BITS)) - g;     // (1−γ) in 2^F units
            for (int k = 0; k < BP_VAR_DEG; k++) begin
              if (lo + k < hi) begin
                e = lo + k;                          // edges are variable-major (contiguous)
                ev  = signed'(WACC'(e_cv[e]));
                old = signed'(WACC'(m_vc[e]));
                computed = total - ev;
                num   = omg * computed + g * old;
                blend = num >>> FRAC_BITS;           // truncating (floor) rounding
                if (blend > signed'(WACC'(MAX_MAG)))       blend = signed'(WACC'(MAX_MAG));
                else if (blend < -signed'(WACC'(MAX_MAG))) blend = -signed'(WACC'(MAX_MAG));
                m_vc[e] <= blend[MSG_BITS-1:0];
              end
            end
          end
          ehat_w <= WW'(wsum);
          state  <= S_SAT;
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- H·ehat == s ? keep best
        // All 72 parity checks in one cycle; combinational all_sat (no cross-cycle carry).
        S_SAT: begin
          automatic int lo, hi;               // `automatic`: per-check combinational temps (see S_CHECK)
          automatic logic p, sat;
          sat = 1'b1;
          for (int c = 0; c < BP_C; c++) begin
            lo = BP_CHECK_OFF[c];
            hi = BP_CHECK_OFF[c + 1];
            p = s_reg[c];
            for (int k = 0; k < BP_CHK_DEG; k++)
              if (lo + k < hi) p = p ^ ehat[BP_EDGE_VAR[BP_CHECK_EDGES[lo + k]]];
            if (p != 1'b0) sat = 1'b0;
          end
          if (sat) begin
            found <= 1'b1;
            if (ehat_w < best_w) begin
              best_w <= ehat_w;
              for (int v = 0; v < BP_N; v++) best_e[v] <= ehat[v];
            end
          end
          // advance iteration / leg (relay-BP keeps best across ALL legs — no early exit)
          if (iter == BP_ITERS - 1) begin
            iter <= '0;
            if (leg == BP_LEGS - 1) state <= S_EMIT;
            else begin leg <= leg + 1; state <= S_CHECK; end
          end else begin
            iter <= iter + 1;
            state <= S_CHECK;
          end
          lat <= lat + 16'd1;
        end

        // ----------------------------------------------------------------- reduce chosen ehat → obs
        S_EMIT: begin
          automatic logic [BP_OBS-1:0] acc;   // `automatic`: per-var combinational temps (see S_CHECK)
          automatic logic b;
          acc = '0;
          for (int v = 0; v < BP_N; v++) begin
            b = found ? best_e[v] : ehat[v];
            corr_out[v] <= b;
            if (b) acc = acc ^ BP_OBS_MASK[v][BP_OBS-1:0];
          end
          obs_acc <= acc;
          state <= S_DONE;
          lat <= lat + 16'd1;
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
