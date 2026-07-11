// Q7-02 M7/M8 — standalone min-sum check-node submodule (`check_minsum`), 2- or 3-cycle pipelined
// (parameter STAGES), generic MW/DEG.
//
// Bit-exact twin of ONE check's inner loop from `bp_relay_fast.sv`'s S_CHECK state (lines 121-145):
// min-sum with exclusive minimum via the running (min1,min2,argmin) trick, alpha=7/8 multiply-free
// scaling (`mag = exmin - (exmin>>3)`), and a per-edge sign from the check's syndrome bit XOR'd with
// each real edge's own message sign. `present[k]` masks real vs unused slots exactly like the source
// loop's `if (lo+k<hi)` guard — DEG is the graph's max check degree (BP_CHK_DEG), one module variant
// handles every check regardless of its real degree.
//
// This is the FIRST of the two small per-check/per-var submodules the M7 hierarchically-modular
// relay-BP core stamps out many times (one per check node); pipelining it (rather than leaving it one
// combinational cloud like `bp_check_update.sv`) is what makes that stamping synthesis-friendly:
//   stage 1 (posedge, gated by `en`): reduce over the DEG (present-masked) slots -> registered
//            min1/min2/argmin/neg; also register `m_in`/`present` themselves so stage 2 sees the
//            SAME snapshot the reduction ran over.
//   stage 2 (posedge, unconditional — free-running off the stage-1 registers): per real slot,
//            exclude this edge's own contribution and emit the signed check->variable message.
//            Unused slots emit 0 (no inferred latch, deterministic pad; bp_relay_fast never writes
//            e_cv for a non-real slot at all, so 0 there is a don't-care downstream — picked here
//            for a fully-defined, bit-reproducible output over the WHOLE e_out array).
//
// M8: `STAGES` (default 2, M7-compatible) optionally adds a THIRD pipeline stage — a mid-tree register
// plane inserted after tournament-tree level SPLIT_LVL=3 — to shorten the stage-1 reduction's critical
// path for higher Fmax. It changes only WHEN values land in min1_r/min2_r/argmin_r/neg_r/m_in_r/
// present_r (one clock later), never WHAT lands there: see the `gplane`/`gpass` generate split below.
// The mid-plane register is `en`-gated exactly like today's stage-1 register (so back-to-back `en`
// pulses at initiation-interval 1 keep pipelining, not latching); the (now one-cycle-later) former
// stage-1 register becomes free-running/unconditional in the STAGES==3 branch, mirroring how stage 2
// already free-runs off stage 1 today.
//
// `INF` (all-ones of width MW) is the "no message yet" sentinel exactly as in bp_relay_fast /
// bp_check_update; a degree-1 (or all-unused) check would leave the excluded min at INF, treated as 0
// (no constraint) — same as `bp_relay_fast.sv:141`.

`timescale 1ns / 1ps

module check_minsum #(
    parameter int MW     = 8,
    parameter int DEG    = 25,
    parameter int STAGES = 2   // 2 = M7-compatible; 3 = extra register plane after tree level SPLIT_LVL
) (
    input  logic                 clk,
    input  logic                 en,             // pulse to start; result valid STAGES clocks later
    input  logic                 sbit,           // this check's syndrome bit
    input  logic signed [MW-1:0] m_in    [DEG],
    input  logic                 present [DEG],  // 1 = real edge, 0 = unused slot (deg < DEG)
    output logic signed [MW-1:0] e_out   [DEG]
);
  localparam logic [MW-1:0] INF = '1;                 // NEUTRAL: all-ones = unsigned max; > any real |mag|

  // ------------------------------------------------------------------- stage 1: reduction registers
  logic [MW-1:0] min1_r, min2_r;
  int            argmin_r;
  logic          neg_r;
  logic signed [MW-1:0] m_in_r    [DEG];
  logic                 present_r [DEG];

  // ---- TOURNAMENT-TREE reduction (bit-exact twin of the serial (min1,min2,argmin) fold) ------------
  // The serial fold `if (a<min1){min2=min1;min1=a;argmin=k} else if (a<min2){min2=a}` over ascending k
  // is a DEG-deep SERIAL compare-select chain that Vivado cannot rebalance (25 logic levels here). It is
  // re-expressed as a balanced tournament tree over ceil-log2(DEG) levels. Each leaf k carries (m1,m2,idx):
  // present real slots hold their magnitude, everything else holds NEUTRAL (=INF, > any real |mag| since
  // |mag| <= 2^(MW-1) < 2^MW-1). Merge keeps the STRICT minimum (a tie keeps the LEFT operand = lower k =
  // the "first occurrence" the serial `<` selects) and folds the loser's minimum into m2.
  //   DUPLICATE proof: if L.m1==R.m1, `R.m1 < L.m1` is FALSE so winner=L (idx=L.idx, the lower k), loser=R;
  //   merged.m2 = min(loser.m1, winner.m2) = min(R.m1, L.m2) = min(L.m1, L.m2) = L.m1 (m2>=m1 always) = the
  //   duplicate value — exactly the serial result (min1==min2 when two present slots tie).
  //   SECOND-ORDER-STATISTIC proof: winner.m1 is the subtree minimum; the next-smallest of the union is
  //   min(winner.m2, loser.m1) because every loser element is >= loser.m1 and the winner's runner-up is m2.
  localparam int IDXW  = (DEG > 1) ? $clog2(DEG) : 1;
  localparam int NLEAF = 1 << ((DEG > 1) ? $clog2(DEG) : 0);   // pad leaves up to a power of two

  // M8 mid-tree register plane (STAGES==3 only): split the NLEAF-leaf tree after SPLIT_LVL span-halving
  // folds (levels 1..SPLIT_LVL before the plane, SPLIT_LVL+1..NLVL after — NLVL=$clog2(NLEAF)=5 for the
  // DEG=25 graph this module is actually instantiated at, so SPLIT_LVL=3 always leaves >=1 level after).
  localparam int SPLIT_LVL = 3;
  localparam int PLANE_SZ  = NLEAF >> SPLIT_LVL;   // node count registered at the plane (4 for DEG=25)

  typedef struct packed {
    logic [MW-1:0]   m1;      // smallest magnitude in this subtree (NEUTRAL if none present)
    logic [MW-1:0]   m2;      // second-smallest counting duplicates (NEUTRAL if <2 present)
    logic [IDXW-1:0] idx;     // leaf index attaining m1 (lowest-k on ties)
  } node_t;

  function automatic logic [MW-1:0] mag_of(input logic signed [MW-1:0] m);
    return m[MW-1] ? unsigned'(-m) : unsigned'(m);   // identical to the serial |m|, incl. the -2^(MW-1) wrap
  endfunction

  function automatic node_t merge_nodes(input node_t l, input node_t r);
    node_t o;
    logic [MW-1:0] loser_m1, winner_m2;
    if (r.m1 < l.m1) begin          // STRICT: equal magnitudes keep LEFT (lower k = first occurrence)
      o.m1 = r.m1; o.idx = r.idx; winner_m2 = r.m2; loser_m1 = l.m1;
    end else begin
      o.m1 = l.m1; o.idx = l.idx; winner_m2 = l.m2; loser_m1 = r.m1;
    end
    o.m2 = (loser_m1 < winner_m2) ? loser_m1 : winner_m2;
    return o;
  endfunction

  // M8: the reduction+stage-1-register plumbing generate-splits on STAGES. `gpass` (STAGES==2) is
  // TODAY'S code verbatim (byte-for-byte, unchanged) so the default-parameter consumers
  // (bp_relay_unroll_pipe.sv, bp_unroll_skeleton.sv) elaborate to bit-identical logic. `gplane`
  // (STAGES==3) is a self-contained twin that inserts one extra `en`-gated register plane after
  // tournament-tree level SPLIT_LVL, then free-runs the (renamed-in-time) former stage-1 register one
  // cycle later off that plane — same merges, same tie-break order, same values, one clock later.
  generate
    if (STAGES == 2) begin : gpass
      node_t red_node;       // tree root -> min1/min2/argmin
      logic  neg_comb;       // XOR-reduced sign fold
      logic  any_present;    // any real slot present -> argmin valid (else -1, matching serial contract)

      always_comb begin
        automatic node_t          lvl [NLEAF];
        automatic logic [DEG-1:0] sgn_terms;   // per-slot sign contribution (present && negative)
        // leaves: real present slots carry their magnitude; all other leaves are NEUTRAL (never win a
        // strict <)
        for (int k = 0; k < NLEAF; k++) begin
          if (k < DEG && present[k]) begin
            lvl[k].m1  = mag_of(m_in[k]);
            lvl[k].m2  = INF;
            lvl[k].idx = IDXW'(k);
          end else begin
            lvl[k].m1  = INF;
            lvl[k].m2  = INF;
            lvl[k].idx = (k < DEG) ? IDXW'(k) : '0;
          end
        end
        // balanced reduction: fold pairs bottom-up in place (index i reads 2i/2i+1, never a slot it
        // clobbered)
        for (int span = NLEAF; span > 1; span = span >> 1)
          for (int i = 0; i < (span >> 1); i++)
            lvl[i] = merge_nodes(lvl[2*i], lvl[2*i+1]);
        red_node = lvl[0];
        // neg = sbit XOR (XOR of present slots' sign bits): a pure XOR reduction, balanced by synthesis
        for (int k = 0; k < DEG; k++)
          sgn_terms[k] = present[k] & m_in[k][MW-1];
        neg_comb = sbit ^ (^sgn_terms);
        // argmin contract: -1 when no real slot present (a don't-care downstream, but kept value-exact)
        any_present = 1'b0;
        for (int k = 0; k < DEG; k++) any_present = any_present | present[k];
      end

      always_ff @(posedge clk) begin
        if (en) begin
          min1_r   <= red_node.m1;
          min2_r   <= red_node.m2;
          argmin_r <= any_present ? int'(red_node.idx) : -1;
          neg_r    <= neg_comb;
          for (int k = 0; k < DEG; k++) begin
            m_in_r[k]    <= m_in[k];
            present_r[k] <= present[k];
          end
        end
      end
    end else begin : gplane
      // ---- plane-side combinational: leaves + first SPLIT_LVL folds -> lvl3_arr[PLANE_SZ] -----------
      node_t lvl3_arr [PLANE_SZ];

      always_comb begin
        automatic node_t lvl [NLEAF];
        for (int k = 0; k < NLEAF; k++) begin
          if (k < DEG && present[k]) begin
            lvl[k].m1  = mag_of(m_in[k]);
            lvl[k].m2  = INF;
            lvl[k].idx = IDXW'(k);
          end else begin
            lvl[k].m1  = INF;
            lvl[k].m2  = INF;
            lvl[k].idx = (k < DEG) ? IDXW'(k) : '0;
          end
        end
        for (int span = NLEAF; span > PLANE_SZ; span = span >> 1)
          for (int i = 0; i < (span >> 1); i++)
            lvl[i] = merge_nodes(lvl[2*i], lvl[2*i+1]);
        for (int i = 0; i < PLANE_SZ; i++) lvl3_arr[i] = lvl[i];
      end

      // ---- the mid-tree register plane itself: `en`-gated, exactly like today's (sole) stage-1 reg --
      // registers the partial node array PLUS the sbit/m_in/present pass-throughs the emit stage needs,
      // so the continuation below (and the free-running register after it) sees one aligned snapshot.
      node_t                lvl3_r        [PLANE_SZ];
      logic                 sbit_mid_r;
      logic signed [MW-1:0] m_in_mid_r    [DEG];
      logic                 present_mid_r [DEG];

      always_ff @(posedge clk) begin
        if (en) begin
          for (int i = 0; i < PLANE_SZ; i++) lvl3_r[i] <= lvl3_arr[i];
          sbit_mid_r <= sbit;
          for (int k = 0; k < DEG; k++) begin
            m_in_mid_r[k]    <= m_in[k];
            present_mid_r[k] <= present[k];
          end
        end
      end

      // ---- continue levels SPLIT_LVL+1..NLVL combinationally off the REGISTERED plane -------------
      node_t red_node3;
      logic  neg_comb3;
      logic  any_present3;

      always_comb begin
        automatic node_t          lvl2 [PLANE_SZ];
        automatic logic [DEG-1:0] sgn_terms3;
        for (int i = 0; i < PLANE_SZ; i++) lvl2[i] = lvl3_r[i];
        for (int span = PLANE_SZ; span > 1; span = span >> 1)
          for (int i = 0; i < (span >> 1); i++)
            lvl2[i] = merge_nodes(lvl2[2*i], lvl2[2*i+1]);
        red_node3 = lvl2[0];
        for (int k = 0; k < DEG; k++)
          sgn_terms3[k] = present_mid_r[k] & m_in_mid_r[k][MW-1];
        neg_comb3 = sbit_mid_r ^ (^sgn_terms3);
        any_present3 = 1'b0;
        for (int k = 0; k < DEG; k++) any_present3 = any_present3 | present_mid_r[k];
      end

      // ---- former stage-1 register, now free-running (unconditional) one cycle further out ---------
      // `en` already gated the mid-plane capture above; this plane just advances every clock (mirrors
      // how stage 2 already free-runs off stage 1 today) so back-to-back `en` pulses (initiation
      // interval 1) stay pipelined instead of latching a stale value.
      always_ff @(posedge clk) begin
        min1_r   <= red_node3.m1;
        min2_r   <= red_node3.m2;
        argmin_r <= any_present3 ? int'(red_node3.idx) : -1;
        neg_r    <= neg_comb3;
        for (int k = 0; k < DEG; k++) begin
          m_in_r[k]    <= m_in_mid_r[k];
          present_r[k] <= present_mid_r[k];
        end
      end
    end
  endgenerate

  // ------------------------------------------------------------------- stage 2: emit (free-running)
  always_ff @(posedge clk) begin
    for (int k = 0; k < DEG; k++) begin
      if (present_r[k]) begin
        automatic logic signed [MW-1:0] m;
        automatic logic                 excl;
        automatic logic [MW-1:0]        exmin, mag;

        m     = m_in_r[k];
        excl  = (m < 0) ? ~neg_r : neg_r;
        exmin = (k == argmin_r) ? min2_r : min1_r;
        if (exmin == INF) exmin = '0;
        mag = exmin - (exmin >> 3);           // alpha = 7/8, multiply-free
        e_out[k] <= excl ? -$signed(mag) : $signed(mag);
      end else begin
        e_out[k] <= '0;
      end
    end
  end
endmodule
