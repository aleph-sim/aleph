// Q7-02 M7 — standalone min-sum check-node submodule (`check_minsum`), 2-cycle pipelined, generic MW/DEG.
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
// `INF` (all-ones of width MW) is the "no message yet" sentinel exactly as in bp_relay_fast /
// bp_check_update; a degree-1 (or all-unused) check would leave the excluded min at INF, treated as 0
// (no constraint) — same as `bp_relay_fast.sv:141`.

`timescale 1ns / 1ps

module check_minsum #(
    parameter int MW  = 8,
    parameter int DEG = 25
) (
    input  logic                 clk,
    input  logic                 en,             // pulse to start; result valid 2 clocks later
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

  node_t red_node;       // tree root -> min1/min2/argmin
  logic  neg_comb;       // XOR-reduced sign fold
  logic  any_present;    // any real slot present -> argmin valid (else -1, matching the serial contract)

  always_comb begin
    automatic node_t          lvl [NLEAF];
    automatic logic [DEG-1:0] sgn_terms;   // per-slot sign contribution (present && negative)
    // leaves: real present slots carry their magnitude; all other leaves are NEUTRAL (never win a strict <)
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
    // balanced reduction: fold pairs bottom-up in place (index i reads 2i/2i+1, never a slot it clobbered)
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
