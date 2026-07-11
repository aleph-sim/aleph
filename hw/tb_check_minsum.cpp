// Q7-02 M7/M8 — standalone Verilator testbench for the `check_minsum` submodule.
//
// `check_minsum` is a bit-exact, STAGES-cycle-pipelined (2 by default, 3 with M8's optional mid-tree
// register plane) twin of ONE check's inner loop from `bp_relay_fast.sv`'s S_CHECK state (lines
// 121-145): min-sum with exclusive minimum via the running (min1,min2,argmin) trick, alpha=7/8
// multiply-free scaling, and a per-edge sign from the syndrome bit XOR'd with each edge's own message
// sign. This TB reimplements that arithmetic directly in C++ (`reference()` below — no golden-vector
// file, no dependency on the Rust crate) and drives the RTL with >=10000 random (m_in, present, sbit)
// cases, exercising random unused slots (present[k]=0), and asserts every one of the DEG registered
// `e_out` lanes is bit-identical to the reference. STAGES only changes latency, never the value, so this
// TB is identical at both pipeline depths save for how many clocks it waits (LATENCY, see below).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "Vcheck_minsum.h"
#include "verilated.h"

// Must match the module's parameter defaults (MW=8, DEG=25) and the real graph's MSG_BITS/BP_CHK_DEG
// (see hw/bb_gross_tanner.svh / hw/bb_circuit_tanner.svh: MSG_BITS=8, BP_CHK_DEG=25 for the circuit
// graph — the largest check degree the M7 core needs to instantiate this module for).
static constexpr int MW  = 8;
static constexpr int DEG = 25;
static constexpr int INF = (1 << MW) - 1;  // MW-wide all-ones sentinel, matching bp_relay_fast's INF

// Q7-02 M8: check_minsum's pipeline depth (module parameter STAGES) is chosen at Verilator elaboration
// time via `-GSTAGES=N`; LATENCY must be passed in lockstep via `-CFLAGS -DLATENCY=N` so this TB waits
// the right number of clocks after `en` before sampling e_out. The reference model is untouched — the
// computed VALUES never depend on STAGES, only when they land.
#ifndef LATENCY
#define LATENCY 2
#endif

static Vcheck_minsum *top;

static void tick() {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

// Sign-extend the low `w` bits of `v` to a full int (models a Verilog `logic signed [w-1:0]`).
static int sext(int v, int w) {
  int shift = 32 - w;
  return (int)((int32_t)(v << shift) >> shift);
}

// Two's-complement negate-then-reinterpret-as-unsigned over `w` bits, exactly matching the RTL's
// `a = m[MW-1] ? unsigned'(-m) : unsigned'(m);` (including the m == -2^(w-1) wraparound case, which
// bp_check_update.sv documents as "shouldn't happen in practice" but which is still well-defined HW
// behavior — this TB does not special-case it away).
static int magnitude(int m, int w) {
  uint32_t mask = (1u << w) - 1;
  if (m >= 0) return m & mask;
  uint32_t um = ((uint32_t)m) & mask;   // m's w-bit two's-complement bit pattern
  return (uint32_t)(-um) & mask;        // negate mod 2^w, reinterpret as unsigned
}

// Bit-exact reimplementation of bp_relay_fast.sv S_CHECK (lines 121-145) for ONE check of up to DEG
// edges. `present[k]` mirrors the source loop's `if (lo+k<hi)` guard — gates BOTH the reduction pass
// AND the emit pass (an unused slot never contributes and never gets a "real" output; this TB defines
// its output as 0, matching the RTL's deterministic pad for the unused lanes).
static void reference(const std::vector<int> &m_in, const std::vector<int> &present, bool sbit,
                       std::vector<int> &e_out) {
  int min1 = INF, min2 = INF, argmin = -1;
  bool neg = sbit;

  for (int k = 0; k < DEG; ++k) {
    if (!present[k]) continue;
    int m = m_in[k];
    if (m < 0) neg = !neg;
    int a = magnitude(m, MW);
    if (a < min1) {
      min2   = min1;
      min1   = a;
      argmin = k;
    } else if (a < min2) {
      min2 = a;
    }
  }

  for (int k = 0; k < DEG; ++k) {
    if (!present[k]) {
      e_out[k] = 0;
      continue;
    }
    int m     = m_in[k];
    bool excl = (m < 0) ? !neg : neg;
    int exmin = (k == argmin) ? min2 : min1;
    if (exmin == INF) exmin = 0;
    int mag = exmin - (exmin >> 3);  // alpha = 7/8, multiply-free
    e_out[k] = excl ? -mag : mag;
  }
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  const int NCASES = (argc > 1) ? std::atoi(argv[1]) : 10000;
  std::mt19937 rng(2024);
  // Full MW-bit signed range, INCLUDING the -2^(MW-1) edge (see magnitude() above) — the module must
  // stay bit-exact even there, not just over the "realistic" saturated [-127,127] operating range.
  std::uniform_int_distribution<int> mdist(-(1 << (MW - 1)), (1 << (MW - 1)) - 1);
  std::uniform_int_distribution<int> bit(0, 1);
  // Vary how many of the DEG slots are "real" per case (a degree-1 check is a documented corner case),
  // then shuffle which slots those are, so both low- and full-degree checks and scattered gaps show up.
  std::uniform_int_distribution<int> degdist(1, DEG);

  top = new Vcheck_minsum;
  top->clk = 0;
  top->en  = 0;
  for (int k = 0; k < DEG; ++k) {
    top->m_in[k]    = 0;
    top->present[k] = 0;
  }
  top->eval();

  int checked = 0, mismatches = 0, cases_ok = 0;

  for (int t = 0; t < NCASES; ++t) {
    std::vector<int> m_in(DEG), present(DEG, 0), e_out(DEG);
    bool sbit = bit(rng);

    int deg = degdist(rng);
    std::vector<int> slots(DEG);
    for (int k = 0; k < DEG; ++k) slots[k] = k;
    std::shuffle(slots.begin(), slots.end(), rng);
    for (int i = 0; i < deg; ++i) present[slots[i]] = 1;

    for (int k = 0; k < DEG; ++k) m_in[k] = mdist(rng);

    reference(m_in, present, sbit, e_out);

    // Drive one `en` pulse, then wait LATENCY clocks (per the module's documented STAGES-clock latency:
    // LATENCY=2 for the default 2-stage tree, LATENCY=3 when built with -GSTAGES=3's extra mid-tree
    // register plane).
    top->sbit = sbit ? 1 : 0;
    for (int k = 0; k < DEG; ++k) {
      top->m_in[k]    = sext(m_in[k], MW);
      top->present[k] = present[k] ? 1 : 0;
    }
    top->en = 1;
    tick();  // first clock: `en` pulse captured by the reduction's (first) register stage
    top->en = 0;
    for (int c = 1; c < LATENCY; ++c) tick();  // remaining clocks: free-running stages drain to e_out

    bool case_ok = true;
    for (int k = 0; k < DEG; ++k) {
      int got = sext((int)top->e_out[k], MW);
      ++checked;
      if (got != e_out[k]) {
        case_ok = false;
        if (mismatches < 10)
          std::fprintf(stderr, "  case %d slot %d: RTL %d != reference %d (present=%d)\n", t, k, got,
                       e_out[k], present[k]);
        ++mismatches;
      }
    }
    if (case_ok) ++cases_ok;
  }

  top->final();
  delete top;

  if (mismatches == 0) {
    std::printf(
        "PASS: %d/%d check_minsum (LATENCY=%d) outputs bit-identical to bp_relay_fast reference\n",
        cases_ok, NCASES, LATENCY);
    return 0;
  }
  std::printf("FAIL: %d/%d check_minsum test cases had a mismatch (%d/%d individual outputs)\n",
              NCASES - cases_ok, NCASES, mismatches, checked);
  return 1;
}
