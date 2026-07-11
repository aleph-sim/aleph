// Q7-02 M7 — standalone Verilator testbench for the `var_update` submodule.
//
// `var_update` is a bit-exact, 2-cycle-pipelined twin of ONE variable's inner loop from
// `bp_relay_fast.sv`'s S_VAR state (lines 153-182): `total = lambda + sum_present e_cv`;
// `ehat_bit = total`'s sign bit (`total[WACC-1]`); per real edge `computed = total - e_cv[edge]`,
// `num = omg*computed + gam*old` (`omg = (1<<FRAC)-gam`, `old` = the edge's OWN current m_vc), right
// shift by FRAC, clamp to +-MAXMAG. This TB reimplements that arithmetic directly in C++
// (`reference()` below — no golden-vector file, no dependency on the Rust crate) and drives the RTL
// with >=10000 random (lam, gam, e_in, m_in, present) cases, exercising random unused slots
// (present[k]=0) and driving inputs that saturate the +-MAX_MAG clamp, asserting every one of the DEG
// registered `m_out` lanes plus `ehat_bit` are bit-identical to the reference.
//
// IMPORTANT width note (see var_update.sv's header comment): bp_relay_fast declares ALL of
// total/g/omg/ev/old/computed/num/blend at WACC(=16) bits — including `num`'s multiply-accumulate,
// which is NOT double-width. For MW=8 / DEG=6 operating magnitudes, `omg*computed` alone can already
// exceed +-32767, so the real RTL wraps that product mod 2^16 before the FRAC shift and clamp. The
// reference below reproduces that exactly (truncate-to-16-bits before the arithmetic shift), rather
// than computing in wider-than-RTL precision, so it stays bit-exact with the actual synthesized
// behavior instead of an idealized one.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "Vvar_update.h"
#include "verilated.h"

// Must match the module's parameter defaults (MW=8, WACC=16, FRAC=3, DEG=6, MAXMAG=127) and the real
// circuit graph's MSG_BITS/BP_VAR_DEG/MAX_MAG/FRAC_BITS (see hw/bb_circuit_tanner.svh: MSG_BITS=8,
// BP_VAR_DEG=6 — the largest variable degree the M7 core needs to instantiate this module for).
static constexpr int MW     = 8;
static constexpr int WACC   = 16;
static constexpr int FRAC   = 3;
static constexpr int DEG    = 6;
static constexpr int MAXMAG = 127;

static Vvar_update *top;

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

// Bit-exact reimplementation of bp_relay_fast.sv S_VAR (lines 153-182) for ONE variable of up to DEG
// edges. `present[k]` mirrors the source loop's `if (lo+k<hi)` guard. Unused slots emit 0, matching
// the RTL's deterministic pad for lanes it never writes in the source FSM.
static void reference(int lam, int gam, const std::vector<int> &e_in, const std::vector<int> &m_in,
                       const std::vector<int> &present, std::vector<int> &m_out, bool &ehat_bit) {
  int total = sext(lam, MW);
  for (int k = 0; k < DEG; ++k)
    if (present[k]) total = sext(total + sext(e_in[k], MW), WACC);
  ehat_bit = total < 0;  // total[WACC-1], total already a correctly sign-extended WACC-bit value

  int g   = sext(gam, MW);
  int omg = sext((1 << FRAC) - g, WACC);

  for (int k = 0; k < DEG; ++k) {
    if (!present[k]) {
      m_out[k] = 0;
      continue;
    }
    int ev       = sext(e_in[k], MW);
    int old      = sext(m_in[k], MW);
    int computed = sext(total - ev, WACC);
    // bp_relay_fast declares `num` at WACC bits too -> the multiply-accumulate itself wraps mod 2^16
    // (it is NOT computed at double width then narrowed). Compute in a wider C++ int to avoid actual
    // UB/overflow, then truncate to WACC bits exactly like the RTL's narrow accumulator.
    long long num64 = (long long)omg * (long long)computed + (long long)g * (long long)old;
    int       num   = sext((int)(((uint64_t)num64) & 0xFFFFu), WACC);

    int blend = num >> FRAC;  // arithmetic shift right on the already-WACC-truncated signed value
    if (blend > MAXMAG) blend = MAXMAG;
    else if (blend < -MAXMAG) blend = -MAXMAG;
    m_out[k] = sext(blend, MW);
  }
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  const int NCASES = (argc > 1) ? std::atoi(argv[1]) : 10000;
  std::mt19937 rng(2025);
  // Full MW-bit signed range, INCLUDING the -2^(MW-1) edge (see sext() above) — the module must stay
  // bit-exact even there, not just over the "realistic" saturated [-127,127] operating range.
  std::uniform_int_distribution<int> mdist(-(1 << (MW - 1)), (1 << (MW - 1)) - 1);
  // Vary how many of the DEG slots are "real" per case (a degree-1 variable is a documented corner
  // case), then shuffle which slots those are, so both low- and full-degree variables and scattered
  // gaps show up.
  std::uniform_int_distribution<int> degdist(1, DEG);

  top = new Vvar_update;
  top->clk = 0;
  top->en  = 0;
  top->lam = 0;
  top->gam = 0;
  for (int k = 0; k < DEG; ++k) {
    top->e_in[k]    = 0;
    top->m_in[k]    = 0;
    top->present[k] = 0;
  }
  top->eval();

  int checked = 0, mismatches = 0, cases_ok = 0, ehat_mismatches = 0;

  for (int t = 0; t < NCASES; ++t) {
    std::vector<int> e_in(DEG), m_in(DEG), present(DEG, 0), m_out(DEG);
    int lam = mdist(rng);
    int gam = mdist(rng);

    int deg = degdist(rng);
    std::vector<int> slots(DEG);
    for (int k = 0; k < DEG; ++k) slots[k] = k;
    std::shuffle(slots.begin(), slots.end(), rng);
    for (int i = 0; i < deg; ++i) present[slots[i]] = 1;

    for (int k = 0; k < DEG; ++k) {
      e_in[k] = mdist(rng);
      m_in[k] = mdist(rng);
    }

    bool ehat_ref;
    reference(lam, gam, e_in, m_in, present, m_out, ehat_ref);

    // Drive one `en` pulse, then wait 2 clocks (per the module's documented 2-cycle latency).
    top->lam = (uint8_t)(lam & 0xFF);
    top->gam = (uint8_t)(gam & 0xFF);
    for (int k = 0; k < DEG; ++k) {
      top->e_in[k]    = (uint8_t)(e_in[k] & 0xFF);
      top->m_in[k]    = (uint8_t)(m_in[k] & 0xFF);
      top->present[k] = present[k] ? 1 : 0;
    }
    top->en = 1;
    tick();  // stage 1 captures lam/gam/e_in/m_in/present -> total/ehat_bit_s1 (+ registered snapshot)
    top->en = 0;
    tick();  // stage 2 emits m_out + ehat_bit from the stage-1 registers

    bool case_ok = true;
    bool got_ehat = top->ehat_bit ? true : false;
    if (got_ehat != ehat_ref) {
      case_ok = false;
      ++ehat_mismatches;
      if (mismatches < 10)
        std::fprintf(stderr, "  case %d: ehat_bit RTL %d != reference %d\n", t, got_ehat, ehat_ref);
    }
    for (int k = 0; k < DEG; ++k) {
      int got = sext((int)top->m_out[k], MW);
      ++checked;
      if (got != m_out[k]) {
        case_ok = false;
        if (mismatches < 10)
          std::fprintf(stderr, "  case %d slot %d: RTL %d != reference %d (present=%d)\n", t, k, got,
                       m_out[k], present[k]);
        ++mismatches;
      }
    }
    if (case_ok) ++cases_ok;
  }

  top->final();
  delete top;

  if (mismatches == 0 && ehat_mismatches == 0) {
    std::printf("PASS: %d/%d var_update outputs bit-identical to bp_relay_fast reference\n", cases_ok,
                NCASES);
    return 0;
  }
  std::printf("FAIL: %d/%d var_update test cases had a mismatch (%d/%d m_out mismatches, %d ehat_bit "
              "mismatches)\n",
              NCASES - cases_ok, NCASES, mismatches, checked, ehat_mismatches);
  return 1;
}
