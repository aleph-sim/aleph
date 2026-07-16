// Q7-04 M9c Step 5c — standalone Verilator testbench for the two AS-Waksman permutation-network
// fabrics (`hw/bp_asw.sv`): bp_asw_ecm_read (N=400) and bp_asw_mcm_wr (N=800).
//
// This TB (a) INDEPENDENTLY reimplements `aswaksman_apply`/`apply_block` from
// crates/aleph-qec/src/aswaksman.rs directly in C++ (`apply_block()` below -- the SAME recursive
// block decomposition with a running control offset, the SAME bar/cross convention: ctrl bit
// false=straight/true=cross, the SAME odd-n bypass + hardwired-output conventions) -- it is a port
// of the ALGORITHM, NOT a translation of the SV; (b) STREAMS >=10000 random `ctrl`+`din` cases per
// fabric -- a FRESH, independent case applied EVERY cycle, so many cases are in flight in the
// pipeline at once (initiation interval = 1) -- and (c) asserts `dout` equals the C++ reference
// EXACTLY `PIPE` clocks after the corresponding `din`/`ctrl` was applied, via a FIFO of pending
// expected results. `ctrl` is randomized DIRECTLY (as the Beneš TB does): the fabric is a pure
// switch network that interprets ANY ctrl assignment, so random ctrl exercises every switch setting
// -- strictly stronger coverage than only valid-permutation control words, and the C++ apply and
// the SV read the SAME flat running-counter ctrl layout, so they agree bit-for-bit iff the wiring
// is right. A zero-flush single-shot `check_latency_precision()` additionally proves the first
// correct output lands at EXACTLY cycle PIPE (not earlier), ruling out a coincidental streaming match.
//
// Structure mirrors hw/tb_bp_benes.cpp (same wide-port cbit()/VlWide<> idiom, same streaming FIFO,
// same latency-precision check). Selects the fabric via a compile-time define (-CFLAGS -DASW_READ /
// -DASW_WR). The N/W/PIPE constants below MUST match the `-G` values the Makefile's `bpasw` target
// passes to Verilator for that build.

#include <cstdint>
#include <cstdio>
#include <deque>
#include <random>
#include <vector>

#if defined(ASW_READ)
#include "Vbp_asw_ecm_read.h"
using VTop                        = Vbp_asw_ecm_read;
static constexpr int N            = 400;
static constexpr int W            = 6;
#ifndef ASW_PIPE
#define ASW_PIPE 4
#endif
static constexpr const char *NAME = "bp_asw_ecm_read";
#elif defined(ASW_WR)
#include "Vbp_asw_mcm_wr.h"
using VTop                        = Vbp_asw_mcm_wr;
static constexpr int N            = 800;
static constexpr int W            = 11;
#ifndef ASW_PIPE
#define ASW_PIPE 5
#endif
static constexpr const char *NAME = "bp_asw_mcm_wr";
#else
#error "define exactly one of ASW_READ / ASW_WR"
#endif
static constexpr int PIPE = ASW_PIPE;

#include "verilated.h"

// asw_sw_count(n) = ceil(n*log2 n) - n + 1 -- MUST match aswaksman.rs::aswaksman_switch_count and
// bp_asw.sv::asw_sw_count (n=400->3089, n=800->6977). Computed via the same recursion the routing uses.
static int asw_sw_count(int n) {
  if (n <= 1) return 0;
  return (n - 1) + asw_sw_count((n + 1) / 2) + asw_sw_count(n / 2);
}

static const int CTRL_BITS = asw_sw_count(N);
static const int DIN_BITS   = N * W;

// -------------------------------------------------------------------------------------------
// Wide-port helpers (din/ctrl/dout all exceed 64 bits here). Identical to tb_bp_benes.cpp.
// -------------------------------------------------------------------------------------------
template <std::size_t NW>
static inline uint32_t getbits(const VlWide<NW> &v, int lo, int width) {
  uint32_t val = 0;
  for (int i = 0; i < width; ++i) {
    int      bit = lo + i;
    uint32_t b   = (v[bit >> 5] >> (bit & 31)) & 1u;
    val |= b << i;
  }
  return val;
}
template <std::size_t NW>
static inline int getbit(const VlWide<NW> &v, int idx) {
  return (int)((v[idx >> 5] >> (idx & 31)) & 1u);
}
template <std::size_t NW>
static inline void randomize_masked(VlWide<NW> &v, std::mt19937 &rng, int total_bits) {
  for (std::size_t i = 0; i < NW; ++i) v[i] = rng();
  int rem = total_bits % 32;
  if (rem != 0) v[NW - 1] &= (uint32_t)((1u << rem) - 1);
}
template <std::size_t NW>
static inline void zero_wide(VlWide<NW> &v) {
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0;
}

// -------------------------------------------------------------------------------------------
// C++ port of aswaksman.rs::apply_block / aswaksman_apply -- SAME recursive block decomposition
// with a running control offset (`base`, threaded by reference exactly as `route`/`apply_block`
// thread their running offset), SAME bar/cross convention (ctrl false=straight, true=cross), SAME
// odd-n bypass (input n-1 -> upper subnet last position) and hardwired-output conventions. `ctrl`
// is a flat bit-per-byte vector in running-counter order; `input` is one payload value per wire.
// -------------------------------------------------------------------------------------------
static std::vector<uint32_t> apply_block(const std::vector<uint8_t> &ctrl,
                                         const std::vector<uint32_t> &input, int &base) {
  int n = (int)input.size();
  if (n <= 1) return input;  // base unchanged
  int  in_cnt  = n / 2;
  int  out_cnt = (n + 1) / 2 - 1;
  int  m_up    = (n + 1) / 2;
  int  m_lo    = n / 2;
  bool has_byp = (n % 2 == 1);
  int  sw_out  = 2 * out_cnt;

  int                   in_base = base;
  std::vector<uint32_t> upper_in(m_up), lower_in(m_lo);
  for (int i = 0; i < in_cnt; ++i) {
    if (ctrl[in_base + i]) {  // cross
      lower_in[i] = input[2 * i];
      upper_in[i] = input[2 * i + 1];
    } else {  // bar
      upper_in[i] = input[2 * i];
      lower_in[i] = input[2 * i + 1];
    }
  }
  if (has_byp) upper_in[m_up - 1] = input[n - 1];

  base = in_base + in_cnt;                        // after input switches -> upper subnet base
  std::vector<uint32_t> upper_out = apply_block(ctrl, upper_in, base);  // advances base
  std::vector<uint32_t> lower_out = apply_block(ctrl, lower_in, base);  // advances base
  int out_base = base;                            // output switches start here

  std::vector<uint32_t> out(n);
  for (int j = 0; j < out_cnt; ++j) {
    if (ctrl[out_base + j]) {  // cross
      out[2 * j]     = lower_out[j];
      out[2 * j + 1] = upper_out[j];
    } else {  // bar
      out[2 * j]     = upper_out[j];
      out[2 * j + 1] = lower_out[j];
    }
  }
  out[sw_out] = upper_out[m_up - 1];
  if (!has_byp) out[sw_out + 1] = lower_out[m_lo - 1];

  base = out_base + out_cnt;
  return out;
}

static std::vector<uint32_t> asw_apply(const std::vector<uint8_t> &ctrl,
                                       const std::vector<uint32_t> &input) {
  int base = 0;
  return apply_block(ctrl, input, base);
}

static VTop *top;

static void tick() {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

static std::vector<uint32_t> read_din() {
  std::vector<uint32_t> v(N);
  for (int i = 0; i < N; ++i) v[i] = getbits(top->din, i * W, W);
  return v;
}
static std::vector<uint8_t> read_ctrl() {
  std::vector<uint8_t> v(CTRL_BITS);
  for (int i = 0; i < CTRL_BITS; ++i) v[i] = (uint8_t)getbit(top->ctrl, i);
  return v;
}
static std::vector<uint32_t> read_dout() {
  std::vector<uint32_t> v(N);
  for (int i = 0; i < N; ++i) v[i] = getbits(top->dout, i * W, W);
  return v;
}

// Zero-flush the pipeline, apply ONE guaranteed-nonzero case, and assert dout stays flushed-zero for
// cycles 0..PIPE-1 and becomes EXACTLY the reference at cycle PIPE -- a coincidence-proof latency check.
static bool check_latency_precision(std::mt19937 &rng) {
  if (PIPE == 0) {
    zero_wide(top->din);
    zero_wide(top->ctrl);
    top->eval();
    randomize_masked(top->ctrl, rng, CTRL_BITS);
    top->ctrl[0] |= 1u;
    randomize_masked(top->din, rng, DIN_BITS);
    top->din[0] |= 1u;
    top->eval();
    std::vector<uint8_t>  ctrl_bits = read_ctrl();
    std::vector<uint32_t> din_vals  = read_din();
    std::vector<uint32_t> expected  = asw_apply(ctrl_bits, din_vals);
    std::vector<uint32_t> got       = read_dout();
    if (got != expected) {
      std::fprintf(stderr, "  latency check (PIPE=0): dout did not match combinationally\n");
      return false;
    }
    return true;
  }

  zero_wide(top->din);
  zero_wide(top->ctrl);
  for (int i = 0; i < PIPE + 2; ++i) tick();

  randomize_masked(top->ctrl, rng, CTRL_BITS);
  top->ctrl[0] |= 1u;
  randomize_masked(top->din, rng, DIN_BITS);
  top->din[0] |= 1u;

  std::vector<uint8_t>  ctrl_bits = read_ctrl();
  std::vector<uint32_t> din_vals  = read_din();
  std::vector<uint32_t> expected  = asw_apply(ctrl_bits, din_vals);

  bool all_zero_expected = true;
  for (auto v : expected)
    if (v != 0) { all_zero_expected = false; break; }
  if (all_zero_expected) {
    std::fprintf(stderr, "  latency check: reference case was unexpectedly all-zero\n");
    return false;
  }

  bool ok = true;
  for (int edges = 1; edges <= PIPE; ++edges) {
    tick();
    std::vector<uint32_t> got = read_dout();
    bool                  eq  = (got == expected);
    if (edges < PIPE) {
      if (eq) {
        std::fprintf(stderr, "  latency check: dout matched TOO EARLY at edge %d (PIPE=%d)\n", edges,
                     PIPE);
        ok = false;
      }
      bool all_zero_got = true;
      for (auto v : got)
        if (v != 0) { all_zero_got = false; break; }
      if (!all_zero_got) {
        std::fprintf(stderr,
                     "  latency check: dout NOT flushed-zero at edge %d (PIPE=%d)\n", edges, PIPE);
        ok = false;
      }
    } else {
      if (!eq) {
        std::fprintf(stderr, "  latency check: dout did NOT match at edge PIPE=%d\n", PIPE);
        ok = false;
      }
    }
  }
  return ok;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  const int    NCASES = (argc > 1) ? std::atoi(argv[1]) : 10000;
  std::mt19937 rng(2025);

  top      = new VTop;
  top->clk = 0;
  zero_wide(top->din);
  zero_wide(top->ctrl);
  top->eval();

  std::printf("== %s: N=%d W=%d PIPE=%d SWITCHES=%d ==\n", NAME, N, W, PIPE, CTRL_BITS);

  if (!check_latency_precision(rng)) {
    std::printf("FAIL: %s latency-precision check failed (see above)\n", NAME);
    top->final();
    delete top;
    return 1;
  }
  std::printf("latency-precision check: OK (first correct output at exactly cycle PIPE=%d)\n", PIPE);

  int mismatches = 0, cases_checked = 0;
  auto report_mismatch = [&](int case_idx, const std::vector<uint32_t> &got,
                             const std::vector<uint32_t> &expected) {
    ++mismatches;
    if (mismatches <= 10) {
      std::fprintf(stderr, "  case %d: mismatch (first differing lane): ", case_idx);
      for (int i = 0; i < N; ++i) {
        if (got[i] != expected[i]) {
          std::fprintf(stderr, "lane %d got=%u expected=%u\n", i, got[i], expected[i]);
          break;
        }
      }
    }
  };

  if (PIPE == 0) {
    for (int t = 0; t < NCASES; ++t) {
      randomize_masked(top->ctrl, rng, CTRL_BITS);
      randomize_masked(top->din, rng, DIN_BITS);
      top->eval();
      std::vector<uint8_t>  ctrl_bits = read_ctrl();
      std::vector<uint32_t> din_vals  = read_din();
      std::vector<uint32_t> expected  = asw_apply(ctrl_bits, din_vals);
      std::vector<uint32_t> got       = read_dout();
      ++cases_checked;
      if (got != expected) report_mismatch(t, got, expected);
      tick();
    }
  } else {
    struct PendingCase {
      int                   ready_at;
      std::vector<uint32_t> expected;
    };
    std::deque<PendingCase> pending;
    const int               total_iters = NCASES + PIPE - 1;
    for (int t = 0; t < total_iters; ++t) {
      if (t < NCASES) {
        randomize_masked(top->ctrl, rng, CTRL_BITS);
        randomize_masked(top->din, rng, DIN_BITS);
        top->eval();
        std::vector<uint8_t>  ctrl_bits = read_ctrl();
        std::vector<uint32_t> din_vals  = read_din();
        std::vector<uint32_t> expected  = asw_apply(ctrl_bits, din_vals);
        pending.push_back({t + PIPE - 1, std::move(expected)});
      }
      tick();
      if (!pending.empty() && pending.front().ready_at == t) {
        std::vector<uint32_t> got = read_dout();
        ++cases_checked;
        if (got != pending.front().expected) report_mismatch(t - PIPE + 1, got, pending.front().expected);
        pending.pop_front();
      }
    }
    if (!pending.empty()) {
      std::fprintf(stderr, "  %zu case(s) never drained from the pending queue\n", pending.size());
      mismatches += (int)pending.size();
    }
  }

  top->final();
  delete top;

  if (mismatches == 0 && cases_checked == NCASES) {
    std::printf(
        "PASS: %s %d/%d STREAMED (overlapping in-flight) cases bit-exact vs C++ aswaksman_apply "
        "reference, latency==PIPE=%d\n",
        NAME, cases_checked, NCASES, PIPE);
    return 0;
  }
  std::printf("FAIL: %s %d/%d cases checked, %d mismatches\n", NAME, cases_checked, NCASES, mismatches);
  return 1;
}
