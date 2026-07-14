// Q7-04 M9c Step 2.3 — standalone Verilator testbench for the three Beneš permutation-network
// fabrics (`hw/bp_benes.sv`): bp_benes_ecm_read, bp_benes_ecm_addr, bp_benes_mcm_wr.
//
// This TB (a) reimplements `benes_apply`/`apply_block` from crates/aleph-qec/src/benes.rs
// directly in C++ (`apply_block()` below — same (col0,row0,stride) recursion, same bar/cross
// convention: ctrl bit false=straight/true=cross), (b) drives >=10000 random `ctrl`+`din` cases
// per fabric, (c) asserts `dout` equals the C++ reference EXACTLY `PIPE` clocks after the
// corresponding `din`/`ctrl` was applied. `ctrl` is a ROM-fed configuration input read LIVE
// (combinationally) by every column of the recursion -- it is deliberately NOT itself pipelined
// (only the data path is, per the "PIPE registers on din" spec), so each case's `ctrl`/`din` is
// held stable for the full PIPE cycles that case's data takes to cross every column before the
// next case is applied (an overlapped one-case-per-cycle stream would let a later case's `ctrl`
// corrupt an earlier case's still-in-flight data on the un-registered column spans between
// register boundaries -- this is not a limitation of the TB, it mirrors the real usage: a
// control ROM holds one routing pattern stable while data streams through it). A dedicated
// zero-flush + single-shot precision check (`check_latency_precision()`) additionally proves the
// FIRST correct output appears at exactly cycle PIPE and not one cycle earlier, ruling out a
// coincidental match.
//
// Structure mirrors hw/tb_var_update.cpp (generic module under Verilator, TB reimplements the
// reference, >=10000 random cases) and hw/tb_uf_surface_scale.cpp's `cbit()`/`VlWide<>` idiom for
// reading wide (>64-bit) Verilated ports (din/ctrl/dout here are all wider than 64 bits for every
// fabric size this TB builds).
//
// Selects which of the three fabrics to build via a compile-time define (-CFLAGS -DBENES_READ /
// -DBENES_ADDR / -DBENES_WR), mirroring the `-CFLAGS -DBRAM`-style reuse pattern other multi-
// variant TBs in this directory use (e.g. tb_bp_relay.cpp's -DBRAM/-DFAST/-DPARTIAL). The
// N/W/PIPE constants below MUST match the corresponding `-G` values the Makefile's `bpbenes`
// target passes to Verilator for that build.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#if defined(BENES_READ)
#include "Vbp_benes_ecm_read.h"
using VTop                    = Vbp_benes_ecm_read;
static constexpr int N        = 512;
static constexpr int W        = 6;
static constexpr int PIPE     = 3;
static constexpr const char *NAME = "bp_benes_ecm_read";
#elif defined(BENES_ADDR)
#include "Vbp_benes_ecm_addr.h"
using VTop                    = Vbp_benes_ecm_addr;
static constexpr int N        = 512;
static constexpr int W        = 5;
static constexpr int PIPE     = 3;
static constexpr const char *NAME = "bp_benes_ecm_addr";
#elif defined(BENES_WR)
#include "Vbp_benes_mcm_wr.h"
using VTop                    = Vbp_benes_mcm_wr;
static constexpr int N        = 1024;
static constexpr int W        = 11;
static constexpr int PIPE     = 4;
static constexpr const char *NAME = "bp_benes_mcm_wr";
#else
#error "define exactly one of BENES_READ / BENES_ADDR / BENES_WR"
#endif

#include "verilated.h"

// benes_columns(m) = 2*log2(m) - 1 (m a power of two >= 2). Mirrors benes.rs::benes_columns.
static int benes_columns(int m) {
  int k = 0;
  while ((1 << k) < m) ++k;
  return 2 * k - 1;
}

static const int     COLS_N     = benes_columns(N);
static const int     CTRL_BITS  = COLS_N * (N / 2);
static const int     DIN_BITS   = N * W;

// -------------------------------------------------------------------------------------------
// Wide-port helpers (din/ctrl/dout all exceed 64 bits for every fabric size this TB builds, so
// Verilator represents them as VlWide<NW> -- an array of 32-bit words). Mirrors the cbit()
// idiom from tb_uf_surface_scale.cpp, generalised to multi-bit fields (din/dout elements) and to
// randomizing + masking a whole wide port at word granularity (fast: O(words), not O(bits)).
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

// VlWide has no begin()/end() (see verilated_types.h) so a range-for won't compile; zero it word
// by word via .size()/operator[] instead.
template <std::size_t NW>
static inline void zero_wide(VlWide<NW> &v) {
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0;
}

// -------------------------------------------------------------------------------------------
// C++ port of benes.rs::apply_block / benes_apply -- SAME (col0,row0,stride) recursive block
// decomposition, SAME bar/cross convention (ctrl bit false=straight, true=cross). `ctrl` is a
// flat bit-per-byte vector (index = column-major control index, matching benes_control's
// `ctrl[col*(m/2)+switch]` layout); `input` is one payload value per input wire.
// -------------------------------------------------------------------------------------------
static std::vector<uint32_t> apply_block(const std::vector<uint8_t> &ctrl,
                                          const std::vector<uint32_t> &input, int col0, int row0,
                                          int stride) {
  int n = (int)input.size();
  if (n == 2) {
    if (ctrl[col0 * stride + row0]) return {input[1], input[0]};
    return {input[0], input[1]};
  }
  int                   half = n / 2;
  std::vector<uint32_t> upper_in(half), lower_in(half);
  for (int isw = 0; isw < half; ++isw) {
    if (ctrl[col0 * stride + row0 + isw]) {
      upper_in[isw] = input[2 * isw + 1];
      lower_in[isw] = input[2 * isw];
    } else {
      upper_in[isw] = input[2 * isw];
      lower_in[isw] = input[2 * isw + 1];
    }
  }
  int out_col = col0 + benes_columns(n) - 1;
  auto upper_out = apply_block(ctrl, upper_in, col0 + 1, row0, stride);
  auto lower_out = apply_block(ctrl, lower_in, col0 + 1, row0 + half / 2, stride);

  std::vector<uint32_t> out(n);
  for (int osw = 0; osw < half; ++osw) {
    if (ctrl[out_col * stride + row0 + osw]) {
      out[2 * osw]     = lower_out[osw];
      out[2 * osw + 1] = upper_out[osw];
    } else {
      out[2 * osw]     = upper_out[osw];
      out[2 * osw + 1] = lower_out[osw];
    }
  }
  return out;
}

static std::vector<uint32_t> benes_apply(const std::vector<uint8_t> &ctrl,
                                          const std::vector<uint32_t> &input) {
  return apply_block(ctrl, input, 0, 0, (int)input.size() / 2);
}

static VTop *top;

static void tick() {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

// Extract the current din/ctrl into plain C++ vectors (per-element payloads / per-bit control),
// suitable for feeding straight into `benes_apply` -- reads back EXACTLY what was driven into the
// DUT this cycle, so the reference always matches the actual stimulus bit-for-bit.
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

// Zero-flush the pipeline, then apply ONE guaranteed-nonzero case and step cycle-by-cycle,
// asserting dout stays all-zero (the flushed case's result) for cycles 0..PIPE-1 and becomes
// EXACTLY the nonzero reference at cycle PIPE -- a coincidence-proof latency check (as opposed
// to the streaming loop's statistical one), since a linear PIPE-deep pipeline's output at cycle c
// after a fresh application depends ONLY on whatever was applied c cycles ago.
static bool check_latency_precision(std::mt19937 &rng) {
  // Flush: PIPE+2 cycles of all-zero din/ctrl so every register stage holds a known all-zero
  // case before we apply the distinguishing one.
  zero_wide(top->din);
  zero_wide(top->ctrl);
  for (int i = 0; i < PIPE + 2; ++i) tick();

  // A guaranteed-nonzero case: force ctrl bit 0 = 1 (so at least one switch crosses) and din[0]
  // to a nonzero payload; randomize the rest.
  randomize_masked(top->ctrl, rng, CTRL_BITS);
  top->ctrl[0] |= 1u;
  randomize_masked(top->din, rng, DIN_BITS);
  top->din[0] |= 1u;

  std::vector<uint8_t>  ctrl_bits = read_ctrl();
  std::vector<uint32_t> din_vals  = read_din();
  std::vector<uint32_t> expected  = benes_apply(ctrl_bits, din_vals);

  bool all_zero_expected = true;
  for (auto v : expected)
    if (v != 0) {
      all_zero_expected = false;
      break;
    }
  if (all_zero_expected) {
    std::fprintf(stderr, "  latency check: reference case was unexpectedly all-zero\n");
    return false;
  }

  // `edges` = number of clock ticks since the case above was applied. dout should reflect the
  // still-flushed all-zero pipeline contents for edges=1..PIPE-1 and become EXACTLY `expected`
  // at edges==PIPE (not one edge earlier, not one edge later).
  bool ok = true;
  for (int edges = 1; edges <= PIPE; ++edges) {
    tick();
    std::vector<uint32_t> got = read_dout();
    bool                  eq  = (got == expected);
    if (edges < PIPE) {
      if (eq) {
        std::fprintf(stderr,
                     "  latency check: dout matched expected TOO EARLY at edge %d (PIPE=%d)\n", edges,
                     PIPE);
        ok = false;
      }
      // Not just "not yet equal to expected" -- assert it's the TRUE flushed value (all-zero,
      // per the zero_wide(din)/zero_wide(ctrl) flush above), so a bug that emits some other
      // wrong-but-different value a cycle early is caught too, not just a lucky early match.
      bool all_zero_got = true;
      for (auto v : got)
        if (v != 0) {
          all_zero_got = false;
          break;
        }
      if (!all_zero_got) {
        std::fprintf(stderr,
                     "  latency check: dout NOT flushed-zero at edge %d (PIPE=%d) -- expected the "
                     "still-flushed all-zero pipeline contents\n",
                     edges, PIPE);
        ok = false;
      }
    } else {  // edges == PIPE
      if (!eq) {
        std::fprintf(stderr, "  latency check: dout did NOT match expected at edge PIPE=%d\n", PIPE);
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

  top = new VTop;
  top->clk = 0;
  zero_wide(top->din);
  zero_wide(top->ctrl);
  top->eval();

  std::printf("== %s: N=%d W=%d PIPE=%d COLS=%d CTRL_BITS=%d ==\n", NAME, N, W, PIPE, COLS_N,
              CTRL_BITS);

  bool latency_ok = check_latency_precision(rng);
  if (!latency_ok) {
    std::printf("FAIL: %s latency-precision check failed (see above)\n", NAME);
    top->final();
    delete top;
    return 1;
  }
  std::printf("latency-precision check: OK (first correct output at exactly cycle PIPE=%d)\n", PIPE);

  // Volume run: `ctrl` is a ROM-fed configuration input, read LIVE (combinationally) by every
  // column in the recursion -- it is deliberately NOT itself pipelined (only the DATA path is,
  // per the PIPE-registers-on-din spec). So each case's `ctrl`/`din` must stay stable for the
  // full PIPE cycles it takes that case's data to cross every column, exactly like the real
  // usage (a control ROM holds one routing pattern stable while data streams through it) --
  // NOT a fully-overlapped one-case-per-cycle stream, which would let a later case's `ctrl`
  // corrupt an earlier case's still-in-flight data on the un-registered column spans between
  // register boundaries. Apply, hold for PIPE ticks, check -- repeated NCASES times.
  int mismatches = 0, cases_checked = 0;

  for (int t = 0; t < NCASES; ++t) {
    randomize_masked(top->ctrl, rng, CTRL_BITS);
    randomize_masked(top->din, rng, DIN_BITS);

    std::vector<uint8_t>  ctrl_bits = read_ctrl();
    std::vector<uint32_t> din_vals  = read_din();
    std::vector<uint32_t> expected  = benes_apply(ctrl_bits, din_vals);

    for (int e = 0; e < PIPE; ++e) tick();  // ctrl/din held constant across all PIPE edges

    std::vector<uint32_t> got = read_dout();
    ++cases_checked;
    if (got != expected) {
      ++mismatches;
      if (mismatches <= 10) {
        std::fprintf(stderr, "  case %d: mismatch (first differing lane): ", t);
        for (int i = 0; i < N; ++i) {
          if (got[i] != expected[i]) {
            std::fprintf(stderr, "lane %d got=%u expected=%u\n", i, got[i], expected[i]);
            break;
          }
        }
      }
    }
  }

  top->final();
  delete top;

  if (mismatches == 0 && cases_checked == NCASES) {
    std::printf("PASS: %s %d/%d cases bit-exact vs C++ benes_apply reference, latency==PIPE=%d\n", NAME,
                cases_checked, NCASES, PIPE);
    return 0;
  }
  std::printf("FAIL: %s %d/%d cases checked, %d mismatches\n", NAME, cases_checked, NCASES, mismatches);
  return 1;
}
