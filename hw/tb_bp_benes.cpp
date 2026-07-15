// Q7-04 M9c Step 2.3b — standalone Verilator testbench for the three Beneš permutation-network
// fabrics (`hw/bp_benes.sv`): bp_benes_ecm_read, bp_benes_ecm_addr, bp_benes_mcm_wr.
//
// This TB (a) reimplements `benes_apply`/`apply_block` from crates/aleph-qec/src/benes.rs
// directly in C++ (`apply_block()` below — same (col0,row0,stride) recursion, same bar/cross
// convention: ctrl bit false=straight/true=cross), (b) STREAMS >=10000 random `ctrl`+`din` cases
// per fabric -- a FRESH, independent case applied EVERY cycle, so many cases are in flight in the
// pipeline simultaneously (initiation interval = 1) -- and (c) asserts `dout` equals the C++
// reference EXACTLY `PIPE` clocks after the corresponding `din`/`ctrl` was applied, via a FIFO of
// pending expected results (push one per cycle when applied, pop/check exactly PIPE cycles
// later). This is the NEW contract as of Step 2.3b: `ctrl` is now pipelined internally in
// lockstep with `din` (see the file banner in bp_benes.sv), so a later case's `ctrl` can no
// longer corrupt an earlier case's still-in-flight data -- the exact overlapping-in-flight
// scenario the old (pre-2.3b) TB deliberately avoided by holding `ctrl`/`din` stable for PIPE
// cycles per case. That old hold-stable contract is gone; this TB proves the new one instead. A
// dedicated zero-flush + single-shot precision check (`check_latency_precision()`) additionally
// proves the FIRST correct output appears at exactly cycle PIPE (or, for PIPE=0, in the SAME
// cycle, combinationally) and not one cycle earlier, ruling out a coincidental match.
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
#include <deque>
#include <random>
#include <vector>

// BENES_PIPE overrides the fabric's default PIPE (which must match the Verilator `-GPIPE=`
// passed for the build). Used to exercise PIPE=0 (fully combinational) one-off builds without
// touching the Makefile's production PIPE=3/3/4 defaults, e.g.:
//   verilator ... -GPIPE=0 -CFLAGS -DBENES_READ -CFLAGS -DBENES_PIPE=0 bp_benes.sv tb_bp_benes.cpp
#if defined(BENES_READ)
#include "Vbp_benes_ecm_read.h"
using VTop                    = Vbp_benes_ecm_read;
static constexpr int N        = 512;
static constexpr int W        = 6;
#ifndef BENES_PIPE
#define BENES_PIPE 3
#endif
static constexpr const char *NAME = "bp_benes_ecm_read";
#elif defined(BENES_ADDR)
#include "Vbp_benes_ecm_addr.h"
using VTop                    = Vbp_benes_ecm_addr;
static constexpr int N        = 512;
static constexpr int W        = 5;
#ifndef BENES_PIPE
#define BENES_PIPE 3
#endif
static constexpr const char *NAME = "bp_benes_ecm_addr";
#elif defined(BENES_WR)
#include "Vbp_benes_mcm_wr.h"
using VTop                    = Vbp_benes_mcm_wr;
static constexpr int N        = 1024;
static constexpr int W        = 11;
#ifndef BENES_PIPE
#define BENES_PIPE 4
#endif
static constexpr const char *NAME = "bp_benes_mcm_wr";
#else
#error "define exactly one of BENES_READ / BENES_ADDR / BENES_WR"
#endif
static constexpr int PIPE = BENES_PIPE;

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
  if (PIPE == 0) {
    // Fully combinational: dout must equal the reference in the SAME cycle with no clock edge
    // at all. The edge-counting loop below only makes sense for PIPE>=1 (it would degenerate to
    // a no-op loop `for edges=1..0`), so check PIPE=0 explicitly here instead.
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
    std::vector<uint32_t> expected  = benes_apply(ctrl_bits, din_vals);
    std::vector<uint32_t> got       = read_dout();
    if (got != expected) {
      std::fprintf(stderr,
                    "  latency check (PIPE=0): dout did not match the reference combinationally\n");
      return false;
    }
    return true;
  }

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

  // Volume run (Step 2.3b streaming contract): `ctrl` now travels WITH its own `din` through the
  // pipeline (see bp_benes.sv file banner), so a FRESH, independent `(din,ctrl)` case may be
  // applied EVERY cycle -- many cases are in flight simultaneously -- and each one's `dout` must
  // appear EXACTLY `PIPE` cycles after it was applied, unperturbed by the cases applied before or
  // after it. This is the exact overlapping-in-flight scenario the OLD (pre-2.3b) TB deliberately
  // avoided (it held `ctrl`/`din` stable for PIPE cycles per case); this TB now proves the new,
  // fully-pipelined contract instead.
  //
  // Implementation: a FIFO of pending expected results. At loop iteration `t` (t=0..NCASES-1) a
  // fresh case is applied (combinationally visible via top->eval()) and its expected result is
  // pushed with `ready_at = t + PIPE - 1` (see derivation below), then the clock is advanced by
  // one tick. Because exactly one tick happens per iteration -- including the iteration that
  // pushed the case -- the case pushed at iteration `t` has received `(current_t - t + 1)` ticks
  // by the end of iteration `current_t`; setting that equal to PIPE gives
  // `current_t = t + PIPE - 1` for PIPE>=1. `total_iters = NCASES + PIPE - 1` gives every pushed
  // case's `ready_at` an iteration to be checked in (the last case, pushed at NCASES-1, has
  // `ready_at = NCASES + PIPE - 2`, the final loop index). PIPE==0 has no clock delay at all, so
  // it is handled as a fully separate (simpler) same-cycle check.
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
    // Fully combinational: no queue needed -- each case's result is visible on dout the SAME
    // cycle it is applied, before any clock edge.
    for (int t = 0; t < NCASES; ++t) {
      randomize_masked(top->ctrl, rng, CTRL_BITS);
      randomize_masked(top->din, rng, DIN_BITS);
      top->eval();

      std::vector<uint8_t>  ctrl_bits = read_ctrl();
      std::vector<uint32_t> din_vals  = read_din();
      std::vector<uint32_t> expected  = benes_apply(ctrl_bits, din_vals);
      std::vector<uint32_t> got       = read_dout();
      ++cases_checked;
      if (got != expected) report_mismatch(t, got, expected);

      tick();  // exercise clk every case anyway, mirroring normal streaming usage.
    }
  } else {
    struct PendingCase {
      int                    ready_at;
      std::vector<uint32_t>  expected;
    };
    std::deque<PendingCase> pending;

    const int total_iters = NCASES + PIPE - 1;
    for (int t = 0; t < total_iters; ++t) {
      if (t < NCASES) {
        randomize_masked(top->ctrl, rng, CTRL_BITS);
        randomize_masked(top->din, rng, DIN_BITS);
        top->eval();

        std::vector<uint8_t>  ctrl_bits = read_ctrl();
        std::vector<uint32_t> din_vals  = read_din();
        std::vector<uint32_t> expected  = benes_apply(ctrl_bits, din_vals);
        pending.push_back({t + PIPE - 1, std::move(expected)});
      }
      // t >= NCASES: no new case applied (queue is draining); ctrl/din simply hold their last
      // driven values, which is harmless since no NEW expected result depends on them.

      tick();

      if (!pending.empty() && pending.front().ready_at == t) {
        std::vector<uint32_t> got = read_dout();
        ++cases_checked;
        if (got != pending.front().expected) {
          report_mismatch(t - PIPE + 1, got, pending.front().expected);
        }
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
        "PASS: %s %d/%d STREAMED (overlapping in-flight) cases bit-exact vs C++ benes_apply "
        "reference, latency==PIPE=%d\n",
        NAME, cases_checked, NCASES, PIPE);
    return 0;
  }
  std::printf("FAIL: %s %d/%d cases checked, %d mismatches\n", NAME, cases_checked, NCASES, mismatches);
  return 1;
}
