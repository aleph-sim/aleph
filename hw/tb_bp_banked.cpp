// Q7-02 M7 — Verilator testbench for the K-BANKED relay-BP core (`bp_relay_banked`).
//
// Drives the SAME circuit-level golden vectors (`bp_circ_vectors.txt`, 40 shots) that the `bp_relay_fast`
// reference and the modular partial-unroll core (`bp_relay_unroll_pipe`) pass, through the beta-split
// check-major LUTRAM-bank core, and asserts corr_out[BP_N] / obs_flip / valid_flag bit-for-bit vs the
// golden. `early_exit` is driven 0 for these golden runs (full best-kept decode). Because the (W,V) bank
// geometry is a pure scheduling/placement change (it only changes cycle count, not the decode), the
// `bpbanked` Makefile target builds+runs this at THREE (W,V) configs (8/24, 12/36, 16/48) and requires
// 40/40 at each — the (W,V)-invariance check that catches any bank gather/scatter/schedule bug. Prints
// the per-shot latency (cycle) distribution + worst/mean.
//
// Ported from tb_bp_unroll_pipe.cpp; deltas: DUT is bp_relay_banked, the `early_exit` port is driven 0,
// and `latency_cycles` is 32-bit (uint32_t). Golden format + compare logic are unchanged.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

// Q7-02 B0 (Option B): the same circuit-level golden also drives the M4 spatially-unrolled core.
// Both cores present the same syndrome-in / corr_out+obs_flip+valid_flag contract, so the compare
// logic below is shared verbatim rather than forked into a second testbench -- the point of the
// exercise is that the two cores are bit-identical on the same vectors, and a duplicated harness
// would weaken exactly that claim.
#ifdef UNROLLED_CORE
#include "Vbp_relay_unrolled.h"
using Dut = Vbp_relay_unrolled;
#else
#include "Vbp_relay_banked.h"
using Dut = Vbp_relay_banked;
#endif
#include "verilated.h"
#ifdef BP_TRACE
// Q7-02 Task A4: optional VCD window for gate-level switching-activity power. Compiled only with
// -DBP_TRACE (the `bpgate-asap7` target); the co-simulation harness above is byte-identical without it.
// +trace=<file> +tfrom=<cycle> +tto=<cycle> dump the clock cycles in [tfrom, tto) of the whole run.
#include "verilated_vcd_c.h"
static VerilatedVcdC *tfp = nullptr;
static uint64_t sim_cycle = 0, tfrom = 0, tto = 0;
static VerilatedContext *ctx = nullptr;
#endif

static Dut *top;

static void tick() {
  top->clk = 0;
  top->eval();
#ifdef BP_TRACE
  ctx->timeInc(500);  // 1 ns period: matches the SDC target so OpenSTA activity is per-clock
  if (tfp && sim_cycle >= tfrom && sim_cycle < tto) tfp->dump(ctx->time());
#endif
  top->clk = 1;
  top->eval();
#ifdef BP_TRACE
  ctx->timeInc(500);
  if (tfp && sim_cycle >= tfrom && sim_cycle < tto) tfp->dump(ctx->time());
  ++sim_cycle;
#endif
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vec = (argc > 1) ? argv[1] : "bp_circ_vectors.txt";

  std::ifstream f(vec);
  if (!f) {
    std::fprintf(stderr, "FAIL: open %s\n", vec.c_str());
    return 2;
  }

  int T = 0, N = 0, C = 0, OBS = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream(line) >> T >> N >> C >> OBS;
    break;
  }
  if (T <= 0 || N <= 0 || C <= 0 || OBS <= 0) {
    std::fprintf(stderr, "FAIL: bad header T=%d N=%d C=%d OBS=%d\n", T, N, C, OBS);
    return 2;
  }

  top = new Dut;
#ifdef BP_TRACE
  ctx = top->contextp();
  {
    // commandArgsPlusMatch returns the whole matching argument; take everything after its '='.
    auto plus = [&](const char *key) -> std::string {
      const char *m = ctx->commandArgsPlusMatch(key);
      std::string v = m ? m : "";
      size_t eq = v.find('=');
      return eq == std::string::npos ? std::string() : v.substr(eq + 1);
    };
    std::string tf = plus("trace=");
    if (!tf.empty()) {
      std::string a = plus("tfrom="), b = plus("tto=");
      tfrom = a.empty() ? 0 : std::strtoull(a.c_str(), nullptr, 10);
      tto = b.empty() ? ~0ULL : std::strtoull(b.c_str(), nullptr, 10);
      ctx->traceEverOn(true);
      tfp = new VerilatedVcdC;
      top->trace(tfp, 99);
      tfp->open(tf.c_str());
      std::fprintf(stderr, "trace: %s cycles [%llu, %llu)\n", tf.c_str(), (unsigned long long)tfrom,
                   (unsigned long long)tto);
    }
  }
#endif
  // Reset (synchronous).
  top->rst_n = 0;
  top->in_valid = 0;
#ifndef UNROLLED_CORE
  top->early_exit = 0;  // full best-kept decode for the golden runs
#endif  // the unrolled core has no early_exit port: it always runs the full best-kept schedule
  for (int i = 0; i < 4; ++i) tick();
  top->rst_n = 1;
  tick();

  auto tagged = [&](char tag, std::string &out) {
    while (std::getline(f, line))
      if (!line.empty() && line[0] == tag) {
        size_t p = line.find_first_not_of(" \t", 1);
        out = (p == std::string::npos) ? "" : line.substr(p);
        return true;
      }
    return false;
  };

  int mism = 0;
  uint32_t worst_lat = 0;
  uint64_t lat_sum = 0;
  std::map<uint32_t, int> lat_hist;
  for (int t = 0; t < T; ++t) {
    std::string s_str, h_str, o_str, v_str;
    if (!tagged('s', s_str) || !tagged('h', h_str) || !tagged('o', o_str) || !tagged('v', v_str)) {
      std::fprintf(stderr, "FAIL: truncated vectors at test %d\n", t);
      return 2;
    }

    // Drive the syndrome and pulse in_valid for one cycle.
#ifdef BP_GATE_PORTS
    // Gate netlists (yosys/OpenROAD write_verilog) expose the unpacked-array ports as packed vectors,
    // which Verilator hands over as 32-bit words: address bit c as word c>>5, bit c&31.
    for (int w = 0; w < (C + 31) / 32; ++w) top->syndrome_in[w] = 0;
    for (int c = 0; c < C; ++c)
      if (c < (int)s_str.size() && s_str[c] == '1') top->syndrome_in[c >> 5] |= 1u << (c & 31);
#else
    for (int c = 0; c < C; ++c) top->syndrome_in[c] = (c < (int)s_str.size() && s_str[c] == '1');
#endif
    top->in_valid = 1;
    tick();
    top->in_valid = 0;

    int guard = 0;
    while (!top->out_valid && guard < 8000000) {
      tick();
      ++guard;
    }
    if (!top->out_valid) {
      std::fprintf(stderr, "FAIL: test %d never asserted out_valid\n", t);
      return 2;
    }

    // Compare corr_out[N], obs_flip[OBS], valid_flag against the golden.
    int local = 0;
    for (int v = 0; v < N; ++v) {
      int want = (v < (int)h_str.size() && h_str[v] == '1') ? 1 : 0;
#ifdef BP_GATE_PORTS
      int got = (top->corr_out[v >> 5] >> (v & 31)) & 1;
#else
      int got = (int)top->corr_out[v];
#endif
      if (got != want) {
        if (mism < 2 && local < 16) std::fprintf(stderr, "    corr_out[%d]: got %d want %d\n", v, got, want);
        ++local;
      }
    }
    uint32_t obs = top->obs_flip;
    for (int o = 0; o < OBS; ++o) {
      int want = (o < (int)o_str.size() && o_str[o] == '1') ? 1 : 0;
      if ((int)((obs >> o) & 1) != want) {
        if (mism < 2) std::fprintf(stderr, "    obs_flip[%d]: got %d want %d\n", o, (int)((obs >> o) & 1), want);
        ++local;
      }
    }
    int vwant = (!v_str.empty() && v_str[0] == '1') ? 1 : 0;
    if ((int)top->valid_flag != vwant) {
      if (mism < 2) std::fprintf(stderr, "    valid_flag: got %d want %d\n", (int)top->valid_flag, vwant);
      ++local;
    }

    uint32_t lat = top->latency_cycles;
    if (lat > worst_lat) worst_lat = lat;
    lat_sum += lat;
    ++lat_hist[lat];
    if (local) {
      if (mism < 8) std::fprintf(stderr, "  test %d: %d field mismatches\n", t, local);
      ++mism;
    }
  }

#ifdef BP_TRACE
  // +idle=<cycles>: keep clocking with in_valid=0 after the last decode, so a second trace window can
  // capture the idle regime (the decoder parked in its wait state) for the power comparison.
  {
    const char *ia = ctx->commandArgsPlusMatch("idle=");
    std::string iv = ia ? ia : "";
    size_t eq = iv.find('=');
    uint64_t idle = eq == std::string::npos ? 0 : std::strtoull(iv.c_str() + eq + 1, nullptr, 10);
    std::fprintf(stderr, "idle window: %llu cycles from cycle %llu\n", (unsigned long long)idle,
                 (unsigned long long)sim_cycle);
    for (uint64_t k = 0; k < idle; ++k) tick();
  }
#endif
  top->final();
#ifdef BP_TRACE
  if (tfp) { tfp->close(); delete tfp; }
  std::printf("total cycles simulated: %llu\n", (unsigned long long)sim_cycle);
#endif
  delete top;

  std::printf("latency distribution (cycles -> shots):");
  for (auto &kv : lat_hist) std::printf(" %u:%d", kv.first, kv.second);
  std::printf("\n");
  std::printf("latency: worst = %u cycles, mean = %.1f cycles\n",
              worst_lat, (double)lat_sum / (double)T);

  if (mism == 0) {
    std::printf("PASS: %d full decodes bit-identical to the fixed-point golden; worst latency = %u cycles\n",
                T, worst_lat);
    return 0;
  }
  std::printf("FAIL: %d/%d decodes mismatched\n", mism, T);
  return 1;
}
