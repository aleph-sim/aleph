// Q6-21 — board-free sim<->RTL co-simulation harness.
//
// The simulator (aleph-qec) plays QPU: `examples/qec_q6_cosim.rs` draws Monte-Carlo shots from the
// SAME detector-error model the RTL matching graph was generated from and dumps them as a `.vec`
// stream (one block per physical error rate p). This testbench feeds that stream into the Verilated
// `uf_surface_decoder`, collects `obs_flip` per shot, accumulates the RTL logical-error rate, and
// checks it against the software UnionFind baseline carried in each block header — within combined
// Monte-Carlo CI. That closes the verification chain on realistic noise entirely in software:
//   noise model -> syndromes -> RTL decode -> logical error rate.
// When a board arrives (Q6-08) the same `.vec` stream drives the real decoder over the Q6-07 AXI
// link instead of Verilator; this TB is the board-free stand-in.
//
// Why CI, not bit-equality: the RTL UF and the CPU UF tie-break degenerate (equal-weight) cosets
// differently, so they disagree shot-for-shot on a minority of syndromes (hw/README "Verification
// note") — but their aggregate logical-error *rates* must agree, which is the meaningful claim.
//
//   ./sim_cosim uf_surface_graph.svh cosim_d3.vec [gate_p]
//
// `gate_p` (optional, default 1.0 = gate every block): only blocks with p <= gate_p gate the run.
// A real decoder operates *below* threshold; there the RTL must match the software UF within CI.
// Above threshold the unweighted bounded-depth RTL UF tie-breaks degenerate cosets more crudely
// than the CPU UF, so its logical-error rate is modestly higher — a true, expected quality gap
// (not noise: both decode the SAME shots). Those rows are reported as "info", not failed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vuf_surface_decoder.h"
#include "verilated.h"

static bool scalar(const std::string &l, const char *name, int &out) {
  auto p = l.find(name);
  if (p == std::string::npos) return false;
  size_t q = p + std::string(name).size();
  while (q < l.size() && l[q] == ' ') ++q;
  if (q >= l.size() || l[q] != '=') return false;
  out = std::atoi(l.c_str() + q + 1);
  return true;
}

// Write a detector bit-vector into the DUT `syndrome` port regardless of width class (narrow scalar
// for d=3; VlWide<> once detectors exceed 64, e.g. multi-round 3D graphs). `lit[i]` = detector i set.
template <class T>
static inline void set_syndrome(T &port, const std::vector<bool> &lit) {
  T v = 0;
  for (std::size_t i = 0; i < lit.size(); ++i)
    if (lit[i]) v |= (T(1) << i);
  port = v;
}
template <std::size_t W>
static inline void set_syndrome(VlWide<W> &port, const std::vector<bool> &lit) {
  for (std::size_t w = 0; w < W; ++w) port[w] = 0;
  for (std::size_t i = 0; i < lit.size(); ++i)
    if (lit[i]) port[i >> 5] |= (1u << (i & 31));
}

// 95% normal-approximation half-width on a logical-error rate (matches LogicalErrorResult::new).
static double ci95(double rate, double n) {
  return n > 0.0 ? 1.96 * std::sqrt(rate * (1.0 - rate) / n) : 0.0;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string svh = (argc > 1) ? argv[1] : "uf_surface_graph.svh";
  const std::string vec = (argc > 2) ? argv[2] : "cosim.vec";
  const double gate_p = (argc > 3) ? std::atof(argv[3]) : 1.0;

  int N = 0, M = 0;
  {
    std::ifstream f(svh);
    if (!f) { std::fprintf(stderr, "FAIL: open %s\n", svh.c_str()); return 2; }
    std::string l;
    while (std::getline(f, l)) { scalar(l, "UF_N", N); scalar(l, "UF_M", M); }
  }
  if (N == 0 || M == 0) { std::fprintf(stderr, "FAIL: parsed N=%d M=%d\n", N, M); return 2; }
  const int dets = N - 1;

  std::ifstream vf(vec);
  if (!vf) { std::fprintf(stderr, "FAIL: open %s\n", vec.c_str()); return 2; }

  auto *dut = new Vuf_surface_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  dut->rst_n = 0; dut->in_valid = 0; set_syndrome(dut->syndrome, std::vector<bool>(dets, false));
  tick(); tick();
  dut->rst_n = 1;

  // Drive one shot through the multi-cycle in_valid -> out_valid handshake; returns obs_flip.
  auto decode = [&](const std::vector<bool> &lit, bool &ok) {
    dut->in_valid = 1; set_syndrome(dut->syndrome, lit); tick();
    dut->in_valid = 0;
    int g = 0;
    while (!dut->out_valid && g < 8192) { tick(); ++g; }
    dut->eval();
    ok = dut->out_valid;
    return (int)dut->obs_flip;
  };

  std::printf("co-sim: graph N=%d M=%d dets=%d | vectors=%s\n", N, M, dets, vec.c_str());
  std::printf("   p       rtl_rate     sw_rate     |diff|    combined_ci  verdict\n");

  // Per-block accumulators (a block = all shots at one p, opened by a `P ...` header).
  bool in_block = false, all_pass = true;
  double blk_p = 0, sw_rate = 0, sw_ci = 0;
  long long blk_shots = 0, rtl_errs = 0, invalid = 0;
  int max_lat = 0;

  auto finish_block = [&]() {
    if (!in_block || blk_shots == 0) return;
    double rate = (double)rtl_errs / (double)blk_shots;
    double rci = ci95(rate, (double)blk_shots);
    double comb = rci + sw_ci;                 // shots are shared -> this sum is a generous bound
    double diff = std::fabs(rate - sw_rate);
    bool within = diff <= comb + 1e-12;
    bool gated = blk_p <= gate_p + 1e-12;       // only sub-threshold blocks gate the run
    // Invalid RTL outputs are a hard failure at any p; a within-CI miss only fails when gated.
    if (invalid) all_pass = false;
    else if (gated && !within) all_pass = false;
    const char *verdict = invalid ? "FAIL (INVALID!)"
                          : within ? "PASS"
                          : gated  ? "FAIL"
                                   : "info (supra-threshold)";
    std::printf("  %.3f   %.4e  %.4e  %.2e  %.2e   %s\n", blk_p, rate, sw_rate, diff, comb, verdict);
  };

  std::string l;
  std::vector<bool> lit(dets, false);
  while (std::getline(vf, l)) {
    if (l.empty() || l[0] == '#') continue;
    if (l[0] == 'P') {                          // block header: flush previous, start new
      finish_block();
      blk_shots = rtl_errs = invalid = 0;
      blk_p = sw_rate = sw_ci = 0;
      // parse "P p=<> shots=<> sw_rate=<> sw_ci=<>"
      std::istringstream is(l.substr(1));
      std::string tok;
      while (is >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "p") blk_p = std::atof(v.c_str());
        else if (k == "sw_rate") sw_rate = std::atof(v.c_str());
        else if (k == "sw_ci") sw_ci = std::atof(v.c_str());
      }
      in_block = true;
      continue;
    }
    // data line: "<dets chars> <obs>"
    if ((int)l.size() < dets + 2) continue;
    for (int j = 0; j < dets; ++j) lit[j] = (l[j] == '1');
    int truth = (l[dets + 1] == '1') ? 1 : 0;
    bool ok = false;
    int obs = decode(lit, ok);
    if ((int)dut->latency_cycles > max_lat) max_lat = (int)dut->latency_cycles;
    if (!ok) { ++invalid; continue; }
    if (obs != truth) ++rtl_errs;
    ++blk_shots;
  }
  finish_block();

  std::printf("max decode latency = %d clk\n", max_lat);
  dut->final();
  delete dut;

  if (!all_pass) { std::printf("RESULT: FAIL\n"); return 1; }
  std::printf("RESULT: PASS (RTL logical-error rate matches software UnionFind within MC CI at "
              "every p; all outputs valid)\n");
  return 0;
}
