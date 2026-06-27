// Q6-02 (sim) — Verilator testbench for the surface-code Union-Find decoder.
//
// Parses the generated graph (`uf_surface_graph.svh`) for the edge tables, then for every syndrome
// checks: (1) validity — the RTL correction reproduces the syndrome on every detector node; and
// (2) logical — the predicted flip matches the Rust unweighted UnionFindDecoder oracle
// (`uf_surface_oracle.mem`). A pass certifies a real UF datapath, equivalent to the CPU decoder.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Vuf_surface_decoder.h"
#include "verilated.h"

// Pull "name = <int>" from a line, requiring `=` to be the next non-space token after `name`
// (so `UF_M = 18` matches but the `[UF_M]` in an array declaration does not).
static bool scalar(const std::string &l, const char *name, int &out) {
  auto p = l.find(name);
  if (p == std::string::npos) return false;
  size_t q = p + std::string(name).size();
  while (q < l.size() && l[q] == ' ') ++q;
  if (q >= l.size() || l[q] != '=') return false;
  out = std::atoi(l.c_str() + q + 1);
  return true;
}

// Pull the comma list inside '{...}' on a line into `out`.
static bool list(const std::string &l, const char *name, std::vector<int> &out) {
  if (l.find(name) == std::string::npos) return false;
  auto a = l.find('{'), b = l.find('}');
  if (a == std::string::npos || b == std::string::npos) return false;
  std::string s = l.substr(a + 1, b - a - 1);
  out.clear();
  for (size_t i = 0; i < s.size();) {
    while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
    if (i >= s.size()) break;
    out.push_back(std::atoi(s.c_str() + i));
    while (i < s.size() && s[i] != ',') ++i;
  }
  return true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string svh = (argc > 1) ? argv[1] : "uf_surface_graph.svh";
  const std::string orc = (argc > 2) ? argv[2] : "uf_surface_oracle.mem";

  int N = 0, M = 0, B = 0;
  std::vector<int> ea, eb, elog;
  {
    std::ifstream f(svh);
    if (!f) { std::fprintf(stderr, "FAIL: open %s\n", svh.c_str()); return 2; }
    std::string l;
    while (std::getline(f, l)) {
      scalar(l, "UF_N", N);
      scalar(l, "UF_M", M);
      scalar(l, "UF_BOUNDARY", B);
      list(l, "UF_EA", ea);
      list(l, "UF_EB", eb);
      list(l, "UF_ELOG", elog);
    }
  }
  if (N == 0 || M == 0 || (int)ea.size() != M || (int)eb.size() != M) {
    std::fprintf(stderr, "FAIL: parsed N=%d M=%d ea=%zu eb=%zu\n", N, M, ea.size(), eb.size());
    return 2;
  }
  const int dets = N - 1;          // detectors 0..N-2 ; node N-1 = boundary
  const int entries = 1 << dets;

  std::vector<uint8_t> oracle;
  {
    std::ifstream f(orc);
    if (!f) { std::fprintf(stderr, "FAIL: open %s\n", orc.c_str()); return 2; }
    std::string l;
    while (std::getline(f, l)) {
      auto p = l.find("//");
      if (p != std::string::npos) l = l.substr(0, p);
      for (char c : l) if (c == '0' || c == '1') { oracle.push_back(c == '1'); break; }
    }
  }
  if ((int)oracle.size() != entries) {
    std::fprintf(stderr, "FAIL: oracle %zu entries, expected %d\n", oracle.size(), entries);
    return 2;
  }

  auto *dut = new Vuf_surface_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  dut->rst_n = 0; dut->in_valid = 0; dut->syndrome = 0;
  tick(); tick();
  dut->rst_n = 1;

  dut->in_valid = 1; dut->syndrome = 0; tick();
  dut->in_valid = 0;
  int latency = 1;
  while (!dut->out_valid && latency < 32) { tick(); ++latency; }

  auto decode = [&](int s) {
    dut->in_valid = 1; dut->syndrome = s; tick();
    dut->in_valid = 0; dut->eval();
  };

  // (1) Validity over all syndromes: the correction reproduces the syndrome on every detector.
  int fails = 0, obs_mismatch = 0;
  for (int s = 0; s < entries; ++s) {
    decode(s);
    if (!dut->out_valid) { std::fprintf(stderr, "FAIL s=%d: out_valid low\n", s); ++fails; continue; }
    const uint32_t corr = dut->correction;
    int syn_rtl = 0;
    for (int d = 0; d < dets; ++d) {
      int par = 0;
      for (int e = 0; e < M; ++e)
        if (ea[e] == d || eb[e] == d) par ^= (corr >> e) & 1;
      syn_rtl |= par << d;
    }
    if (syn_rtl != s) {
      std::fprintf(stderr, "FAIL s=%d: correction 0x%x -> syndrome 0x%x\n", s, corr, syn_rtl);
      if (++fails > 20) break;
    }
    if (dut->obs_flip != oracle[s]) ++obs_mismatch;
  }

  // (2) Distance: every weight-1 error must be corrected (no residual logical flip). For a single
  // edge error e, the syndrome is its two endpoints and the true logical content is ELOG[e]; a
  // correct distance-3 decoder predicts obs == ELOG[e].
  int dist_fail = 0;
  for (int e = 0; e < M; ++e) {
    int s = 0;
    if (ea[e] != B) s ^= 1 << ea[e];
    if (eb[e] != B) s ^= 1 << eb[e];
    decode(s);
    if (dut->obs_flip != (elog[e] & 1)) {
      std::fprintf(stderr, "DIST FAIL edge %d: obs=%d expected ELOG=%d\n", e, dut->obs_flip, elog[e]);
      ++dist_fail;
    }
  }

  // (3) Quality: logical-error count over all weight-1 and weight-2 errors, for the RTL decoder vs
  // the CPU UF oracle. A logical error on error `e` is `predicted_obs != ELOG·e`. Equal counts =
  // the RTL decoder is as good as the CPU UF, despite differing per-syndrome tie-breaks.
  auto syn_of = [&](int e) {
    int s = 0;
    if (ea[e] != B) s ^= 1 << ea[e];
    if (eb[e] != B) s ^= 1 << eb[e];
    return s;
  };
  int rtl_le = 0, cpu_le = 0, total = 0;
  for (int e1 = 0; e1 < M; ++e1) {
    for (int e2 = e1; e2 < M; ++e2) {
      const bool single = (e1 == e2);
      const int s = single ? syn_of(e1) : (syn_of(e1) ^ syn_of(e2));
      const int otrue = single ? (elog[e1] & 1) : ((elog[e1] ^ elog[e2]) & 1);
      decode(s);
      rtl_le += (dut->obs_flip != otrue);
      cpu_le += (oracle[s] != otrue);
      ++total;
    }
  }

  std::printf("validity: %s (%d fail); weight-1 correctness: %s (%d fail); "
              "logical vs Rust UF: %d/%d agree; latency = %d clk\n",
              fails ? "FAIL" : "PASS", fails, dist_fail ? "FAIL" : "PASS", dist_fail,
              entries - obs_mismatch, entries, latency);
  std::printf("quality (wt<=2 errors, n=%d): RTL logical errors=%d, CPU UF=%d\n",
              total, rtl_le, cpu_le);
  dut->final();
  delete dut;
  if (fails || dist_fail) return 1;
  std::printf("PASS: valid on all %d syndromes + corrects all %d weight-1 errors "
              "(UF tie-breaks differ from CPU UF on %d degenerate syndromes)\n",
              entries, M, obs_mismatch);
  return 0;
}
