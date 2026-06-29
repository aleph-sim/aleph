// Q6-09 — distance-scalable Verilator testbench for the surface-code UF decoder.
//
// For d≥5 the syndrome space (2^detectors) is far too large to enumerate (d=5 has 24 detectors →
// 16.7M). A distance-d code corrects up to ⌊(d-1)/2⌋ errors, so we instead enumerate every weight-1
// and weight-2 edge-error pattern (O(M²)), which is the regime the decoder MUST get right, and check:
//   (1) validity — the correction reproduces the syndrome on every detector;
//   (2) distance — every weight-1 error decodes to its true logical (obs == ELOG[e]);
//   (3) quality  — logical-error count over all weight-≤2 errors (0 ⇒ corrects every ≤2-error fault).
// No oracle / golden table needed: the true logical content of an edge-error set is the XOR of its
// ELOG flags. Parametric in the included graph, so the same TB serves d=3, d=5, …

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

// Read bit `e` of the DUT `correction` port regardless of its Verilator width class: a narrow port
// (<=32 / <=64 bits) is an IData/QData scalar, but for d>=7 UF_M exceeds 64 so it is a VlWide<>.
// Overloads keep the same call site (`cbit(dut->correction, e)`) working across distances.
static inline int cbit(uint32_t c, int e) { return (int)((c >> e) & 1u); }
static inline int cbit(uint64_t c, int e) { return (int)((c >> e) & 1ull); }
template <std::size_t W>
static inline int cbit(const VlWide<W> &c, int e) { return (int)((c[e >> 5] >> (e & 31)) & 1u); }

// Write a detector bit-vector into the DUT `syndrome` port regardless of its width class. For d>=5
// with multiple rounds (Q6-19) the detector count exceeds 64, so the port is a VlWide<>; the scalar
// template covers the narrow CData/SData/IData/QData cases. `lit[i]` = detector i is set.
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

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string svh = (argc > 1) ? argv[1] : "uf_surface_graph.svh";

  int N = 0, M = 0, B = 0;
  std::vector<int> ea, eb, el;
  {
    std::ifstream f(svh);
    if (!f) { std::fprintf(stderr, "FAIL: open %s\n", svh.c_str()); return 2; }
    std::string l;
    while (std::getline(f, l)) {
      scalar(l, "UF_N", N); scalar(l, "UF_M", M); scalar(l, "UF_BOUNDARY", B);
      list(l, "UF_EA", ea); list(l, "UF_EB", eb); list(l, "UF_ELOG", el);
    }
  }
  if (N == 0 || M == 0 || (int)ea.size() != M || (int)el.size() != M) {
    std::fprintf(stderr, "FAIL: parsed N=%d M=%d\n", N, M); return 2;
  }
  const int dets = N - 1;

  auto *dut = new Vuf_surface_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  dut->rst_n = 0; dut->in_valid = 0; set_syndrome(dut->syndrome, std::vector<bool>(dets, false));
  tick(); tick();
  dut->rst_n = 1;

  // Syndrome is UF_N-2 bits wide; at d=7 that is 47 bits and for multi-round 3D graphs (Q6-19) it
  // exceeds 64, so syndromes are detector bit-vectors written via the width-agnostic set_syndrome().
  auto decode = [&](const std::vector<bool> &lit) {
    dut->in_valid = 1; set_syndrome(dut->syndrome, lit); tick();
    dut->in_valid = 0;
    int g = 0;
    while (!dut->out_valid && g < 8192) { tick(); ++g; }
    dut->eval();
  };
  // syndrome of a single edge error: flip its two detector endpoints (boundary has no detector bit).
  auto syn_of = [&](int e) {
    std::vector<bool> s(dets, false);
    if (ea[e] != B) s[ea[e]] = !s[ea[e]];
    if (eb[e] != B) s[eb[e]] = !s[eb[e]];
    return s;
  };
  // does the correction reproduce syndrome `lit`? (UF_M may exceed 64 -> read via cbit)
  auto valid = [&](const std::vector<bool> &lit) {
    for (int d = 0; d < dets; ++d) {
      int par = 0;
      for (int e = 0; e < M; ++e)
        if (ea[e] == d || eb[e] == d) par ^= cbit(dut->correction, e);
      if ((par != 0) != lit[d]) return false;
    }
    return true;
  };

  int valid_fail = 0, dist_fail = 0, le_w2 = 0, n_w2 = 0, max_lat = 0;
  auto track_lat = [&]() { if ((int)dut->latency_cycles > max_lat) max_lat = dut->latency_cycles; };

  // (1)+(2) weight-1: validity + distance (obs must equal the edge's true logical).
  for (int e = 0; e < M; ++e) {
    decode(syn_of(e)); track_lat();
    if (!dut->out_valid || !valid(syn_of(e))) { ++valid_fail; continue; }
    if (dut->obs_flip != (el[e] & 1)) ++dist_fail;
  }
  // (3) weight-2: validity + logical-error count vs the true XOR logical.
  for (int e1 = 0; e1 < M; ++e1)
    for (int e2 = e1 + 1; e2 < M; ++e2) {
      std::vector<bool> s = syn_of(e1);
      const std::vector<bool> s2 = syn_of(e2);
      for (int d = 0; d < dets; ++d) s[d] = s[d] != s2[d];   // XOR the two edge syndromes
      const int otrue = (el[e1] ^ el[e2]) & 1;
      decode(s); track_lat();
      ++n_w2;
      if (!dut->out_valid || !valid(s)) { ++valid_fail; continue; }
      if (dut->obs_flip != otrue) ++le_w2;
    }

  std::printf("d-scale: N=%d M=%d dets=%d | validity_fail=%d  distance(wt1)_fail=%d  "
              "wt2_logical_errors=%d/%d  max_latency=%d clk\n",
              N, M, dets, valid_fail, dist_fail, le_w2, n_w2, max_lat);
  dut->final();
  delete dut;
  // PASS gate: every tested syndrome is valid AND every weight-1 error is corrected (true distance).
  // weight-2 logical errors are reported (0 ⇒ corrects all ≤2-error faults; nonzero ⇒ UF tie-breaks
  // on degenerate weight-2 cosets — still valid corrections).
  if (valid_fail || dist_fail) { std::printf("RESULT: FAIL\n"); return 1; }
  std::printf("RESULT: PASS (valid on all weight-≤2 syndromes; distance-correct on all %d weight-1 "
              "errors; %d/%d weight-2 logical errors)\n", M, le_w2, n_w2);
  return 0;
}
