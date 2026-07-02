// Q7-02 M2 — Verilator testbench for the sequential fixed-point relay-BP decoder.
//
// Replays the Rust full-decode golden (`bp_dec_vectors.txt` from `FixedRelayBp::decode_fixed_ehat`):
// each test drives a syndrome through the FSM, waits for `out_valid`, and checks the chosen error
// pattern (`corr_out[BP_N]`), the observable flips (`obs_flip[BP_OBS]`), and the validity flag
// bit-for-bit against the golden. A pass certifies the RTL relay-BP decode is the exact silicon twin
// of the M0/M1 golden at Q5.3.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_relay_decoder.h"
#include "verilated.h"

static Vbp_relay_decoder *top;

static void tick() {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vec = (argc > 1) ? argv[1] : "bp_dec_vectors.txt";

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

  top = new Vbp_relay_decoder;
  // Reset.
  top->rst_n = 0;
  top->in_valid = 0;
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

  int mism = 0, worst_lat = 0;
  for (int t = 0; t < T; ++t) {
    std::string s_str, h_str, o_str, v_str;
    if (!tagged('s', s_str) || !tagged('h', h_str) || !tagged('o', o_str) || !tagged('v', v_str)) {
      std::fprintf(stderr, "FAIL: truncated vectors at test %d\n", t);
      return 2;
    }

    // Drive the syndrome and pulse in_valid for one cycle.
    for (int c = 0; c < C; ++c) top->syndrome_in[c] = (c < (int)s_str.size() && s_str[c] == '1');
    top->in_valid = 1;
    tick();
    top->in_valid = 0;

    // Run until out_valid (bounded guard).
    int guard = 0;
    while (!top->out_valid && guard < 200000) {
      tick();
      ++guard;
    }
    if (!top->out_valid) {
      std::fprintf(stderr, "FAIL: test %d never asserted out_valid\n", t);
      return 2;
    }

    // Compare corr_out[N], obs_flip[OBS], valid_flag.
    int local = 0;
    for (int v = 0; v < N; ++v) {
      int want = (v < (int)h_str.size() && h_str[v] == '1') ? 1 : 0;
      if ((int)top->corr_out[v] != want) ++local;
    }
    uint32_t obs = top->obs_flip;
    for (int o = 0; o < OBS; ++o) {
      int want = (o < (int)o_str.size() && o_str[o] == '1') ? 1 : 0;
      if ((int)((obs >> o) & 1) != want) ++local;
    }
    int vwant = (!v_str.empty() && v_str[0] == '1') ? 1 : 0;
    if ((int)top->valid_flag != vwant) ++local;

    if (top->latency_cycles > worst_lat) worst_lat = top->latency_cycles;
    if (local) {
      if (mism < 8) std::fprintf(stderr, "  test %d: %d field mismatches\n", t, local);
      ++mism;
    }
  }

  top->final();
  delete top;

  if (mism == 0) {
    std::printf("PASS: %d full decodes bit-identical to the fixed-point golden; worst latency = %d cycles\n",
                T, worst_lat);
    return 0;
  }
  std::printf("FAIL: %d/%d decodes mismatched\n", mism, T);
  return 1;
}
