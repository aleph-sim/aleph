// Q7-02 M1 — Verilator testbench for the fixed-point min-sum check-update datapath.
//
// Replays the Rust golden vectors (`bp_check_vectors.txt` from
// `FixedRelayBp::check_update_once`): for each `(syndrome, m_vc)` it drives the combinational RTL and
// checks every one of the 432 output messages `e_cv` bit-for-bit against the fixed-point golden. A
// pass certifies the RTL check-update is the exact silicon twin of the M0 golden at Q5.3.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_check_update.h"
#include "verilated.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vec = (argc > 1) ? argv[1] : "bp_check_vectors.txt";

  std::ifstream f(vec);
  if (!f) {
    std::fprintf(stderr, "FAIL: open %s\n", vec.c_str());
    return 2;
  }

  int T = 0, C = 0, E = 0;
  std::string line;
  // First non-comment line is the header "T C E".
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream(line) >> T >> C >> E;
    break;
  }
  if (T <= 0 || C <= 0 || E <= 0) {
    std::fprintf(stderr, "FAIL: bad header T=%d C=%d E=%d\n", T, C, E);
    return 2;
  }

  auto *top = new Vbp_check_update;
  int checked = 0, mismatches = 0;

  for (int t = 0; t < T; ++t) {
    std::vector<int> mvc(E, 0), egold(E, 0);
    std::vector<int> sbits(C, 0);

    // 's' line: C bits, no spaces.
    while (std::getline(f, line) && (line.empty() || line[0] == '#')) {
    }
    {
      // strip leading 's' tag and spaces
      size_t p = line.find_first_not_of(" \t", 1);
      for (int c = 0; c < C && p + c < line.size(); ++c)
        sbits[c] = (line[p + c] == '1') ? 1 : 0;
    }
    // 'm' line: E ints.
    std::getline(f, line);
    {
      std::istringstream ss(line.substr(1));
      for (int e = 0; e < E; ++e) ss >> mvc[e];
    }
    // 'e' line: E ints (expected).
    std::getline(f, line);
    {
      std::istringstream ss(line.substr(1));
      for (int e = 0; e < E; ++e) ss >> egold[e];
    }

    // Drive inputs (8-bit signed words as raw bytes).
    for (int c = 0; c < C; ++c) top->s_in[c] = sbits[c] & 1;
    for (int e = 0; e < E; ++e) top->m_vc[e] = (uint8_t)(mvc[e] & 0xFF);
    top->eval();

    for (int e = 0; e < E; ++e) {
      int got = (int8_t)top->e_cv[e];  // sign-extend the 8-bit output
      ++checked;
      if (got != egold[e]) {
        if (mismatches < 10)
          std::fprintf(stderr, "  test %d edge %d: RTL %d != golden %d\n", t, e, got, egold[e]);
        ++mismatches;
      }
    }
  }

  top->final();
  delete top;

  if (mismatches == 0) {
    std::printf("PASS: %d check-update outputs over %d vectors match the fixed-point golden\n",
                checked, T);
    return 0;
  }
  std::printf("FAIL: %d/%d check-update outputs mismatched\n", mismatches, checked);
  return 1;
}
