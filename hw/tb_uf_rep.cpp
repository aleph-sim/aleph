// Q6-02 (sim) — Verilator testbench for the repetition-code Union-Find decoder.
//
// For every one of the 2^(D-1) syndromes it drives the RTL and checks two things:
//   1. validity   — the RTL correction reproduces the input syndrome (c_i = chat_i ^ chat_{i+1}),
//                    i.e. the decoder actually solves H*chat = s;
//   2. logical    — the RTL's predicted logical flip matches the Rust UnionFindDecoder oracle
//                    (`rep_uf_vectors.mem`), i.e. it picks the same (minimum-weight) coset.
// Also reports the input->output latency in clocks. A pass certifies the RTL is a correct,
// Rust-equivalent decoder datapath in simulation.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Vuf_rep_decoder.h"
#include "verilated.h"

static constexpr int kD = 7;           // must match uf_rep_decoder's D and the Rust generator
static constexpr int kChecks = kD - 1;
static constexpr int kEntries = 1 << kChecks;

// Load the binary oracle (`//` comments skipped), as $readmemb would.
static std::vector<uint8_t> load_oracle(const std::string &path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "FAIL: cannot open %s\n", path.c_str()); std::exit(2); }
  std::vector<uint8_t> t;
  std::string line;
  while (std::getline(f, line)) {
    auto pos = line.find("//");
    if (pos != std::string::npos) line = line.substr(0, pos);
    for (char c : line) if (c == '0' || c == '1') { t.push_back(c == '1'); break; }
  }
  return t;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string mem = (argc > 1) ? argv[1] : "rep_uf_vectors.mem";
  std::vector<uint8_t> oracle = load_oracle(mem);
  if (static_cast<int>(oracle.size()) != kEntries) {
    std::fprintf(stderr, "FAIL: oracle has %zu entries, expected %d\n", oracle.size(), kEntries);
    return 2;
  }

  auto *dut = new Vuf_rep_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };

  dut->rst_n = 0; dut->in_valid = 0; dut->syndrome = 0;
  tick(); tick();
  dut->rst_n = 1;

  // Latency probe.
  dut->in_valid = 1; dut->syndrome = 0; tick();
  dut->in_valid = 0;
  int latency = 1;
  while (!dut->out_valid && latency < 16) { tick(); ++latency; }

  int fails = 0;
  for (int s = 0; s < kEntries; ++s) {
    dut->in_valid = 1; dut->syndrome = s; tick();
    dut->in_valid = 0; dut->eval();
    if (!dut->out_valid) { std::fprintf(stderr, "FAIL s=%d: out_valid low\n", s); ++fails; continue; }

    // Validity: recompute the syndrome from the RTL correction.
    const uint32_t corr = dut->correction;
    int syn_rtl = 0;
    for (int i = 0; i < kChecks; ++i)
      syn_rtl |= (((corr >> i) ^ (corr >> (i + 1))) & 1) << i;
    if (syn_rtl != s) {
      std::fprintf(stderr, "FAIL s=%d: correction 0x%x reproduces syndrome 0x%x\n", s, corr, syn_rtl);
      if (++fails > 20) break;
    }
    // Logical: matches the Rust UF oracle.
    if (dut->obs_flip != oracle[s]) {
      std::fprintf(stderr, "FAIL s=%d: obs rtl=%d oracle=%d\n", s, dut->obs_flip, oracle[s]);
      if (++fails > 20) break;
    }
  }

  dut->final();
  delete dut;
  if (fails) { std::printf("FAILED: %d mismatches\n", fails); return 1; }
  std::printf("PASS: %d/%d syndromes valid + match Rust UF oracle; decode latency = %d clock(s)\n",
              kEntries, kEntries, latency);
  return 0;
}
