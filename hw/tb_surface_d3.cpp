// Q6-01 (sim) — Verilator testbench for the d=3 surface-code LUT decoder.
//
// Drives every one of the 2^8 syndromes through the RTL and checks the registered `correction`
// output against the Rust-generated oracle (`surface_d3_lut.mem`, the same table the Rust Union-Find
// decoder produced). A pass certifies the RTL ROM, addressing, and 1-cycle pipeline faithfully
// realise the decoder LUT in simulation. Also reports the measured input->output latency in clocks.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Vsurface_d3_decoder.h"
#include "verilated.h"

static constexpr int kSyndromeBits = 8;
static constexpr int kDepth = 1 << kSyndromeBits;

// Load the oracle .mem (binary lines, `//` comments skipped), exactly as $readmemb does.
static std::vector<uint8_t> load_oracle(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "FAIL: cannot open oracle %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<uint8_t> table;
  std::string line;
  while (std::getline(f, line)) {
    // Strip a `//` comment and surrounding whitespace.
    auto pos = line.find("//");
    if (pos != std::string::npos) line = line.substr(0, pos);
    std::string tok;
    for (char c : line)
      if (c == '0' || c == '1') tok.push_back(c);
    if (!tok.empty()) table.push_back(tok.back() == '1' ? 1 : 0);
  }
  return table;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string mem = (argc > 1) ? argv[1] : "surface_d3_lut.mem";
  std::vector<uint8_t> oracle = load_oracle(mem);
  if (static_cast<int>(oracle.size()) != kDepth) {
    std::fprintf(stderr, "FAIL: oracle has %zu entries, expected %d\n", oracle.size(), kDepth);
    return 2;
  }

  auto *dut = new Vsurface_d3_decoder;
  auto tick = [&](void) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
  };

  // Reset.
  dut->rst_n = 0; dut->in_valid = 0; dut->syndrome = 0;
  tick(); tick();
  dut->rst_n = 1;

  // Latency probe: present a request, count clocks until out_valid.
  dut->in_valid = 1; dut->syndrome = 0; tick();
  dut->in_valid = 0;
  int latency = 1;          // out_valid is registered from the cycle in_valid was sampled
  while (!dut->out_valid && latency < 16) { tick(); ++latency; }

  // Exhaustive check: drive each syndrome, read the correction one clock later.
  int fails = 0;
  for (int s = 0; s < kDepth; ++s) {
    dut->in_valid = 1; dut->syndrome = s; tick();   // sample request
    dut->in_valid = 0; dut->eval();
    // `correction`/`out_valid` are now valid (registered on the edge just clocked).
    if (!dut->out_valid) { std::fprintf(stderr, "FAIL s=%d: out_valid low\n", s); ++fails; }
    if (dut->correction != oracle[s]) {
      std::fprintf(stderr, "FAIL s=%d: rtl=%d oracle=%d\n", s, dut->correction, oracle[s]);
      if (++fails > 20) break;
    }
  }

  dut->final();
  delete dut;
  if (fails) { std::printf("FAILED: %d mismatches\n", fails); return 1; }
  std::printf("PASS: %d/%d syndromes match oracle; decode latency = %d clock(s)\n",
              kDepth, kDepth, latency);
  return 0;
}
