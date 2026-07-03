// Q7-02 M5-followup — Verilator TB for the WIDE AXI4-Lite wrapper (bp_axi_wrap_wide) around the M2
// circuit-level decoder. Drives REAL AXI4-Lite transactions per golden vector — the sim twin of the
// on-board driver — over the generic register map (NS syndrome words, NC correction words derived from
// BP_C/BP_N). A pass certifies the wide PS<->PL shell before the Vivado board build.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_axi_wrap_wide.h"
#include "verilated.h"

static const int A_CTRL = 0x00, A_STATUS = 0x04, A_LAT = 0x08, A_OBS = 0x0C, A_ID = 0x10;
static const int SYND_BASE = 0x40, CORR_BASE = 0x80;
static const uint32_t IDCODE = 0x42500002u;

static Vbp_axi_wrap_wide *top;

static void tick() {
  top->aclk = 0;
  top->eval();
  top->aclk = 1;
  top->eval();
}

static void axil_write(int addr, uint32_t data) {
  top->s_axil_awaddr = addr;
  top->s_axil_awvalid = 1;
  top->s_axil_wdata = data;
  top->s_axil_wstrb = 0xF;
  top->s_axil_wvalid = 1;
  int g = 0;
  while (!(top->s_axil_awready && top->s_axil_wready) && g++ < 1000) tick();
  tick();
  top->s_axil_awvalid = 0;
  top->s_axil_wvalid = 0;
  top->s_axil_bready = 1;
  g = 0;
  while (!top->s_axil_bvalid && g++ < 1000) tick();
  tick();
  top->s_axil_bready = 0;
}

static uint32_t axil_read(int addr) {
  top->s_axil_araddr = addr;
  top->s_axil_arvalid = 1;
  int g = 0;
  while (!top->s_axil_arready && g++ < 1000) tick();
  tick();
  top->s_axil_arvalid = 0;
  top->s_axil_rready = 1;
  g = 0;
  while (!top->s_axil_rvalid && g++ < 1000) tick();
  uint32_t d = top->s_axil_rdata;
  tick();
  top->s_axil_rready = 0;
  return d;
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
  const int NS = (C + 31) / 32, NC = (N + 31) / 32;

  top = new Vbp_axi_wrap_wide;
  top->aresetn = 0;
  top->s_axil_awvalid = top->s_axil_wvalid = top->s_axil_bready = 0;
  top->s_axil_arvalid = top->s_axil_rready = 0;
  for (int i = 0; i < 4; ++i) tick();
  top->aresetn = 1;
  tick();

  uint32_t id = axil_read(A_ID);
  if (id != IDCODE) {
    std::fprintf(stderr, "FAIL: IDCODE=0x%08x expected 0x%08x\n", id, IDCODE);
    return 1;
  }

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
  long worst_lat = 0;
  for (int t = 0; t < T; ++t) {
    std::string s_str, h_str, o_str, v_str;
    if (!tagged('s', s_str) || !tagged('h', h_str) || !tagged('o', o_str) || !tagged('v', v_str)) {
      std::fprintf(stderr, "FAIL: truncated vectors at test %d\n", t);
      return 2;
    }
    // pack syndrome (bit c <- s_str[c]) into NS words, write them, then START
    std::vector<uint32_t> sw(NS, 0);
    for (int c = 0; c < C; ++c)
      if (c < (int)s_str.size() && s_str[c] == '1') sw[c / 32] |= (1u << (c % 32));
    for (int w = 0; w < NS; ++w) axil_write(SYND_BASE + 4 * w, sw[w]);
    axil_write(A_CTRL, 1);

    int g = 0;
    uint32_t st = 0;
    while (g++ < 4000000) {
      st = axil_read(A_STATUS);
      if (st & 0x2) break;
    }
    if (!(st & 0x2)) {
      std::fprintf(stderr, "FAIL: test %d DONE never asserted\n", t);
      return 2;
    }
    int got_v = (st >> 2) & 1;

    std::vector<uint32_t> cw(NC);
    for (int w = 0; w < NC; ++w) cw[w] = axil_read(CORR_BASE + 4 * w);
    uint32_t obs = axil_read(A_OBS);
    long lat = axil_read(A_LAT);
    if (lat > worst_lat) worst_lat = lat;

    int local = 0;
    for (int v = 0; v < N; ++v) {
      int want = (v < (int)h_str.size() && h_str[v] == '1') ? 1 : 0;
      int got = (cw[v / 32] >> (v % 32)) & 1;
      if (got != want) ++local;
    }
    for (int o = 0; o < OBS; ++o) {
      int want = (o < (int)o_str.size() && o_str[o] == '1') ? 1 : 0;
      if ((int)((obs >> o) & 1) != want) ++local;
    }
    int vwant = (!v_str.empty() && v_str[0] == '1') ? 1 : 0;
    if (got_v != vwant) ++local;

    if (local) {
      if (mism < 8) std::fprintf(stderr, "  test %d: %d field mismatches\n", t, local);
      ++mism;
    }
  }
  top->final();
  delete top;

  if (mism == 0) {
    std::printf(
        "PASS: %d circuit-level decodes bit-identical to golden over wide AXI4-Lite; worst latency = %ld cycles\n",
        T, worst_lat);
    return 0;
  }
  std::printf("FAIL: %d/%d decodes mismatched over wide AXI4-Lite\n", mism, T);
  return 1;
}
