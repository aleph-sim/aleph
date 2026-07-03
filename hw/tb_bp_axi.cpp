// Q7-02 board bring-up — Verilator testbench for the AXI4-Lite wrapper (bp_axi_wrap) around the
// partial relay-BP decoder. Drives REAL AXI4-Lite transactions (multi-word syndrome writes, START,
// poll DONE, multi-word correction reads) exactly as the on-board `bp_pynq.py` driver does, and checks
// corr_out[BP_N]/obs_flip[BP_OBS]/valid_flag bit-for-bit against the Rust golden (`bp_dec_vectors.txt`).
// A pass certifies the PS<->PL wrapper is a faithful shell over the already-verified decoder before we
// spend a Vivado build — the wrapper is the only new logic on the board path.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "Vbp_axi_wrap.h"
#include "verilated.h"

// Register map (byte offsets, mirrors bp_axi_wrap.sv)
static const int A_CTRL = 0x00, A_STATUS = 0x04, A_SYND0 = 0x08 /*..0x10*/;
static const int A_CORR0 = 0x14 /*..0x24*/, A_OBS = 0x28, A_LAT = 0x2C, A_ID = 0x30;
static const uint32_t IDCODE = 0x42500001u;

static Vbp_axi_wrap *top;

static void tick() {
  top->aclk = 0;
  top->eval();
  top->aclk = 1;
  top->eval();
}

// ---- minimal AXI4-Lite master ----
static void axil_write(int addr, uint32_t data) {
  top->s_axil_awaddr = addr;
  top->s_axil_awvalid = 1;
  top->s_axil_wdata = data;
  top->s_axil_wstrb = 0xF;
  top->s_axil_wvalid = 1;
  int g = 0;
  while (!(top->s_axil_awready && top->s_axil_wready) && g++ < 1000) tick();
  tick();  // consume the accept cycle
  top->s_axil_awvalid = 0;
  top->s_axil_wvalid = 0;
  // handshake the write response
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

  top = new Vbp_axi_wrap;
  top->aresetn = 0;
  top->s_axil_awvalid = top->s_axil_wvalid = top->s_axil_bready = 0;
  top->s_axil_arvalid = top->s_axil_rready = 0;
  for (int i = 0; i < 4; ++i) tick();
  top->aresetn = 1;
  tick();

  // IDCODE sanity (the on-board `probe()`)
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

  int mism = 0, worst_lat = 0;
  for (int t = 0; t < T; ++t) {
    std::string s_str, h_str, o_str, v_str;
    if (!tagged('s', s_str) || !tagged('h', h_str) || !tagged('o', o_str) || !tagged('v', v_str)) {
      std::fprintf(stderr, "FAIL: truncated vectors at test %d\n", t);
      return 2;
    }

    // pack syndrome bits (bit i <- s_str[i]) into 3 words, write them, then START
    uint32_t sw[3] = {0, 0, 0};
    for (int c = 0; c < C; ++c)
      if (c < (int)s_str.size() && s_str[c] == '1') sw[c / 32] |= (1u << (c % 32));
    for (int w = 0; w < 3; ++w) axil_write(A_SYND0 + 4 * w, sw[w]);
    axil_write(A_CTRL, 1);  // START

    // poll STATUS.DONE
    int g = 0;
    uint32_t st = 0;
    while (g++ < 400000) {
      st = axil_read(A_STATUS);
      if (st & 0x2) break;  // DONE
    }
    if (!(st & 0x2)) {
      std::fprintf(stderr, "FAIL: test %d DONE never asserted\n", t);
      return 2;
    }
    int got_v = (st >> 2) & 1;

    // read correction (5 words) and obs
    uint32_t cw[5];
    for (int w = 0; w < 5; ++w) cw[w] = axil_read(A_CORR0 + 4 * w);
    uint32_t obs = axil_read(A_OBS);
    int lat = axil_read(A_LAT) & 0xFFFF;
    if (lat > worst_lat) worst_lat = lat;

    // compare
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
        "PASS: %d full decodes bit-identical to golden over AXI4-Lite; worst latency = %d cycles\n", T,
        worst_lat);
    return 0;
  }
  std::printf("FAIL: %d/%d decodes mismatched over AXI4-Lite\n", mism, T);
  return 1;
}
