// Q7-06 (AC-1) — Verilator batch co-sim for the AXI4-Stream banked-decoder front-end
// (`bp_stream_banked_core`). Drives the SAME 40-shot circuit-level golden (`bp_circ_vectors.txt`) that
// `bpbanked`/`bpaxibanked` pass, but as ONE batched AXI-DMA-style transfer: all T experiments streamed
// back-to-back through MM2S (NS=ceil(C/32) syndrome beats each), results collected from S2MM (one status
// word each), asserting each experiment's {obs, vflag} is bit-identical to golden. Gates:
//   1. golden-equality  — 40/40 obs+vflag match, streamed as a single batch (the AXI-Lite twin's result).
//   2. tlast framing    — exactly T result words, tlast set on the last one only.
//   3. back-pressure invariance — rerun with a randomly-stalled S2MM (m_axis_tready toggled): identical
//      results, proving the shallow output FIFO + input gate never drop or duplicate a decode.
// A pass certifies the batch shell before the Vivado DMA bitstream build.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_stream_banked_core.h"
#include "verilated.h"

static Vbp_stream_banked_core *top;

static void tick() {
  top->aclk = 0;
  top->eval();
  top->aclk = 1;
  top->eval();
}

struct Vec {
  std::vector<uint32_t> synd;  // NS words
  uint32_t obs;                // OBS bits
  int vflag;
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vpath = (argc > 1) ? argv[1] : "bp_circ_vectors.txt";

  std::ifstream f(vpath);
  if (!f) {
    std::fprintf(stderr, "FAIL: open %s\n", vpath.c_str());
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
  const int NS = (C + 31) / 32;

  auto tagged = [&](char tag, std::string &out) {
    while (std::getline(f, line))
      if (!line.empty() && line[0] == tag) {
        size_t p = line.find_first_not_of(" \t", 1);
        out = (p == std::string::npos) ? "" : line.substr(p);
        return true;
      }
    return false;
  };

  std::vector<Vec> vecs;
  for (int t = 0; t < T; ++t) {
    std::string s_str, h_str, o_str, v_str;
    if (!tagged('s', s_str) || !tagged('h', h_str) || !tagged('o', o_str) || !tagged('v', v_str)) {
      std::fprintf(stderr, "FAIL: truncated vectors at test %d\n", t);
      return 2;
    }
    Vec vv;
    vv.synd.assign(NS, 0);
    for (int c = 0; c < C; ++c)
      if (c < (int)s_str.size() && s_str[c] == '1') vv.synd[c / 32] |= (1u << (c % 32));
    vv.obs = 0;
    for (int o = 0; o < OBS; ++o)
      if (o < (int)o_str.size() && o_str[o] == '1') vv.obs |= (1u << o);
    vv.vflag = (!v_str.empty() && v_str[0] == '1') ? 1 : 0;
    vecs.push_back(vv);
  }

  const uint32_t OBS_MASK = (OBS >= 32) ? 0xFFFFFFFFu : ((1u << OBS) - 1);

  // Flatten the batch into a beat list: NS beats per experiment, tlast on the final beat.
  struct Beat { uint32_t data; int last; };
  std::vector<Beat> beats;
  for (int t = 0; t < T; ++t)
    for (int w = 0; w < NS; ++w)
      beats.push_back({vecs[t].synd[w], 0});
  beats.back().last = 1;

  // One run of the whole batch through the AXIS shell. `stall_mask` toggles S2MM ready to exercise
  // back-pressure (0 = always ready). Returns collected {obs,vflag,last} per result word.
  auto run_batch = [&](unsigned stall_mask,
                       std::vector<uint32_t> &obs_out, std::vector<int> &vf_out,
                       std::vector<int> &last_out) -> const char * {
    top->aresetn = 0;
    top->early_exit_i = 0;
    top->s_axis_tvalid = 0;
    top->s_axis_tdata = 0;
    top->s_axis_tlast = 0;
    top->m_axis_tready = 0;
    for (int i = 0; i < 4; ++i) tick();
    top->aresetn = 1;

    size_t bi = 0;                 // next input beat
    int collected = 0;
    unsigned lfsr = 0xACE1u ^ stall_mask;
    long guard = 0;
    const long GUARD_MAX = 200000L * (long)(T + 1);

    while (collected < T && guard++ < GUARD_MAX) {
      // ---- drive inputs for this cycle ----
      if (bi < beats.size()) {
        top->s_axis_tvalid = 1;
        top->s_axis_tdata = beats[bi].data;
        top->s_axis_tlast = beats[bi].last;
      } else {
        top->s_axis_tvalid = 0;
        top->s_axis_tlast = 0;
      }
      int mready = stall_mask ? (lfsr & 1) : 1;
      top->m_axis_tready = mready;

      // ---- settle combinational, sample handshakes ----
      top->aclk = 0;
      top->eval();
      int in_fire = top->s_axis_tvalid && top->s_axis_tready;
      int out_fire = top->m_axis_tvalid && top->m_axis_tready;
      uint32_t rdata = top->m_axis_tdata;
      int rlast = top->m_axis_tlast;

      // ---- clock edge ----
      top->aclk = 1;
      top->eval();

      if (in_fire && bi < beats.size()) ++bi;
      if (out_fire) {
        obs_out.push_back((rdata >> 20) & OBS_MASK);
        vf_out.push_back((rdata >> 19) & 1);
        last_out.push_back(rlast);
        ++collected;
      }
      if (stall_mask) lfsr = (lfsr >> 1) ^ (-(int)(lfsr & 1) & 0xB400u);
    }
    if (collected != T) return "timeout / wrong result count";
    return nullptr;
  };

  top = new Vbp_stream_banked_core;

  // ---- run 1: no back-pressure ----
  std::vector<uint32_t> obs1;
  std::vector<int> vf1, last1;
  if (const char *e = run_batch(0, obs1, vf1, last1)) {
    std::printf("FAIL: batch run (no stall): %s\n", e);
    return 1;
  }

  int mism = 0;
  for (int t = 0; t < T; ++t) {
    if (obs1[t] != (vecs[t].obs & OBS_MASK) || vf1[t] != vecs[t].vflag) {
      if (mism < 8)
        std::fprintf(stderr, "  test %d: obs got 0x%03x want 0x%03x, vflag got %d want %d\n",
                     t, obs1[t], vecs[t].obs & OBS_MASK, vf1[t], vecs[t].vflag);
      ++mism;
    }
  }
  // tlast framing: exactly the last word tagged.
  int tlast_errs = 0;
  for (int t = 0; t < T; ++t) {
    int want = (t == T - 1) ? 1 : 0;
    if (last1[t] != want) ++tlast_errs;
  }

  // ---- run 2: randomly-stalled S2MM (back-pressure invariance) ----
  std::vector<uint32_t> obs2;
  std::vector<int> vf2, last2;
  if (const char *e = run_batch(0xFFFFu, obs2, vf2, last2)) {
    std::printf("FAIL: batch run (stalled S2MM): %s\n", e);
    return 1;
  }
  int bp_errs = 0;
  for (int t = 0; t < T; ++t)
    if (obs2[t] != obs1[t] || vf2[t] != vf1[t] || last2[t] != last1[t]) ++bp_errs;

  top->final();
  delete top;

  if (mism == 0 && tlast_errs == 0 && bp_errs == 0) {
    std::printf(
        "PASS: %d circuit-level decodes bit-identical to golden as ONE batched AXIS transfer; "
        "tlast framing exact; back-pressure invariant\n",
        T);
    return 0;
  }
  std::printf("FAIL: golden-mism=%d tlast-errs=%d backpressure-errs=%d (of %d)\n",
              mism, tlast_errs, bp_errs, T);
  return 1;
}
