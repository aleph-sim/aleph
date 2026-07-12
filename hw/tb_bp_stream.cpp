// Q7-04 M9b — Verilator testbench for the SLIDING-WINDOW STREAMING wrapper (`bp_streaming_decoder`).
//
// Drives the HW-schedule goldens (40 trials; each = SLICES raw measurement rounds of BP_DPR bits streamed
// in, producing SLOTS committed slots) through the wrapper's warm/run/wait/commit/slide/reload FSM and
// asserts the per-slot {commit_corr[BP_N], out_obs, out_vflag, out_commit_clean} (+ out_last placement)
// bit-exact against the golden. The 40 trials are driven BACK-TO-BACK without an external reset between
// them — the frame-independence check (the FSM returns to S_WARM after each final slot).
//
// PER-MODE GOLDENS (the M6–M8 circvectors/circvectorsearly house pattern): the RTL core's early-exit mode
// commits the FIRST syndrome-valid leg, while the full decode commits the BEST-KEPT (lowest-weight valid)
// decision over the whole schedule — where first-valid != best-kept the decisions legitimately differ, so
// each mode is gated against its own golden:
//   argv[1] = best-kept golden  (bp_stream_vectors.txt)       -> run with early_exit=0
//   argv[2] = early-exit golden (bp_stream_vectors_early.txt) -> run with early_exit=1
// Same format, same header, identical r-lines; only the w-lines differ (25/280 slots on the committed
// vectors — matching the divergence set the single-golden co-sim found).
//
// Handshake: `in_ready` is high only in warm/reload before the stream's last round is consumed; the TB
// holds a round on in_round/in_valid and ticks until the DUT accepts it (in_ready high at the sampling
// edge), meanwhile capturing any out_valid slot pulses. After the last real round, the TB keeps ticking
// (in_valid=0) so the internal zero-pad drain can emit the tail slots. Guard: 32 M cycles/trial.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_streaming_decoder.h"
#include "verilated.h"

using Dut = Vbp_streaming_decoder;

struct GSlot {
  std::string committed;  // BP_N bits, var 0 first
  std::string obs;        // BP_OBS bits, bit 0 first
  int vflag;
  int clean;
};
struct GTrial {
  std::vector<std::string> rounds;  // SLICES strings of BP_DPR bits, det 0 first
  std::vector<GSlot> slots;         // SLOTS golden slots
};
struct Golden {
  int T = 0, SLICES = 0, DPR = 0, SLOTS = 0, N = 0, OBS = 0;
  std::vector<GTrial> trials;
};

static Dut *top;
static uint64_t g_cycles = 0;  // cycles in the current trial (guard)

// One posedge. Samples in_ready BEFORE the edge (combinational), advances the clock, and returns whether
// the DUT accepted an input this edge (in_ready was high). out_valid is sampled by the caller after.
static bool posedge() {
  top->clk = 0;
  top->eval();
  bool ready = top->in_ready;
  top->clk = 1;
  top->eval();
  ++g_cycles;
  return ready;
}

// Parse one golden file: header 'T SLICES DPR SLOTS BP_N BP_OBS', then per trial SLICES 'r' lines and
// SLOTS 'w' lines. Returns false (with a message) on any format problem.
static bool parse_golden(const std::string &path, Golden &g) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "FAIL: open %s\n", path.c_str());
    return false;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream(line) >> g.T >> g.SLICES >> g.DPR >> g.SLOTS >> g.N >> g.OBS;
    break;
  }
  if (g.T <= 0 || g.SLICES <= 0 || g.DPR <= 0 || g.SLOTS <= 0 || g.N <= 0 || g.OBS <= 0) {
    std::fprintf(stderr, "FAIL: bad header in %s: T=%d SLICES=%d DPR=%d SLOTS=%d N=%d OBS=%d\n",
                 path.c_str(), g.T, g.SLICES, g.DPR, g.SLOTS, g.N, g.OBS);
    return false;
  }
  g.trials.assign(g.T, GTrial{});
  for (int t = 0; t < g.T; ++t) {
    g.trials[t].rounds.reserve(g.SLICES);
    while ((int)g.trials[t].rounds.size() < g.SLICES && std::getline(f, line)) {
      if (line.empty() || line[0] != 'r') continue;
      std::istringstream is(line);
      char tag;
      std::string bits;
      is >> tag >> bits;
      g.trials[t].rounds.push_back(bits);
    }
    g.trials[t].slots.reserve(g.SLOTS);
    while ((int)g.trials[t].slots.size() < g.SLOTS && std::getline(f, line)) {
      if (line.empty() || line[0] != 'w') continue;
      std::istringstream is(line);
      char tag;
      int slot;
      GSlot gs;
      is >> tag >> slot >> gs.committed >> gs.obs >> gs.vflag >> gs.clean;
      g.trials[t].slots.push_back(gs);
    }
    if ((int)g.trials[t].rounds.size() != g.SLICES || (int)g.trials[t].slots.size() != g.SLOTS) {
      std::fprintf(stderr, "FAIL: truncated vectors in %s at trial %d (rounds=%zu slots=%zu)\n",
                   path.c_str(), t, g.trials[t].rounds.size(), g.trials[t].slots.size());
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vec_full = (argc > 1) ? argv[1] : "bp_stream_vectors.txt";
  const std::string vec_early = (argc > 2) ? argv[2] : "bp_stream_vectors_early.txt";

  Golden gold[2];  // [0] = best-kept (early_exit=0), [1] = first-valid (early_exit=1)
  if (!parse_golden(vec_full, gold[0]) || !parse_golden(vec_early, gold[1])) return 2;
  if (gold[0].T != gold[1].T || gold[0].SLICES != gold[1].SLICES || gold[0].DPR != gold[1].DPR ||
      gold[0].SLOTS != gold[1].SLOTS || gold[0].N != gold[1].N || gold[0].OBS != gold[1].OBS) {
    std::fprintf(stderr, "FAIL: golden headers disagree between %s and %s\n", vec_full.c_str(),
                 vec_early.c_str());
    return 2;
  }

  top = new Dut;
  const int DPR = gold[0].DPR;
  const int NWORDS = (DPR + 31) / 32;

  auto set_round = [&](const std::string &bits) {
    uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < DPR; ++i)
      if (i < (int)bits.size() && bits[i] == '1') w[i / 32] |= (1u << (i % 32));
    for (int k = 0; k < NWORDS; ++k) top->in_round[k] = w[k];
  };

  // Run every trial back-to-back for one early_exit mode against its own golden; returns field-mismatch
  // count. Fills the per-window latency stats.
  auto run_mode = [&](int ee, const Golden &g, uint32_t &worst_lat, uint64_t &lat_sum, uint64_t &lat_cnt,
                      std::map<uint32_t, int> &lat_hist) -> long {
    const int T = g.T, SLICES = g.SLICES, SLOTS = g.SLOTS, N = g.N, OBS = g.OBS;

    // Synchronous reset for this mode.
    top->rst_n = 0;
    top->in_valid = 0;
    top->in_last = 0;
    top->early_exit = ee;
    for (int i = 0; i < 5; ++i) posedge();
    top->rst_n = 1;
    posedge();

    long mism = 0;
    int printed = 0;
    const uint64_t GUARD = 32ull * 1000ull * 1000ull;

    for (int t = 0; t < T; ++t) {
      g_cycles = 0;
      int seen = 0;  // slots captured this trial

      // Capture one slot pulse and compare against golden slot `seen`.
      auto capture = [&]() {
        if (!top->out_valid) return;
        int k = seen;
        if (k >= SLOTS) {
          std::fprintf(stderr, "  mode %d trial %d: extra out_valid beyond %d slots\n", ee, t, SLOTS);
          ++mism;
          ++seen;
          return;
        }
        const GSlot &gs = g.trials[t].slots[k];
        int local = 0;
        for (int v = 0; v < N; ++v) {
          int want = (v < (int)gs.committed.size() && gs.committed[v] == '1') ? 1 : 0;
          if ((int)top->commit_corr[v] != want) ++local;
        }
        uint32_t obs = top->out_obs;
        for (int o = 0; o < OBS; ++o) {
          int want = (o < (int)gs.obs.size() && gs.obs[o] == '1') ? 1 : 0;
          if ((int)((obs >> o) & 1) != want) ++local;
        }
        if ((int)top->out_vflag != gs.vflag) ++local;
        if ((int)top->out_commit_clean != gs.clean) ++local;
        if ((int)top->out_last != (k == SLOTS - 1 ? 1 : 0)) ++local;

        uint32_t lat = top->last_latency;
        if (lat > worst_lat) worst_lat = lat;
        lat_sum += lat;
        ++lat_cnt;
        ++lat_hist[lat];

        if (local && printed < 40) {
          ++printed;
          std::fprintf(stderr,
                       "  mode %d trial %d slot %d: %d field mismatches (vflag dut=%d gold=%d, "
                       "clean dut=%d gold=%d)\n",
                       ee, t, k, local, (int)top->out_vflag, gs.vflag, (int)top->out_commit_clean,
                       gs.clean);
        }
        mism += local;
        ++seen;
      };

      // Drive the SLICES raw rounds; hold each until accepted, capturing slot pulses meanwhile.
      for (int s = 0; s < SLICES; ++s) {
        set_round(g.trials[t].rounds[s]);
        top->in_valid = 1;
        top->in_last = (s == SLICES - 1) ? 1 : 0;
        bool accepted = false;
        while (!accepted) {
          if (g_cycles > GUARD) {
            std::fprintf(stderr, "FAIL: mode %d trial %d exceeded %llu cycles driving round %d\n", ee, t,
                         (unsigned long long)GUARD, s);
            return mism + 1;
          }
          bool ready = posedge();
          capture();
          if (ready) accepted = true;
        }
        top->in_valid = 0;
        top->in_last = 0;
      }

      // Drain the tail (internal zero-pad) until all SLOTS are emitted.
      while (seen < SLOTS) {
        if (g_cycles > GUARD) {
          std::fprintf(stderr, "FAIL: mode %d trial %d exceeded %llu cycles draining (seen=%d)\n", ee, t,
                       (unsigned long long)GUARD, seen);
          return mism + 1;
        }
        posedge();
        capture();
      }
    }
    return mism;
  };

  long total_mism = 0;
  uint32_t worst_lat[2] = {0, 0};
  uint64_t lat_sum[2] = {0, 0}, lat_cnt[2] = {0, 0};
  std::map<uint32_t, int> lat_hist[2];
  const char *gname[2] = {vec_full.c_str(), vec_early.c_str()};

  for (int ee = 0; ee <= 1; ++ee) {
    long m = run_mode(ee, gold[ee], worst_lat[ee], lat_sum[ee], lat_cnt[ee], lat_hist[ee]);
    if (m == 0)
      std::printf("PASS: early_exit=%d — %d trials x %d slots bit-identical to %s\n", ee, gold[ee].T,
                  gold[ee].SLOTS, gname[ee]);
    else
      std::printf("FAIL: early_exit=%d — %ld field mismatch(es) vs %s\n", ee, m, gname[ee]);
    total_mism += m;
  }

  top->final();
  delete top;

  for (int ee = 0; ee <= 1; ++ee) {
    std::printf("early_exit=%d per-window latency (cycles -> slots):", ee);
    for (auto &kv : lat_hist[ee]) std::printf(" %u:%d", kv.first, kv.second);
    std::printf("\n");
    std::printf(
        "early_exit=%d per-window latency: worst = %u cycles, mean = %.1f cycles over %llu windows\n", ee,
        worst_lat[ee], lat_cnt[ee] ? (double)lat_sum[ee] / (double)lat_cnt[ee] : 0.0,
        (unsigned long long)lat_cnt[ee]);
  }

  if (total_mism == 0) {
    std::printf(
        "PASS: %d trials x %d slots x 2 early-exit modes, each bit-exact vs its own HW-schedule golden\n",
        gold[0].T, gold[0].SLOTS);
    return 0;
  }
  std::printf("FAIL: %ld total field mismatches\n", total_mism);
  return 1;
}
