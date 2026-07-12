// Q7-04 M9b (Task 6) — Verilator testbench for the AXI4-Stream front-end of the sliding-window banked-BP
// streaming decoder (`bp_stream_win_core`, wrapping `bp_streaming_decoder` from Task 5).
//
// The bare decoder is already bit-exact-verified against the HW-schedule goldens (tb_bp_stream.cpp, both
// early_exit modes). What THIS test gates is that the AXI shell preserves that behaviour under real DMA
// framing: 3 MM2S beats per round in, one 32-bit S2MM result word per committed slot out, correct field
// packing, correct tlast, and — crucially — that back-pressure and repeated (no-external-reset) frames
// never drop, corrupt, or reorder a result word.
//
// PER-MODE GOLDENS (M6-M8 house pattern, same as tb_bp_stream.cpp): early_exit commits the FIRST
// syndrome-valid leg while the full decode commits the BEST-KEPT decision, so each mode is gated against
// its own golden file:
//   argv[1] = best-kept golden  (bp_stream_vectors.txt)       -> run with early_exit=0
//   argv[2] = early-exit golden (bp_stream_vectors_early.txt) -> run with early_exit=1
//
// Gates (each run for BOTH early_exit modes):
//   1. ZERO STREAM   — SLICES all-zero rounds -> SLOTS words, every word obs=0/vflag=1/commit_clean=1,
//                       tlast on the last word only (no premature tlast).
//   2. GOLDEN EQUALITY — all T golden trials, each driven with a fresh external reset; the output words'
//                       {obs, vflag, commit_clean} fields bit-equal the golden w-lines (latency: asserted
//                       > 0, not golden-compared).
//   3. BACK-PRESSURE INVARIANCE — same T trials, full-speed vs splitmix64-random `m_axis_tready`; the
//                       output word sequence (data + tlast) must be byte-identical.
//   4. FRAME INDEPENDENCE — 3 golden trials driven BACK-TO-BACK with NO external reset between them (only
//                       the shell's own per-frame `frame_rst` re-arm), each still producing the exact
//                       golden slot count/fields/tlast placement (gates 1+2, per frame).
//   5. ADVERSARIAL DRAIN STALL — after `in_last`, the decoder self-drives through the tail slots with NO
//                       input handshake (its zero-fill branches never consult in_valid and in_ready is
//                       low), so S2MM back-pressure CANNOT stall the drain: the shell's result FIFO must
//                       absorb the whole tail. Golden trial 0 is driven full-speed, then m_axis_tready is
//                       held LOW from the moment the final round is accepted for STALL_CYCLES = 40000
//                       cycles (> 2x the max observed core latency 16298, so at least two tail slots are
//                       forced to retire INTO the stall), then released; all SLOTS words must arrive
//                       intact (field-compared vs the golden). On the pre-review 1-deep shell this gate
//                       fails: each new dec_out_valid overwrote the unconsumed parked word, losing every
//                       stalled tail slot but the last (verified in the Task-6 report's negative-control
//                       build). Note tready is NOT dropped while the final round's beats are still being
//                       presented: the shell (old and new) only accepts input while the result buffer is
//                       empty, so a pre-drain parked word + early tready-low would deadlock the handshake
//                       by construction (input blocked on the parked word, parked word blocked on tready)
//                       — the adversarial window is the drain itself, which is exactly the un-back-
//                       pressurable region.
//
// Gates 1-4 mirror tb_uf_stream_win.cpp; gate 5 is BP-specific (the UF streaming decoder cannot retire a
// window without consuming input first, so its 1-deep slot was sufficient there).
//
// Result-word layout (bit-exact to the Task-6 contract): [31:20]=obs[11:0], [19]=vflag, [18]=commit_clean,
// [17:16]=00, [15:0]=latency (already 16-bit saturated by the decoder).
//
// Framing: BP_DPR=72 > 32, so one round is 3 MM2S beats: beat0=bits[31:0], beat1=bits[63:32],
// beat2={24'b0,bits[71:64]}. tlast rides the round's final (3rd) beat of the frame's final round.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Vbp_stream_win_core.h"
#include "verilated.h"

using Dut = Vbp_stream_win_core;

// ---- golden parsing (same format as tb_bp_stream.cpp; `committed` (BP_N bits) is parsed but unused here
// — commit_corr is not visible on the AXI result word) ----
struct GSlot {
  std::string committed;  // unused (debug-only field on the bare decoder)
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

// splitmix64 — deterministic per-seed stream (same generator as tb_uf_stream_win.cpp / tb_bp_stream.cpp
// house style).
struct Rng {
  uint64_t z;
  explicit Rng(uint64_t seed) : z(seed) {}
  uint64_t next() {
    z += 0x9E3779B97F4A7C15ull;
    uint64_t x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }
};

struct OutWord {
  uint32_t data;
  bool last;
};

static uint32_t obs_of(uint32_t w) { return (w >> 20) & 0xFFFu; }
static int vflag_of(uint32_t w) { return (int)((w >> 19) & 1u); }
static int clean_of(uint32_t w) { return (int)((w >> 18) & 1u); }
static uint32_t lat_of(uint32_t w) { return w & 0xFFFFu; }

// Pack one round's bit string (det 0 first) into the 3 MM2S beats: beat0=[31:0], beat1=[63:32],
// beat2={24'b0,[DPR-1:32*2]}.
static std::array<uint32_t, 3> pack_round(const std::string &bits, int dpr) {
  std::array<uint32_t, 3> w{0u, 0u, 0u};
  for (int i = 0; i < dpr; ++i)
    if (i < (int)bits.size() && bits[i] == '1') w[i / 32] |= (1u << (i % 32));
  return w;
}

static Dut *dut = nullptr;

static void tick() {
  dut->aclk = 0;
  dut->eval();
  dut->aclk = 1;
  dut->eval();
}

static void reset(int early_exit) {
  dut->aresetn = 0;
  dut->early_exit_i = early_exit;
  dut->s_axis_tvalid = 0;
  dut->s_axis_tdata = 0;
  dut->s_axis_tlast = 0;
  dut->m_axis_tready = 0;
  tick();
  tick();
  dut->aresetn = 1;
}

// Drive one frame (SLICES rounds, each already packed into 3 beats) over s_axis and collect the S2MM
// result words. `bp_seed`==0 -> m_axis always ready (full speed); otherwise pseudo-random back-pressure.
// `do_reset`==false runs the frame on a DUT that was NOT externally re-armed (the wrapper's own
// `frame_rst` must make it independent).
static std::vector<OutWord> run_frame(const std::vector<std::array<uint32_t, 3>> &round_beats,
                                      uint64_t bp_seed, bool do_reset, int early_exit) {
  std::vector<OutWord> out;
  Rng bp(bp_seed ? bp_seed : 1);
  if (do_reset) reset(early_exit);

  size_t round_idx = 0;
  int beat_idx = 0;
  const size_t total_rounds = round_beats.size();
  long guard = 0, guard_max = (long)total_rounds * 200 + 200000;
  bool done_sending = false;

  while (guard++ < guard_max) {
    bool have = round_idx < total_rounds;
    bool is_final_beat = have && (round_idx + 1 == total_rounds) && (beat_idx == 2);

    dut->s_axis_tvalid = have ? 1 : 0;
    dut->s_axis_tdata = have ? round_beats[round_idx][beat_idx] : 0;
    dut->s_axis_tlast = is_final_beat ? 1 : 0;
    dut->m_axis_tready = bp_seed ? (uint8_t)(bp.next() & 1) : 1;

    dut->eval();  // settle combinational tready/tvalid for this cycle

    bool beat = have && dut->s_axis_tready;
    if (dut->m_axis_tvalid && dut->m_axis_tready)
      out.push_back({(uint32_t)dut->m_axis_tdata, (bool)dut->m_axis_tlast});

    tick();

    if (beat) {
      if (beat_idx == 2) {
        beat_idx = 0;
        ++round_idx;
      } else {
        ++beat_idx;
      }
    }
    if (!have) done_sending = true;
    if (done_sending && !out.empty() && out.back().last) break;
  }
  dut->s_axis_tvalid = 0;
  return out;
}

// Gate-5 driver: stream all rounds full-speed, then hold m_axis_tready LOW for `stall_cycles` starting
// the cycle AFTER the final round's last beat is accepted (the start of the drain), then release and
// collect. `phase3_words` reports how many words were collected after the release — i.e., how many tail
// slots the shell had to hold across / emit after the stall. tready must stay high until the final round
// is accepted: the shell only accepts input while the result buffer is empty, so a pre-drain parked word
// under early tready-low would deadlock the handshake by design (see the gate-5 header comment).
static std::vector<OutWord> run_frame_drain_stall(const std::vector<std::array<uint32_t, 3>> &round_beats,
                                                  int early_exit, long stall_cycles,
                                                  size_t &phase3_words) {
  std::vector<OutWord> out;
  reset(early_exit);

  // Phase 1: full-speed stream of every round, exactly as run_frame's bp_seed==0 path.
  size_t round_idx = 0;
  int beat_idx = 0;
  const size_t total_rounds = round_beats.size();
  long guard = 0, guard_max = (long)total_rounds * 200 + 400000 + 2 * stall_cycles;
  while (round_idx < total_rounds && guard++ < guard_max) {
    bool is_final_beat = (round_idx + 1 == total_rounds) && (beat_idx == 2);
    dut->s_axis_tvalid = 1;
    dut->s_axis_tdata = round_beats[round_idx][beat_idx];
    dut->s_axis_tlast = is_final_beat ? 1 : 0;
    dut->m_axis_tready = 1;
    dut->eval();
    bool beat = dut->s_axis_tready;
    if (dut->m_axis_tvalid && dut->m_axis_tready)
      out.push_back({(uint32_t)dut->m_axis_tdata, (bool)dut->m_axis_tlast});
    tick();
    if (beat) {
      if (beat_idx == 2) {
        beat_idx = 0;
        ++round_idx;
      } else {
        ++beat_idx;
      }
    }
  }
  dut->s_axis_tvalid = 0;
  dut->s_axis_tlast = 0;

  // Phase 2: the drain — hold m_axis_tready low across (multiple) tail-slot retirements.
  dut->m_axis_tready = 0;
  for (long i = 0; i < stall_cycles; ++i) {
    dut->eval();
    tick();
  }

  // Phase 3: release and consume everything up to the tlast word.
  phase3_words = 0;
  dut->m_axis_tready = 1;
  while (guard++ < guard_max) {
    dut->eval();
    bool got = dut->m_axis_tvalid;
    bool last = got && dut->m_axis_tlast;
    if (got) {
      out.push_back({(uint32_t)dut->m_axis_tdata, (bool)dut->m_axis_tlast});
      ++phase3_words;
    }
    tick();
    if (last) break;
  }
  dut->m_axis_tready = 0;
  return out;
}

// Check one frame's captured words against a golden trial's fields (obs/vflag/commit_clean), the exact
// slot count, and tlast placement (only on the last word). Also asserts latency > 0 on every word.
// Returns the number of mismatches found; appends up to `budget` diagnostic lines to stderr.
static long check_trial(const std::vector<OutWord> &out, const GTrial &gt, int ee, int t, int &budget) {
  long mism = 0;
  if (out.size() != gt.slots.size()) {
    std::fprintf(stderr, "  mode %d trial %d: got %zu words, expected %zu\n", ee, t, out.size(),
                 gt.slots.size());
    return std::labs((long)out.size() - (long)gt.slots.size());
  }
  for (size_t k = 0; k < out.size(); ++k) {
    const GSlot &gs = gt.slots[k];
    long local = 0;
    uint32_t want_obs = 0;
    for (size_t o = 0; o < gs.obs.size(); ++o)
      if (gs.obs[o] == '1') want_obs |= (1u << o);
    if (obs_of(out[k].data) != want_obs) ++local;
    if (vflag_of(out[k].data) != gs.vflag) ++local;
    if (clean_of(out[k].data) != gs.clean) ++local;
    bool want_last = (k + 1 == out.size());
    if (out[k].last != want_last) ++local;
    if (lat_of(out[k].data) == 0) ++local;  // latency must be > 0 (not golden-compared otherwise)

    if (local && budget > 0) {
      --budget;
      std::fprintf(stderr,
                   "  mode %d trial %d word %zu: %ld mismatch(es) (obs dut=%u gold=%u, vflag dut=%d "
                   "gold=%d, clean dut=%d gold=%d, last dut=%d want=%d, lat=%u)\n",
                   ee, t, k, local, obs_of(out[k].data), want_obs, vflag_of(out[k].data), gs.vflag,
                   clean_of(out[k].data), gs.clean, (int)out[k].last, (int)want_last, lat_of(out[k].data));
    }
    mism += local;
  }
  return mism;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  const std::string vec_full = (argc > 1) ? argv[1] : "bp_stream_vectors.txt";
  const std::string vec_early = (argc > 2) ? argv[2] : "bp_stream_vectors_early.txt";

  Golden gold[2];  // [0] = best-kept (early_exit=0), [1] = first-valid (early_exit=1)
  if (!parse_golden(vec_full, gold[0]) || !parse_golden(vec_early, gold[1])) return 2;
  if (gold[0].T != gold[1].T || gold[0].SLICES != gold[1].SLICES || gold[0].DPR != gold[1].DPR ||
      gold[0].SLOTS != gold[1].SLOTS || gold[0].OBS != gold[1].OBS) {
    std::fprintf(stderr, "FAIL: golden headers disagree between %s and %s\n", vec_full.c_str(),
                 vec_early.c_str());
    return 2;
  }

  dut = new Dut;
  const char *gname[2] = {vec_full.c_str(), vec_early.c_str()};
  int fail = 0;

  for (int ee = 0; ee <= 1; ++ee) {
    const Golden &g = gold[ee];
    const int DPR = g.DPR, SLICES = g.SLICES, SLOTS = g.SLOTS, T = g.T;
    int printed = 0;

    // ---- Gate 1: zero stream ----
    {
      std::vector<std::array<uint32_t, 3>> zero_rounds(SLICES, std::array<uint32_t, 3>{0u, 0u, 0u});
      auto out = run_frame(zero_rounds, /*bp_seed=*/0, /*do_reset=*/true, ee);
      long mism = 0;
      if ((int)out.size() != SLOTS) {
        std::printf("FAIL(zero) mode %d: got %zu words, expected %d\n", ee, out.size(), SLOTS);
        ++mism;
      }
      for (size_t k = 0; k < out.size(); ++k) {
        if (obs_of(out[k].data) != 0) { std::printf("FAIL(zero) mode %d word %zu: obs != 0\n", ee, k); ++mism; }
        if (vflag_of(out[k].data) != 1) { std::printf("FAIL(zero) mode %d word %zu: vflag != 1\n", ee, k); ++mism; }
        if (clean_of(out[k].data) != 1) { std::printf("FAIL(zero) mode %d word %zu: commit_clean != 1\n", ee, k); ++mism; }
        bool want_last = (k + 1 == out.size());
        if (out[k].last != want_last) {
          std::printf("FAIL(zero) mode %d word %zu: tlast dut=%d want=%d\n", ee, k, (int)out[k].last,
                      (int)want_last);
          ++mism;
        }
      }
      if (mism == 0)
        std::printf("PASS: mode %d gate1(zero-stream) — %d words, obs=0/vflag=1/clean=1 all, tlast@%d only\n",
                    ee, SLOTS, SLOTS - 1);
      else
        std::printf("FAIL: mode %d gate1(zero-stream) — %ld mismatch(es)\n", ee, mism);
      fail += (mism != 0);
    }

    // ---- Gate 2: golden equality (all T trials, each with a fresh external reset) ----
    {
      long total = 0;
      for (int t = 0; t < T; ++t) {
        std::vector<std::array<uint32_t, 3>> rb(SLICES);
        for (int s = 0; s < SLICES; ++s) rb[s] = pack_round(g.trials[t].rounds[s], DPR);
        auto out = run_frame(rb, /*bp_seed=*/0, /*do_reset=*/true, ee);
        total += check_trial(out, g.trials[t], ee, t, printed);
      }
      if (total == 0)
        std::printf("PASS: mode %d gate2(golden-equality) — %d trials x %d slots bit-exact vs %s\n", ee, T,
                    SLOTS, gname[ee]);
      else
        std::printf("FAIL: mode %d gate2(golden-equality) — %ld field mismatch(es) vs %s\n", ee, total,
                    gname[ee]);
      fail += (total != 0);
    }

    // ---- Gate 3: back-pressure invariance (same T trials, full-speed vs random m_axis_tready) ----
    {
      int bp_match = 0;
      for (int t = 0; t < T; ++t) {
        std::vector<std::array<uint32_t, 3>> rb(SLICES);
        for (int s = 0; s < SLICES; ++s) rb[s] = pack_round(g.trials[t].rounds[s], DPR);
        auto full = run_frame(rb, /*bp_seed=*/0, /*do_reset=*/true, ee);
        auto bpp = run_frame(rb, /*bp_seed=*/0xC0FFEEull + (uint64_t)t + (ee ? 0x9000ull : 0ull),
                             /*do_reset=*/true, ee);
        bool same = full.size() == bpp.size();
        for (size_t i = 0; same && i < full.size(); ++i)
          same = (full[i].data == bpp[i].data) && (full[i].last == bpp[i].last);
        if (same)
          ++bp_match;
        else
          std::printf("FAIL: mode %d trial %d: back-pressure changed the output word sequence\n", ee, t);
      }
      if (bp_match == T)
        std::printf("PASS: mode %d gate3(back-pressure-invariance) — %d/%d trials byte-identical\n", ee, T,
                    T);
      else
        std::printf("FAIL: mode %d gate3(back-pressure-invariance) — %d/%d trials byte-identical\n", ee,
                    bp_match, T);
      fail += (bp_match != T);
    }

    // ---- Gate 4: frame independence (3 golden trials back-to-back, NO external reset between them) ----
    {
      const int FRAMES = 3;
      int ok = 0;
      for (int fnum = 0; fnum < FRAMES; ++fnum) {
        std::vector<std::array<uint32_t, 3>> rb(SLICES);
        for (int s = 0; s < SLICES; ++s) rb[s] = pack_round(g.trials[fnum].rounds[s], DPR);
        auto out = run_frame(rb, /*bp_seed=*/0, /*do_reset=*/(fnum == 0), ee);
        long m = check_trial(out, g.trials[fnum], ee, fnum, printed);
        if (m == 0)
          ++ok;
        else
          std::printf("FAIL: mode %d frame %d: %ld mismatch(es) (no external reset since frame 0)\n", ee,
                      fnum, m);
      }
      if (ok == FRAMES)
        std::printf("PASS: mode %d gate4(frame-independence) — %d/%d frames back-to-back, gates 1+2 hold\n",
                    ee, ok, FRAMES);
      else
        std::printf("FAIL: mode %d gate4(frame-independence) — %d/%d frames ok\n", ee, ok, FRAMES);
      fail += (ok != FRAMES);
    }

    // ---- Gate 5: adversarial drain stall (m_axis_tready low across the whole drain) ----
    {
      // > 2x the max observed core latency (16298 cycles, bpstream latency stats), so at least two tail
      // slots retire INTO the stall even in full-decode mode — the 1-deep-overwrite trigger condition.
      const long STALL_CYCLES = 40000;
      std::vector<std::array<uint32_t, 3>> rb(SLICES);
      for (int s = 0; s < SLICES; ++s) rb[s] = pack_round(g.trials[0].rounds[s], DPR);
      size_t phase3_words = 0;
      auto out = run_frame_drain_stall(rb, ee, STALL_CYCLES, phase3_words);
      long m = check_trial(out, g.trials[0], ee, 0, printed);
      // The stall must actually have covered the tail: every remaining word arrives after the release.
      if (phase3_words < 2) {
        std::printf("FAIL: mode %d gate5: only %zu word(s) after the stall release — stall did not span "
                    "the drain\n",
                    ee, phase3_words);
        ++m;
      }
      if (m == 0)
        std::printf("PASS: mode %d gate5(drain-stall) — tready low %ld cycles across the drain; all %d "
                    "words intact (%zu held across/after the stall)\n",
                    ee, STALL_CYCLES, SLOTS, phase3_words);
      else
        std::printf("FAIL: mode %d gate5(drain-stall) — %ld mismatch(es)\n", ee, m);
      fail += (m != 0);
    }
  }

  dut->final();
  delete dut;

  if (fail == 0) {
    std::printf(
        "PASS: bp_stream_win_core — all 5 gates x 2 early-exit modes green (zero-stream, golden-equality, "
        "back-pressure-invariance, frame-independence, drain-stall)\n");
    return 0;
  }
  std::printf("FAIL: %d gate(s) failed\n", fail);
  return 1;
}
