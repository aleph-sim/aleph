// Q6-20 (on silicon) — correctness test for the AXI4-Stream front-end of the streaming decoder.
//
// The bare `uf_streaming_decoder` is already validity-verified (tb_uf_streaming, #399): a stream of
// defects pushed through the commit region and drained leaves an empty residual. What THIS test gates
// is that the AXI wrapper (`uf_stream_win_core`) preserves that behaviour under real stream framing:
// one round per MM2S beat, one result per committed window on S2MM, correct tlast, and — crucially —
// that S2MM back-pressure never drops or corrupts a window (the 1-deep result slot + input gating).
//
// Checks:
//   (1) a zero round-stream commits zero logical, every window reports residual-empty, exactly the
//       expected number of windows arrive, and the last carries tlast;
//   (2) VALIDITY: a random defect stream + zero-drain fully clears the residual (last window
//       residual-empty), on the exact window boundary;
//   (3) BACK-PRESSURE INVARIANCE: the same defect stream decoded with random S2MM back-pressure yields
//       a bit-identical output-word sequence to the full-speed run (no drops, no reordering).
//
// Window params (UF_ACTIVE/UF_DPR/UF_LOAD_LO) are passed as -D defines from the generated window graph
// header by the Makefile, so warm-up = UF_ACTIVE/UF_DPR rounds and each slide reloads
// (UF_ACTIVE-UF_LOAD_LO)/UF_DPR = C rounds — matching the FSM's byte-cursor fill exactly.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "Vuf_stream_win_core.h"
#include "verilated.h"

#ifndef UF_ACTIVE
#define UF_ACTIVE 36
#endif
#ifndef UF_DPR
#define UF_DPR 4
#endif
#ifndef UF_LOAD_LO
#define UF_LOAD_LO 24
#endif

static const int WARMUP_ROUNDS = UF_ACTIVE / UF_DPR;             // rounds to fill the warm-up window (W)
static const int RELOAD_ROUNDS = (UF_ACTIVE - UF_LOAD_LO) / UF_DPR; // rounds reloaded per slide (C)
static const uint32_t ROUND_MASK = (UF_DPR >= 32) ? 0xFFFFFFFFu : ((1u << UF_DPR) - 1u);

struct OutWord {
  uint32_t data;
  bool last;
};

// splitmix64 — deterministic per-seed stream.
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

static Vuf_stream_win_core *dut = nullptr;
static void tick() {
  dut->aclk = 0; dut->eval();
  dut->aclk = 1; dut->eval();
}
static void reset() {
  dut->aresetn = 0; dut->s_axis_tvalid = 0; dut->s_axis_tdata = 0; dut->s_axis_tlast = 0;
  dut->m_axis_tready = 0;
  tick(); tick();
  dut->aresetn = 1;
}

// Drive `rounds` over AXI (one round/beat, tlast on the last beat) and collect the per-window result
// words. `bp_seed`==0 -> S2MM always ready (full speed); otherwise pseudo-random back-pressure.
// `do_reset`==false skips the initial reset, so a second frame runs back-to-back on a DUT that was NOT
// re-armed externally — the on-board case (one overlay, many DMA transfers). The wrapper's per-frame
// reset must make that frame independent (fresh warm-up), or its window count/drain would be wrong.
static std::vector<OutWord> run_stream(const std::vector<uint32_t> &rounds, uint64_t bp_seed,
                                       bool do_reset = true) {
  std::vector<OutWord> out;
  Rng bp(bp_seed ? bp_seed : 1);
  if (do_reset) reset();
  size_t sent = 0;
  int guard = 0, guard_max = (int)rounds.size() * 200 + 200000;
  bool done_sending = false;
  while (guard++ < guard_max) {
    bool have = sent < rounds.size();
    bool last = have && (sent + 1 == rounds.size());
    dut->s_axis_tvalid = have ? 1 : 0;
    dut->s_axis_tdata  = have ? (rounds[sent] & ROUND_MASK) : 0;
    dut->s_axis_tlast  = last ? 1 : 0;
    dut->m_axis_tready = bp_seed ? (uint8_t)(bp.next() & 1) : 1;

    dut->eval();  // settle combinational tready / tvalid for this cycle

    bool beat = have && dut->s_axis_tready;
    if (dut->m_axis_tvalid && dut->m_axis_tready)
      out.push_back({(uint32_t)dut->m_axis_tdata, (bool)dut->m_axis_tlast});

    tick();
    if (beat) ++sent;
    if (!have) done_sending = true;
    // stop once everything is sent and the final (tlast) window has drained
    if (done_sending && !out.empty() && out.back().last) break;
  }
  dut->s_axis_tvalid = 0;
  return out;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vuf_stream_win_core;

  int fail = 0;
  const int K = 8;  // windows after warm-up
  const int R = WARMUP_ROUNDS + K * RELOAD_ROUNDS;
  const int EXPECT_WIN = 1 + K;

  auto obs_of   = [](uint32_t w) { return (w >> 31) & 1u; };
  auto rese_of  = [](uint32_t w) { return (w >> 30) & 1u; };

  // (1) zero stream -> zero logical, all windows residual-empty, exact count, tlast on last.
  {
    std::vector<uint32_t> z(R, 0u);
    auto out = run_stream(z, 0);
    if ((int)out.size() != EXPECT_WIN) {
      std::printf("FAIL(zero): got %zu windows, expected %d\n", out.size(), EXPECT_WIN); ++fail;
    }
    uint32_t obs_sum = 0; int nonempty = 0;
    for (auto &o : out) { obs_sum ^= obs_of(o.data); if (!rese_of(o.data)) ++nonempty; }
    if (obs_sum != 0) { std::printf("FAIL(zero): committed nonzero logical\n"); ++fail; }
    if (nonempty != 0) { std::printf("FAIL(zero): a window was not residual-empty\n"); ++fail; }
    if (!out.empty() && !out.back().last) { std::printf("FAIL(zero): last window lacks tlast\n"); ++fail; }
    for (size_t i = 0; i + 1 < out.size(); ++i)
      if (out[i].last) { std::printf("FAIL(zero): premature tlast at window %zu\n", i); ++fail; }
  }

  // (2)+(3) random defect stream, drained, with and without back-pressure.
  const int TRIALS = 40;
  int drained_ok = 0, bp_match = 0;
  for (int t = 0; t < TRIALS; ++t) {
    Rng g(0x1234ull + 0x1000ull * (uint64_t)t);
    // Defects only in the first half; the rest is zero-drain so every defect is pushed through commit
    // and the boundary lands on a window (R = WARMUP + K*RELOAD).
    std::vector<uint32_t> rounds(R, 0u);
    for (int i = 0; i < R / 2; ++i)
      rounds[i] = (uint32_t)((g.next() & ROUND_MASK) & ((g.next() & 0x3) == 0 ? ROUND_MASK : 0u));

    auto full = run_stream(rounds, 0);
    auto bpp  = run_stream(rounds, 0xC0FFEEull + t);

    if ((int)full.size() == EXPECT_WIN && !full.empty() && rese_of(full.back().data) &&
        full.back().last)
      ++drained_ok;
    else
      std::printf("FAIL: trial %d not drained/framed (win=%zu)\n", t, full.size());

    bool same = full.size() == bpp.size();
    for (size_t i = 0; same && i < full.size(); ++i)
      same = (full[i].data == bpp[i].data) && (full[i].last == bpp[i].last);
    if (same) ++bp_match;
    else std::printf("FAIL: trial %d back-pressure changed the output sequence\n", t);
  }
  if (drained_ok != TRIALS) { std::printf("FAIL: %d/%d drained+framed\n", drained_ok, TRIALS); ++fail; }
  if (bp_match != TRIALS)  { std::printf("FAIL: %d/%d back-pressure-invariant\n", bp_match, TRIALS); ++fail; }

  // (4) FRAME INDEPENDENCE: many DMA frames back-to-back with NO external reset between them (the
  // on-board case: one overlay, repeated transfers). Each frame must produce exactly EXPECT_WIN windows
  // and drain — proving the wrapper re-arms the decoder to warm-up at every tlast. Without the per-frame
  // reset a follow-on frame resumes mid-stream and its window count/drain would be wrong.
  int frame_ok = 0;
  const int FRAMES = 6;
  reset();  // fresh once; subsequent frames rely on the wrapper's own re-arm
  for (int fnum = 0; fnum < FRAMES; ++fnum) {
    Rng g(0xABCDull + 0x777ull * (uint64_t)fnum);
    std::vector<uint32_t> rounds(R, 0u);
    for (int i = 0; i < R / 2; ++i)
      rounds[i] = (uint32_t)((g.next() & ROUND_MASK) & ((g.next() & 0x3) == 0 ? ROUND_MASK : 0u));
    auto out = run_stream(rounds, 0, /*do_reset=*/false);
    if ((int)out.size() == EXPECT_WIN && !out.empty() && rese_of(out.back().data) && out.back().last)
      ++frame_ok;
    else
      std::printf("FAIL: frame %d wrong count/undrained (win=%zu)\n", fnum, out.size());
  }
  if (frame_ok != FRAMES) { std::printf("FAIL: %d/%d frames independent\n", frame_ok, FRAMES); ++fail; }

  std::printf("stream-axi: W=%d C=%d R=%d win/stream=%d; zero-stream OK; drain %d/%d; bp-invariant %d/%d; "
              "frame-indep %d/%d\n",
              WARMUP_ROUNDS, RELOAD_ROUNDS, R, EXPECT_WIN, drained_ok, TRIALS, bp_match, TRIALS,
              frame_ok, FRAMES);
  std::printf("%s\n", fail ? "RESULT: FAIL"
                           : "RESULT: PASS (AXI framing preserves validity; tlast + back-pressure correct)");
  dut->final();
  delete dut;
  return fail ? 1 : 0;
}
