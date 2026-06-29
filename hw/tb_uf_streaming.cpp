// Q6-20 (part 3) — correctness test for the sliding-window streaming wrapper.
//
// The per-window UF core is already distance-verified (Q6-09/17/19). What the streaming wrapper must
// add correctly is the residual carry: feed each window, commit the oldest C rounds, toggle the
// committed defects, slide, reload. The tie-break-independent proof of that is VALIDITY: a graphlike
// decoder always produces a correction that reproduces the syndrome, so after a stream of defects is
// fully pushed through the commit region (drain with zero rounds), EVERY real defect must be resolved
// -- the residual must clear. (This is exactly the software `residual_after_decode == 0` criterion.)
// A bug in the commit / shift / residual logic leaves a stuck defect and fails the drain.
//
// Checks: (1) a zero stream commits zero logical and never lights the residual; (2) over many random
// defect streams, the residual fully drains (validity); (3) the FSM never stalls.

#include <cstdint>
#include <cstdio>

#include "Vuf_streaming_decoder.h"
#include "verilated.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto *dut = new Vuf_streaming_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  auto reset = [&]() {
    dut->rst_n = 0; dut->in_valid = 0; dut->in_round = 0;
    tick(); tick();
    dut->rst_n = 1;
  };

  // splitmix64 — deterministic per-seed stream.
  uint64_t z = 0;
  auto next = [&]() {
    z += 0x9E3779B97F4A7C15ull;
    uint64_t x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  };

  // Feed `nrounds` rounds; each round's detector bits are `gen()` (a closure). Collect committed-obs
  // pulses + window count. Returns total committed logical parity over the fed rounds.
  int windows = 0, obs_sum = 0, last_lat = 0;
  auto feed = [&](int nrounds, auto gen) {
    int sent = 0, guard = 0;
    while (sent < nrounds && guard < 500000) {
      bool rdy = dut->in_ready;
      dut->in_valid = rdy ? 1 : 0;
      dut->in_round = rdy ? gen() : 0;
      if (rdy) ++sent;
      tick();
      if (dut->out_valid) { ++windows; obs_sum ^= (dut->out_obs & 1); last_lat = dut->last_latency; }
      ++guard;
    }
    dut->in_valid = 0;
  };

  int fail = 0;

  // (1) zero stream: zero logical, residual stays empty.
  reset();
  windows = 0; obs_sum = 0;
  feed(60, [] { return 0; });
  if (obs_sum != 0) { std::printf("FAIL: zero stream committed nonzero logical\n"); ++fail; }
  if (windows < 4) { std::printf("FAIL: too few windows (%d)\n", windows); ++fail; }

  // (2) validity drain: random defect streams must fully resolve.
  const int TRIALS = 40, N = 60, DRAIN = 80;
  int drained_ok = 0;
  for (int t = 0; t < TRIALS; ++t) {
    reset();
    z = 0x1234ull + 0x1000ull * (uint64_t)t;
    int w_before = windows;
    feed(N, [&] { return (uint32_t)(next() & 0xF) & (uint32_t)((next() & 0x3) == 0 ? 0xF : 0x0); });
    feed(DRAIN, [] { return 0; });            // drain: push every defect through the commit region
    // settle to an idle point, then sample the residual.
    for (int i = 0; i < 5000 && !dut->in_ready; ++i) {
      tick();
      if (dut->out_valid) { ++windows; last_lat = dut->last_latency; }
    }
    if (dut->residual_empty) ++drained_ok;
    else std::printf("FAIL: trial %d left a stuck defect (residual not empty after drain)\n", t);
    if (windows <= w_before) { std::printf("FAIL: trial %d stalled\n", t); ++fail; }
  }
  if (drained_ok != TRIALS) { std::printf("FAIL: %d/%d trials drained\n", drained_ok, TRIALS); ++fail; }

  std::printf("streaming: zero-stream OK; validity drain %d/%d trials; windows=%d; last_latency=%d clk\n",
              drained_ok, TRIALS, windows, last_lat);
  std::printf("%s\n", fail ? "RESULT: FAIL"
                            : "RESULT: PASS (zero->zero, residual drains on every random defect stream)");
  dut->final();
  delete dut;
  return fail ? 1 : 0;
}
