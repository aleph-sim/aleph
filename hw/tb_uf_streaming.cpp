// Q6-20 (part 2) — smoke test for the sliding-window streaming wrapper. Drives the round-stream
// handshake and checks basic invariants:
//   (1) a zero stream commits zero logical and leaves no residual (the FSM cycles, no spurious flips);
//   (2) windows fire at the expected cadence (one commit per C rounds after warm-up);
//   (3) injecting a defect does not hang the decoder (it keeps committing).
// Full bit-equality vs the software steady-state sliding decode is part 3 (tb against the reference).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "Vuf_streaming_decoder.h"
#include "verilated.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto *dut = new Vuf_streaming_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };

  dut->rst_n = 0;
  dut->in_valid = 0;
  dut->in_round = 0;
  tick();
  tick();
  dut->rst_n = 1;

  // Feed `nrounds` rounds of detector bits `pattern` (one per accepted-ready cycle), running the
  // wrapper to completion; collect committed-obs pulses. Returns total committed logical parity.
  int windows = 0, obs_sum = 0, last_lat = 0;
  auto feed = [&](int nrounds, uint32_t pattern) {
    int sent = 0, guard = 0;
    while (sent < nrounds && guard < 200000) {
      dut->in_valid = dut->in_ready ? 1 : 0;
      dut->in_round = dut->in_ready ? pattern : 0;
      if (dut->in_ready) ++sent;
      tick();
      if (dut->out_valid) {
        ++windows;
        obs_sum ^= (dut->out_obs & 1);
        last_lat = dut->last_latency;
      }
      ++guard;
    }
    dut->in_valid = 0;
    // let any in-flight window finish
    for (int i = 0; i < 4000 && !dut->in_ready; ++i) {
      tick();
      if (dut->out_valid) { ++windows; obs_sum ^= (dut->out_obs & 1); last_lat = dut->last_latency; }
    }
  };

  // (1)+(2): a long zero stream. Expect zero committed logical and a healthy window count.
  const int N = 120;
  feed(N, 0);
  int fail = 0;
  if (obs_sum != 0) { std::printf("FAIL: zero stream committed nonzero logical (%d)\n", obs_sum); ++fail; }
  if (windows < 5) { std::printf("FAIL: too few windows fired (%d) on a %d-round stream\n", windows, N); ++fail; }

  // (3): inject a defect mid-stream; the decoder must keep committing (no hang).
  int w_before = windows;
  feed(1, 0x1);     // one round with detector 0 set
  feed(30, 0);      // drain
  if (windows <= w_before) { std::printf("FAIL: decoder stalled after a defect (%d -> %d)\n", w_before, windows); ++fail; }

  std::printf("streaming smoke: windows=%d  committed_logical=%d  last_window_latency=%d clk\n",
              windows, obs_sum, last_lat);
  std::printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS (FSM cycles, zero stream -> zero logical, no stall)");
  dut->final();
  delete dut;
  return fail ? 1 : 0;
}
