// Q6-04 helper — dump the CURRENT combinational uf_surface_decoder output for all syndromes.
// Produces the frozen golden table {obs_flip,correction} that the sequential FSM rewrite must
// reproduce bit-for-bit. Build against the pre-refactor RTL, run once, commit the .mem, then refactor.
//   verilator --cc --exe --build -Wall --top-module uf_surface_decoder \
//       uf_surface_decoder.sv tb_dump_golden.cpp -o dump_golden && ./obj_dir/dump_golden > uf_surface_golden.mem

#include <cstdint>
#include <cstdio>
#include "Vuf_surface_decoder.h"
#include "verilated.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto *dut = new Vuf_surface_decoder;
  auto tick = [&]() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); };
  dut->rst_n = 0; dut->in_valid = 0; dut->syndrome = 0;
  tick(); tick();
  dut->rst_n = 1;

  const int dets = 8;                 // d=3: N-1 = 8 detector nodes
  const int entries = 1 << dets;
  std::printf("// GENERATED golden table — current combinational uf_surface_decoder, all syndromes.\n");
  std::printf("// one row per syndrome s=0..%d : <obs_flip><correction[M-1:0]> in hex.\n", entries - 1);
  for (int s = 0; s < entries; ++s) {
    dut->in_valid = 1; dut->syndrome = s; tick();
    dut->in_valid = 0; dut->eval();
    // {obs_flip, correction} packed: obs in bit M, correction in bits [M-1:0].
    const uint32_t corr = dut->correction;
    const uint32_t obs  = dut->obs_flip & 1u;
    std::printf("%X\n", (obs << 18) | (corr & 0x3FFFF));   // M=18
  }
  dut->final();
  delete dut;
  return 0;
}
