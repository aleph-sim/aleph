// Q6-06 — SystemVerilog testbench for Vivado xsim gate-level sign-off.
//
// The Verilator TB (`tb_uf_surface.cpp`) only exercises behavioral RTL. This SV TB runs in xsim and
// is reused across THREE elaborations of the SAME `uf_surface_decoder` top:
//   1. behavioral RTL (sanity that the TB itself is correct),
//   2. post-synthesis / post-route FUNCTIONAL netlist (catches synth/sim mismatch, latches),
//   3. post-route TIMING netlist with SDF (catches X-prop / setup issues at the closed clock).
//
// Self-checking: every syndrome must (a) bit-match the frozen Q6-02 golden table, (b) have a
// correction that reproduces the syndrome, and (c) drive no X on the outputs after reset.

`timescale 1ns / 1ps
`include "uf_surface_graph.svh"

module tb_uf_surface_xsim;
  // Clock half-period in ns. Behavioral/functional sims have no cell delays → keep 5 (100 MHz).
  // For the SDF timing sim, override above the closed Fmax (e.g. -generic_top "HALF_NS=10" = 50 MHz)
  // so the run reflects real cell delays without spurious setup violations from over-clocking.
  parameter int HALF_NS = 5;

  localparam int DETS = UF_N - 1;     // 8 detector nodes for d=3
  localparam int NSYN = 1 << DETS;    // 256 syndromes

  logic              clk = 1'b0;
  logic              rst_n = 1'b0;
  logic              in_valid = 1'b0;
  logic [UF_N-2:0]   syndrome = '0;
  logic              busy;
  logic              out_valid;
  logic [UF_M-1:0]   correction;
  logic              obs_flip;
  logic [15:0]       latency_cycles;

  // {obs_flip, correction} per syndrome, snapshotted from the Q6-02 RTL.
  logic [UF_M:0]     golden [0:NSYN-1];

  uf_surface_decoder dut (
    .clk, .rst_n, .in_valid, .syndrome, .busy, .out_valid,
    .correction, .obs_flip, .latency_cycles
  );

  always #(HALF_NS) clk = ~clk;

  int fails = 0, gold_fail = 0, valid_fail = 0, x_fail = 0;

  // validity: the correction must reproduce the syndrome on every detector node.
  function automatic logic validity_ok(input logic [UF_M-1:0] corr, input int s);
    int synr, par;
    synr = 0;
    for (int d = 0; d < DETS; d++) begin
      par = 0;
      for (int e = 0; e < UF_M; e++)
        if (UF_EA[e] == d || UF_EB[e] == d) par ^= corr[e];
      synr |= par << d;
    end
    return (synr == s);
  endfunction

  task automatic do_decode(input int s);
    int g;
    @(posedge clk); in_valid <= 1'b1; syndrome <= s[UF_N-2:0];
    @(posedge clk); in_valid <= 1'b0;
    g = 0;
    while (out_valid !== 1'b1 && g < 4096) begin @(posedge clk); g++; end
  endtask

  initial begin
    $readmemh("uf_surface_golden.mem", golden);
    rst_n = 1'b0; in_valid = 1'b0; syndrome = '0;
    // Hold reset past the glbl GSR window (~100 ns in gate-level sim) before the first decode,
    // else the first in_valid is swallowed while the FFs are still globally reset.
    repeat (20) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    for (int s = 0; s < NSYN; s++) begin
      do_decode(s);
      if (out_valid !== 1'b1) begin
        $display("FAIL s=%0d: out_valid never asserted", s); fails++; continue;
      end
      if ($isunknown({obs_flip, correction})) begin
        $display("X FAIL s=%0d: outputs contain X (obs=%b corr=%h)", s, obs_flip, correction);
        x_fail++;
      end
      if ({obs_flip, correction} !== golden[s]) begin
        $display("GOLDEN FAIL s=%0d: rtl=%h golden=%h", s, {obs_flip, correction}, golden[s]);
        gold_fail++;
      end
      if (!validity_ok(correction, s)) begin
        $display("VALIDITY FAIL s=%0d: corr=%h", s, correction); valid_fail++;
      end
    end

    $display("xsim: syndromes=%0d out_valid_fail=%0d golden_fail=%0d validity_fail=%0d x_fail=%0d",
             NSYN, fails, gold_fail, valid_fail, x_fail);
    if (fails || gold_fail || valid_fail || x_fail) begin
      $display("RESULT: FAIL");
      $fatal(1, "gate-level sign-off failed");
    end else begin
      $display("RESULT: PASS (all %0d syndromes bit-match golden + valid, no X)", NSYN);
    end
    $finish;
  end
endmodule
