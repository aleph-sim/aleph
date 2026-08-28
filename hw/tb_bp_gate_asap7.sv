// Q7-02 Task A4 — event-driven gate-level testbench for the ASAP7 routed netlist of `bp_relay_banked`.
//
// Same contract as tb_bp_banked.cpp, but written for an event-driven simulator (Icarus) so the VENDOR
// sequential models (UDP primitives, which Verilator cannot compile) drive the netlist. Its job is to
// arbitrate Verilator + behavioural DHLx1/DFFHQN replacements against the foundry's own cell models.
// Ports are the packed vectors the netlist exposes.
//
// Input is the pre-flattened vector file written by `hw/sw/gate_vectors.py` from bp_circ_vectors.txt:
//   line 1: T N C OBS ; then per test four lines: s h o v as %b strings (bit 0 = rightmost char).
// Plain Verilog-2001 I/O on purpose: Icarus' SV string support crashed its backend on the first draft.
//   +vec=<file>  +ntests=<k> (default: all)
`timescale 1ns/1ps
module tb_bp_gate_asap7;
  reg clk = 0, rst_n = 0, in_valid = 0, early_exit = 0;
  reg [143:0] syndrome_in = 0;
  wire busy, out_valid, valid_flag;
  wire [863:0] corr_out;
  wire [31:0] latency_cycles;
  wire [11:0] obs_flip;

  bp_relay_banked dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .early_exit(early_exit),
                       .syndrome_in(syndrome_in), .busy(busy), .out_valid(out_valid),
                       .corr_out(corr_out), .obs_flip(obs_flip), .valid_flag(valid_flag),
                       .latency_cycles(latency_cycles));

  always #0.5 clk = ~clk;  // 1 ns period, matches the SDC target

  integer fd, T, N, C, OBS, ntests, mism, t, i, local_m, r;
  reg [1023:0] vec;
  reg [143:0] s_vec;
  reg [863:0] h_vec;
  reg [11:0] o_vec;
  reg v_want;

  initial begin
    if (!$value$plusargs("vec=%s", vec)) vec = "bp_circ_vectors.gate.txt";
    if (!$value$plusargs("ntests=%d", ntests)) ntests = -1;
    fd = $fopen(vec, "r");
    if (fd == 0) begin $display("FAIL: open vector file"); $finish; end
    r = $fscanf(fd, "%d %d %d %d\n", T, N, C, OBS);
    if (r != 4 || N != 864 || C != 144 || OBS != 12) begin $display("FAIL: bad header r=%0d", r); $finish; end
    if (ntests < 0 || ntests > T) ntests = T;
    $display("header T=%0d N=%0d C=%0d OBS=%0d, running %0d", T, N, C, OBS, ntests);
    repeat (4) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);
    mism = 0;
    for (t = 0; t < ntests; t = t + 1) begin
      r = $fscanf(fd, "%b\n", s_vec); r = r + $fscanf(fd, "%b\n", h_vec);
      r = r + $fscanf(fd, "%b\n", o_vec); r = r + $fscanf(fd, "%b\n", v_want);
      if (r != 4) begin $display("FAIL: truncated vectors at test %0d", t); $finish; end
      syndrome_in <= s_vec;
      in_valid <= 1;
      @(posedge clk);
      in_valid <= 0;
      i = 0;
      while (!out_valid && i < 8000000) begin @(posedge clk); i = i + 1; end
      #0.1;
      if (!out_valid) begin $display("FAIL: test %0d never asserted out_valid", t); $finish; end
      local_m = 0;
      for (i = 0; i < N; i = i + 1) if (corr_out[i] !== h_vec[i]) begin
        if (local_m < 16) $display("    corr_out[%0d]: got %b want %b", i, corr_out[i], h_vec[i]);
        local_m = local_m + 1;
      end
      for (i = 0; i < OBS; i = i + 1) if (obs_flip[i] !== o_vec[i]) begin
        $display("    obs_flip[%0d]: got %b want %b", i, obs_flip[i], o_vec[i]); local_m = local_m + 1;
      end
      if (valid_flag !== v_want) begin
        $display("    valid_flag: got %b want %b", valid_flag, v_want); local_m = local_m + 1;
      end
      $display("test %0d: latency %0d cycles, %0d field mismatches", t, latency_cycles, local_m);
      if (local_m) mism = mism + 1;
    end
    if (mism == 0) $display("PASS: %0d gate-level decodes bit-identical to the golden", ntests);
    else $display("FAIL: %0d/%0d decodes mismatched", mism, ntests);
    $finish;
  end
endmodule
