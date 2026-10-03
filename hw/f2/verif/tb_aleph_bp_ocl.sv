// P2 appliance v2-large — xsim testbench for aleph_bp_ocl (the F2 OCL front end + decoder core).
//
// Drives the module exactly as the host will: OCL AXI4-Lite (PUSH, POP peek + POP_ACK) and the PCIS
// AXI4 batch path (experiment slots in, result ring out), with the real MMCM (unisim) and the real XPM
// FIFOs/CDC. Gates:
//   * MAGIC, GEOM, VERSION, SLOTS read back; GEOM must match the header the core was built with
//   * the measured clk_core / clk_main ratio matches the MMCM divide
//   * OCL: a partial-strobe PUSH pushes nothing and sets err_strb; a POP_ACK with the wrong seq pops
//     nothing; reading POP twice returns the same word
//   * OCL: every one of the T golden vectors decodes to the golden {obs, valid_flag}
//   * PCIS: the same T vectors, written as slot stores (some split in two), read back from the ring
//     bit-exact; counters agree (PCIS_ACC, RESULTS, PUSH_WORDS)
//   * after a soft reset back to OCL mode the decoder decodes again (first 5 vectors)
// Prints "PASS: ..." on success; any failure ends with $fatal.
//
// Usage: xsim with +VEC=<path to bp_circ_vectors.txt>; see hw/f2/verif/run_xsim.sh.

`timescale 1ps / 1ps
`include "bb_gross_tanner.svh"

module tb_aleph_bp_ocl;
  localparam int NS = (BP_C + 31) / 32;

  logic clk_main = 0, clk_ref = 0, rst_main_n = 0;
  always #2000 clk_main = ~clk_main;   // 250 MHz
  always #5000 clk_ref  = ~clk_ref;    // 100 MHz

  logic [31:0] awaddr, wdata, araddr, rdata;
  logic [3:0]  wstrb = 4'hF;
  logic [63:0]  p_awaddr, p_araddr, p_wstrb;
  logic [511:0] p_wdata, p_rdata;
  logic         p_awvalid, p_awready, p_wvalid, p_wready, p_bvalid, p_bready;
  logic         p_arvalid, p_arready, p_rvalid, p_rready;
  logic        awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [1:0]  bresp, rresp;

  aleph_bp_ocl #(.CORE_DIV_F(`TB_DIV_F), .CORE_KHZ(`TB_KHZ)) dut (
      .clk_main(clk_main), .rst_main_n(rst_main_n), .clk_ref(clk_ref),
      .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready),
      .s_wdata(wdata), .s_wstrb(wstrb), .s_wvalid(wvalid), .s_wready(wready),
      .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready),
      .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
      .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready),
      .p_awaddr(p_awaddr), .p_awid(16'h0), .p_awsize(3'd6), .p_awvalid(p_awvalid), .p_awready(p_awready),
      .p_wdata(p_wdata), .p_wstrb(p_wstrb), .p_wlast(1'b1), .p_wvalid(p_wvalid), .p_wready(p_wready),
      .p_bid(), .p_bresp(), .p_bvalid(p_bvalid), .p_bready(p_bready),
      .p_araddr(p_araddr), .p_arid(16'h0), .p_arlen(8'd0), .p_arsize(3'd6), .p_arvalid(p_arvalid),
      .p_arready(p_arready), .p_rid(), .p_rdata(p_rdata), .p_rresp(), .p_rlast(), .p_rvalid(p_rvalid),
      .p_rready(p_rready)
  );

  // ---- PCIS AXI4 master: single-beat 64-byte writes and reads (the bursts, narrow and split cases are
  // covered by tb_aleph_pcis_ring)

  task automatic pcis_write(input logic [63:0] a, input logic [511:0] d, input logic [63:0] st);
    @(posedge clk_main);
    p_awaddr <= a; p_awvalid <= 1; p_wdata <= d; p_wstrb <= st; p_wvalid <= 1;
    do @(posedge clk_main); while (!p_awready);
    p_awvalid <= 0;
    while (!p_wready) @(posedge clk_main);
    @(posedge clk_main);
    p_wvalid <= 0; p_bready <= 1;
    while (!p_bvalid) @(posedge clk_main);
    @(posedge clk_main);
    p_bready <= 0;
  endtask

  task automatic pcis_read(input logic [63:0] a, output logic [511:0] d);
    @(posedge clk_main);
    p_araddr <= a; p_arvalid <= 1;
    do @(posedge clk_main); while (!p_arready);
    p_arvalid <= 0; p_rready <= 1;
    do @(posedge clk_main); while (!p_rvalid);
    d = p_rdata;
    p_rready <= 0;
  endtask

  // ---- AXI4-Lite master. Address and data are presented in different cycles on purpose: the shell is
  // allowed to do that, and the slave must wait for both.
  task automatic axil_write(input logic [31:0] a, input logic [31:0] d);
    @(posedge clk_main);
    awaddr <= a; awvalid <= 1;
    @(posedge clk_main);
    wdata <= d; wvalid <= 1;
    do @(posedge clk_main); while (!(awready && wready));
    awvalid <= 0; wvalid <= 0; bready <= 1;
    do @(posedge clk_main); while (!bvalid);
    bready <= 0;
  endtask

  task automatic axil_read(input logic [31:0] a, output logic [31:0] d);
    @(posedge clk_main);
    araddr <= a; arvalid <= 1;
    do @(posedge clk_main); while (!arready);
    arvalid <= 0; rready <= 1;
    do @(posedge clk_main); while (!rvalid);
    d = rdata;
    rready <= 0;
  endtask

  // ---- golden vectors
  string       synd[$];
  int          gobs[$], gval[$];

  task automatic load_vectors(input string path);
    int fd, t, n, c, o;
    string line, body;
    fd = $fopen(path, "r");
    if (fd == 0) $fatal(1, "cannot open %s", path);
    t = -1;
    while ($fgets(line, fd)) begin
      if (line.len() == 0 || line[0] == "#" || line[0] == "\n") continue;
      if (t < 0) begin
        void'($sscanf(line, "%d %d %d %d", t, n, c, o));
        if (c != BP_C) $fatal(1, "vectors have C=%0d, core has BP_C=%0d", c, BP_C);
        continue;
      end
      body = line.substr(2, line.len() - 1);
      // strip trailing newline
      while (body.len() > 0 && (body[body.len()-1] == "\n" || body[body.len()-1] == " "))
        body = body.substr(0, body.len() - 2);
      case (line[0])
        "s": synd.push_back(body);
        "o": begin
          int ob = 0;
          for (int i = 0; i < body.len(); i++) if (body[i] == "1") ob |= (1 << i);
          gobs.push_back(ob);
        end
        "v": gval.push_back(body[0] == "1");
        default: ;
      endcase
    end
    $fclose(fd);
    if (synd.size() != t || gobs.size() != t || gval.size() != t)
      $fatal(1, "parsed %0d/%0d/%0d tests, header says %0d", synd.size(), gobs.size(), gval.size(), t);
  endtask

  task automatic push_syndrome(input int k);
    logic [NS*32-1:0] w = '0;
    for (int c = 0; c < BP_C; c++) if (synd[k][c] == "1") w[c] = 1'b1;
    for (int i = 0; i < NS; i++) axil_write(32'h10, w[i*32 +: 32]);
  endtask

  task automatic wait_ready();
    logic [31:0] st;
    int guard = 0;
    do begin
      axil_read(32'h08, st);
      if (++guard > 20000) $fatal(1, "never became ready, STATUS=%h", st);
    end while (!st[4] || st[5] || st[6] || st[7]);   // locked, core out of reset, FIFOs out of reset
  endtask

  task automatic run_and_check(input int first, input int count, output int worst);
    logic [31:0] r;
    int mism = 0, got_obs, got_v, lat, guard;
    worst = 0;
    for (int k = first; k < first + count; k++) push_syndrome(k);
    for (int k = first; k < first + count; k++) begin
      guard = 0;
      do begin
        axil_read(32'h14, r);
        if (++guard > 200000) $fatal(1, "no result for test %0d", k);
      end while (r == 32'hFFFF_FFFF);
      if (r[17:16] != 2'(k)) $fatal(1, "test %0d: POP seq %0d, want %0d", k, r[17:16], k % 4);
      axil_write(32'h24, {30'd0, r[17:16]});
      got_obs = (r >> 20) & ((1 << BP_OBS) - 1);
      got_v   = r[19];
      lat     = r[15:0];
      if (lat > worst) worst = lat;
      if ((k + 1) % 10 == 0) $display("[tb] %0d decodes done", k + 1);  // progress: xsim is slow here
      if (got_obs != (gobs[k] & ((1 << BP_OBS) - 1)) || got_v != gval[k]) begin
        $display("MISMATCH test %0d: got obs=%03h v=%0d, want obs=%03h v=%0d", k, got_obs, got_v,
                 gobs[k], gval[k]);
        mism++;
      end
    end
    if (mism != 0) $fatal(1, "%0d/%0d mismatches", mism, count);
  endtask

  // PCIS mode: every vector as one slot store (every third one split into two half stores), then poll
  // the ring rows until each slot shows its tag
  task automatic run_pcis(output int worst);
    logic [511:0] line, row;
    logic [31:0]  r;
    int           mism = 0, s, guard;
    worst = 0;
    for (int k = 0; k < gobs.size(); k++) begin
      line = '0;
      for (int c = 0; c < BP_C; c++) if (synd[k][c] == "1") line[c] = 1'b1;
      line[255:224] = 32'h7A00_0000 + k;
      s = k % 128;
      if (k % 3 == 2) begin
        pcis_write(64'(s) * 64, line, 64'h0000_0000_0000_FFFF);
        pcis_write(64'(s) * 64, line, 64'h0000_0000_FFFF_0000);
      end else begin
        pcis_write(64'(s) * 64, line, {64{1'b1}});
      end
    end
    for (int k = 0; k < gobs.size(); k++) begin
      s = k % 128;
      guard = 0;
      do begin
        pcis_read(64'h1_0000 + 64 * (s / 8), row);
        if (++guard > 200000) $fatal(1, "PCIS: no result for test %0d, entry %h", k, row[64*(s%8) +: 64]);
      end while (row[64*(s%8) + 32 +: 32] != 32'h7A00_0000 + k);
      r = row[64*(s%8) +: 32];
      if (r[15:0] > worst) worst = r[15:0];
      if (((r >> 20) & ((1 << BP_OBS) - 1)) != (gobs[k] & ((1 << BP_OBS) - 1)) || r[19] != gval[k]) begin
        $display("PCIS MISMATCH test %0d: got %h", k, r);
        mism++;
      end
    end
    if (mism != 0) $fatal(1, "PCIS: %0d/%0d mismatches", mism, gobs.size());
  endtask

  initial begin
    string vec;
    logic [31:0] r, c0, m0, c1, m1;
    int worst, worst2;
    real ratio, mhz;
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
    p_awvalid = 0; p_wvalid = 0; p_bready = 0; p_arvalid = 0; p_rready = 0;
    awaddr = 0; wdata = 0; araddr = 0;
    if (!$value$plusargs("VEC=%s", vec)) vec = "bp_circ_vectors.txt";
    load_vectors(vec);

    repeat (20) @(posedge clk_main);
    rst_main_n = 1;
    wait_ready();

    axil_read(32'h00, r);
    if (r != 32'hA1E9_B0F2) $fatal(1, "MAGIC=%h", r);
    axil_read(32'h04, r);
    if (r[31:16] != BP_BANK_W || r[15:0] != BP_BANK_V) $fatal(1, "GEOM=%h", r);
    $display("[tb] MAGIC ok, GEOM %0d/%0d", r[31:16], r[15:0]);
    axil_read(32'h14, r);
    if (r != 32'hFFFF_FFFF) $fatal(1, "POP on empty FIFO returned %h", r);
    axil_read(32'h38, r);
    if (r != 2) $fatal(1, "VERSION=%h", r);
    axil_read(32'h3C, r);
    if (r != 128) $fatal(1, "SLOTS=%h", r);

    // a PUSH without all four strobes must push nothing and raise err_strb; clear_err clears it
    wstrb = 4'h3;
    axil_write(32'h10, 32'h1234_5678);
    wstrb = 4'hF;
    axil_read(32'h28, r);
    if (r != 0) $fatal(1, "partial-strobe PUSH pushed (PUSH_WORDS=%0d)", r);
    axil_read(32'h08, r);
    if (!r[8]) $fatal(1, "err_strb not set, STATUS=%h", r);
    axil_write(32'h0C, 32'h8);
    axil_read(32'h08, r);
    if (r[8]) $fatal(1, "clear_err did not clear err_strb, STATUS=%h", r);

    // POP is a peek and POP_ACK checks seq: decode vector 0, read it twice, ack it with a wrong seq
    // (nothing happens), then with the right one (it is gone)
    push_syndrome(0);
    do axil_read(32'h14, r); while (r == 32'hFFFF_FFFF);
    axil_read(32'h14, c0);
    if (c0 != r || r[17:16] != 0) $fatal(1, "POP peek not stable: %h then %h", r, c0);
    axil_write(32'h24, 32'h1);
    axil_read(32'h14, c0);
    if (c0 != r) $fatal(1, "POP_ACK with wrong seq popped (%h -> %h)", r, c0);
    axil_write(32'h24, 32'h0);
    axil_read(32'h14, c0);
    if (c0 != 32'hFFFF_FFFF) $fatal(1, "POP_ACK with right seq did not pop: %h", c0);
    // restart counters so seq == test index again
    axil_write(32'h0C, 32'h1);
    repeat (50) @(posedge clk_main);
    axil_write(32'h0C, 32'h0);
    wait_ready();

    axil_read(32'h18, c0); axil_read(32'h1C, m0);
    repeat (4000) @(posedge clk_main);
    axil_read(32'h18, c1); axil_read(32'h1C, m1);
    ratio = real'(c1 - c0) / real'(m1 - m0);
    mhz = ratio * 250.0;
    $display("[tb] measured clk_core = %.2f MHz (nominal %0d kHz)", mhz, `TB_KHZ);
    if (mhz < 0.98 * `TB_KHZ / 1000.0 || mhz > 1.02 * `TB_KHZ / 1000.0)
      $fatal(1, "clk_core measured %.2f MHz", mhz);

    run_and_check(0, gobs.size(), worst);
    $display("[tb] %0d/%0d golden decodes match, worst latency %0d cycles", gobs.size(), gobs.size(), worst);

    // PCIS mode: soft reset with pcis_mode set, then release with it still set
    axil_write(32'h0C, 32'h5);
    repeat (50) @(posedge clk_main);
    axil_write(32'h0C, 32'h4);
    wait_ready();
    do axil_read(32'h08, r); while (r[11]);   // ring clearing
    if (!r[12]) $fatal(1, "pcis_mode not set, STATUS=%h", r);
    run_pcis(worst2);
    axil_read(32'h30, r);
    if (r != gobs.size()) $fatal(1, "PCIS_ACC=%0d, want %0d", r, gobs.size());
    axil_read(32'h2C, r);
    if (r != gobs.size()) $fatal(1, "RESULTS=%0d, want %0d", r, gobs.size());
    axil_read(32'h28, r);
    if (r != gobs.size() * NS) $fatal(1, "PUSH_WORDS=%0d, want %0d", r, gobs.size() * NS);
    axil_read(32'h34, r);
    if (r != 0) $fatal(1, "PCIS_DROP=%0d", r);
    $display("[tb] PCIS: %0d/%0d golden decodes match, worst latency %0d cycles", gobs.size(), gobs.size(),
             worst2);

    axil_write(32'h0C, 32'h1);
    repeat (50) @(posedge clk_main);
    axil_read(32'h08, r);
    if (!r[5]) $fatal(1, "soft reset did not reach the core, STATUS=%h", r);
    axil_write(32'h0C, 32'h0);
    wait_ready();
    run_and_check(0, 5, worst2);
    $display("[tb] after soft reset: 5/5 match");

    $display("PASS: %0d golden decodes bit-exact through OCL AXI-Lite and %0d through PCIS + XPM FIFOs, worst latency %0d cycles at %.2f MHz",
             gobs.size(), gobs.size(), worst, mhz);
    $finish;
  end
endmodule
