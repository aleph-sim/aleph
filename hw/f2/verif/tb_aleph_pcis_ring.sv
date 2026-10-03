// P2 appliance v2-large — unit testbench for aleph_pcis_ring (the PCIS batch path), with a stand-in
// decoder so it runs in seconds. The full-core gate is tb_aleph_bp_ocl.
//
// The stand-in takes NS words per experiment from the input-FIFO port and, after a random delay, offers
// a result word computed from those words. The test then checks, for every experiment, that the ring
// entry of its slot carries its tag and the result of exactly its syndrome. The host side does what a
// real host may do to a write-combining BAR: whole 64-byte stores, the same store split into halves or
// 4-byte pieces, pieces of different slots interleaved and reordered, multi-beat bursts across slots,
// and narrow / burst reads of the ring. Negative checks: writes outside PCIS mode and into the ring
// window are refused and counted, a slot whose bytes are not all in does not commit, and the ring reads
// all-ones after reset.
// Prints "PASS: ..." on success; any failure ends with $fatal.

`timescale 1ns / 1ps

module tb_aleph_pcis_ring;
  localparam int NS = 5, SLOTS = 128, N = 600;
  // explicit all-ones: '1 inside a queue literal is not widened by every simulator
  localparam logic [63:0] S_ALL = {64{1'b1}};
  localparam logic [511:0] D_ALL = {512{1'b1}};

  logic clk = 0, rst = 1, enable = 0;
  always #2 clk = ~clk;   // 250 MHz

  logic [63:0]  awaddr, araddr;
  logic [15:0]  awid, arid, bid, rid;
  logic [2:0]   awsize, arsize;
  logic [7:0]   arlen;
  logic         awvalid, awready, wlast, wvalid, wready, bvalid, bready;
  logic         arvalid, arready, rlast, rvalid, rready;
  logic [511:0] wdata, rdata;
  logic [63:0]  wstrb;
  logic [1:0]   bresp, rresp;
  logic         in_wr_en, in_can_write, out_valid, out_rd_en, commit_pulse, drop_pulse, clearing;
  logic [31:0]  in_din, out_dout;

  aleph_pcis_ring #(.NS(NS), .SLOTS(SLOTS)) dut (.*);

  always @(posedge clk)
    if (dut.beat && $test$plusargs("DEBUG"))
      $display("[%0t] beat addr %h slot %0d strb %h mask_new %h complete %0d wlast %0d", $time, dut.b_addr,
               dut.wslot, wstrb, dut.mask_new, dut.complete, wlast);

  // ---- stand-in decoder: in-order, random input back-pressure and random decode time
  int     commits = 0, drops = 0;
  always @(posedge clk) begin
    if (commit_pulse) commits++;
    if (drop_pulse) drops++;
  end

  function automatic logic [31:0] fake_result(input logic [NS*32-1:0] s);
    logic [31:0] h = 32'h9E37_79B9;
    for (int i = 0; i < NS; i++) h = {h[26:0], h[31:27]} ^ s[32*i +: 32] ^ (h >> 7);
    return h & 32'hFFF8_FFFF;   // bits [18:16] are 0 in a real result word
  endfunction

  logic [NS*32-1:0] asm_w;
  int               asm_n = 0;
  logic [31:0]      outq[$];
  logic [31:0]      pend_r[$];       // decodes in progress, serial like the real core:
  longint           pend_t[$];       // each finishes a random time after the previous one
  longint           cyc = 0, last_done = 0;
  // out_valid / out_dout are registered (updated in the NBA region like the real FIFO), so the DUT
  // samples them race-free at the clock edge
  always @(posedge clk) begin
    in_can_write <= ($urandom % 4) != 0;
    if (in_wr_en) begin
      asm_w[32*asm_n +: 32] = in_din;
      if (++asm_n == NS) begin
        asm_n = 0;
        last_done = (last_done > cyc ? last_done : cyc) + 1 + $urandom % 40;
        pend_r.push_back(fake_result(asm_w));
        pend_t.push_back(last_done);
      end
    end
    cyc++;
    if (pend_t.size() > 0 && pend_t[0] <= cyc) begin
      outq.push_back(pend_r.pop_front());
      void'(pend_t.pop_front());
    end
    if (out_rd_en) void'(outq.pop_front());
    out_valid <= outq.size() > 0;
    out_dout  <= outq.size() > 0 ? outq[0] : 32'hX;
  end
  initial out_valid = 0;

  // ---- AXI4 master (one transaction at a time; the slave must cope with W arriving before AW)
  task automatic axi_write(input logic [63:0] a, input logic [2:0] size, input logic [511:0] d[$],
                           input logic [63:0] s[$]);
    @(posedge clk);
    awid <= 16'($urandom); awaddr <= a; awsize <= size;
    fork
      begin
        if ($urandom % 2) repeat ($urandom % 3) @(posedge clk);
        awvalid <= 1;
        do @(posedge clk); while (!awready);
        awvalid <= 0;
      end
      begin
        for (int i = 0; i < d.size(); i++) begin
          wdata <= d[i]; wstrb <= s[i]; wlast <= (i == d.size() - 1); wvalid <= 1;
          do @(posedge clk); while (!wready);
          wvalid <= 0;
          if ($urandom % 3 == 0) @(posedge clk);
        end
      end
    join
    bready <= 1;
    do @(posedge clk); while (!bvalid);
    if (bresp != 2'b00) $fatal(1, "BRESP %0d", bresp);
    bready <= 0;
  endtask

  task automatic axi_read(input logic [63:0] a, input logic [2:0] size, input int len,
                          output logic [511:0] d[$]);
    logic [15:0] id = 16'($urandom);
    d = {};
    @(posedge clk);
    arid <= id; araddr <= a; arsize <= size; arlen <= 8'(len - 1); arvalid <= 1;
    do @(posedge clk); while (!arready);
    arvalid <= 0;
    for (int i = 0; i < len; i++) begin
      rready <= ($urandom % 2) != 0;
      do begin
        @(posedge clk);
        if (!rready) rready <= 1;
      end while (!(rvalid && rready));
      if (rid != id) $fatal(1, "RID %h, want %h", rid, id);
      if (rlast != (i == len - 1)) $fatal(1, "RLAST wrong on beat %0d/%0d", i, len);
      d.push_back(rdata);
    end
    rready <= 0;
  endtask

  // ---- host model
  logic [NS*32-1:0] synd[N];
  logic [31:0]      tag[N];
  // burst_k >= 0: a two-beat burst carrying experiments burst_k and burst_k + 1
  typedef struct { logic [63:0] a; logic [2:0] size; logic [511:0] d; logic [63:0] s; int burst_k; } piece_t;

  function automatic logic [511:0] slot_line(input int k);
    logic [511:0] l = '0;
    l[NS*32-1:0] = synd[k];
    l[255:224]   = tag[k];
    l[511:256]   = {8{32'hBAD0_BAD0}};   // ignored half: must never reach the decoder
    return l;
  endfunction

  // the pieces one slot's store may arrive as
  task automatic split(input int k, inout piece_t q[$]);
    logic [63:0]  base = 64'(k % SLOTS) * 64;
    logic [511:0] l = slot_line(k);
    case ($urandom % 3)
      0: q.push_back('{base, 3'd6, l, S_ALL, -1});
      1: begin
        q.push_back('{base, 3'd6, l, {32'h0, 32'hFFFF_FFFF}, -1});
        q.push_back('{base, 3'd6, l, {32'hFFFF_FFFF, 32'h0}, -1});
      end
      default:
        for (int b = 0; b < 32; b += 4)
          q.push_back('{base + b, 3'd2, l, 64'hF << b, -1});
    endcase
  endtask

  int got = 0;
  task automatic collect(input int first, input int last);
    // poll the ring rows until entries first..last show their tags
    logic [511:0] d[$];
    int k = first, guard = 0;
    while (k <= last) begin
      int s = k % SLOTS;
      int row = s / 8;
      // mix of whole-row reads, a 2-row burst, and narrow 8-byte reads
      case ($urandom % 3)
        0: axi_read(64'h1_0000 + 64 * row, 3'd6, 1, d);
        1: axi_read(64'h1_0000 + 64 * row, 3'd6, (row < SLOTS / 8 - 1) ? 2 : 1, d);
        default: axi_read(64'h1_0000 + 8 * s, 3'd3, 1, d);
      endcase
      if (d[0][64*(s%8) + 32 +: 32] === tag[k]) begin
        logic [31:0] r = d[0][64*(s%8) +: 32];
        if (r !== fake_result(synd[k])) $fatal(1, "exp %0d slot %0d: result %h, want %h", k, s, r,
                                                fake_result(synd[k]));
        got++; k++; guard = 0;
      end else if (++guard > 5000) begin
        $fatal(1, "exp %0d (slot %0d, tag %h) never completed; entry %h; commits %0d drops %0d cq %0d/%0d tq %0d/%0d sst %0d outq %0d",
               k, s, tag[k], d[0][64*(s%8) +: 64], commits, drops, dut.cq_wp, dut.cq_rp, dut.tq_wp,
               dut.tq_rp, dut.sst, outq.size());
      end
    end
  endtask

  initial begin
    logic [511:0] d[$];
    piece_t       q[$];
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0; wlast = 0;
    for (int k = 0; k < N; k++) begin
      for (int i = 0; i < NS; i++) synd[k][32*i +: 32] = $urandom;
      tag[k] = 32'h5A00_0000 + k;
    end
    repeat (5) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    while (clearing) @(posedge clk);

    // after reset every ring entry is all-ones (one 16-beat burst over the whole ring)
    axi_read(64'h1_0000, 3'd6, SLOTS / 8, d);
    foreach (d[i]) if (d[i] !== D_ALL) $fatal(1, "ring row %0d after reset: %h", i, d[i]);
    // the input window reads as zero
    axi_read(64'h0, 3'd6, 1, d);
    if (d[0] !== '0) $fatal(1, "input window read %h", d[0]);

    // not in PCIS mode: refused and counted, nothing reaches the decoder
    axi_write(64'h0, 3'd6, '{slot_line(0)}, '{S_ALL});
    repeat (20) @(posedge clk);
    if (drops != 1 || commits != 0) $fatal(1, "disabled write: drops %0d commits %0d", drops, commits);

    enable <= 1;
    @(posedge clk);
    // a write into the ring window is refused too
    axi_write(64'h1_0000, 3'd6, '{D_ALL}, '{S_ALL});
    repeat (5) @(posedge clk);
    if (drops != 2) $fatal(1, "ring-window write not refused");
    // the 32 high bytes alone are not an experiment and not an error
    axi_write(64'h0, 3'd6, '{slot_line(0)}, '{{32'hFFFF_FFFF, 32'h0}});
    repeat (5) @(posedge clk);
    if (drops != 2 || commits != 0) $fatal(1, "high-half-only write acted");

    // main run: batches of up to 16 experiments, pieces shuffled across the batch; every so often a
    // two-beat burst covering two neighbouring slots
    for (int k0 = 0; k0 < N; ) begin
      int nb = 1 + $urandom % 16;
      if (k0 + nb > N) nb = N - k0;
      q = {};
      for (int k = k0; k < k0 + nb; k++) begin
        if (k + 1 < k0 + nb && (k % SLOTS) != SLOTS - 1 && $urandom % 5 == 0) begin
          q.push_back('{64'(k % SLOTS) * 64, 3'd6, '0, S_ALL, k});
          k++;
        end else begin
          split(k, q);
        end
      end
      q.shuffle();
      foreach (q[i]) begin
        if (q[i].burst_k >= 0) begin
          axi_write(q[i].a, 3'd6, '{slot_line(q[i].burst_k), slot_line(q[i].burst_k + 1)}, '{S_ALL, S_ALL});
        end else begin
          axi_write(q[i].a, q[i].size, '{q[i].d}, '{q[i].s});
        end
      end
      collect(k0, k0 + nb - 1);
      k0 += nb;
    end
    if (commits != N || drops != 2) $fatal(1, "commits %0d (want %0d), drops %0d (want 2)", commits, N, drops);

    // a slot with bytes missing does not commit; the missing bytes (which carry a new tag) complete it
    synd[N - 1] = ~synd[N - 1];
    axi_write(64'((N - 1) % SLOTS) * 64, 3'd6, '{slot_line(N - 1)}, '{64'h0000_0000_00FF_FFFF});
    repeat (100) @(posedge clk);
    if (commits != N) $fatal(1, "partial slot committed");
    tag[N - 1] = 32'h6B00_0001;
    axi_write(64'((N - 1) % SLOTS) * 64, 3'd6, '{slot_line(N - 1)}, '{64'h0000_0000_FF00_0000});
    collect(N - 1, N - 1);
    if (commits != N + 1) $fatal(1, "completed slot did not commit");

    $display("PASS: %0d experiments through PCIS slots (split, reordered, burst) and back via the ring, all bit-exact; %0d refused writes counted",
             got, drops);
    $finish;
  end
endmodule
