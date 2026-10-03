// P2 appliance v2-large — batched host path over F2 PCIS (AppPF BAR4, 512-bit AXI4 slave).
//
// The F2 small shell has no DMA engine, so a batch has to arrive as host MMIO stores. A 64-byte store
// sequence into a write-combining mapping of BAR4 (`resource4_wc`) usually leaves the CPU as one 64-byte
// PCIe write, i.e. one 512-bit beat here, but nothing guarantees it: a WC buffer may be flushed in pieces
// and different lines may be flushed out of program order. And the BAR is prefetchable, so reads must be
// side-effect free (`hdk/docs/AWS_Fpga_Pcie_Memory_Map.md`, "Prefetchable BAR Setting"). The protocol
// below is built to be correct under all of that, and it never stalls the bus (the shell times a PCIS
// transaction out after 8 us, `AWS_Shell_Interface_Specification.md` "PCIS Interface Timeout Details"):
//
//   * Input = SLOTS experiment slots of 64 bytes at BAR4 offset 64*s (s < SLOTS). Bytes [4*NS-1:0] are
//     the syndrome in the same word order as OCL PUSH, bytes [31:28] a host-chosen 32-bit tag (never
//     0xFFFF_FFFF), bytes [63:32] are ignored. Each slot keeps a byte mask of what has arrived; when all
//     32 low bytes are in, the slot commits (mask cleared) and its experiment queues for the decoder. So
//     a store split into any number of pieces, in any order, still yields exactly one experiment.
//   * Output = a result ring at BAR4 offset 0x1_0000: slot s's 64-bit entry {tag, result word} at
//     0x1_0000 + 8*s, so one 64-byte read returns 8 entries. Entries are written in place and reads have
//     no side effect. The host knows experiment k (tag t, slot s) is done when entry s reads back tag t,
//     whatever order things arrived or completed in. After reset every entry reads all-ones.
//   * Flow control is the host's: at most one experiment in flight per slot (reuse slot s only after its
//     result arrived). Nothing ever back-pressures PCIS; a commit that finds the queue full (only possible
//     if the host breaks that rule) is dropped and counted.
//
// The decoder core is in-order, so the slot/tag of each experiment rides a FIFO alongside it, and the
// result mover pairs the next result word with the next {slot, tag}.

`timescale 1ns / 1ps

module aleph_pcis_ring #(
    parameter int NS    = 5,     // syndrome words per experiment (<= 7: bytes [31:28] are the tag)
    parameter int SLOTS = 128
) (
    input  logic         clk,          // clk_main_a0
    input  logic         rst,          // active-high, synchronous (soft reset / shell reset)
    input  logic         enable,       // PCIS mode: accept experiments and drain decoder results

    // PCIS AXI4 slave (only the fields the protocol needs; the rest are ignored by the CL top)
    input  logic [63:0]  awaddr,
    input  logic [15:0]  awid,
    input  logic [2:0]   awsize,
    input  logic         awvalid,
    output logic         awready,
    input  logic [511:0] wdata,
    input  logic [63:0]  wstrb,
    input  logic         wlast,
    input  logic         wvalid,
    output logic         wready,
    output logic [15:0]  bid,
    output logic [1:0]   bresp,
    output logic         bvalid,
    input  logic         bready,
    input  logic [63:0]  araddr,
    input  logic [15:0]  arid,
    input  logic [7:0]   arlen,
    input  logic [2:0]   arsize,
    input  logic         arvalid,
    output logic         arready,
    output logic [15:0]  rid,
    output logic [511:0] rdata,
    output logic [1:0]   rresp,
    output logic         rlast,
    output logic         rvalid,
    input  logic         rready,

    // decoder input FIFO (write side)
    output logic         in_wr_en,
    output logic [31:0]  in_din,
    input  logic         in_can_write,  // ~full & ~wr_rst_busy
    // decoder output FIFO (read side, FWFT)
    input  logic         out_valid,     // ~empty & ~rd_rst_busy
    input  logic [31:0]  out_dout,
    output logic         out_rd_en,

    output logic         commit_pulse,  // an experiment was queued
    output logic         drop_pulse,    // a write beat was refused (not in PCIS mode, ring window, queue full)
    output logic         clearing       // result ring is being reset to all-ones
);
  localparam int SW    = $clog2(SLOTS);
  localparam int RROWS = SLOTS / 8;          // ring rows of 8 x 64-bit entries
  localparam int RW    = $clog2(RROWS);
  localparam int TAGW  = SW + 32;            // {slot, tag}

`ifndef SYNTHESIS
  initial begin
    if (NS > 7) $fatal(1, "aleph_pcis_ring: NS=%0d does not leave bytes [31:28] for the tag", NS);
    if (SLOTS < 8 || (SLOTS & (SLOTS - 1)) != 0) $fatal(1, "aleph_pcis_ring: SLOTS must be a power of 2 >= 8");
  end
`endif

  // ================================================================ write channel
  typedef enum logic [1:0] { W_ADDR, W_DATA, W_RESP } wstate_t;
  wstate_t     wst;
  logic [63:0] waddr;
  logic [2:0]  wsize;

  assign awready = (wst == W_ADDR);
  assign wready  = (wst == W_DATA);
  assign bvalid  = (wst == W_RESP);
  assign bresp   = 2'b00;

  // Each accepted W beat is registered before it touches the masks or the slot RAM: the shell's PCIS
  // pins may sit in another SLR, and this stage never stalls (one beat per cycle in, one processed).
  logic          beat;
  logic [63:0]   b_addr;
  logic [255:0]  b_data;
  logic [31:0]   strb_lo;
  always_ff @(posedge clk) begin
    beat    <= ~rst & wready & wvalid;
    b_addr  <= waddr;
    b_data  <= wdata[255:0];
    strb_lo <= wstrb[31:0];
  end
  wire           in_win   = ~b_addr[16];
  wire [SW-1:0]  wslot    = b_addr[6 +: SW];

  // per-slot arrival masks (flops: read-modify-write every beat, no RAM hazard)
  logic [31:0] mask [SLOTS];
  wire  [31:0] mask_new = mask[wslot] | strb_lo;
  wire         complete = beat & in_win & enable & (|strb_lo) & (&mask_new);

  // commit queue: slots whose 32 bytes are all in
  logic [SW-1:0] cq [SLOTS];
  logic [SW:0]   cq_wp, cq_rp;
  wire           cq_empty = (cq_wp == cq_rp);
  wire           cq_full  = (cq_wp - cq_rp) == (SW + 1)'(SLOTS);
  logic          cq_pop;

  assign commit_pulse = complete & ~cq_full;
  assign drop_pulse   = beat & (|strb_lo) & (~in_win | ~enable | (complete & cq_full));

  // slot payload RAM, byte-write
  logic [255:0] sbuf [SLOTS];
  logic [SW-1:0] sb_raddr;
  logic [255:0]  sb_q;
  always_ff @(posedge clk) begin
    if (beat & in_win & enable)
      for (int b = 0; b < 32; b++)
        if (strb_lo[b]) sbuf[wslot][8*b +: 8] <= b_data[8*b +: 8];
    sb_q <= sbuf[sb_raddr];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      wst   <= W_ADDR;
      cq_wp <= '0;
      for (int s = 0; s < SLOTS; s++) mask[s] <= '0;
    end else begin
      unique case (wst)
        W_ADDR: if (awvalid) begin
          waddr <= awaddr;
          wsize <= awsize;
          bid   <= awid;
          wst   <= W_DATA;
        end
        W_DATA: if (wvalid) begin
          // INCR: the next beat's address advances by the beat size (narrow beats keep their lanes)
          waddr <= waddr + (64'd1 << wsize);
          if (wlast) wst <= W_RESP;
        end
        W_RESP: if (bready) wst <= W_ADDR;
        default: wst <= W_ADDR;
      endcase
      if (beat & in_win & enable & (|strb_lo)) mask[wslot] <= complete ? '0 : mask_new;
      if (commit_pulse) begin
        cq[cq_wp[SW-1:0]] <= wslot;
        cq_wp <= cq_wp + 1'b1;
      end
    end
  end

  // ================================================================ serializer: slot -> NS decoder words
  typedef enum logic [1:0] { S_IDLE, S_LOAD, S_PUSH } sstate_t;
  sstate_t      sst;
  logic [SW-1:0] s_slot;
  logic [255:0]  s_row;
  logic [2:0]    s_word;

  // {slot, tag} of experiments inside the decoder, in order
  logic [TAGW-1:0] tq [SLOTS];
  logic [SW:0]     tq_wp, tq_rp;
  wire             tq_empty = (tq_wp == tq_rp);
  wire             tq_full  = (tq_wp - tq_rp) == (SW + 1)'(SLOTS);
  logic            tq_push, tq_pop;

  assign sb_raddr = cq[cq_rp[SW-1:0]];
  assign cq_pop   = (sst == S_IDLE) & ~cq_empty & ~tq_full & enable;
  assign in_din   = s_row[32*s_word +: 32];
  assign in_wr_en = (sst == S_PUSH) & in_can_write;
  assign tq_push  = in_wr_en & (s_word == 3'(NS - 1));

  always_ff @(posedge clk) begin
    if (rst) begin
      sst   <= S_IDLE;
      cq_rp <= '0;
      tq_wp <= '0;
    end else begin
      unique case (sst)
        S_IDLE: if (cq_pop) begin
          s_slot <= sb_raddr;
          cq_rp  <= cq_rp + 1'b1;
          sst    <= S_LOAD;
        end
        S_LOAD: begin            // sb_q now holds sbuf[s_slot]
          s_row  <= sb_q;
          s_word <= '0;
          sst    <= S_PUSH;
        end
        S_PUSH: if (in_can_write) begin
          s_word <= s_word + 1'b1;
          if (s_word == 3'(NS - 1)) sst <= S_IDLE;
        end
        default: sst <= S_IDLE;
      endcase
      if (tq_push) begin
        tq[tq_wp[SW-1:0]] <= {s_slot, s_row[255:224]};
        tq_wp <= tq_wp + 1'b1;
      end
    end
  end

  // ================================================================ result ring
  logic [511:0] ring [RROWS];
  logic [RW-1:0] ring_waddr, ring_raddr, clr_row;
  logic [7:0]    ring_we;
  logic [63:0]   ring_entry;
  logic [511:0]  ring_q;

  wire [TAGW-1:0] tq_head = tq[tq_rp[SW-1:0]];
  wire [SW-1:0]   m_slot  = tq_head[TAGW-1 -: SW];
  assign tq_pop    = enable & ~clearing & out_valid & ~tq_empty;
  assign out_rd_en = tq_pop;

  always_comb begin
    if (clearing) begin
      ring_waddr = clr_row;
      ring_we    = 8'hFF;
      ring_entry = '1;
    end else begin
      ring_waddr = m_slot[SW-1:3];
      ring_we    = tq_pop ? (8'd1 << m_slot[2:0]) : 8'd0;
      ring_entry = {tq_head[31:0], out_dout};
    end
  end

  always_ff @(posedge clk) begin
    for (int l = 0; l < 8; l++)
      if (ring_we[l]) ring[ring_waddr][64*l +: 64] <= ring_entry;
    ring_q <= ring[ring_raddr];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      clearing <= 1'b1;
      clr_row  <= '0;
      tq_rp    <= '0;
    end else begin
      if (clearing) begin
        clr_row <= clr_row + 1'b1;
        if (clr_row == RW'(RROWS - 1)) clearing <= 1'b0;
      end
      if (tq_pop) tq_rp <= tq_rp + 1'b1;
    end
  end

  // ================================================================ read channel (side-effect free)
  typedef enum logic [1:0] { R_ADDR, R_FETCH, R_WAIT, R_DATA } rstate_t;
  rstate_t     rst_q;
  logic [63:0] raddr;
  logic [2:0]  rsize;
  logic [7:0]  rleft;

  assign arready    = (rst_q == R_ADDR);
  assign rvalid     = (rst_q == R_DATA);
  assign rresp      = 2'b00;
  assign rlast      = (rleft == 8'd0);
  assign ring_raddr = raddr[6 +: RW];

  always_ff @(posedge clk) begin
    if (rst) begin
      rst_q <= R_ADDR;
    end else begin
      unique case (rst_q)
        R_ADDR: if (arvalid) begin
          raddr <= araddr;
          rsize <= arsize;
          rleft <= arlen;
          rid   <= arid;
          rst_q <= R_FETCH;
        end
        R_FETCH: rst_q <= R_WAIT;              // ring_raddr presented this cycle
        R_WAIT: begin
          // the input window and anything outside the ring read as zero
          rdata <= (raddr[16] && raddr[15:6 + RW] == '0) ? ring_q : '0;
          rst_q <= R_DATA;
        end
        R_DATA: if (rready) begin
          if (rleft == 8'd0) begin
            rst_q <= R_ADDR;
          end else begin
            raddr <= raddr + (64'd1 << rsize);
            rleft <= rleft - 1'b1;
            rst_q <= R_FETCH;
          end
        end
        default: rst_q <= R_ADDR;
      endcase
    end
  end
endmodule
