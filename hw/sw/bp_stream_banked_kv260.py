#!/usr/bin/env python3
# Q7-06 (AC-1) — KV260 batched AXI-DMA runner for the banked relay-BP BLOCK decoder
# (bp_stream_banked overlay). This is the >=100x-throughput path over the per-word AXI-Lite runner
# (bp_circ_kv260.py): a whole BATCH of independent syndrome->result experiments is pushed through ONE
# AXI-DMA transfer (MM2S in, S2MM out) instead of NS MMIO writes + a Python poll loop + reads per
# experiment. NS=ceil(C/32)=5 syndrome words in per experiment; one status word out per experiment
# ({obs[11:0], vflag@19, latency[15:0]}, see bp_stream_banked_core.sv).
#
# Two things it does:
#   1. CORRECTNESS — decode the 40-shot circuit-level golden (bp_circ_vectors.txt, the SAME golden the
#      co-sim and the AXI-Lite overlay pass) as one batch; assert 40/40 obs+vflag bit-exact on silicon.
#   2. THROUGHPUT — replay a large batch many times, report experiments/sec (batched harness rate). Divide
#      by the per-word AXI-Lite rate (bp_circ_kv260.py, measured separately) for the AC-1 >=100x figure.
#
# Kria-PYNQ 3.0.1 note: pynq.Overlay(bit) dies on designs with no PL DRAM banks (the stub xclbinutil
# masks the empty-MEM_TOPOLOGY failure -> t.xclbin never built -> Overlay raises). This design routes DMA
# through the PS HP port to PS DDR (no PL DRAM banks), so we AVOID Overlay entirely: program the PL with
# pynq.Bitstream(bit).download() and drive the AXI-DMA engine's registers directly over MMIO (exactly what
# pynq.lib.dma does internally), with pynq.allocate() for the physically-contiguous DMA buffers.
#
# Usage (as root, pynq venv + XRT, from a dir holding the .bit + vectors):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        bp_stream_banked_kv260.py bp_kv260_stream_banked.bit bp_circ_vectors.txt \
#        [--base 0xA0000000] [--clk 100e6] [--bench-batch 100000] [--bench-reps 20]

import sys
import time

# ---- AXI DMA (simple/direct register mode) register map (PG021) ----
MM2S_DMACR = 0x00  # control; bit0 = RS (run/stop)
MM2S_DMASR = 0x04  # status;  bit0 = Halted, bit1 = Idle
MM2S_SA = 0x18  # source address (low 32)
MM2S_SA_MSB = 0x1C  # source address (high 32)
MM2S_LENGTH = 0x28  # bytes to transfer; WRITING THIS ARMS/STARTS the MM2S channel
S2MM_DMACR = 0x30  # control; bit0 = RS
S2MM_DMASR = 0x34  # status;  bit1 = Idle
S2MM_DA = 0x48  # destination address (low 32)
S2MM_DA_MSB = 0x4C  # destination address (high 32)
S2MM_LENGTH = 0x58  # bytes to transfer; WRITING THIS ARMS the S2MM channel

DMACR_RS = 0x1
DMASR_IDLE = 0x2
DMASR_HALTED = 0x1


def load_vectors(path):
    """Parse 'T N C OBS' header + per-test s/h/o/v lines. Returns (T,N,C,OBS, [(synd_bits, obs, vflag)])."""
    T = N = C = OBS = 0
    tests = []
    cur = {}
    with open(path) as f:
        header_done = False
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            if not header_done:
                T, N, C, OBS = (int(x) for x in line.split()[:4])
                header_done = True
                continue
            tag = line[0]
            body = line[1:].strip()
            if tag == "s":
                cur = {"s": body}
            elif tag == "o":
                cur["o"] = body
            elif tag == "v":
                cur["v"] = body
                # commit test on 'v' (s,o already seen; 'h' correction is unused here)
                synd = cur["s"]
                obs = 0
                for i, ch in enumerate(cur.get("o", "")):
                    if ch == "1":
                        obs |= 1 << i
                vflag = 1 if cur.get("v", "0").startswith("1") else 0
                tests.append((synd, obs, vflag))
    return T, N, C, OBS, tests


def pack_syndrome(synd_bits, C, NS):
    """Pack a C-char '0/1' string into NS little-endian uint32 words (bit c -> word c//32, bit c%32)."""
    words = [0] * NS
    for c in range(C):
        if c < len(synd_bits) and synd_bits[c] == "1":
            words[c // 32] |= 1 << (c % 32)
    return words


def main(argv):
    import numpy as np
    from pynq import Bitstream, MMIO, allocate

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vecfile = next((a for a in argv[1:] if a.endswith(".txt")), "bp_circ_vectors.txt")
    base = 0xA0000000
    clk = 100_000_000
    bench_batch = 100_000
    bench_reps = 20
    it = iter(argv[1:])
    for a in it:
        if a == "--base":
            base = int(next(it), 0)
        elif a == "--clk":
            clk = int(float(next(it)))
        elif a == "--bench-batch":
            bench_batch = int(next(it))
        elif a == "--bench-reps":
            bench_reps = int(next(it))
    if not bitfile:
        print("usage: bp_stream_banked_kv260.py <design.bit> <bp_circ_vectors.txt> "
              "[--base 0xADDR] [--clk 100e6] [--bench-batch N] [--bench-reps R]")
        return 2

    T, N, C, OBS, tests = load_vectors(vecfile)
    if not tests:
        print("FAIL: no golden vectors parsed from %s" % vecfile)
        return 2
    NS = (C + 31) // 32
    print("[info] %d checks / %d vars / %d obs ; %d golden tests ; NS=%d words/exp" % (C, N, OBS, len(tests), NS))

    print("[board] programming PL with %s ..." % bitfile)
    Bitstream(bitfile).download()
    dma = MMIO(base, 0x1000)

    def reset_dma():
        # RS=0 then RS=1 on both channels; writing LENGTH later actually starts a transfer.
        dma.write(MM2S_DMACR, 0)
        dma.write(S2MM_DMACR, 0)
        dma.write(MM2S_DMACR, DMACR_RS)
        dma.write(S2MM_DMACR, DMACR_RS)

    def run_batch(in_buf, out_buf, n_exp):
        """One batched transfer: MM2S sends n_exp*NS words, S2MM receives n_exp words. Returns elapsed s."""
        in_buf.flush()
        reset_dma()
        t0 = time.perf_counter()
        # arm S2MM (receive) first so it is ready when results stream out
        dma.write(S2MM_DA, out_buf.physical_address & 0xFFFFFFFF)
        dma.write(S2MM_DA_MSB, (out_buf.physical_address >> 32) & 0xFFFFFFFF)
        dma.write(S2MM_LENGTH, n_exp * 4)
        # start MM2S (send)
        dma.write(MM2S_SA, in_buf.physical_address & 0xFFFFFFFF)
        dma.write(MM2S_SA_MSB, (in_buf.physical_address >> 32) & 0xFFFFFFFF)
        dma.write(MM2S_LENGTH, n_exp * NS * 4)
        # wait for both channels idle
        g = 0
        while not (dma.read(MM2S_DMASR) & DMASR_IDLE) and g < 100_000_000:
            g += 1
        g = 0
        while not (dma.read(S2MM_DMASR) & DMASR_IDLE) and g < 100_000_000:
            g += 1
        dt = time.perf_counter() - t0
        out_buf.invalidate()
        if not (dma.read(MM2S_DMASR) & DMASR_IDLE) or not (dma.read(S2MM_DMASR) & DMASR_IDLE):
            raise RuntimeError("DMA channel did not go idle (MM2S_SR=0x%08x S2MM_SR=0x%08x)"
                               % (dma.read(MM2S_DMASR), dma.read(S2MM_DMASR)))
        return dt

    obs_mask = (1 << OBS) - 1

    # ---- 1. correctness: the 40 golden shots as one batch ----
    in_buf = allocate(shape=(len(tests) * NS,), dtype=np.uint32)
    out_buf = allocate(shape=(len(tests),), dtype=np.uint32)
    for t, (synd, _, _) in enumerate(tests):
        w = pack_syndrome(synd, C, NS)
        in_buf[t * NS:(t + 1) * NS] = np.asarray(w, dtype=np.uint32)
    out_buf[:] = 0
    run_batch(in_buf, out_buf, len(tests))

    mism = 0
    for t, (_, want_obs, want_v) in enumerate(tests):
        word = int(out_buf[t])
        got_obs = (word >> 20) & obs_mask
        got_v = (word >> 19) & 1
        if got_obs != (want_obs & obs_mask) or got_v != want_v:
            if mism < 8:
                print("  test %d: obs got 0x%03x want 0x%03x, vflag got %d want %d"
                      % (t, got_obs, want_obs & obs_mask, got_v, want_v))
            mism += 1
    del in_buf, out_buf
    ok = (mism == 0)
    print("CORRECTNESS: %s (%d/%d batched decodes match golden on KV260 silicon)"
          % ("PASS" if ok else "FAIL", len(tests) - mism, len(tests)))
    if not ok:
        return 1

    # ---- 2. throughput: replay a large batch and report experiments/sec ----
    # Reuse the golden syndromes tiled up to bench_batch so the decoder sees realistic (not all-zero) work.
    nb = max(1, bench_batch)
    bin_ = allocate(shape=(nb * NS,), dtype=np.uint32)
    bout = allocate(shape=(nb,), dtype=np.uint32)
    packed = [pack_syndrome(s, C, NS) for (s, _, _) in tests]
    flat = np.asarray([w for word in packed for w in word], dtype=np.uint32)
    reps_tile = (nb * NS + flat.size - 1) // flat.size
    tiled = np.tile(flat, reps_tile)[: nb * NS]
    bin_[:] = tiled

    # warm-up (first transfer pays allocation/driver setup), then timed reps
    run_batch(bin_, bout, nb)
    best = None
    total = 0.0
    for _ in range(bench_reps):
        dt = run_batch(bin_, bout, nb)
        total += dt
        best = dt if best is None else min(best, dt)
    mean = total / bench_reps
    print("THROUGHPUT: batch=%d exp, %d reps @ clk=%.0f MHz" % (nb, bench_reps, clk / 1e6))
    print("  best  %.4f s  -> %.3e exp/s  (%.3f us/exp)" % (best, nb / best, 1e6 * best / nb))
    print("  mean  %.4f s  -> %.3e exp/s  (%.3f us/exp)" % (mean, nb / mean, 1e6 * mean / nb))
    print("  (divide by the per-word AXI-Lite exp/s from bp_circ_kv260.py for the AC-1 speedup)")
    del bin_, bout
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
