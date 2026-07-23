#!/usr/bin/env python3
# Q7-06 AC-2 — KV260 on-silicon 10^6-shot LER campaign for the batched banked relay-BP decoder.
#
# Streams a large set of REAL DEM shots (binary <prefix>.syn = NS little-endian u32 syndrome words per
# shot) through the batched AXI-DMA decoder overlay, collects the RTL's per-shot observable-flip
# prediction, and compares logical-error rates against the software FixedRelayBp golden carried in
# <prefix>.ref (u16 true_obs, u16 sw_obs per shot). Reports, per physical-error point:
#   RTL LER   = mean(rtl_obs != true_obs)   -- the silicon decoder's logical-error rate
#   SW  LER   = mean(sw_obs  != true_obs)   -- the software golden's rate (the reference)
#   |diff|    within combined 95% CI ?      -- AC-2 acceptance (RTL LER within CI of software golden)
#   divergence= mean(rtl_obs != sw_obs)     -- direct bit-exactness check (should be ~0)
# across the >=3 points given on the command line.
#
# The syndromes stream as one contiguous DMA input (no per-shot Python repacking); results come back one
# u16-in-u32 status word per shot. Chunked to keep each DMA buffer within the CMA pool.
#
# Usage (root, pynq venv + XRT, from a dir with the FULL-SCHEDULE .bit + the <prefix>.syn/.ref files):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        bp_stream_banked_ler_kv260.py bp_kv260_stream_banked.bit p003 p005 p007 [--chunk 100000] [--obs 12] [--ns 5]

import math
import os
import sys
import time

MM2S_DMACR, MM2S_DMASR, MM2S_SA, MM2S_SA_MSB, MM2S_LENGTH = 0x00, 0x04, 0x18, 0x1C, 0x28
S2MM_DMACR, S2MM_DMASR, S2MM_DA, S2MM_DA_MSB, S2MM_LENGTH = 0x30, 0x34, 0x48, 0x4C, 0x58


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def main(argv):
    import numpy as np
    from pynq import Overlay, MMIO, allocate

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    base = 0xA0000000
    chunk = 100_000
    OBS = 12
    NS = 5
    prefixes = []
    it = iter(argv[1:])
    for a in it:
        if a.endswith(".bit"):
            continue
        elif a == "--chunk":
            chunk = int(next(it))
        elif a == "--obs":
            OBS = int(next(it))
        elif a == "--ns":
            NS = int(next(it))
        elif a == "--base":
            base = int(next(it), 0)
        else:
            prefixes.append(a)
    if not bitfile or not prefixes:
        print("usage: bp_stream_banked_ler_kv260.py <bit> <prefix1> [prefix2 ...] [--chunk N] [--obs 12] [--ns 5]")
        return 2
    obs_mask = (1 << OBS) - 1

    # sidecar xclbin so Overlay bypasses the Kria stub-xclbinutil bug
    xclbin_side = os.path.splitext(bitfile)[0] + ".xclbin"
    if not os.path.exists(xclbin_side):
        import pynq as _pq, shutil
        d = os.path.join(os.path.dirname(_pq.__file__), "pl_server", "default.xclbin")
        if os.path.exists(d):
            shutil.copyfile(d, xclbin_side)

    print("[board] programming PL with %s ..." % bitfile)
    Overlay(bitfile)  # registers the device (CMA allocate) + programs the PL
    dma = MMIO(base, 0x1000)

    # pre-allocate max-size chunk buffers, reused across chunks/points
    ib = allocate(shape=(chunk * NS,), dtype=np.uint32)
    ob = allocate(shape=(chunk,), dtype=np.uint32)

    def run_chunk(syn_words, n):
        ib[: n * NS] = syn_words
        ob[:n] = 0
        ib.flush()
        dma.write(MM2S_DMACR, 0); dma.write(S2MM_DMACR, 0)
        dma.write(MM2S_DMACR, 1); dma.write(S2MM_DMACR, 1)
        dma.write(S2MM_DA, ob.physical_address & 0xFFFFFFFF)
        dma.write(S2MM_DA_MSB, (ob.physical_address >> 32) & 0xFFFFFFFF)
        dma.write(S2MM_LENGTH, n * 4)
        dma.write(MM2S_SA, ib.physical_address & 0xFFFFFFFF)
        dma.write(MM2S_SA_MSB, (ib.physical_address >> 32) & 0xFFFFFFFF)
        dma.write(MM2S_LENGTH, n * NS * 4)
        g = 0
        while g < 500_000_000:
            if (dma.read(MM2S_DMASR) & 2) and (dma.read(S2MM_DMASR) & 2):
                break
            g += 1
        ob.invalidate()
        if not ((dma.read(MM2S_DMASR) & 2) and (dma.read(S2MM_DMASR) & 2)):
            raise RuntimeError("DMA stall MM2S=0x%08x S2MM=0x%08x" % (dma.read(MM2S_DMASR), dma.read(S2MM_DMASR)))
        return (np.asarray(ob[:n]) >> 20) & obs_mask  # rtl_obs per shot

    print("  point        n        sw_ler       rtl_ler       |diff|     comb_ci   divergence  verdict")
    all_pass = True
    for prefix in prefixes:
        syn = np.fromfile(prefix + ".syn", dtype=np.uint32)
        ref = np.fromfile(prefix + ".ref", dtype=np.uint16)
        n = syn.size // NS
        assert ref.size == 2 * n, "%s.ref size mismatch (%d vs %d)" % (prefix, ref.size, 2 * n)
        true_obs = ref[0::2].astype(np.uint32)
        sw_obs = ref[1::2].astype(np.uint32)

        rtl_obs = np.empty(n, dtype=np.uint32)
        t0 = time.perf_counter()
        off = 0
        while off < n:
            m = min(chunk, n - off)
            rtl_obs[off:off + m] = run_chunk(syn[off * NS:(off + m) * NS], m)
            off += m
        dt = time.perf_counter() - t0

        rtl_err = int(np.count_nonzero(rtl_obs != true_obs))
        sw_err = int(np.count_nonzero(sw_obs != true_obs))
        div = int(np.count_nonzero(rtl_obs != sw_obs))
        rtl_ler = rtl_err / n
        sw_ler = sw_err / n
        comb = ci95(rtl_ler, n) + ci95(sw_ler, n)
        diff = abs(rtl_ler - sw_ler)
        within = diff <= comb + 1e-12
        if not within:
            all_pass = False
        print("  %-8s %9d  %.4e  %.4e  %.3e  %.3e  %6d/%-8d  %s"
              % (prefix, n, sw_ler, rtl_ler, diff, comb, div, n, "PASS" if within else "FAIL"))
        print("           (%.2f s, %.2f us/shot; rtl_err=%d sw_err=%d)" % (dt, 1e6 * dt / n, rtl_err, sw_err))

    del ib, ob
    print("\nAC-2 RESULT:", "PASS (RTL LER within CI of software golden at every point)"
          if all_pass else "FAIL (see rows)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
