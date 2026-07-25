#!/usr/bin/env python3
# Q7-06 AC-2 — KV260 on-silicon 10^6-shot LER campaign for the batched banked relay-BP decoder.
#
# Streams a large set of REAL DEM shots (binary <prefix>.syn = NS little-endian u32 syndrome words per
# shot) through the batched AXI-DMA decoder overlay, collects the RTL's per-shot observable-flip
# prediction, and compares logical-error rates against the software FixedRelayBp golden carried in
# <prefix>.ref (aleph_qec::refvec v2: true_obs, sw_obs, valid+iters per shot). Reports, per
# physical-error point:
#   RTL LER   = mean(rtl_obs != true_obs)   -- the silicon decoder's logical-error rate
#   SW  LER   = mean(sw_obs  != true_obs)   -- the software golden's rate (the reference)
#   |diff|    within combined 95% CI ?      -- AC-2 acceptance (RTL LER within CI of software golden)
#   divergence= mean(rtl_obs != sw_obs)     -- direct bit-exactness check (should be ~0)
#   valid_mismatch = count(rtl_valid != sw_valid)  -- Q7-07 gate, must be 0
# across the >=3 points given on the command line.
#
# The syndromes stream as one contiguous DMA input (no per-shot Python repacking); results come back one
# u16-in-u32 status word per shot. Chunked to keep each DMA buffer within the CMA pool.
#
# Usage (root, pynq venv + XRT, from a dir with the FULL-SCHEDULE .bit + the <prefix>.syn/.ref files;
# <prefix>.ref must be v2 -- generate with `qec_q7_bp_graph -- silvectors ...`):
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
        return np.asarray(ob[:n]).copy()  # full status word: [31:20]=obs, [19]=valid, [15:0]=cycles

    print("  point        n        sw_ler       rtl_ler       |diff|     comb_ci   divergence  verdict")
    all_pass = True
    for prefix in prefixes:
        syn = np.fromfile(prefix + ".syn", dtype=np.uint32)
        # .ref v2 (aleph_qec::refvec): header [magic, version, words_per_shot, 0] then
        # [true_obs, sw_obs, meta] per shot, meta = (valid << 15) | iters.
        ref = np.fromfile(prefix + ".ref", dtype="<u2")
        n = syn.size // NS
        if ref.size < 4 or ref[0] != 0xA1E7:
            raise SystemExit(
                "%s.ref is a legacy v1 file (no header). Regenerate it: "
                "qec_q7_bp_graph -- silvectors <rounds> <p> <n> <seed> %s <decoder_p>"
                % (prefix, prefix))
        if ref[1] != 2 or ref[2] != 3:
            raise SystemExit("%s.ref: unsupported version/width %d/%d" % (prefix, ref[1], ref[2]))
        body = ref[4:]
        assert body.size == 3 * n, "%s.ref size mismatch (%d vs %d)" % (prefix, body.size, 3 * n)
        true_obs = body[0::3].astype(np.uint32)
        sw_obs = body[1::3].astype(np.uint32)
        sw_valid = (body[2::3] >> 15).astype(np.uint32)

        status = np.empty(n, dtype=np.uint32)
        t0 = time.perf_counter()
        off = 0
        while off < n:
            m = min(chunk, n - off)
            status[off:off + m] = run_chunk(syn[off * NS:(off + m) * NS], m)
            off += m
        dt = time.perf_counter() - t0

        rtl_obs = (status >> 20) & obs_mask
        rtl_valid = (status >> 19) & 1
        rtl_cycles = status & 0xFFFF
        valid_mismatch = int(np.count_nonzero(rtl_valid != sw_valid))
        rtl_nonconv = int(np.count_nonzero(rtl_valid == 0))

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
        print("           (valid: rtl_nonconv=%d (%.4f%%), sw_nonconv=%d, mismatch=%d; "
              "cycles mean=%.1f max=%d)"
              % (rtl_nonconv, 100.0 * rtl_nonconv / n,
                 int(np.count_nonzero(sw_valid == 0)), valid_mismatch,
                 float(rtl_cycles.mean()), int(rtl_cycles.max())))
        if valid_mismatch:
            all_pass = False

    del ib, ob
    print("\nAC-2 RESULT:", "PASS (RTL LER within CI of software golden; valid_flag matches at every point)"
          if all_pass else "FAIL (see rows)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
