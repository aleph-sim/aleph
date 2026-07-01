#!/usr/bin/env python3
# Q6-03 (throughput) — on-board AXI-DMA driver for the streaming decoder (uf_stream / arty_z7_dma_bd).
#
# Streams a whole batch of syndromes from DDR through the decoder and back via AXI DMA, so the PS is
# NOT in the per-decode loop: measured throughput is decoder-bound (one decode per the core's latency
# + stream overhead), unlike the AXI4-Lite-polled uf_hil.py (~7k decodes/s, host-bound). Reuses the
# co-sim Monte-Carlo stream (cosim_d3.vec) so it also re-checks the on-board logical-error rate.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 uf_dma.py uf_arty_dma.bit cosim_d3.vec
#
# Result word from the decoder: bit31 = obs_flip, bits[30:0] = correction (low bits).

import math
import sys
import time


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def syndrome_int(bits):
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_blocks(vec_path):
    """Parse the .vec into [(p, sw_rate, sw_ci, [syndrome_int...], [truth...]), ...] and dets."""
    blocks = []
    cur = None
    dets = None
    with open(vec_path) as f:
        for l in f:
            if not l or l[0] == "#":
                continue
            if l[0] == "P":
                kv = dict(t.split("=", 1) for t in l[1:].split() if "=" in t)
                cur = [float(kv["p"]), float(kv["sw_rate"]), float(kv["sw_ci"]), [], []]
                blocks.append(cur)
                continue
            if " " not in l:
                continue
            s, obs = l.split(" ", 1)
            if dets is None:
                dets = len(s)
            if cur is None:
                continue
            cur[3].append(syndrome_int(s))
            cur[4].append(1 if obs.strip() == "1" else 0)
    return blocks, dets


def main(argv):
    from pynq import Overlay, allocate
    import numpy as np

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vec = next((a for a in argv[1:] if a.endswith(".vec")), "cosim_d3.vec")
    if not bitfile:
        print("usage: uf_dma.py <design.bit> <cosim.vec>")
        return 2

    blocks, dets = load_blocks(vec)
    print("[board] loading overlay %s (dets=%d, %d blocks)" % (bitfile, dets, len(blocks)))
    ol = Overlay(bitfile)
    dma_name = next(k for k in ol.ip_dict if "dma" in k.lower())
    dma = getattr(ol, dma_name)

    def run_batch(syndromes):
        n = len(syndromes)
        ib = allocate(shape=(n,), dtype=np.uint32)
        ob = allocate(shape=(n,), dtype=np.uint32)
        ib[:] = np.asarray(syndromes, dtype=np.uint32)
        ob[:] = 0
        ib.flush()
        t0 = time.perf_counter()
        dma.recvchannel.transfer(ob)   # arm S2MM first
        dma.sendchannel.transfer(ib)   # start MM2S
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        dt = time.perf_counter() - t0
        ob.invalidate()
        obs = (np.asarray(ob) >> 31) & 1
        res = obs.copy()
        del ib, ob
        return res, dt

    # warm-up (first transfer pays allocation/driver setup) using the first block's head.
    warm = blocks[0][3][: min(256, len(blocks[0][3]))]
    run_batch(warm)

    print("   p       rtl_rate     sw_rate     |diff|     comb_ci    verdict")
    all_pass = True
    total_shots = 0
    total_time = 0.0
    for p, sw_rate, sw_ci, syns, truth in blocks:
        obs, dt = run_batch(syns)
        n = len(syns)
        errs = int((obs != np.asarray(truth, dtype=np.uint32)).sum())
        rate = errs / n
        comb = ci95(rate, n) + sw_ci
        diff = abs(rate - sw_rate)
        within = diff <= comb + 1e-12
        gated = p <= 0.011 + 1e-12
        if within:
            verdict = "PASS"
        elif gated:
            all_pass = False
            verdict = "FAIL"
        else:
            verdict = "info (supra-threshold)"
        print(
            "  %.3f   %.4e  %.4e  %.3e  %.3e   %s"
            % (p, rate, sw_rate, diff, comb, verdict)
        )
        total_shots += n
        total_time += dt

    thru = total_shots / total_time if total_time > 0 else 0.0
    print(
        "\non-board DMA: shots=%d  wall=%.3f s  throughput=%.0f decodes/s (%.3f us/decode)"
        % (total_shots, total_time, thru, 1e6 / thru if thru else 0.0)
    )
    print("RESULT:", "PASS (on-board LER within CI at all gated p)" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
