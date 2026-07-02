#!/usr/bin/env python3
# Q6-22 (streaming LER) — on-board finite-experiment logical-error-rate for the sliding-window
# streaming decoder (uf_stream_win / arty_z7_dma_win_bd; reuses the Q6-20 bitstream unchanged).
#
# Each experiment is one finite memory-Z run of `slices` rounds. Because correct per-experiment decode
# needs a fresh warm-up + empty residual, each experiment is ONE DMA transfer (the wrapper's per-frame
# re-arm resets the decoder at every tlast). The predicted logical is the XOR of every window's obs bit
# over the transfer (experiment rounds + a zero-drain that commits the tail). We compare the RTL
# streaming LER to the boundary-aware software SlidingWindowDecoder baseline (carried in the .vec P
# headers) within Monte-Carlo CI — measuring whether the interior-window + drain finite handling costs
# accuracy vs the software's real-boundary last window.
#
# Buffers are the same size for every experiment (constant R), so ib/ob are allocated ONCE and reused.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_dma_stream_ler.py uf_arty_dma_win.bit cosim_stream_ler_d3.vec

import math
import re
import sys
import time


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def round_word(bits):
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_ler_vec(path):
    """Parse the finite-experiment .vec: (meta, [(p, sw_rate, sw_ci, [(truth, [round_word,...]),...])])."""
    meta = {}
    blocks = []
    cur = None
    exp = None
    with open(path) as f:
        for l in f:
            l = l.rstrip("\n")
            if not l:
                continue
            if l[0] == "#":
                for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                    meta.setdefault(k, v)
                continue
            if l[0] == "P":
                kv = dict(t.split("=", 1) for t in l[1:].split() if "=" in t)
                cur = (float(kv["p"]), float(kv["sw_rate"]), float(kv["sw_ci"]), [])
                blocks.append(cur)
                exp = None
                continue
            if l[0] == "E":
                exp = (int(l.split()[1]), [])
                cur[3].append(exp)
                continue
            if exp is not None:
                exp[1].append(round_word(l.strip()))
    return meta, blocks


def main(argv):
    from pynq import Overlay, allocate
    import numpy as np

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vec = next((a for a in argv[1:] if a.endswith(".vec")), "cosim_stream_ler_d3.vec")
    if not bitfile:
        print("usage: uf_dma_stream_ler.py <design.bit> <cosim_stream_ler.vec>")
        return 2

    meta, blocks = load_ler_vec(vec)
    W = int(meta.get("W", 9))
    C = int(meta.get("C", 3))
    slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)

    # Every experiment is `slices` rounds + drain, padded to a window boundary (W + k*C). Constant size.
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    print(
        "[board] overlay %s  W=%d C=%d slices=%d  %d p-blocks  (per-exp: %d rounds -> %d windows)"
        % (bitfile, W, C, slices, len(blocks), total, nwin)
    )

    ol = Overlay(bitfile)
    dma_name = next(k2 for k2 in ol.ip_dict if "dma" in k2.lower())
    dma = getattr(ol, dma_name)

    ib = allocate(shape=(total,), dtype=np.uint32)  # reused across all experiments
    ob = allocate(shape=(nwin,), dtype=np.uint32)

    def decode_exp(rounds):
        ib[:] = 0
        ib[: len(rounds)] = np.asarray(rounds, dtype=np.uint32)  # rest is the zero-drain + pad
        ob[:] = 0
        ib.flush()
        dma.recvchannel.transfer(ob)
        dma.sendchannel.transfer(ib)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        ob.invalidate()
        pred = int(np.bitwise_xor.reduce((np.asarray(ob) >> 31) & 1)) & 1  # committed logical parity
        last_empty = int((np.asarray(ob)[-1] >> 30) & 1)
        return pred, last_empty

    # warm-up (first transfer pays driver setup)
    decode_exp(blocks[0][3][0][1])

    print("   p       rtl_rate     sw_rate     |diff|     comb_ci   drained   verdict")
    all_pass = True
    t0 = time.perf_counter()
    n_exp = 0
    for p, sw_rate, sw_ci, exps in blocks:
        errs = 0
        drained = 0
        for truth, rounds in exps:
            pred, last_empty = decode_exp(rounds)
            drained += last_empty
            errs += int(pred != truth)
        n = len(exps)
        n_exp += n
        rate = errs / n
        comb = ci95(rate, n) + sw_ci
        diff = abs(rate - sw_rate)
        within = diff <= comb + 1e-12
        gated = p <= 0.011 + 1e-12  # gate the sub-threshold (operating) regime
        if within:
            verdict = "PASS"
        elif gated:
            all_pass = False
            verdict = "FAIL"
        else:
            verdict = "info (supra-threshold)"
        print(
            "  %.3f   %.4e  %.4e  %.3e  %.3e   %4d/%-4d  %s"
            % (p, rate, sw_rate, diff, comb, drained, n, verdict)
        )
    dt = time.perf_counter() - t0

    print(
        "\non-board streaming LER: %d experiments, %.2f s (%.2f ms/exp)"
        % (n_exp, dt, 1e3 * dt / n_exp if n_exp else 0.0)
    )
    print("RESULT:", "PASS (RTL streaming LER within CI of software at the gated sub-threshold p)"
          if all_pass else "FAIL (see gated rows)")
    del ib, ob
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
