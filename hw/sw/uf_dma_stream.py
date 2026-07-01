#!/usr/bin/env python3
# Q6-20 (on silicon) — on-board AXI-DMA driver for the SLIDING-WINDOW STREAMING decoder
# (uf_stream_win / arty_z7_dma_win_bd). Streams a continuous run of measurement ROUNDS from DDR through
# the decoder and collects one result word per committed window, so the PS is not in the per-round loop
# and the measured window rate is decoder-bound.
#
# Unlike the block driver (uf_dma.py: one syndrome in / one result out), the streaming decoder consumes
# one round per MM2S beat and emits one word per committed window (every C rounds). Correctness on the
# board is VALIDITY (tie-break- and boundary-independent, matching the #399 Verilator proof): after the
# stream is drained (a tail of zero rounds pushes every defect through the commit region) the residual
# must clear. Each result word carries that flag:  bit31 = committed logical parity (obs),
# bit30 = residual_empty,  bits[15:0] = the window's core decode latency (cycles).
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_dma_stream.py uf_arty_dma_win.bit cosim_stream_d3.vec

import re
import sys
import time


def round_word(bits):
    """Pack a round's detector-bit string into the low bits of a 32-bit stream word."""
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_vec(path):
    """Parse the streaming .vec: returns (meta dict, [(p, [round_word, ...]), ...])."""
    meta = {}
    blocks = []
    cur = None
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
                cur = (float(kv["p"]), [])
                blocks.append(cur)
                continue
            if cur is not None:
                cur[1].append(round_word(l.strip()))
    return meta, blocks


def main(argv):
    from pynq import Overlay, allocate
    import numpy as np

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vec = next((a for a in argv[1:] if a.endswith(".vec")), "cosim_stream_d3.vec")
    if not bitfile:
        print("usage: uf_dma_stream.py <design.bit> <cosim_stream.vec>")
        return 2

    meta, blocks = load_vec(vec)
    W = int(meta.get("W", 9))
    C = int(meta.get("C", 3))
    dpr = int(meta.get("dpr", 4))
    drain = max(2 * W, 16)  # zero-round tail: push every in-flight defect through the commit region
    print(
        "[board] overlay %s  W=%d C=%d dpr=%d  %d p-blocks (drain=%d rounds/block)"
        % (bitfile, W, C, dpr, len(blocks), drain)
    )

    ol = Overlay(bitfile)
    dma_name = next(k for k in ol.ip_dict if "dma" in k.lower())
    dma = getattr(ol, dma_name)

    def nwindows(total_rounds):
        # Warm-up loads W rounds -> 1 window; every C further rounds -> 1 more. The stream must land on a
        # window boundary (total = W + k*C); pad up so the final round completes (and tlast-tags) a window.
        k = max(1, -(-(total_rounds - W) // C))  # ceil
        total = W + k * C
        return total, 1 + k

    def run_stream(rounds):
        """Feed `rounds` (a list of round words) + drain; return (per-window result words, wall seconds)."""
        padded = list(rounds) + [0] * drain
        total, nwin = nwindows(len(padded))
        padded += [0] * (total - len(padded))  # pad to a window boundary; DMA sets tlast on the last beat
        ib = allocate(shape=(total,), dtype=np.uint32)
        ob = allocate(shape=(nwin,), dtype=np.uint32)
        ib[:] = np.asarray(padded, dtype=np.uint32)
        ob[:] = 0
        ib.flush()
        t0 = time.perf_counter()
        dma.recvchannel.transfer(ob)  # arm S2MM (nwin result words) first
        dma.sendchannel.transfer(ib)  # start MM2S (total round beats)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        dt = time.perf_counter() - t0
        ob.invalidate()
        out = np.asarray(ob).copy()
        del ib, ob
        return out, dt, nwin

    # warm-up (first transfer pays allocation/driver setup)
    run_stream(blocks[0][1][: min(256, len(blocks[0][1]))])

    # ---- validity: every p-block's stream must fully drain (last window residual-empty) ----
    print("   p       windows   nonempty_mid   last_resid_empty   verdict")
    all_pass = True
    all_rounds = []
    for p, rounds in blocks:
        out, _dt, nwin = run_stream(rounds)
        all_rounds.extend(rounds)
        obs = (out >> 31) & 1
        rese = (out >> 30) & 1
        last_empty = int(rese[-1]) if len(rese) else 0
        nonempty_mid = int((rese == 0).sum())
        ok = last_empty == 1 and len(out) == nwin
        if not ok:
            all_pass = False
        print(
            "  %.3f    %6d    %10d    %14d     %s"
            % (p, len(out), nonempty_mid, last_empty, "PASS" if ok else "FAIL")
        )
        _ = obs  # committed-logical parity available per window; validity is the on-board gate

    # ---- sustained throughput: one big streamed run (windows/s = decoder-bound window rate) ----
    out, dt, nwin = run_stream(all_rounds)
    thru = nwin / dt if dt > 0 else 0.0
    us_win = 1e6 / thru if thru else 0.0
    budget_us = float(C)  # one window per commit period = C rounds ≈ C µs at ~1 µs/round
    print(
        "\non-board streaming DMA: windows=%d  wall=%.4f s  %.0f windows/s (%.3f µs/window)"
        % (nwin, dt, thru, us_win)
    )
    print(
        "real-time vs %.0f µs commit budget: %s (%.1fx headroom)"
        % (budget_us, "YES" if us_win < budget_us else "NO", budget_us / us_win if us_win else 0.0)
    )
    print("RESULT:", "PASS (every stream drains on silicon; window rate meets the commit budget)"
          if all_pass and us_win < budget_us else "FAIL")
    return 0 if (all_pass and us_win < budget_us) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
