#!/usr/bin/env python3
# Q6-03/Q6-08 — on-board decode-latency distribution (honest real-time profile).
#
# The "600 ns" headline is the worst case; a real-time claim also needs the distribution. This drives
# the co-sim Monte-Carlo stream through the decoder over AXI4-Lite (which exposes the PL-measured
# LATENCY register per decode) and reports min / mean / p50 / p90 / p99 / max latency per physical
# error rate p. Sub-threshold p (the real operating regime) vs supra-threshold shows how the tail
# grows with syndrome weight.
#
# Usage on the board (root + XRT env), against the AXI4-Lite bring-up bitstream:
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 uf_latstat.py uf_arty.bit cosim_d3.vec

import sys

import uf_pynq as U


def main(argv):
    import numpy as np

    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vec = next((a for a in argv[1:] if a.endswith(".vec")), "cosim_d3.vec")
    if not bitfile:
        print("usage: uf_latstat.py <axilite_design.bit> <cosim.vec>")
        return 2

    # detector count from the first data line
    dets = None
    with open(vec) as f:
        for l in f:
            if l and l[0] not in "#P" and " " in l:
                dets = len(l.split(" ", 1)[0])
                break

    dev = U.UfDecoder.from_overlay(bitfile)
    if not dev.probe():
        print("FAIL: IDCODE probe")
        return 1
    ns = lambda c: dev.latency_ns(c)

    print("[board] latency distribution over %s (clk @ %.0f MHz)" % (vec, dev.clk_hz / 1e6))
    print("   p       n     min    p50    mean    p90    p99    max     (max ns)")

    cur_p = None
    lats = []

    def flush():
        if cur_p is None or not lats:
            return
        a = np.asarray(lats)
        print(
            "  %.3f  %6d  %4d  %4d  %6.1f  %4d  %4d  %4d    %5d ns  %s"
            % (
                cur_p, a.size, a.min(), int(np.percentile(a, 50)), a.mean(),
                int(np.percentile(a, 90)), int(np.percentile(a, 99)), a.max(),
                ns(int(a.max())), "OVER 1us" if ns(int(a.max())) > 1000 else "",
            )
        )

    with open(vec) as f:
        for l in f:
            if not l or l[0] == "#":
                continue
            if l[0] == "P":
                flush()
                kv = dict(t.split("=", 1) for t in l[1:].split() if "=" in t)
                cur_p = float(kv["p"])
                lats = []
                continue
            if len(l) < dets + 2:
                continue
            bits = l[:dets]
            syn = sum(1 << j for j, c in enumerate(bits) if c == "1")
            _c, _o, lat = dev.decode(syn)
            lats.append(lat)
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
