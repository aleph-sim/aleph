#!/usr/bin/env python3
# Q6-03 / Q6-08 — on-board Hardware-in-the-Loop: replay the co-simulation Monte-Carlo syndrome stream
# (hw/cosim_d3.vec, produced by `qec_q6_cosim.rs` from the SAME detector-error model the RTL matching
# graph was generated from) through the REAL decoder on the board, over AXI4-Lite, and:
#   * accumulate the on-board RTL logical-error rate per physical error rate p and compare it to the
#     software Union-Find baseline within Monte-Carlo CI (the on-silicon version of the Q6-21 board-free
#     co-sim / tb_uf_cosim.cpp);
#   * measure end-to-end decode throughput (shots/s) over the AXI4-Lite control plane.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 uf_hil.py uf_arty.bit cosim_d3.vec
#
# Vec format (see tb_uf_cosim.cpp): `P p=.. shots=.. sw_rate=.. sw_ci=..` opens a block; data lines are
# "<dets-char string> <actual_obs>" where detector j = char j and actual_obs is the ground-truth
# observable flip. A shot is a logical error iff the decoder's obs_flip != actual_obs.

import math
import sys
import time

import uf_pynq as U


def ci95(p, n):
    # 95% normal-approximation half-width (matches ci95 in tb_uf_cosim.cpp / LogicalErrorResult).
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def syndrome_int(bits):
    # detector j = char j (LSB-first), matching the RTL syndrome[j] port packing.
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def run_vec(dev, vec_path, dets, gate_p=0.011):
    print("   p       rtl_rate     sw_rate     |diff|     comb_ci   max_latency    verdict")
    all_pass = True
    total_shots = 0
    t_decode = 0.0
    max_lat = 0

    blk = None  # (p, sw_rate, sw_ci); accumulate rtl_errs, shots, invalid

    def finish(blk, rtl_errs, shots, invalid, blk_lat):
        nonlocal all_pass
        if blk is None or shots == 0:
            return
        p, sw_rate, sw_ci = blk
        rate = rtl_errs / shots
        comb = ci95(rate, shots) + sw_ci
        diff = abs(rate - sw_rate)
        within = diff <= comb + 1e-12
        gated = p <= gate_p + 1e-12
        if invalid:
            all_pass = False
            verdict = "FAIL (INVALID!)"
        elif within:
            verdict = "PASS"
        elif gated:
            all_pass = False
            verdict = "FAIL"
        else:
            verdict = "info (supra-threshold)"
        print(
            "  %.3f   %.4e  %.4e  %.3e  %.3e  %4d clk=%4d ns   %s"
            % (p, rate, sw_rate, diff, comb, blk_lat, dev.latency_ns(blk_lat), verdict)
        )

    rtl_errs = shots = invalid = blk_lat = 0
    with open(vec_path) as f:
        for l in f:
            if not l or l[0] == "#":
                continue
            if l[0] == "P":
                finish(blk, rtl_errs, shots, invalid, blk_lat)
                total_shots += shots
                kv = dict(
                    tok.split("=", 1) for tok in l[1:].split() if "=" in tok
                )
                blk = (float(kv["p"]), float(kv["sw_rate"]), float(kv["sw_ci"]))
                rtl_errs = shots = invalid = blk_lat = 0
                continue
            if len(l) < dets + 2:
                continue
            bits = l[:dets]
            truth = 1 if l[dets + 1] == "1" else 0
            t0 = time.perf_counter()
            corr, obs, lat = dev.decode(syndrome_int(bits))
            t_decode += time.perf_counter() - t0
            if lat > blk_lat:
                blk_lat = lat
            if lat > max_lat:
                max_lat = lat
            if obs != truth:
                rtl_errs += 1
            shots += 1
    finish(blk, rtl_errs, shots, invalid, blk_lat)
    total_shots += shots

    thru = total_shots / t_decode if t_decode > 0 else 0.0
    print(
        "\non-board HiL: shots=%d  max latency=%d clk = %d ns @ %.0f MHz  "
        "throughput=%.0f decodes/s (%.1f us/decode, AXI4-Lite polled)"
        % (
            total_shots,
            max_lat,
            dev.latency_ns(max_lat),
            dev.clk_hz / 1e6,
            thru,
            1e6 / thru if thru else 0.0,
        )
    )
    return all_pass


def main(argv):
    bitfile = next((a for a in argv[1:] if a.endswith(".bit")), None)
    vec = next((a for a in argv[1:] if a.endswith(".vec")), "cosim_d3.vec")
    if not bitfile:
        print("usage: uf_hil.py <design.bit> <cosim.vec>")
        return 2
    # detectors = characters before the first space on the first data line.
    dets = None
    with open(vec) as f:
        for l in f:
            if l and l[0] not in "#P" and " " in l:
                dets = len(l.split(" ", 1)[0])
                break
    if not dets:
        print("FAIL: could not determine detector count from %s" % vec)
        return 2

    print("[board] loading overlay %s (dets=%d)" % (bitfile, dets))
    dev = U.UfDecoder.from_overlay(bitfile)
    if not dev.probe():
        print("FAIL: IDCODE probe")
        return 1

    ok = run_vec(dev, vec, dets)
    print("RESULT:", "PASS (on-board LER within MC CI at all gated p)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
