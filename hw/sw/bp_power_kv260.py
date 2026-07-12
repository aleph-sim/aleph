#!/usr/bin/env python3
# Q7-05 — KV260 power / energy-per-decode for the shipped M8 banked relay-BP overlay.
#
# The Kria K26 SOM carries an INA260 (hwmon name `ina260_u14`) on the 5 V SOM input rail, so it
# reports *SOM-total* power (PS + PL + DDR + fan), not PL-only. We therefore measure a *delta*: PL
# programmed-but-idle vs PL under sustained decode load. The delta cancels the static PS/DDR baseline
# and isolates the dynamic power drawn while decoding. A PS-poll-only control (MMIO reads, no START)
# bounds the share of that delta owed to the host/AXI loop rather than the PL datapath itself.
#
# Power is derived as in1_input(mV) * curr1_input(mA) — the current channel is 1 mA (~5 mW) resolution,
# finer than power1_input's 10 mW quantum. Sampling is interleaved with the decode loop in the main
# thread (a batch of decodes, then one rail read) so no GIL-starved sampler thread can miss the load.
#
# Energy per decode is duty-corrected: with the PL busy t_hw = cycles/f_clk per decode but the host
# loop stretching wall-per-decode to T/n, the average dynamic power measured over the window is
# dP = P_load - P_idle = duty * (P_active - P_idle), so (P_active - P_idle) * t_hw = dP * (T/n). Hence
# E_decode = dP * wall_per_decode regardless of duty.
#
# Usage (as root, pynq venv + XRT, from a dir with bp_m8.bit + bp_circ_pynq.py + bp_circ_vectors.txt):
#   XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 bp_power_kv260.py \
#       bp_m8.bit bp_circ_vectors.txt [--base 0xA0000000] [--clk 133332000] [--idcode 0x42500003] \
#       [--idle 8] [--load 20]
import glob
import importlib.util
import os
import sys
import time

REG_ID = 0x10


def _load_driver():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "bp_circ_pynq.py")
    if not os.path.exists(path):
        path = "bp_circ_pynq.py"
    spec = importlib.util.spec_from_file_location("bp_circ_pynq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_ina():
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(os.path.join(h, "name")).read().strip().startswith("ina260"):
                return h
        except OSError:
            continue
    raise RuntimeError("no ina260 hwmon found")


def _read_power_w(vin, iin):
    # SOM 5 V rail power from the finer current/voltage channels (mV * mA -> uW -> W).
    return int(open(vin).read()) * int(open(iin).read()) / 1e6


def _summ(ps):
    xs = sorted(ps)
    n = len(xs)
    return {
        "n": n,
        "mean": sum(xs) / n,
        "p50": xs[n // 2],
        "min": xs[0],
        "max": xs[-1],
    }


def _sample_idle(vin, iin, dur):
    ps = []
    t0 = time.time()
    while time.time() - t0 < dur:
        ps.append(_read_power_w(vin, iin))
        time.sleep(0.02)
    return _summ(ps)


def _hammer(dev, syndromes, vin, iin, dur, batch=40):
    """Run decodes back-to-back for `dur` seconds, sampling the rail once per batch.
    Returns (n_decodes, total_cycles, wall_s, power_summary)."""
    ps = []
    n = 0
    cyc = 0
    j = 0
    m = len(syndromes)
    t0 = time.time()
    while time.time() - t0 < dur:
        for _ in range(batch):
            _, _, _, lat = dev.decode(syndromes[j % m])
            j += 1
            n += 1
            cyc += lat
        ps.append(_read_power_w(vin, iin))
    wall = time.time() - t0
    return n, cyc, wall, _summ(ps)


def _ps_poll_ctrl(mmio, vin, iin, dur, batch=2000):
    """Control: hammer MMIO reads (no START, PL quiescent) to bound the host/AXI power share."""
    ps = []
    n = 0
    t0 = time.time()
    while time.time() - t0 < dur:
        for _ in range(batch):
            mmio.read(REG_ID)
            n += 1
        ps.append(_read_power_w(vin, iin))
    return n, time.time() - t0, _summ(ps)


def main(argv):
    bitfile = vecfile = None
    base = 0xA0000000
    clk = 133_332_000
    idcode = 0x42500003
    idle_s = 8.0
    load_s = 20.0
    it = iter(argv[1:])
    for a in it:
        if a.endswith(".bit"):
            bitfile = a
        elif a.endswith(".txt"):
            vecfile = a
        elif a == "--base":
            base = int(next(it), 0)
        elif a == "--clk":
            clk = int(float(next(it)))
        elif a == "--idcode":
            idcode = int(next(it), 0)
        elif a == "--idle":
            idle_s = float(next(it))
        elif a == "--load":
            load_s = float(next(it))
    if not bitfile or not vecfile:
        print("usage: bp_power_kv260.py <bit> <vectors.txt> [--base A] [--clk H] [--idcode I]"
              " [--idle S] [--load S]")
        return 2

    from pynq import MMIO, Bitstream

    bpc = _load_driver()
    tests, n_checks, n_vars, n_obs = bpc.load_vectors(vecfile)
    syndromes = [t[0] for t in tests]
    hwmon = _find_ina()
    vin = os.path.join(hwmon, "in1_input")
    iin = os.path.join(hwmon, "curr1_input")
    print("[info] %d checks / %d vars / %d obs ; %d syndromes ; rail=%s"
          % (n_checks, n_vars, n_obs, len(syndromes), os.path.basename(hwmon)))

    print("[board] programming PL with %s ..." % bitfile)
    Bitstream(bitfile).download()
    mmio = MMIO(base, 0x1000)
    idc = mmio.read(REG_ID)
    if idc != idcode:
        print("FAIL: IDCODE 0x%08x (expected 0x%08x)" % (idc, idcode))
        return 1
    print("[board] IDCODE ok (0x%08x) at base 0x%08x, clk %.3f MHz" % (idc, base, clk / 1e6))

    # (a) PL programmed, quiescent.
    print("\n[idle ] sampling %.0fs (PL programmed, no decodes) ..." % idle_s)
    idle = _sample_idle(vin, iin, idle_s)

    # (b) full-schedule hammer.
    print("[full ] hammering full-schedule decodes %.0fs ..." % load_s)
    devf = bpc.BpCircDecoder(mmio, n_checks, n_vars, n_obs, clk_hz=clk, early=False)
    nf, cf, wf, pf = _hammer(devf, syndromes, vin, iin, load_s)

    # (c) early-exit hammer.
    print("[early] hammering early-exit decodes %.0fs ..." % load_s)
    deve = bpc.BpCircDecoder(mmio, n_checks, n_vars, n_obs, clk_hz=clk, early=True)
    ne, ce, we, pe = _hammer(deve, syndromes, vin, iin, load_s)

    # (d) PS-poll-only control.
    print("[psctl] MMIO-read control (no START) %.0fs ..." % (load_s / 2))
    n_ctl, w_ctl, pc = _ps_poll_ctrl(mmio, vin, iin, load_s / 2)

    def report(name, n, cyc, wall, p):
        thr = n / wall
        wall_per = wall / n
        mean_cyc = cyc / n
        t_hw = mean_cyc / clk
        duty = t_hw / wall_per
        dP = p["mean"] - idle["mean"]
        e_decode = dP * wall_per  # duty-corrected dynamic energy per decode (J)
        print("  [%s] decodes=%d  thr=%.0f/s  wall/dec=%.1f us  mean_hw=%.0f cyc (%.2f us)  duty=%.1f%%"
              % (name, n, thr, wall_per * 1e6, mean_cyc, t_hw * 1e6, duty * 100))
        print("        P_load: mean=%.3f W p50=%.3f min=%.3f max=%.3f  |  dP=+%.0f mW over idle"
              % (p["mean"], p["p50"], p["min"], p["max"], dP * 1000))
        print("        energy/decode (dynamic, SOM-total delta) = %.1f uJ" % (e_decode * 1e6))
        return dP, e_decode

    print("\n===== Q7-05 power (SOM-total INA260 @ %s) =====" % os.path.basename(hwmon))
    print("  [idle ] P: mean=%.3f W p50=%.3f min=%.3f max=%.3f  (n=%d, PL quiescent)"
          % (idle["mean"], idle["p50"], idle["min"], idle["max"], idle["n"]))
    dPf, ef = report("full ", nf, cf, wf, pf)
    dPe, ee = report("early", ne, ce, we, pe)
    dPc = pc["mean"] - idle["mean"]
    print("  [psctl] P: mean=%.3f W  |  dP=+%.0f mW over idle  (%d reads/s, PL quiescent host/AXI share)"
          % (pc["mean"], dPc * 1000, int(n_ctl / w_ctl)))
    print("\n  PL-attributable dynamic (dP_load - dP_psctl): full=+%.0f mW  early=+%.0f mW"
          % ((dPf - dPc) * 1000, (dPe - dPc) * 1000))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
