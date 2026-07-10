#!/usr/bin/env python3
# Q7-02 M6 — KV260 (Zynq UltraScale+) board runner for the circuit-level relay-BP decoder.
#
# Why this exists instead of just using bp_circ_pynq.py's Overlay path: on the Kria-PYNQ 3.0.1 image,
# pynq.Overlay(bit) dies with `FileNotFoundError: .../t.xclbin` for a design that has NO PL DRAM banks.
# The image ships a *stub* /usr/bin/xclbinutil that wraps unwrapped/xclbinutil and `exit 0`s regardless of
# its return code; the empty MEM_TOPOLOGY makes the real tool fail, the failure is masked, t.xclbin is
# never produced, and Overlay raises. Bypass: program the PL with pynq.Bitstream(bit).download() (which
# skips the metadata/xclbin path) and MMIO the wide-wrap IP at its assigned base directly.
#
# Usage (as root, with the pynq venv + XRT, from a dir holding the .bit/.hwh + vectors):
#   XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 bp_circ_kv260.py \
#       bp_kv260_circ.bit bp_circ_vectors.txt [--base 0xA0000000] [--clk 100e6]
#
# The IP base comes from the .hwh (C_BASEADDR); for the M6 build with M_AXI_HPM0_FPD it is 0xA0000000.
import importlib.util
import os
import sys


def _load_driver():
    # bp_circ_pynq.py lives next to this file on the repo; on the board it is copied alongside.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "bp_circ_pynq.py")
    if not os.path.exists(path):
        path = "bp_circ_pynq.py"  # fall back to cwd (flat board dir)
    spec = importlib.util.spec_from_file_location("bp_circ_pynq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    bitfile = vecfile = None
    base = 0xA0000000
    clk = 100_000_000
    idcode_override = None
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
            idcode_override = int(next(it), 0)
    if not bitfile or not vecfile:
        print("usage: bp_circ_kv260.py <bit> <vectors.txt> [--base 0xADDR] [--clk 100e6] [--idcode 0x42500003]")
        return 2

    from pynq import Bitstream, MMIO

    bpc = _load_driver()
    if idcode_override is not None:
        bpc.IDCODE_EXPECTED = idcode_override
    tests, n_checks, n_vars, n_obs = bpc.load_vectors(vecfile)
    if not tests:
        print("FAIL: no golden vectors parsed from %s" % vecfile)
        return 2
    print("[info] %d checks / %d vars / %d obs ; %d tests" % (n_checks, n_vars, n_obs, len(tests)))

    print("[board] programming PL with %s ..." % bitfile)
    Bitstream(bitfile).download()
    mmio = MMIO(base, 0x1000)
    idc = mmio.read(bpc.REG_ID)
    if idc != bpc.IDCODE_EXPECTED:
        print("FAIL: IDCODE 0x%08x (expected 0x%08x) at base 0x%08x" % (idc, bpc.IDCODE_EXPECTED, base))
        return 1
    print("[board] IDCODE ok (0x%08x) at base 0x%08x" % (idc, base))

    print("\n=== full-schedule (worst-case) ===")
    fails = bpc.run_check(bpc.BpCircDecoder(mmio, n_checks, n_vars, n_obs, clk_hz=clk), tests)

    print("\n=== early-exit (first-valid, average-case lever) ===")
    fails_e = bpc.run_check(
        bpc.BpCircDecoder(mmio, n_checks, n_vars, n_obs, clk_hz=clk, early=True), tests)

    ok = (fails == 0 and fails_e == 0)
    print("\nRESULT: %s (%d/%d circuit-level decodes match golden on KV260 silicon)"
          % ("PASS" if ok else "FAIL", len(tests) - max(fails, fails_e), len(tests)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
