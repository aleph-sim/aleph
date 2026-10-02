#!/usr/bin/env python3
"""P2 appliance v2-large: self-test driver for the aleph decoder AFI on an AWS F2 instance.

Talks to the CL through OCL BAR0 (register map in hw/f2/design/aleph_bp_ocl.sv) by mmap'ing the PCI
resource file, so it needs nothing but Python 3 and root. Load the AFI first:

    sudo fpga-load-local-image -S 0 -I agfi-...
    sudo python3 bp_f2_ocl.py bp_circ_vectors.txt [--slot 0] [--max-mhz 125]

It checks MAGIC and the bank geometry, measures the decoder clock against the shell's fixed 250 MHz
clk_main_a0, decodes every golden vector and compares {obs, valid_flag} bit-exactly. The latency it
prints is the core's own cycle count (result word bits [15:0]), so the PCIe round trip is not in it.
Exit status 0 only on a full bit-exact pass.
"""

import glob
import mmap
import os
import struct
import sys
import time

VENDOR, DEVICE = 0x1D0F, 0xF0A1
MAGIC = 0xA1E9B0F2
R_MAGIC, R_GEOM, R_STATUS, R_CTRL, R_PUSH, R_POP, R_CORE, R_MAIN, R_KHZ = (
    0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20)
EMPTY = 0xFFFFFFFF
MAIN_MHZ = 250.0


def load_vectors(path):
    """Header 'T N C OBS', then per test 's' / 'h' / 'o' / 'v' lines (same file the co-sim gates on)."""
    T = C = OBS = 0
    tests, cur, header = [], {}, False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            if not header:
                T, _, C, OBS = (int(x) for x in line.split()[:4])
                header = True
                continue
            tag, body = line[0], line[1:].strip()
            if tag == "s":
                cur = {"s": body}
            elif tag == "o":
                cur["o"] = body
            elif tag == "v":
                obs = sum(1 << i for i, ch in enumerate(cur.get("o", "")) if ch == "1")
                tests.append((cur["s"], obs, 1 if body.startswith("1") else 0))
    if len(tests) != T:
        raise SystemExit("vector file says %d tests, parsed %d" % (T, len(tests)))
    return C, OBS, tests


def find_bar0(slot):
    devs = []
    for d in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            v = int(open(d + "/vendor").read(), 16)
            p = int(open(d + "/device").read(), 16)
        except OSError:
            continue
        if v == VENDOR and p == DEVICE:
            devs.append(d)
    if not devs:
        raise SystemExit("no %04x:%04x device. Is the aleph AFI loaded? "
                         "(sudo fpga-load-local-image -S %d -I agfi-...)" % (VENDOR, DEVICE, slot))
    if slot >= len(devs):
        raise SystemExit("slot %d requested, found %d aleph device(s): %s" % (slot, len(devs), devs))
    return devs[slot]


class Ocl:
    def __init__(self, dev):
        path = dev + "/resource0"
        size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def rd(self, off):
        return struct.unpack_from("<I", self.mm, off)[0]

    def wr(self, off, val):
        struct.pack_into("<I", self.mm, off, val & 0xFFFFFFFF)


def wait_ready(ocl, timeout=2.0):
    t0 = time.time()
    while True:
        st = ocl.rd(R_STATUS)
        # locked, core out of reset, both FIFOs out of reset
        if (st >> 4) & 1 and not (st >> 5) & 1 and not (st >> 6) & 3:
            return st
        if time.time() - t0 > timeout:
            raise SystemExit("decoder never became ready, STATUS=0x%08x" % st)


def main(argv):
    args = [a for a in argv[1:]]
    vecfile = next((a for a in args if a.endswith(".txt")), "bp_circ_vectors.txt")
    slot, max_mhz = 0, None
    it = iter(args)
    for a in it:
        if a == "--slot":
            slot = int(next(it))
        elif a == "--max-mhz":
            max_mhz = float(next(it))

    C, OBS, tests = load_vectors(vecfile)
    ns = (C + 31) // 32
    dev = find_bar0(slot)
    ocl = Ocl(dev)
    print("[f2] device %s" % dev)

    magic = ocl.rd(R_MAGIC)
    if magic != MAGIC:
        raise SystemExit("MAGIC reads 0x%08x, expected 0x%08x: not the aleph AFI" % (magic, MAGIC))
    geom = ocl.rd(R_GEOM)
    print("[f2] aleph decoder, banked %d/%d, built for %.3f MHz"
          % (geom >> 16, geom & 0xFFFF, ocl.rd(R_KHZ) / 1000.0))

    # Measure the decoder clock against clk_main_a0 over ~0.5 s.
    c0, m0 = ocl.rd(R_CORE), ocl.rd(R_MAIN)
    time.sleep(0.5)
    c1, m1 = ocl.rd(R_CORE), ocl.rd(R_MAIN)
    mhz = ((c1 - c0) & 0xFFFFFFFF) / ((m1 - m0) & 0xFFFFFFFF) * MAIN_MHZ
    print("[f2] decoder clock measured: %.3f MHz" % mhz)
    if max_mhz is not None and mhz > max_mhz * 1.001:
        print("FAIL: decoder clock %.3f MHz is above %.3f MHz; nothing was decoded" % (mhz, max_mhz))
        return 1

    # Clean start: pulse the soft reset, which also empties both FIFOs.
    ocl.wr(R_CTRL, 1)
    time.sleep(0.01)
    ocl.wr(R_CTRL, 0)
    wait_ready(ocl)
    if ocl.rd(R_POP) != EMPTY:
        raise SystemExit("output FIFO not empty after reset")

    obs_mask = (1 << OBS) - 1
    mism, lats = 0, []
    t0 = time.time()
    # One experiment at a time keeps the input FIFO far from its 512-word depth regardless of batch size.
    for k, (synd, want_obs, want_v) in enumerate(tests):
        words = [0] * ns
        for c in range(C):
            if synd[c] == "1":
                words[c // 32] |= 1 << (c % 32)
        for w in words:
            ocl.wr(R_PUSH, w)
        deadline = time.time() + 1.0
        while True:
            r = ocl.rd(R_POP)
            if r != EMPTY:
                break
            if time.time() > deadline:
                raise SystemExit("no result for test %d (STATUS=0x%08x)" % (k, ocl.rd(R_STATUS)))
        lats.append(r & 0xFFFF)
        got_obs, got_v = (r >> 20) & obs_mask, (r >> 19) & 1
        if got_obs != (want_obs & obs_mask) or got_v != want_v:
            print("  MISMATCH test %d: got obs=0x%03x v=%d, want obs=0x%03x v=%d"
                  % (k, got_obs, got_v, want_obs & obs_mask, want_v))
            mism += 1
    host_s = time.time() - t0

    ok = mism == 0
    print("CORRECTNESS: %s (%d/%d decodes match golden on this F2 FPGA)"
          % ("PASS" if ok else "FAIL", len(tests) - mism, len(tests)))
    print("LATENCY: worst %d cycles, mean %.1f cycles = worst %.2f us at %.3f MHz (core-counted)"
          % (max(lats), sum(lats) / len(lats), max(lats) / mhz, mhz))
    print("HOST: %d decodes in %.3f s over OCL MMIO (%.1f us/decode incl. PCIe round trips)"
          % (len(tests), host_s, host_s / len(tests) * 1e6))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
