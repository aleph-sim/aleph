#!/usr/bin/env python3
"""P2 appliance v2-large: self-test driver for the aleph decoder AFI on an AWS F2 instance.

Talks to the CL by mmap'ing the PCI resource files, so it needs nothing but Python 3 and root: OCL BAR0
for registers and the per-word path, PCIS BAR4 for the batched path (register map and protocol in
hw/f2/design/aleph_bp_ocl.sv and aleph_pcis_ring.sv). Load the AFI first:

    sudo fpga-load-local-image -S 0 -I agfi-...
    sudo python3 bp_f2_ocl.py bp_circ_vectors.txt [--slot 0] [--max-mhz 150] [--pcis] [--repeat R]
                                                  [--window W]

It checks MAGIC and the bank geometry, measures the decoder clock against the shell's fixed 250 MHz
clk_main_a0, decodes every golden vector (R times over with --repeat) and compares {obs, valid_flag}
bit-exactly. The latency it prints is the core's own cycle count (result word bits [15:0]), so the PCIe
round trip is not in it; the HOST line is the end-to-end rate the host saw.

  default  OCL: five 32-bit PUSH stores per experiment, then POP peek + POP_ACK, one experiment at a time.
           On the first public image (no VERSION register) POP pops on read instead.
  --pcis   BAR4: each experiment is one 64-byte store into a slot through a write-combining mapping,
           results come back eight per 64-byte read from the result ring, up to W (default 64)
           experiments in flight.
Exit status 0 only on a full bit-exact pass.
"""

import ctypes
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
R_POP_ACK, R_PUSH_WORDS, R_RESULTS, R_PCIS_ACC, R_PCIS_DROP, R_VERSION, R_SLOTS = (
    0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C)
CTRL_RESET, CTRL_PCIS = 0x1, 0x4
EMPTY = 0xFFFFFFFF
LEGACY = 0xDEADBEEF          # VERSION on the first public image: no such register, POP pops on read
MAIN_MHZ = 250.0
RING = 0x10000               # result ring offset in BAR4
TAG_OFF = 28                 # byte offset of the tag in a 64-byte experiment slot


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


def find_device(slot):
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


def result_fields(r, obs_mask):
    return (r >> 20) & obs_mask, (r >> 19) & 1, r & 0xFFFF


class Ocl:
    """32-bit MMIO on OCL BAR0.

    Every access MUST be a single aligned 32-bit load/store. struct.pack_into / unpack_from on an mmap
    let Python copy the 4 bytes in pieces, and on an F2 instance that reaches the CL as several narrower
    AXI-Lite transactions: one PUSH became 3 FIFO words and one POP read popped more than one result
    (measured: 5 struct writes -> 3 results; 5 ctypes uint32 writes -> 1). ctypes.c_uint32 at the mapped
    address compiles to one 32-bit access.
    """

    def __init__(self, dev):
        path = dev + "/resource0"
        size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self._buf = (ctypes.c_char * 0x100).from_buffer(self.mm)
        self.base = ctypes.addressof(self._buf)

    def rd(self, off):
        return ctypes.c_uint32.from_address(self.base + off).value

    def wr(self, off, val):
        ctypes.c_uint32.from_address(self.base + off).value = val & 0xFFFFFFFF


class Pcis:
    """BAR4: experiment slots written through resource4_wc, the result ring read through resource4.

    Unlike OCL, these accesses may be split, merged or reordered by the CPU and the shell: the CL keeps
    a byte mask per slot and commits a slot only when all its bytes are in, and ring reads have no side
    effects. So plain mmap slice copies are fine here.
    """

    SIZE = 0x20000

    def __init__(self, dev):
        wc = dev + "/resource4_wc"
        if not os.path.exists(wc):
            raise SystemExit("%s missing: BAR4 is not prefetchable on this kernel/device" % wc)
        self.fds = [os.open(wc, os.O_RDWR | os.O_SYNC), os.open(dev + "/resource4", os.O_RDWR | os.O_SYNC)]
        self.wc = mmap.mmap(self.fds[0], self.SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.uc = mmap.mmap(self.fds[1], self.SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def put(self, slot, line):
        self.wc[64 * slot:64 * slot + 64] = line

    def row(self, r):
        return self.uc[RING + 64 * r:RING + 64 * r + 64]


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
    slot, max_mhz, use_pcis, repeat, window = 0, None, False, 1, 64
    it = iter(args)
    for a in it:
        if a == "--slot":
            slot = int(next(it))
        elif a == "--max-mhz":
            max_mhz = float(next(it))
        elif a == "--pcis":
            use_pcis = True
        elif a == "--repeat":
            repeat = int(next(it))
        elif a == "--window":
            window = int(next(it))

    C, OBS, tests = load_vectors(vecfile)
    ns = (C + 31) // 32
    dev = find_device(slot)
    ocl = Ocl(dev)
    print("[f2] device %s" % dev)

    magic = ocl.rd(R_MAGIC)
    if magic != MAGIC:
        raise SystemExit("MAGIC reads 0x%08x, expected 0x%08x: not the aleph AFI" % (magic, MAGIC))
    geom = ocl.rd(R_GEOM)
    version = ocl.rd(R_VERSION)
    legacy = version == LEGACY
    print("[f2] aleph decoder, banked %d/%d, built for %.3f MHz, interface %s"
          % (geom >> 16, geom & 0xFFFF, ocl.rd(R_KHZ) / 1000.0,
             "v1 (pop-on-read)" if legacy else "v%d" % version))
    if use_pcis and legacy:
        raise SystemExit("--pcis needs an image with interface v2 or later")

    # Measure the decoder clock against clk_main_a0 over ~0.5 s.
    c0, m0 = ocl.rd(R_CORE), ocl.rd(R_MAIN)
    time.sleep(0.5)
    c1, m1 = ocl.rd(R_CORE), ocl.rd(R_MAIN)
    mhz = ((c1 - c0) & 0xFFFFFFFF) / ((m1 - m0) & 0xFFFFFFFF) * MAIN_MHZ
    print("[f2] decoder clock measured: %.3f MHz" % mhz)
    if max_mhz is not None and mhz > max_mhz * 1.001:
        print("FAIL: decoder clock %.3f MHz is above %.3f MHz; nothing was decoded" % (mhz, max_mhz))
        return 1

    # Clean start: pulse the soft reset, which also empties both FIFOs, zeroes the counters and (v2)
    # clears the result ring. The mode bit is set with the reset and kept after it.
    mode = CTRL_PCIS if use_pcis else 0
    ocl.wr(R_CTRL, CTRL_RESET | mode)
    time.sleep(0.01)
    ocl.wr(R_CTRL, mode)
    wait_ready(ocl)
    if ocl.rd(R_POP) != EMPTY:
        raise SystemExit("output FIFO not empty after reset")

    obs_mask = (1 << OBS) - 1
    lines = []
    for synd, _, _ in tests:
        words = [0] * ns
        for c in range(C):
            if synd[c] == "1":
                words[c // 32] |= 1 << (c % 32)
        lines.append(words)
    n = len(tests) * repeat
    lats = [0] * n
    mism = 0

    def check(i, r):
        nonlocal mism
        _, want_obs, want_v = tests[i % len(tests)]
        got_obs, got_v, lats[i] = result_fields(r, obs_mask)
        if got_obs != (want_obs & obs_mask) or got_v != want_v:
            if mism < 20:
                print("  MISMATCH decode %d (test %d): got obs=0x%03x v=%d, want obs=0x%03x v=%d"
                      % (i, i % len(tests), got_obs, got_v, want_obs & obs_mask, want_v))
            mism += 1

    t0 = time.time()
    if use_pcis:
        slots = ocl.rd(R_SLOTS)
        window = max(1, min(window, slots))
        pcis = Pcis(dev)
        pay = [b"".join(struct.pack("<I", w) for w in words).ljust(64, b"\0") for words in lines]
        buf = bytearray(64)
        sent = done = 0
        last_progress = time.time()
        while done < n:
            while sent < n and sent - done < window:
                buf[:] = pay[sent % len(tests)]
                struct.pack_into("<I", buf, TAG_OFF, sent)   # tag = decode index (never 0xFFFFFFFF)
                pcis.put(sent % slots, buf)
                sent += 1
            s = done % slots
            row = pcis.row(s // 8)
            progressed = False
            for lane in range(s % 8, 8):
                if done >= sent:
                    break
                r, tag = struct.unpack_from("<II", row, 8 * lane)
                if tag != done:
                    break
                check(done, r)
                done += 1
                progressed = True
            if progressed:
                last_progress = time.time()
            elif time.time() - last_progress > 1.0:
                raise SystemExit("PCIS: no result for decode %d (slot %d) in 1 s: STATUS=0x%08x PCIS_ACC=%d "
                                 "PCIS_DROP=%d RESULTS=%d" % (done, s, ocl.rd(R_STATUS), ocl.rd(R_PCIS_ACC),
                                                             ocl.rd(R_PCIS_DROP), ocl.rd(R_RESULTS)))
    else:
        # One experiment at a time keeps the input FIFO far from its 512-word depth.
        for i in range(n):
            for w in lines[i % len(tests)]:
                ocl.wr(R_PUSH, w)
            deadline = time.time() + 1.0
            while True:
                r = ocl.rd(R_POP)
                if r != EMPTY:
                    break
                if time.time() > deadline:
                    raise SystemExit("no result for decode %d (STATUS=0x%08x)" % (i, ocl.rd(R_STATUS)))
            if not legacy:
                if (r >> 16) & 3 != i & 3:
                    raise SystemExit("decode %d: POP seq %d, expected %d" % (i, (r >> 16) & 3, i & 3))
                ocl.wr(R_POP_ACK, (r >> 16) & 3)
            check(i, r)
    host_s = time.time() - t0

    if not legacy:
        st, acc, drop, res, pw = (ocl.rd(R_STATUS), ocl.rd(R_PCIS_ACC), ocl.rd(R_PCIS_DROP),
                                  ocl.rd(R_RESULTS), ocl.rd(R_PUSH_WORDS))
        print("[f2] counters: PUSH_WORDS=%d RESULTS=%d PCIS_ACC=%d PCIS_DROP=%d STATUS=0x%08x"
              % (pw, res, acc, drop, st))
        if res != n or pw != n * ns or (st >> 8) & 7 or (use_pcis and (acc != n or drop != 0)):
            print("  counter or error-flag mismatch (want RESULTS=%d PUSH_WORDS=%d, no error flags)"
                  % (n, n * ns))
            mism += 1

    ok = mism == 0
    print("CORRECTNESS: %s (%d/%d decodes match golden on this F2 FPGA)"
          % ("PASS" if ok else "FAIL", n - mism, n))
    print("LATENCY: worst %d cycles, mean %.1f cycles = worst %.2f us at %.3f MHz (core-counted)"
          % (max(lats), sum(lats) / len(lats), max(lats) / mhz, mhz))
    print("HOST: %d decodes in %.3f s over %s (%.2f us/decode = %.0f decodes/s incl. PCIe; "
          "the core alone needs %.2f us/decode)"
          % (n, host_s, "PCIS BAR4, window %d" % window if use_pcis else "OCL MMIO",
             host_s / n * 1e6, n / host_s, sum(lats) / n / mhz))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
