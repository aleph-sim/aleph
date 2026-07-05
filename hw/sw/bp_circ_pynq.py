#!/usr/bin/env python3
# Q7-02 M5-followup (PYNQ) — host driver for the CIRCUIT-LEVEL M2 relay-BP decoder behind the WIDE
# AXI4-Lite wrapper (bp_axi_wrap_wide). Python twin of tb_bp_axi_wide.cpp: generic in the graph size —
# it reads BP_N / BP_C from the golden-vector header and derives the number of 32-bit syndrome /
# correction words. Talks to the PL through a duck-typed MMIO backend so the SAME logic runs on the
# board (pynq.MMIO) or off-board (a GoldenModel backed by the vectors) for a board-free self-test.
#
# Register map (mirrors bp_axi_wrap_wide.sv; NS = ceil(BP_C/32), NC = ceil(BP_N/32)):
#   0x00 CTRL [W] START ; 0x04 STATUS [R] busy/done/valid ; 0x08 LATENCY [R] 32-bit ;
#   0x0C OBS [R] obs_flip ; 0x10 IDCODE [R] 0x4250_0002 ;
#   0x40 SYND0.. [RW] NS words ; 0x80 CORR0.. [R] NC words

import os
import sys

REG_CTRL = 0x00
REG_STATUS = 0x04
REG_LAT = 0x08
REG_OBS = 0x0C
REG_ID = 0x10
SYND_BASE = 0x40
CORR_BASE = 0x80

CTRL_START = 1 << 0
CTRL_EARLY = 1 << 1        # sticky: 1 = stop at first syndrome-valid decision
STATUS_DONE = 1 << 1
STATUS_VALID = 1 << 2
IDCODE_EXPECTED = 0x42500002


class BpCircDecoder:
    """Host driver for bp_axi_wrap_wide over any MMIO backend exposing read(off)/write(off, val)."""

    def __init__(self, mmio, n_checks, n_vars, n_obs, clk_hz=50_000_000, poll_limit=2_000_000,
                 early=False):
        self.mmio = mmio
        self.n_checks = n_checks
        self.n_vars = n_vars
        self.n_obs = n_obs
        self.ns = (n_checks + 31) // 32
        self.nc = (n_vars + 31) // 32
        self.clk_hz = clk_hz
        self.poll_limit = poll_limit
        self.early = early     # drive CTRL bit1 so the core stops at the first valid decision

    @classmethod
    def from_overlay(cls, bitfile, n_checks, n_vars, n_obs, ip_name=None, **kw):
        from pynq import Overlay  # imported lazily so the module also loads off-board

        ol = Overlay(bitfile)
        if ip_name is None:
            matches = [n for n in ol.ip_dict if "bp" in n.lower()]
            if not matches:
                raise RuntimeError("no bp* IP in overlay; available: %s" % list(ol.ip_dict))
            ip_name = matches[0]
        ip = getattr(ol, ip_name)
        dev = cls(ip.mmio, n_checks, n_vars, n_obs, **kw)
        dev._overlay = ol
        return dev

    def probe(self):
        return self.mmio.read(REG_ID) == IDCODE_EXPECTED

    def decode(self, syndrome_bits):
        """Decode one syndrome (list of 0/1 length n_checks, or an int). Returns
        (correction_int, obs, valid, latency_cycles)."""
        s = syndrome_bits
        if not isinstance(s, int):
            v = 0
            for i, b in enumerate(s):
                if b:
                    v |= 1 << i
            s = v
        for w in range(self.ns):
            self.mmio.write(SYND_BASE + 4 * w, (s >> (32 * w)) & 0xFFFFFFFF)
        self.mmio.write(REG_CTRL, CTRL_START | (CTRL_EARLY if self.early else 0))
        for _ in range(self.poll_limit):
            status = self.mmio.read(REG_STATUS)
            if status & STATUS_DONE:
                valid = 1 if (status & STATUS_VALID) else 0
                corr = 0
                for w in range(self.nc):
                    corr |= (self.mmio.read(CORR_BASE + 4 * w) & 0xFFFFFFFF) << (32 * w)
                corr &= (1 << self.n_vars) - 1
                obs = self.mmio.read(REG_OBS) & ((1 << self.n_obs) - 1)
                lat = self.mmio.read(REG_LAT) & 0xFFFFFFFF
                return corr, obs, valid, lat
        raise TimeoutError("DONE not asserted")

    def latency_ns(self, cycles):
        return 0 if self.clk_hz == 0 else cycles * 1_000_000_000 // self.clk_hz


def _bits_to_int(bitstr):
    v = 0
    for i, ch in enumerate(bitstr):
        if ch == "1":
            v |= 1 << i
    return v


def load_vectors(path):
    """Parse the bp_circ_vectors.txt: header 'T BP_N BP_C BP_OBS', then s/h/o/v per test."""
    tests = []
    n_vars = n_checks = n_obs = 0
    cur = {}
    with open(path) as f:
        header_seen = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not header_seen:
                parts = s.split()
                _, n_vars, n_checks, n_obs = (int(x) for x in parts[:4])
                header_seen = True
                continue
            tag, body = s[0], s[1:].strip()
            if tag == "s":
                cur = {"s": _bits_to_int(body)}
            elif tag == "h":
                cur["h"] = _bits_to_int(body)
            elif tag == "o":
                cur["o"] = _bits_to_int(body)
            elif tag == "v":
                cur["v"] = 1 if body.startswith("1") else 0
                tests.append((cur["s"], cur["h"], cur["o"], cur["v"]))
    return tests, n_checks, n_vars, n_obs


class GoldenModel:
    """Software MMIO model backed by the golden vectors (board-free protocol oracle)."""

    def __init__(self, tests, latency=69984):
        self.table = {s: (h, o, v) for (s, h, o, v) in tests}
        self.syndrome = 0
        self.status = 0
        self.corr = 0
        self.obs = 0
        self.lat = latency

    def write(self, off, val):
        if SYND_BASE <= off < SYND_BASE + 4 * 16:
            w = (off - SYND_BASE) // 4
            self.syndrome = (self.syndrome & ~(0xFFFFFFFF << (32 * w))) | ((val & 0xFFFFFFFF) << (32 * w))
        elif off == REG_CTRL and (val & CTRL_START):
            h, o, v = self.table.get(self.syndrome, (0, 0, 0))
            self.corr, self.obs = h, o
            self.status = STATUS_DONE | (STATUS_VALID if v else 0)

    def read(self, off):
        if off == REG_ID:
            return IDCODE_EXPECTED
        if off == REG_STATUS:
            return self.status
        if off == REG_OBS:
            return self.obs
        if off == REG_LAT:
            return self.lat
        if CORR_BASE <= off < CORR_BASE + 4 * 64:
            w = (off - CORR_BASE) // 4
            return (self.corr >> (32 * w)) & 0xFFFFFFFF
        return 0


def run_check(dev, tests, verbose=True):
    fails, lats = 0, []
    for (s, want_h, want_o, want_v) in tests:
        corr, obs, valid, lat = dev.decode(s)
        lats.append(lat)
        if corr != want_h or obs != want_o or valid != want_v:
            fails += 1
            if verbose and fails <= 10:
                print("FAIL s=0x%036x: obs got %03x want %03x, v %d/%d, corr %s"
                      % (s, obs, want_o, valid, want_v, "ok" if corr == want_h else "MISMATCH"))
    if verbose:
        mode = "early-exit" if dev.early else "full-schedule"
        sl = sorted(lats)
        n = len(sl)
        pct = lambda q: sl[min(int(n * q), n - 1)]
        mean = sum(sl) // n
        us = lambda c: dev.latency_ns(c) / 1000.0
        print("bp-circ driver [%s]: decodes=%d fails=%d @ %.0f MHz" % (mode, n, fails, dev.clk_hz / 1e6))
        print("  latency cycles: min=%d p50=%d mean=%d p99=%d max=%d"
              % (sl[0], pct(0.50), mean, pct(0.99), sl[-1]))
        print("  latency    ms : min=%.3f p50=%.3f mean=%.3f p99=%.3f max=%.3f"
              % (us(sl[0]) / 1000, us(pct(0.50)) / 1000, us(mean) / 1000,
                 us(pct(0.99)) / 1000, us(sl[-1]) / 1000))
    return fails


def _default_vec_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "bp_circ_vectors.txt")


def main(argv):
    vec_path = _default_vec_path()
    bitfile = None
    early = False
    for a in argv[1:]:
        if a.endswith(".bit"):
            bitfile = a
        elif a.endswith(".txt"):
            vec_path = a
        elif a == "early":
            early = True

    tests, n_checks, n_vars, n_obs = load_vectors(vec_path)
    if not tests:
        print("FAIL: no golden vectors parsed from %s" % vec_path)
        return 2
    print("[info] circuit graph: %d checks / %d vars / %d obs (%d synd words, %d corr words)"
          % (n_checks, n_vars, n_obs, (n_checks + 31) // 32, (n_vars + 31) // 32))

    if bitfile:
        print("[board] loading overlay %s" % bitfile)
        dev = BpCircDecoder.from_overlay(bitfile, n_checks, n_vars, n_obs, clk_hz=50_000_000, early=early)
        if not dev.probe():
            print("FAIL: IDCODE probe (read 0x%08x)" % dev.mmio.read(REG_ID))
            return 1
        print("[board] IDCODE ok (0x%08x)" % IDCODE_EXPECTED)
    else:
        print("[host] no .bit given -> software golden-model self-test (no board)")
        dev = BpCircDecoder(GoldenModel(tests), n_checks, n_vars, n_obs, early=early)
        if not dev.probe():
            print("FAIL: IDCODE probe")
            return 1

    fails = run_check(dev, tests)
    if fails:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS (%d/%d circuit-level decodes match golden; IDCODE ok)" % (len(tests), len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
