#!/usr/bin/env python3
# Q7-02 board bring-up (PYNQ) — Python host driver for the partial relay-BP decoder (bp_axi_wrap),
# for the LAN/SSH path on the Arty Z7-20 (PYNQ-Z1 image). Python twin of the Verilator AXI TB
# (hw/tb_bp_axi.cpp): same AXI4-Lite register map, same protocol (write SYND0..2 -> pulse START ->
# poll DONE -> read CORR0..4 / OBS / LATENCY). Talks to the PL through a duck-typed MMIO backend so
# the SAME driver logic runs two ways:
#
#   * on the board -> BpDecoder.from_overlay("bp_arty.bit") uses pynq.MMIO (real hardware);
#   * on any host  -> BpDecoder(GoldenModel(vectors)) uses a software register model backed by the
#                     frozen golden vectors (`bp_dec_vectors.txt`), so the protocol is verifiable with
#                     no board. Run `python3 bp_pynq.py` to self-check all 65 golden decodes.
#
# Register map (mirrors bp_axi_wrap.sv exactly; byte offsets):
#   0x00 CTRL     [W]  bit0 START (self-clearing)
#   0x04 STATUS   [R]  bit0 BUSY, bit1 DONE (sticky), bit2 VALID (=valid_flag)
#   0x08 SYND0..0x10 SYND2 [RW] syndrome[71:0] (3 words, low 8 bits of SYND2)
#   0x14 CORR0..0x24 CORR4 [R]  correction[143:0] (5 words, low 16 bits of CORR4)
#   0x28 OBS      [R]  obs_flip[11:0]
#   0x2C LATENCY  [R]  last decode latency in cycles
#   0x30 IDCODE   [R]  0x4250_0001 ('BP', v1)

import os
import sys

# ---- AXI4-Lite register map (byte offsets) ----
BP_REG_CTRL = 0x00
BP_REG_STATUS = 0x04
BP_REG_SYND0 = 0x08  # SYND1=0x0C, SYND2=0x10
BP_REG_CORR0 = 0x14  # CORR1..4 = 0x18,0x1C,0x20,0x24
BP_REG_OBS = 0x28
BP_REG_LATENCY = 0x2C
BP_REG_IDCODE = 0x30

BP_CTRL_START = 1 << 0
BP_STATUS_BUSY = 1 << 0
BP_STATUS_DONE = 1 << 1
BP_STATUS_VALID = 1 << 2

BP_IDCODE_EXPECTED = 0x42500001

BP_N = 144  # data qubits (correction width)
BP_C = 72  # checks (syndrome width)
BP_OBS = 12  # logical observables


class BpDecoder:
    """Host driver for bp_axi_wrap over any MMIO backend exposing read(off)/write(off, val)."""

    def __init__(self, mmio, clk_hz=25_000_000, poll_limit=200_000):
        # 25 MHz: the PL clock the bp_arty bitstream is built at (WNS +4.57 ns, in-context Fmax ~28 MHz;
        # the 12/24 partial's OOC Fmax was 35.5 MHz). Pass clk_hz to match a differently-clocked build.
        self.mmio = mmio
        self.clk_hz = clk_hz
        self.poll_limit = poll_limit

    @classmethod
    def from_overlay(cls, bitfile, ip_name=None, **kw):
        """Load a PYNQ overlay and bind to the bp IP's MMIO. `ip_name` defaults to the first IP whose
        name contains 'bp'."""
        from pynq import Overlay  # imported lazily so the module also loads off-board

        ol = Overlay(bitfile)
        if ip_name is None:
            matches = [n for n in ol.ip_dict if "bp" in n.lower()]
            if not matches:
                raise RuntimeError("no bp* IP in overlay; available: %s" % list(ol.ip_dict))
            ip_name = matches[0]
        ip = getattr(ol, ip_name)
        dev = cls(ip.mmio, **kw)
        dev._overlay = ol  # keep a reference so the overlay isn't GC'd
        return dev

    def probe(self):
        """True iff IDCODE reads back the expected constant (PS<->PL link sanity)."""
        return self.mmio.read(BP_REG_IDCODE) == BP_IDCODE_EXPECTED

    def decode(self, syndrome):
        """Decode one syndrome (72-bit int, bit c = check c). Returns (correction_int, obs, valid,
        latency_cycles). Raises TimeoutError if DONE never asserts within poll_limit reads."""
        s = int(syndrome)
        self.mmio.write(BP_REG_SYND0 + 0, s & 0xFFFFFFFF)
        self.mmio.write(BP_REG_SYND0 + 4, (s >> 32) & 0xFFFFFFFF)
        self.mmio.write(BP_REG_SYND0 + 8, (s >> 64) & 0xFF)
        self.mmio.write(BP_REG_CTRL, BP_CTRL_START)
        for _ in range(self.poll_limit):
            status = self.mmio.read(BP_REG_STATUS)
            if status & BP_STATUS_DONE:
                valid = 1 if (status & BP_STATUS_VALID) else 0
                corr = 0
                for w in range(5):
                    corr |= (self.mmio.read(BP_REG_CORR0 + 4 * w) & 0xFFFFFFFF) << (32 * w)
                corr &= (1 << BP_N) - 1
                obs = self.mmio.read(BP_REG_OBS) & ((1 << BP_OBS) - 1)
                lat = self.mmio.read(BP_REG_LATENCY) & 0xFFFF
                return corr, obs, valid, lat
        raise TimeoutError("DONE not asserted for syndrome 0x%x" % s)

    def latency_ns(self, cycles):
        if self.clk_hz == 0:
            return 0
        return cycles * 1_000_000_000 // self.clk_hz


def _bits_to_int(bitstr):
    """Golden bit strings are MSB-at-index-0-is-bit-0 (char at position i is bit i)."""
    v = 0
    for i, ch in enumerate(bitstr):
        if ch == "1":
            v |= 1 << i
    return v


def load_vectors(path):
    """Parse bp_dec_vectors.txt into a list of (syndrome_int, corr_int, obs_int, valid) tuples."""
    tests = []
    cur = {}
    with open(path) as f:
        header_seen = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not header_seen:
                # header: 'T BP_N BP_C BP_OBS'
                header_seen = True
                continue
            tag = s[0]
            body = s[1:].strip()
            if tag == "s":
                cur = {"s": _bits_to_int(body)}
            elif tag == "h":
                cur["h"] = _bits_to_int(body)
            elif tag == "o":
                cur["o"] = _bits_to_int(body)
            elif tag == "v":
                cur["v"] = 1 if body.startswith("1") else 0
                tests.append((cur["s"], cur["h"], cur["o"], cur["v"]))
    return tests


class GoldenModel:
    """Software MMIO model backed by the golden vectors — the board-free protocol oracle.
    Keyed by syndrome integer (only the golden syndromes are exercised in the self-test)."""

    def __init__(self, tests):
        self.table = {s: (h, o, v) for (s, h, o, v) in tests}
        self.syndrome = 0
        self.status = 0
        self.corr = 0
        self.obs = 0
        self.lat = 0

    def write(self, off, val):
        if off == BP_REG_SYND0 + 0:
            self.syndrome = (self.syndrome & ~0xFFFFFFFF) | (val & 0xFFFFFFFF)
        elif off == BP_REG_SYND0 + 4:
            self.syndrome = (self.syndrome & ~(0xFFFFFFFF << 32)) | ((val & 0xFFFFFFFF) << 32)
        elif off == BP_REG_SYND0 + 8:
            self.syndrome = (self.syndrome & ~(0xFF << 64)) | ((val & 0xFF) << 64)
        elif off == BP_REG_CTRL and (val & BP_CTRL_START):
            h, o, v = self.table.get(self.syndrome, (0, 0, 0))
            self.corr, self.obs = h, o
            self.lat = 1086  # worst-case partial-unroll latency (informational)
            self.status = BP_STATUS_DONE | (BP_STATUS_VALID if v else 0)

    def read(self, off):
        if off == BP_REG_IDCODE:
            return BP_IDCODE_EXPECTED
        if off == BP_REG_STATUS:
            return self.status
        if off == BP_REG_OBS:
            return self.obs
        if off == BP_REG_LATENCY:
            return self.lat
        if BP_REG_CORR0 <= off < BP_REG_CORR0 + 20:
            w = (off - BP_REG_CORR0) // 4
            return (self.corr >> (32 * w)) & 0xFFFFFFFF
        return 0


def run_check(dev, tests, verbose=True):
    """Drive all golden syndromes through `dev`, compare correction/obs/valid. Returns fail count."""
    fails = 0
    max_lat = 0
    for (s, want_h, want_o, want_v) in tests:
        corr, obs, valid, lat = dev.decode(s)
        max_lat = max(max_lat, lat)
        if corr != want_h or obs != want_o or valid != want_v:
            fails += 1
            if verbose and fails <= 10:
                print(
                    "FAIL s=0x%018x: got {corr=0x%036x,obs=0x%03x,v=%d} want {corr=0x%036x,obs=0x%03x,v=%d}"
                    % (s, corr, obs, valid, want_h, want_o, want_v)
                )
    if verbose:
        print(
            "bp-pynq driver: decodes=%d  fails=%d  worst latency=%d clk = %d ns @ %.0f MHz"
            % (len(tests), fails, max_lat, dev.latency_ns(max_lat), dev.clk_hz / 1e6)
        )
    return fails


def _default_vec_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "bp_dec_vectors.txt")


def main(argv):
    vec_path = _default_vec_path()
    bitfile = None
    for a in argv[1:]:
        if a.endswith(".bit"):
            bitfile = a
        elif a.endswith(".txt"):
            vec_path = a

    tests = load_vectors(vec_path)
    if not tests:
        print("FAIL: no golden vectors parsed from %s" % vec_path)
        return 2

    if bitfile:
        print("[board] loading overlay %s" % bitfile)
        dev = BpDecoder.from_overlay(bitfile)
        if not dev.probe():
            print("FAIL: IDCODE probe (read 0x%08x)" % dev.mmio.read(BP_REG_IDCODE))
            return 1
        print("[board] IDCODE ok (0x%08x)" % BP_IDCODE_EXPECTED)
    else:
        print("[host] no .bit given -> software golden-model self-test (no board)")
        dev = BpDecoder(GoldenModel(tests))
        if not dev.probe():
            print("FAIL: IDCODE probe")
            return 1

    fails = run_check(dev, tests)
    if fails:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS (%d/%d decodes match golden; IDCODE ok)" % (len(tests), len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
