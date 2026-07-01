#!/usr/bin/env python3
# Q6-08 (PYNQ) — Python host driver for the surface-code Union-Find decoder (uf_axi_wrap),
# for the LAN/SSH bring-up path on a PYNQ board (Arty Z7-20 running the PYNQ-Z1 image).
#
# This is the Python twin of the bare-metal C driver (`uf_decoder.c`): same AXI4-Lite register
# map, same protocol (write SYNDROME -> pulse START -> poll DONE -> read CORRECTION/OBS/LATENCY).
# It talks to the PL through a duck-typed MMIO backend so the SAME driver logic runs two ways:
#
#   * on the board  -> `UfDecoder.from_overlay("design_1.bit")` uses `pynq.MMIO` (real hardware);
#   * on any host   -> `UfDecoder(GoldenModel(...))` uses a software register model backed by the
#                      frozen golden table, so the protocol is verifiable with no board (mirrors
#                      `hw/sw/test/` — run `python3 uf_pynq.py` to self-check all 256 syndromes).
#
# Register map (mirrors uf_axi_wrap.sv exactly):
#   0x00 CTRL       [W]  bit0 START (self-clearing)
#   0x04 STATUS     [R]  bit0 BUSY, bit1 DONE (sticky, cleared on next START), bit2 OBS_FLIP
#   0x08 SYNDROME   [RW] syndrome bits
#   0x0C CORRECTION [R]  correction[M-1:0]
#   0x10 LATENCY    [R]  last decode latency in cycles
#   0x14 IDCODE     [R]  0x5546_0003

import os
import sys

# ---- AXI4-Lite register map (byte offsets) ----
UF_REG_CTRL = 0x00
UF_REG_STATUS = 0x04
UF_REG_SYNDROME = 0x08
UF_REG_CORRECTION = 0x0C
UF_REG_LATENCY = 0x10
UF_REG_IDCODE = 0x14

UF_CTRL_START = 1 << 0
UF_STATUS_BUSY = 1 << 0
UF_STATUS_DONE = 1 << 1
UF_STATUS_OBS = 1 << 2

UF_IDCODE_EXPECTED = 0x55460003

# correction occupies the low 18 bits of the golden pack; OBS_FLIP is bit 18 (see uf_surface_golden.mem).
CORR_MASK = 0x3FFFF
OBS_SHIFT = 18


class UfDecoder:
    """Host driver for uf_axi_wrap over any MMIO backend exposing read(off)/write(off, val)."""

    def __init__(self, mmio, clk_hz=50_000_000, poll_limit=100_000):
        # clk_hz defaults to 50 MHz: the timing-closed PL clock for d=3 on xc7z020 (OOC Fmax was
        # 58.7 MHz worst-case; FCLK is set to 50 MHz in the block design to close in-context).
        self.mmio = mmio
        self.clk_hz = clk_hz
        self.poll_limit = poll_limit

    @classmethod
    def from_overlay(cls, bitfile, ip_name=None, **kw):
        """Load a PYNQ overlay and bind to the uf_axi_wrap IP's MMIO. `ip_name` defaults to the
        first IP whose name contains 'uf_axi_wrap'."""
        from pynq import Overlay  # imported lazily so the module also loads off-board

        ol = Overlay(bitfile)
        if ip_name is None:
            # The decoder IP's block-design cell may be named uf_axi_wrap / uf_axi_top / uf_0 depending
            # on how the BD instantiated it, so match any "uf"-prefixed IP (excludes ps7/interconnect).
            matches = [n for n in ol.ip_dict if "uf" in n.lower()]
            if not matches:
                raise RuntimeError(
                    "no uf* IP in overlay; available: %s" % list(ol.ip_dict)
                )
            ip_name = matches[0]
        ip = getattr(ol, ip_name)
        dev = cls(ip.mmio, **kw)
        dev._overlay = ol  # keep a reference so the overlay isn't GC'd
        return dev

    def probe(self):
        """True iff IDCODE reads back the expected constant (PS<->PL link sanity)."""
        return self.mmio.read(UF_REG_IDCODE) == UF_IDCODE_EXPECTED

    def decode(self, syndrome):
        """Decode one syndrome. Returns (correction, obs_flip, latency_cycles).
        Raises TimeoutError if DONE never asserts within poll_limit reads."""
        self.mmio.write(UF_REG_SYNDROME, int(syndrome))
        self.mmio.write(UF_REG_CTRL, UF_CTRL_START)
        for _ in range(self.poll_limit):
            status = self.mmio.read(UF_REG_STATUS)
            if status & UF_STATUS_DONE:
                obs = 1 if (status & UF_STATUS_OBS) else 0
                corr = self.mmio.read(UF_REG_CORRECTION)
                lat = self.mmio.read(UF_REG_LATENCY) & 0xFFFF
                return corr, obs, lat
        raise TimeoutError("DONE not asserted for syndrome 0x%x" % syndrome)

    def latency_ns(self, cycles):
        if self.clk_hz == 0:
            return 0
        return cycles * 1_000_000_000 // self.clk_hz


def load_golden(path):
    """Parse uf_surface_golden.mem: one hex value {obs<<18 | corr} per non-comment line."""
    packed = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            packed.append(int(s, 16))
    return packed


class GoldenModel:
    """Software MMIO model backed by the golden table — the board-free protocol oracle.
    Mirrors hw/sw/test/uf_mmio_model.c so the driver can be verified on a laptop."""

    def __init__(self, packed, latency_cycles=47):
        self.packed = packed
        self.lat = latency_cycles
        self.syndrome = 0
        self.status = 0
        self.correction = 0

    def write(self, off, val):
        if off == UF_REG_SYNDROME:
            self.syndrome = val & 0xFFFFFFFF
        elif off == UF_REG_CTRL and (val & UF_CTRL_START):
            pk = self.packed[self.syndrome]  # one-shot combinational model: result ready immediately
            self.correction = pk & CORR_MASK
            obs = (pk >> OBS_SHIFT) & 1
            self.status = UF_STATUS_DONE | (UF_STATUS_OBS if obs else 0)

    def read(self, off):
        if off == UF_REG_IDCODE:
            return UF_IDCODE_EXPECTED
        if off == UF_REG_STATUS:
            return self.status
        if off == UF_REG_SYNDROME:
            return self.syndrome
        if off == UF_REG_CORRECTION:
            return self.correction
        if off == UF_REG_LATENCY:
            return self.lat
        return 0


def run_golden_check(dev, packed, verbose=True):
    """Drive all syndromes through `dev`, compare to `packed`. Returns fail count."""
    n = len(packed)
    fails = 0
    max_lat = 0
    for s in range(n):
        corr, obs, lat = dev.decode(s)
        max_lat = max(max_lat, lat)
        want_corr = packed[s] & CORR_MASK
        want_obs = (packed[s] >> OBS_SHIFT) & 1
        if corr != want_corr or obs != want_obs:
            fails += 1
            if verbose and fails <= 10:
                print(
                    "FAIL s=%d: got {obs=%d,corr=0x%05x} want {obs=%d,corr=0x%05x}"
                    % (s, obs, corr, want_obs, want_corr)
                )
    if verbose:
        print(
            "uf-pynq driver: syndromes=%d  fails=%d  worst latency=%d clk = %d ns @ %.0f MHz"
            % (n, fails, max_lat, dev.latency_ns(max_lat), dev.clk_hz / 1e6)
        )
    return fails


def _default_golden_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "uf_surface_golden.mem")


def main(argv):
    golden_path = _default_golden_path()
    bitfile = None
    for a in argv[1:]:
        if a.endswith(".bit"):
            bitfile = a
        elif a.endswith(".mem"):
            golden_path = a

    packed = load_golden(golden_path)
    if len(packed) != 256:
        print("FAIL: golden has %d entries, expected 256" % len(packed))
        return 2

    if bitfile:
        print("[board] loading overlay %s" % bitfile)
        dev = UfDecoder.from_overlay(bitfile)
        if not dev.probe():
            print("FAIL: IDCODE probe (read 0x%08x)" % dev.mmio.read(UF_REG_IDCODE))
            return 1
        print("[board] IDCODE ok (0x%08x)" % UF_IDCODE_EXPECTED)
    else:
        print("[host] no .bit given -> software golden-model self-test (no board)")
        dev = UfDecoder(GoldenModel(packed))
        if not dev.probe():
            print("FAIL: IDCODE probe")
            return 1

    fails = run_golden_check(dev, packed)
    if fails:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS (%d/%d syndromes match golden; IDCODE ok)" % (len(packed), len(packed)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
