#!/usr/bin/env python3
# Q6-25 — FEED-FORWARD on real silicon: the decode result of a logical measurement conditions the next
# logical operation (the teleportation byproduct Pauli). This is the FT primitive Q6-24 was the
# substrate for — the decode no longer just scores a passive frame, it *steers a conditional quantum
# gate* in a genuine Clifford teleportation gadget that runs here on the board, with the two byproduct
# measurements resolved in real time by the sliding-window decoder on the Arty Z7-20.
#
#   |ψ⟩ ─┐   Bell-measure → two logical outcomes (m_x, m_z)
#        │        each is a CODE-PROTECTED measurement: raw = m ⊕ e (logical meas error e)
#        │        the REAL Arty decoder resolves ê from the syndrome block → corrected byproduct bit
#        └──► X^{b_x} Z^{b_z} on the teleported qubit   ← conditional gate driven by the on-silicon decode
#
# We contrast, per trial (paired, same Bell outcomes):
#   * ON  — byproduct = decoder-corrected outcome (raw ⊕ ê)  → teleportation succeeds at ~(1−LER);
#   * OFF — byproduct = raw undecoded outcome        (raw)   → teleportation corrupted at the raw rate.
# The ON−OFF fidelity gap is the on-silicon proof the real-time decode steers the computation — a thing
# that cannot exist in open-loop trace-replay.
#
# Honest scope: for Clifford inputs teleportation success reduces to "was the byproduct decode right"
# (= composed memory-LER); the new content is the conditional-operation control flow executed by a real
# stabilizer gadget + the ON/OFF contrast, not a new error mechanism.
#
# The teleportation gadget is a genuine Aaronson–Gottesman CHP stabilizer tableau (§ "Improved
# Simulation of Stabilizer Circuits", 2004): real prep, Bell pair, Bell measurement (genuine outcome
# randomness), conditional X/Z byproducts, and a real verification measurement.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_feedforward.py uf_arty_dma_win.bit cosim_ff_d3.vec [--trials 8000]
# Off-board gadget self-check (perfect decoder, no board): validates ON=100%, OFF<100%:
#   python3 uf_qubit_feedforward.py --selfcheck cosim_ff_d3.vec

import argparse
import math
import random
import re
import sys
import time


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def round_word(bits):
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_ff_vec(path):
    """Parse the feed-forward .vec -> (meta, [(input, e_x, e_z, [bx words], [bz words]), ...])."""
    meta = {}
    trials = []
    slices = None
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(k, v)
    slices = int(meta.get("slices", 18))
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        if not l or l[0] == "#" or l[0] == "P":
            i += 1
            continue
        if l[0] == "T":
            _, inp, ex, ez = l.split()
            bx = [round_word(lines[i + 1 + k]) for k in range(slices)]
            bz = [round_word(lines[i + 1 + slices + k]) for k in range(slices)]
            trials.append((int(inp), int(ex), int(ez), bx, bz))
            i += 1 + 2 * slices
            continue
        i += 1
    return meta, trials


# --------------------------------------------------------------------------------------------------
# Aaronson–Gottesman CHP stabilizer tableau (n qubits). Rows 0..n-1 destabilizers, n..2n-1 stabilizers,
# row 2n scratch. Each row: x[n], z[n], phase r.
# --------------------------------------------------------------------------------------------------
class Chp:
    def __init__(self, n):
        self.n = n
        self.x = [[0] * n for _ in range(2 * n + 1)]
        self.z = [[0] * n for _ in range(2 * n + 1)]
        self.r = [0] * (2 * n + 1)
        for i in range(n):
            self.x[i][i] = 1        # destabilizers X_i
            self.z[n + i][i] = 1    # stabilizers   Z_i

    def copy(self):
        c = Chp.__new__(Chp)
        c.n = self.n
        c.x = [row[:] for row in self.x]
        c.z = [row[:] for row in self.z]
        c.r = self.r[:]
        return c

    def cnot(self, a, b):
        for i in range(2 * self.n):
            self.r[i] ^= self.x[i][a] & self.z[i][b] & (self.x[i][b] ^ self.z[i][a] ^ 1)
            self.x[i][b] ^= self.x[i][a]
            self.z[i][a] ^= self.z[i][b]

    def h(self, a):
        for i in range(2 * self.n):
            self.r[i] ^= self.x[i][a] & self.z[i][a]
            self.x[i][a], self.z[i][a] = self.z[i][a], self.x[i][a]

    def x_gate(self, a):  # X conjugation flips sign where the row has a Z component on a
        for i in range(2 * self.n):
            self.r[i] ^= self.z[i][a]

    def z_gate(self, a):  # Z conjugation flips sign where the row has an X component on a
        for i in range(2 * self.n):
            self.r[i] ^= self.x[i][a]

    @staticmethod
    def _g(x1, z1, x2, z2):
        if x1 == 0 and z1 == 0:
            return 0
        if x1 == 1 and z1 == 1:
            return z2 - x2
        if x1 == 1 and z1 == 0:
            return z2 * (2 * x2 - 1)
        return x2 * (1 - 2 * z2)  # x1==0, z1==1

    def _rowsum(self, h, i):
        s = 2 * self.r[h] + 2 * self.r[i]
        xi, zi, xh, zh = self.x[i], self.z[i], self.x[h], self.z[h]
        for j in range(self.n):
            s += self._g(xi[j], zi[j], xh[j], zh[j])
        self.r[h] = 1 if (s % 4) != 0 else 0  # (s mod 4) is 0 or 2
        for j in range(self.n):
            xh[j] ^= xi[j]
            zh[j] ^= zi[j]

    def measure(self, a, rng):
        n = self.n
        p = next((i for i in range(n, 2 * n) if self.x[i][a]), None)
        if p is not None:  # random outcome
            for i in range(2 * n):
                if i != p and self.x[i][a]:
                    self._rowsum(i, p)
            self.x[p - n] = self.x[p][:]
            self.z[p - n] = self.z[p][:]
            self.r[p - n] = self.r[p]
            self.x[p] = [0] * n
            self.z[p] = [0] * n
            self.z[p][a] = 1
            self.r[p] = rng.getrandbits(1)
            return self.r[p]
        # deterministic outcome via scratch row 2n
        self.x[2 * n] = [0] * n
        self.z[2 * n] = [0] * n
        self.r[2 * n] = 0
        for i in range(n):
            if self.x[i][a]:
                self._rowsum(2 * n, i + n)
        return self.r[2 * n]


# Single-qubit stabilizer inputs. |0>,|1> are Z-eigenstates (verify in Z); |+>,|-> are X-eigenstates
# (H then verify in Z).  H|1> = |->, so |-> preps as X then H.
INPUT_PREP = {0: (), 1: ("x",), 2: ("h",), 3: ("x", "h")}
INPUT_XBASIS = {0: False, 1: False, 2: True, 3: True}
INPUT_EXPECT = {0: 0, 1: 1, 2: 0, 3: 1}
INPUT_NAME = {0: "|0>", 1: "|1>", 2: "|+>", 3: "|->"}


def teleport_phase1(input_label, rng):
    """Prep input on q0, Bell pair on (q1,q2), Bell-measure (q0,q1). Returns (tableau, m_x, m_z)."""
    t = Chp(3)
    for g in INPUT_PREP[input_label]:
        (t.x_gate if g == "x" else t.h)(0)
    t.h(1); t.cnot(1, 2)              # Bell pair on q1,q2
    t.cnot(0, 1); t.h(0)             # Bell measurement basis on q0,q1
    m_x = t.measure(1, rng)          # X-correction control (q1 outcome)
    m_z = t.measure(0, rng)          # Z-correction control (q0 outcome)
    return t, m_x, m_z


def teleport_verify(t, input_label, a_x, a_z, rng):
    """Apply byproducts X^{a_x} Z^{a_z} on q2 and verify the teleported state equals the input."""
    if a_x:
        t.x_gate(2)
    if a_z:
        t.z_gate(2)
    if INPUT_XBASIS[input_label]:
        t.h(2)
    return t.measure(2, rng) == INPUT_EXPECT[input_label]


class FfStats:
    def __init__(self, clk_hz, commit_c):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.trials = 0
        self.on_pass = 0
        self.off_pass = 0
        self.max_lat = 0
        self.sum_lat = 0
        self.n_lat = 0
        self.t_wall = 0.0
        self.rounds = 0

    def add_decode(self, lat_cyc, dt, rounds):
        self.sum_lat += lat_cyc
        self.n_lat += 1
        self.max_lat = max(self.max_lat, lat_cyc)
        self.t_wall += dt
        self.rounds += rounds

    def add_trial(self, on_ok, off_ok):
        self.trials += 1
        self.on_pass += on_ok
        self.off_pass += off_ok

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def dashboard(self):
        on = self.on_pass / self.trials if self.trials else 0.0
        off = self.off_pass / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        budget = self.commit_c * 1.0
        thr = (self.rounds / self.t_wall / 1000.0) if self.t_wall > 0 else 0.0
        return (
            "[ff] trial %5d | decode-ON fidelity %5.1f%% | raw-OFF %5.1f%% | gain %+5.1fpp | "
            "decode %.2fµs worst %.2fµs (%.1fx budget) | %4.1fk decodes/s"
            % (self.trials, 100 * on, 100 * off, 100 * (on - off), mean, worst,
               (budget / worst) if worst > 0 else float("inf"), thr)
        )

    def summary(self, p, per_input):
        on = self.on_pass / self.trials if self.trials else 0.0
        off = self.off_pass / self.trials if self.trials else 0.0
        on_ci = ci95(on, self.trials)
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        budget = self.commit_c * 1.0
        gap_real = (on - off) > 6 * (on_ci + ci95(off, self.trials))  # decode clearly steers the result
        lines = [
            "",
            "=" * 82,
            "  FEED-FORWARD ON REAL SILICON  —  teleportation byproduct driven by the Arty decoder",
            "=" * 82,
            "  operating point   : p = %.4f  (d=3, %d-round byproduct-measurement blocks, 2 decodes/trial)"
            % (p, int(self.rounds / max(1, self.n_lat))),
            "  trials            : %d" % self.trials,
            "  teleport fidelity : ON  (decode-corrected byproduct) = %.2f%% ± %.2f"
            % (100 * on, 100 * on_ci),
            "                      OFF (raw undecoded byproduct)     = %.2f%%" % (100 * off),
            "                      GAIN from the real-time decoder    = %+.2f pp" % (100 * (on - off)),
            "  per-input ON/OFF  : " + "  ".join(
                "%s %.0f/%.0f%%" % (INPUT_NAME[k], 100 * v[0], 100 * v[1]) for k, v in sorted(per_input.items())
            ),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs  vs %.0f µs commit budget -> %s"
            % (mean, worst, budget, "real-time" if worst < budget else "OVER budget"),
            "  verdict           : %s"
            % ("the on-silicon decode STEERS the computation (ON >> OFF): feed-forward works"
               if gap_real else "no significant ON/OFF gap — check operating point / decoder"),
            "=" * 82,
        ]
        return "\n".join(lines), gap_real


def run_trials(trials, decode_fn, stats, rng, n_trials):
    """Core loop shared by board + selfcheck. `decode_fn(block_words) -> (ehat, lat_cyc, dt)`."""
    per_input_on = {k: 0 for k in range(4)}
    per_input_off = {k: 0 for k in range(4)}
    per_input_n = {k: 0 for k in range(4)}
    for i in range(n_trials):
        inp, e_x, e_z, bx, bz = trials[i % len(trials)]
        ehat_x, lat_x, dt_x = decode_fn(bx)
        ehat_z, lat_z, dt_z = decode_fn(bz)
        stats.add_decode(lat_x, dt_x, len(bx))
        stats.add_decode(lat_z, dt_z, len(bz))
        # genuine teleportation gadget: one Bell measurement, paired ON/OFF verification
        t0, m_x, m_z = teleport_phase1(inp, rng)
        raw_x, raw_z = m_x ^ e_x, m_z ^ e_z          # code-level raw outcomes (corrupted by meas error)
        on_ok = teleport_verify(t0.copy(), inp, raw_x ^ ehat_x, raw_z ^ ehat_z, rng)   # decoder-corrected
        off_ok = teleport_verify(t0.copy(), inp, raw_x, raw_z, rng)                     # raw, undecoded
        stats.add_trial(on_ok, off_ok)
        per_input_n[inp] += 1
        per_input_on[inp] += on_ok
        per_input_off[inp] += off_ok
        if (i + 1) % 500 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
        if (i + 1) % 5000 == 0:
            print()
    print("\r" + stats.dashboard())
    per_input = {k: (per_input_on[k] / max(1, per_input_n[k]),
                     per_input_off[k] / max(1, per_input_n[k])) for k in range(4)}
    return per_input


def run_selfcheck(vec, n_trials):
    meta, trials = load_ff_vec(vec)
    p = float(meta.get("p", 0.01)) if "p" in meta else 0.01
    C = int(meta.get("C", 3))
    n = n_trials or len(trials)
    print("[selfcheck] %d trials, PERFECT decoder (ê=e) — expect ON≈100%%, OFF<100%%" % n)
    stats = FfStats(clk_hz=50_000_000, commit_c=C)
    rng = random.Random(0xFEE1)
    # Perfect decoder: ê = e (the block's true logical flip, carried on the T line). This isolates the
    # gadget from the RTL so a bug shows as ON < 100%. Board mode replaces this with the real decode.
    per_input_on = {k: 0 for k in range(4)}
    per_input_off = {k: 0 for k in range(4)}
    per_input_n = {k: 0 for k in range(4)}
    for i in range(n):
        inp, e_x, e_z, bx, bz = trials[i % len(trials)]
        ehat_x, ehat_z = e_x, e_z                     # perfect decoder
        stats.add_decode(20, 0.0, len(bx))
        stats.add_decode(20, 0.0, len(bz))
        t0, m_x, m_z = teleport_phase1(inp, rng)
        raw_x, raw_z = m_x ^ e_x, m_z ^ e_z
        on_ok = teleport_verify(t0.copy(), inp, raw_x ^ ehat_x, raw_z ^ ehat_z, rng)
        off_ok = teleport_verify(t0.copy(), inp, raw_x, raw_z, rng)
        stats.add_trial(on_ok, off_ok)
        per_input_n[inp] += 1
        per_input_on[inp] += on_ok
        per_input_off[inp] += off_ok
        if (i + 1) % 5000 == 0:
            print("\r" + stats.dashboard())
    print("\r" + stats.dashboard())
    per_input = {k: (per_input_on[k] / max(1, per_input_n[k]),
                     per_input_off[k] / max(1, per_input_n[k])) for k in range(4)}
    text, ok = stats.summary(p, per_input)
    print(text)
    on = stats.on_pass / stats.trials
    print("[selfcheck] ON fidelity = %.4f (expect 1.0 with a perfect decoder) -> %s"
          % (on, "OK" if on > 0.999 else "FAIL — gadget bug"))
    return 0 if on > 0.999 else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate
    import numpy as np

    meta, trials = load_ff_vec(vec)
    p = float(meta.get("p", 0.01))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    print("[ff] overlay %s  W=%d C=%d slices=%d p=%.4f  (%d rounds -> %d windows/block, 2 blocks/trial)"
          % (bitfile, W, C, slices, p, total, nwin))

    ol = Overlay(bitfile)
    dma = getattr(ol, next(kk for kk in ol.ip_dict if "dma" in kk.lower()))
    ib = allocate(shape=(total,), dtype=np.uint32)
    ob = allocate(shape=(nwin,), dtype=np.uint32)

    def decode_block(block):
        ib[:] = 0
        ib[: len(block)] = np.asarray(block, dtype=np.uint32)
        ob[:] = 0
        ib.flush()
        t0 = time.perf_counter()
        dma.recvchannel.transfer(ob)
        dma.sendchannel.transfer(ib)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        dt = time.perf_counter() - t0
        ob.invalidate()
        words = np.asarray(ob)
        ehat = int(np.bitwise_xor.reduce((words >> 31) & 1)) & 1
        lat = int((words & 0xFFFF).max())
        return ehat, lat, dt

    decode_block(trials[0][3])  # warm-up
    stats = FfStats(clk_hz=50_000_000, commit_c=C)
    rng = random.Random(0xFEE1)
    n = n_trials or len(trials)
    print("[ff] running the teleportation gadget with the silicon decoder in the loop...\n")
    per_input = run_trials(trials, decode_block, stats, rng, n)
    text, ok = stats.summary(p, per_input)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Feed-forward teleportation driven by the on-silicon decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the gadget off-board (perfect decoder)")
    ap.add_argument("--trials", type=int, default=None, help="number of teleportation trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_ff_d3.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_feedforward.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_feedforward.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
