#!/usr/bin/env python3
# Q6-30 — LARGER ALGORITHM / T-COUNT SCALING end-to-end from the decoder: a multi-controlled-X (C^kX)
# run on a real n-qubit state vector, with all 14(k-1) of its T-gate magic-state measurements resolved
# in real time by the sliding-window decoder on the Arty Z7-20. Sweeping the control count k scales the
# T-gate count (14,28,42,56 for k=2..5), so the fidelity(T) curve shows the (1-LER) compounding: more
# non-Clifford gates -> sharper dependence on decoder quality.
#
# C^kX (the multi-controlled-X at the heart of Grover oracles and reversible arithmetic) is built from a
# compute/uncompute cascade of 2(k-1) Toffolis on (k-1) ancillas; each Toffoli (Q6-27) = 7 T/T. Each T
# is applied by gate teleportation whose magic-ancilla Z-measurement is code-protected (raw=m^e) and
# DECODED on the real Arty. X/CNOT are Clifford (no decode). A wrong decode inserts an extra S, corrupting
# the result -- we verify the truth table (target flips iff all k controls are 1), decoder ON vs OFF.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_mcx.py uf_arty_dma_win.bit cosim_mcx_k3.vec [--trials 800]
# Off-board self-check (perfect decoder -> exact C^kX truth table):
#   python3 uf_qubit_mcx.py --selfcheck cosim_mcx_k3.vec

import argparse
import math
import re
import sys
import time

import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)


def apply_1q(st, q, U, nq):
    """Apply 2x2 gate U to qubit q of an n-qubit state vector (qubit 0 = MSB). Vectorized."""
    step = 1 << (nq - 1 - q)
    idx = np.arange(1 << nq)
    i0 = idx[(idx & step) == 0]
    i1 = i0 | step
    a0, a1 = st[i0], st[i1]
    out = st.copy()
    out[i0] = U[0, 0] * a0 + U[0, 1] * a1
    out[i1] = U[1, 0] * a0 + U[1, 1] * a1
    return out


def cnot(st, ctrl, targ, nq):
    sc = 1 << (nq - 1 - ctrl)
    stt = 1 << (nq - 1 - targ)
    idx = np.arange(1 << nq)
    out = st.copy()
    sel = (idx & sc) != 0
    out[sel] = st[idx[sel] ^ stt]
    return out


def apply_ccx(st, a, b, c, wrongs, gi, nq):
    """7-T Toffoli decomposition (controls a,b; target c); insert extra S after T-gate g if wrongs[g]."""
    def tg(st, q, dag):
        st = apply_1q(st, q, TDG if dag else T, nq)
        if wrongs[gi[0]]:
            st = apply_1q(st, q, SDG if dag else S, nq)
        gi[0] += 1
        return st

    st = apply_1q(st, c, H, nq)
    st = cnot(st, b, c, nq); st = tg(st, c, True)
    st = cnot(st, a, c, nq); st = tg(st, c, False)
    st = cnot(st, b, c, nq); st = tg(st, c, True)
    st = cnot(st, a, c, nq); st = tg(st, b, False)
    st = tg(st, c, False)
    st = cnot(st, a, b, nq)
    st = apply_1q(st, c, H, nq)
    st = tg(st, a, False)
    st = tg(st, b, True)
    st = cnot(st, a, b, nq)
    return st


def run_mcx(k, control, wrongs):
    """C^kX on |control>|anc=0>|t=0>: compute AND of controls into ancillas, flip target, uncompute.

    Layout (nq=2k): controls 0..k-1, ancillas k..2k-2, target 2k-1. Returns P(correct truth-table output).
    """
    nq = 2 * k
    tqi = 2 * k - 1
    st = np.zeros(1 << nq, dtype=complex)
    # input: controls set from `control` (MSB=control 0), ancillas 0, target 0
    idx = 0
    for q in range(k):
        if (control >> (k - 1 - q)) & 1:
            idx |= 1 << (nq - 1 - q)
    st[idx] = 1.0
    gi = [0]

    toffs = []  # (a, b, c) compute cascade
    toffs.append((0, 1, k))                       # anc_0 = c0 & c1
    for j in range(2, k):
        toffs.append((j, k + j - 2, k + j - 1))   # anc_{j-1} = c_j & anc_{j-2}
    anc_top = k + (k - 2)                          # = 2k-2, holds AND of all controls
    for (a, b, cc) in toffs:                        # compute
        st = apply_ccx(st, a, b, cc, wrongs, gi, nq)
    st = cnot(st, anc_top, tqi, nq)                 # flip target iff all controls 1
    for (a, b, cc) in reversed(toffs):              # uncompute
        st = apply_ccx(st, a, b, cc, wrongs, gi, nq)

    # expected output: controls unchanged, ancillas restored to 0, target = AND(all controls)
    all_one = int(all((control >> (k - 1 - q)) & 1 for q in range(k)))
    exp = idx | (all_one << (nq - 1 - tqi))
    return float(abs(st[exp]) ** 2)


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_mcx_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for kk, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(kk, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 28))
    trials = []
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            parts = l.split()
            control = int(parts[1])
            es = [int(x) for x in parts[2:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + j]) for j in range(slices)])
                off += slices
            trials.append((control, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class McxStats:
    def __init__(self, clk_hz, commit_c, k, gates):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.k = k
        self.gates = gates
        self.trials = 0
        self.on = 0.0
        self.off = 0.0
        self.max_lat = 0
        self.sum_lat = 0
        self.n_lat = 0
        self.t_wall = 0.0
        self.rounds = 0

    def add_decode(self, lat, dt, rounds):
        self.sum_lat += lat
        self.n_lat += 1
        self.max_lat = max(self.max_lat, lat)
        self.t_wall += dt
        self.rounds += rounds

    def add_trial(self, on_p, off_p):
        self.trials += 1
        self.on += on_p
        self.off += off_p

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def dashboard(self):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        thr = (self.rounds / self.t_wall / 1000.0) if self.t_wall > 0 else 0.0
        return ("[mcx k=%d, %d T] trial %5d | ON %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.k, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.03
        lines = [
            "",
            "=" * 84,
            "  T-COUNT SCALING ON REAL SILICON  —  C^%dX, %d T-gate decodes/op" % (self.k, self.gates),
            "=" * 84,
            "  operating point : p = %.4f  (d=3, %d-qubit C^%dX, %d T-decodes)" % (p, 2 * self.k, self.k, self.gates),
            "  trials          : %d" % self.trials,
            "  truth-table fid : ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                    OFF (raw undecoded)      = %.2f%%" % (100 * off),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT k=%d T=%d : ON=%.4f OFF=%.4f" % (self.k, self.gates, on, off),
            "=" * 84,
        ]
        return "\n".join(lines), good


def run_loop(k, gates, trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        control, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        stats.add_trial(run_mcx(k, control, on_wrongs), run_mcx(k, control, off_wrongs))
        if (i + 1) % 100 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_mcx_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    k = int(meta.get("k", 3))
    exact = all(abs(run_mcx(k, ctrl, [False] * gates) - 1.0) < 1e-9 for ctrl in range(1 << k))
    print("[selfcheck] C^%dX truth table (all %d control inputs, perfect decoder): %s"
          % (k, 1 << k, "EXACT" if exact else "MISMATCH — circuit bug"))
    n = n_trials or len(trials)
    stats = McxStats(50_000_000, C, k, gates)
    for i in range(n):
        control, es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        stats.add_trial(run_mcx(k, control, [False] * gates), run_mcx(k, control, [e != 0 for e in es]))
        if (i + 1) % 200 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())
    text, _ = stats.summary(p)
    print(text)
    on = stats.on / stats.trials
    ok = exact and on > 0.999
    print("[selfcheck] perfect-decoder ON fidelity = %.4f (expect 1.0) -> %s" % (on, "OK" if ok else "FAIL"))
    return 0 if ok else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate

    meta, gates, trials = load_mcx_vec(vec)
    p = float(meta.get("p", 0.002))
    k = int(meta.get("k", 3))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    kk = max(1, -(-(total_raw - W) // C))
    total = W + kk * C
    nwin = 1 + kk
    if not all(abs(run_mcx(k, ctrl, [False] * gates) - 1.0) < 1e-9 for ctrl in range(1 << k)):
        print("[mcx] ABORT: C^%dX does not match its truth table" % k)
        return 3
    print("[mcx] overlay %s  C^%dX  gates=%d T-decodes  p=%.4f  (%d-qubit state vector)"
          % (bitfile, k, gates, p, 2 * k))

    ol = Overlay(bitfile)
    dma = getattr(ol, next(nm for nm in ol.ip_dict if "dma" in nm.lower()))
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
        eh = int(np.bitwise_xor.reduce((words >> 31) & 1)) & 1
        lat = int((words & 0xFFFF).max())
        return eh, lat, dt

    decode_block(trials[0][2][0])  # warm-up
    stats = McxStats(50_000_000, C, k, gates)
    n = n_trials or len(trials)
    print("[mcx] running C^%dX (%d T-decodes) with the silicon decoder in the loop...\n" % (k, gates))
    run_loop(k, gates, trials, decode_block, stats, n)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="T-count scaling (multi-controlled-X) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the C^kX circuit off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_mcx_k3.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_mcx.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_mcx.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
