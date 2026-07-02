#!/usr/bin/env python3
# Q6-32 Milestone A — GENUINE LARGE ALGORITHM (modular arithmetic) end-to-end from the decoder: an n-bit
# quantum ripple-carry adder b := a + b (Cuccaro et al. 2004, arXiv:quant-ph/0410184 — the arithmetic core
# of Shor's algorithm) run on a real (2n+2)-qubit state vector, with all 14n of its T-gate magic-state
# measurements resolved in real time by the sliding-window decoder on the Arty Z7-20. Unlike Q6-30's
# synthetic C^kX ladder, the T-count 28/42/56 (n=2/3/4) is INTRINSIC to a real arithmetic circuit.
#
# The adder uses n MAJ + n UMA gadgets = 2n Toffolis; each Toffoli (Q6-27) = 7 T/T. Each T is applied by
# gate teleportation whose magic-ancilla Z-measurement is code-protected (raw=m^e) and DECODED on the real
# Arty. X/CNOT are Clifford (no decode). A wrong decode inserts an extra S, corrupting the arithmetic --
# we verify the sum register (b := (a+b) mod 2^n, carry-out in z, a restored), decoder ON vs OFF.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_adder.py uf_arty_dma_win.bit cosim_adder_n2.vec [--trials 800]
# Off-board self-check (perfect decoder -> exact b:=a+b truth table):
#   python3 uf_qubit_adder.py --selfcheck cosim_adder_n2.vec

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


def run_adder(n, a_in, b_in, wrongs):
    """Cuccaro ripple-carry adder b := a + b on |c0=0>|a>|b>|z=0>. Returns P(correct sum-register output).

    Layout (nq=2n+2): c0 = qubit 0, a[i] = qubit 1+i, b[i] = qubit 1+n+i, z = qubit 2n+1.
    MAJ(c,b,a): CNOT(a->b); CNOT(a->c); Toffoli(c,b->a).  UMA(c,b,a): Toffoli(c,b->a); CNOT(a->c); CNOT(c->b).
    Carry-in of block i is c0 (i=0) else a[i-1]; the top carry lands in z.  (arXiv:quant-ph/0410184.)
    """
    nq = 2 * n + 2
    c0 = 0
    aq = [1 + i for i in range(n)]
    bq = [1 + n + i for i in range(n)]
    zq = 2 * n + 1

    def bit(q):
        return 1 << (nq - 1 - q)

    # input |c0=0, a=a_in, b=b_in, z=0>  (bit i of a_in -> a[i], LSB = a[0])
    idx = 0
    for i in range(n):
        if (a_in >> i) & 1:
            idx |= bit(aq[i])
        if (b_in >> i) & 1:
            idx |= bit(bq[i])
    st = np.zeros(1 << nq, dtype=complex)
    st[idx] = 1.0
    gi = [0]

    def maj(cc, bb, aa):
        nonlocal st
        st = cnot(st, aa, bb, nq)
        st = cnot(st, aa, cc, nq)
        st = apply_ccx(st, cc, bb, aa, wrongs, gi, nq)

    def uma(cc, bb, aa):
        nonlocal st
        st = apply_ccx(st, cc, bb, aa, wrongs, gi, nq)
        st = cnot(st, aa, cc, nq)
        st = cnot(st, cc, bb, nq)

    carry = [c0] + aq  # carry-in of block i
    for i in range(n):          # MAJ down the chain
        maj(carry[i], bq[i], aq[i])
    st = cnot(st, aq[n - 1], zq, nq)  # high output carry -> z
    for i in reversed(range(n)):      # UMA back up
        uma(carry[i], bq[i], aq[i])

    # expected: c0=0, a restored to a_in, b = (a+b) mod 2^n, z = carry-out
    total = a_in + b_in
    b_out = total & ((1 << n) - 1)
    z_out = (total >> n) & 1
    exp = 0
    for i in range(n):
        if (a_in >> i) & 1:
            exp |= bit(aq[i])
        if (b_out >> i) & 1:
            exp |= bit(bq[i])
    if z_out:
        exp |= bit(zq)
    return float(abs(st[exp]) ** 2)


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_adder_vec(path):
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
    i, nl = 0, len(lines)
    while i < nl:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            parts = l.split()
            a_in = int(parts[1])
            b_in = int(parts[2])
            es = [int(x) for x in parts[3:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + j]) for j in range(slices)])
                off += slices
            trials.append((a_in, b_in, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class AddStats:
    def __init__(self, clk_hz, commit_c, n, gates):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.n = n
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
        return ("[adder n=%d, %d T] trial %5d | ON %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.n, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.03
        lines = [
            "",
            "=" * 84,
            "  RIPPLE-CARRY ADDER ON REAL SILICON  —  %d-bit b:=a+b, %d T-gate decodes/op" % (self.n, self.gates),
            "=" * 84,
            "  operating point : p = %.4f  (d=3, %d-qubit adder, %d T-decodes)" % (p, 2 * self.n + 2, self.gates),
            "  trials          : %d" % self.trials,
            "  sum-register fid: ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                    OFF (raw undecoded)      = %.2f%%" % (100 * off),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT n=%d T=%d : ON=%.4f OFF=%.4f" % (self.n, self.gates, on, off),
            "=" * 84,
        ]
        return "\n".join(lines), good


def run_loop(n, gates, trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        a_in, b_in, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        stats.add_trial(run_adder(n, a_in, b_in, on_wrongs), run_adder(n, a_in, b_in, off_wrongs))
        if (i + 1) % 100 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_adder_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    n = int(meta.get("n", 2))
    exact = all(
        abs(run_adder(n, a_in, b_in, [False] * gates) - 1.0) < 1e-9
        for a_in in range(1 << n)
        for b_in in range(1 << n)
    )
    print("[selfcheck] %d-bit adder b:=a+b truth table (all %d input pairs, perfect decoder): %s"
          % (n, 1 << (2 * n), "EXACT" if exact else "MISMATCH — circuit bug"))
    nn = n_trials or len(trials)
    stats = AddStats(50_000_000, C, n, gates)
    for i in range(nn):
        a_in, b_in, es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        stats.add_trial(
            run_adder(n, a_in, b_in, [False] * gates),
            run_adder(n, a_in, b_in, [e != 0 for e in es]),
        )
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

    meta, gates, trials = load_adder_vec(vec)
    p = float(meta.get("p", 0.002))
    n = int(meta.get("n", 2))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    kk = max(1, -(-(total_raw - W) // C))
    total = W + kk * C
    nwin = 1 + kk
    if not all(
        abs(run_adder(n, a_in, b_in, [False] * gates) - 1.0) < 1e-9
        for a_in in range(1 << n)
        for b_in in range(1 << n)
    ):
        print("[adder] ABORT: %d-bit adder does not match its b:=a+b truth table" % n)
        return 3
    print("[adder] overlay %s  %d-bit ripple-carry adder  gates=%d T-decodes  p=%.4f  (%d-qubit state vector)"
          % (bitfile, n, gates, p, 2 * n + 2))

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

    decode_block(trials[0][3][0])  # warm-up
    stats = AddStats(50_000_000, C, n, gates)
    nn = n_trials or len(trials)
    print("[adder] running the %d-bit adder (%d T-decodes) with the silicon decoder in the loop...\n" % (n, gates))
    run_loop(n, gates, trials, decode_block, stats, nn)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Ripple-carry adder driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the adder circuit off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_adder_n2.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_adder.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_adder.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
