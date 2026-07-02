#!/usr/bin/env python3
# Q6-32 Milestone C — the next layer of Shor's ladder from the decoder: out-of-place modular multiplication
# by a constant, y := (a * x) mod N, on |x>|0> -> |x>|(a*x) mod N>. This is the operation Shor's modular
# exponentiation is built from (as its controlled version): a product is n controlled-modular-additions of
# the classical constants a*2^i mod N. Every T-gate magic-state measurement is resolved in real time by the
# sliding-window decoder on the Arty Z7-20. The T-count is INTRINSIC and climbs another order: 70n^2
# (280/630 for n=2/3), n modular adders deep.
#
# Each bit x[i] of the multiplier controls a modular add of the classical constant c_i = a*2^i mod N into
# the accumulator y. The control is realised by CNOT-loading c_i into the addend register iff x[i] (Clifford,
# free), then running an UNCONDITIONAL VBE modular adder (70n T = 10n Toffolis) -- which is the identity when
# the addend is 0 (x[i]=0). So the whole multiply is n VBE modular adders = 70n^2 T-gate decodes on the Arty;
# X/CNOT (constant load, controlled load) are Clifford. A wrong decode corrupts the product -- we verify
# y == (a*x) mod N, decoder ON vs OFF.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_mulmod.py uf_arty_dma_win.bit cosim_mulmod_n2.vec [--trials 60]
# Off-board self-check (perfect decoder -> exact (a*x) mod N truth table):
#   python3 uf_qubit_mulmod.py --selfcheck cosim_mulmod_n2.vec

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
X = np.array([[0, 1], [1, 0]], dtype=complex)


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


def cuccaro(st, X_reg, Ylow, yovf, c0, wrongs, gi, nq, inverse=False):
    """Cuccaro ripple-carry add/sub: Y += X_reg (forward) or Y -= X_reg (inverse). quant-ph/0410184.

    MAJ(c,b,a): CNOT(a->b); CNOT(a->c); TOFF(c,b->a).  UMA(c,b,a): TOFF(c,b->a); CNOT(a->c); CNOT(c->b).
    inverse = the forward gate list reversed, each gate inverted (Toffoli/CNOT self-inverse).
    """
    n = len(X_reg)
    carry = [c0] + list(X_reg)

    def maj(c, b, a):
        nonlocal st
        st = cnot(st, a, b, nq); st = cnot(st, a, c, nq)
        st = apply_ccx(st, c, b, a, wrongs, gi, nq)

    def inv_maj(c, b, a):
        nonlocal st
        st = apply_ccx(st, c, b, a, wrongs, gi, nq)
        st = cnot(st, a, c, nq); st = cnot(st, a, b, nq)

    def uma(c, b, a):
        nonlocal st
        st = apply_ccx(st, c, b, a, wrongs, gi, nq)
        st = cnot(st, a, c, nq); st = cnot(st, c, b, nq)

    def inv_uma(c, b, a):
        nonlocal st
        st = cnot(st, c, b, nq); st = cnot(st, a, c, nq)
        st = apply_ccx(st, c, b, a, wrongs, gi, nq)

    if not inverse:
        for i in range(n):
            maj(carry[i], Ylow[i], X_reg[i])
        st = cnot(st, X_reg[n - 1], yovf, nq)
        for i in reversed(range(n)):
            uma(carry[i], Ylow[i], X_reg[i])
    else:
        for i in range(n):
            inv_uma(carry[i], Ylow[i], X_reg[i])
        st = cnot(st, X_reg[n - 1], yovf, nq)
        for i in reversed(range(n)):
            inv_maj(carry[i], Ylow[i], X_reg[i])
    return st


def vbe_modadd(st, areg, ylow, yovf, ncq, tq, c0, N, wrongs, gi, nq):
    """VBE modular adder: y := (y + areg) mod N, in place. areg, ncq, tq, yovf, c0 all restored.

    Five Cuccaro adders + a conditional subtract of N (arXiv:quant-ph/9511018). Identity in y when areg=0.
    """
    n = len(areg)
    nbits = [(N >> i) & 1 for i in range(n)]

    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=False)   # 1. y += a
    for i in range(n):                                                      # load constant N (Clifford)
        if nbits[i]:
            st = apply_1q(st, ncq[i], X, nq)
    st = cuccaro(st, ncq, ylow, yovf, c0, wrongs, gi, nq, inverse=True)     # 2. y -= N
    for i in range(n):
        if nbits[i]:
            st = apply_1q(st, ncq[i], X, nq)
    st = cnot(st, yovf, tq, nq)                                            # 3. t <- overflow
    for i in range(n):                                                     # t-controlled load of N
        if nbits[i]:
            st = cnot(st, tq, ncq[i], nq)
    st = cuccaro(st, ncq, ylow, yovf, c0, wrongs, gi, nq, inverse=False)   # 4. y += (t? N : 0)
    for i in range(n):
        if nbits[i]:
            st = cnot(st, tq, ncq[i], nq)
    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=True)   # 5. y -= a
    st = apply_1q(st, yovf, X, nq)                                         # 6. reset t: X;CNOT;X
    st = cnot(st, yovf, tq, nq)
    st = apply_1q(st, yovf, X, nq)
    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=False)  # 7. y += a
    return st


def run_mulmod(n, N, a_const, x_in, wrongs):
    """Out-of-place modular multiply y := (a_const * x) mod N on |x>|0>. Returns P(correct product).

    Layout (nq=4n+3): c0=q0, x[i]=q(1+i), y[i]=q(1+n+i), yovf=q(1+2n), areg[i]=q(2+2n+i),
    Ncst[i]=q(2+3n+i), t=q(2+4n).  For each bit x[i]: CNOT-load c_i=a*2^i mod N into areg, VBE modadd, unload.
    """
    nq = 4 * n + 3
    c0 = 0
    xq = [1 + i for i in range(n)]
    yq = [1 + n + i for i in range(n)]
    yovf = 1 + 2 * n
    areg = [2 + 2 * n + i for i in range(n)]
    ncq = [2 + 3 * n + i for i in range(n)]
    tq = 2 + 4 * n

    def bit(q):
        return 1 << (nq - 1 - q)

    idx = 0
    for i in range(n):
        if (x_in >> i) & 1:
            idx |= bit(xq[i])
    st = np.zeros(1 << nq, dtype=complex)
    st[idx] = 1.0
    gi = [0]

    for i in range(n):
        c_i = (a_const * (1 << i)) % N          # classical partial constant
        cbits = [(c_i >> j) & 1 for j in range(n)]
        for j in range(n):                       # load c_i into areg iff x[i]  (Clifford)
            if cbits[j]:
                st = cnot(st, xq[i], areg[j], nq)
        st = vbe_modadd(st, areg, yq, yovf, ncq, tq, c0, N, wrongs, gi, nq)
        for j in range(n):                       # unload c_i
            if cbits[j]:
                st = cnot(st, xq[i], areg[j], nq)

    y_out = (a_const * x_in) % N
    exp = idx  # x preserved
    for i in range(n):
        if (y_out >> i) & 1:
            exp |= bit(yq[i])
    return float(abs(st[exp]) ** 2)


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_mulmod_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for kk, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(kk, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 280))
    trials = []
    i, nl = 0, len(lines)
    while i < nl:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            parts = l.split()
            x_in = int(parts[1])
            es = [int(x) for x in parts[2:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + j]) for j in range(slices)])
                off += slices
            trials.append((x_in, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class MulStats:
    def __init__(self, clk_hz, commit_c, n, N, a_const, gates):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.n = n
        self.N = N
        self.a_const = a_const
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
        return ("[mulmod n=%d N=%d a=%d, %d T] trial %5d | ON %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.n, self.N, self.a_const, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.03
        lines = [
            "",
            "=" * 90,
            "  MODULAR MULTIPLIER ON REAL SILICON  —  %d-bit y:=(%d*x) mod %d, %d T-gate decodes/op" % (self.n, self.a_const, self.N, self.gates),
            "=" * 90,
            "  operating point : p = %.4f  (d=3, %d-qubit VBE modular multiplier, %d T-decodes)" % (p, 4 * self.n + 3, self.gates),
            "  trials          : %d" % self.trials,
            "  product fidelity: ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                    OFF (raw undecoded)      = %.2f%%" % (100 * off),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT n=%d N=%d a=%d T=%d : ON=%.4f OFF=%.4f" % (self.n, self.N, self.a_const, self.gates, on, off),
            "=" * 90,
        ]
        return "\n".join(lines), good


def run_loop(n, N, a_const, gates, trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        x_in, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        stats.add_trial(run_mulmod(n, N, a_const, x_in, on_wrongs), run_mulmod(n, N, a_const, x_in, off_wrongs))
        if (i + 1) % 10 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def _truth_exact(n, N, a_const, gates):
    return all(abs(run_mulmod(n, N, a_const, x, [False] * gates) - 1.0) < 1e-9 for x in range(N))


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_mulmod_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    exact = _truth_exact(n, N, a_const, gates)
    print("[selfcheck] %d-bit y:=(%d*x) mod %d truth table (all %d residues x, perfect decoder): %s"
          % (n, a_const, N, N, "EXACT" if exact else "MISMATCH — circuit bug"))
    nn = n_trials or len(trials)
    stats = MulStats(50_000_000, C, n, N, a_const, gates)
    for i in range(nn):
        x_in, es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        stats.add_trial(
            run_mulmod(n, N, a_const, x_in, [False] * gates),
            run_mulmod(n, N, a_const, x_in, [e != 0 for e in es]),
        )
        if (i + 1) % 10 == 0:
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

    meta, gates, trials = load_mulmod_vec(vec)
    p = float(meta.get("p", 0.002))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    kk = max(1, -(-(total_raw - W) // C))
    total = W + kk * C
    nwin = 1 + kk
    if not _truth_exact(n, N, a_const, gates):
        print("[mulmod] ABORT: %d-bit multiplier does not match its y:=(%d*x) mod %d truth table" % (n, a_const, N))
        return 3
    print("[mulmod] overlay %s  %d-bit y:=(%d*x) mod %d  gates=%d T-decodes  p=%.4f  (%d-qubit state vector)"
          % (bitfile, n, a_const, N, gates, p, 4 * n + 3))

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
    stats = MulStats(50_000_000, C, n, N, a_const, gates)
    nn = n_trials or len(trials)
    print("[mulmod] running the %d-bit modular multiplier (%d T-decodes) with the silicon decoder in the loop...\n" % (n, gates))
    run_loop(n, N, a_const, gates, trials, decode_block, stats, nn)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Modular multiplier (a*x mod N) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the modular multiplier off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_mulmod_n2.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_mulmod.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_mulmod.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
