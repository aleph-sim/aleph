#!/usr/bin/env python3
# Q6-32 Milestone F — the front half of Shor from the decoder: a short MODULAR EXPONENTIATION
# |k>|1> -> |k>|a^k mod N>, the map at the heart of Shor's period-finding. It is a chain of m controlled
# in-place multipliers: for each phase-register bit k[j], apply controlled-U_{a^{2^j}} to the work register
# (U_b|x> = |b*x mod N>), so the work register accumulates a^(sum_j k[j]*2^j) = a^k mod N. Every T-gate
# magic-state measurement is resolved in real time by the sliding-window decoder on the Arty Z7-20.
#
# On a computational-basis k this computes the modexp truth table a^k mod N; on a superposition (Hadamards,
# not run here) it prepares the periodic state |k>|a^k mod N> whose period r = ord_N(a) the inverse QFT
# (the remaining, Clifford+T back half of Shor) would extract. For a=2,N=3 the period is r=2, so a^k mod 3
# = 1,2,1,2 -- a genuinely periodic function. Each c-U_{a^{2^j}} is the Milestone-E controlled multiplier
# (control turns its constant-loads into Toffolis and its SWAP into a Fredkin), so the whole exponentiation
# is sum_j 7*(20n^2 + n + 2*Hamming(load consts of a^{2^j})) T-gate decodes on the real Arty. A wrong decode
# corrupts a^k mod N -- we verify the modexp truth table, decoder ON vs OFF.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_modexp.py uf_arty_dma_win.bit cosim_modexp_n2.vec [--trials 24]
# Off-board self-check (perfect decoder -> exact a^k mod N truth table):
#   python3 uf_qubit_modexp.py --selfcheck cosim_modexp_n2.vec

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


def modinv(a, N):
    g, x, _ = _egcd(a % N, N)
    if g != 1:
        raise ValueError("a=%d has no inverse mod N=%d (gcd=%d)" % (a, N, g))
    return x % N


def _egcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = _egcd(b % a, a)
    return g, y - (b // a) * x, x


def _cua_toffolis(n, N, a):
    """Decoded-Toffoli count of one c-U_a: 20n^2 VBE + n Fredkins + 2*Hamming(load constants)."""
    ainv = modinv(a, N)
    fwd = [(a * (1 << i)) % N for i in range(n)]
    inv = [(N - ((ainv * (1 << i)) % N)) % N for i in range(n)]
    loads = 2 * (sum(bin(c).count("1") for c in fwd) + sum(bin(c).count("1") for c in inv))
    return 20 * n * n + n + loads


def total_toffolis(n, N, a, m):
    """Sum over the m chained controlled multipliers c-U_{a^{2^j}}."""
    return sum(_cua_toffolis(n, N, pow(a, 1 << j, N)) for j in range(m))


def apply_1q(st, q, U, nq):
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


def fredkin(st, ctrl, q1, q2, wrongs, gi, nq):
    """Controlled-SWAP(ctrl; q1,q2) = CNOT(q2->q1); Toffoli(ctrl,q1->q2); CNOT(q2->q1). One decoded Toffoli."""
    st = cnot(st, q2, q1, nq)
    st = apply_ccx(st, ctrl, q1, q2, wrongs, gi, nq)
    st = cnot(st, q2, q1, nq)
    return st


def cuccaro(st, X_reg, Ylow, yovf, c0, wrongs, gi, nq, inverse=False):
    """Cuccaro ripple-carry add/sub: Y += X_reg (forward) or Y -= X_reg (inverse). quant-ph/0410184."""
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
    """VBE modular adder: y := (y + areg) mod N in place; areg/ncq/tq/yovf/c0 restored. Identity when areg=0."""
    n = len(areg)
    nbits = [(N >> i) & 1 for i in range(n)]
    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=False)
    for i in range(n):
        if nbits[i]:
            st = apply_1q(st, ncq[i], X, nq)
    st = cuccaro(st, ncq, ylow, yovf, c0, wrongs, gi, nq, inverse=True)
    for i in range(n):
        if nbits[i]:
            st = apply_1q(st, ncq[i], X, nq)
    st = cnot(st, yovf, tq, nq)
    for i in range(n):
        if nbits[i]:
            st = cnot(st, tq, ncq[i], nq)
    st = cuccaro(st, ncq, ylow, yovf, c0, wrongs, gi, nq, inverse=False)
    for i in range(n):
        if nbits[i]:
            st = cnot(st, tq, ncq[i], nq)
    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=True)
    st = apply_1q(st, yovf, X, nq)
    st = cnot(st, yovf, tq, nq)
    st = apply_1q(st, yovf, X, nq)
    st = cuccaro(st, areg, ylow, yovf, c0, wrongs, gi, nq, inverse=False)
    return st


def cmuladd_ctrl(st, ctrl, mult, accum_low, accum_ovf, consts, areg, ncq, tq, c0, N, wrongs, gi, nq):
    """Controlled modular MAC: accum += (ctrl AND mult)·consts mod N. Loads are Toffolis (ctrl AND mult[i])."""
    n = len(mult)
    for i in range(n):
        cbits = [(consts[i] >> j) & 1 for j in range(n)]
        for j in range(n):
            if cbits[j]:
                st = apply_ccx(st, ctrl, mult[i], areg[j], wrongs, gi, nq)
        st = vbe_modadd(st, areg, accum_low, accum_ovf, ncq, tq, c0, N, wrongs, gi, nq)
        for j in range(n):
            if cbits[j]:
                st = apply_ccx(st, ctrl, mult[i], areg[j], wrongs, gi, nq)
    return st


def apply_cua(st, ctrl, r1, r2, r2ovf, areg, ncq, tq, c0, N, a_j, wrongs, gi, nq):
    """Controlled in-place modular multiply c-U_{a_j}: if ctrl, R1 := (a_j * R1) mod N. (Milestone E.)"""
    ainv = modinv(a_j, N)
    consts_fwd = [(a_j * (1 << i)) % N for i in range(len(r1))]
    consts_inv = [(N - ((ainv * (1 << i)) % N)) % N for i in range(len(r1))]
    st = cmuladd_ctrl(st, ctrl, r1, r2, r2ovf, consts_fwd, areg, ncq, tq, c0, N, wrongs, gi, nq)
    for i in range(len(r1)):
        st = fredkin(st, ctrl, r1[i], r2[i], wrongs, gi, nq)
    st = cmuladd_ctrl(st, ctrl, r1, r2, r2ovf, consts_inv, areg, ncq, tq, c0, N, wrongs, gi, nq)
    return st


def run_modexp(n, N, a_const, m, k_in, wrongs):
    """Modular exponentiation |k>|1> -> |k>|a^k mod N> via m chained controlled multipliers. P(correct a^k).

    Layout (nq=m+4n+3): k[j]=q(j), c0=q(m), R1[i]=q(m+1+i), R2[i]=q(m+1+n+i), R2ovf=q(m+1+2n),
    areg[i]=q(m+2+2n+i), Ncq[i]=q(m+2+3n+i), t=q(m+2+4n).  R1 starts at 1; step j multiplies by a^{2^j} if k[j].
    """
    nq = m + 4 * n + 3
    kq = [j for j in range(m)]
    c0 = m
    r1 = [m + 1 + i for i in range(n)]
    r2 = [m + 1 + n + i for i in range(n)]
    r2ovf = m + 1 + 2 * n
    areg = [m + 2 + 2 * n + i for i in range(n)]
    ncq = [m + 2 + 3 * n + i for i in range(n)]
    tq = m + 2 + 4 * n

    def bit(q):
        return 1 << (nq - 1 - q)

    idx = bit(r1[0])                         # work register R1 = 1
    for j in range(m):
        if (k_in >> j) & 1:
            idx |= bit(kq[j])                # phase register = k_in
    st = np.zeros(1 << nq, dtype=complex)
    st[idx] = 1.0
    gi = [0]

    for j in range(m):                       # chain: R1 *= a^{2^j} controlled on k[j]
        a_j = pow(a_const, 1 << j, N)
        st = apply_cua(st, kq[j], r1, r2, r2ovf, areg, ncq, tq, c0, N, a_j, wrongs, gi, nq)

    y_out = pow(a_const, k_in, N)            # a^k mod N lands in R1
    exp = 0
    for j in range(m):
        if (k_in >> j) & 1:
            exp |= bit(kq[j])
    for i in range(n):
        if (y_out >> i) & 1:
            exp |= bit(r1[i])
    return float(abs(st[exp]) ** 2)


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_modexp_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for kk, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(kk, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 1260))
    trials = []
    i, nl = 0, len(lines)
    while i < nl:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            parts = l.split()
            k_in = int(parts[1])
            es = [int(x) for x in parts[2:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + j]) for j in range(slices)])
                off += slices
            trials.append((k_in, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class ExpStats:
    def __init__(self, clk_hz, commit_c, n, N, a_const, m, gates):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.n = n
        self.N = N
        self.a_const = a_const
        self.m = m
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
        return ("[a^k mod N, n=%d N=%d a=%d m=%d, %d T] trial %5d | ON %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.n, self.N, self.a_const, self.m, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.03
        r = _order(self.a_const, self.N)
        lines = [
            "",
            "=" * 96,
            "  MODULAR EXPONENTIATION ON REAL SILICON (front half of Shor)  —  a^k mod N = %d^k mod %d, %d T-decodes/op" % (self.a_const, self.N, self.gates),
            "=" * 96,
            "  operating point : p = %.4f  (d=3, %d-qubit modexp, %d-bit phase reg, %d T-decodes, period r=%d)" % (p, self.m + 4 * self.n + 3, self.m, self.gates, r),
            "  trials          : %d" % self.trials,
            "  a^k mod N fidelity: ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                      OFF (raw undecoded)      = %.2f%%" % (100 * off),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT n=%d N=%d a=%d m=%d T=%d : ON=%.4f OFF=%.4f" % (self.n, self.N, self.a_const, self.m, self.gates, on, off),
            "=" * 96,
        ]
        return "\n".join(lines), good


def _order(a, N):
    x, r = a % N, 1
    while x != 1 and r < N:
        x = (x * a) % N
        r += 1
    return r


def run_loop(n, N, a_const, m, gates, trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        k_in, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        stats.add_trial(run_modexp(n, N, a_const, m, k_in, on_wrongs),
                        run_modexp(n, N, a_const, m, k_in, off_wrongs))
        if (i + 1) % 5 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def _truth_exact(n, N, a_const, m, gates):
    return all(abs(run_modexp(n, N, a_const, m, k, [False] * gates) - 1.0) < 1e-9 for k in range(1 << m))


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_modexp_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    m = int(meta.get("m", 2))
    exact = _truth_exact(n, N, a_const, m, gates)
    print("[selfcheck] %d^k mod %d truth table (all %d exponents k, perfect decoder): %s  [period r=%d]"
          % (a_const, N, 1 << m, "EXACT" if exact else "MISMATCH — circuit bug", _order(a_const, N)))
    nn = n_trials or len(trials)
    stats = ExpStats(50_000_000, C, n, N, a_const, m, gates)
    for i in range(nn):
        k_in, es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        stats.add_trial(
            run_modexp(n, N, a_const, m, k_in, [False] * gates),
            run_modexp(n, N, a_const, m, k_in, [e != 0 for e in es]),
        )
        if (i + 1) % 5 == 0:
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

    meta, gates, trials = load_modexp_vec(vec)
    p = float(meta.get("p", 0.002))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    m = int(meta.get("m", 2))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    kk = max(1, -(-(total_raw - W) // C))
    total = W + kk * C
    nwin = 1 + kk
    if not _truth_exact(n, N, a_const, m, gates):
        print("[modexp] ABORT: %d^k mod %d does not match its truth table" % (a_const, N))
        return 3
    print("[modexp] overlay %s  %d^k mod %d  m=%d-bit phase reg  gates=%d T-decodes  p=%.4f  (%d-qubit state vector, period r=%d)"
          % (bitfile, a_const, N, m, gates, p, m + 4 * n + 3, _order(a_const, N)))

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
    stats = ExpStats(50_000_000, C, n, N, a_const, m, gates)
    nn = n_trials or len(trials)
    print("[modexp] running %d^k mod %d (%d T-decodes) with the silicon decoder in the loop...\n" % (a_const, N, gates))
    run_loop(n, N, a_const, m, gates, trials, decode_block, stats, nn)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Modular exponentiation a^k mod N (front half of Shor) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the modexp circuit off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_modexp_n2.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_modexp.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_modexp.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
