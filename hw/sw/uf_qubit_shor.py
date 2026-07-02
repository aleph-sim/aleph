#!/usr/bin/env python3
# Q6-32 Milestone G — Shor's algorithm end-to-end from the decoder: quantum ORDER-FINDING for a mod N.
# Hadamard an m-qubit phase register into a uniform superposition of exponents, run the modular
# exponentiation |k>|1> -> |k>|a^k mod N> (Milestone F, every T-gate decoded on the Arty), apply the inverse
# QFT to the phase register, and measure -- the outcome distribution concentrates on multiples of 2^m/r,
# from which the period r = ord_N(a) (the quantum heart of Shor's factoring) follows. For a=2,N=3 the order
# is r=2, so with m=2 phase qubits the ideal outcomes are exactly {0,2} and 2/2^m = 1/2 -> r=2.
#
# For m=2 the inverse QFT is Clifford (H, controlled-S, SWAP) -- so EVERY decoded T-gate is in the modular
# exponentiation (1260 T at n=2), and the metric is how much measurement probability the decoder keeps on
# the period-revealing peaks. Perfect decoder -> all probability on the ideal peaks (P=1). A wrong decode
# corrupts the modexp and smears the distribution off the peaks -- we report P(peaks), decoder ON vs OFF.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_shor.py uf_arty_dma_win.bit cosim_shor_n2.vec [--trials 24]
# Off-board self-check (perfect decoder -> ideal peaks reveal r):
#   python3 uf_qubit_shor.py --selfcheck cosim_shor_n2.vec

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


def _order(a, N):
    x, r = a % N, 1
    while x != 1 and r < N:
        x = (x * a) % N
        r += 1
    return r


def _cua_toffolis(n, N, a):
    ainv = modinv(a, N)
    fwd = [(a * (1 << i)) % N for i in range(n)]
    inv = [(N - ((ainv * (1 << i)) % N)) % N for i in range(n)]
    loads = 2 * (sum(bin(c).count("1") for c in fwd) + sum(bin(c).count("1") for c in inv))
    return 20 * n * n + n + loads


def total_toffolis(n, N, a, m):
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


def swap(st, q1, q2, nq):
    st = cnot(st, q1, q2, nq)
    st = cnot(st, q2, q1, nq)
    st = cnot(st, q1, q2, nq)
    return st


def cphase(st, ctrl, targ, theta, nq):
    """Controlled phase: multiply the |ctrl=1,targ=1> amplitudes by e^{i*theta} (symmetric in ctrl,targ)."""
    sc = 1 << (nq - 1 - ctrl)
    stt = 1 << (nq - 1 - targ)
    idx = np.arange(1 << nq)
    out = st.copy()
    sel = ((idx & sc) != 0) & ((idx & stt) != 0)
    out[sel] = st[sel] * np.exp(1j * theta)
    return out


def iqft(st, qubits, nq):
    """Inverse QFT on `qubits` (qubits[0] = most significant). Clifford for m<=2 (H, controlled-S, SWAP)."""
    m = len(qubits)
    for i in range(m // 2):
        st = swap(st, qubits[i], qubits[m - 1 - i], nq)
    for j in reversed(range(m)):
        for k in reversed(range(j + 1, m)):
            st = cphase(st, qubits[k], qubits[j], -2 * math.pi / (1 << (k - j + 1)), nq)
        st = apply_1q(st, qubits[j], H, nq)
    return st


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
    st = cnot(st, q2, q1, nq)
    st = apply_ccx(st, ctrl, q1, q2, wrongs, gi, nq)
    st = cnot(st, q2, q1, nq)
    return st


def cuccaro(st, X_reg, Ylow, yovf, c0, wrongs, gi, nq, inverse=False):
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
    ainv = modinv(a_j, N)
    consts_fwd = [(a_j * (1 << i)) % N for i in range(len(r1))]
    consts_inv = [(N - ((ainv * (1 << i)) % N)) % N for i in range(len(r1))]
    st = cmuladd_ctrl(st, ctrl, r1, r2, r2ovf, consts_fwd, areg, ncq, tq, c0, N, wrongs, gi, nq)
    for i in range(len(r1)):
        st = fredkin(st, ctrl, r1[i], r2[i], wrongs, gi, nq)
    st = cmuladd_ctrl(st, ctrl, r1, r2, r2ovf, consts_inv, areg, ncq, tq, c0, N, wrongs, gi, nq)
    return st


def run_shor(n, N, a_const, m, wrongs):
    """Full order-finding: Hadamard phase reg -> modexp -> inverse QFT. Returns the 2^m phase-outcome dist.

    Layout (nq=m+4n+3): k[j]=q(j) (phase reg, q[0] MSB), c0=q(m), R1[i]=q(m+1+i), R2[i]=q(m+1+n+i),
    R2ovf=q(m+1+2n), areg[i]=q(m+2+2n+i), Ncq[i]=q(m+2+3n+i), t=q(m+2+4n).  R1 starts at 1.
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

    st = np.zeros(1 << nq, dtype=complex)
    st[bit(r1[0])] = 1.0                      # work register R1 = 1, phase register = 0
    for j in range(m):                         # Hadamard phase register -> superposition of exponents
        st = apply_1q(st, kq[j], H, nq)
    gi = [0]
    for j in range(m):                         # modular exponentiation (decoded)
        a_j = pow(a_const, 1 << j, N)
        st = apply_cua(st, kq[j], r1, r2, r2ovf, areg, ncq, tq, c0, N, a_j, wrongs, gi, nq)
    # phase bit k[j] carries exponent weight 2^j, so the iQFT runs on the reversed qubit order and the
    # measured value is read with k[j] as bit j (weight 2^j) -- the convention that peaks on multiples of 2^m/r.
    st = iqft(st, kq[::-1], nq)                # inverse QFT on the phase register (Clifford for m<=2)

    probs = np.abs(st) ** 2                     # marginal over the phase register, value = sum_j k[j]*2^j
    idx = np.arange(1 << nq)
    yvals = np.zeros(1 << nq, dtype=int)
    for j in range(m):
        yvals |= ((idx >> (nq - 1 - kq[j])) & 1) << j
    dist = np.zeros(1 << m)
    np.add.at(dist, yvals, probs)
    return dist


def peak_set(N, a, m):
    """Ideal period-revealing outcomes: multiples of 2^m/r, r=ord_N(a). Requires r | 2^m for exact peaks."""
    r = _order(a, N)
    return sorted({(j * (1 << m)) // r for j in range(r)}), r


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_shor_vec(path):
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
            es = [int(x) for x in parts[1:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + j]) for j in range(slices)])
                off += slices
            trials.append((es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class ShorStats:
    def __init__(self, clk_hz, commit_c, n, N, a_const, m, gates, peaks, r):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.n = n
        self.N = N
        self.a_const = a_const
        self.m = m
        self.gates = gates
        self.peaks = peaks
        self.r = r
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
        return ("[Shor %d mod %d r=%d, %d T] trial %5d | ON(peaks) %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.a_const, self.N, self.r, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.03
        lines = [
            "",
            "=" * 96,
            "  SHOR ORDER-FINDING ON REAL SILICON  —  order of %d mod %d (r=%d) end-to-end, %d T-decodes/op" % (self.a_const, self.N, self.r, self.gates),
            "=" * 96,
            "  operating point : p = %.4f  (d=3, %d-qubit Shor, %d-bit phase reg, %d T-decodes)" % (p, self.m + 4 * self.n + 3, self.m, self.gates),
            "  ideal peaks     : phase outcomes %s  ->  %d/2^%d gives r=%d" % (self.peaks, self.peaks[1] if len(self.peaks) > 1 else self.peaks[0], self.m, self.r),
            "  trials          : %d" % self.trials,
            "  P(peaks) fidelity: ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                     OFF (raw undecoded)      = %.2f%%" % (100 * off),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT n=%d N=%d a=%d m=%d T=%d : ON=%.4f OFF=%.4f" % (self.n, self.N, self.a_const, self.m, self.gates, on, off),
            "=" * 96,
        ]
        return "\n".join(lines), good


def run_loop(n, N, a_const, m, peaks, gates, trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        p_on = float(run_shor(n, N, a_const, m, on_wrongs)[peaks].sum())
        p_off = float(run_shor(n, N, a_const, m, off_wrongs)[peaks].sum())
        stats.add_trial(p_on, p_off)
        if (i + 1) % 5 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def _oracle(n, N, a_const, m, gates):
    """Perfect-decoder Shor: ideal distribution must sit exactly on the period-revealing peaks."""
    peaks, r = peak_set(N, a_const, m)
    ideal = run_shor(n, N, a_const, m, [False] * gates)
    on_peaks = sum(ideal[y] for y in peaks)
    off_peaks = sum(ideal[y] for y in range(1 << m) if y not in peaks)
    clean = (1 << m) % r == 0
    ok = clean and on_peaks > 0.999 and off_peaks < 1e-6
    return ok, peaks, r, on_peaks


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_shor_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    m = int(meta.get("m", 2))
    ok, peaks, r, on_peaks = _oracle(n, N, a_const, m, gates)
    print("[selfcheck] Shor order of %d mod %d: ideal peaks %s carry %.4f of the probability, r=%d -> %s"
          % (a_const, N, peaks, on_peaks, r, "EXACT" if ok else "MISMATCH — circuit bug"))
    nn = n_trials or len(trials)
    stats = ShorStats(50_000_000, C, n, N, a_const, m, gates, peaks, r)
    pk = np.array(peaks)
    for i in range(nn):
        es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        p_on = float(run_shor(n, N, a_const, m, [False] * gates)[pk].sum())
        p_off = float(run_shor(n, N, a_const, m, [e != 0 for e in es])[pk].sum())
        stats.add_trial(p_on, p_off)
        if (i + 1) % 5 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())
    text, _ = stats.summary(p)
    print(text)
    on = stats.on / stats.trials
    okf = ok and on > 0.999
    print("[selfcheck] perfect-decoder P(peaks) = %.4f (expect 1.0) -> %s" % (on, "OK" if okf else "FAIL"))
    return 0 if okf else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate

    meta, gates, trials = load_shor_vec(vec)
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
    ok, peaks, r, on_peaks = _oracle(n, N, a_const, m, gates)
    if not ok:
        print("[shor] ABORT: perfect-decoder Shor does not reveal a clean order (peaks %s carry %.4f)" % (peaks, on_peaks))
        return 3
    print("[shor] overlay %s  order of %d mod %d (r=%d)  m=%d phase qubits  gates=%d T-decodes  p=%.4f  (%d-qubit state vector)"
          % (bitfile, a_const, N, r, m, gates, p, m + 4 * n + 3))

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

    decode_block(trials[0][1][0])  # warm-up
    stats = ShorStats(50_000_000, C, n, N, a_const, m, gates, np.array(peaks), r)
    nn = n_trials or len(trials)
    print("[shor] running Shor order-finding of %d mod %d (%d T-decodes) with the silicon decoder in the loop...\n" % (a_const, N, gates))
    run_loop(n, N, a_const, m, np.array(peaks), gates, trials, decode_block, stats, nn)
    text, okf = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if okf else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Shor order-finding a mod N (end-to-end) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the Shor circuit off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_shor_n2.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_shor.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_shor.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
