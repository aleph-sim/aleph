#!/usr/bin/env python3
# Q6-32 Milestone H — Shor with a DECODED inverse QFT: the m=3 finer-resolution order-finding whose inverse
# QFT's own non-Clifford gates are also decoded on the Arty, so the decode load spans the QFT as well as the
# modular exponentiation. Milestone G treated the m=2 inverse QFT as free Clifford glue, but its
# controlled-phase gates are NOT Clifford: controlled-S = diag(1,1,1,i) sits in the third level of the
# Clifford hierarchy (like T). Here each controlled-S is decomposed into Clifford + 3 T and every T is a
# code-protected magic-state measurement DECODED on the board -- the honest fault-tolerant accounting.
#
# We use m=3 phase qubits (peaks at multiples of 2^m/r; for a=2,N=3,r=2 -> {0,4}) and a band-2 approximate
# QFT (Coppersmith): keep H and the controlled-S gates, drop the controlled-T (R_3), which is not
# ancilla-free Clifford+T. So the whole Shor circuit is Clifford+T with every T decoded: 7*(modexp Toffolis)
# + 3*(m-1) controlled-S T-gates. HONEST CAVEAT: for r=2 the period is coarse enough that the controlled-S
# gates do not change the ideal outcome (band-1 already peaks at {0,4}); decoding them therefore adds decode
# load (and can only lower ON) without changing the answer. The generic case where the QFT gates are
# ESSENTIAL needs r not dividing 2^m, i.e. n>=3 work qubits (>15 qubits, past the Arty's state-vector reach).
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_shorqft.py uf_arty_dma_win.bit cosim_shorqft_n2.vec [--trials 12]
# Off-board self-check (perfect decoder -> ideal peaks reveal r):
#   python3 uf_qubit_shorqft.py --selfcheck cosim_shorqft_n2.vec

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


def modexp_toffolis(n, N, a, m):
    return sum(_cua_toffolis(n, N, pow(a, 1 << j, N)) for j in range(m))


def total_gates(n, N, a, m):
    """Every decoded T: 7 per modexp Toffoli + 3 per band-2 inverse-QFT controlled-S (m-1 of them)."""
    return 7 * modexp_toffolis(n, N, a, m) + 3 * (m - 1)


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


def _decoded_t(st, q, dag, wrongs, gi, nq):
    """A single magic-state T (or T†); a wrong decode inserts the extra S (or S†). One decoded gate."""
    st = apply_1q(st, q, TDG if dag else T, nq)
    if wrongs[gi[0]]:
        st = apply_1q(st, q, SDG if dag else S, nq)
    gi[0] += 1
    return st


def csdg_decoded(st, c, t, wrongs, gi, nq):
    """Controlled-S† = diag(1,1,1,-i) as Clifford + 3 decoded T: CNOT; T_t; CNOT; T†_t; T†_c."""
    st = cnot(st, c, t, nq)
    st = _decoded_t(st, t, False, wrongs, gi, nq)
    st = cnot(st, c, t, nq)
    st = _decoded_t(st, t, True, wrongs, gi, nq)
    st = _decoded_t(st, c, True, wrongs, gi, nq)
    return st


def iqft_decoded(st, qubits, wrongs, gi, nq):
    """Band-2 inverse QFT: H (Clifford), SWAP (Clifford), and adjacent controlled-S† (decoded, 3 T each).

    Drops the controlled-T (R_3) rotations (not ancilla-free Clifford+T) -- the Coppersmith band-2 AQFT.
    """
    m = len(qubits)
    for i in range(m // 2):
        st = swap(st, qubits[i], qubits[m - 1 - i], nq)
    for j in reversed(range(m)):
        for k in reversed(range(j + 1, m)):
            if k - j + 1 == 2:                      # adjacent -> controlled-S† (decoded)
                st = csdg_decoded(st, qubits[k], qubits[j], wrongs, gi, nq)
        st = apply_1q(st, qubits[j], H, nq)
    return st


def apply_ccx(st, a, b, c, wrongs, gi, nq):
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


def run_shorqft(n, N, a_const, m, wrongs):
    """Full order-finding with a DECODED band-2 inverse QFT. Returns the 2^m phase-outcome distribution."""
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
    st[bit(r1[0])] = 1.0
    for j in range(m):
        st = apply_1q(st, kq[j], H, nq)
    gi = [0]
    for j in range(m):                          # modular exponentiation (decoded Toffolis)
        a_j = pow(a_const, 1 << j, N)
        st = apply_cua(st, kq[j], r1, r2, r2ovf, areg, ncq, tq, c0, N, a_j, wrongs, gi, nq)
    st = iqft_decoded(st, kq[::-1], wrongs, gi, nq)  # inverse QFT with decoded controlled-S gates

    probs = np.abs(st) ** 2
    idx = np.arange(1 << nq)
    yvals = np.zeros(1 << nq, dtype=int)
    for j in range(m):
        yvals |= ((idx >> (nq - 1 - kq[j])) & 1) << j
    dist = np.zeros(1 << m)
    np.add.at(dist, yvals, probs)
    return dist


def peak_set(N, a, m):
    r = _order(a, N)
    return sorted({(j * (1 << m)) // r for j in range(r)}), r


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_shorqft_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for kk, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(kk, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 1896))
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


class ShorQftStats:
    def __init__(self, clk_hz, commit_c, n, N, a_const, m, gates, peaks, r, cs):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.n = n
        self.N = N
        self.a_const = a_const
        self.m = m
        self.gates = gates
        self.peaks = peaks
        self.r = r
        self.cs = cs
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
        return ("[Shor+QFT %d mod %d r=%d m=%d, %d T] trial %5d | ON %5.1f%% | OFF %5.1f%% | worst %.2fµs | %4.1fk dec/s"
                % (self.a_const, self.N, self.r, self.m, self.gates, self.trials, 100 * on, 100 * off, worst, thr))

    def summary(self, p):
        on = self.on / self.trials if self.trials else 0.0
        off = self.off / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        good = on > off + 0.02
        floor = 100.0 * len(self.peaks) / (1 << self.m)
        lines = [
            "",
            "=" * 98,
            "  SHOR WITH DECODED INVERSE QFT ON REAL SILICON  —  order of %d mod %d (r=%d), m=%d, %d T-decodes/op" % (self.a_const, self.N, self.r, self.m, self.gates),
            "=" * 98,
            "  operating point : p = %.4f  (d=3, %d-qubit Shor, %d-bit phase reg, %d modexp-T + %d QFT-T decoded)"
            % (p, self.m + 4 * self.n + 3, self.m, self.gates - 3 * self.cs, 3 * self.cs),
            "  ideal peaks     : %s  (multiples of 2^%d/%d=%d)  ->  reveals r=%d" % (self.peaks, self.m, self.r, (1 << self.m) // self.r, self.r),
            "  trials          : %d" % self.trials,
            "  P(peaks) fidelity: ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                     OFF (raw undecoded)      = %.2f%%   (random floor %.1f%%)" % (100 * off, floor),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  RESULT n=%d N=%d a=%d m=%d T=%d (QFT-T=%d) : ON=%.4f OFF=%.4f" % (self.n, self.N, self.a_const, self.m, self.gates, 3 * self.cs, on, off),
            "=" * 98,
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
        p_on = float(run_shorqft(n, N, a_const, m, on_wrongs)[peaks].sum())
        p_off = float(run_shorqft(n, N, a_const, m, off_wrongs)[peaks].sum())
        stats.add_trial(p_on, p_off)
        if (i + 1) % 3 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())


def _oracle(n, N, a_const, m, gates):
    peaks, r = peak_set(N, a_const, m)
    ideal = run_shorqft(n, N, a_const, m, [False] * gates)
    on_peaks = sum(ideal[y] for y in peaks)
    off_peaks = sum(ideal[y] for y in range(1 << m) if y not in peaks)
    clean = (1 << m) % r == 0
    ok = clean and on_peaks > 0.999 and off_peaks < 1e-6
    return ok, peaks, r, on_peaks


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_shorqft_vec(vec)
    p = float(meta.get("p", 0.002))
    C = int(meta.get("C", 3))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    m = int(meta.get("m", 3))
    cs = m - 1
    ok, peaks, r, on_peaks = _oracle(n, N, a_const, m, gates)
    print("[selfcheck] Shor+decoded-QFT order of %d mod %d: ideal peaks %s carry %.4f, r=%d -> %s"
          % (a_const, N, peaks, on_peaks, r, "EXACT" if ok else "MISMATCH — circuit bug"))
    nn = n_trials or len(trials)
    stats = ShorQftStats(50_000_000, C, n, N, a_const, m, gates, peaks, r, cs)
    pk = np.array(peaks)
    for i in range(nn):
        es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        p_on = float(run_shorqft(n, N, a_const, m, [False] * gates)[pk].sum())
        p_off = float(run_shorqft(n, N, a_const, m, [e != 0 for e in es])[pk].sum())
        stats.add_trial(p_on, p_off)
        if (i + 1) % 3 == 0:
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

    meta, gates, trials = load_shorqft_vec(vec)
    p = float(meta.get("p", 0.002))
    n = int(meta.get("n", 2))
    N = int(meta.get("N", (1 << n) - 1))
    a_const = int(meta.get("a", 2))
    m = int(meta.get("m", 3))
    cs = m - 1
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    kk = max(1, -(-(total_raw - W) // C))
    total = W + kk * C
    nwin = 1 + kk
    ok, peaks, r, on_peaks = _oracle(n, N, a_const, m, gates)
    if not ok:
        print("[shor+qft] ABORT: perfect-decoder Shor does not reveal a clean order (peaks %s carry %.4f)" % (peaks, on_peaks))
        return 3
    print("[shor+qft] overlay %s  order of %d mod %d (r=%d)  m=%d phase qubits  gates=%d T-decodes (%d in the QFT)  p=%.4f  (%d-qubit)"
          % (bitfile, a_const, N, r, m, gates, 3 * cs, p, m + 4 * n + 3))

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
    stats = ShorQftStats(50_000_000, C, n, N, a_const, m, gates, np.array(peaks), r, cs)
    nn = n_trials or len(trials)
    print("[shor+qft] running Shor of %d mod %d with a DECODED inverse QFT (%d T-decodes) in the loop...\n" % (a_const, N, gates))
    run_loop(n, N, a_const, m, np.array(peaks), gates, trials, decode_block, stats, nn)
    text, okf = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if okf else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Shor with a decoded inverse QFT, driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the Shor+QFT circuit off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_shorqft_n2.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_shorqft.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_shorqft.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
