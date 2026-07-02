#!/usr/bin/env python3
# Q6-28 — SMALL LOGICAL ALGORITHM end-to-end from the decoder: 3-qubit **Grover search** run on a real
# state vector, with all 28 of its T-gate magic-state measurements resolved in real time by the
# sliding-window decoder on the Arty Z7-20, verified by the output distribution.
#
# Grover on N=8 with one marked state: H^3 then 2 iterations of {oracle, diffusion}; the optimal 2
# iterations peak the marked-state probability at ~94.5%. Both the oracle and the diffusion contain a
# CCZ = H·CCX·H, and CCX (Toffoli) = 7 T/T† gates (Q6-27). So the algorithm is 4 CCZ = 28 T-gate
# magic-state injections, each code-protected (raw = m ⊕ e) and DECODED on the real Arty. X/H/CNOT are
# Clifford (no decode). A wrong decode inserts an extra S mid-algorithm, corrupting the amplitude
# amplification — so the board measures the marked-state probability with the decoder ON (corrected) vs
# OFF (raw), and vs the uniform baseline 1/8.
#
# The 3 logical qubits are a genuine 8-amplitude state vector (numpy); the decoder drives the conditional
# S corrections. A full non-Clifford ALGORITHM, not one gate, end-to-end on real silicon.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_grover.py uf_arty_dma_win.bit cosim_grover_d3.vec [--trials 1024]
# Off-board gadget self-check (perfect decoder → ideal Grover peak ~94.5% on every marked state):
#   python3 uf_qubit_grover.py --selfcheck cosim_grover_d3.vec

import argparse
import math
import re
import sys
import time

import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
IDEAL_PEAK = math.sin(5 * math.asin(1 / math.sqrt(8))) ** 2  # 2-iteration N=8 Grover success ~0.9453


def apply_1q(st, q, U):
    out = st.copy()
    sh = 2 - q
    for i in range(8):
        if not ((i >> sh) & 1):
            j = i | (1 << sh)
            a0, a1 = st[i], st[j]
            out[i] = U[0, 0] * a0 + U[0, 1] * a1
            out[j] = U[1, 0] * a0 + U[1, 1] * a1
    return out


def cnot(st, ctrl, targ):
    out = st.copy()
    sc, stt = 2 - ctrl, 2 - targ
    for i in range(8):
        if (i >> sc) & 1:
            out[i] = st[i ^ (1 << stt)]
    return out


def apply_ccx(st, a, b, c, wrongs, gi):
    """7-T Toffoli decomposition on (controls a,b; target c); insert extra S after T-gate g if wrongs[g]."""
    def tg(st, q, dag):
        st = apply_1q(st, q, TDG if dag else T)
        if wrongs[gi[0]]:
            st = apply_1q(st, q, SDG if dag else S)
        gi[0] += 1
        return st

    st = apply_1q(st, c, H)
    st = cnot(st, b, c); st = tg(st, c, True)
    st = cnot(st, a, c); st = tg(st, c, False)
    st = cnot(st, b, c); st = tg(st, c, True)
    st = cnot(st, a, c); st = tg(st, b, False)
    st = tg(st, c, False)
    st = cnot(st, a, b)
    st = apply_1q(st, c, H)
    st = tg(st, a, False)
    st = tg(st, b, True)
    st = cnot(st, a, b)
    return st


def apply_ccz(st, wrongs, gi):
    st = apply_1q(st, 2, H)
    st = apply_ccx(st, 0, 1, 2, wrongs, gi)
    st = apply_1q(st, 2, H)
    return st


def grover(marked, wrongs):
    """Run 3-qubit Grover (2 iterations) with per-T-gate `wrongs`. Returns (P(marked), full 8-prob dist)."""
    st = np.zeros(8, dtype=complex)
    st[0] = 1.0
    for q in range(3):
        st = apply_1q(st, q, H)
    gi = [0]
    mbits = [(marked >> 2) & 1, (marked >> 1) & 1, marked & 1]
    for _it in range(2):
        for q in range(3):            # oracle: phase-flip |marked> via X-wrapped CCZ
            if mbits[q] == 0:
                st = apply_1q(st, q, X)
        st = apply_ccz(st, wrongs, gi)
        for q in range(3):
            if mbits[q] == 0:
                st = apply_1q(st, q, X)
        for q in range(3):            # diffusion: H^3 X^3 CCZ X^3 H^3
            st = apply_1q(st, q, H)
        for q in range(3):
            st = apply_1q(st, q, X)
        st = apply_ccz(st, wrongs, gi)
        for q in range(3):
            st = apply_1q(st, q, X)
        for q in range(3):
            st = apply_1q(st, q, H)
    probs = np.abs(st) ** 2
    return float(probs[marked]), probs


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_grover_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(k, v)
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
            marked = int(parts[1])
            es = [int(x) for x in parts[2:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + k]) for k in range(slices)])
                off += slices
            trials.append((marked, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class GroverStats:
    def __init__(self, clk_hz, commit_c):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.trials = 0
        self.on_pm = 0.0
        self.off_pm = 0.0
        self.on_hit = 0      # argmax == marked (ON)
        self.off_hit = 0
        self.by_m_on = [0.0] * 8
        self.by_m_n = [0] * 8
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

    def add_trial(self, marked, on_pm, on_probs, off_pm, off_probs):
        self.trials += 1
        self.on_pm += on_pm
        self.off_pm += off_pm
        self.on_hit += int(int(np.argmax(on_probs)) == marked)
        self.off_hit += int(int(np.argmax(off_probs)) == marked)
        self.by_m_on[marked] += on_pm
        self.by_m_n[marked] += 1

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def dashboard(self):
        on = self.on_pm / self.trials if self.trials else 0.0
        off = self.off_pm / self.trials if self.trials else 0.0
        hit = self.on_hit / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        thr = (self.rounds / self.t_wall / 1000.0) if self.t_wall > 0 else 0.0
        return (
            "[grover] trial %5d | ON P(marked) %5.1f%% (found %5.1f%%) | OFF %5.1f%% | uniform 12.5%% | "
            "worst %.2fµs | %4.1fk dec/s"
            % (self.trials, 100 * on, 100 * hit, 100 * off, worst, thr)
        )

    def summary(self, p):
        on = self.on_pm / self.trials if self.trials else 0.0
        off = self.off_pm / self.trials if self.trials else 0.0
        hit = self.on_hit / self.trials if self.trials else 0.0
        offhit = self.off_hit / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        tbl = "  ".join(
            "|%d%d%d>%.0f%%" % ((m >> 2) & 1, (m >> 1) & 1, m & 1, 100 * self.by_m_on[m] / self.by_m_n[m])
            for m in range(8) if self.by_m_n[m]
        )
        good = on > 3 * 0.125 and (on - off) > 0.1  # amplified well above uniform, and ON >> OFF
        lines = [
            "",
            "=" * 92,
            "  SMALL LOGICAL ALGORITHM ON REAL SILICON  —  3-qubit Grover, 28 T-gates decoded by the Arty",
            "=" * 92,
            "  operating point : p = %.4f  (d=3, 3 logical qubits, 28 T-gate decodes/search)" % p,
            "  trials          : %d   (2 Grover iterations; ideal marked-state peak %.1f%%)"
            % (self.trials, 100 * IDEAL_PEAK),
            "  P(marked state) : ON  (decoder-corrected) = %.2f%%   [found-as-argmax %.1f%%]"
            % (100 * on, 100 * hit),
            "                    OFF (raw undecoded)      = %.2f%%   [found %.1f%%]" % (100 * off, 100 * offhit),
            "                    uniform (no search)      = 12.50%%",
            "  per-marked P (ON): " + tbl,
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs budget -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  verdict         : %s"
            % ("a real 3-qubit non-Clifford ALGORITHM (Grover) amplifies the marked state end-to-end with "
               "the silicon decoder in the loop (ON >> OFF, >> uniform)" if good
               else "amplification unclear — check operating point / #gates"),
            "=" * 92,
        ]
        return "\n".join(lines), good


def run_loop(trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        marked, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]
        off_wrongs = [e != 0 for e in es]
        on_pm, on_probs = grover(marked, on_wrongs)
        off_pm, off_probs = grover(marked, off_wrongs)
        stats.add_trial(marked, on_pm, on_probs, off_pm, off_probs)
        if (i + 1) % 100 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
        if (i + 1) % 1000 == 0:
            print()
    print("\r" + stats.dashboard())


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_grover_vec(vec)
    p = float(meta.get("p", 0.003))
    C = int(meta.get("C", 3))
    ideal = all(abs(grover(m, [False] * gates)[0] - IDEAL_PEAK) < 1e-9 for m in range(8))
    print("[selfcheck] Grover peak vs ideal on all 8 marked states (perfect decoder): %s (ideal %.4f)"
          % ("EXACT" if ideal else "MISMATCH — circuit bug", IDEAL_PEAK))
    n = n_trials or len(trials)
    stats = GroverStats(50_000_000, C)
    for i in range(n):
        marked, es, blocks = trials[i % len(trials)]
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        on_pm, on_probs = grover(marked, [False] * gates)     # perfect decoder
        off_pm, off_probs = grover(marked, [e != 0 for e in es])
        stats.add_trial(marked, on_pm, on_probs, off_pm, off_probs)
        if (i + 1) % 500 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())
    text, _ = stats.summary(p)
    print(text)
    on = stats.on_pm / stats.trials
    ok = ideal and abs(on - IDEAL_PEAK) < 1e-6
    print("[selfcheck] perfect-decoder ON P(marked) = %.4f (expect %.4f) -> %s"
          % (on, IDEAL_PEAK, "OK" if ok else "FAIL"))
    return 0 if ok else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate

    meta, gates, trials = load_grover_vec(vec)
    p = float(meta.get("p", 0.003))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    if not all(abs(grover(m, [False] * gates)[0] - IDEAL_PEAK) < 1e-9 for m in range(8)):
        print("[grover] ABORT: circuit does not reach the ideal Grover peak")
        return 3
    print("[grover] overlay %s  W=%d C=%d slices=%d p=%.4f  (28 T-decodes/search, 3 logical qubits)"
          % (bitfile, W, C, slices, p))

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
        eh = int(np.bitwise_xor.reduce((words >> 31) & 1)) & 1
        lat = int((words & 0xFFFF).max())
        return eh, lat, dt

    decode_block(trials[0][2][0])  # warm-up
    stats = GroverStats(50_000_000, C)
    n = n_trials or len(trials)
    print("[grover] running 3-qubit Grover with the silicon decoder in the loop...\n")
    run_loop(trials, decode_block, stats, n)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Small logical algorithm (3-qubit Grover) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the Grover circuit + gadget off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of Grover searches (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_grover_d3.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_grover.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_grover.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
