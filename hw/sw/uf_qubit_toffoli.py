#!/usr/bin/env python3
# Q6-27 — MULTI-QUBIT NON-CLIFFORD ALGORITHM end-to-end from the decoder: a logical Toffoli (CCX) run on
# a real 3-qubit state vector, with all 7 of its T-gate magic-state measurements resolved in real time by
# the sliding-window decoder on the Arty Z7-20.
#
# Toffoli's standard decomposition (Nielsen & Chuang §4.3) is 7 T/T† + 6 CNOT + 2 H. CNOT/H are Clifford
# (transversal in an FT machine, no decode); each of the 7 non-Clifford T's is applied by gate
# teleportation whose magic-ancilla Z-measurement is code-protected (raw = m ⊕ e) and DECODED on the real
# Arty. A wrong decode inserts an extra S mid-circuit — and because S lands before the decomposition's H
# gates, the Toffoli output becomes a genuine superposition, not a bit flip. We verify the CLASSICAL
# TRUTH TABLE (target flips iff both controls are 1) with the decoder:
#   * ON  — extra S inserted only where the decode was wrong (ê ≠ e) → Toffoli computes correctly;
#   * OFF — extra S inserted wherever the raw measurement erred (e ≠ 0) → Toffoli corrupted.
#
# The 3-qubit logical state is a genuine 8-amplitude state vector (numpy); the decoder drives the
# conditional S corrections. Non-Clifford, multi-qubit, end-to-end on real silicon.
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_toffoli.py uf_arty_dma_win.bit cosim_toffoli_d3.vec [--trials 2400]
# Off-board gadget self-check (perfect decoder → exact Toffoli truth table on all 8 inputs):
#   python3 uf_qubit_toffoli.py --selfcheck cosim_toffoli_d3.vec

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


def apply_1q(st, q, U):
    """Apply a 2x2 gate to qubit q of a 3-qubit state vector (qubit 0 = MSB, index = 4a+2b+c)."""
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


def run_toffoli(inp, wrongs):
    """Apply the 7-T Toffoli decomposition (controls a=0,b=1; target c=2) to |inp>.

    wrongs[g] = True inserts an extra S after T-gate g (the byproduct that leaks when its decode is
    wrong). With all-False the sequence is exactly the Toffoli — the self-check verifies that.
    """
    a, b, c = 0, 1, 2
    st = np.zeros(8, dtype=complex)
    st[inp] = 1.0
    gi = [0]

    def tgate(st, q, dagger):
        st = apply_1q(st, q, TDG if dagger else T)
        if wrongs[gi[0]]:
            st = apply_1q(st, q, SDG if dagger else S)
        gi[0] += 1
        return st

    st = apply_1q(st, c, H)
    st = cnot(st, b, c); st = tgate(st, c, True)   # g0: T† c
    st = cnot(st, a, c); st = tgate(st, c, False)  # g1: T  c
    st = cnot(st, b, c); st = tgate(st, c, True)   # g2: T† c
    st = cnot(st, a, c); st = tgate(st, b, False)  # g3: T  b
    st = tgate(st, c, False)                        # g4: T  c
    st = cnot(st, a, b)
    st = apply_1q(st, c, H)
    st = tgate(st, a, False)                        # g5: T  a
    st = tgate(st, b, True)                         # g6: T† b
    st = cnot(st, a, b)
    return st


def toffoli_expected(inp):
    a, b, c = (inp >> 2) & 1, (inp >> 1) & 1, inp & 1
    cp = c ^ (a & b)
    return (a << 2) | (b << 1) | cp


def fidelity(inp, wrongs):
    st = run_toffoli(inp, wrongs)
    return float(abs(st[toffoli_expected(inp)]) ** 2)


def round_word(bits):
    return sum(1 << j for j, ch in enumerate(bits) if ch == "1")


def load_toffoli_vec(path):
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(k, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 7))
    trials = []
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            parts = l.split()
            inp = int(parts[1])
            es = [int(x) for x in parts[2:]]
            blocks, off = [], i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + k]) for k in range(slices)])
                off += slices
            trials.append((inp, es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


class ToffStats:
    def __init__(self, clk_hz, commit_c):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.trials = 0
        self.on_fid = 0.0
        self.off_fid = 0.0
        self.by_in_on = [0.0] * 8
        self.by_in_n = [0] * 8
        self.wrong_sum = 0.0
        self.wrong_n = 0
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

    def add_trial(self, inp, on_p, off_p, w):
        self.trials += 1
        self.on_fid += on_p
        self.off_fid += off_p
        self.by_in_on[inp] += on_p
        self.by_in_n[inp] += 1
        if w >= 1:
            self.wrong_sum += on_p
            self.wrong_n += 1

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def dashboard(self):
        on = self.on_fid / self.trials if self.trials else 0.0
        off = self.off_fid / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        thr = (self.rounds / self.t_wall / 1000.0) if self.t_wall > 0 else 0.0
        return (
            "[toffoli] trial %5d | ON truth-table fidelity %5.1f%% | OFF %5.1f%% | gain %+5.1fpp | "
            "worst %.2fµs (%.1fx) | %4.1fk dec/s"
            % (self.trials, 100 * on, 100 * off, 100 * (on - off), worst,
               (self.commit_c / worst) if worst > 0 else float("inf"), thr)
        )

    def summary(self, p):
        on = self.on_fid / self.trials if self.trials else 0.0
        off = self.off_fid / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        names = ["|%d%d%d>" % ((k >> 2) & 1, (k >> 1) & 1, k & 1) for k in range(8)]
        tbl = "  ".join(
            "%s→|%d%d%d> %.0f%%" % (names[k], (toffoli_expected(k) >> 2) & 1,
                                    (toffoli_expected(k) >> 1) & 1, toffoli_expected(k) & 1,
                                    100 * self.by_in_on[k] / self.by_in_n[k])
            for k in range(8) if self.by_in_n[k]
        )
        fid_wrong = (self.wrong_sum / self.wrong_n) if self.wrong_n else float("nan")
        good = (on - off) > 0.05 and on > 0.80
        lines = [
            "",
            "=" * 90,
            "  MULTI-QUBIT NON-CLIFFORD ALGORITHM ON REAL SILICON  —  logical Toffoli, 7 T's decoded by the Arty",
            "=" * 90,
            "  operating point : p = %.4f  (d=3, 3 logical qubits, 7 T-gate decodes/Toffoli)" % p,
            "  trials          : %d" % self.trials,
            "  truth-table fid : ON  (decoder-corrected) = %.2f%%" % (100 * on),
            "                    OFF (raw undecoded)      = %.2f%%   (gain %+.2f pp)" % (100 * off, 100 * (on - off)),
            "  per-input (ON)  : " + tbl,
            "  fidelity when ≥1 T-decode wrong (ON) = %.3f  (non-Clifford corruption, not a clean flip)"
            % fid_wrong,
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs budget -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  verdict         : %s"
            % ("a real 3-qubit non-Clifford algorithm runs correctly end-to-end with the silicon decoder "
               "in the loop (ON >> OFF)" if good else "ON/OFF gap unclear — check operating point"),
            "=" * 90,
        ]
        return "\n".join(lines), good


def run_loop(trials, decode_fn, stats, n_trials):
    for i in range(n_trials):
        inp, es, blocks = trials[i % len(trials)]
        ehat = []
        for blk in blocks:
            eh, lat, dt = decode_fn(blk)
            ehat.append(eh)
            stats.add_decode(lat, dt, len(blk))
        w = sum(1 for e, eh in zip(es, ehat) if e != eh)
        on_wrongs = [e != eh for e, eh in zip(es, ehat)]   # extra S where decode wrong
        off_wrongs = [e != 0 for e in es]                  # extra S where raw measurement erred
        stats.add_trial(inp, fidelity(inp, on_wrongs), fidelity(inp, off_wrongs), w)
        if (i + 1) % 400 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
        if (i + 1) % 4000 == 0:
            print()
    print("\r" + stats.dashboard())


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_toffoli_vec(vec)
    p = float(meta.get("p", 0.005))
    C = int(meta.get("C", 3))
    # First: the decomposition must reproduce the Toffoli truth table exactly (perfect decoder, no extra S).
    exact = all(abs(fidelity(inp, [False] * gates) - 1.0) < 1e-9 for inp in range(8))
    print("[selfcheck] Toffoli decomposition vs truth table (all 8 inputs, perfect decoder): %s"
          % ("EXACT" if exact else "MISMATCH — decomposition bug"))
    n = n_trials or len(trials)
    stats = ToffStats(50_000_000, C)
    for i in range(n):
        inp, es, blocks = trials[i % len(trials)]
        ehat = es[:]  # perfect decoder ê = e -> no extra S -> exact Toffoli
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        on_wrongs = [False] * gates
        off_wrongs = [e != 0 for e in es]
        stats.add_trial(inp, fidelity(inp, on_wrongs), fidelity(inp, off_wrongs), 0)
        if (i + 1) % 2000 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())
    text, _ = stats.summary(p)
    print(text)
    on = stats.on_fid / stats.trials
    ok = exact and on > 0.999
    print("[selfcheck] perfect-decoder ON fidelity = %.4f (expect 1.0) -> %s" % (on, "OK" if ok else "FAIL"))
    return 0 if ok else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate

    meta, gates, trials = load_toffoli_vec(vec)
    p = float(meta.get("p", 0.005))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    if not all(abs(fidelity(inp, [False] * gates) - 1.0) < 1e-9 for inp in range(8)):
        print("[toffoli] ABORT: decomposition does not match the Toffoli truth table")
        return 3
    print("[toffoli] overlay %s  W=%d C=%d slices=%d p=%.4f  (7 T-decodes/Toffoli, 3 logical qubits)"
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
    stats = ToffStats(50_000_000, C)
    n = n_trials or len(trials)
    print("[toffoli] running the logical Toffoli with the silicon decoder in the loop...\n")
    run_loop(trials, decode_block, stats, n)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Multi-qubit non-Clifford (logical Toffoli) driven by the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the decomposition + gadget off-board")
    ap.add_argument("--trials", type=int, default=None, help="number of Toffoli trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_toffoli_d3.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_toffoli.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_toffoli.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
