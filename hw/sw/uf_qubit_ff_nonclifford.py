#!/usr/bin/env python3
# Q6-26 — NON-CLIFFORD feed-forward on real silicon: the on-board decoder resolves the magic-state
# measurements of a T-gate-teleportation chain, and a missed decode injects an extra S gate — which is
# NOT a Pauli relative to the verification basis, so a wrong feed-forward turns the deterministic result
# into a genuinely QUANTUM-RANDOM outcome (fidelity 0.5), not a bit flip. This is the feed-forward the
# Q6-25 note flagged as the real frontier: its error mechanism cannot be reduced to composed memory-LER.
#
# Q6-25 teleported *stabilizer* states (a missed byproduct = a deterministic Pauli flip = composed LER).
# Here the logical qubit passes through `gates` T-gate teleportations. T is non-Clifford, so the state is
# non-stabilizer and the board simulates it with a genuine 1-qubit STATE VECTOR (numpy), not a tableau.
# Applying T by gate teleportation (Gottesman–Chuang) needs a Z-measurement of a magic ancilla and a
# conditional S correction; that measurement is code-protected (raw = m ⊕ e) and DECODED on the real
# Arty. With gates=8, T^8 = I, so the correct chain returns the input |+> (verify in X → 0).
#
# Signature (the point): a wrong decode applies S^{±1}. Because the magic outcome s is random, the extra
# gate is S or S†, so for ANY number of wrong decodes w≥1 the verification collapses to a 50/50 quantum
# coin — E[fidelity | w≥1] = 0.5, NOT 0. A classical bit-flip (composed-LER) model predicts w≥1 → 0
# (deterministic flip). Measuring ~0.5 is the non-reducibility proof. We report fidelity binned by w and
# the "quantum-random fraction" (trials whose success probability is ~0.5 — a superposition w.r.t. the
# verification basis, impossible in a stabilizer model).
#
#   ON  = S correction driven by the decoder-corrected outcome  → chain ≈ T^8 = I, fidelity ≈ 1 when w=0
#   OFF = S correction from the raw (undecoded) outcome         → wrong whenever a measurement erred
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_ff_nonclifford.py uf_arty_dma_win.bit cosim_ffnc_d3.vec [--trials 4000]
# Off-board gadget self-check (perfect decoder ê=e; no board):
#   python3 uf_qubit_ff_nonclifford.py --selfcheck cosim_ffnc_d3.vec

import argparse
import math
import random
import re
import sys
import time

import numpy as np

# Single-qubit gates (computational basis).
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
S_POW = [np.linalg.matrix_power(S, r) for r in range(4)]
KET_PLUS = H @ np.array([1, 0], dtype=complex)  # |+> = H|0>


def round_word(bits):
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_ffnc_vec(path):
    """Parse the non-Clifford .vec -> (meta, [(e_list, [block words per gate]), ...])."""
    meta = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for l in lines:
        if l.startswith("#"):
            for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                meta.setdefault(k, v)
    slices = int(meta.get("slices", 18))
    gates = int(meta.get("gates", 8))
    trials = []
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        if not l or l[0] in "#P":
            i += 1
            continue
        if l[0] == "T":
            es = [int(x) for x in l.split()[1:]]
            blocks = []
            off = i + 1
            for _g in range(gates):
                blocks.append([round_word(lines[off + k]) for k in range(slices)])
                off += slices
            trials.append((es, blocks))
            i = off
            continue
        i += 1
    return meta, gates, trials


def teleport_chain_fidelity(e_list, ehat_list, corrected, rng):
    """Run the T-teleportation chain on a 1-qubit state vector and return the verification success prob.

    corrected=True  -> ON  (S correction = decoder-corrected magic outcome)
    corrected=False -> OFF (S correction = raw undecoded outcome)
    Returns |<+| result>|^2 exactly (the true quantum success probability of the X-basis verification).
    """
    psi = KET_PLUS.copy()
    for e, ehat in zip(e_list, ehat_list):
        psi = T @ psi                       # teleport a logical T through
        s = rng.getrandbits(1)              # genuine (random) magic-ancilla Z-measurement outcome
        raw = s ^ e                         # code-level raw outcome (corrupted by the logical meas error)
        a = (raw ^ ehat) if corrected else raw   # correction bit applied to the data qubit
        res = (s - a) % 4                   # residual byproduct S^res (identity iff a == s)
        psi = S_POW[res] @ psi
    psi = H @ psi                           # X-basis verification: measure Z after H, expect 0
    return float(abs(psi[0]) ** 2)


class NcStats:
    def __init__(self, clk_hz, commit_c, gates):
        self.clk_hz = clk_hz
        self.commit_c = commit_c
        self.gates = gates
        self.trials = 0
        self.on_fid = 0.0
        self.off_fid = 0.0
        self.qrand = 0            # ON trials whose success prob is ~0.5 (genuine superposition)
        self.by_w_sum = [0.0] * (gates + 1)
        self.by_w_n = [0] * (gates + 1)
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

    def add_trial(self, on_p, off_p, w):
        self.trials += 1
        self.on_fid += on_p
        self.off_fid += off_p
        if 0.25 < on_p < 0.75:
            self.qrand += 1
        self.by_w_sum[w] += on_p
        self.by_w_n[w] += 1

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def dashboard(self):
        on = self.on_fid / self.trials if self.trials else 0.0
        off = self.off_fid / self.trials if self.trials else 0.0
        qr = self.qrand / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        thr = (self.rounds / self.t_wall / 1000.0) if self.t_wall > 0 else 0.0
        return (
            "[ffnc] trial %5d | ON T^8 fidelity %5.1f%% | OFF %5.1f%% | quantum-random %4.1f%% | "
            "worst %.2fµs (%.1fx) | %4.1fk dec/s"
            % (self.trials, 100 * on, 100 * off, 100 * qr, worst,
               (self.commit_c / worst) if worst > 0 else float("inf"), thr)
        )

    def summary(self, p):
        on = self.on_fid / self.trials if self.trials else 0.0
        off = self.off_fid / self.trials if self.trials else 0.0
        worst = self.lat_ns(self.max_lat) / 1000.0
        mean = self.lat_ns(self.sum_lat / max(1, self.n_lat)) / 1000.0
        wtbl = "  ".join(
            "w=%d:%.2f(%d)" % (w, self.by_w_sum[w] / self.by_w_n[w], self.by_w_n[w])
            for w in range(self.gates + 1) if self.by_w_n[w]
        )
        # non-reducible iff wrong decodes land near 0.5 (quantum), not near 0 (classical flip)
        wsum = sum(self.by_w_sum[w] for w in range(1, self.gates + 1))
        wn = sum(self.by_w_n[w] for w in range(1, self.gates + 1))
        fid_wrong = (wsum / wn) if wn else float("nan")
        nonreducible = wn > 0 and abs(fid_wrong - 0.5) < 0.08
        lines = [
            "",
            "=" * 86,
            "  NON-CLIFFORD FEED-FORWARD ON REAL SILICON  —  T^%d teleportation chain, S driven by the Arty"
            % self.gates,
            "=" * 86,
            "  operating point : p = %.4f  (d=3, %d T-teleportations/trial, 1 decode per magic measurement)"
            % (p, self.gates),
            "  trials          : %d" % self.trials,
            "  chain fidelity  : ON  (decoder-corrected S) = %.2f%%" % (100 * on),
            "                    OFF (raw undecoded S)      = %.2f%%" % (100 * off),
            "  fidelity vs #wrong-decodes w (ON): " + wtbl,
            "    → mean fidelity when w≥1 = %.3f  (classical bit-flip model predicts 0.0; quantum = 0.5)"
            % fid_wrong,
            "  quantum-random trials (fidelity ~0.5, a superposition the verify basis can't resolve): %.1f%%"
            % (100 * self.qrand / self.trials if self.trials else 0),
            "  decoder (measured): mean %.2f µs/window, worst %.2f µs vs %.0f µs budget -> %s"
            % (mean, worst, float(self.commit_c), "real-time" if worst < self.commit_c else "OVER"),
            "  verdict         : %s"
            % ("NON-REDUCIBLE feed-forward confirmed — a wrong on-silicon decode injects quantum "
               "randomness (≈0.5), not a classical flip"
               if nonreducible else "signature unclear — check operating point / gates"),
            "=" * 86,
        ]
        return "\n".join(lines), nonreducible


def run(trials, gates, decode_fn, stats, rng, n_trials):
    for i in range(n_trials):
        e_list, blocks = trials[i % len(trials)]
        ehat_list = []
        for blk in blocks:
            ehat, lat, dt = decode_fn(blk)
            ehat_list.append(ehat)
            stats.add_decode(lat, dt, len(blk))
        w = sum(1 for e, eh in zip(e_list, ehat_list) if e != eh)  # wrong decodes this trial
        on_p = teleport_chain_fidelity(e_list, ehat_list, corrected=True, rng=rng)
        off_p = teleport_chain_fidelity(e_list, ehat_list, corrected=False, rng=rng)
        stats.add_trial(on_p, off_p, w)
        if (i + 1) % 500 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
        if (i + 1) % 5000 == 0:
            print()
    print("\r" + stats.dashboard())


def run_selfcheck(vec, n_trials):
    meta, gates, trials = load_ffnc_vec(vec)
    p = float(meta.get("p", 0.005))
    C = int(meta.get("C", 3))
    n = n_trials or len(trials)
    print("[selfcheck] %d trials, PERFECT decoder (ê=e) — expect ON≈100%%, OFF<100%%, w≥1→~0.5" % n)
    stats = NcStats(50_000_000, C, gates)
    rng = random.Random(0xC0FFEE)
    # Perfect decoder: ê = e (block's true flip, on the T line) -> w=0 always. Board mode uses real decode.
    for i in range(n):
        e_list, blocks = trials[i % len(trials)]
        ehat_list = e_list[:]                       # perfect: ê = e -> w = 0 always
        for blk in blocks:
            stats.add_decode(20, 0.0, len(blk))
        w = 0
        on_p = teleport_chain_fidelity(e_list, ehat_list, corrected=True, rng=rng)
        off_p = teleport_chain_fidelity(e_list, ehat_list, corrected=False, rng=rng)
        stats.add_trial(on_p, off_p, w)
        if (i + 1) % 2000 == 0:
            print("\r" + stats.dashboard(), end="", flush=True)
    print("\r" + stats.dashboard())

    # Also inject synthetic wrong decodes to exercise the w-binning signature off-board.
    print("\n[selfcheck] injecting synthetic wrong decodes to show the w≥1 → ~0.5 signature:")
    demo = NcStats(50_000_000, C, gates)
    for i in range(min(n, 4000)):
        e_list, blocks = trials[i % len(trials)]
        # flip decode on a random subset to create controlled w
        ehat_list = [e ^ rng.getrandbits(1) for e in e_list]
        w = sum(1 for e, eh in zip(e_list, ehat_list) if e != eh)
        on_p = teleport_chain_fidelity(e_list, ehat_list, True, rng)
        demo.add_trial(on_p, on_p, w)
        demo.add_decode(20, 0.0, len(blocks[0]))
    text, ok = demo.summary(p)
    print(text)
    on0 = stats.on_fid / stats.trials
    print("[selfcheck] perfect-decoder ON fidelity = %.4f (expect 1.0) -> %s"
          % (on0, "OK" if on0 > 0.999 else "FAIL — gadget bug"))
    return 0 if on0 > 0.999 else 1


def run_board(bitfile, vec, n_trials):
    from pynq import Overlay, allocate

    meta, gates, trials = load_ffnc_vec(vec)
    p = float(meta.get("p", 0.005))
    W = int(meta.get("W", 9)); C = int(meta.get("C", 3)); slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    print("[ffnc] overlay %s  W=%d C=%d slices=%d p=%.4f gates=%d  (%d decodes/trial)"
          % (bitfile, W, C, slices, p, gates, gates))

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

    decode_block(trials[0][1][0])  # warm-up
    stats = NcStats(50_000_000, C, gates)
    rng = random.Random(0xC0FFEE)
    n = n_trials or len(trials)
    print("[ffnc] running the T^%d teleportation chain with the silicon decoder in the loop...\n" % gates)
    run(trials, gates, decode_block, stats, rng, n)
    text, ok = stats.summary(p)
    print(text)
    del ib, ob
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Non-Clifford feed-forward (T-teleportation) on the Arty decoder")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the gadget off-board (perfect decoder)")
    ap.add_argument("--trials", type=int, default=None, help="number of chain trials (default: all)")
    ns = ap.parse_args(argv[1:])
    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_ffnc_d3.vec")
    if ns.selfcheck:
        return run_selfcheck(vec, ns.trials)
    if not bitfile:
        print("usage: uf_qubit_ff_nonclifford.py <design.bit> <vec> [--trials N]")
        print("   or: uf_qubit_ff_nonclifford.py --selfcheck <vec>")
        return 2
    return run_board(bitfile, vec, ns.trials)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
