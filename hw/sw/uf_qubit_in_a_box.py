#!/usr/bin/env python3
# "Logical qubit in a box" — a CLOSED-LOOP lifetime monitor for the sliding-window streaming decoder
# on the real Arty Z7-20 (uf_stream_win / arty_z7_dma_win_bd; reuses the Q6-20/Q6-22 bitstream
# unchanged, uf_arty_dma_win.bit).
#
# Every prior on-board driver is OPEN-LOOP trace-replay: a pre-generated syndrome stream is decoded and
# the decoder's guess is scored against ground truth offline (uf_dma_stream_ler.py = an LER table). The
# decoder never touches the state. This driver closes the loop:
#
#     [sim: memory-Z cycle] --detector rounds--> [REAL decoder on silicon] --logical correction-->
#          ^                                                                                     |
#          +------------------ correction applied to the tracked logical frame -----------------+
#
# We hold a logical Pauli frame (starts at identity |0>_L). Each cycle is one finite memory-Z experiment
# of `slices` rounds streamed through the decoder; the decoder returns a proposed logical correction
# (XOR of the committed windows' obs bits). We apply it to the frame: the corrected frame is identity
# iff the decoder's correction matches the cycle's true accumulated logical flip. A residual (mismatch)
# is a LOGICAL FAILURE — the qubit "died"; we record the survival interval, re-sync the frame to truth,
# and keep the box running. That inter-failure statistic IS the logical qubit's lifetime, kept alive in
# real time by the silicon decoder. (For a memory qubit, "apply correction, check identity" equals
# pred==truth; the new thing here is the maintained state + live time-to-failure loop, and it is the
# substrate for feed-forward, where `pred` would instead condition the NEXT logical operation.)
#
# The decoder throughput/latency is MEASURED on the silicon (wall clock + the RTL per-window latency
# field, result word bits[15:0]). The logical lifetime is reported in physical time under an ASSUMED
# syndrome-extraction cycle time (--round-ns, default 1000 ns ~ a superconducting round); that
# assumption scales only the time axis, not the decoder verdict.
#
# Result word layout (matches uf_dma_stream.py):  bit31 = committed logical parity (obs),
#   bit30 = residual_empty (validity),  bits[15:0] = that window's core decode latency (cycles).
#
# Usage on the board (root + XRT env):
#   sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
#        uf_qubit_in_a_box.py uf_arty_dma_win.bit cosim_qubit_box_d3.vec [--p 0.005] [--cycles 20000]
#
# Off-board math self-check (no board, no pynq — validates the lifetime/LER bookkeeping):
#   python3 uf_qubit_in_a_box.py --selfcheck cosim_qubit_box_d3.vec [--p 0.01]

import argparse
import math
import re
import sys
import time


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0


def round_word(bits):
    return sum(1 << j for j, c in enumerate(bits) if c == "1")


def load_ler_vec(path):
    """Parse the finite-experiment .vec -> (meta, [(p, sw_rate, sw_ci, [(truth, [round_word,...]),...])])."""
    meta = {}
    blocks = []
    cur = None
    exp = None
    with open(path) as f:
        for l in f:
            l = l.rstrip("\n")
            if not l:
                continue
            if l[0] == "#":
                for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", l):
                    meta.setdefault(k, v)
                continue
            if l[0] == "P":
                kv = dict(t.split("=", 1) for t in l[1:].split() if "=" in t)
                cur = (float(kv["p"]), float(kv["sw_rate"]), float(kv["sw_ci"]), [])
                blocks.append(cur)
                exp = None
                continue
            if l[0] == "E":
                exp = (int(l.split()[1]), [])
                cur[3].append(exp)
                continue
            if exp is not None:
                exp[1].append(round_word(l.strip()))
    return meta, blocks


class LifetimeMonitor:
    """Tracks the closed-loop logical qubit: survival streaks, death events, decode latency, throughput."""

    def __init__(self, slices, round_ns, clk_hz, commit_c):
        self.slices = slices              # physical rounds per memory cycle
        self.round_ns = round_ns          # assumed physical syndrome-extraction cycle time
        self.clk_hz = clk_hz              # PL clock -> decode latency ns
        self.commit_c = commit_c          # rounds per committed window (real-time budget = C * round_ns)
        self.cycles = 0
        self.failures = 0
        self.age = 0                      # consecutive survived cycles (current logical qubit's age)
        self.intervals = []               # survived-cycle counts between successive deaths (empirical lifetime)
        self.max_lat_cyc = 0
        self.sum_lat_cyc = 0
        self.t_wall = 0.0                 # wall-clock spent in decode transfers (for stream throughput)
        self.rounds_streamed = 0

    def step(self, survived, lat_cyc, dt):
        self.cycles += 1
        self.sum_lat_cyc += lat_cyc
        self.max_lat_cyc = max(self.max_lat_cyc, lat_cyc)
        self.t_wall += dt
        self.rounds_streamed += self.slices
        if survived:
            self.age += 1
        else:
            self.failures += 1
            self.intervals.append(self.age)  # this qubit lived `age` cycles before dying
            self.age = 0                      # re-sync frame to truth and keep the box running

    # --- derived metrics ---
    @property
    def ler(self):
        return self.failures / self.cycles if self.cycles else 0.0

    @property
    def ler_ci(self):
        return ci95(self.ler, self.cycles)

    def lifetime_cycles_est(self):
        # 1/LER = mean cycles-to-failure (geometric); the model-free estimate.
        return (1.0 / self.ler) if self.ler > 0 else float("inf")

    def lifetime_cycles_emp(self):
        # Directly measured mean survival interval (cycles), if any deaths have occurred.
        return (sum(self.intervals) / len(self.intervals)) if self.intervals else float("inf")

    def lifetime_us(self, cycles):
        return cycles * self.slices * self.round_ns / 1000.0 if math.isfinite(cycles) else float("inf")

    def lat_ns(self, cyc):
        return cyc * 1_000_000_000 // self.clk_hz if self.clk_hz else 0

    def rounds_per_s(self):
        return self.rounds_streamed / self.t_wall if self.t_wall > 0 else 0.0

    def dashboard(self):
        est = self.lifetime_cycles_est()
        alive = "ALIVE ✓" if self.age > 0 or self.failures == 0 else "died ✗ (re-synced)"
        tl_us = self.lifetime_us(est)
        tl = ("%.2f ms" % (tl_us / 1000.0)) if math.isfinite(tl_us) and tl_us >= 1000 else \
             (("%.0f µs" % tl_us) if math.isfinite(tl_us) else "∞")
        worst_us = self.lat_ns(self.max_lat_cyc) / 1000.0
        mean_us = self.lat_ns(self.sum_lat_cyc / max(1, self.cycles)) / 1000.0
        budget_us = self.commit_c * self.round_ns / 1000.0
        head = (budget_us / worst_us) if worst_us > 0 else float("inf")
        return (
            "[box] cyc %6d | age %4d %-18s | deaths %4d | LER %.2e±%.1e | "
            "T_L≈%s (%.0f cyc) | decode %.2fµs worst %.2fµs (%.1fx budget) | %5.0fk rounds/s"
            % (self.cycles, self.age, alive, self.failures, self.ler, self.ler_ci,
               tl, est if math.isfinite(est) else 0, mean_us, worst_us, head,
               self.rounds_per_s() / 1000.0)
        )

    def summary(self, p, sw_rate, sw_ci):
        est_cyc = self.lifetime_cycles_est()
        emp_cyc = self.lifetime_cycles_emp()
        budget_us = self.commit_c * self.round_ns / 1000.0
        worst_us = self.lat_ns(self.max_lat_cyc) / 1000.0
        mean_us = self.lat_ns(self.sum_lat_cyc / max(1, self.cycles)) / 1000.0
        comb = self.ler_ci + sw_ci
        within = abs(self.ler - sw_rate) <= comb + 1e-12
        lines = [
            "",
            "=" * 78,
            "  LOGICAL QUBIT IN A BOX  —  closed-loop memory lifetime on real silicon",
            "=" * 78,
            "  operating point      : p = %.4f  (d=3 surface code, %d-round memory cycles)" % (p, self.slices),
            "  cycles run           : %d   (deaths: %d)" % (self.cycles, self.failures),
            "  per-cycle logical LER: %.3e ± %.1e   [software UF baseline %.3e ± %.1e -> %s]"
            % (self.ler, self.ler_ci, sw_rate, sw_ci, "MATCH" if within else "DIVERGE"),
            "  logical lifetime     : %.0f cycles (1/LER)  =  %s   @ %d ns/round"
            % (est_cyc if math.isfinite(est_cyc) else 0,
               self._fmt_us(self.lifetime_us(est_cyc)), self.round_ns),
            "     (empirical mean survival interval: %s cycles = %s over %d deaths)"
            % (("%.0f" % emp_cyc) if math.isfinite(emp_cyc) else "∞",
               self._fmt_us(self.lifetime_us(emp_cyc)), len(self.intervals)),
            "  decoder (measured)   : mean %.2f µs/window, worst %.2f µs  vs %.0f µs commit budget"
            % (mean_us, worst_us, budget_us),
            "                         real-time: %s (%.1fx headroom worst-case)"
            % ("YES" if worst_us < budget_us else "NO",
               (budget_us / worst_us) if worst_us > 0 else float("inf")),
            "  stream throughput    : %.0fk detector-rounds/s sustained through the silicon decoder"
            % (self.rounds_per_s() / 1000.0),
            "  verdict              : %s"
            % ("the real decoder keeps the logical qubit alive as well as software UF (within CI)"
               if within else "on-silicon lifetime DIVERGES from software baseline — investigate"),
            "=" * 78,
        ]
        return "\n".join(lines), within

    @staticmethod
    def _fmt_us(us):
        if not math.isfinite(us):
            return "∞"
        if us >= 1000:
            return "%.2f ms" % (us / 1000.0)
        return "%.0f µs" % us


def pick_block(blocks, p_arg):
    if p_arg is not None:
        for b in blocks:
            if abs(b[0] - p_arg) < 1e-9:
                return b
        raise SystemExit("p=%g not in vec (have %s)" % (p_arg, [b[0] for b in blocks]))
    return min(blocks, key=lambda b: b[0])  # default: lowest (most sub-threshold) operating point


def run_selfcheck(vec, p_arg, round_ns, clk_hz):
    """Validate the lifetime/LER bookkeeping off-board with synthetic per-cycle outcomes at the sw rate.

    This does NOT touch the RTL — it proves the monitor's statistics/dashboard are correct so the only
    unknown on the board is the decoder itself. Deterministic (LCG) so it's reproducible without numpy.
    """
    meta, blocks = load_ler_vec(vec)
    slices = int(meta.get("slices", 18))
    C = int(meta.get("C", 3))
    p, sw_rate, sw_ci, exps = pick_block(blocks, p_arg)
    print("[selfcheck] p=%.4f  slices=%d  synthesizing %d cycles at the software rate %.4f"
          % (p, slices, len(exps), sw_rate))
    mon = LifetimeMonitor(slices, round_ns, clk_hz, C)
    seed = 0x2024ABCD
    for i in range(len(exps)):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF  # LCG
        died = (seed / 0x7FFFFFFF) < sw_rate
        # synthetic latency ~ a plausible small window decode (in PL cycles), varied per cycle
        lat = 20 + (seed % 8)
        mon.step(survived=not died, lat_cyc=lat, dt=0.0004)
        if (i + 1) % 4000 == 0:
            print("\r" + mon.dashboard(), end="", flush=True)
    print("\r" + mon.dashboard())
    text, ok = mon.summary(p, sw_rate, sw_ci)
    print(text)
    print("[selfcheck] monitor LER %.4e vs synthesized-at %.4e -> %s"
          % (mon.ler, sw_rate, "OK" if abs(mon.ler - sw_rate) <= mon.ler_ci + 1e-3 else "MISMATCH"))
    return 0


def run_board(bitfile, vec, p_arg, cycles_arg, round_ns, clk_hz):
    from pynq import Overlay, allocate
    import numpy as np

    meta, blocks = load_ler_vec(vec)
    W = int(meta.get("W", 9))
    C = int(meta.get("C", 3))
    slices = int(meta.get("slices", 18))
    drain = max(2 * W, 16)
    p, sw_rate, sw_ci, exps = pick_block(blocks, p_arg)

    # Constant per-cycle transfer size (slices + drain, padded to a window boundary W + k*C).
    total_raw = slices + drain
    k = max(1, -(-(total_raw - W) // C))
    total = W + k * C
    nwin = 1 + k
    print("[box] overlay %s  W=%d C=%d slices=%d  operating point p=%.4f  (%d rounds -> %d windows/cycle)"
          % (bitfile, W, C, slices, p, total, nwin))

    ol = Overlay(bitfile)
    dma_name = next(kk for kk in ol.ip_dict if "dma" in kk.lower())
    dma = getattr(ol, dma_name)
    ib = allocate(shape=(total,), dtype=np.uint32)
    ob = allocate(shape=(nwin,), dtype=np.uint32)

    def decode_cycle(rounds):
        ib[:] = 0
        ib[: len(rounds)] = np.asarray(rounds, dtype=np.uint32)  # rest = zero-drain + boundary pad
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
        pred = int(np.bitwise_xor.reduce((words >> 31) & 1)) & 1     # decoder's proposed logical correction
        last_empty = int((words[-1] >> 30) & 1)                      # stream drained -> valid
        lat_cyc = int((words & 0xFFFF).max())                        # worst window decode latency this cycle
        return pred, last_empty, lat_cyc, dt

    # warm-up (first transfer pays driver/allocation setup; not scored)
    decode_cycle(exps[0][1])

    mon = LifetimeMonitor(slices, round_ns, clk_hz, C)
    n_cycles = cycles_arg if cycles_arg else len(exps)
    invalid = 0
    print("[box] running the logical qubit under the silicon decoder... (Ctrl-C to stop)\n")
    try:
        for i in range(n_cycles):
            truth, rounds = exps[i % len(exps)]  # cycle the pool if --cycles exceeds the vec
            pred, last_empty, lat_cyc, dt = decode_cycle(rounds)
            invalid += (last_empty == 0)
            # CLOSE THE LOOP: apply the decoder's correction to the tracked logical frame.
            survived = (pred == truth)
            mon.step(survived=survived, lat_cyc=lat_cyc, dt=dt)
            if (i + 1) % 500 == 0:
                print("\r" + mon.dashboard(), end="", flush=True)
            if (i + 1) % 5000 == 0:
                print()  # newline for scrollback
    except KeyboardInterrupt:
        print("\n[box] stopped by user.")
    print("\r" + mon.dashboard())
    if invalid:
        print("[box] WARNING: %d cycles did not drain (residual not empty) — decoder validity broken!" % invalid)
    text, ok = mon.summary(p, sw_rate, sw_ci)
    print(text)
    del ib, ob
    return 0 if (ok and invalid == 0) else 1


def main(argv):
    ap = argparse.ArgumentParser(description="Logical qubit in a box — closed-loop lifetime monitor")
    ap.add_argument("args", nargs="*", help="<design.bit> <vec>  (or just <vec> with --selfcheck)")
    ap.add_argument("--selfcheck", action="store_true", help="validate the monitor math off-board (no pynq)")
    ap.add_argument("--p", type=float, default=None, help="operating point to run (default: lowest p in the vec)")
    ap.add_argument("--cycles", type=int, default=None, help="number of memory cycles to run (default: all in vec)")
    ap.add_argument("--round-ns", type=int, default=1000, help="assumed physical syndrome-cycle time (ns)")
    ap.add_argument("--clk-hz", type=int, default=50_000_000, help="PL clock for decode-latency ns (default 50 MHz)")
    ns = ap.parse_args(argv[1:])

    bitfile = next((a for a in ns.args if a.endswith(".bit")), None)
    vec = next((a for a in ns.args if a.endswith(".vec")), "cosim_qubit_box_d3.vec")

    if ns.selfcheck:
        return run_selfcheck(vec, ns.p, ns.round_ns, ns.clk_hz)
    if not bitfile:
        print("usage: uf_qubit_in_a_box.py <design.bit> <vec> [--p P] [--cycles N]")
        print("   or: uf_qubit_in_a_box.py --selfcheck <vec> [--p P]")
        return 2
    return run_board(bitfile, vec, ns.p, ns.cycles, ns.round_ns, ns.clk_hz)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
