#!/usr/bin/env python3
# Q6-32 Milestone A — plot ripple-carry adder (Cuccaro b:=a+b) fidelity vs T-gate count on the real
# Arty Z7-20. Unlike Q6-30's synthetic C^kX ladder, the T-count here is INTRINSIC to a real arithmetic
# circuit (n=2,3,4 -> 28,42,56 T). Renders docs/perf/qec-q6-adder.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = [2, 3, 4]
TCOUNT = [28, 42, 56]
ON = [98.83, 95.57, 94.73]      # filled from the board sweep
OFF = [45.61, 31.77, 23.60]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Ripple-carry adder (b:=a+b) fidelity vs T-count — real Arty Z7-20 (d=3), p=0.002, silicon decoder in the loop",
             fontsize=11.5, fontweight="bold")

# Left: ON vs OFF, linear — a REAL arithmetic algorithm's fidelity vs its intrinsic T-count
ax1.plot(TCOUNT, ON, "o-", color="C2", lw=2, ms=9, label="ON — decoder in the loop")
ax1.plot(TCOUNT, OFF, "s--", color="C3", lw=2, ms=8, label="OFF — raw / no decoder")
ax1.fill_between(TCOUNT, OFF, ON, color="C2", alpha=0.08)
for x, on, off in zip(TCOUNT, ON, OFF):
    ax1.annotate("%.0fpp" % (on - off), (x, (on + off) / 2), fontsize=8, ha="center", color="gray")
for x, nb in zip(TCOUNT, N):
    ax1.annotate("%d-bit" % nb, (x, 100.5), fontsize=8, ha="center", color="C2")
ax1.set_title("sum-register fidelity  (n = 2,3,4-bit adder)")
ax1.set_xlabel("T-gate count  (= 14·n, intrinsic to the adder — each a decode on the Arty)")
ax1.set_ylabel("sum-register fidelity  [%]")
ax1.set_ylim(15, 104)
ax1.set_xticks(TCOUNT)
ax1.grid(alpha=0.3)
ax1.legend(loc="center right", fontsize=9)

# Right: infidelity (1 − fidelity) on log-y — compounding with the algorithm's intrinsic T-count
ax2.semilogy(TCOUNT, [100 - v for v in ON], "o-", color="C2", lw=2, ms=9, label="ON — decoder in the loop")
ax2.semilogy(TCOUNT, [100 - v for v in OFF], "s--", color="C3", lw=2, ms=8, label="OFF — raw / no decoder")
ax2.set_title("infidelity (1 − fidelity) — compounding with T-count")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(TCOUNT)
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=9)

fig.text(0.5, 0.005,
         "A genuine arithmetic algorithm (Cuccaro adder, the Shor core), not a synthetic ladder: T-count is intrinsic (14n). "
         "ON holds near-ideal while undecoded OFF collapses toward random — the decoder's value compounds with the algorithm's size.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("docs/perf/qec-q6-adder.png", dpi=130)
print("wrote docs/perf/qec-q6-adder.png")
