#!/usr/bin/env python3
# Q6-30 — plot algorithm fidelity vs T-gate count (C^kX, k=2..5) on the real Arty Z7-20.
# Data from the on-board sweep. Renders docs/perf/qec-q6-tcount-scaling.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = [2, 3, 4, 5]
TCOUNT = [14, 28, 42, 56]
ON = [99.41, 97.07, 95.70, 94.53]
OFF = [69.43, 46.87, 36.21, 25.12]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Algorithm fidelity vs non-Clifford (T-gate) count — real Arty Z7-20 (d=3), p=0.002, silicon decoder in the loop",
             fontsize=11.5, fontweight="bold")

# Left: ON vs OFF, linear — the widening gap = compounding decoder value
ax1.plot(TCOUNT, ON, "o-", color="C0", lw=2, ms=9, label="ON — decoder in the loop")
ax1.plot(TCOUNT, OFF, "s--", color="C3", lw=2, ms=8, label="OFF — raw / no decoder")
ax1.fill_between(TCOUNT, OFF, ON, color="C0", alpha=0.08)
for x, on, off in zip(TCOUNT, ON, OFF):
    ax1.annotate("%.0fpp" % (on - off), (x, (on + off) / 2), fontsize=8, ha="center", color="gray")
ax1.set_title("C^kX truth-table fidelity  (k = 2,3,4,5)")
ax1.set_xlabel("T-gate count  (= 14·(k−1), each a decode on the Arty)")
ax1.set_ylabel("truth-table fidelity  [%]")
ax1.set_ylim(15, 102)
ax1.set_xticks(TCOUNT)
ax1.grid(alpha=0.3)
ax1.legend(loc="center right", fontsize=9)

# Right: infidelity (1 − fidelity) on log-y — ON grows ~linearly in T (exponential decay), steeper for OFF
ax2.semilogy(TCOUNT, [100 - v for v in ON], "o-", color="C0", lw=2, ms=9, label="ON — decoder in the loop")
ax2.semilogy(TCOUNT, [100 - v for v in OFF], "s--", color="C3", lw=2, ms=8, label="OFF — raw / no decoder")
ax2.set_title("infidelity (1 − fidelity) — compounding with T-count")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(TCOUNT)
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=9)

fig.text(0.5, 0.005,
         "Each added non-Clifford gate is another decode; infidelity compounds with T-count. "
         "ON stays near-ideal (99.4→94.5%) while undecoded OFF collapses (69→25%): the decoder's value compounds (gap 30→69pp).",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("docs/perf/qec-q6-tcount-scaling.png", dpi=130)
print("wrote docs/perf/qec-q6-tcount-scaling.png")
