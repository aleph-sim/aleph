#!/usr/bin/env python3
# Q6-32 Milestone H — plot Shor with a DECODED inverse QFT (m=3). Left: the m=3 period-finding distribution
# for order of 2 mod 3 (peaks at {0,4} reveal r=2, finer 3-bit resolution). Right: where the decoded T-gates
# live -- modexp (1890) vs the newly-decoded inverse QFT controlled-S gates (6). Renders qec-q6-shorqft.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ON = 0.3549     # board P(peaks), fraction
OFF = 0.2536
FLOOR = 2.0 / 8     # 2 peaks of 8 outcomes = 25% random floor

# --- Left: m=3 phase-measurement distribution (peaks {0,4}) ---
IDEAL = [0.5, 0, 0, 0, 0.5, 0, 0, 0]
# illustrative ON: peak mass = ON split across {0,4}, rest spread over the other 6 outcomes
peakmass = ON / 2
rest = (1 - ON) / 6
ONd = [peakmass if y in (0, 4) else rest for y in range(8)]
OFFd = [1 / 8] * 8
x = np.arange(8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Shor with a decoded inverse QFT — m=3 order-finding of 2 mod 3 (r=2), real Arty Z7-20 (d=3), p=0.002",
             fontsize=11, fontweight="bold")

w = 0.27
ax1.bar(x - w, IDEAL, w, color="C7", label="ideal (perfect decoder)")
ax1.bar(x, ONd, w, color="C0", label="decoder ON  (P(peaks)=%.0f%%)" % (100 * ON))
ax1.bar(x + w, OFFd, w, color="C3", alpha=0.7, label="decoder OFF (washed out)")
ax1.axhline(FLOOR / 2, color="gray", ls=":", lw=1, alpha=0.6)
for xi in (0, 4):
    ax1.annotate("peak", (xi, 0.52), ha="center", fontsize=8, color="C0")
ax1.set_title("m=3 phase measurement: peaks {0,4} ⇒ 4/2³ = ½ ⇒ r = 2")
ax1.set_xlabel("measured phase outcome y  (m=3 phase qubits, 8 outcomes)")
ax1.set_ylabel("probability")
ax1.set_xticks(x)
ax1.set_ylim(0, 0.6)
ax1.legend(loc="upper right", fontsize=8)

# --- Right: where the decoded T-gates live ---
labels = ["modular\nexponentiation", "inverse QFT\n(controlled-S)"]
counts = [1890, 6]
colors = ["C4", "C1"]
bars = ax2.bar(labels, counts, color=colors)
ax2.bar_label(bars, fmt="%d T", fontsize=10)
ax2.set_title("decoded T-gates now span the QFT (m=3): 1890 modexp + 6 QFT = 1896 T")
ax2.set_ylabel("decoded T-gate count")
ax2.set_ylim(0, 2100)
ax2.text(0.5, 0.80,
         "Milestone G treated the inverse QFT as free Clifford glue, but controlled-S = diag(1,1,1,i)\n"
         "is non-Clifford (Clifford-hierarchy level 3). Here each is Clifford + 3 decoded T.\n\n"
         "Board: P(peaks) ON %.1f%% vs OFF %.1f%% (25%% random floor); r=2 still recovered.\n"
         "Caveat: for r=2 the QFT gates don't change the answer — this measures their decode COST."
         % (100 * ON, 100 * OFF),
         transform=ax2.transAxes, ha="center", va="top", fontsize=8,
         bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("docs/perf/qec-q6-shorqft.png", dpi=130)
print("wrote docs/perf/qec-q6-shorqft.png")
