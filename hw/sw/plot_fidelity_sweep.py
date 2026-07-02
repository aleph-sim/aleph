#!/usr/bin/env python3
# Q6-29 — plot algorithm fidelity vs physical error rate, measured on the real Arty Z7-20.
# Data from hw/sw/algo_fidelity_sweep.sh. Renders docs/perf/qec-q6-fidelity-sweep.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = [0.001, 0.002, 0.003, 0.005]
GROVER_ON = [94.24, 91.24, 86.41, 76.15]
GROVER_OFF = [52.95, 35.63, 26.99, 18.79]
GROVER_FOUND = [100.0, 99.6, 96.5, 90.6]
GROVER_IDEAL = 94.53
GROVER_UNIFORM = 12.5
TOFFOLI_ON = [99.94, 99.44, 98.56, 96.19]
TOFFOLI_OFF = [90.06, 83.75, 78.25, 69.94]
TOFFOLI_IDEAL = 100.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Logical algorithm fidelity vs physical error rate — real Arty Z7-20 (d=3), silicon decoder in the loop",
             fontsize=12, fontweight="bold")

# Grover
ax1.axhline(GROVER_IDEAL, ls=":", color="green", lw=1.5, label=f"ideal Grover ({GROVER_IDEAL:.1f}%)")
ax1.axhline(GROVER_UNIFORM, ls=":", color="gray", lw=1, label=f"uniform / no search ({GROVER_UNIFORM:.1f}%)")
ax1.plot(P, GROVER_ON, "o-", color="C0", lw=2, ms=8, label="ON — decoder in the loop")
ax1.plot(P, GROVER_FOUND, "^--", color="C2", lw=1.5, ms=6, label="ON — marked found (argmax)")
ax1.plot(P, GROVER_OFF, "s--", color="C3", lw=1.5, ms=6, label="OFF — raw / no decoder")
ax1.set_title("3-qubit Grover search (28 T-gate decodes)")
ax1.set_xlabel("physical error rate p")
ax1.set_ylabel("P(marked state)  [%]")
ax1.set_ylim(0, 105)
ax1.grid(alpha=0.3)
ax1.legend(loc="center left", fontsize=9)

# Toffoli
ax2.axhline(TOFFOLI_IDEAL, ls=":", color="green", lw=1.5, label=f"ideal Toffoli ({TOFFOLI_IDEAL:.0f}%)")
ax2.plot(P, TOFFOLI_ON, "o-", color="C0", lw=2, ms=8, label="ON — decoder in the loop")
ax2.plot(P, TOFFOLI_OFF, "s--", color="C3", lw=1.5, ms=6, label="OFF — raw / no decoder")
ax2.set_title("Logical Toffoli (7 T-gate decodes)")
ax2.set_xlabel("physical error rate p")
ax2.set_ylabel("truth-table fidelity  [%]")
ax2.set_ylim(60, 102)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower left", fontsize=9)

for ax in (ax1, ax2):
    ax.set_xticks(P)
    ax.set_xticklabels([f"{p:g}" for p in P])

fig.text(0.5, 0.005,
         "As p→0 the decoder drives the algorithm to its ideal output (at p=0.001: Grover 94.2%≈94.5%, Toffoli 99.9%≈100%). "
         "A lower effective LER — a better/larger decoder — directly buys algorithm fidelity.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig("docs/perf/qec-q6-fidelity-sweep.png", dpi=130)
print("wrote docs/perf/qec-q6-fidelity-sweep.png")
