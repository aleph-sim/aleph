#!/usr/bin/env python3
# Q6-32 Milestone D — plot the in-place modular multiplier U_a (140n^2 T) as the capstone of the full
# Shor arithmetic ladder on the real Arty Z7-20: adder (14n) -> modular adder (70n) -> modular multiplier
# (70n^2) -> in-place multiplier U_a (140n^2). Four real algorithms, T-count intrinsic and climbing to 560 T.
# Renders docs/perf/qec-q6-imul.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The full ladder, real Arty board points (d=3, p=0.002)
LADDER = [
    ("adder b:=a+b",            [28, 42, 56], [98.83, 95.57, 94.73], [45.61, 31.77, 23.60], "C2", "o"),
    ("mod-adder (a+b) mod N",   [140, 210],   [87.50, 81.46],        [4.06, 0.83],          "C0", "s"),
    ("mod-mult (a·x) mod N",    [280],        [73.06],               [0.68],                "C3", "D"),
    ("in-place U_a (a·x) mod N", [560],       [52.08],          [0.10],         "C4", "^"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("The full Shor arithmetic ladder from the silicon decoder — adder → modular adder → multiplier → in-place U_a, real Arty Z7-20 (d=3), p=0.002",
             fontsize=11, fontweight="bold")

for label, T, ON, OFF, color, marker in LADDER:
    ax1.plot(T, ON, marker + "-", color=color, lw=2, ms=9, label=label + " — ON")
    ax1.plot(T, OFF, marker + "--", color=color, lw=1.4, ms=7, alpha=0.5)
ax1.set_title("output fidelity vs INTRINSIC T-count (14n → 70n → 70n² → 140n²)")
ax1.set_xlabel("T-gate count  (each a decode on the Arty; dashed = OFF/undecoded)")
ax1.set_ylabel("output-register fidelity  [%]")
ax1.set_ylim(-3, 104)
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right", fontsize=7.5)

allT = sorted(set(t for _, T, *_ in LADDER for t in T))
for label, T, ON, OFF, color, marker in LADDER:
    ax2.semilogy(T, [max(1e-2, 100 - v) for v in ON], marker + "-", color=color, lw=2, ms=9, label=label + " — ON")
    ax2.semilogy(T, [max(1e-2, 100 - v) for v in OFF], marker + "--", color=color, lw=1.4, ms=7, alpha=0.5)
ax2.set_title("infidelity (1 − fidelity) — compounding up the whole ladder")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(allT)
ax2.set_xticklabels([str(t) for t in allT], fontsize=7.5)
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=7.5)

fig.text(0.5, 0.005,
         "Four real arithmetic algorithms up Shor's ladder. U_a — the complete in-place modular-multiply unitary whose controlled powers ARE modular exponentiation — reaches 560 T at n=2. "
         "The decoder keeps it usable where the undecoded circuit is annihilated (OFF ~0.1%). n=3 U_a (1260 T) is verified exact off-board.",
         ha="center", fontsize=8, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig("docs/perf/qec-q6-imul.png", dpi=130)
print("wrote docs/perf/qec-q6-imul.png")
