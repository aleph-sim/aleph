#!/usr/bin/env python3
# Q6-32 Milestone C — plot the modular multiplier (VBE y:=(a*x) mod N) alongside the plain adder
# (Milestone A) and modular adder (Milestone B), on the real Arty Z7-20. Three real arithmetic algorithms
# up Shor's ladder, T-count intrinsic and climbing: 14n -> 70n -> 70n^2 (to 280 T at n=2).
# Renders docs/perf/qec-q6-mulmod.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Milestone A — plain ripple-carry adder b:=a+b  (14n T)
A_T = [28, 42, 56]
A_ON = [98.83, 95.57, 94.73]
A_OFF = [45.61, 31.77, 23.60]

# Milestone B — VBE modular adder b:=(a+b) mod N  (70n T)
B_T = [140, 210]
B_ON = [87.50, 81.46]
B_OFF = [4.06, 0.83]

# Milestone C — VBE modular multiplier y:=(a*x) mod N  (70n^2 T); n=2 on board, n=3 off-board verified
C_T = [280]
C_ON = [73.06]
C_OFF = [0.68]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Up Shor's ladder from the silicon decoder — adder → modular adder → modular multiplier, real Arty Z7-20 (d=3), p=0.002",
             fontsize=11.5, fontweight="bold")

def series(ax, T, ON, OFF, color, marker, label):
    ax.plot(T, ON, marker + "-", color=color, lw=2, ms=9, label=label + " — ON")
    ax.plot(T, OFF, marker + "--", color=color, lw=1.5, ms=7, alpha=0.5, label=label + " — OFF")

series(ax1, A_T, A_ON, A_OFF, "C2", "o", "adder b:=a+b")
series(ax1, B_T, B_ON, B_OFF, "C0", "s", "mod-adder (a+b) mod N")
series(ax1, C_T, C_ON, C_OFF, "C3", "D", "mod-mult (a·x) mod N")
ax1.set_title("fidelity vs INTRINSIC T-count (14n → 70n → 70n²)")
ax1.set_xlabel("T-gate count  (each a decode on the Arty)")
ax1.set_ylabel("output-register fidelity  [%]")
ax1.set_ylim(-3, 104)
ax1.grid(alpha=0.3)
ax1.legend(loc="center right", fontsize=8)

allT = A_T + B_T + C_T
for T, ON, OFF, color, marker, label in [
    (A_T, A_ON, A_OFF, "C2", "o", "adder"),
    (B_T, B_ON, B_OFF, "C0", "s", "mod-adder"),
    (C_T, C_ON, C_OFF, "C3", "D", "mod-mult"),
]:
    ax2.semilogy(T, [max(1e-2, 100 - v) for v in ON], marker + "-", color=color, lw=2, ms=9, label=label + " — ON")
    ax2.semilogy(T, [max(1e-2, 100 - v) for v in OFF], marker + "--", color=color, lw=1.5, ms=7, alpha=0.5, label=label + " — OFF")
ax2.set_title("infidelity (1 − fidelity) — compounding up the ladder")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(sorted(set(allT)))
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=8)

fig.text(0.5, 0.005,
         "Three real arithmetic algorithms up Shor's ladder: the modular multiplier (the operation modular exponentiation is built from) reaches 280 T at n=2. "
         "The decoder keeps the product usable where the undecoded circuit is annihilated (OFF ~0.7%). n=3 (630 T) is verified exact off-board.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("docs/perf/qec-q6-mulmod.png", dpi=130)
print("wrote docs/perf/qec-q6-mulmod.png")
