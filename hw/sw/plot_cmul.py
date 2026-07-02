#!/usr/bin/env python3
# Q6-32 Milestone E — plot the controlled in-place modular multiplier c-U_a (the exact Shor exponentiation
# step) as the fifth rung of the arithmetic ladder on the real Arty Z7-20:
# adder (14n) -> modular adder (70n) -> modular multiplier (70n^2) -> in-place U_a (140n^2) -> c-U_a (630 T
# at n=2, control turns loads into Toffolis + SWAP into a Fredkin). Renders docs/perf/qec-q6-cmul.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The full ladder, real Arty board points (d=3, p=0.002)
LADDER = [
    ("adder b:=a+b",             [28, 42, 56], [98.83, 95.57, 94.73], [45.61, 31.77, 23.60], "C2", "o"),
    ("mod-adder (a+b) mod N",    [140, 210],   [87.50, 81.46],        [4.06, 0.83],          "C0", "s"),
    ("mod-mult (a·x) mod N",     [280],        [73.06],               [0.68],                "C3", "D"),
    ("in-place U_a (a·x) mod N",  [560],       [52.08],               [0.10],                "C4", "^"),
    ("controlled c-U_a (Shor step)", [630],    [50.31],          [0.12],         "C1", "*"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Shor's arithmetic ladder from the silicon decoder — up to the exact exponentiation step c-U_a, real Arty Z7-20 (d=3), p=0.002",
             fontsize=11, fontweight="bold")

for label, Tc, ON, OFF, color, marker in LADDER:
    ms = 14 if marker == "*" else 9
    ax1.plot(Tc, ON, marker + "-", color=color, lw=2, ms=ms, label=label + " — ON")
    ax1.plot(Tc, OFF, marker + "--", color=color, lw=1.4, ms=ms - 2, alpha=0.5)
ax1.set_title("output fidelity vs INTRINSIC T-count (14n → 70n → 70n² → 140n² → c-U_a)")
ax1.set_xlabel("T-gate count  (each a decode on the Arty; dashed = OFF/undecoded)")
ax1.set_ylabel("output-register fidelity  [%]")
ax1.set_ylim(-3, 104)
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right", fontsize=7)

allT = sorted(set(t for _, Tc, *_ in LADDER for t in Tc))
for label, Tc, ON, OFF, color, marker in LADDER:
    ms = 14 if marker == "*" else 9
    ax2.semilogy(Tc, [max(1e-2, 100 - v) for v in ON], marker + "-", color=color, lw=2, ms=ms, label=label + " — ON")
    ax2.semilogy(Tc, [max(1e-2, 100 - v) for v in OFF], marker + "--", color=color, lw=1.4, ms=ms - 2, alpha=0.5)
ax2.set_title("infidelity (1 − fidelity) — compounding up the whole ladder")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(allT)
ax2.set_xticklabels([str(t) for t in allT], fontsize=7, rotation=45)
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=7)

fig.text(0.5, 0.005,
         "Five real arithmetic algorithms up Shor's ladder, capped by the controlled multiplier c-U_a — the literal modular-exponentiation step (period-finding = a product of controlled-U_{a^{2^k}}). "
         "The control turns the constant-loads into Toffolis and the SWAP into a Fredkin (630 T at n=2); the decoder keeps it usable where the raw circuit is annihilated.",
         ha="center", fontsize=7.6, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig("docs/perf/qec-q6-cmul.png", dpi=130)
print("wrote docs/perf/qec-q6-cmul.png")
