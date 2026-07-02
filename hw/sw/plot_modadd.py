#!/usr/bin/env python3
# Q6-32 Milestone B — plot the modular adder (VBE b:=(a+b) mod N) alongside Milestone A's plain adder,
# on the real Arty Z7-20. Both are REAL arithmetic algorithms whose T-count is intrinsic; the modular
# adder is the Shor-relevant primitive and reaches deep into the high-T region (140/210 T for n=2/3).
# Renders docs/perf/qec-q6-modadd.png.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Milestone A — plain ripple-carry adder b:=a+b  (14n T)
A_T = [28, 42, 56]
A_ON = [98.83, 95.57, 94.73]
A_OFF = [45.61, 31.77, 23.60]

# Milestone B — VBE modular adder b:=(a+b) mod N  (70n T)
B_T = [140, 210]          # n=2 (N=3), n=3 (N=7)
B_ON = [87.50, 81.46]
B_OFF = [4.06, 0.83]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
fig.suptitle("Real arithmetic algorithms from the silicon decoder — plain vs modular adder, real Arty Z7-20 (d=3), p=0.002",
             fontsize=11.5, fontweight="bold")

# Left: sum-register fidelity vs intrinsic T-count, both algorithms, ON vs OFF
ax1.plot(A_T, A_ON, "o-", color="C2", lw=2, ms=9, label="adder b:=a+b  — ON (decoder)")
ax1.plot(A_T, A_OFF, "o--", color="C2", lw=1.5, ms=7, alpha=0.55, label="adder — OFF (raw)")
ax1.plot(B_T, B_ON, "s-", color="C0", lw=2, ms=9, label="mod-adder b:=(a+b) mod N — ON")
ax1.plot(B_T, B_OFF, "s--", color="C0", lw=1.5, ms=7, alpha=0.55, label="mod-adder — OFF (raw)")
ax1.set_title("fidelity vs INTRINSIC T-count of a real algorithm")
ax1.set_xlabel("T-gate count  (14n plain / 70n modular — each a decode on the Arty)")
ax1.set_ylabel("output-register fidelity  [%]")
ax1.set_ylim(-3, 104)
ax1.grid(alpha=0.3)
ax1.legend(loc="center left", fontsize=8.5)

# Right: infidelity (1-fid), log-y — ON stays low deep into the high-T region; OFF saturates near 1
allT = A_T + B_T
ax2.semilogy(A_T, [max(1e-2, 100 - v) for v in A_ON], "o-", color="C2", lw=2, ms=9, label="adder — ON")
ax2.semilogy(B_T, [max(1e-2, 100 - v) for v in B_ON], "s-", color="C0", lw=2, ms=9, label="mod-adder — ON")
ax2.semilogy(A_T, [max(1e-2, 100 - v) for v in A_OFF], "o--", color="C2", lw=1.5, ms=7, alpha=0.55, label="adder — OFF")
ax2.semilogy(B_T, [max(1e-2, 100 - v) for v in B_OFF], "s--", color="C0", lw=1.5, ms=7, alpha=0.55, label="mod-adder — OFF")
ax2.set_title("infidelity (1 − fidelity) — compounding into the high-T region")
ax2.set_xlabel("T-gate count")
ax2.set_ylabel("1 − fidelity  [%]  (log scale)")
ax2.set_xticks(sorted(set(allT)))
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower right", fontsize=8.5)

fig.text(0.5, 0.005,
         "The modular adder — the primitive Shor stacks into modular exponentiation — has an intrinsic 70n T-count, reaching 140–210 T. "
         "The decoder keeps the mod-sum fidelity usable (87.5% at 140 T) where the undecoded circuit is destroyed (4% -> ~1%).",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("docs/perf/qec-q6-modadd.png", dpi=130)
print("wrote docs/perf/qec-q6-modadd.png")
