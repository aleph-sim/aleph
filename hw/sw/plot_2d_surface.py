#!/usr/bin/env python3
# Q6-31 — 2-D fidelity surface: algorithm fidelity vs (physical error rate p, T-gate count), real Arty.
# Reads the CSV from tcount_p_surface.sh (k,tcount,p,on,off) and renders docs/perf/qec-q6-2d-surface.png.
#
#   python3 hw/sw/plot_2d_surface.py <surface.csv>

import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "surface.csv"
rows = [r for r in csv.DictReader(open(path)) if r.get("on")]
TS = sorted({int(r["tcount"]) for r in rows})
PS = sorted({float(r["p"]) for r in rows})
ON = np.full((len(TS), len(PS)), np.nan)
OFF = np.full((len(TS), len(PS)), np.nan)
for r in rows:
    i, j = TS.index(int(r["tcount"])), PS.index(float(r["p"]))
    ON[i, j] = 100 * float(r["on"])
    OFF[i, j] = 100 * float(r["off"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Algorithm fidelity surface vs (physical error rate, T-gate count) — real Arty Z7-20 (d=3), silicon decoder",
             fontsize=12, fontweight="bold")

for ax, data, title, cmap in ((ax1, ON, "ON — decoder in the loop", "viridis"),
                              (ax2, OFF, "OFF — raw / no decoder", "magma")):
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   extent=[-0.5, len(PS) - 0.5, -0.5, len(TS) - 0.5])
    ax.set_xticks(range(len(PS)))
    ax.set_xticklabels([f"{p:g}" for p in PS])
    ax.set_yticks(range(len(TS)))
    ax.set_yticklabels([f"{t}" for t in TS])
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("T-gate count  (= 14·(k−1))")
    ax.set_title(title)
    for i in range(len(TS)):
        for j in range(len(PS)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, "%.1f" % v, ha="center", va="center",
                        color="white" if v < 55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="fidelity [%]", fraction=0.046, pad=0.04)

fig.text(0.5, 0.008,
         "Fidelity peaks at low p + low T-count and falls toward high p + high T-count. The decoder (left) holds the whole "
         "surface high; without it (right) it collapses — the gap between the two panels is the decoder's value.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("docs/perf/qec-q6-2d-surface.png", dpi=130)
print("wrote docs/perf/qec-q6-2d-surface.png")
