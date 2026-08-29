#!/usr/bin/env python3
"""Generate the three paper figures (schedule.pdf, herald.pdf, latency.pdf).

Run from the repo root:
    scripts/qiskit-baseline/.venv/bin/python paper/figs/make_figs.py

Two-column arXiv layout: 3.4 in wide, 8-9 pt text, colour-blind-safe
(Okabe-Ito) palette, tight bbox, PDF output. Every literal number carries a
comment naming the file (and section/line) it was verified against.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BUDGET_CSV = os.path.join(ROOT, "docs", "perf", "data", "qec-q7-budget.csv")

# Okabe-Ito, colour-blind safe (fixed assignment order, never cycled).
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY = "#7F7F7F"

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
W = 3.4  # in

used = []  # (figure, label, value, source) rows for the audit table


def note(fig, label, value, src):
    used.append((fig, label, value, src))


# ---------------------------------------------------------------------------
# 1. schedule.pdf — LER ratio vs latency, per p, per (legs,iters)
# ---------------------------------------------------------------------------
def fig_schedule():
    rows = list(csv.DictReader(open(BUDGET_CSV)))
    for r in rows:
        for k in ("p", "latency_us", "ratio_to_base"):
            r[k] = float(r[k])
        r["legs"] = int(r["legs"])
        r["iters"] = int(r["iters"])
        r["within_ci"] = int(r["within_ci"])
    ps = sorted({r["p"] for r in rows})
    pcol = {ps[0]: C_BLUE, ps[1]: C_ORANGE, ps[2]: C_GREEN}
    # marker encodes legs (secondary encoding so identity is not colour-alone)
    legs_marker = {2: "v", 3: "s", 4: "o", 5: "D", 6: "*"}

    fig, ax = plt.subplots(figsize=(W, 2.9))
    for p in ps:
        sub = sorted((r for r in rows if r["p"] == p), key=lambda r: r["latency_us"])
        # Series line through the legs=4 iteration sweep (the main knob).
        s4 = [r for r in sub if r["legs"] == 4]
        ax.plot(
            [r["latency_us"] for r in s4],
            [r["ratio_to_base"] for r in s4],
            color=pcol[p],
            lw=0.9,
            alpha=0.6,
            zorder=1,
        )
        for r in sub:
            face = pcol[p] if r["within_ci"] else "white"
            ax.scatter(
                r["latency_us"],
                r["ratio_to_base"],
                marker=legs_marker[r["legs"]],
                s=28 if r["legs"] != 6 else 60,
                facecolor=face,
                edgecolor=pcol[p],
                linewidth=0.9,
                zorder=3,
            )
    # Adopted point: 6 legs x 10 iters (present at every p; same latency).
    adopted = [r for r in rows if r["legs"] == 6 and r["iters"] == 10]
    if adopted:
        x = adopted[0]["latency_us"]
        ax.axvline(x, color=C_GRAY, lw=0.7, ls=":", zorder=0)
        ax.annotate(
            "adopted 6 legs × 10 iters",
            xy=(x, max(r["ratio_to_base"] for r in adopted)),
            xytext=(1.3, 1.53),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.6),
        )
        note("schedule", "adopted 6x10 latency_us", x, "qec-q7-budget.csv (legs=6,iters=10 rows)")
    ax.axhline(1.0, color=C_GRAY, lw=0.6, zorder=0)
    ax.set_xlabel("decode latency (µs)")
    ax.set_ylabel("LER / LER(full schedule)")
    ax.set_ylim(0.95, 1.6)

    # Legends: colour = p (line), marker = legs (open = outside 95% CI).
    from matplotlib.lines import Line2D

    h_p = [Line2D([], [], color=pcol[p], lw=1.2, label=f"p = {p:g}") for p in ps]
    h_l = [
        Line2D(
            [],
            [],
            marker=m,
            color=C_GRAY,
            ls="",
            markersize=5 if l != 6 else 7,
            markerfacecolor="white",
            label=f"{l} legs",
        )
        for l, m in legs_marker.items()
    ]
    h_ci = [
        Line2D([], [], marker="o", color=C_GRAY, ls="", markerfacecolor=C_GRAY, markersize=5, label="within CI"),
        Line2D([], [], marker="o", color=C_GRAY, ls="", markerfacecolor="white", markersize=5, label="outside CI"),
    ]
    leg1 = ax.legend(handles=h_p, loc="upper right", handlelength=1.5)
    ax.add_artist(leg1)
    ax.legend(handles=h_l + h_ci, loc="upper center", ncol=4, columnspacing=0.8, handletextpad=0.3,
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    out = os.path.join(HERE, "schedule.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. herald.pdf — LER and non-convergence rate vs p (Q7-07)
# ---------------------------------------------------------------------------
# Source: docs/perf/data/qec-q7-nonconv-block.csv (columns p, r, r_ci95, ler,
# ler_ci95, p_err_given_conv), reproduced in
# docs/qec/q7-07-nonconvergence-policy.md § "AC-1 — non-convergence rate, block
# path (primary)" and § "The ceiling — A(p), and what it forbids".
# [[144,12,12]], circuit-level rounds=1, 6 legs x 10 iters, Q4.3, 1e6 shots/pt.
HERALD = [
    # p,     r=P(valid=0), r_ci95,     LER,        ler_ci95,   P(err|valid=1)
    (0.003, 0.00116800, 0.00006695, 0.00083200, 0.00005651, None),  # 0/998832 -> <=3.0e-6 (rule of three)
    (0.005, 0.00847300, 0.00017965, 0.00705200, 0.00016401, 0.00001916),
    (0.007, 0.03259100, 0.00034802, 0.02877400, 0.00032765, 0.00011784),
]
P_ERR_CONV_UB_003 = 3.0e-6  # q7-07-nonconvergence-policy.md § ceiling: 3/998832


def fig_herald():
    p = [h[0] for h in HERALD]
    r = [h[1] for h in HERALD]
    rci = [h[2] for h in HERALD]
    ler = [h[3] for h in HERALD]
    lci = [h[4] for h in HERALD]
    for h in HERALD:
        note("herald", f"p={h[0]} r", h[1], "qec-q7-nonconv-block.csv")
        note("herald", f"p={h[0]} LER", h[3], "qec-q7-nonconv-block.csv")
        note("herald", f"p={h[0]} P(err|valid=1)", h[5] if h[5] is not None else f"<= {P_ERR_CONV_UB_003}",
             "qec-q7-nonconv-block.csv / q7-07 md § ceiling")

    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.errorbar(p, r, yerr=rci, color=C_ORANGE, marker="s", markersize=5, capsize=2,
                label="non-convergence rate  P(valid=0)", zorder=3)
    ax.errorbar(p, ler, yerr=lci, color=C_BLUE, marker="o", markersize=5, capsize=2,
                markerfacecolor="white", label="logical error rate", zorder=4)
    # Converged-and-wrong: measured at p=0.005, 0.007; upper bound at 0.003.
    pc = [h[0] for h in HERALD if h[5] is not None]
    vc = [h[5] for h in HERALD if h[5] is not None]
    ax.plot(pc, vc, color=C_VERM, marker="^", markersize=5, ls="--",
            label="P(err | valid=1)", zorder=3)
    ax.plot([0.003], [P_ERR_CONV_UB_003], color=C_VERM, marker="v", markersize=5, ls="", zorder=3)
    ax.annotate("≤ 3.0e-6\n(0 / 998 832)", xy=(0.003, P_ERR_CONV_UB_003), xytext=(0.00335, 2.2e-6),
                fontsize=7, color=C_VERM, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("rate per shot")
    ax.set_xticks(p)
    ax.set_xlim(0.0025, 0.0075)
    ax.set_ylim(1e-6, 1e-1)
    ax.legend(loc="upper left", handlelength=2.0)
    fig.tight_layout()
    out = os.path.join(HERE, "herald.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. latency.pdf — decode latency vs geometry / target
# ---------------------------------------------------------------------------
# Each tuple: (target, geometry, cycles, MHz, latency_us, source)
LATENCY = [
    # KV260 M8 bitstream: 2085 cyc @ 133.332 MHz = 15.64 us
    # docs/perf/qec-q7-fixed-bp.md  M8 (lines ~1220, 1462, 1799)
    ("KV260", "16/48", 2085, 133.332, 15.64, "docs/perf/qec-q7-fixed-bp.md M8"),
    # VU47P full-parallel: docs/perf/q7-02-fullparallel-fpga.md (table rows 210-211, 229-230)
    ("VU47P", "64/192", 913, 150.4, 6.07, "docs/perf/q7-02-fullparallel-fpga.md"),
    ("VU47P", "144/864", 543, 97.3, 5.58, "docs/perf/q7-02-fullparallel-fpga.md"),
    # ASAP7 16/48: Fmax 686.13 MHz (docs/perf/q7-02-asap7-timing.md 6_finish.rpt, line 25/44),
    # 2085-cycle schedule (same file line 188/210); 2085/686.13 = 3.04 us
    # (matches docs/perf/q7-02-b3-asap7-fullparallel.md table line 35).
    ("ASAP7", "16/48", 2085, 686.13, 2085 / 686.13, "docs/perf/q7-02-asap7-timing.md"),
    # ASAP7 144/864: 543 cyc @ 614.59 MHz = 0.88 us
    # docs/perf/q7-02-b3-asap7-fullparallel.md (lines 3, 40)
    ("ASAP7", "144/864", 543, 614.59, 0.88, "docs/perf/q7-02-b3-asap7-fullparallel.md"),
]


def fig_latency():
    for t, g, cyc, mhz, us, src in LATENCY:
        note("latency", f"{t} {g}", f"{cyc} cyc @ {mhz} MHz = {us:.2f} us", src)

    geoms = ["16/48", "64/192", "144/864"]
    targets = ["KV260", "VU47P", "ASAP7"]
    tcol = {"KV260": C_BLUE, "VU47P": C_ORANGE, "ASAP7": C_GREEN}
    bw = 0.92  # slot pitch; a 7 pt "133.3 MHz" label (~33 pt) must clear the neighbouring bar
    fig, ax = plt.subplots(figsize=(W, 2.7))
    seen = set()
    for gi, g in enumerate(geoms):
        present = [x for x in LATENCY if x[1] == g]
        n = len(present)
        for k, (t, _, cyc, mhz, us, _) in enumerate(present):
            x = gi + (k - (n - 1) / 2) * bw
            ax.bar(x, us, width=bw - 0.2, color=tcol[t], edgecolor="white", linewidth=0.5,
                   label=None if t in seen else t, zorder=3)
            seen.add(t)
            ax.text(x, us * 1.10, f"{us:.2f} µs\n{cyc} cyc\n{mhz:.1f} MHz", ha="center", va="bottom",
                    fontsize=7, linespacing=1.0, color="#333333", zorder=4)
    ax.axhline(1.0, color=C_GRAY, lw=0.7, ls=":", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 60)
    ax.set_xlim(-0.98, 2.98)
    ax.set_xticks(range(len(geoms)))
    ax.set_xticklabels(geoms)
    ax.set_xlabel("decoder geometry (parallel units)")
    ax.set_ylabel("decode latency (µs)")
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(t) for t in targets if t in ls]
    ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="lower center", ncol=3,
              columnspacing=1.5, handlelength=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    out = os.path.join(HERE, "latency.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    outs = [fig_schedule(), fig_herald(), fig_latency()]
    print("\nLiterals used:")
    print(f"{'figure':9} {'label':26} {'value':36} source")
    for f, l, v, s in used:
        print(f"{f:9} {l:26} {str(v):36} {s}")
    print()
    for o in outs:
        print(f"{o}: {os.path.getsize(o)} bytes")
