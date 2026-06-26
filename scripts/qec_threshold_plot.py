#!/usr/bin/env python3
"""Plot the Q0-05 surface-code threshold sweep and estimate the threshold p_th.

Reads the CSV emitted by the `qec_threshold` example (columns:
source,d,rounds,p,shots,logical_errors,rate,ci95), draws logical error rate vs p with one
curve per distance, and estimates the threshold two ways:

  1. Curve crossing — the p where the small-d and large-d curves intersect (where the
     distance ordering of the rate inverts). Robust and assumption-free.
  2. Finite-size scaling collapse (Wang/Harrington/Preskill, quant-ph/0207088): fit
     p_logical = A + B·x + C·x² with x = (p - p_th)·d^(1/ν), optimising p_th, ν, A, B, C so
     all curves collapse onto one parabola. Reported when scipy is available.

Usage:
    python scripts/qec_threshold_plot.py docs/perf/data/qec-q0-threshold.csv \
        --out docs/perf/data/qec-q0-threshold.png
"""
import argparse
import csv
import sys
from collections import defaultdict


def load(path):
    """Return {d: (ps, rates, ci95s)} sorted by p, plus shot count, source, and decoder.

    Accepts both the Q0-05 schema (no `decoder` column) and the Q1-04 schema (with it)."""
    by_d = defaultdict(list)
    shots = source = decoder = None
    with open(path) as f:
        for row in csv.DictReader(f):
            d = int(row["d"])
            by_d[d].append((float(row["p"]), float(row["rate"]), float(row["ci95"])))
            shots = int(row["shots"])
            source = row.get("source")
            decoder = row.get("decoder")
    out = {}
    for d, pts in by_d.items():
        pts.sort()
        ps = [a for a, _, _ in pts]
        rates = [b for _, b, _ in pts]
        cis = [c for _, _, c in pts]
        out[d] = (ps, rates, cis)
    return out, shots, source, decoder


def crossing_threshold(by_d):
    """Estimate p_th as the mean pairwise crossing of adjacent-distance curves."""
    ds = sorted(by_d)
    crossings = []
    for d_lo, d_hi in zip(ds, ds[1:]):
        ps = by_d[d_lo][0]
        lo = by_d[d_lo][1]
        hi = by_d[d_hi][1]
        # diff = rate(d_hi) - rate(d_lo): negative below threshold (suppression), positive above.
        diff = [h - l for h, l in zip(hi, lo)]
        for i in range(len(diff) - 1):
            if diff[i] == 0.0:
                crossings.append(ps[i])
            elif diff[i] < 0.0 < diff[i + 1] or diff[i] > 0.0 > diff[i + 1]:
                # Linear interpolation of the zero crossing.
                t = diff[i] / (diff[i] - diff[i + 1])
                crossings.append(ps[i] + t * (ps[i + 1] - ps[i]))
    if not crossings:
        return None
    return sum(crossings) / len(crossings)


def scaling_threshold(by_d):
    """Finite-size scaling fit; returns (p_th, nu) or None if scipy is unavailable."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    ds, ps, rates = [], [], []
    for d, (pp, rr, _) in by_d.items():
        for p, r in zip(pp, rr):
            ds.append(d)
            ps.append(p)
            rates.append(r)
    ds = np.array(ds, float)
    ps = np.array(ps, float)
    rates = np.array(rates, float)

    def model(X, p_th, nu, a, b, c):
        d, p = X
        x = (p - p_th) * d ** (1.0 / nu)
        return a + b * x + c * x * x

    p0 = [0.03, 1.5, float(rates.mean()), 0.0, 0.0]
    try:
        popt, _ = curve_fit(
            model, (ds, ps), rates, p0=p0, maxfev=200000,
            bounds=([ps.min(), 0.5, -1, -50, -500], [ps.max(), 5.0, 1, 50, 500]),
        )
    except Exception as e:  # noqa: BLE001 — diagnostic only
        print(f"  (scaling fit failed: {e})", file=sys.stderr)
        return None
    return popt[0], popt[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None, help="PNG output path")
    ap.add_argument(
        "--overlay", default=None,
        help="reference CSV (e.g. the PyMatching-oracle sweep) drawn as faint dashed curves",
    )
    ap.add_argument("--overlay-label", default="PyMatching (ref)")
    args = ap.parse_args()

    by_d, shots, source, decoder = load(args.csv)

    p_cross = crossing_threshold(by_d)
    fit = scaling_threshold(by_d)

    print(f"decoder={decoder} source={source} shots={shots}")
    if p_cross is not None:
        print(f"threshold (curve crossing): p_th = {p_cross*100:.3f}%")
    if fit is not None:
        print(f"threshold (finite-size scaling): p_th = {fit[0]*100:.3f}%, nu = {fit[1]:.2f}")

    if args.overlay:
        ref_by_d, _, _, ref_dec = load(args.overlay)
        ref_cross = crossing_threshold(ref_by_d)
        ref_fit = scaling_threshold(ref_by_d)
        print(f"overlay decoder={ref_dec}:")
        if ref_cross is not None:
            print(f"  threshold (curve crossing): p_th = {ref_cross*100:.3f}%")
        if ref_fit is not None:
            print(f"  threshold (finite-size scaling): p_th = {ref_fit[0]*100:.3f}%, nu = {ref_fit[1]:.2f}")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipping plot", file=sys.stderr)
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        # Reference (oracle) curves first, faint dashed, so the native curves sit on top.
        if args.overlay:
            ref_by_d, _, _, _ = load(args.overlay)
            for i, d in enumerate(sorted(ref_by_d)):
                ps, rates, _ = ref_by_d[d]
                ax.plot(
                    [p * 100 for p in ps], rates,
                    ls="--", lw=1, alpha=0.45, color="gray", marker="x", ms=4,
                    label=args.overlay_label if i == 0 else None,
                )
        for d in sorted(by_d):
            ps, rates, cis = by_d[d]
            ax.errorbar(
                [p * 100 for p in ps], rates, yerr=cis,
                marker="o", capsize=3, label=f"d = {d}",
            )
        if p_cross is not None:
            ax.axvline(
                p_cross * 100, color="gray", ls="--", lw=1,
                label=f"$p_{{th}}\\approx{p_cross*100:.2f}\\%$",
            )
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate p (%)")
        ax.set_ylabel("logical error rate")
        dec_label = {"mwpm": "aleph-MWPM", "pymatching": "PyMatching"}.get(decoder, decoder or "MWPM")
        ax.set_title(
            f"Rotated surface-code memory-Z threshold ({source} DEM, {dec_label}, {shots:,} shots)"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
