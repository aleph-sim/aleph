#!/usr/bin/env python3
"""Plot the Q0-05 surface-code threshold sweep and estimate the threshold p_th.

Reads the CSV emitted by the `qec_threshold` example (columns:
source,d,rounds,p,shots,logical_errors,rate,ci95 plus explicit ci95_lower/upper/method),
draws logical error rate vs p with asymmetric bounds and one
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
import math
import sys
import warnings
from collections import defaultdict


def load(path):
    """Return {d: (ps, rates, [lower_errors, upper_errors])} sorted by p.

    Accepts both the Q0-05 schema (no `decoder` column) and the Q1-04 schema (with it)."""
    by_d = defaultdict(list)
    shots = source = decoder = None
    warned_legacy = False
    shot_counts = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            d = int(row["d"])
            shots = int(row["shots"])
            shot_counts.add(shots)
            errors = int(row["logical_errors"])
            if not 0 <= errors <= shots:
                raise ValueError("invalid Bernoulli tally")
            rate = float(row["rate"])
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("invalid rate")
            if "ci95_lower" in row or "ci95_upper" in row:
                if not row.get("ci95_lower") or not row.get("ci95_upper"):
                    raise ValueError("missing confidence endpoint")
                if row.get("ci95_method") != "clopper-pearson":
                    raise ValueError("unsupported interval method")
                lower, upper = float(row["ci95_lower"]), float(row["ci95_upper"])
                if not (math.isfinite(lower) and math.isfinite(upper)
                        and 0 <= lower <= rate <= upper <= 1):
                    raise ValueError("invalid confidence bounds")
                ci = (rate - lower, upper - rate)
            else:
                if not warned_legacy:
                    warnings.warn("legacy CSV: ci95 is a Wald half-width; boundary/rare-event intervals are unreliable")
                    warned_legacy = True
                half = float(row["ci95"])
                if not math.isfinite(half) or half < 0:
                    raise ValueError("invalid legacy half-width")
                ci = (half, half)
            # Zero trials are no estimate, not a point at zero risk.
            if shots == 0:
                rate = math.nan
                ci = (math.nan, math.nan)
            by_d[d].append((float(row["p"]), rate, ci))
            source = row.get("source")
            decoder = row.get("decoder")
    out = {}
    for d, pts in by_d.items():
        pts.sort()
        ps = [a for a, _, _ in pts]
        rates = [b for _, b, _ in pts]
        cis = [[c[side] for _, _, c in pts] for side in (0, 1)]
        out[d] = (ps, rates, cis)
    return out, shots if len(shot_counts) == 1 else None, source, decoder


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
            if not (math.isfinite(p) and math.isfinite(r)):
                continue
            ds.append(d)
            ps.append(p)
            rates.append(r)
    if len(rates) < 5:
        return None
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
        ax.set_yscale("log")
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
        limit_label_used = False
        for d in sorted(by_d):
            ps, rates, cis = by_d[d]
            # A zero observed rate has no finite position on a log axis. Draw its
            # two-sided interval's upper endpoint as an upper-limit marker instead.
            upper_limits = [r == 0 and c > 0 for r, c in zip(rates, cis[1])]
            display_rates = [math.nan if limit else r for r, limit in zip(rates, upper_limits)]
            display_errors = [
                [math.nan if limit else c for c, limit in zip(cis[0], upper_limits)],
                [math.nan if limit else c for c, limit in zip(cis[1], upper_limits)],
            ]
            series = ax.errorbar(
                [p * 100 for p in ps], display_rates, yerr=display_errors,
                marker="o", capsize=3, label=f"d = {d}",
            )
            if any(upper_limits):
                # Limits are not estimates: do not connect them into the rate curve.
                limit_ps = [p * 100 for p, limit in zip(ps, upper_limits) if limit]
                limits = [c for c, limit in zip(cis[1], upper_limits) if limit]
                ax.errorbar(limit_ps, limits, yerr=[c * 0.5 for c in limits],
                            uplims=True, fmt="none", capsize=3,
                            color=series.lines[0].get_color(),
                            label=None if limit_label_used else "Zero failures: two-sided 95% upper endpoint")
                limit_label_used = True
        if p_cross is not None:
            ax.axvline(
                p_cross * 100, color="gray", ls="--", lw=1,
                label=f"$p_{{th}}\\approx{p_cross*100:.2f}\\%$",
            )
        ax.set_xlabel("physical error rate p (%)")
        ax.set_ylabel("logical error rate")
        dec_label = {"mwpm": "aleph-MWPM", "pymatching": "PyMatching"}.get(decoder, decoder or "MWPM")
        ax.set_title(
            f"Rotated surface-code memory-Z threshold\n({source} DEM, {dec_label}, "
            + (f"{shots:,} shots)" if shots is not None else "varying shots per cell)")
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
