#!/usr/bin/env python3
"""Offline SciPy reference, exact finite-sum and repeated-sampling coverage checks.

Requires numpy/scipy; the interval implementation under test is the supplied Rust binary.
Run: cargo build -p aleph-qec --example binomial_intervals
     python scripts/python/check_binomial_intervals.py --binary target/debug/examples/binomial_intervals --output results.json
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import scipy
from scipy.stats import beta, binom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed, repeats = 566, 20000
    rng = np.random.default_rng(seed)
    exact_ns = [1, 2, 5, 10, 20, 100]
    ps = [0, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.2, 0.5, 0.9, 0.99, 0.9999, 1]
    sampled = []
    pairs = {(n, k) for n in exact_ns for k in range(n + 1)}
    for n in [1000, 1000000]:
        for p in [0.000001, 0.0001, 0.01, 0.5, 0.9999]:
            draws = rng.binomial(n, p, size=repeats)
            pairs.update((n, int(k)) for k in np.unique(draws))
            sampled.append((n, p, draws))
    pairs.update([(0, 0), (1000000000, 0), (1000000000, 1), (1000000000, 1000)])
    result = subprocess.run([args.binary], input="".join(f"{n} {k}\n" for n, k in sorted(pairs)),
                            text=True, capture_output=True, check=True, timeout=120)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != len(pairs):
        raise AssertionError("missing evaluator results")
    bounds, numerical_rejections, max_error = {}, [], 0.0
    for row in rows:
        n, k = int(row["shots"]), int(row["errors"])
        if row["status"] != "ok":
            numerical_rejections.append({"shots": n, "errors": k, "reason": row["status"]})
            if n <= 1000000:
                raise AssertionError(row)
            continue
        lo, hi = float(row["lower"]), float(row["upper"])
        ref_lo = 0 if k == 0 else beta.ppf(0.025, k, n-k+1)
        ref_hi = 1 if k == n else beta.isf(0.025, k+1, n-k)
        if n == 0:
            ref_lo, ref_hi = 0, 1
        error = max(abs(lo-ref_lo), abs(hi-ref_hi))
        if error > 1e-8:
            raise AssertionError((n, k, lo, hi, ref_lo, ref_hi))
        max_error = max(max_error, error)
        bounds[n, k] = (lo, hi)
    exact = []
    for n in exact_ns:
        for p in ps:
            coverage = sum(binom.pmf(k, n, p) for k in range(n+1)
                           if bounds[n, k][0] <= p <= bounds[n, k][1])
            if coverage < 0.95 - 1e-12:
                raise AssertionError((n, p, coverage))
            exact.append({"shots": n, "probability": p, "coverage": float(coverage)})
    simulation = []
    for n, p, draws in sampled:
        hits = sum(bounds[n, int(k)][0] <= p <= bounds[n, int(k)][1] for k in draws)
        coverage = hits / repeats
        se = (coverage * (1-coverage) / repeats) ** 0.5
        if coverage + 5*se < 0.95:
            raise AssertionError((n, p, coverage, se))
        simulation.append({"shots": n, "probability": p, "repeats": repeats,
                           "coverage": coverage, "monte_carlo_standard_error": se})
    report = {"method": "two-sided equal-tailed Clopper-Pearson 95%", "seed": seed,
              "scipy": scipy.__version__, "numpy": np.__version__,
              "reference_cases": len(bounds), "max_absolute_reference_error": max_error,
              "numerical_rejections": numerical_rejections,
              "exact_coverage": exact, "simulated_coverage": simulation}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["reference_cases", "max_absolute_reference_error", "numerical_rejections"]}))


if __name__ == "__main__":
    main()
