#!/usr/bin/env python3
"""Independent Decimal Poisson-binomial mass checks and SciPy binomial interval checks."""
import decimal
import json
import re
import math
import sys
from pathlib import Path
from scipy.stats import beta
import scipy
import stim
import pymatching
import numpy as np


def mass_bounds(case):
    probs = [decimal.Decimal.from_float(float(p))
             for p in re.findall(r"^error\(([^)]+)\)", case["dem"], re.M)]
    kmax = case["max_faults"]
    weights = [decimal.Decimal(0)] * (kmax + 1)
    weights[0] = decimal.Decimal(1)
    tail = decimal.Decimal(0)
    for p in probs:
        tail += weights[-1] * p
        for k in range(kmax, -1, -1):
            weights[k] = weights[k] * (1 - p) + (weights[k - 1] * p if k else 0)
    for run in case["runs"]:
        result = run["stratified"]
        m = sum(row["mode"] == "ConditionalMonteCarlo" for row in result["strata"])
        alpha = (1 - result["confidence"]) / max(m, 1)
        for row, exact in zip(result["strata"], weights):
            weight = row["weight"]
            assert decimal.Decimal.from_float(weight["lower"]) <= exact
            assert decimal.Decimal.from_float(weight["upper"]) >= exact
            if row["mode"] == "ConditionalMonteCarlo":
                n, failures = row["decoder_calls"], row["failures"]
                lower = beta.ppf(alpha / 2, failures, n - failures + 1) if failures else 0.0
                upper = beta.ppf(1 - alpha / 2, failures + 1, n - failures) if failures < n else 1.0
                bounds = row["conditional"]
                assert abs(bounds["lower"] - lower) < 1e-10, (case["case"], row["faults"], "lower")
                assert abs(bounds["upper"] - upper) < 1e-10, (case["case"], row["faults"], "upper")
        # These fixed plans sample every possible stratum through kmax, so omitted mass is the tail.
        assert all(row["mode"] != "Unsampled" for row in result["strata"])
        assert decimal.Decimal.from_float(result["unsampled_mass"]["lower"]) <= tail
        assert decimal.Decimal.from_float(result["unsampled_mass"]["upper"]) >= tail
        assert result["decoder_calls"] == run["direct"]["decoder_calls"]
    return {"case": case["case"], "runs": len(case["runs"]), "mass_and_interval_checks": "PASS"}


def main():
    decimal.getcontext().prec = 100
    report = json.loads(Path(sys.argv[1]).read_text())
    coverage_checks = []
    for design, cov in report["coverage"].items():
        model = stim.DetectorErrorModel(cov["dem"])
        default_matching = pymatching.Matching.from_detector_error_model(model)
        errors = list(model.flattened())
        mechanisms = [e for e in errors if e.type == "error"]
        # This coverage DEM has only boundary mechanisms. Aleph retains distinct
        # logical-mask edges; PyMatching's DEM importer merges endpoints and keeps
        # the first mask. Explicit smallest-weight boundary edges reproduce the
        # minimum-cost classifier here, without changing sampling probabilities.
        matching = pymatching.Matching()
        for e in mechanisms:
            dets = [t.val for t in e.targets_copy() if t.is_relative_detector_id()]
            obs = [t.val for t in e.targets_copy() if t.is_logical_observable_id()]
            assert len(dets) <= 1
            if dets:
                p = e.args_copy()[0]
                matching.add_boundary_edge(dets[0], fault_ids=set(obs),
                                          weight=math.log1p(-p) - math.log(p),
                                          merge_strategy="smallest-weight")
        exact = decimal.Decimal(0)
        default_exact = decimal.Decimal(0)
        for mask in range(1 << len(mechanisms)):
            dets = np.zeros(model.num_detectors, dtype=np.uint8)
            truth = np.zeros(model.num_observables, dtype=np.uint8)
            probability = decimal.Decimal(1)
            for i, e in enumerate(mechanisms):
                p = decimal.Decimal.from_float(e.args_copy()[0])
                fired = bool(mask >> i & 1)
                probability *= p if fired else 1 - p
                if fired:
                    for t in e.targets_copy():
                        if t.is_relative_detector_id():
                            dets[t.val] ^= 1
                        elif t.is_logical_observable_id():
                            truth[t.val] ^= 1
            if np.any(matching.decode(dets) != truth):
                exact += probability
            if np.any(default_matching.decode(dets) != truth):
                default_exact += probability
        assert abs(float(exact) - cov["exact_failure_probability"]) < 1e-12
        covered = sum(lo <= float(exact) <= hi for lo, hi in cov["intervals"])
        assert covered == cov["covered"]
        trials = cov["trials"]
        lo = beta.ppf(0.025, covered, trials - covered + 1) if covered else 0.0
        hi = beta.ppf(0.975, covered + 1, trials - covered) if covered < trials else 1.0
        assert abs(cov["coverage_binomial95"]["lower"] - lo) < 1e-10
        assert abs(cov["coverage_binomial95"]["upper"] - hi) < 1e-10
        coverage_checks.append({"design": design, "exact_pymatching_aligned_truth": float(exact), "default_dem_import_truth": float(default_exact), "parallel_edge_policy": "explicit minimum-weight boundary classifier; default importer differs", "covered": covered, "trials": trials})
    print(json.dumps({"decimal_digits": 100, "scipy": scipy.__version__,
                      "cases": [mass_bounds(case) for case in report["cases"]],
                      "coverage_checks": coverage_checks}, indent=2))


if __name__ == "__main__":
    main()
