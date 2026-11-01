#!/usr/bin/env python3
"""Audit legacy output hashes and explicit relay budget/early-exit trade-offs.

Use --legacy-only against the baseline extension; compare hashes with the candidate.
This is a small CPU/API audit, not a statistically powered decoder-quality campaign.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time

import numpy as np
import aleph._native as native
import aleph.qec as qec
from test_qec import RELAY_BUDGET_DEM, xor_reduce_dem_text


def signature(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 1:
        parser.error("shots and repeats must be positive")
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error("set RAYON_NUM_THREADS=1 for reproducible per-core timing")
    cases = [("toy", qec.DetectorErrorModel(RELAY_BUDGET_DEM)),
             ("gross-circuit-r2-p0.003", qec.gross_code_dem(2, 0.003))]
    rows = []
    for label, dem in cases:
        # BP uses original mechanism columns. Reduce ^ supports before the matrix
        # helper, so sampling does not accidentally make correlated parts independent.
        H, O, rates = qec.dem_to_matrices(xor_reduce_dem_text(dem.to_dem_string()))
        faults = (np.random.default_rng(514).random((args.shots, len(rates))) < rates).astype(np.uint8)
        dets, truth = faults @ H.T % 2, faults @ O.T % 2
        modes = [("default", {}), ("legacy-6", {"legs": 6, "seed": 7})]
        if not args.legacy_only:
            modes += [("explicit-default", {"legs": 4, "iters_per_leg": 25, "early_exit": False}),
                      ("explicit-legacy-6", {"legs": 6, "iters_per_leg": 16,
                                             "early_exit": False, "seed": 7}),
                      ("6x10-full", {"legs": 6, "iters_per_leg": 10, "early_exit": False, "seed": 7}),
                      ("6x10-first", {"legs": 6, "iters_per_leg": 10, "early_exit": True, "seed": 7})]
        for name in ("relay-bp", "relay-bp-osd"):
            defaults = {}
            for mode, params in modes:
                decoder = qec.Decoder(dem, name, **params)
                decoder.decode_batch_errors(dets[:min(16, args.shots)])
                seconds = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    estimate, valid = decoder.decode_batch_errors(dets)
                    seconds.append(time.perf_counter() - start)
                predicted = estimate @ O.T % 2
                assert not np.any(np.any(estimate @ H.T % 2 != dets, axis=1) & valid)
                np.testing.assert_array_equal(decoder.decode_batch(dets), predicted.astype(bool))
                digest = signature(estimate, valid, predicted)
                if mode.startswith("explicit-"):
                    assert digest == defaults[mode.removeprefix("explicit-")]
                else:
                    defaults[mode] = digest
                rows.append({"case": label, "decoder": name, "mode": mode, "params": params,
                             "shots": args.shots, "detectors": H.shape[0], "columns": H.shape[1],
                             "seed": 514, "signature": digest, "nonconverged": int((~valid).sum()),
                             "logical_mismatches": int(np.any(predicted != truth, axis=1).sum()),
                             "seconds": seconds, "median_shots_per_second": args.shots/statistics.median(seconds)})
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "numpy": np.__version__, "native_module": native.__file__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
