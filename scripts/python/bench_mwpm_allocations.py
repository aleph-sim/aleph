#!/usr/bin/env python3
"""Single-thread MWPM follow-up using bench_qec's exact surface fixtures/timing helper.

Run with RAYON_NUM_THREADS=1 and the chosen native module on PYTHONPATH.
Outputs raw JSON samples; compare baseline/candidate in alternating external runs.
"""
import argparse
import hashlib
import json
import os
import platform

import numpy as np
import pymatching
import stim
import aleph._native as native

import bench_qec as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error("set RAYON_NUM_THREADS=1 for a comparable per-core result")
    rows = []
    for distance, shots in [(5, 20000), (9, 5000)]:
        circuit = bench.surface(distance)
        sdem = circuit.detector_error_model(decompose_errors=True)
        decoder = bench.qec.Decoder(bench.qec.DetectorErrorModel(sdem), "mwpm")
        reference = pymatching.Matching.from_detector_error_model(sdem)
        packed = bench.sample(circuit, shots)
        dense = np.unpackbits(packed, axis=1, bitorder="little", count=sdem.num_detectors)
        samples = []
        for i in range(args.repeats):
            rates = {}
            # Alternate timing order to avoid always giving one decoder the hotter CPU.
            arms = [("aleph", decoder.decode_batch_bit_packed, packed),
                    ("pymatching", reference.decode_batch, dense)]
            for name, decode, syndromes in arms[::1 if i % 2 == 0 else -1]:
                rates[name] = bench.rate(decode, syndromes, shots)
            samples.append(rates)
        rows.append({"distance": distance, "shots": shots, "probability": 0.003,
                     "seed": 1, "prediction_sha256": hashlib.sha256(
                         decoder.decode_batch_bit_packed(packed).tobytes()).hexdigest(),
                     "samples_shots_per_second": samples})
    print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                      "native_module": native.__file__, "threads": 1,
                      "numpy_version": np.__version__, "stim_version": stim.__version__,
                      "pymatching_version": pymatching.__version__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
