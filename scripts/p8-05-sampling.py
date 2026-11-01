#!/usr/bin/env python3
"""Bounded, sampling-only sweep. Build sampling_audit first; services stay running.

python3 scripts/p8-05-sampling.py target/release/examples/sampling_audit results.jsonl
Add --profile for synchronized stage attribution (separate from throughput runs).
"""
import argparse
import itertools
import json
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary")
parser.add_argument("output")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()


def gpu():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


sizes = [22] if args.profile else [14, 18, 22]
shots = [1024, 65536] if args.profile else [1, 1024, 65536, 1000000]
modes = ["reference", "hierarchical", "block", "auto"] if args.profile else ["reference", "hierarchical", "block"]
with open(args.output, "w") as output:
    for precision, n, count, mode in itertools.product(["f64", "f32"], sizes, shots, modes):
        for attempt in range(5):
            before = gpu()
            if int(before.split(",")[0]) > 5:
                time.sleep(2)
                continue
            result = subprocess.run(
                [args.binary, precision, mode, str(n), str(count), "9", str(int(args.profile))],
                text=True, capture_output=True, check=True,
            )
            row = json.loads(result.stdout)
            row["gpu_before"] = before
            row["gpu_after"] = gpu()
            output.write(json.dumps(row) + "\n")
            output.flush()
            break
        else:
            raise RuntimeError("GPU busy; stopped timing without changing services")
