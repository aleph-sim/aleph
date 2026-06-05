#!/usr/bin/env python3
"""Extract criterion medians into the unified Phase-4 aleph results JSON.

Reads target/criterion/<group>/<param>/new/estimates.json for each parameter
(the qubit count) and writes {workloads: {<family>_n<n>: {n, family,
aleph_ms_median, aleph_rsd}}}. Deterministic; no network.
"""
import argparse
import json
from pathlib import Path


def extract(criterion_root: Path, group: str, family: str) -> dict:
    workloads = {}
    group_dir = criterion_root / group
    for param_dir in sorted(group_dir.iterdir()):
        if not param_dir.is_dir() or param_dir.name == "report":
            continue
        est = param_dir / "new" / "estimates.json"
        if not est.exists():
            continue
        e = json.loads(est.read_text())
        median_ns = float(e["median"]["point_estimate"])
        std_ns = float(e["std_dev"]["point_estimate"])
        n = int(param_dir.name)
        workloads[f"{family}_n{n}"] = {
            "n": n,
            "family": family,
            "aleph_ms_median": median_ns / 1e6,
            "aleph_rsd": (std_ns / median_ns) if median_ns else 0.0,
        }
    return {"schema_version": 1, "workloads": workloads}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion-root", required=True, type=Path)
    ap.add_argument("--group", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = extract(args.criterion_root, args.group, args.family)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[extract] {len(data['workloads'])} workloads -> {args.out}")


if __name__ == "__main__":
    main()
