"""Time one Stim surface-code cycle per committed .stim file. Writes
surface-stim.json: {"workloads": {"3": {"d":3,"qubits":17,"median_s":...}, ...}}.
Single-thread; median of N runs (default 50, the protocol in surface_code.md)."""
import argparse
import json
import statistics
import time
from pathlib import Path

import stim


def time_one(path: Path, runs: int) -> float:
    circuit = stim.Circuit(path.read_text())
    samples = []
    for _ in range(runs):
        sim = stim.TableauSimulator()
        t0 = time.perf_counter()
        sim.do(circuit)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=Path, default=Path(__file__).parent / "circuits")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    workloads = {}
    for d in [3, 5, 7, 9, 11]:
        p = args.circuits / f"surface_d{d}.stim"
        median = time_one(p, args.runs)
        workloads[str(d)] = {"d": d, "qubits": 2 * d * d - 1, "median_s": median}
    args.out.write_text(json.dumps({"workloads": workloads}, indent=2) + "\n")


if __name__ == "__main__":
    main()
