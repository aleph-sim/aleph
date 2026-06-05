"""Qiskit Aer baseline harness for aleph Phase 1.

Builds the full Phase-1 matrix — {GHZ, QFT, Grover, random-brickwall} ×
n ∈ {15, 20, 22, 25} — transpiles each to the basis aleph-parser supports,
exports QASM3 (committed under circuits/), and times
AerSimulator(method='statevector') under single-thread pinning.

Flags: `--gen-only` regenerates circuits without timing; `--workloads a,b`
times only the named subset (default: full matrix).

Specs: docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md
       docs/superpowers/specs/2026-05-30-p1-14-phase1-perf-report-design.md
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import QFT, GroverOperator
from qiskit_aer import AerSimulator

# Grover-10 transpiles to ~192k gates (well over the spec's 100k threshold).
# Per design doc section 8, drop to 5 iters; the kernel mix is identical, the
# wall-clock just halves.  Documented choice; not a defect.
# Per-family qubit sizes. QFT extends to the P4-01 matrix {10,15,20,25,30}
# (n=30 is the AC ceiling, measured on the EPYC box); other families keep the
# Stage-0 sizes. A global list would regenerate every family and make
# Grover/random at n=30 intractable.
FAMILY_SIZES = {
    "ghz": [15, 20, 22, 25],
    "qft": [10, 15, 20, 25, 30],
    "grover": [15, 20, 22, 25],
    "random_brickwall": [15, 20, 22, 25],
}
# Union for the results header (sorted, de-duplicated).
N_QUBITS_LIST = sorted({n for sizes in FAMILY_SIZES.values() for n in sizes})
GROVER_ITERS = 5
RANDOM_DEPTH = 20
BASIS_GATES = ["h", "x", "z", "rz", "rx", "ry", "cx", "cz", "ccx", "p"]

CIRCUITS_DIR = Path(__file__).parent / "circuits"
RESULTS_PATH = Path(__file__).parent / "results-qiskit.json"


def build_qft(n: int) -> QuantumCircuit:
    """Textbook QFT on `n` qubits, no closing SWAPs (matches aleph_benches::qft_circuit)."""
    qc = QuantumCircuit(n, name=f"qft_n{n}")
    qc.compose(QFT(num_qubits=n, do_swaps=False, inverse=False), inplace=True)
    return qc


def build_grover(n: int, iters: int) -> QuantumCircuit:
    """Grover on `n` qubits with 1 marked state |0...01>, applied `iters` times.

    Oracle: phase-flip on |0...01>. We encode it by X-ing qubits [1..n) so that
    the target maps to the all-ones state, then a multi-controlled Z on the
    full register (H + mcx + H sandwich), then undo the X layer.
    """
    oracle = QuantumCircuit(n, name="oracle")
    for q in range(1, n):
        oracle.x(q)
    oracle.h(0)
    oracle.mcx(list(range(1, n)), 0)
    oracle.h(0)
    for q in range(1, n):
        oracle.x(q)

    grover_op = GroverOperator(oracle, insert_barriers=False)

    qc = QuantumCircuit(n, name=f"grover_n{n}_iters{iters}")
    # Initial superposition.
    qc.h(range(n))
    # Iters x Grover operator.
    for _ in range(iters):
        qc.compose(grover_op, inplace=True)
    return qc


def build_random_brickwall(n: int, depth: int) -> QuantumCircuit:
    """Brick-wall random circuit mirroring `aleph_benches::random_brickwall_circuit`
    bit-for-bit.  Angles are deterministic: cos(layer + qubit*0.37) and the same * 1.13.
    """
    qc = QuantumCircuit(n, name=f"random_brickwall_n{n}_d{depth}")
    for layer in range(depth):
        for q in range(n):
            theta = math.cos(layer + q * 0.37)
            qc.rz(theta, q)
            qc.rx(theta * 1.13, q)
        offset = layer & 1
        q = offset
        while q + 1 < n:
            qc.cx(q, q + 1)
            q += 2
    return qc


def build_ghz(n: int) -> QuantumCircuit:
    """GHZ state on `n` qubits: H on q0, then a CNOT chain q_i -> q_{i+1}."""
    qc = QuantumCircuit(n, name=f"ghz_n{n}")
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


FAMILY_BUILDERS = {
    "ghz": lambda n: build_ghz(n),
    "qft": lambda n: build_qft(n),
    "grover": lambda n: build_grover(n, GROVER_ITERS),
    "random_brickwall": lambda n: build_random_brickwall(n, RANDOM_DEPTH),
}


def workload_name(family: str, n: int) -> str:
    if family == "grover":
        return f"grover_n{n}_iters{GROVER_ITERS}"
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"


def all_workloads() -> list[tuple[str, str, int]]:
    """(name, family, n) for the full matrix, families in stable order."""
    return [
        (workload_name(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in FAMILY_SIZES[fam]
    ]


def timing_runs_for(n: int) -> int:
    """Fewer timed Aer runs at large n (each is minutes). Disclosed in the report."""
    if n <= 20:
        return 10
    if n <= 22:
        return 5
    if n <= 25:
        return 3
    return 2  # n >= 28: a single Aer statevector run is many minutes


def transpile_and_export(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    """Transpile to aleph's basis (level 0, no optimisation) and write QASM3."""
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    qasm = qasm3.dumps(tqc)
    out = CIRCUITS_DIR / f"{name}.qasm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(qasm)
    return tqc


def time_aer(tqc: QuantumCircuit, runs: int) -> dict:
    """Run `tqc` through AerSimulator(method='statevector') `runs` times under
    single-thread pinning. Returns dict with median, mean, stdev (seconds)."""
    sim = AerSimulator(
        method="statevector",
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    # Aer needs a save-statevector to actually compute the state.
    tqc_with_save = tqc.copy()
    tqc_with_save.save_statevector()
    # Warm-up: one run not timed.
    sim.run(tqc_with_save).result()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sim.run(tqc_with_save).result()
        samples.append(time.perf_counter() - t0)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Qiskit Aer Phase-1 baseline harness")
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="Only generate/export circuits/*.qasm; do not time Aer.",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default="",
        help="Comma-separated workload names to time (default: full matrix).",
    )
    args = parser.parse_args()

    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = all_workloads()
    selected = (
        set(args.workloads.split(",")) if args.workloads else {name for name, _, _ in matrix}
    )

    results: dict = {
        "schema_version": 2,
        "n_qubits_list": N_QUBITS_LIST,
        "grover_iters": GROVER_ITERS,
        "random_depth": RANDOM_DEPTH,
        "basis_gates": BASIS_GATES,
        "workloads": {},
    }
    for name, family, n in matrix:
        print(f"[build] {name} ...", flush=True)
        qc = FAMILY_BUILDERS[family](n)
        tqc = transpile_and_export(qc, name)
        gate_count = len(tqc.data)
        print(f"[build] {name}: {gate_count} gates after transpile", flush=True)
        if args.gen_only or name not in selected:
            continue
        runs = timing_runs_for(n)
        print(f"[time]  {name} (Aer, {runs} runs) ...", flush=True)
        timing = time_aer(tqc, runs)
        print(
            f"[time]  {name}: median={timing['median_s']*1000:.2f} ms "
            f"stdev={timing['stdev_s']*1000:.2f} ms",
            flush=True,
        )
        results["workloads"][name] = {
            "n": n,
            "family": family,
            "timing_runs": runs,
            "gate_count_post_transpile": gate_count,
            "qiskit_aer": timing,
        }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[done] results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
