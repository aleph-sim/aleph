#!/usr/bin/env python3
"""P4.5-01 MPS baseline: build the three MPS workload families, export QASM3
fixtures (the single source of truth both sides consume byte-identically),
and time Aer's matrix_product_state method on them.

Families (χ per family chosen in CHI below):
- brickwork_n128_d6 — mirrors crates/aleph-mps/tests/shallow_100q.rs: H wall,
  alternating even/odd CX·RZ(θ)·CX bonds (θ = 0.3 + 0.05·q), RX(φ) mixer wall
  (φ = 0.4 + 0.03·layer). Max bond 8 ⇒ χ=64 is exact on both sides.
- long_range_n12_dist{4,8,11} — H wall + one NN CX·RZ·CX ladder, then a single
  long-range CX(0, dist). χ=64 = 2^(12/2) is exact at n=12 ⇒ no truncation on
  either side (fidelity equal by construction).
- wide_bond_n26_d12 — seeded random-SU(4) brickwall, 12 layers, saturates the
  χ cap. Truncation semantics differ between implementations ⇒ the report
  carries both sides' truncation metrics and a fairness caveat.

Aer config: matrix_product_state_max_bond_dimension=χ,
matrix_product_state_truncation_threshold=1e-16 (bond cap binding, matching
aleph's FixedBond), max_parallel_threads=1 (aleph-mps default is sequential —
default-vs-default).

Timing caveat: Aer needs a save_matrix_product_state() instruction to compute
anything, so the timed region includes serializing the MPS tensors into the
result — a cost the aleph side (criterion bench timing run() only) does not
pay. Negligible at wide_bond_n26 scale; disclosed in docs/perf/parity.md, and
if a small-n cell lands near the 1.2× bar, the save's isolated cost must be
measured before calling the verdict.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator

BASIS_GATES = ["h", "x", "z", "rz", "rx", "ry", "cx", "cz", "ccx", "p"]
CIRCUITS_DIR = Path(__file__).parent / "circuits"
CHI = {"brickwork_n128_d6": 64, "long_range_n12_dist4": 64,
       "long_range_n12_dist8": 64, "long_range_n12_dist11": 64,
       "wide_bond_n26_d12": 256}


def brickwork(n: int, layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(layers):
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(0.3 + 0.05 * q, q + 1)
            qc.cx(q, q + 1)
        phi = 0.4 + 0.03 * layer
        for q in range(n):
            qc.rx(phi, q)
    return qc


def long_range(n: int, dist: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for q in range(n - 1):
        qc.cx(q, q + 1)
        qc.rz(0.3 + 0.05 * q, q + 1)
        qc.cx(q, q + 1)
    qc.cx(0, dist)
    return qc


def wide_bond(n: int, layers: int, seed: int = 0x5121A6E0) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    k = 0
    for layer in range(layers):
        start = 0 if layer % 2 == 0 else 1
        for q in range(start, n - 1, 2):
            qc.append(UnitaryGate(random_unitary(4, seed=seed + k)), [q, q + 1])
            k += 1
    return qc


def export(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    (CIRCUITS_DIR / f"{name}.qasm").write_text(qasm3.dumps(tqc))
    return tqc


def time_aer_mps(tqc: QuantumCircuit, chi: int, runs: int) -> dict:
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=chi,
        matrix_product_state_truncation_threshold=1e-16,
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    t = tqc.copy()
    t.save_matrix_product_state()
    sim.run(t).result()  # warm-up
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sim.run(t).result()
        samples.append(time.perf_counter() - t0)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P4.5-01 Aer MPS baseline")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only export circuits/*.qasm; do not time Aer.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=str, default="results-aer-mps.json")
    args = parser.parse_args()

    builders = {
        "brickwork_n128_d6": lambda: brickwork(128, 6),
        "long_range_n12_dist4": lambda: long_range(12, 4),
        "long_range_n12_dist8": lambda: long_range(12, 8),
        "long_range_n12_dist11": lambda: long_range(12, 11),
        "wide_bond_n26_d12": lambda: wide_bond(26, 12),
    }
    results = {"schema_version": 1, "workloads": {}}
    for name, build in builders.items():
        tqc = export(build(), name)
        gate_count = len(tqc.data)
        print(f"[gen] {name}: n={tqc.num_qubits} gates={gate_count}", flush=True)
        if args.gen_only:
            continue
        print(f"[time] {name}: chi={CHI[name]} runs={args.runs}", flush=True)
        results["workloads"][name] = {
            "n": tqc.num_qubits,
            "chi": CHI[name],
            "gate_count_post_transpile": gate_count,
            "aer_mps": time_aer_mps(tqc, CHI[name], args.runs),
        }
    if not args.gen_only:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
