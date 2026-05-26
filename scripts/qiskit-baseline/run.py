"""Qiskit Aer baseline harness for aleph Phase 1, Stage 0.

Builds three workloads (QFT-20, Grover-20 x 10 iters, random-brickwall-20),
transpiles each to the basis aleph-parser supports, exports QASM3, and times
AerSimulator(method='statevector') under single-thread pinning.

Spec: docs/superpowers/specs/2026-05-26-stage0-qiskit-baseline-design.md
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
N_QUBITS = 20
GROVER_ITERS = 5
RANDOM_DEPTH = 20
TIMING_RUNS = 10
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


WORKLOADS = {
    "qft_n20": lambda: build_qft(N_QUBITS),
    "grover_n20_iters5": lambda: build_grover(N_QUBITS, GROVER_ITERS),
    "random_brickwall_n20_d20": lambda: build_random_brickwall(N_QUBITS, RANDOM_DEPTH),
}


def transpile_and_export(qc: QuantumCircuit, name: str) -> QuantumCircuit:
    """Transpile to aleph's basis (level 0, no optimisation) and write QASM3."""
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    qasm = qasm3.dumps(tqc)
    out = CIRCUITS_DIR / f"{name}.qasm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(qasm)
    return tqc


def main() -> None:
    CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)
    transpiled = {}
    for name, builder in WORKLOADS.items():
        print(f"[build] {name} ...", flush=True)
        qc = builder()
        tqc = transpile_and_export(qc, name)
        print(
            f"[build] {name}: {len(tqc.data)} gates after transpile "
            f"(was {len(qc.data)} pre-transpile)",
            flush=True,
        )
        transpiled[name] = tqc
    # (Aer timing lands in Task 4.)


if __name__ == "__main__":
    main()
