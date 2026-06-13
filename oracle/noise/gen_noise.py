"""Generate exact noisy measurement distributions via Qiskit Aer.

For each fixture we build a byte-identical NoiseModel, run the gate-only
circuit through the density-matrix method with save_probabilities, and dump
the exact P(outcome) vector. The matching aleph NoiseModel is constructed in
Rust in crates/aleph-sv/tests/noise_oracle.rs — keep the two in sync.

Run from repo root with the oracle venv that has qiskit-aer:
    oracle/.venv/bin/python oracle/noise/gen_noise.py
"""
from __future__ import annotations

import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
)

OUT = Path(__file__).resolve().parent


def quantum_probs(qc: QuantumCircuit, nm: NoiseModel, n: int) -> list[float]:
    """Exact diagonal of the noisy density matrix in the computational basis.

    Index i = qubit values, little-endian (qubit q = bit q of i), matching
    aleph's |i⟩ convention. This captures QUANTUM channel noise (depolarizing,
    amplitude/phase damping) but NOT classical readout error — Aer's
    save_probabilities reads ρ before measurement sampling, so readout
    confusion never touches it (verified empirically). Readout error is folded
    in analytically by apply_readout() below.
    """
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    tqc = qc.copy()
    tqc.save_probabilities()  # exact diagonal of ρ in the computational basis
    res = sim.run(tqc).result()
    probs = res.data(0)["probabilities"]  # dict or vector over 2^n
    out = [0.0] * (1 << n)
    if isinstance(probs, dict):
        for k, v in probs.items():
            out[int(k)] = float(v)
    else:
        for i, v in enumerate(probs):
            out[i] = float(v)
    return out


def apply_readout(probs: list[float], n: int, confusion: dict) -> list[float]:
    """Fold per-qubit readout confusion matrices into a true-outcome dist.

    `confusion[q]` is a 2x2 matrix M where M[t][m] = P(measure m | true t),
    matching Qiskit's ReadoutError([[P(0|0),P(1|0)],[P(0|1),P(1|1)]]) layout.
    For a true index `t` we scatter its mass across all measured indices `m`
    with weight prod_q M[bit_q(t)][bit_q(m)]. Qubit q is bit q of the index
    (little-endian), so this stays in aleph's |i⟩ ordering. Exact (no MC).
    """
    dim = 1 << n
    out = [0.0] * dim
    for t in range(dim):
        pt = probs[t]
        if pt == 0.0:
            continue
        for m in range(dim):
            w = 1.0
            for q in range(n):
                tb = (t >> q) & 1
                mb = (m >> q) & 1
                M = confusion.get(q)
                if M is None:
                    # No readout error on this qubit: identity (m bit must match).
                    if tb != mb:
                        w = 0.0
                        break
                else:
                    w *= M[tb][mb]
            out[m] += pt * w
    return out


def exact_probs(
    qc: QuantumCircuit, nm: NoiseModel, n: int, readout: dict | None = None
) -> list[float]:
    """Exact noisy measurement distribution over 2^n outcomes.

    Quantum-channel noise comes from the density-matrix diagonal; classical
    readout error (which Aer's save_probabilities ignores) is applied
    analytically via `readout` = {qubit: 2x2 confusion matrix}.
    """
    p = quantum_probs(qc, nm, n)
    if readout:
        p = apply_readout(p, n, readout)
    return p


def dump(name: str, n: int, probs: list[float]) -> None:
    (OUT / f"{name}.json").write_text(
        json.dumps({"name": name, "num_qubits": n, "exact_probs": probs}, indent=2)
    )
    print(f"wrote {name}.json  (Sum p={sum(probs):.6f})")


def depol_h() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["h"])
    dump("depol_h", 1, exact_probs(qc, nm, 1))


def depol_cx() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    nm = NoiseModel()
    nm.add_quantum_error(depolarizing_error(0.1, 2), ["cx"], [0, 1])
    dump("depol_cx", 2, exact_probs(qc, nm, 2))


def amp_damp() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    nm = NoiseModel()
    nm.add_quantum_error(amplitude_damping_error(0.2), ["id"], [0])
    dump("amp_damp_h", 1, exact_probs(qc, nm, 1))


def phase_damp() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    nm = NoiseModel()
    nm.add_quantum_error(phase_damping_error(0.3), ["id"], [0])
    dump("phase_damp_h", 1, exact_probs(qc, nm, 1))


def readout() -> None:
    # Deterministic |1> via X, asymmetric readout error.
    # ReadoutError([[P(0|0),P(1|0)],[P(0|1),P(1|1)]]); save_probabilities does
    # NOT include readout confusion, so we fold it in analytically.
    qc = QuantumCircuit(1)
    qc.x(0)
    nm = NoiseModel()
    M = [[0.98, 0.02], [0.05, 0.95]]
    nm.add_readout_error(ReadoutError(M), [0])
    dump("readout_x", 1, exact_probs(qc, nm, 1, readout={0: M}))


def combined_ghz() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    M = [[0.97, 0.03], [0.04, 0.96]]
    for q in range(3):
        nm.add_readout_error(ReadoutError(M), [q])
    dump("combined_ghz3", 3, exact_probs(qc, nm, 3, readout={0: M, 1: M, 2: M}))


if __name__ == "__main__":
    depol_h()
    depol_cx()
    amp_damp()
    phase_damp()
    readout()
    combined_ghz()
