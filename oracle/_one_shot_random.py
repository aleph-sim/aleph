"""One-shot generator for the two pseudo-random QASM circuits in the
corpus. Run **once** to produce the .qasm files; they are then
committed and the regen flow leaves them alone.

Re-running with the same seeds produces byte-identical output, so
bumping Qiskit may move the circuit text — that's accepted, and any
drift gets reviewed in the PR that bumps it.

The random Clifford is post-processed via `transpile(..., basis_gates=
["h","s","sdg","cx"])` so the resulting QASM only contains gates the
aleph parser supports today (P0-08).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from qiskit import qasm3, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_clifford

ROOT = Path(__file__).resolve().parent
CIRCUITS_DIR = ROOT / "circuits"

N = 4
DEPTH = 20

# --- Clifford ---------------------------------------------------------
# random_clifford samples uniformly from the n-qubit Clifford group.
# `to_circuit()` may produce gates beyond {h, s, sdg, cx}; transpile
# down to that basis so the aleph parser (P0-08) accepts the result.
rng = random.Random(0)
attempt = 0
while True:
    attempt += 1
    seed = rng.getrandbits(31)
    cliff = random_clifford(N, seed=seed)
    raw = cliff.to_circuit()
    qc = transpile(
        raw,
        basis_gates=["h", "s", "sdg", "cx"],
        optimization_level=0,
    )
    if qc.size() >= DEPTH:
        break
    if attempt > 10_000:
        raise RuntimeError("could not find a long-enough random Clifford")
(CIRCUITS_DIR / "random_clifford_n4_d20.qasm").write_text(qasm3.dumps(qc) + "\n")

# --- Non-Clifford -----------------------------------------------------
# Hand-rolled: at each depth step, randomly apply (RX/RY/RZ on a random
# qubit) or (CX on a random ordered pair). Angles drawn from a fixed
# seed. The result has explicit, parser-supported gates only.
rng = random.Random(1)
qc2 = QuantumCircuit(N)
for _ in range(DEPTH):
    kind = rng.choice(["rx", "ry", "rz", "cx"])
    if kind == "cx":
        a, b = rng.sample(range(N), 2)
        qc2.cx(a, b)
    else:
        q = rng.randrange(N)
        angle = rng.uniform(-math.pi, math.pi)
        getattr(qc2, kind)(angle, q)
(CIRCUITS_DIR / "random_nonclifford_n4_d20.qasm").write_text(qasm3.dumps(qc2) + "\n")

print("wrote random_clifford_n4_d20.qasm and random_nonclifford_n4_d20.qasm")
