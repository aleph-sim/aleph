"""Generate oracle fixtures from oracle/circuits/*.qasm using Qiskit Aer.

Each fixture captures the exact final state vector plus 100k-shot counts
obtained with seed_simulator=0. See spec §4 for the JSON schema. The
fixtures are byte-stable across regen runs except for the
``generated_at`` field.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import qasm3
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parent
CIRCUITS_DIR = ROOT / "circuits"
FIXTURES_DIR = ROOT / "fixtures"
SCHEMA_VERSION = 1
SHOTS = 100_000
SEED = 0


def gen_one(qasm_path: Path) -> dict:
    name = qasm_path.stem
    qasm_src = qasm_path.read_text()
    qc = qasm3.loads(qasm_src)
    n = qc.num_qubits

    # State vector via a save_statevector instruction on a fresh copy.
    sv_qc = qc.copy()
    sv_qc.save_statevector()
    sv_backend = AerSimulator(method="statevector")
    sv = sv_backend.run(sv_qc).result().get_statevector(sv_qc)
    amps = [[c.real, c.imag] for c in sv.data]

    # Counts via a measure-all copy with a fixed seed.
    m_qc = qc.copy()
    m_qc.measure_all()
    counts_backend = AerSimulator()
    counts = (
        counts_backend.run(m_qc, shots=SHOTS, seed_simulator=SEED)
        .result()
        .get_counts(m_qc)
    )
    # measure_all() produces a single classical register so keys are
    # plain bitstrings (no inter-register space).
    counts = dict(sorted(counts.items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "num_qubits": n,
        "qasm_path": f"circuits/{qasm_path.name}",
        "qiskit_version": qiskit.__version__,
        "aer_version": qiskit_aer.__version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shots": SHOTS,
        "rng_seed": SEED,
        "statevector": {"endianness": "little", "amplitudes": amps},
        "counts": counts,
    }


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    qasm_files = sorted(CIRCUITS_DIR.glob("*.qasm"))
    if not qasm_files:
        print(f"no .qasm files in {CIRCUITS_DIR}", flush=True)
        return 1
    for qasm_path in qasm_files:
        fx = gen_one(qasm_path)
        out = FIXTURES_DIR / f"{fx['name']}.json"
        out.write_text(
            json.dumps(fx, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        print(
            f"  ✓ {fx['name']:32s} "
            f"{len(fx['statevector']['amplitudes']):>8d} amps  "
            f"{len(fx['counts']):>5d} counts",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
