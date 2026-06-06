"""VQE-H2 driver: convergence demo + aleph-vs-Qiskit energy-eval benchmark.

Requires aleph-py built into the active venv (`maturin develop --features python`
from crates/aleph-py) and qiskit/qiskit-aer for --bench. Single-thread both sides.

Usage:
    python scripts/vqe/vqe.py --converge          # rotosolve H2 4q -> FCI
    python scripts/vqe/vqe.py --bench --out results-vqe.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rotosolve import rotosolve  # noqa: E402

HAM_DIR = Path(__file__).parent / "hamiltonians"
DEPTH = 4
SIZES = [4, 6, 8]


def load_terms(n):
    """Parse the committed Hamiltonian file into [(coeff, 'IXYZ...'), ...]."""
    terms = []
    for line in (HAM_DIR / f"vqe_n{n}.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        coeff, pauli = line.split()
        terms.append((float(coeff), pauli))
    return terms


def aleph_energy_fn(n, ham_obj):
    import aleph
    return lambda thetas: aleph.hea_energy(n, DEPTH, list(thetas), ham_obj)


def converge():
    import aleph
    n = 4
    ham = aleph.PauliSum.load(str(HAM_DIR / "vqe_n4.txt"), n)
    fci = float((HAM_DIR / "vqe_n4.fci").read_text().strip())
    p = n * (DEPTH + 1)
    theta0 = [0.1 * (i + 1) for i in range(p)]
    theta, e, n_evals = rotosolve(aleph_energy_fn(n, ham), theta0)
    gap = e - fci
    print(f"H2 VQE: E={e:.6f} Ha, FCI={fci:.6f} Ha, gap={gap:.2e} ({n_evals} evals)")
    ok = e <= fci + 1.6e-3
    print("CONVERGED within chemical accuracy" if ok else "DID NOT CONVERGE")
    return 0 if ok else 1


_AER_SIM = None


def _aer_sim():
    global _AER_SIM
    if _AER_SIM is None:
        from qiskit_aer import AerSimulator
        _AER_SIM = AerSimulator(
            method="statevector",
            max_parallel_threads=1,
            max_parallel_experiments=1,
        )
    return _AER_SIM


def qiskit_energy(n, terms, thetas):
    """<H> of the same Ry+CNOT HEA via Qiskit Aer (statevector, single-thread)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    import qiskit_aer.library  # registers save_expectation_value on QuantumCircuit  # noqa: F401
    qc = QuantumCircuit(n)
    idx = 0
    for _layer in range(DEPTH):
        for q in range(n):
            qc.ry(thetas[idx], q); idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(thetas[idx], q); idx += 1
    # Our pauli string char i = qubit i; Aer/SparsePauliOp labels are
    # little-endian (leftmost char = highest qubit), so reverse each label.
    op = SparsePauliOp.from_list([(p[::-1], c) for c, p in terms])
    qc.save_expectation_value(op, list(range(n)))
    result = _aer_sim().run(qc).result()
    return float(result.data(0)["expectation_value"].real)


def time_median(fn, runs):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def bench(out_path):
    import aleph
    results = {"schema_version": 1, "depth": DEPTH, "workloads": {}}
    for n in SIZES:
        terms = load_terms(n)
        ham = aleph.PauliSum.load(str(HAM_DIR / f"vqe_n{n}.txt"), n)
        p = n * (DEPTH + 1)
        thetas = [0.1 * (i + 1) for i in range(p)]
        a_fn = lambda: aleph.hea_energy(n, DEPTH, thetas, ham)
        q_fn = lambda: qiskit_energy(n, terms, thetas)
        a_fn(); q_fn()  # warm-up
        runs = 50 if n <= 6 else 20
        a_med, a_sd = time_median(a_fn, runs)
        q_med, q_sd = time_median(q_fn, runs)
        results["workloads"][f"vqe_n{n}"] = {
            "n": n, "family": "vqe", "n_terms": len(terms),
            "aleph_ms_median": a_med * 1e3, "aleph_rsd": (a_sd / a_med if a_med else 0.0),
            "qiskit_ms_median": q_med * 1e3, "qiskit_rsd": (q_sd / q_med if q_med else 0.0),
        }
        print(f"vqe_n{n}: aleph={a_med*1e3:.3f}ms qiskit={q_med*1e3:.3f}ms ({len(terms)} terms)")
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="results-vqe.json")
    args = ap.parse_args()
    if args.converge:
        return converge()
    if args.bench:
        bench(args.out)
        return 0
    ap.error("pass --converge or --bench")


if __name__ == "__main__":
    sys.exit(main())
