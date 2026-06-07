"""QAOA Max-Cut driver: approximation ratios (COBYLA) + multi-backend energy-eval
benchmark (aleph SV, aleph MPS, Qiskit Aer). Requires aleph-py built into the
venv (maturin build --release --features python) and scipy/qiskit/qiskit-aer.

Usage:
    python scripts/qaoa/qaoa.py --ratio   # optimize, print ratios, assert >=0.9 @ p=3 n=6,10
    python scripts/qaoa/qaoa.py --bench --out docs/perf/data/qaoa-results.json
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

GRAPHS = Path(__file__).parent / "graphs"
SIZES = [6, 10, 14]
PS = [1, 2, 3]
RESTARTS = 16


def load_graph(n):
    edges = []
    for line in (GRAPHS / f"qaoa_n{n}.edges").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        i, j = line.split()
        edges.append((int(i), int(j)))
    maxcut = int((GRAPHS / f"qaoa_n{n}.maxcut").read_text().strip())
    return edges, maxcut


def optimize(n, edges, p, restarts=RESTARTS):
    """Maximize <H_C> over (gammas,betas) via COBYLA, best of `restarts`."""
    import numpy as np
    from scipy.optimize import minimize
    import aleph
    rng = np.random.default_rng(12345 + n * 100 + p)

    def neg_energy(x):
        g, b = list(x[:p]), list(x[p:])
        return -aleph.qaoa_energy(n, edges, g, b, "sv")

    best = -math.inf
    best_x = None
    for _ in range(restarts):
        x0 = rng.uniform(0, math.pi, size=2 * p)
        res = minimize(neg_energy, x0, method="COBYLA",
                       options={"maxiter": 200, "rhobeg": 0.5})
        if -res.fun > best:
            best, best_x = -res.fun, res.x
    return best, list(best_x[:p]), list(best_x[p:])


def ratios():
    out = {}
    ok = True
    print(f"{'graph':>8} {'p':>2} {'cut':>8} {'maxcut':>6} {'ratio':>6}")
    for n in SIZES:
        edges, maxcut = load_graph(n)
        for p in PS:
            energy, _, _ = optimize(n, edges, p)
            r = energy / maxcut
            out[f"qaoa_n{n}_p{p}"] = {"n": n, "p": p, "edges": len(edges),
                                     "maxcut": maxcut, "cut": energy, "ratio": r}
            print(f"{'n'+str(n):>8} {p:>2} {energy:>8.3f} {maxcut:>6} {r:>6.3f}")
            if p == 3 and n in (6, 10) and r < 0.9:
                ok = False
                print(f"  !! n={n} p=3 ratio {r:.3f} < 0.9")
    print("APPROXIMATION OK (>=0.9 @ p=3 for n=6,10)" if ok else "RATIO AC NOT MET")
    return out, ok


def qiskit_cut_energy(n, edges, gammas, betas):
    """Same QAOA circuit in Qiskit; <H_C> via Aer save_expectation_value."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    import qiskit_aer.library  # noqa: F401  (registers save_expectation_value)
    from qiskit_aer import AerSimulator
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (i, j) in edges:
            qc.cx(i, j); qc.rz(2 * g, j); qc.cx(i, j)
        for q in range(n):
            qc.rx(2 * b, q)
    # H_C = sum_edges 0.5*(I - Z_iZ_j); reverse labels for Qiskit endianness.
    terms = [("I" * n, 0.5 * len(edges))]
    for (i, j) in edges:
        lbl = ["I"] * n
        lbl[i] = "Z"; lbl[j] = "Z"
        terms.append(("".join(reversed(lbl)), -0.5))
    op = SparsePauliOp.from_list(terms)
    qc.save_expectation_value(op, list(range(n)))
    sim = AerSimulator(method="statevector", max_parallel_threads=1,
                       max_parallel_experiments=1)
    return float(sim.run(qc).result().data(0)["expectation_value"].real)


def time_median(fn, runs):
    s = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); s.append(time.perf_counter() - t0)
    return statistics.median(s)


def bench(out_path, base=None):
    import aleph
    results = {"schema_version": 1, "restarts": RESTARTS,
               "workloads": base.get("workloads", {}) if base else {}}
    for n in SIZES:
        edges, maxcut = load_graph(n)
        for p in PS:
            g = [0.4] * p; b = [0.3] * p  # fixed angles; timing is angle-independent
            f_sv = lambda: aleph.qaoa_energy(n, edges, g, b, "sv")
            f_mps = lambda: aleph.qaoa_energy(n, edges, g, b, "mps")
            f_aer = lambda: qiskit_cut_energy(n, edges, g, b)
            f_sv(); f_mps(); f_aer()  # warm-up
            runs = 30 if n <= 10 else 15
            key = f"qaoa_n{n}_p{p}"
            row = results["workloads"].setdefault(key, {})
            row.update({"n": n, "p": p, "edges": len(edges), "maxcut": maxcut,
                        "sv_ms": time_median(f_sv, runs) * 1e3,
                        "mps_ms": time_median(f_mps, runs) * 1e3,
                        "aer_ms": time_median(f_aer, runs) * 1e3})
            print(f"qaoa_n{n}_p{p}: sv={row['sv_ms']:.3f} mps={row['mps_ms']:.3f} "
                  f"aer={row['aer_ms']:.3f} ms")
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="docs/perf/data/qaoa-results.json")
    args = ap.parse_args()
    base = {}
    out_p = Path(args.out)
    if out_p.exists():
        base = json.loads(out_p.read_text())
    if args.ratio:
        rat, ok = ratios()
        # merge ratios into the results file (preserve any timing already there)
        base.setdefault("workloads", {})
        for k, v in rat.items():
            base.setdefault("schema_version", 1)
            base["workloads"].setdefault(k, {}).update(v)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(base, indent=2))
        print(f"[ratios] -> {out_p}")
        return 0 if ok else 1
    if args.bench:
        bench(args.out, base)
        return 0
    ap.error("pass --ratio or --bench")


if __name__ == "__main__":
    sys.exit(main())
