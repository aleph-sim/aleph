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
    "qpe": [10, 15, 20, 25],
    # P4-02: optimal-iteration Grover at small n (tiny cache-resident state,
    # tractable even at 2.26M gates for n=16). The legacy grover_n{15..25}_iters5
    # fixtures stay on disk (frozen Phase-1/2 bench artifacts) but are no longer
    # regenerated here.
    "grover": [4, 8, 12, 16],
    "random_brickwall": [15, 20, 22, 25],
}
# Union for the results header (sorted, de-duplicated).
N_QUBITS_LIST = sorted({n for sizes in FAMILY_SIZES.values() for n in sizes})
# Legacy iteration count for the frozen grover_n{15..25}_iters5 fixtures only.
# The active P4-02 matrix uses grover_optimal_iters(n) instead (see below).
GROVER_ITERS = 5


def grover_optimal_iters(n: int) -> int:
    """Optimal Grover iteration count for a single marked state.

    The success probability peaks at ~round(pi/4 * sqrt(2^n)) Grover operators
    (Nielsen & Chuang sec. 6.1). n in {4,8,12,16} -> {3, 13, 50, 201}.
    """
    return max(1, round(math.pi / 4 * math.sqrt(2**n)))


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


def build_qpe(n: int) -> QuantumCircuit:
    """Quantum phase estimation of U = P(2*pi*phi) on a single target qubit.

    Counting register = qubits [0, m); target = qubit m, where m = n - 1. The
    eigenphase phi = (2^m - 1)/2^m has all m fractional bits set, so it is
    exactly representable in the counting register: QPE returns it exactly and
    the final state is the all-ones basis state |1...1> (counting register all
    1, target stays |1>) = amplitude index 2^n - 1 in ANY qubit ordering and
    inverse-QFT swap convention (all-ones is bit-reversal-invariant) -> a
    layout-free accuracy oracle (asserted by the QPE accuracy test added
    later in this P4-03 change set, benches/tests/qpe_accuracy.rs).
    Nielsen & Chuang sec. 5.2.
    """
    m = n - 1
    phi = (2**m - 1) / 2**m
    qc = QuantumCircuit(n, name=f"qpe_n{n}")
    counting = list(range(m))
    target = m
    qc.x(target)  # prepare eigenstate |1> of P(2*pi*phi)
    qc.h(counting)
    for j, ctrl in enumerate(counting):
        # controlled-U^{2^j}: phase e^{2*pi*i*phi*2^j} kicks back onto |1> of ctrl.
        qc.cp(2 * math.pi * phi * (2**j), ctrl, target)
    # Inverse QFT on the counting register recovers the estimate.
    qc.compose(QFT(num_qubits=m, do_swaps=True, inverse=True), qubits=counting, inplace=True)
    return qc


FAMILY_BUILDERS = {
    "ghz": lambda n: build_ghz(n),
    "qft": lambda n: build_qft(n),
    "qpe": lambda n: build_qpe(n),
    "grover": lambda n: build_grover(n, grover_optimal_iters(n)),
    "random_brickwall": lambda n: build_random_brickwall(n, RANDOM_DEPTH),
}


def corpus_stem(family: str, n: int) -> str:
    """QASM filename stem and QuantumCircuit name. Grover/Random embed their
    iteration/depth count so the on-disk corpus is self-describing
    (e.g. grover_n16_iters201.qasm)."""
    if family == "grover":
        return f"grover_n{n}_iters{grover_optimal_iters(n)}"
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"


def workload_key(family: str, n: int) -> str:
    """Join key into the unified results JSON. MUST equal extract_criterion.py's
    `{family}_n{n}` so report.py lines up the aleph (criterion) and Aer rows.
    The grover *file* carries the iter count; the *key* does not — the criterion
    BenchmarkId parameter is just n. (Random keeps its legacy depth-suffixed key;
    it is not a Phase-4 criterion consumer.)"""
    if family == "random_brickwall":
        return f"random_brickwall_n{n}_d{RANDOM_DEPTH}"
    return f"{family}_n{n}"


def all_workloads() -> list[tuple[str, str, str, int]]:
    """(key, stem, family, n) for the full matrix, families in stable order."""
    return [
        (workload_key(fam, n), corpus_stem(fam, n), fam, n)
        for fam in FAMILY_BUILDERS
        for n in FAMILY_SIZES[fam]
    ]


def timing_runs_for(n: int, gate_count: int) -> int:
    """Fewer timed Aer runs for costlier circuits. A single Aer statevector pass
    costs ~ gate_count * 2^n (each gate sweeps the 2^n-amplitude state), so the
    budget keys on that product rather than n alone — Grover-16 is only 16 qubits
    but 2.26M gates. Verified: QFT-25 (5.1e10)->3, QFT-30 (2.4e12)->1,
    Grover-16 (1.5e11)->2. Disclosed in the report."""
    cost = gate_count * (2**n)
    if cost > 1e12:
        return 1
    if cost > 1e11:
        return 2
    if cost > 1e10:
        return 3
    if cost > 1e9:
        return 5
    return 10


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
        set(args.workloads.split(",")) if args.workloads else {key for key, _, _, _ in matrix}
    )

    results: dict = {
        "schema_version": 2,
        "n_qubits_list": N_QUBITS_LIST,
        "grover_iters": GROVER_ITERS,
        "random_depth": RANDOM_DEPTH,
        "basis_gates": BASIS_GATES,
        "workloads": {},
    }
    for key, stem, family, n in matrix:
        print(f"[build] {stem} ...", flush=True)
        qc = FAMILY_BUILDERS[family](n)
        tqc = transpile_and_export(qc, stem)
        gate_count = len(tqc.data)
        print(f"[build] {stem}: {gate_count} gates after transpile", flush=True)
        if args.gen_only or key not in selected:
            continue
        runs = timing_runs_for(n, gate_count)
        print(f"[time]  {key} (Aer, {runs} runs) ...", flush=True)
        timing = time_aer(tqc, runs)
        print(
            f"[time]  {key}: median={timing['median_s']*1000:.2f} ms "
            f"stdev={timing['stdev_s']*1000:.2f} ms",
            flush=True,
        )
        results["workloads"][key] = {
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
