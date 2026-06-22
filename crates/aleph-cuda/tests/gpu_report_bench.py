#!/usr/bin/env python3
"""P5-08 Qiskit-Aer-GPU column for the GPU benchmark report.

Builds the same Tier-1 + Tier-2 workloads as `gpu_report_bench.rs` and times them
on AerSimulator(method='statevector', device='GPU') — which dispatches to NVIDIA
cuStateVec when cuQuantum is present, so this is "the cuStateVec kernel + Qiskit
transpile/Python overhead". Prints `workload  seconds` (best of N).

    ALEPH_REPORT_N=28 /root/aervenv/bin/python gpu_report_bench.py
"""
import math
import os
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def ghz(n):
    c = QuantumCircuit(n)
    c.h(0)
    for q in range(1, n):
        c.cx(0, q)
    return c


def qft(n):
    c = QuantumCircuit(n)
    for j in range(n):
        c.h(j)
        for off, k in enumerate(range(j + 1, n)):
            c.cp(math.pi / (1 << (off + 1)), k, j)
    return c


def grover(n, iters):
    c = QuantumCircuit(n)
    c.h(range(n))
    ctrls = list(range(min(n - 1, 8)))

    def mcz():
        # (k)-controlled Z on the last qubit.
        c.h(n - 1)
        c.mcx(ctrls, n - 1)
        c.h(n - 1)

    for _ in range(iters):
        mcz()
        for q in range(n):
            c.h(q); c.x(q)
        mcz()
        for q in range(n):
            c.x(q); c.h(q)
    return c


def random_brickwall(n, depth, seed=0x5208):
    import random
    rng = random.Random(seed)
    c = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            t = rng.random() * 2 * math.pi
            g = rng.randrange(3)
            (c.rx if g == 0 else c.ry if g == 1 else c.rz)(t, q)
        for q in range(0, n - 1, 2):
            c.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            c.cx(q, q + 1)
    return c


def qpe(n):
    c = QuantumCircuit(n)
    eig, m = n - 1, n - 1
    c.x(eig)
    c.h(range(m))
    theta = 2 * math.pi * 0.123
    for k in range(m):
        c.cp(theta * (1 << k), k, eig)
    for j in reversed(range(m)):
        for off, k in enumerate(reversed(range(j))):
            c.cp(-math.pi / (1 << (off + 1)), j, k)
        c.h(j)
    return c


def vqe(n, layers):
    c = QuantumCircuit(n)
    t = 0.1
    for _ in range(layers):
        for q in range(n):
            c.ry(t, q); t += 0.017
        for q in range(n - 1):
            c.cx(q, q + 1)
        for q in range(n):
            c.rz(t, q); t += 0.013
    return c


def qaoa(n, p):
    c = QuantumCircuit(n)
    c.h(range(n))
    gamma, beta = 0.7, 0.4
    for _ in range(p):
        for i in range(n):
            j = (i + 1) % n
            a, b = min(i, j), max(i, j)
            c.cx(a, b); c.rz(2 * gamma, b); c.cx(a, b)
        for q in range(n):
            c.rx(2 * beta, q)
    return c


def main():
    n = int(os.environ.get("ALEPH_REPORT_N", "28"))
    reps = int(os.environ.get("ALEPH_REPORT_REPS", "5"))
    sim = AerSimulator(method="statevector", device="GPU")
    workloads = [
        ("ghz", ghz(n)),
        ("qft", qft(n)),
        ("grover(4it)", grover(n, 4)),
        ("random(d20)", random_brickwall(n, 20)),
        ("qpe", qpe(n)),
        ("vqe(8L)", vqe(n, 8)),
        ("qaoa(p4)", qaoa(n, 4)),
    ]
    print(f"== Aer-GPU (statevector) n={n}, best of {reps} ==")
    for name, circ in workloads:
        circ = circ.copy()
        circ.save_statevector()
        tc = transpile(circ, sim)
        sim.run(tc).result()  # warmup
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            sim.run(tc).result()
            best = min(best, time.perf_counter() - t)
        print(f"{name:<14} {best:.4f}")


if __name__ == "__main__":
    main()
