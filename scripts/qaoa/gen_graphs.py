"""Generate committed 3-regular Max-Cut graph instances (single source of truth
for aleph + Qiskit) and their exact max-cut values.

For n in {6,10,14}: networkx.random_regular_graph(3, n, seed) -> edge list, and
the exact max-cut by brute force over all 2^n bipartitions (n<=14 -> <=16384).
Run: scripts/qiskit-baseline/.venv/bin/python scripts/qaoa/gen_graphs.py
"""
from pathlib import Path
import networkx as nx

OUT = Path(__file__).parent / "graphs"
SIZES = [6, 10, 14]
SEED = 20260607


def max_cut_bruteforce(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = sum(1 for i, j in edges if ((mask >> i) & 1) != ((mask >> j) & 1))
        if cut > best:
            best = cut
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        g = nx.random_regular_graph(3, n, seed=SEED + n)
        edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())
        (OUT / f"qaoa_n{n}.edges").write_text(
            f"# 3-regular n={n}, {len(edges)} edges, seed={SEED + n}\n"
            + "\n".join(f"{i} {j}" for i, j in edges) + "\n")
        mc = max_cut_bruteforce(n, edges)
        (OUT / f"qaoa_n{n}.maxcut").write_text(f"{mc}\n")
        print(f"[gen] qaoa_n{n}: {len(edges)} edges, max-cut={mc}")


if __name__ == "__main__":
    main()
