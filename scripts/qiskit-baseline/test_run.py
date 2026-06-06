"""Unit tests for the pure helpers in run.py (optimal iters, run budget, keys).

run.py imports Qiskit at module load, so these tests skip when Qiskit is not
installed (e.g. CI). Run locally under the harness venv:

    scripts/qiskit-baseline/.venv/bin/python -m unittest \
        discover -s scripts/qiskit-baseline -p 'test_run.py'
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    import run  # noqa: E402  (Qiskit-importing module)
    HAVE_RUN = True
except Exception:  # pragma: no cover - environment without Qiskit
    HAVE_RUN = False


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestGroverOptimalIters(unittest.TestCase):
    def test_table(self):
        self.assertEqual(run.grover_optimal_iters(4), 3)
        self.assertEqual(run.grover_optimal_iters(8), 13)
        self.assertEqual(run.grover_optimal_iters(12), 50)
        self.assertEqual(run.grover_optimal_iters(16), 201)


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestTimingRunsForCost(unittest.TestCase):
    def test_qft_budget_preserved(self):
        # QFT-25: 1525 * 2^25 = 5.1e10 -> 3 (unchanged vs the n-only function).
        self.assertEqual(run.timing_runs_for(25, 1525), 3)
        # QFT-30: 2205 * 2^30 = 2.4e12 -> 1 (spec: "1-2 runs, as before").
        self.assertEqual(run.timing_runs_for(30, 2205), 1)

    def test_grover_costs(self):
        self.assertEqual(run.timing_runs_for(4, 268), 10)        # 4.3e3
        self.assertEqual(run.timing_runs_for(8, 17974), 10)      # 4.6e6
        self.assertEqual(run.timing_runs_for(12, 264312), 5)     # 1.08e9
        self.assertEqual(run.timing_runs_for(16, 2258854), 2)    # 1.48e11


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestKeysAndStems(unittest.TestCase):
    def test_grover_stem_carries_iters(self):
        self.assertEqual(run.corpus_stem("grover", 16), "grover_n16_iters201")
        self.assertEqual(run.corpus_stem("grover", 4), "grover_n4_iters3")

    def test_grover_key_has_no_iters(self):
        self.assertEqual(run.workload_key("grover", 16), "grover_n16")

    def test_qft_key_and_stem_align(self):
        self.assertEqual(run.workload_key("qft", 20), "qft_n20")
        self.assertEqual(run.corpus_stem("qft", 20), "qft_n20")

    def test_family_sizes(self):
        self.assertEqual(run.FAMILY_SIZES["grover"], [4, 8, 12, 16])

    def test_qpe_key_and_stem_align(self):
        self.assertEqual(run.workload_key("qpe", 20), "qpe_n20")
        self.assertEqual(run.corpus_stem("qpe", 20), "qpe_n20")

    def test_qpe_family_sizes(self):
        self.assertEqual(run.FAMILY_SIZES["qpe"], [10, 15, 20, 25])


@unittest.skipUnless(HAVE_RUN, "Qiskit not installed")
class TestBuildQpe(unittest.TestCase):
    def test_all_ones_recovery(self):
        # phi = (2^m - 1)/2^m is exactly representable, so QPE collapses to the
        # all-ones state |1...1> = index 2^n - 1 (counting all 1, target |1>).
        from qiskit.quantum_info import Statevector
        for n in (4, 6):
            probs = Statevector(run.build_qpe(n)).probabilities()
            self.assertEqual(int(probs.argmax()), 2**n - 1)
            self.assertGreater(probs[2**n - 1], 0.999)

    def test_num_qubits(self):
        self.assertEqual(run.build_qpe(10).num_qubits, 10)


if __name__ == "__main__":
    unittest.main()
