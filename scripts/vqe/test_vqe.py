"""VQE driver tests. The aleph-py-dependent tests skip when the `aleph` module
is not built (run `maturin build --release --features python` in crates/aleph-py first)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rotosolve import rotosolve  # noqa: E402
import vqe  # noqa: E402

try:
    import aleph  # noqa: F401
    HAVE_ALEPH = True
except Exception:
    HAVE_ALEPH = False


class TestRotosolve(unittest.TestCase):
    def test_minimizes_simple_sinusoid_sum(self):
        # E(t) = sum_i (1 - cos(t_i)); min at t_i = 0 -> E = 0.
        import math
        fn = lambda t: sum(1 - math.cos(x) for x in t)
        _, e, _ = rotosolve(fn, [1.0, 2.0, 3.0])
        self.assertLess(e, 1e-6)


class TestHamiltonianFiles(unittest.TestCase):
    def test_term_counts(self):
        self.assertEqual(len(vqe.load_terms(4)), 15)
        self.assertEqual(len(vqe.load_terms(6)), 51)
        self.assertEqual(len(vqe.load_terms(8)), 92)


@unittest.skipUnless(HAVE_ALEPH, "aleph-py not built (maturin develop --features python)")
class TestConvergence(unittest.TestCase):
    def test_h2_reaches_fci(self):
        self.assertEqual(vqe.converge(), 0)


if __name__ == "__main__":
    unittest.main()
