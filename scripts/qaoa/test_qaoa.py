"""QAOA driver tests. aleph-py-dependent tests skip if `aleph` isn't built."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qaoa  # noqa: E402

try:
    import aleph  # noqa: F401
    HAVE_ALEPH = True
except Exception:
    HAVE_ALEPH = False


class TestGraphs(unittest.TestCase):
    def test_edge_counts_three_regular(self):
        for n in (6, 10, 14):
            edges, maxcut = qaoa.load_graph(n)
            self.assertEqual(len(edges), 3 * n // 2)  # 3-regular
            self.assertGreater(maxcut, 0)
            # every node degree 3
            deg = {}
            for i, j in edges:
                deg[i] = deg.get(i, 0) + 1
                deg[j] = deg.get(j, 0) + 1
            self.assertTrue(all(d == 3 for d in deg.values()))


@unittest.skipUnless(HAVE_ALEPH, "aleph-py not built")
class TestRatio(unittest.TestCase):
    def test_n6_p3_reaches_0_9(self):
        edges, maxcut = qaoa.load_graph(6)
        energy, _, _ = qaoa.optimize(6, edges, 3)
        self.assertGreaterEqual(energy / maxcut, 0.9)

    def test_sv_mps_agree(self):
        # SV and MPS must compute the same <H_C> for the same angles.
        edges, _ = qaoa.load_graph(6)
        g, b = [0.4, 0.3, 0.5], [0.2, 0.6, 0.1]
        sv = aleph.qaoa_energy(6, edges, g, b, "sv")
        mps = aleph.qaoa_energy(6, edges, g, b, "mps")
        self.assertAlmostEqual(sv, mps, places=6)


if __name__ == "__main__":
    unittest.main()
