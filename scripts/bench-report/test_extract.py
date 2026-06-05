import json, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).parent

def _write_estimates(root, group, param, median_ns, std_ns):
    d = root / "criterion" / group / param / "new"
    d.mkdir(parents=True, exist_ok=True)
    (d / "estimates.json").write_text(json.dumps({
        "median": {"point_estimate": median_ns},
        "std_dev": {"point_estimate": std_ns},
    }))

class TestExtract(unittest.TestCase):
    def test_extract_qft(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_estimates(root, "phase4_qft", "10", 1_000_000.0, 5_000.0)
            _write_estimates(root, "phase4_qft", "25", 500_000_000.0, 250_000.0)
            out = root / "phase4-aleph.json"
            subprocess.run([sys.executable, str(HERE / "extract_criterion.py"),
                "--criterion-root", str(root / "criterion"),
                "--group", "phase4_qft", "--family", "qft", "--out", str(out)], check=True)
            w = json.loads(out.read_text())["workloads"]
            self.assertEqual(w["qft_n10"]["aleph_ms_median"], 1.0)
            self.assertEqual(w["qft_n10"]["n"], 10)
            self.assertEqual(w["qft_n10"]["family"], "qft")
            self.assertAlmostEqual(w["qft_n25"]["aleph_ms_median"], 500.0, places=6)
            self.assertAlmostEqual(w["qft_n25"]["aleph_rsd"], 250_000.0/500_000_000.0, places=12)

if __name__ == "__main__":
    unittest.main()
