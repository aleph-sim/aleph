import subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).parent
TD = HERE / "testdata"

class TestReport(unittest.TestCase):
    def test_report_matches_golden(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "phase4.md"
            subprocess.run([sys.executable, str(HERE / "report.py"),
                "--aleph", str(TD / "aleph.json"), "--aer", str(TD / "aer.json"),
                "--meta", str(TD / "meta.json"), "--out", str(out)], check=True)
            self.assertEqual(out.read_text(), (TD / "phase4.golden.md").read_text())

if __name__ == "__main__":
    unittest.main()
