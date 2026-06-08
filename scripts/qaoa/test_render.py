import json, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import render_report  # noqa: E402

TD = Path(__file__).parent / "testdata"


class TestRender(unittest.TestCase):
    def test_renders_both_tables(self):
        data = json.loads((TD / "results.json").read_text())
        meta = json.loads((TD / "meta.json").read_text())
        md = render_report.render(data, meta)
        self.assertIn("## Approximation ratio", md)
        self.assertIn("## Energy-eval time per call", md)
        self.assertIn("0.930", md)        # n6 p3 ratio formatted
        self.assertIn("| 6 | 9 | 7 |", md)  # ratio table row
        self.assertIn("aleph MPS", md)
        # deterministic
        self.assertEqual(md, render_report.render(data, meta))


if __name__ == "__main__":
    unittest.main()
