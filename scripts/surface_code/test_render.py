"""Golden test for render_report.render (stdlib unittest)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import render_report  # noqa: E402

HERE = Path(__file__).parent
TD = HERE / "testdata"


class RenderTest(unittest.TestCase):
    def test_render_golden(self):
        aleph = json.loads((TD / "aleph.json").read_text())
        stim = json.loads((TD / "stim.json").read_text())
        meta = json.loads((TD / "meta.json").read_text())
        md = render_report.render(aleph, stim, meta)
        # Structural assertions (robust to float formatting).
        self.assertIn("# Phase 4 — Surface-code syndrome extraction (stabilizer)", md)
        self.assertIn("| 11 | 241 |", md)
        self.assertIn("aleph / Stim", md)
        # d=3: 0.00001*1000 = 0.010 ms aleph / 0.000002*1000 = 0.002 ms stim = 5.00×
        self.assertIn("| 3 | 17 | 0.010 | 0.002 | 5.00× |", md)
        # All five distances present.
        for d in (3, 5, 7, 9, 11):
            self.assertRegex(md, rf"\n\| {d} \|")


if __name__ == "__main__":
    unittest.main()
