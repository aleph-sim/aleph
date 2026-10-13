"""Threshold CSV migration: asymmetric bounds, legacy data and missing evidence."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import warnings

spec = importlib.util.spec_from_file_location(
    "qec_threshold_plot", Path(__file__).resolve().parents[1] / "qec_threshold_plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class TestThresholdBounds(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text(text)
            return plot.load(path)[0][3]

    def test_asymmetric_bounds_and_zero_trials(self):
        _, rates, errors = self.load(
            "d,p,shots,logical_errors,rate,ci95,ci95_lower,ci95_upper,ci95_method\n"
            "3,0.01,1000,0,0,0,0,0.0036820839,clopper-pearson\n"
            "3,0.02,0,0,0,0,0,1,clopper-pearson\n")
        self.assertEqual(rates[0], 0)
        self.assertEqual(errors[0][0], 0)
        self.assertAlmostEqual(errors[1][0], 0.0036820839)
        self.assertTrue(math.isnan(rates[1]))

    def test_legacy_half_width_is_not_reinterpreted(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, errors = self.load(
                "d,p,shots,logical_errors,rate,ci95\n3,0.01,100,10,0.1,0.02\n")
        self.assertEqual(errors, [[0.02], [0.02]])
        self.assertIn("Wald", str(caught[0].message))

    def test_bad_counts_and_bounds(self):
        header = "d,p,shots,logical_errors,rate,ci95,ci95_lower,ci95_upper,ci95_method\n"
        for row in [
            "3,0.01,1,2,0.1,0,0,1,clopper-pearson\n",
            "3,0.01,100,10,0.1,0,0.2,0.3,clopper-pearson\n",
            "3,0.01,100,10,0.1,0,nan,1,clopper-pearson\n",
            "3,0.01,100,10,0.1,0,0,1,weighted\n",
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load(header + row)


if __name__ == "__main__":
    unittest.main()
