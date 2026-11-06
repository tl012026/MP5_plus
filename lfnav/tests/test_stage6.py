"""Summary statistics and the cost measurement."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lfnav.eval.measure_cost import measure, svg_scatter
from lfnav.eval.metrics import group_rates, mcnemar, wilson_interval


class MetricsTest(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        interval = wilson_interval(8, 10)
        self.assertLess(interval["low"], 0.8)
        self.assertGreater(interval["high"], 0.8)
        paired = mcnemar([True, True, False, False], [True, False, True, False])
        self.assertEqual(paired["only_reference"], 1)
        self.assertEqual(paired["only_other"], 1)
        grouped = group_rates([{"group": "easy", "success": True}, {"group": "easy", "success": False}])
        self.assertEqual(grouped["easy"]["n"], 2)

    def test_cost_svg(self):
        report = measure()
        self.assertIn("mean_s", report["fake_encoder"])
        self.assertIsNone(report["gpu"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.svg"
            svg_scatter([(report["c2"]["mean_s"], 0.5)], path)
            self.assertIn(b"<svg", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
