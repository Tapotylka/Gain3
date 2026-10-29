import unittest

import numpy as np

from tspga.experiments import estimate_runs, run_series, summarize
from tspga.factory import IslandConfig
from tspga.instance import circle_instance


class EstimateRunsTest(unittest.TestCase):
    def test_formula_uses_relative_half_width(self):
        costs = np.array([100.0, 110.0, 90.0, 105.0, 95.0])          # mean 100
        s = costs.std(ddof=1)
        expected = int(np.ceil((1.96 * s / (0.01 * 100.0)) ** 2))
        self.assertEqual(estimate_runs(costs, rel_error=0.01, minimum=1, maximum=10_000), expected)

    def test_result_is_clamped(self):
        noisy = np.array([1.0, 100.0, 1.0, 100.0])
        self.assertEqual(estimate_runs(noisy, maximum=30), 30)
        constant = np.array([5.0, 5.0, 5.0])
        self.assertEqual(estimate_runs(constant, minimum=5), 5)

    def test_needs_at_least_two_observations(self):
        with self.assertRaises(ValueError):
            estimate_runs(np.array([1.0]))


class SeriesTest(unittest.TestCase):
    def test_run_series_and_summary(self):
        inst = circle_instance(20)
        cfg = IslandConfig(n_islands=2, pop_size=20, epochs=4, interval=5, migrants=2)
        series = run_series(inst, cfg, runs=3, base_seed=100)
        self.assertEqual(len(series.costs), 3)
        s = summarize(series, inst.known_optimum)
        self.assertEqual(s["runs"], 3)
        self.assertLessEqual(s["min"], s["mean"] + 1e-9)
        self.assertGreaterEqual(s["max"], s["mean"] - 1e-9)
        self.assertGreaterEqual(s["gap_mean_pct"], 0.0)

    def test_different_seeds_are_used_for_each_run(self):
        inst = circle_instance(20)
        cfg = IslandConfig(n_islands=2, pop_size=20, epochs=2, interval=3, migrants=1)
        series = run_series(inst, cfg, runs=4, base_seed=0)
        self.assertGreater(len(set(series.costs)), 1)

    def test_runs_must_be_positive(self):
        with self.assertRaises(ValueError):
            run_series(circle_instance(10), IslandConfig(), runs=0)


if __name__ == "__main__":
    unittest.main()
