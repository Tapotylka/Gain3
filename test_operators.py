import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tspga.factory import IslandConfig, build_model
from tspga.instance import random_instance
from tspga.visualization import animate_routes, plot_convergence, plot_routes


class VisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance = random_instance(20, seed=2)
        cls.result = build_model(cls.instance, IslandConfig(n_islands=3, pop_size=20, epochs=4, interval=4, migrants=2)).run()

    def test_files_are_created(self):
        with tempfile.TemporaryDirectory() as d:
            routes, conv, gif = Path(d, "r.png"), Path(d, "c.png"), Path(d, "a.gif")
            plot_routes(self.instance, self.result, routes)
            early = Path(d, "early.png")
            plot_routes(self.instance, self.result, early, epoch=1)
            self.assertGreater(early.stat().st_size, 1000)
            plot_convergence(self.result, conv)
            animate_routes(self.instance, self.result, gif)
            for f in (routes, conv, gif):
                self.assertGreater(f.stat().st_size, 1000, f.name)

    def test_epoch_out_of_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                plot_routes(self.instance, self.result, Path(d, "x.png"), epoch=99)


if __name__ == "__main__":
    unittest.main()
