import unittest

import numpy as np

from tspga.factory import IslandConfig, build_model
from tspga.instance import circle_instance, is_valid_tour, random_instance

INSTANCE = random_instance(25, seed=11)


def small(**kw):
    base = dict(n_islands=3, pop_size=20, epochs=6, interval=5, migrants=2, seed=1)
    base.update(kw)
    return IslandConfig(**base)


class IslandModelTest(unittest.TestCase):
    def test_result_structure(self):
        cfg = small()
        r = build_model(INSTANCE, cfg).run()
        self.assertEqual(len(r.best_history), 3)
        self.assertEqual(len(r.best_history[0]), cfg.epochs * cfg.interval + 1)
        self.assertEqual(len(r.mean_history[0]), cfg.epochs * cfg.interval + 1)
        self.assertEqual(len(r.snapshots[0]), cfg.epochs)
        self.assertTrue(is_valid_tour(r.best_tour, 25))
        self.assertAlmostEqual(INSTANCE.tour_length(r.best_tour), r.best_cost)
        self.assertEqual(r.evaluations, 3 * 20 * (cfg.epochs * cfg.interval + 1))

    def test_best_island_has_the_lowest_final_cost(self):
        r = build_model(INSTANCE, small()).run()
        finals = [h[-1] for h in r.best_history]
        self.assertEqual(r.best_island, int(np.argmin(finals)))
        self.assertAlmostEqual(r.best_cost, min(finals))

    def test_same_seed_gives_identical_results(self):
        a = build_model(INSTANCE, small()).run()
        b = build_model(INSTANCE, small()).run()
        self.assertEqual(a.best_cost, b.best_cost)
        np.testing.assert_array_equal(a.best_tour, b.best_tour)

    def test_result_does_not_depend_on_number_of_worker_threads(self):
        serial = build_model(INSTANCE, small(max_workers=1)).run()
        parallel = build_model(INSTANCE, small(max_workers=3)).run()
        for s, p in zip(serial.best_history, parallel.best_history):
            np.testing.assert_array_equal(s, p)

    def test_migration_goes_through_hop_islands(self):
        r = build_model(INSTANCE, small(topology="ring")).run()
        self.assertGreater(r.migrants_sent, 0)
        self.assertGreater(r.migrants_received, 0)
        self.assertLessEqual(r.migrants_received, r.migrants_sent)

    def test_isolated_islands_exchange_nothing(self):
        r = build_model(INSTANCE, small(topology="none")).run()
        self.assertEqual(r.migrants_sent, 0)
        self.assertEqual(r.migrants_received, 0)

    def test_migration_changes_the_evolution(self):
        with_migration = build_model(INSTANCE, small(topology="ring")).run()
        without = build_model(INSTANCE, small(topology="none")).run()
        self.assertNotEqual([h[-1] for h in with_migration.best_history], [h[-1] for h in without.best_history])

    def test_single_island_works_as_plain_ga(self):
        r = build_model(INSTANCE, small(n_islands=1)).run()
        self.assertEqual(len(r.best_history), 1)
        self.assertEqual(r.best_island, 0)

    def test_stagnation_stops_early(self):
        cfg = small(epochs=200, interval=3, patience_epochs=4, pop_size=10)
        r = build_model(INSTANCE, cfg).run()
        self.assertLess(r.epochs_run, 200)
        self.assertEqual(len(r.best_history[0]), r.epochs_run * 3 + 1)

    def test_asynchronous_mode_runs_and_returns_valid_tours(self):
        r = build_model(INSTANCE, small(synchronous=False)).run()
        self.assertTrue(is_valid_tour(r.best_tour, 25))
        self.assertEqual(len(r.best_history[0]), 6 * 5 + 1)
        self.assertGreater(r.migrants_sent, 0)

    def test_finds_near_optimal_tour_on_circle(self):
        circle = circle_instance(30)
        r = build_model(circle, IslandConfig(n_islands=3, pop_size=60, epochs=20, interval=15, migrants=3, seed=0)).run()
        self.assertLessEqual(r.best_cost, circle.known_optimum * 1.05)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            build_model(INSTANCE, small(topology="star"))
        with self.assertRaises(ValueError):
            build_model(INSTANCE, small(n_islands=0))
        with self.assertRaises(ValueError):
            build_model(INSTANCE, small(migrants=20, pop_size=20))


if __name__ == "__main__":
    unittest.main()
