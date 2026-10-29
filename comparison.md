import unittest

import numpy as np

from tspga.instance import is_valid_tour
from tspga.operators import InversionMutation, OrderCrossover, SwapMutation, TournamentSelection, order_crossover
from tspga.population import random_tours


class TournamentTest(unittest.TestCase):
    def test_prefers_fitter_individuals(self):
        rng = np.random.default_rng(0)
        fitness = np.arange(40, dtype=float)
        idx = TournamentSelection(3).select(fitness, 4000, rng)
        self.assertEqual(idx.shape, (4000,))
        self.assertTrue(((idx >= 0) & (idx < 40)).all())
        self.assertGreater(fitness[idx].mean(), fitness.mean())

    def test_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            TournamentSelection(0)


class OrderCrossoverTest(unittest.TestCase):
    def test_keeps_segment_of_first_parent_and_order_of_second(self):
        p1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        p2 = np.array([7, 6, 5, 4, 3, 2, 1, 0])
        child = order_crossover(p1, p2, 2, 5)
        np.testing.assert_array_equal(child[2:5], [2, 3, 4])
        self.assertTrue(is_valid_tour(child, 8))
        # остальные города идут в порядке второго родителя, начиная после второй точки разрыва
        np.testing.assert_array_equal(child, [6, 5, 2, 3, 4, 1, 0, 7])

    def test_children_are_always_valid_tours(self):
        rng = np.random.default_rng(1)
        parents = random_tours(rng, 40, 25)
        children = OrderCrossover(1.0).cross(parents, rng)
        self.assertEqual(children.shape, parents.shape)
        self.assertTrue(all(is_valid_tour(c, 25) for c in children))
        self.assertFalse((children == parents).all())

    def test_rate_zero_returns_parents(self):
        rng = np.random.default_rng(2)
        parents = random_tours(rng, 10, 12)
        np.testing.assert_array_equal(OrderCrossover(0.0).cross(parents, rng), parents)

    def test_odd_number_of_parents_rejected(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            OrderCrossover(1.0).cross(random_tours(rng, 5, 8), rng)

    def test_invalid_rate_rejected(self):
        with self.assertRaises(ValueError):
            OrderCrossover(1.2)


class MutationTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.tours = random_tours(self.rng, 60, 30)

    def test_mutants_stay_valid_and_change(self):
        for mutation in (InversionMutation(1.0), SwapMutation(1.0)):
            out = mutation.mutate(self.tours, self.rng)
            self.assertTrue(all(is_valid_tour(t, 30) for t in out))
            self.assertTrue((out != self.tours).any(axis=1).all())

    def test_rate_zero_changes_nothing(self):
        for mutation in (InversionMutation(0.0), SwapMutation(0.0)):
            np.testing.assert_array_equal(mutation.mutate(self.tours, self.rng), self.tours)

    def test_input_is_not_modified(self):
        before = self.tours.copy()
        InversionMutation(1.0).mutate(self.tours, self.rng)
        np.testing.assert_array_equal(self.tours, before)

    def test_inversion_reverses_a_contiguous_segment(self):
        tour = np.arange(12)[None, :]
        out = InversionMutation(1.0).mutate(tour, self.rng)[0]
        changed = np.flatnonzero(out != tour[0])
        lo, hi = changed.min(), changed.max()
        np.testing.assert_array_equal(out[lo:hi + 1], tour[0][lo:hi + 1][::-1])


if __name__ == "__main__":
    unittest.main()
