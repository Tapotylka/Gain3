import math
import unittest

import numpy as np

from tspga.instance import TSPInstance, circle_instance, is_valid_tour, random_instance


class InstanceTest(unittest.TestCase):
    def setUp(self):
        self.square = TSPInstance.from_coords("square", [(0, 0), (0, 1), (1, 1), (1, 0)])

    def test_distance_matrix_is_symmetric_with_zero_diagonal(self):
        d = random_instance(12, seed=1).dist
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_allclose(np.diag(d), 0.0)

    def test_tour_length_of_square(self):
        self.assertAlmostEqual(self.square.tour_length(np.array([0, 1, 2, 3])), 4.0)
        self.assertAlmostEqual(self.square.tour_length(np.array([0, 2, 1, 3])), 2 + 2 * math.sqrt(2))

    def test_tour_lengths_is_vectorised_and_matches_scalar_version(self):
        inst = random_instance(15, seed=3)
        rng = np.random.default_rng(0)
        tours = np.array([rng.permutation(15) for _ in range(6)])
        np.testing.assert_allclose(inst.tour_lengths(tours), [inst.tour_length(t) for t in tours])

    def test_tour_length_does_not_depend_on_start_or_direction(self):
        inst = random_instance(10, seed=5)
        tour = np.random.default_rng(1).permutation(10)
        base = inst.tour_length(tour)
        self.assertAlmostEqual(inst.tour_length(np.roll(tour, 4)), base)
        self.assertAlmostEqual(inst.tour_length(tour[::-1]), base)

    def test_rounded_distances_follow_tsplib_euc_2d(self):
        inst = TSPInstance.from_coords("r", [(0, 0), (1, 1), (5, 0)], rounded=True)
        self.assertEqual(inst.dist[0, 1], 1.0)          # nint(1.414) = 1

    def test_circle_instance_has_known_optimum(self):
        inst = circle_instance(40, radius=10.0)
        self.assertAlmostEqual(inst.known_optimum, 2 * 40 * 10.0 * math.sin(math.pi / 40))
        self.assertAlmostEqual(inst.tour_length(np.arange(40)), inst.known_optimum)

    def test_random_instance_is_reproducible(self):
        np.testing.assert_array_equal(random_instance(8, seed=2).coords, random_instance(8, seed=2).coords)

    def test_is_valid_tour(self):
        self.assertTrue(is_valid_tour(np.array([2, 0, 1]), 3))
        self.assertFalse(is_valid_tour(np.array([0, 0, 1]), 3))
        self.assertFalse(is_valid_tour(np.array([0, 1]), 3))

    def test_instance_needs_at_least_three_cities(self):
        with self.assertRaises(ValueError):
            TSPInstance.from_coords("tiny", [(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
