import unittest

from tspga.topology import FullyConnectedTopology, IsolatedTopology, RingTopology


class TopologyTest(unittest.TestCase):
    def test_ring(self):
        self.assertEqual(RingTopology().edges(4), [(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_ring_of_two_is_bidirectional(self):
        self.assertEqual(RingTopology().edges(2), [(0, 1), (1, 0)])

    def test_single_island_has_no_edges(self):
        for topology in (RingTopology(), FullyConnectedTopology(), IsolatedTopology()):
            self.assertEqual(topology.edges(1), [])

    def test_fully_connected_has_all_ordered_pairs(self):
        edges = FullyConnectedTopology().edges(4)
        self.assertEqual(len(edges), 12)
        self.assertNotIn((2, 2), edges)
        self.assertEqual(len(set(edges)), 12)

    def test_isolated_has_no_edges(self):
        self.assertEqual(IsolatedTopology().edges(5), [])

    def test_island_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            RingTopology().edges(0)


if __name__ == "__main__":
    unittest.main()
