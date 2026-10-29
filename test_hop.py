import unittest
from pathlib import Path

from tspga.tsplib import load_tsplib, parse_tsplib

SAMPLE = """NAME : demo4
COMMENT : four cities
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 3
3 4 3
4 4 0
EOF
"""


class TsplibTest(unittest.TestCase):
    def test_parses_header_and_coordinates(self):
        inst = parse_tsplib(SAMPLE)
        self.assertEqual(inst.name, "demo4")
        self.assertEqual(inst.n_cities, 4)
        self.assertEqual(inst.dist[0, 2], 5.0)           # 3-4-5
        self.assertIsNone(inst.known_optimum)

    def test_rejects_unsupported_weight_type(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SAMPLE.replace("EUC_2D", "GEO"))

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            parse_tsplib(SAMPLE.replace("DIMENSION : 4", "DIMENSION : 5"))

    def test_bundled_kro_a100(self):
        inst = load_tsplib(Path(__file__).resolve().parent.parent / "data" / "kroA100.tsp")
        self.assertEqual(inst.name, "kroA100")
        self.assertEqual(inst.n_cities, 100)
        self.assertEqual(inst.known_optimum, 21282)       # оптимум из TSPLIB
        self.assertEqual(tuple(inst.coords[0]), (1380, 939))
        self.assertEqual(tuple(inst.coords[-1]), (3950, 1558))


if __name__ == "__main__":
    unittest.main()
