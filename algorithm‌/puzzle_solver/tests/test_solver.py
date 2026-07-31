import math
import unittest

import numpy as np

from puzzle_solver.solver import (
    SolverConfig,
    _score_edge_patterns,
    polygon_area,
    solve_puzzle,
)


PIECES = [
    [[40.0, 10.0], [40.0, 30.0], [28.0, 46.0], [20.0, 10.0]],
    [[180.0, 40.0], [144.0, 48.0], [104.0, 18.0], [180.0, 30.0]],
    [[0.0, 110.0], [76.0, 122.0], [100.0, 140.0], [0.0, 140.0]],
    [[120.0, 80.0], [200.0, 80.0], [200.0, 140.0], [176.0, 122.0], [136.0, 92.0]],
]


class SolverTest(unittest.TestCase):
    def test_fixed_four_piece_example(self):
        result = solve_puzzle(
            PIECES,
            ["upper_left", "middle_left", "bottom", "upper_right"],
            target_origin_mm=(55.0, 190.0),
            config=SolverConfig(max_search_nodes_per_rectangle=300_000),
        )

        rectangle = result["rectangle"]
        self.assertAlmostEqual(rectangle["width_mm"], 100.0, delta=1.0)
        self.assertAlmostEqual(rectangle["height_mm"], 60.0, delta=1.0)
        self.assertLess(result["metrics"]["hole_ratio"], 0.02)
        self.assertLess(result["metrics"]["overlap_ratio"], 0.002)
        self.assertLessEqual(
            result["metrics"]["final_overlap_area_mm2"],
            result["config"]["final_overlap_tolerance_mm2"],
        )
        self.assertGreaterEqual(result["metrics"]["applied_placement_gap_mm"], 1.5)
        self.assertLessEqual(result["metrics"]["max_adjacent_vertex_distance_mm"], 20.0)
        self.assertGreaterEqual(len(result["adjacencies"]), 3)
        for adjacency in result["adjacencies"]:
            self.assertLessEqual(adjacency["max_corresponding_vertex_distance_mm"], 20.0)
        self.assertEqual(len(result["pieces"]), 4)

        source_areas = [polygon_area(np.asarray(piece)) for piece in PIECES]
        target_areas = [
            polygon_area(np.asarray(piece["target_polygon_mm"]))
            for piece in result["pieces"]
        ]
        for source, target in zip(source_areas, target_areas):
            self.assertTrue(math.isclose(source, target, rel_tol=1e-9, abs_tol=1e-6))

    def test_submillimetre_vertex_noise(self):
        random = np.random.default_rng(7)
        noisy_pieces = [
            (np.asarray(piece, dtype=float) + random.normal(0.0, 0.25, (len(piece), 2))).tolist()
            for piece in PIECES
        ]

        result = solve_puzzle(
            noisy_pieces,
            config=SolverConfig(
                dimension_area_tolerance=0.06,
                max_hole_ratio=0.04,
            ),
        )

        self.assertAlmostEqual(result["rectangle"]["width_mm"], 100.0, delta=1.0)
        self.assertAlmostEqual(result["rectangle"]["height_mm"], 60.0, delta=1.0)
        self.assertLess(result["metrics"]["hole_ratio"], 0.01)
        self.assertLess(result["metrics"]["overlap_ratio"], 0.003)

    def test_reversed_edge_pattern_profiles_match(self):
        profile = np.column_stack(
            [
                np.linspace(20.0, 220.0, 32),
                np.linspace(180.0, 40.0, 32),
                np.full(32, 120.0),
            ]
        )
        adjacency = {
            "piece_a": 0,
            "piece_b": 1,
            "edge_a": 0,
            "edge_b": 0,
            "edge_b_reversed": True,
            "edge_a_interval": [0.0, 1.0],
            "edge_b_interval": [1.0, 0.0],
        }
        _, mismatch, evidence = _score_edge_patterns(
            [adjacency], [[profile], [profile[::-1]]]
        )
        self.assertGreater(evidence, 0.5)
        self.assertLess(mismatch, 1e-9)


if __name__ == "__main__":
    unittest.main()
