import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from puzzle_solver.solver import (
    PoseCandidate,
    SolverConfig,
    _apply_safe_placement_gap,
    _polygon_edge_distance,
    _resolve_placement_overlaps,
    _score_card_features,
    _score_edge_patterns,
    _violates_trusted_corner_constraint,
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
    @staticmethod
    def _evaluated_candidate(
        search_phase: str,
        fallback_reasons: list[str] | None = None,
    ) -> tuple:
        polygon = np.asarray(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]],
            dtype=float,
        )
        pose = PoseCandidate(
            piece_index=0,
            rotation_rad=0.0,
            translation=(0.0, 0.0),
            polygon=polygon,
            mask=0,
            cell_count=6000,
            boundary_side="top",
            source_edge=0,
        )
        return (
            0.1,
            0.1,
            100.0,
            60.0,
            [pose],
            {"hole_ratio": 0.0, "overlap_ratio": 0.0},
            [polygon],
            [np.zeros(2)],
            [],
            1.5,
            0.0,
            math.inf,
            True,
            True,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            {},
            0.0,
            0.0,
            {},
            list(fallback_reasons or []),
            search_phase,
        )

    def test_vectorized_polygon_edge_distance(self):
        square = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        separated = square + [5.0, 0.0]
        touching = square + [2.0, 0.0]
        crossing = np.asarray([[1, -1], [3, -1], [3, 1], [1, 1]], dtype=float)

        self.assertAlmostEqual(_polygon_edge_distance(square, separated), 3.0)
        self.assertEqual(_polygon_edge_distance(square, touching), 0.0)
        self.assertEqual(_polygon_edge_distance(square, crossing), 0.0)

    def test_invalid_overlap_skips_clearance_calculation(self):
        polygons = [
            np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
            np.asarray([[2, 0], [4, 0], [4, 2], [2, 2]], dtype=float),
        ]
        config = SolverConfig(
            placement_gap_mm=1.5,
            max_placement_gap_mm=1.5,
            final_overlap_tolerance_mm2=0.25,
        )
        unresolved = (polygons, [np.zeros(2), np.zeros(2)], 1.0)

        with patch(
            "puzzle_solver.solver._resolve_placement_overlaps",
            return_value=unresolved,
        ), patch("puzzle_solver.solver._minimum_piece_clearance") as clearance:
            with self.assertRaises(RuntimeError):
                _apply_safe_placement_gap(polygons, ["a", "b"], 4.0, 2.0, config)

        clearance.assert_not_called()

    def test_pairwise_repulsion_removes_residual_overlap(self):
        polygons = [
            np.asarray([[0, 0], [50, 0], [50, 30], [0, 30]], dtype=float),
            np.asarray([[48, 0], [98, 0], [98, 30], [48, 30]], dtype=float),
        ]
        placed, offsets, overlap = _resolve_placement_overlaps(
            polygons,
            [np.zeros(2), np.zeros(2)],
            maximum_offset_mm=8.0,
            tolerance_mm2=0.25,
        )
        self.assertLessEqual(overlap, 0.25)
        self.assertTrue(all(np.linalg.norm(offset) <= 8.0 + 1e-9 for offset in offsets))
        self.assertEqual(len(placed), 2)

    def test_adjacency_offsets_propagate_gap_across_parallel_strips(self):
        polygons = [
            np.asarray(
                [[start, 0.0], [start + 20.0, 0.0], [start + 20.0, 30.0], [start, 30.0]],
                dtype=float,
            )
            for start in (0.0, 20.0, 40.0, 60.0)
        ]
        config = SolverConfig(
            placement_gap_mm=1.5,
            max_placement_gap_mm=8.0,
            final_overlap_tolerance_mm2=0.25,
        )

        placed, offsets, _, _, overlap, achieved, satisfied = (
            _apply_safe_placement_gap(
                polygons, ["a", "b", "c", "d"], 80.0, 30.0, config
            )
        )

        self.assertTrue(satisfied)
        self.assertGreaterEqual(achieved, 1.5 - 1e-6)
        self.assertLessEqual(overlap, 0.25)
        offset_x = [float(offset[0]) for offset in offsets]
        self.assertTrue(
            all(second - first >= 1.5 - 1e-4 for first, second in zip(offset_x, offset_x[1:]))
        )
        self.assertEqual(len(placed), 4)

    def test_best_non_overlapping_gap_is_returned_as_fallback(self):
        polygons = [
            np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
            np.asarray([[10.3, 0], [20.3, 0], [20.3, 10], [10.3, 10]], dtype=float),
        ]
        config = SolverConfig(
            placement_gap_mm=1.5,
            max_placement_gap_mm=2.0,
            final_overlap_tolerance_mm2=0.25,
        )
        zero_offsets = [np.zeros(2), np.zeros(2)]

        with patch(
            "puzzle_solver.solver._placement_offsets",
            return_value=zero_offsets,
        ):
            _, _, _, _, overlap, achieved, satisfied = _apply_safe_placement_gap(
                polygons, ["a", "b"], 20.3, 10.0, config
            )

        self.assertFalse(satisfied)
        self.assertLessEqual(overlap, 0.25)
        self.assertAlmostEqual(achieved, 0.3, places=5)

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
        self.assertTrue(result["metrics"]["placement_gap_satisfied"])
        self.assertGreaterEqual(result["metrics"]["achieved_placement_gap_mm"], 1.5)
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

    def test_best_scored_candidate_is_returned_when_soft_constraint_rejects_all(self):
        rectangle = [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]]
        edge_profiles = [[np.zeros((32, 3), dtype=float) for _ in range(4)]]
        piece_features = [
            {
                "corner_roundness_mm": [0.0] * 4,
                "corner_ink_density": [0.0] * 4,
                "corner_red_density": [0.0] * 4,
                "corner_black_density": [0.0] * 4,
                "ink_points_mm": [[1.0, 1.0]] * 300,
                "ink_point_colours": [0] * 300,
            }
        ]

        with patch(
            "puzzle_solver.solver._score_card_symmetry",
            return_value=(0.9, 1.0, {}),
        ):
            result = solve_puzzle(
                [rectangle],
                config=SolverConfig(
                    width_range=(100.0, 100.0),
                    height_range=(60.0, 60.0),
                    max_card_symmetry_mismatch=0.6,
                    enable_relaxed_retry=False,
                ),
                edge_profiles=edge_profiles,
                piece_features=piece_features,
            )

        self.assertTrue(result["metrics"]["best_effort_fallback_used"])
        self.assertEqual(
            result["metrics"]["best_effort_reasons"],
            ["card_symmetry_mismatch"],
        )
        self.assertEqual(result["metrics"]["search_phase"], "strict")

    def test_rejected_phase_continues_to_the_next_search_phase(self):
        polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]]
        pose = self._evaluated_candidate("strict")[4][0]
        raw_solution = [(0.1, [pose], {"hole_ratio": 0.0, "overlap_ratio": 0.0})]
        relaxed_candidate = self._evaluated_candidate("relaxed")

        with (
            patch(
                "puzzle_solver.solver.candidate_rectangles",
                return_value=[(100.0, 60.0, 0.0)],
            ),
            patch(
                "puzzle_solver.solver._search_rectangle",
                return_value=(raw_solution, 1, False),
            ),
            patch(
                "puzzle_solver.solver._evaluate_solution_candidates",
                side_effect=[([], [], 0, False), ([relaxed_candidate], [], 0, False)],
            ) as evaluate,
        ):
            result = solve_puzzle(
                [polygon],
                config=SolverConfig(
                    width_range=(100.0, 100.0),
                    height_range=(60.0, 60.0),
                    relaxed_search_first=False,
                ),
            )

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["metrics"]["search_phase"], "relaxed")

    def test_timeout_returns_the_best_safe_fallback_candidate(self):
        polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]]
        pose = self._evaluated_candidate("strict")[4][0]
        raw_solution = [(0.1, [pose], {"hole_ratio": 0.0, "overlap_ratio": 0.0})]
        fallback = self._evaluated_candidate("strict", ["pattern_mismatch"])

        with (
            patch(
                "puzzle_solver.solver.candidate_rectangles",
                return_value=[(100.0, 60.0, 0.0)],
            ),
            patch(
                "puzzle_solver.solver._search_rectangle",
                return_value=(raw_solution, 1, False),
            ),
            patch(
                "puzzle_solver.solver._evaluate_solution_candidates",
                return_value=([], [fallback], 1, True),
            ) as evaluate,
        ):
            result = solve_puzzle(
                [polygon],
                config=SolverConfig(
                    width_range=(100.0, 100.0),
                    height_range=(60.0, 60.0),
                    relaxed_search_first=False,
                ),
            )

        evaluate.assert_called_once()
        self.assertTrue(result["metrics"]["search_timed_out"])
        self.assertTrue(result["metrics"]["best_effort_fallback_used"])
        self.assertEqual(
            result["metrics"]["best_effort_reasons"],
            ["search_timeout", "pattern_mismatch"],
        )

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

    def test_sparse_print_mismatch_is_not_diluted_by_white_samples(self):
        white = np.full((48, 3), 240.0)
        printed = white.copy()
        printed[20:26] = [25.0, 128.0, 128.0]
        adjacency = {
            "piece_a": 0,
            "piece_b": 1,
            "edge_a": 0,
            "edge_b": 0,
            "edge_b_reversed": False,
            "edge_a_interval": [0.0, 1.0],
            "edge_b_interval": [0.0, 1.0],
        }

        _, mismatch, evidence = _score_edge_patterns(
            [adjacency], [[printed], [white]]
        )

        self.assertGreater(evidence, 0.5)
        self.assertGreater(mismatch, 0.05)

    def test_trusted_rounded_corner_must_land_on_card_corner(self):
        pose = SimpleNamespace(
            piece_index=0,
            polygon=np.asarray(
                [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [20.0, 10.0], [20.0, 20.0]]
            ),
        )
        features = {
            "corner_roundness_mm": [0.0, 0.0, 1.2, 0.0, 0.0],
            "corner_ink_density": [0.0] * 5,
            "corner_red_density": [0.0] * 5,
            "corner_black_density": [0.0] * 5,
        }

        *_, details = _score_card_features(
            [pose], 30.0, 20.0, [features], config=SolverConfig()
        )

        self.assertEqual(details["trusted_rounded_corner_count"], 1)
        self.assertEqual(details["misplaced_rounded_corner_count"], 1)
        self.assertFalse(
            _violates_trusted_corner_constraint(details, SolverConfig())
        )

    def test_trusted_rounded_corner_accepts_outer_card_corner(self):
        pose = SimpleNamespace(
            piece_index=0,
            polygon=np.asarray(
                [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]
            ),
        )
        features = {
            "corner_roundness_mm": [1.2, 0.0, 0.0, 0.0],
            "corner_ink_density": [0.0] * 4,
            "corner_red_density": [0.0] * 4,
            "corner_black_density": [0.0] * 4,
        }

        *_, details = _score_card_features(
            [pose], 20.0, 10.0, [features], config=SolverConfig()
        )

        self.assertEqual(details["trusted_rounded_corner_count"], 1)
        self.assertEqual(details["misplaced_rounded_corner_count"], 0)

    def test_two_misplaced_rounded_corners_reject_candidate(self):
        details = {"misplaced_rounded_corner_count": 2}
        self.assertTrue(
            _violates_trusted_corner_constraint(details, SolverConfig())
        )


if __name__ == "__main__":
    unittest.main()
