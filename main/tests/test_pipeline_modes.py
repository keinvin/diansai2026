import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from main.config import load_config
from main.pipeline import MainPipeline


class PipelineSolverModeTest(unittest.TestCase):
    def setUp(self):
        document = load_config()
        document["timing"] = {"enabled": False, "log_enabled": False}
        self.pipeline = MainPipeline(document)
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.calibration = Mock()
        self.calibration.undistort_image.return_value = self.frame
        self.result = SimpleNamespace(pieces=[])
        self.solution = {
            "rectangle": {"origin_mm": [0.0, 0.0], "width_mm": 100.0, "height_mm": 60.0},
            "pieces": [],
        }

    def _run(self, use_piece_features: bool):
        with (
            patch("main.pipeline.Calibration.from_dict", return_value=self.calibration),
            patch("main.pipeline.detect_pieces", return_value=self.result),
            patch("main.pipeline.extract_edge_profiles", return_value=["edges"]) as edges,
            patch("main.pipeline.extract_piece_features", return_value=["features"]) as features,
            patch("main.pipeline.solve_puzzle", return_value=self.solution) as solve,
        ):
            self.pipeline.recognize(
                self.frame,
                puzzle_search_enabled=True,
                use_piece_features=use_piece_features,
            )
        return edges, features, solve.call_args.kwargs

    def test_white_puzzle_uses_geometry_only_and_strict_search_first(self):
        edges, features, solve_args = self._run(use_piece_features=False)

        edges.assert_not_called()
        features.assert_not_called()
        self.assertIsNone(solve_args["edge_profiles"])
        self.assertIsNone(solve_args["piece_features"])
        self.assertFalse(solve_args["config"].relaxed_search_first)
        self.assertTrue(self.pipeline.solver_config.relaxed_search_first)

    def test_card_puzzle_keeps_appearance_search_configuration(self):
        edges, features, solve_args = self._run(use_piece_features=True)

        edges.assert_called_once()
        features.assert_called_once()
        self.assertEqual(solve_args["edge_profiles"], ["edges"])
        self.assertEqual(solve_args["piece_features"], ["features"])
        self.assertIs(solve_args["config"], self.pipeline.solver_config)


if __name__ == "__main__":
    unittest.main()
