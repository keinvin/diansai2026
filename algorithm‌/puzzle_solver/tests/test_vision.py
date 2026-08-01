import importlib.util
import unittest

import numpy as np


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
class VisionTest(unittest.TestCase):
    def test_multiscale_fit_preserves_concave_piece_area(self):
        import cv2

        from puzzle_solver.vision import VisionConfig, _fit_piece_polygon

        corners = np.asarray([[0, 0], [70, 0], [70, 45], [36, 24], [0, 45]], dtype=float)
        contour = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            contour.extend(start + (end - start) * fraction for fraction in np.linspace(0, 1, 30, endpoint=False))
        contour = np.asarray(contour)
        contour[:, 1] += 0.18 * np.sin(np.arange(len(contour)) * 1.7)
        fitted, _ = _fit_piece_polygon(
            contour,
            VisionConfig(
                approx_epsilon_mm=0.2,
                max_approx_epsilon_mm=3.0,
                max_vertices=5,
            ),
        )
        source_area = abs(cv2.contourArea(contour.astype(np.float32)))
        fitted_area = abs(cv2.contourArea(fitted.astype(np.float32)))
        self.assertLessEqual(len(fitted), 5)
        self.assertGreaterEqual(len(fitted), 3)
        self.assertLess(abs(fitted_area - source_area) / source_area, 0.03)

    def test_rounded_card_corner_pair_becomes_one_corner(self):
        from puzzle_solver.vision import _merge_rounded_corner_vertices

        polygon = np.asarray(
            [[3.0, 0.0], [47.0, 0.0], [50.0, 3.0], [50.0, 30.0], [0.0, 30.0], [0.0, 3.0]]
        )
        cleaned = _merge_rounded_corner_vertices(polygon, 8.0)
        self.assertEqual(len(cleaned), 4)
        expected = np.asarray([[0.0, 0.0], [50.0, 0.0], [50.0, 30.0], [0.0, 30.0]])
        for corner in expected:
            self.assertTrue(np.any(np.all(np.isclose(cleaned, corner), axis=1)))

    def test_legal_ten_millimetre_edge_is_not_rounded_corner(self):
        from puzzle_solver.vision import _merge_rounded_corner_vertices

        polygon = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 10.0], [20.0, 30.0], [0.0, 30.0]]
        )
        cleaned = _merge_rounded_corner_vertices(polygon, 8.0)
        self.assertEqual(len(cleaned), 5)

    def test_rounded_piece_detection_returns_four_corners(self):
        import cv2

        from puzzle_solver.vision import Calibration, VisionConfig, detect_pieces

        pixels_per_mm = 4
        image = np.full((220, 300, 3), (155, 85, 35), dtype=np.uint8)
        rounded_mm = np.asarray(
            [[13, 10], [57, 10], [60, 13], [60, 40], [10, 40], [10, 13]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [rounded_mm * pixels_per_mm], (245, 245, 245))
        calibration = Calibration.from_dict(
            {
                "mm_per_pixel": 1.0 / pixels_per_mm,
                "roi_polygon_px": [[0, 0], [299, 0], [299, 219], [0, 219]],
            },
            image.shape,
        )
        result = detect_pieces(
            image,
            calibration,
            VisionConfig(
                morphology_kernel_px=3,
                min_piece_area_mm2=100.0,
                rounded_corner_max_chord_mm=8.0,
            ),
        )
        self.assertEqual(len(result.pieces), 1)
        self.assertEqual(len(result.pieces[0].polygon_mm), 4)

    def test_corner_touching_pieces_are_split(self):
        import cv2

        from puzzle_solver.vision import Calibration, VisionConfig, detect_pieces

        pixels_per_mm = 5
        image = np.full((750, 650, 3), (35, 35, 35), dtype=np.uint8)
        first = np.asarray([[30, 20], [80, 20], [80, 70], [30, 70]]) * pixels_per_mm
        second = np.asarray([[80, 70], [110, 100], [80, 130], [50, 100]]) * pixels_per_mm
        cv2.fillPoly(image, [first.astype(np.int32)], (245, 245, 245))
        cv2.fillPoly(image, [second.astype(np.int32)], (245, 245, 245))
        calibration = Calibration.from_dict(
            {
                "mm_per_pixel": 1.0 / pixels_per_mm,
                "roi_polygon_px": [[0, 0], [649, 0], [649, 749], [0, 749]],
                "use_camera_intrinsics": False,
            },
            image.shape,
        )
        result = detect_pieces(
            image,
            calibration,
            VisionConfig(
                morphology_kernel_px=3,
                min_piece_area_mm2=100.0,
                collinear_vertex_tolerance_mm=0.5,
            ),
        )
        self.assertEqual(len(result.pieces), 2)
        self.assertTrue(all(3 <= len(piece.polygon_mm) <= 5 for piece in result.pieces))

    def test_nearly_straight_false_vertex_is_removed(self):
        from puzzle_solver.vision import _remove_nearly_straight_vertices

        polygon = np.asarray(
            [[0.0, 0.0], [10.0, 0.5], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]
        )
        cleaned = _remove_nearly_straight_vertices(polygon, 170.0, 20.0)
        self.assertEqual(len(cleaned), 4)
        self.assertFalse(np.any(np.all(np.isclose(cleaned, [10.0, 0.5]), axis=1)))

    def test_real_corner_below_angle_threshold_is_retained(self):
        from puzzle_solver.vision import _remove_nearly_straight_vertices

        polygon = np.asarray(
            [[0.0, 0.0], [10.0, 1.8], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]
        )
        cleaned = _remove_nearly_straight_vertices(polygon, 170.0, 20.0)
        self.assertEqual(len(cleaned), 5)

    def test_shallow_corner_between_long_edges_is_retained(self):
        from puzzle_solver.vision import _remove_nearly_straight_vertices

        # Reproduces the blue piece: the 172.9-degree V1 is shallow but both
        # adjacent physical cut edges are much longer than 20 mm.
        polygon = np.asarray(
            [
                [14.4, 217.4],
                [59.7, 251.5],
                [90.8, 269.3],
                [111.4, 253.9],
                [60.5, 182.0],
            ]
        )
        cleaned = _remove_nearly_straight_vertices(polygon, 170.0, 20.0)
        self.assertEqual(len(cleaned), 5)

    def test_rejected_contours_can_be_hidden_from_overlay(self):
        from puzzle_solver.vision import (
            Calibration,
            DetectionResult,
            RejectedContour,
            draw_detection_overlay,
        )

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        calibration = Calibration(np.eye(3), roi_polygon_px=None)
        result = DetectionResult(
            pieces=[],
            mask=np.zeros(image.shape[:2], dtype=np.uint8),
            roi_mask=np.ones(image.shape[:2], dtype=np.uint8),
            rejected_contours=[
                RejectedContour(
                    np.asarray([[5.0, 5.0], [25.0, 5.0], [15.0, 25.0]]),
                    "测试拒绝轮廓",
                )
            ],
        )

        hidden = draw_detection_overlay(
            image, result, calibration, show_rejected_contours=False
        )
        shown = draw_detection_overlay(
            image, result, calibration, show_rejected_contours=True
        )

        self.assertTrue(np.array_equal(hidden, image))
        self.assertFalse(np.array_equal(shown, image))

    def test_synthetic_detection_and_solve(self):
        import cv2

        from puzzle_solver.solver import SolverConfig, solve_puzzle
        from puzzle_solver.vision import Calibration, VisionConfig, detect_pieces

        polygons_mm = [
            [[40, 10], [40, 30], [28, 46], [20, 10]],
            [[180, 40], [144, 48], [104, 18], [180, 30]],
            [[0, 110], [76, 122], [100, 140], [0, 140]],
            [[120, 80], [200, 80], [200, 140], [176, 122], [136, 92]],
        ]
        pixels_per_mm = 4
        image = np.zeros((592, 840, 3), dtype=np.uint8)
        image[:] = (155, 85, 35)
        for polygon in polygons_mm:
            points = np.asarray(polygon, dtype=np.int32) * pixels_per_mm
            cv2.fillPoly(image, [points], (245, 245, 245))

        # Printed card marks become holes in the white mask, while RETR_EXTERNAL
        # would damage a white-only contour. Clip the print to the physical piece,
        # exactly as a real card pattern is clipped by the cut edge.
        piece_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        fourth_piece = np.asarray(polygons_mm[3], dtype=np.int32) * pixels_per_mm
        cv2.fillPoly(piece_mask, [fourth_piece], 255)
        print_mask = np.zeros_like(piece_mask)
        cv2.circle(print_mask, (640, 440), 14, 255, thickness=cv2.FILLED)
        image[(piece_mask != 0) & (print_mask != 0)] = (20, 20, 170)
        calibration = Calibration.from_dict(
            {
                "mm_per_pixel": 1.0 / pixels_per_mm,
                "roi_polygon_px": [[0, 0], [839, 0], [839, 591], [0, 591]],
            },
            image.shape,
        )
        detection = detect_pieces(
            image,
            calibration,
            VisionConfig(morphology_kernel_px=3, min_piece_area_mm2=100.0),
        )

        self.assertEqual(len(detection.pieces), 4)
        self.assertTrue(all(3 <= len(piece.polygon_mm) <= 5 for piece in detection.pieces))
        self.assertTrue(all(piece.pickup_clearance_mm > 2.0 for piece in detection.pieces))
        self.assertTrue(
            all(
                cv2.pointPolygonTest(
                    piece.polygon_px.astype(np.float32), piece.pickup_point_px, False
                )
                >= 0
                for piece in detection.pieces
            )
        )

        solved = solve_puzzle(
            [piece.polygon_mm for piece in detection.pieces],
            config=SolverConfig(
                dimension_area_tolerance=0.06,
                max_hole_ratio=0.04,
                max_overlap_ratio=0.003,
            ),
        )
        self.assertAlmostEqual(solved["rectangle"]["width_mm"], 100.0, delta=1.0)
        self.assertAlmostEqual(solved["rectangle"]["height_mm"], 60.0, delta=1.0)
        self.assertLess(solved["metrics"]["hole_ratio"], 0.02)


if __name__ == "__main__":
    unittest.main()
