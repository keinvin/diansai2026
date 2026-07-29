import importlib.util
import unittest

import numpy as np


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
class VisionTest(unittest.TestCase):
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
