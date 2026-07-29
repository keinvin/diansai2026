from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .solver import SolverConfig, solve_puzzle


@dataclass
class VisionConfig:
    """Configuration for pieces on a saturated, coloured A4 background."""

    segmentation_mode: str = "background_difference"
    background_distance_min: float = 28.0
    white_max_saturation: int = 95
    white_min_value: int = 125
    morphology_kernel_px: int = 5
    close_iterations: int = 2
    open_iterations: int = 1
    min_piece_area_mm2: float = 80.0
    max_piece_area_mm2: float = 12_000.0
    approx_epsilon_mm: float = 0.7
    max_approx_epsilon_mm: float = 3.0
    min_detected_edge_mm: float = 8.0
    max_vertices: int = 5
    max_pieces: int = 4
    camera_width: int = 1920
    camera_height: int = 1080
    camera_warmup_frames: int = 20


@dataclass
class Calibration:
    pixel_to_mm_homography: np.ndarray
    roi_polygon_px: np.ndarray | None = None

    @classmethod
    def from_dict(cls, document: dict, image_shape: Sequence[int] | None = None):
        if "pixel_to_mm_homography" in document:
            homography = np.asarray(document["pixel_to_mm_homography"], dtype=float)
        elif "a4_corners_px" in document:
            corners = np.asarray(document["a4_corners_px"], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError("a4_corners_px must be [top-left, top-right, bottom-right, bottom-left]")
            width = float(document.get("a4_width_mm", 210.0))
            height = float(document.get("a4_height_mm", 297.0))
            world = np.asarray(
                [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(corners, world)
        elif "mm_per_pixel" in document:
            scale = float(document["mm_per_pixel"])
            homography = np.asarray(
                [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=float,
            )
        else:
            raise ValueError(
                "Calibration needs pixel_to_mm_homography, a4_corners_px, or mm_per_pixel"
            )

        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError("pixel_to_mm_homography must be a finite 3 x 3 matrix")
        if abs(float(np.linalg.det(homography))) < 1e-12:
            raise ValueError("pixel_to_mm_homography is singular")

        roi = document.get("roi_polygon_px")
        if roi is not None:
            roi_polygon = np.asarray(roi, dtype=float)
        elif document.get("use_a4_upper_half", True) and (
            "a4_corners_px" in document or "pixel_to_mm_homography" in document
        ):
            width = float(document.get("a4_width_mm", 210.0))
            height = float(document.get("a4_height_mm", 297.0))
            upper_world = np.asarray(
                [[[0.0, 0.0], [width, 0.0], [width, height / 2.0], [0.0, height / 2.0]]],
                dtype=np.float32,
            )
            inverse = np.linalg.inv(homography)
            roi_polygon = cv2.perspectiveTransform(upper_world, inverse)[0]
        elif image_shape is not None:
            rows, columns = image_shape[:2]
            roi_polygon = np.asarray(
                [[0.0, 0.0], [columns - 1.0, 0.0], [columns - 1.0, rows - 1.0], [0.0, rows - 1.0]]
            )
        else:
            roi_polygon = None

        if roi_polygon is not None and (
            roi_polygon.ndim != 2 or roi_polygon.shape[1] != 2 or len(roi_polygon) < 3
        ):
            raise ValueError("roi_polygon_px must be an N x 2 polygon")
        return cls(homography, roi_polygon)

    def pixels_to_mm(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(values, self.pixel_to_mm_homography)[0].astype(float)

    def mm_to_pixels(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        inverse = np.linalg.inv(self.pixel_to_mm_homography)
        return cv2.perspectiveTransform(values, inverse)[0].astype(float)


@dataclass
class DetectedPiece:
    id: str
    contour_px: np.ndarray
    polygon_px: np.ndarray
    polygon_mm: np.ndarray
    area_mm2: float
    centroid_mm: tuple[float, float]
    pickup_point_px: tuple[float, float]
    pickup_point_mm: tuple[float, float]
    pickup_clearance_mm: float
    min_edge_mm: float
    approximation_epsilon_mm: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "polygon_mm": np.round(self.polygon_mm, 3).tolist(),
            "polygon_px": np.round(self.polygon_px, 2).tolist(),
            "centroid_mm": [round(value, 3) for value in self.centroid_mm],
            "pickup_point_mm": [round(value, 3) for value in self.pickup_point_mm],
            "pickup_point_px": [round(value, 2) for value in self.pickup_point_px],
            "pickup_clearance_mm": round(self.pickup_clearance_mm, 3),
            "area_mm2": round(self.area_mm2, 3),
            "min_edge_mm": round(self.min_edge_mm, 3),
            "vertex_count": len(self.polygon_mm),
            "approximation_epsilon_mm": round(self.approximation_epsilon_mm, 3),
        }


@dataclass
class DetectionResult:
    pieces: list[DetectedPiece]
    mask: np.ndarray
    roi_mask: np.ndarray


def _roi_mask(image_shape: Sequence[int], calibration: Calibration) -> np.ndarray:
    rows, columns = image_shape[:2]
    mask = np.zeros((rows, columns), dtype=np.uint8)
    if calibration.roi_polygon_px is None:
        mask.fill(255)
    else:
        polygon = np.round(calibration.roi_polygon_px).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def segment_white_pieces(
    image_bgr: np.ndarray, calibration: Calibration, config: VisionConfig
) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a non-empty BGR colour image")

    roi = _roi_mask(image_bgr.shape, calibration)
    if config.segmentation_mode == "background_difference":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_pixels = lab[roi != 0]
        if not len(roi_pixels):
            raise ValueError("The configured ROI contains no image pixels")
        background_lab = np.median(roi_pixels, axis=0)
        colour_distance = np.linalg.norm(lab - background_lab, axis=2)
        mask = np.where(colour_distance >= config.background_distance_min, 255, 0).astype(
            np.uint8
        )
    elif config.segmentation_mode == "white":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.asarray([0, 0, config.white_min_value], dtype=np.uint8)
        upper = np.asarray([180, config.white_max_saturation, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        raise ValueError(
            "segmentation_mode must be 'background_difference' or 'white'"
        )
    mask = cv2.bitwise_and(mask, roi)

    kernel_size = max(1, int(config.morphology_kernel_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if config.close_iterations:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
        )
    if config.open_iterations:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=config.open_iterations
        )
    return mask, roi


def _approximate_polygon_mm(
    contour_mm: np.ndarray, config: VisionConfig
) -> tuple[np.ndarray, float]:
    contour = np.asarray(contour_mm, dtype=np.float32).reshape(-1, 1, 2)
    epsilon = config.approx_epsilon_mm
    approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(approximation) > config.max_vertices and epsilon < config.max_approx_epsilon_mm:
        epsilon = min(config.max_approx_epsilon_mm, epsilon * 1.25)
        approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return approximation.astype(float), epsilon


def _polygon_centroid(polygon: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(np.asarray(polygon, dtype=np.float32))
    if abs(moments["m00"]) < 1e-9:
        centre = np.mean(polygon, axis=0)
        return float(centre[0]), float(centre[1])
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _pickup_point(
    contour_px: np.ndarray,
    image_shape: Sequence[int],
    calibration: Calibration,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    piece_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(piece_mask, [contour_px.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    padded = cv2.copyMakeBorder(piece_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, clearance_px, _, maximum_location = cv2.minMaxLoc(distance)
    pickup_px = np.asarray(maximum_location, dtype=float) - 1.0
    pickup_mm = calibration.pixels_to_mm(pickup_px.reshape(1, 2))[0]

    nearby_px = np.asarray(
        [pickup_px, pickup_px + [1.0, 0.0], pickup_px + [0.0, 1.0]], dtype=float
    )
    nearby_mm = calibration.pixels_to_mm(nearby_px)
    local_scale = 0.5 * (
        np.linalg.norm(nearby_mm[1] - nearby_mm[0])
        + np.linalg.norm(nearby_mm[2] - nearby_mm[0])
    )
    clearance_mm = float(clearance_px * local_scale)
    return (
        (float(pickup_px[0]), float(pickup_px[1])),
        (float(pickup_mm[0]), float(pickup_mm[1])),
        clearance_mm,
    )


def detect_pieces(
    image_bgr: np.ndarray,
    calibration: Calibration,
    config: VisionConfig | None = None,
) -> DetectionResult:
    config = config or VisionConfig()
    mask, roi = segment_white_pieces(image_bgr, calibration, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    detected: list[DetectedPiece] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour_px = contour.reshape(-1, 2).astype(float)
        contour_mm = calibration.pixels_to_mm(contour_px)
        area_mm2 = abs(float(cv2.contourArea(contour_mm.astype(np.float32))))
        if not config.min_piece_area_mm2 <= area_mm2 <= config.max_piece_area_mm2:
            continue

        polygon_mm, epsilon = _approximate_polygon_mm(contour_mm, config)
        if not 3 <= len(polygon_mm) <= config.max_vertices:
            continue
        polygon_px = calibration.mm_to_pixels(polygon_mm)
        edges = np.roll(polygon_mm, -1, axis=0) - polygon_mm
        edge_lengths = np.linalg.norm(edges, axis=1)
        if float(edge_lengths.min()) < config.min_detected_edge_mm:
            continue
        centroid_mm = _polygon_centroid(polygon_mm)
        pickup_px, pickup_mm, clearance_mm = _pickup_point(
            contour_px, image_bgr.shape, calibration
        )

        detected.append(
            DetectedPiece(
                id="",
                contour_px=contour_px,
                polygon_px=polygon_px,
                polygon_mm=polygon_mm,
                area_mm2=area_mm2,
                centroid_mm=centroid_mm,
                pickup_point_px=pickup_px,
                pickup_point_mm=pickup_mm,
                pickup_clearance_mm=clearance_mm,
                min_edge_mm=float(edge_lengths.min()),
                approximation_epsilon_mm=epsilon,
            )
        )

    detected.sort(key=lambda piece: (piece.centroid_mm[1], piece.centroid_mm[0]))
    if len(detected) > config.max_pieces:
        raise RuntimeError(
            f"Detected {len(detected)} plausible pieces; expected no more than {config.max_pieces}. "
            "Tighten the ROI, colour threshold, or minimum area."
        )
    for index, piece in enumerate(detected):
        piece.id = f"piece_{index}"
    if not detected:
        raise RuntimeError("No pieces detected; check ROI, exposure, and white thresholds")
    return DetectionResult(detected, mask, roi)


def draw_detection_overlay(
    image_bgr: np.ndarray, result: DetectionResult, calibration: Calibration
) -> np.ndarray:
    overlay = image_bgr.copy()
    if calibration.roi_polygon_px is not None:
        roi = np.round(calibration.roi_polygon_px).astype(np.int32)
        cv2.polylines(overlay, [roi], True, (0, 210, 255), 2, cv2.LINE_AA)

    palette = [(255, 130, 30), (40, 190, 70), (40, 80, 235), (190, 70, 190)]
    for index, piece in enumerate(result.pieces):
        colour = palette[index % len(palette)]
        polygon = np.round(piece.polygon_px).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, colour, 3, cv2.LINE_AA)
        pickup = tuple(np.round(piece.pickup_point_px).astype(int))
        cv2.drawMarker(overlay, pickup, colour, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        label = f"{piece.id}: {len(polygon)}v  A={piece.area_mm2:.0f}mm2"
        anchor = tuple(polygon[np.argmin(polygon[:, 1])])
        cv2.putText(
            overlay,
            label,
            (int(anchor[0]), max(20, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    return overlay


def capture_camera(index: int, config: VisionConfig) -> np.ndarray:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened() and backend != cv2.CAP_ANY:
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        frame = None
        for _ in range(max(1, config.camera_warmup_frames)):
            ok, frame = camera.read()
            if not ok:
                frame = None
        if frame is None:
            raise RuntimeError(f"Camera {index} did not return an image")
        return frame
    finally:
        camera.release()


def _load_configuration(path: Path, image_shape: Sequence[int]):
    document = json.loads(path.read_text(encoding="utf-8"))
    calibration = Calibration.from_dict(document, image_shape)
    vision = VisionConfig(**document.get("vision", {}))
    solver_values = dict(document.get("solver", {}))
    for key in ("width_range", "height_range"):
        if key in solver_values:
            solver_values[key] = tuple(solver_values[key])
    solver = SolverConfig(**solver_values)
    target_origin = document.get("target_origin_mm", [55.0, 190.0])
    return document, calibration, vision, solver, target_origin


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect puzzle pieces with OpenCV and optionally solve their target poses"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image")
    source.add_argument("--camera", type=int, help="OpenCV camera index")
    parser.add_argument("--config", type=Path, required=True, help="Vision calibration JSON")
    parser.add_argument("--detections", type=Path, help="Write detected polygons as JSON")
    parser.add_argument("--solution", type=Path, help="Also solve and write target poses")
    parser.add_argument("--debug-image", type=Path, help="Write contour and pickup overlay")
    parser.add_argument("--mask-image", type=Path, help="Write segmentation mask")
    arguments = parser.parse_args(argv)

    preliminary = json.loads(arguments.config.read_text(encoding="utf-8"))
    preliminary_vision = VisionConfig(**preliminary.get("vision", {}))
    if arguments.image:
        image = cv2.imread(str(arguments.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {arguments.image}")
    else:
        image = capture_camera(arguments.camera, preliminary_vision)

    _, calibration, vision_config, solver_config, target_origin = _load_configuration(
        arguments.config, image.shape
    )
    result = detect_pieces(image, calibration, vision_config)
    detection_document = {
        "target_origin_mm": target_origin,
        "pieces": [piece.to_dict() for piece in result.pieces],
        "vision": asdict(vision_config),
    }
    detection_json = json.dumps(detection_document, ensure_ascii=False, indent=2)
    if arguments.detections:
        arguments.detections.write_text(detection_json + "\n", encoding="utf-8")
    elif not arguments.solution:
        print(detection_json)

    if arguments.debug_image:
        arguments.debug_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(arguments.debug_image), draw_detection_overlay(image, result, calibration)
        ):
            raise RuntimeError(f"Cannot write debug image: {arguments.debug_image}")
    if arguments.mask_image:
        arguments.mask_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(arguments.mask_image), result.mask):
            raise RuntimeError(f"Cannot write mask image: {arguments.mask_image}")

    if arguments.solution:
        solved = solve_puzzle(
            [piece.polygon_mm for piece in result.pieces],
            [piece.id for piece in result.pieces],
            target_origin,
            solver_config,
        )
        pickup_by_id = {piece.id: piece.pickup_point_mm for piece in result.pieces}
        for piece in solved["pieces"]:
            pickup = np.asarray(pickup_by_id[piece["id"]], dtype=float)
            rotation = np.asarray(piece["rotation_matrix"], dtype=float)
            translation = np.asarray(piece["translation_mm"], dtype=float)
            piece["pickup_source_mm"] = pickup.tolist()
            piece["pickup_target_mm"] = (rotation @ pickup + translation).tolist()
        arguments.solution.write_text(
            json.dumps(solved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
