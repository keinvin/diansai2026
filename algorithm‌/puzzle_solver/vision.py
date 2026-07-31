from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .solver import SolverConfig, solve_puzzle


CAMERA_INTRINSICS_PATH = Path(__file__).resolve().parents[2] / "data" / "camera_intrinsics.json"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Lens parameters produced by ``test/camera_calibration.py``."""

    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    optimal_new_camera_matrix: np.ndarray


@lru_cache(maxsize=2)
def load_camera_intrinsics(path: str | Path = CAMERA_INTRINSICS_PATH) -> CameraIntrinsics | None:
    """Load lens calibration, returning ``None`` when no calibration exists."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        size = document["image_size_px"]
        image_size = (int(size["width"]), int(size["height"]))
        camera_matrix = np.asarray(document["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(document["distortion_coefficients"], dtype=np.float64)
        new_matrix = np.asarray(document.get("optimal_new_camera_matrix", camera_matrix), dtype=np.float64)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid camera intrinsics file: {path}") from error
    if camera_matrix.shape != (3, 3) or new_matrix.shape != (3, 3) or distortion.size < 4:
        raise ValueError(f"Invalid camera intrinsics matrix shape: {path}")
    return CameraIntrinsics(image_size, camera_matrix, distortion.reshape(-1, 1), new_matrix)


def undistort_image(image_bgr: np.ndarray, intrinsics: CameraIntrinsics | None = None) -> np.ndarray:
    """Return an undistorted image when calibration matches its resolution."""
    intrinsics = intrinsics if intrinsics is not None else load_camera_intrinsics()
    if intrinsics is None or image_bgr.shape[1::-1] != intrinsics.image_size:
        return image_bgr
    return cv2.undistort(
        image_bgr,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
        None,
        intrinsics.optimal_new_camera_matrix,
    )


def undistort_points(points_px: np.ndarray, intrinsics: CameraIntrinsics | None = None) -> np.ndarray:
    """Map raw pixel points into the coordinate system of ``undistort_image``."""
    intrinsics = intrinsics if intrinsics is not None else load_camera_intrinsics()
    points = np.asarray(points_px, dtype=np.float32).reshape(1, -1, 2)
    if intrinsics is None:
        return points[0].astype(float)
    corrected = cv2.undistortPoints(
        points,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
        P=intrinsics.optimal_new_camera_matrix,
    )
    return corrected.reshape(-1, 2).astype(float)


@dataclass
class VisionConfig:
    """Configuration for pieces on a saturated, coloured A4 background."""

    segmentation_mode: str = "adaptive_gray"
    adaptive_gray_min_threshold: float = 70.0
    adaptive_gray_paper_offset: float = 38.0
    background_distance_min: float = 28.0
    white_max_saturation: int = 95
    white_min_value: int = 125
    morphology_kernel_px: int = 3
    close_iterations: int = 1
    open_iterations: int = 1
    min_piece_area_mm2: float = 80.0
    max_piece_area_mm2: float = 12_000.0
    approx_epsilon_mm: float = 0.7
    max_approx_epsilon_mm: float = 3.0
    approx_epsilon_perimeter_ratio: float = 0.0
    use_convex_hull: bool = False
    min_detected_edge_mm: float = 0.0
    rounded_corner_max_chord_mm: float = 0.0
    collinear_vertex_tolerance_mm: float = 0.0
    straight_vertex_angle_threshold_deg: float = 180.0
    straight_vertex_max_adjacent_edge_mm: float = 0.0
    roi_border_margin_px: int = 0
    max_vertices: int = 5
    max_pieces: int = 4
    camera_width: int = 1920
    camera_height: int = 1080
    camera_warmup_frames: int = 20


@dataclass
class Calibration:
    pixel_to_mm_homography: np.ndarray
    roi_polygon_px: np.ndarray | None = None
    camera_intrinsics: CameraIntrinsics | None = None

    @classmethod
    def from_dict(cls, document: dict, image_shape: Sequence[int] | None = None):
        intrinsics = load_camera_intrinsics() if document.get("use_camera_intrinsics", True) else None
        if image_shape is not None and intrinsics is not None:
            if tuple(image_shape[1::-1]) != intrinsics.image_size:
                intrinsics = None

        if "pixel_to_mm_homography" in document:
            homography = np.asarray(document["pixel_to_mm_homography"], dtype=float)
        elif "a4_corners_px" in document:
            corners = np.asarray(document["a4_corners_px"], dtype=np.float32)
            indices = document.get("a4_corner_indices", list(range(len(corners))))
            indices = [int(index) for index in indices]
            if corners.ndim != 2 or corners.shape[1] != 2 or len(corners) not in (3, 4):
                raise ValueError("a4_corners_px must contain three or four [x, y] points")
            if len(indices) != len(corners) or len(set(indices)) != len(indices) or any(index not in range(4) for index in indices):
                raise ValueError("a4_corner_indices must identify unique A4 corners: 0=TL, 1=TR, 2=BR, 3=BL")
            if intrinsics is not None:
                corners = undistort_points(corners, intrinsics).astype(np.float32)
            width = float(document.get("a4_width_mm", 210.0))
            height = float(document.get("a4_height_mm", 297.0))
            world = np.asarray(
                [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                dtype=np.float32,
            )
            if len(corners) == 4:
                ordered = np.empty((4, 2), dtype=np.float32)
                ordered[indices] = corners
                homography = cv2.getPerspectiveTransform(ordered, world)
            else:
                if intrinsics is None:
                    raise ValueError("Three-point A4 calibration requires data/camera_intrinsics.json")
                object_points = np.column_stack((world[indices], np.zeros(3, dtype=np.float32)))
                ok, rotation, translation = cv2.solvePnP(
                    object_points, corners, intrinsics.optimal_new_camera_matrix,
                    np.zeros((5, 1), dtype=np.float64), flags=cv2.SOLVEPNP_SQPNP,
                )
                if not ok:
                    raise ValueError("Unable to solve A4 pose from three corners")
                projected, _ = cv2.projectPoints(
                    np.column_stack((world, np.zeros(4, dtype=np.float32))), rotation, translation,
                    intrinsics.optimal_new_camera_matrix, np.zeros((5, 1), dtype=np.float64),
                )
                homography = cv2.getPerspectiveTransform(projected.reshape(4, 2).astype(np.float32), world)
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
            if intrinsics is not None:
                roi_polygon = undistort_points(roi_polygon, intrinsics)
        elif (
            document.get("a4_region", "upper" if document.get("use_a4_upper_half", True) else "full")
            in ("upper", "lower")
        ) and (
            "a4_corners_px" in document or "pixel_to_mm_homography" in document
        ):
            width = float(document.get("a4_width_mm", 210.0))
            height = float(document.get("a4_height_mm", 297.0))
            region = document.get("a4_region", "upper" if document.get("use_a4_upper_half", True) else "full")
            y0, y1 = (0.0, height / 2.0) if region == "upper" else (height / 2.0, height)
            region_world = np.asarray(
                [[[0.0, y0], [width, y0], [width, y1], [0.0, y1]]],
                dtype=np.float32,
            )
            inverse = np.linalg.inv(homography)
            roi_polygon = cv2.perspectiveTransform(region_world, inverse)[0]
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
        return cls(homography, roi_polygon, intrinsics)

    def undistort_image(self, image_bgr: np.ndarray) -> np.ndarray:
        return undistort_image(image_bgr, self.camera_intrinsics)

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
class RejectedContour:
    """A segmented contour rejected by a geometric piece filter."""

    contour_px: np.ndarray
    reason: str


@dataclass
class DetectionResult:
    pieces: list[DetectedPiece]
    mask: np.ndarray
    roi_mask: np.ndarray
    rejected_contours: list[RejectedContour]


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
    if config.segmentation_mode == "adaptive_gray":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray[roi != 0]
        if not len(roi_pixels):
            raise ValueError("The configured ROI contains no image pixels")
        paper_level = float(np.median(roi_pixels))
        threshold = max(
            float(config.adaptive_gray_min_threshold),
            paper_level + float(config.adaptive_gray_paper_offset),
        )
        bright = gray > threshold
        # Printed colours can be dark in grayscale but are still unlike the
        # black paper. Preserve them so edge-touching artwork cannot cut a
        # false notch into the physical piece contour.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = np.median(lab[roi != 0], axis=0)
        colour_distance = np.linalg.norm(lab - background_lab, axis=2)
        mask = np.where(
            bright | (colour_distance >= config.background_distance_min), 255, 0
        ).astype(np.uint8)
    elif config.segmentation_mode == "background_difference":
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
            "segmentation_mode must be 'adaptive_gray', 'background_difference', or 'white'"
        )
    mask = cv2.bitwise_and(mask, roi)

    kernel_size = max(1, int(config.morphology_kernel_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Match the simulator detector: a small square kernel preserves straight
    # polygon corners better than an elliptical kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
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
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(
        float(config.approx_epsilon_mm),
        float(config.approx_epsilon_perimeter_ratio) * perimeter,
    )
    approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    # Keep temporary arc endpoints until rounded corners have been reconstructed.
    # A rounded rectangle can contribute one extra vertex at each of four corners.
    approximation_limit = config.max_vertices + (
        4 if config.rounded_corner_max_chord_mm > 0.0 else 0
    )
    while len(approximation) > approximation_limit and epsilon < config.max_approx_epsilon_mm:
        epsilon = min(config.max_approx_epsilon_mm, epsilon * 1.25)
        approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return approximation.astype(float), epsilon


def _vertex_removal_cost(polygon: np.ndarray, index: int) -> float:
    """Measure the boundary distortion caused by removing one vertex."""
    previous = polygon[(index - 1) % len(polygon)]
    current = polygon[index]
    following = polygon[(index + 1) % len(polygon)]
    old_length = np.linalg.norm(current - previous) + np.linalg.norm(following - current)
    new_length = np.linalg.norm(following - previous)
    return float(max(0.0, old_length - new_length))


def _line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    denominator = float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(denominator) <= 1e-9:
        return None
    offset = second_point - first_point
    scale = float(
        (offset[0] * second_direction[1] - offset[1] * second_direction[0])
        / denominator
    )
    return first_point + scale * first_direction


def _merge_rounded_corner_vertices(
    polygon: np.ndarray, max_chord_mm: float
) -> np.ndarray:
    """Replace two vertices on a small rounded-corner chord by the line intersection."""

    result = np.asarray(polygon, dtype=float)
    if max_chord_mm <= 0.0:
        return result
    while len(result) > 3:
        edge_lengths = np.linalg.norm(np.roll(result, -1, axis=0) - result, axis=1)
        merged = False
        for edge_index in np.argsort(edge_lengths):
            if edge_lengths[edge_index] > max_chord_mm:
                break
            rotated = np.roll(result, -int(edge_index), axis=0)
            first, second = rotated[0], rotated[1]
            previous, following = rotated[-1], rotated[2]
            incoming = first - previous
            outgoing = following - second
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if min(incoming_length, outgoing_length) <= max_chord_mm:
                continue

            chord = second - first
            turn_first = float(incoming[0] * chord[1] - incoming[1] * chord[0])
            turn_second = float(chord[0] * outgoing[1] - chord[1] * outgoing[0])
            # A rounded corner bends consistently in one direction. Opposite signs
            # indicate a small notch or a real zig-zag and must not be collapsed.
            if turn_first * turn_second <= 0.0:
                continue
            intersection = _line_intersection(first, incoming, second, outgoing)
            if intersection is None:
                continue
            maximum_offset = max(
                float(np.linalg.norm(intersection - first)),
                float(np.linalg.norm(intersection - second)),
            )
            if maximum_offset > max_chord_mm * 2.0:
                continue
            result = np.vstack((intersection, rotated[2:]))
            merged = True
            break
        if not merged:
            break
    return result


def _merge_artifact_edges(polygon: np.ndarray, config: VisionConfig) -> np.ndarray:
    """Remove only enough low-impact vertices to satisfy the five-edge limit.

    A 10 mm edge can be a valid part of the original outer contour, while newly
    introduced cut edges are at least 20 mm. Vision cannot reliably distinguish
    those two origins, so edge length alone must not remove a vertex.
    """
    result = np.asarray(polygon, dtype=float)
    while len(result) > 3:
        edges = np.roll(result, -1, axis=0) - result
        lengths = np.linalg.norm(edges, axis=1)
        shortest = int(np.argmin(lengths))
        if len(result) <= config.max_vertices:
            break
        first = shortest
        second = (shortest + 1) % len(result)
        remove = min((first, second), key=lambda index: _vertex_removal_cost(result, index))
        result = np.delete(result, remove, axis=0)
    return result


def _remove_near_collinear_vertices(polygon: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Collapse small bends on a physically straight edge without using edge length."""
    result = np.asarray(polygon, dtype=float)
    if tolerance_mm <= 0:
        return result
    while len(result) > 3:
        distances: list[float] = []
        for index in range(len(result)):
            previous = result[(index - 1) % len(result)]
            current = result[index]
            following = result[(index + 1) % len(result)]
            chord = following - previous
            chord_length = float(np.linalg.norm(chord))
            if chord_length <= 1e-9:
                distances.append(0.0)
                continue
            cross = chord[0] * (current - previous)[1] - chord[1] * (current - previous)[0]
            distances.append(abs(float(cross)) / chord_length)
        remove = int(np.argmin(distances))
        if distances[remove] > tolerance_mm:
            break
        result = np.delete(result, remove, axis=0)
    return result


def _remove_nearly_straight_vertices(
    polygon: np.ndarray,
    angle_threshold_deg: float,
    max_adjacent_edge_mm: float,
) -> np.ndarray:
    """Remove nearly straight vertices only when an adjacent segment is short."""

    result = np.asarray(polygon, dtype=float)
    if not 0.0 < angle_threshold_deg < 180.0 or max_adjacent_edge_mm <= 0.0:
        return result
    while len(result) > 3:
        angles: list[float] = []
        eligible: list[bool] = []
        for index in range(len(result)):
            previous_vector = result[(index - 1) % len(result)] - result[index]
            following_vector = result[(index + 1) % len(result)] - result[index]
            shortest_adjacent_edge = min(
                float(np.linalg.norm(previous_vector)),
                float(np.linalg.norm(following_vector)),
            )
            denominator = float(
                np.linalg.norm(previous_vector) * np.linalg.norm(following_vector)
            )
            if denominator <= 1e-9:
                angles.append(180.0)
                eligible.append(True)
                continue
            cosine = float(
                np.clip(
                    np.dot(previous_vector, following_vector) / denominator,
                    -1.0,
                    1.0,
                )
            )
            angles.append(math.degrees(math.acos(cosine)))
            eligible.append(shortest_adjacent_edge <= max_adjacent_edge_mm)
        candidates = [
            index
            for index, angle in enumerate(angles)
            if eligible[index] and angle > angle_threshold_deg
        ]
        if not candidates:
            break
        remove = max(candidates, key=lambda index: angles[index])
        result = np.delete(result, remove, axis=0)
    return result


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
    image_bgr = calibration.undistort_image(image_bgr)
    mask, roi = segment_white_pieces(image_bgr, calibration, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    border_margin = max(0, int(config.roi_border_margin_px))
    inner_roi = roi
    if border_margin:
        size = border_margin * 2 + 1
        inner_roi = cv2.erode(
            roi, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)), iterations=1
        )

    detected: list[DetectedPiece] = []
    rejected: list[RejectedContour] = []
    for contour in contours:
        if len(contour) < 3:
            rejected.append(RejectedContour(contour.reshape(-1, 2).astype(float), "轮廓点不足"))
            continue
        contour_px = contour.reshape(-1, 2).astype(float)
        contour_indices = np.round(contour_px).astype(int)
        contour_indices[:, 0] = np.clip(contour_indices[:, 0], 0, inner_roi.shape[1] - 1)
        contour_indices[:, 1] = np.clip(contour_indices[:, 1], 0, inner_roi.shape[0] - 1)
        if border_margin and np.any(inner_roi[contour_indices[:, 1], contour_indices[:, 0]] == 0):
            rejected.append(RejectedContour(contour_px, "接触 A4 识别区边界"))
            continue
        contour_mm = calibration.pixels_to_mm(contour_px)
        area_mm2 = abs(float(cv2.contourArea(contour_mm.astype(np.float32))))
        if not config.min_piece_area_mm2 <= area_mm2 <= config.max_piece_area_mm2:
            rejected.append(
                RejectedContour(
                    contour_px,
                    f"面积 {area_mm2:.0f} mm²（允许 {config.min_piece_area_mm2:.0f}–{config.max_piece_area_mm2:.0f}）",
                )
            )
            continue

        polygon_source_mm = contour_mm
        if config.use_convex_hull:
            polygon_source_mm = cv2.convexHull(
                contour_mm.astype(np.float32).reshape(-1, 1, 2)
            ).reshape(-1, 2)
        polygon_mm, epsilon = _approximate_polygon_mm(polygon_source_mm, config)
        polygon_mm = _merge_rounded_corner_vertices(
            polygon_mm, config.rounded_corner_max_chord_mm
        )
        polygon_mm = _merge_artifact_edges(polygon_mm, config)
        polygon_mm = _remove_near_collinear_vertices(
            polygon_mm, config.collinear_vertex_tolerance_mm
        )
        polygon_mm = _remove_nearly_straight_vertices(
            polygon_mm,
            config.straight_vertex_angle_threshold_deg,
            config.straight_vertex_max_adjacent_edge_mm,
        )
        if not 3 <= len(polygon_mm) <= config.max_vertices:
            rejected.append(RejectedContour(contour_px, f"顶点 {len(polygon_mm)}（最大 {config.max_vertices}）"))
            continue
        polygon_px = calibration.mm_to_pixels(polygon_mm)
        edges = np.roll(polygon_mm, -1, axis=0) - polygon_mm
        edge_lengths = np.linalg.norm(edges, axis=1)
        if config.min_detected_edge_mm > 0 and float(edge_lengths.min()) < config.min_detected_edge_mm:
            rejected.append(
                RejectedContour(
                    contour_px,
                    f"最短边 {edge_lengths.min():.1f} mm（最小 {config.min_detected_edge_mm:.1f}）",
                )
            )
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

    # Competition prior: at most four pieces. Keep the four largest plausible
    # physical contours and demote smaller fragments/reflections to diagnostics.
    if len(detected) > config.max_pieces:
        detected.sort(key=lambda piece: piece.area_mm2, reverse=True)
        dropped = detected[config.max_pieces :]
        detected = detected[: config.max_pieces]
        rejected.extend(
            RejectedContour(piece.contour_px, f"Top {config.max_pieces} 之外的小轮廓")
            for piece in dropped
        )
    detected.sort(key=lambda piece: (piece.centroid_mm[1], piece.centroid_mm[0]))
    for index, piece in enumerate(detected):
        piece.id = f"piece_{index}"
    if not detected:
        raise RuntimeError("No pieces detected; check ROI, exposure, and white thresholds")
    return DetectionResult(detected, mask, roi, rejected)


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
        for vertex_index, vertex in enumerate(polygon):
            point = (int(vertex[0]), int(vertex[1]))
            cv2.circle(overlay, point, 6, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, point, 6, colour, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"V{vertex_index}",
                (point[0] + 8, point[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                2,
                cv2.LINE_AA,
            )
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
    for rejected in result.rejected_contours:
        contour = np.round(rejected.contour_px).astype(np.int32)
        if len(contour) < 3:
            continue
        colour = (40, 40, 235)
        cv2.polylines(overlay, [contour], True, colour, 2, cv2.LINE_AA)
        anchor = tuple(contour[np.argmin(contour[:, 1])])
        cv2.putText(
            overlay,
            rejected.reason,
            (int(anchor[0]), max(20, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
