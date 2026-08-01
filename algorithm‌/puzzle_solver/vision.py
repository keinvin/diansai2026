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
    split_touching_pieces: bool = True
    touching_split_area_loss_ratio: float = 0.12
    touching_split_max_erosion_mm: float = 12.0
    touching_split_min_seed_ratio: float = 0.08
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
    corner_roundness_mm: tuple[float, ...]

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
            "corner_roundness_mm": [
                round(value, 3) for value in self.corner_roundness_mm
            ],
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


def extract_edge_profiles(
    corrected_frame_bgr: np.ndarray,
    pieces: Sequence[DetectedPiece],
    sample_count: int = 64,
) -> list[list[np.ndarray]]:
    """Sample three LAB strips inside each edge without averaging away detail."""

    lab = cv2.cvtColor(corrected_frame_bgr, cv2.COLOR_BGR2LAB)
    profiles: list[list[np.ndarray]] = []
    positions = np.linspace(0.05, 0.95, sample_count, dtype=np.float32)
    inset_pixels = (2.0, 4.0, 7.0)
    for piece in pieces:
        polygon = np.asarray(piece.polygon_px, dtype=np.float32)
        centroid = polygon.mean(axis=0)
        piece_profiles: list[np.ndarray] = []
        for edge_index in range(len(polygon)):
            start = polygon[edge_index]
            end = polygon[(edge_index + 1) % len(polygon)]
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length <= 1e-6:
                piece_profiles.append(np.zeros((sample_count, 9), dtype=float))
                continue
            normal = np.array([-edge[1], edge[0]], dtype=np.float32) / length
            if float(np.dot(normal, centroid - (start + end) * 0.5)) < 0.0:
                normal *= -1.0
            base = start[None, :] + positions[:, None] * edge[None, :]
            maps = [base + normal[None, :] * inset for inset in inset_pixels]
            map_x = np.stack([points[:, 0] for points in maps]).astype(np.float32)
            map_y = np.stack([points[:, 1] for points in maps]).astype(np.float32)
            sampled = cv2.remap(
                lab,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            piece_profiles.append(
                sampled.transpose(1, 0, 2).reshape(sample_count, -1).astype(float)
            )
        profiles.append(piece_profiles)
    return profiles


def extract_piece_features(
    corrected_frame_bgr: np.ndarray,
    pieces: Sequence[DetectedPiece],
    calibration: Calibration | None = None,
    corner_radius_mm: float = 15.0,
) -> list[dict]:
    """Return rounded-corner and red/black corner-mark evidence per piece."""

    hsv = cv2.cvtColor(corrected_frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corrected_frame_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    red = ((hue <= 12) | (hue >= 168)) & (saturation >= 70) & (gray >= 35)
    dark = gray <= 130
    result: list[dict] = []
    for piece in pieces:
        contour = np.rint(piece.contour_px).astype(np.int32)
        x, y, width, height = cv2.boundingRect(contour)
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1 = min(gray.shape[1], x + width + 2)
        y1 = min(gray.shape[0], y + height + 2)
        local_shape = (y1 - y0, x1 - x0)
        piece_mask = np.zeros(local_shape, dtype=np.uint8)
        cv2.fillPoly(piece_mask, [contour - [x0, y0]], 255)
        piece_mask = cv2.erode(
            piece_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        local_red = red[y0:y1, x0:x1]
        local_dark = dark[y0:y1, x0:x1]
        polygon_px = np.asarray(piece.polygon_px, dtype=float)
        polygon_mm = np.asarray(piece.polygon_mm, dtype=float)
        px_lengths = np.linalg.norm(np.roll(polygon_px, -1, axis=0) - polygon_px, axis=1)
        mm_lengths = np.linalg.norm(np.roll(polygon_mm, -1, axis=0) - polygon_mm, axis=1)
        valid = mm_lengths > 1e-6
        pixels_per_mm = float(np.median(px_lengths[valid] / mm_lengths[valid]))
        radius_px = max(4, int(round(corner_radius_mm * pixels_per_mm)))
        ink_density: list[float] = []
        red_density: list[float] = []
        black_density: list[float] = []
        for vertex in polygon_px:
            corner_mask = np.zeros(local_shape, dtype=np.uint8)
            cv2.circle(
                corner_mask,
                tuple(np.rint(vertex - [x0, y0]).astype(int)),
                radius_px,
                255,
                thickness=cv2.FILLED,
            )
            region = (piece_mask != 0) & (corner_mask != 0)
            count = int(np.count_nonzero(region))
            if count == 0:
                ink_density.append(0.0)
                red_density.append(0.0)
                black_density.append(0.0)
                continue
            red_value = float(np.count_nonzero(region & local_red) / count)
            black_value = float(
                np.count_nonzero(region & local_dark & ~local_red) / count
            )
            red_density.append(red_value)
            black_density.append(black_value)
            ink_density.append(red_value + black_value)
        result.append(
            {
                "corner_roundness_mm": list(piece.corner_roundness_mm),
                "corner_ink_density": ink_density,
                "corner_red_density": red_density,
                "corner_black_density": black_density,
                "ink_points_mm": [],
                "ink_point_colours": [],
            }
        )
        if calibration is not None:
            local_ink = (local_red | local_dark) & (piece_mask != 0)
            rows, columns = np.nonzero(local_ink)
            if len(rows):
                maximum_points = 900
                if len(rows) > maximum_points:
                    selected = np.linspace(
                        0, len(rows) - 1, maximum_points, dtype=int
                    )
                    rows = rows[selected]
                    columns = columns[selected]
                points_px = np.column_stack((columns + x0, rows + y0)).astype(float)
                points_mm = calibration.pixels_to_mm(points_px)
                point_red = local_red[rows, columns]
                result[-1]["ink_points_mm"] = points_mm.tolist()
                result[-1]["ink_point_colours"] = point_red.astype(np.uint8).tolist()
    return result


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


def _clean_polygon_candidate(
    approximation: np.ndarray, config: VisionConfig, force_vertex_limit: bool = False
) -> np.ndarray:
    polygon = _merge_rounded_corner_vertices(
        approximation, config.rounded_corner_max_chord_mm
    )
    if force_vertex_limit:
        polygon = _merge_artifact_edges(polygon, config)
    polygon = _remove_near_collinear_vertices(
        polygon, config.collinear_vertex_tolerance_mm
    )
    return _remove_nearly_straight_vertices(
        polygon,
        config.straight_vertex_angle_threshold_deg,
        config.straight_vertex_max_adjacent_edge_mm,
    )


def _fit_piece_polygon(
    contour_mm: np.ndarray, config: VisionConfig
) -> tuple[np.ndarray, float]:
    """Choose a 3-5 vertex approximation without sacrificing real concavities."""

    contour = np.asarray(contour_mm, dtype=np.float32).reshape(-1, 1, 2)
    contour_area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    initial_epsilon = max(
        float(config.approx_epsilon_mm),
        float(config.approx_epsilon_perimeter_ratio) * perimeter,
    )
    maximum_epsilon = max(initial_epsilon, float(config.max_approx_epsilon_mm))

    epsilons = [initial_epsilon]
    while epsilons[-1] < maximum_epsilon - 1e-9:
        epsilons.append(min(maximum_epsilon, epsilons[-1] * 1.20))

    natural: list[tuple[float, float, np.ndarray]] = []
    forced: list[tuple[float, float, np.ndarray]] = []
    for epsilon in epsilons:
        approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
        cleaned = _clean_polygon_candidate(approximation, config)
        if 3 <= len(cleaned) <= config.max_vertices:
            area_error = abs(abs(float(cv2.contourArea(cleaned.astype(np.float32)))) - contour_area)
            relative_error = area_error / max(contour_area, 1e-9)
            natural.append((relative_error, epsilon, cleaned))
            continue

        if len(cleaned) > config.max_vertices:
            reduced = _clean_polygon_candidate(cleaned, config, force_vertex_limit=True)
            if 3 <= len(reduced) <= config.max_vertices:
                area_error = abs(abs(float(cv2.contourArea(reduced.astype(np.float32)))) - contour_area)
                relative_error = area_error / max(contour_area, 1e-9)
                forced.append((relative_error, epsilon, reduced))

    # A naturally simplified contour is safer unless it loses substantial area.
    # The fallback handles noisy/rounded contours that never reach five vertices
    # within the configured epsilon range.
    pool = natural
    if forced and (not natural or min(item[0] for item in natural) > 0.08):
        pool = natural + forced
    if not pool:
        approximation, epsilon = _approximate_polygon_mm(contour_mm, config)
        return _clean_polygon_candidate(approximation, config, force_vertex_limit=True), epsilon

    _, epsilon, polygon = min(pool, key=lambda item: (item[0], item[1]))
    return polygon, epsilon


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


def _split_touching_contour(
    contour_px: np.ndarray,
    image_shape: Sequence[int],
    calibration: Calibration,
    config: VisionConfig,
) -> list[np.ndarray]:
    """Split a narrow connection only when polygon fitting loses substantial area."""

    if not config.split_touching_pieces or config.touching_split_max_erosion_mm <= 0.0:
        return [contour_px]

    contour_mm = calibration.pixels_to_mm(contour_px)
    fitted, _ = _fit_piece_polygon(contour_mm, config)
    contour_area = abs(float(cv2.contourArea(contour_mm.astype(np.float32))))
    fitted_area = abs(float(cv2.contourArea(fitted.astype(np.float32))))
    area_mismatch = abs(contour_area - fitted_area) / max(contour_area, 1e-9)
    if area_mismatch < config.touching_split_area_loss_ratio:
        return [contour_px]

    x, y, width, height = cv2.boundingRect(contour_px.astype(np.float32))
    padding = 2
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image_shape[1], x + width + padding)
    y1 = min(image_shape[0], y + height + padding)
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = np.rint(contour_px - [x0, y0]).astype(np.int32)
    cv2.fillPoly(local, [shifted], 255)
    original_pixels = int(np.count_nonzero(local))

    centre = np.mean(contour_px, axis=0)
    nearby_mm = calibration.pixels_to_mm(
        np.asarray([centre, centre + [1.0, 0.0], centre + [0.0, 1.0]])
    )
    mm_per_pixel = 0.5 * (
        np.linalg.norm(nearby_mm[1] - nearby_mm[0])
        + np.linalg.norm(nearby_mm[2] - nearby_mm[0])
    )
    maximum_radius_px = max(
        1, int(math.ceil(config.touching_split_max_erosion_mm / max(mm_per_pixel, 1e-9)))
    )

    for radius in range(1, maximum_radius_px + 1):
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(local, kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        minimum_seed_pixels = original_pixels * config.touching_split_min_seed_ratio
        seed_labels = [
            label
            for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= minimum_seed_pixels
        ]
        if not 2 <= len(seed_labels) <= config.max_pieces:
            continue

        distances = []
        for label in seed_labels:
            seed = np.where(labels == label, 0, 255).astype(np.uint8)
            distances.append(cv2.distanceTransform(seed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        ownership = np.argmin(np.stack(distances), axis=0)

        pieces: list[np.ndarray] = []
        valid = True
        for owner in range(len(seed_labels)):
            component = np.where((local != 0) & (ownership == owner), 255, 0).astype(np.uint8)
            component_contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not component_contours:
                valid = False
                break
            piece = max(component_contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            piece += [x0, y0]
            piece_mm = calibration.pixels_to_mm(piece)
            piece_area = abs(float(cv2.contourArea(piece_mm.astype(np.float32))))
            piece_polygon, _ = _fit_piece_polygon(piece_mm, config)
            piece_polygon_area = abs(
                float(cv2.contourArea(piece_polygon.astype(np.float32)))
            )
            piece_mismatch = abs(piece_area - piece_polygon_area) / max(piece_area, 1e-9)
            if (
                not config.min_piece_area_mm2 <= piece_area <= config.max_piece_area_mm2
                or not 3 <= len(piece_polygon) <= config.max_vertices
                or piece_mismatch >= area_mismatch
            ):
                valid = False
                break
            pieces.append(piece)
        if valid:
            return pieces

    return [contour_px]


def detect_pieces(
    image_bgr: np.ndarray,
    calibration: Calibration,
    config: VisionConfig | None = None,
) -> DetectionResult:
    config = config or VisionConfig()
    image_bgr = calibration.undistort_image(image_bgr)
    mask, roi = segment_white_pieces(image_bgr, calibration, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    split_contours: list[np.ndarray] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        split_contours.extend(
            _split_touching_contour(points, image_bgr.shape, calibration, config)
        )
    contours = [contour.reshape(-1, 1, 2) for contour in split_contours]
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
        polygon_mm, epsilon = _fit_piece_polygon(polygon_source_mm, config)
        if not 3 <= len(polygon_mm) <= config.max_vertices:
            rejected.append(RejectedContour(contour_px, f"顶点 {len(polygon_mm)}（最大 {config.max_vertices}）"))
            continue
        polygon_px = calibration.mm_to_pixels(polygon_mm)
        corner_roundness_mm = tuple(
            float(np.linalg.norm(contour_mm - vertex, axis=1).min())
            for vertex in polygon_mm
        )
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
                corner_roundness_mm=corner_roundness_mm,
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
    image_bgr: np.ndarray,
    result: DetectionResult,
    calibration: Calibration,
    *,
    show_rejected_contours: bool = True,
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
    if show_rejected_contours:
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


def render_assembled_image(
    corrected_image_bgr: np.ndarray,
    result: DetectionResult,
    solution: dict,
    calibration: Calibration,
    pixels_per_mm: float = 6.0,
    margin_mm: float = 4.0,
) -> np.ndarray:
    """Warp source-piece pixels into the solved rectangle for visual verification."""

    if corrected_image_bgr is None or corrected_image_bgr.ndim != 3:
        raise ValueError("corrected_image_bgr must be a colour image")
    rectangle = solution["rectangle"]
    origin = np.asarray(rectangle["origin_mm"], dtype=float)
    width = float(rectangle["width_mm"])
    height = float(rectangle["height_mm"])
    if width <= 0.0 or height <= 0.0 or pixels_per_mm <= 0.0:
        raise ValueError("Assembly dimensions and pixels_per_mm must be positive")

    canvas_width = max(1, int(round((width + 2.0 * margin_mm) * pixels_per_mm)))
    canvas_height = max(1, int(round((height + 2.0 * margin_mm) * pixels_per_mm)))
    canvas = np.full((canvas_height, canvas_width, 3), (32, 36, 43), dtype=np.uint8)
    target_to_canvas = np.asarray(
        [
            [pixels_per_mm, 0.0, (margin_mm - origin[0]) * pixels_per_mm],
            [0.0, pixels_per_mm, (margin_mm - origin[1]) * pixels_per_mm],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pieces_by_id = {piece.id: piece for piece in result.pieces}
    for solved_piece in solution["pieces"]:
        piece = pieces_by_id.get(solved_piece["id"])
        if piece is None:
            continue
        rotation = np.asarray(solved_piece["rotation_matrix"], dtype=float)
        translation = np.asarray(solved_piece["translation_mm"], dtype=float)
        rigid_transform = np.asarray(
            [
                [rotation[0, 0], rotation[0, 1], translation[0]],
                [rotation[1, 0], rotation[1, 1], translation[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        source_to_canvas = (
            target_to_canvas
            @ rigid_transform
            @ calibration.pixel_to_mm_homography
        )
        source_mask = np.zeros(corrected_image_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            source_mask,
            [np.rint(piece.contour_px).astype(np.int32)],
            255,
        )
        warped_image = cv2.warpPerspective(
            corrected_image_bgr,
            source_to_canvas,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
        )
        warped_mask = cv2.warpPerspective(
            source_mask,
            source_to_canvas,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
        )
        canvas[warped_mask != 0] = warped_image[warped_mask != 0]

    first = int(round(margin_mm * pixels_per_mm))
    last = (
        int(round((margin_mm + width) * pixels_per_mm)),
        int(round((margin_mm + height) * pixels_per_mm)),
    )
    cv2.rectangle(canvas, (first, first), last, (0, 190, 255), 2, cv2.LINE_AA)
    return canvas


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
    parser.add_argument(
        "--assembled-image", type=Path, help="Write the solved textured assembly"
    )
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
        corrected_image = calibration.undistort_image(image)
        solved = solve_puzzle(
            [piece.polygon_mm for piece in result.pieces],
            [piece.id for piece in result.pieces],
            target_origin,
            solver_config,
            edge_profiles=extract_edge_profiles(corrected_image, result.pieces),
            piece_features=extract_piece_features(
                corrected_image, result.pieces, calibration
            ),
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
        if arguments.assembled_image:
            arguments.assembled_image.parent.mkdir(parents=True, exist_ok=True)
            assembled = render_assembled_image(
                corrected_image, result, solved, calibration
            )
            if not cv2.imwrite(str(arguments.assembled_image), assembled):
                raise RuntimeError(
                    f"Cannot write assembled image: {arguments.assembled_image}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
