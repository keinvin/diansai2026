from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

try:
    from ._native_search import (
        NativeSearchError,
        NativeSearchUnavailable,
        search_candidate_masks,
    )
except ImportError:
    # The optional C++ extension is not bundled on every deployment target.
    # Keep the algorithm usable through the Python DFS implementation below.
    class NativeSearchError(RuntimeError):
        pass

    class NativeSearchUnavailable(RuntimeError):
        pass

    def search_candidate_masks(*_args, **_kwargs):
        raise NativeSearchUnavailable("C++ puzzle search extension is unavailable")


ArrayLike = Sequence[Sequence[float]]


@dataclass
class SolverConfig:
    """All dimensions are in millimetres."""

    width_range: tuple[float, float] = (90.0, 120.0)
    height_range: tuple[float, float] = (50.0, 90.0)
    grid_mm: float = 1.0
    dimension_area_tolerance: float = 0.04
    inside_tolerance_mm: float = 0.75
    max_rectangle_candidates: int = 40
    preferred_aspect_ratio: float = 5.0 / 3.0
    aspect_ratio_score_weight: float = 0.10
    max_hole_ratio: float = 0.025
    max_overlap_ratio: float = 0.002
    max_search_nodes_per_rectangle: int = 300_000
    max_solutions_per_rectangle: int = 8
    pattern_max_solutions_per_rectangle: int = 64
    pattern_rectangle_candidates: int = 3
    max_solve_seconds: float = 0.0
    early_accept_score: float = 0.06
    placement_gap_mm: float = 1.5
    max_placement_gap_mm: float = 8.0
    adjacency_detection_tolerance_mm: float = 8.0
    max_adjacent_vertex_distance_mm: float = 20.0
    final_overlap_tolerance_mm2: float = 0.25
    pattern_score_weight: float = 0.65
    rounded_corner_score_weight: float = 0.20
    min_trusted_corner_roundness_mm: float = 1.20
    max_trusted_corner_distance_mm: float = 4.0
    min_trusted_corner_right_angle_weight: float = 0.55
    min_misplaced_trusted_corner_count: int = 2
    corner_mark_score_weight: float = 0.12
    card_symmetry_score_weight: float = 0.45
    min_card_symmetry_evidence: float = 0.70
    max_card_symmetry_mismatch: float = 0.60
    min_pattern_evidence: float = 0.12
    max_pattern_mismatch: float = 0.45
    require_connected_assembly: bool = True
    enable_relaxed_retry: bool = True
    relaxed_search_first: bool = True
    retry_dimension_area_tolerance: float = 0.06
    retry_inside_tolerance_mm: float = 4.0
    retry_max_hole_ratio: float = 0.06
    retry_max_overlap_ratio: float = 0.02
    retry_early_accept_score: float = 0.12
    use_native_search: bool = True
    native_search_required: bool = False


@dataclass
class PoseCandidate:
    piece_index: int
    rotation_rad: float
    translation: tuple[float, float]
    polygon: np.ndarray
    mask: int
    cell_count: int
    boundary_side: str
    source_edge: int


def polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _validated_polygons(polygons: Sequence[ArrayLike]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index, polygon in enumerate(polygons):
        array = np.asarray(polygon, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError(f"Piece {index} must be an N x 2 polygon with N >= 3")
        if not np.isfinite(array).all():
            raise ValueError(f"Piece {index} contains a non-finite coordinate")
        if polygon_area(array) <= 1e-6:
            raise ValueError(f"Piece {index} has zero area")
        result.append(array)
    if not 1 <= len(result) <= 4:
        raise ValueError("The competition constraint requires between 1 and 4 pieces")
    return result


def _edge_lengths(polygons: Sequence[np.ndarray]) -> list[float]:
    lengths: list[float] = []
    for polygon in polygons:
        vectors = np.roll(polygon, -1, axis=0) - polygon
        lengths.extend(float(value) for value in np.linalg.norm(vectors, axis=1))
    return lengths


def _reachable_dimension_units(
    lengths: Iterable[float], grid_mm: float, maximum_mm: float
) -> set[int]:
    maximum = int(round(maximum_mm / grid_mm))
    reachable_bits = 1
    valid_mask = (1 << (maximum + 1)) - 1
    for length in lengths:
        units = max(1, int(round(length / grid_mm)))
        reachable_bits |= reachable_bits << units
        reachable_bits &= valid_mask
    return {unit for unit in range(maximum + 1) if (reachable_bits >> unit) & 1}


def candidate_rectangles(
    polygons: Sequence[np.ndarray], config: SolverConfig
) -> list[tuple[float, float, float]]:
    """Return (width, height, relative area error), best candidate first."""

    total_area = sum(polygon_area(polygon) for polygon in polygons)
    lengths = _edge_lengths(polygons)
    reachable = _reachable_dimension_units(
        lengths, config.grid_mm, max(config.width_range[1], config.height_range[1])
    )

    min_w = math.ceil(config.width_range[0] / config.grid_mm)
    max_w = math.floor(config.width_range[1] / config.grid_mm)
    min_h = math.ceil(config.height_range[0] / config.grid_mm)
    max_h = math.floor(config.height_range[1] / config.grid_mm)
    width_units = [value for value in reachable if min_w <= value <= max_w]
    height_units = [value for value in reachable if min_h <= value <= max_h]

    candidates: list[tuple[float, float, float]] = []
    for width_unit in width_units:
        for height_unit in height_units:
            width = width_unit * config.grid_mm
            height = height_unit * config.grid_mm
            error = abs(width * height - total_area) / total_area
            if error <= config.dimension_area_tolerance:
                candidates.append((width, height, error))

    # Segmentation noise can make subset sums miss the real side length. Area scanning
    # keeps a bounded fallback while retaining the same finite pose search.
    if not candidates:
        for width_unit in range(min_w, max_w + 1):
            width = width_unit * config.grid_mm
            estimated_height = total_area / width
            centre = int(round(estimated_height / config.grid_mm))
            for height_unit in range(centre - 2, centre + 3):
                if min_h <= height_unit <= max_h:
                    height = height_unit * config.grid_mm
                    error = abs(width * height - total_area) / total_area
                    if error <= config.dimension_area_tolerance:
                        candidates.append((width, height, error))

    def candidate_priority(item: tuple[float, float, float]) -> tuple[float, float, float]:
        width, height, area_error = item
        aspect_penalty = 0.0
        if config.preferred_aspect_ratio > 0.0 and config.aspect_ratio_score_weight > 0.0:
            aspect_penalty = config.aspect_ratio_score_weight * abs(
                math.log((width / height) / config.preferred_aspect_ratio)
            )
        return area_error + aspect_penalty, area_error, width * 2.0 + height

    candidates.sort(key=candidate_priority)
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = (round(candidate[0] / config.grid_mm), round(candidate[1] / config.grid_mm))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique[: config.max_rectangle_candidates]


def _rigid_transform(
    polygon: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source_angle = math.atan2(*(source_b - source_a)[::-1])
    target_angle = math.atan2(*(target_b - target_a)[::-1])
    theta = target_angle - source_angle
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    translation = target_a - rotation @ source_a
    transformed = polygon @ rotation.T + translation
    return transformed, rotation, translation, theta


def _polygon_mask(polygon: np.ndarray, width: float, height: float, grid: float) -> int:
    columns = int(round(width / grid))
    rows = int(round(height / grid))
    x0 = max(0, int(math.floor(float(polygon[:, 0].min()) / grid)))
    x1 = min(columns, int(math.ceil(float(polygon[:, 0].max()) / grid)))
    y0 = max(0, int(math.floor(float(polygon[:, 1].min()) / grid)))
    y1 = min(rows, int(math.ceil(float(polygon[:, 1].max()) / grid)))
    if x0 >= x1 or y0 >= y1:
        return 0

    xs = (np.arange(x0, x1, dtype=float) + 0.5) * grid
    ys = (np.arange(y0, y1, dtype=float) + 0.5) * grid
    x_grid, y_grid = np.meshgrid(xs, ys)
    inside = np.zeros(x_grid.shape, dtype=bool)

    previous = polygon[-1]
    for current in polygon:
        x_i, y_i = current
        x_j, y_j = previous
        crosses = (y_i > y_grid) != (y_j > y_grid)
        denominator = y_j - y_i
        if abs(denominator) < 1e-12:
            previous = current
            continue
        intersection_x = (x_j - x_i) * (y_grid - y_i) / denominator + x_i
        inside ^= crosses & (x_grid < intersection_x)
        previous = current

    full = np.zeros(rows * columns, dtype=np.uint8)
    local_rows, local_columns = np.nonzero(inside)
    global_indices = (local_rows + y0) * columns + (local_columns + x0)
    full[global_indices] = 1
    packed = np.packbits(full, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


def _side_segments(side: str, offset: float, length: float, width: float, height: float):
    if side == "top":
        return np.array([offset, 0.0]), np.array([offset + length, 0.0])
    if side == "bottom":
        return np.array([offset, height]), np.array([offset + length, height])
    if side == "left":
        return np.array([0.0, offset]), np.array([0.0, offset + length])
    if side == "right":
        return np.array([width, offset]), np.array([width, offset + length])
    raise ValueError(f"Unknown rectangle side: {side}")


def generate_piece_poses(
    piece_index: int,
    polygon: np.ndarray,
    width: float,
    height: float,
    config: SolverConfig,
) -> list[PoseCandidate]:
    poses: list[PoseCandidate] = []
    seen: set[tuple[int, int, int]] = set()
    tolerance = config.inside_tolerance_mm

    for edge_index in range(len(polygon)):
        source_a = polygon[edge_index]
        source_b = polygon[(edge_index + 1) % len(polygon)]
        edge_length = float(np.linalg.norm(source_b - source_a))
        if edge_length <= 1e-9:
            continue

        for side in ("top", "bottom", "left", "right"):
            side_length = width if side in ("top", "bottom") else height
            if edge_length > side_length + tolerance:
                continue
            maximum_offset = max(0.0, side_length - edge_length)
            offset_units = int(math.floor(maximum_offset / config.grid_mm + 1e-9))
            offsets = [unit * config.grid_mm for unit in range(offset_units + 1)]
            if maximum_offset - offsets[-1] > 1e-6:
                offsets.append(maximum_offset)

            for offset in offsets:
                target_a, target_b = _side_segments(
                    side, offset, edge_length, width, height
                )
                for reverse in (False, True):
                    first, second = (target_b, target_a) if reverse else (target_a, target_b)
                    transformed, _, translation, theta = _rigid_transform(
                        polygon, source_a, source_b, first, second
                    )
                    if (
                        transformed[:, 0].min() < -tolerance
                        or transformed[:, 0].max() > width + tolerance
                        or transformed[:, 1].min() < -tolerance
                        or transformed[:, 1].max() > height + tolerance
                    ):
                        continue

                    normalized_theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
                    key = (
                        round(normalized_theta * 10_000),
                        round(float(translation[0]) / config.grid_mm * 10),
                        round(float(translation[1]) / config.grid_mm * 10),
                    )
                    if key in seen:
                        continue
                    mask = _polygon_mask(transformed, width, height, config.grid_mm)
                    if not mask:
                        continue
                    seen.add(key)
                    poses.append(
                        PoseCandidate(
                            piece_index=piece_index,
                            rotation_rad=normalized_theta,
                            translation=(float(translation[0]), float(translation[1])),
                            polygon=transformed,
                            mask=mask,
                            cell_count=mask.bit_count(),
                            boundary_side=side,
                            source_edge=edge_index,
                        )
                    )

    return poses


def _search_rectangle(
    polygons: Sequence[np.ndarray],
    width: float,
    height: float,
    dimension_error: float,
    config: SolverConfig,
    deadline: float | None = None,
) -> tuple[list[tuple[float, list[PoseCandidate], dict]], int, bool]:
    def estimated_pose_count(polygon: np.ndarray) -> int:
        estimate = 0
        for edge_index in range(len(polygon)):
            length = float(
                np.linalg.norm(polygon[(edge_index + 1) % len(polygon)] - polygon[edge_index])
            )
            for side_length in (width, width, height, height):
                if length <= side_length + config.inside_tolerance_mm:
                    estimate += 2 * (
                        int(max(0.0, side_length - length) / config.grid_mm) + 1
                    )
        return estimate

    candidates: list[list[PoseCandidate]] = [[] for _ in polygons]
    generation_order = sorted(
        range(len(polygons)), key=lambda index: estimated_pose_count(polygons[index])
    )
    for index in generation_order:
        candidates[index] = generate_piece_poses(index, polygons[index], width, height, config)
        if not candidates[index]:
            return [], 0, False

    rectangle_cells = int(round(width / config.grid_mm)) * int(
        round(height / config.grid_mm)
    )
    allowed_holes = math.ceil(rectangle_cells * config.max_hole_ratio)
    allowed_overlap = math.ceil(rectangle_cells * config.max_overlap_ratio)

    if config.use_native_search:
        remaining_seconds = 0.0
        if deadline is not None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.0:
                return [], 0, True
        try:
            native_result = search_candidate_masks(
                [[pose.mask for pose in piece] for piece in candidates],
                rectangle_cells=rectangle_cells,
                allowed_holes=allowed_holes,
                allowed_overlap=allowed_overlap,
                max_nodes=config.max_search_nodes_per_rectangle,
                max_seconds=remaining_seconds,
                max_solutions=config.max_solutions_per_rectangle,
            )
        except (NativeSearchUnavailable, NativeSearchError, OverflowError, ValueError) as error:
            if config.native_search_required:
                raise RuntimeError("Native puzzle search is required but unavailable") from error
        else:
            solutions: list[tuple[float, list[PoseCandidate], dict]] = []
            for native_solution in native_result.solutions:
                selected = [
                    candidates[piece_index][candidate_index]
                    for piece_index, candidate_index in enumerate(
                        native_solution.candidate_indices
                    )
                ]
                hole_ratio = native_solution.hole_cells / rectangle_cells
                overlap_ratio = native_solution.overlap_cells / rectangle_cells
                score = dimension_error + hole_ratio + 5.0 * overlap_ratio
                solutions.append(
                    (
                        score,
                        selected,
                        {
                            "dimension_area_error": dimension_error,
                            "hole_ratio": hole_ratio,
                            "overlap_ratio": overlap_ratio,
                            "search_nodes": native_result.nodes,
                            "search_backend": "cpp",
                        },
                    )
                )
            return solutions, native_result.nodes, native_result.timed_out

    chosen: list[PoseCandidate | None] = [None] * len(polygons)
    solutions: list[tuple[float, list[PoseCandidate], dict]] = []
    nodes = 0
    timed_out = False

    def dfs(remaining: tuple[int, ...], occupied: int, overlap_cells: int) -> None:
        nonlocal nodes, timed_out
        if nodes >= config.max_search_nodes_per_rectangle:
            return
        if deadline is not None and nodes % 256 == 0 and time.monotonic() >= deadline:
            timed_out = True
            return
        nodes += 1

        if not remaining:
            hole_cells = rectangle_cells - occupied.bit_count()
            if hole_cells > allowed_holes or overlap_cells > allowed_overlap:
                return
            hole_ratio = hole_cells / rectangle_cells
            overlap_ratio = overlap_cells / rectangle_cells
            score = dimension_error + hole_ratio + 5.0 * overlap_ratio
            selected = [pose for pose in chosen if pose is not None]
            metrics = {
                "dimension_area_error": dimension_error,
                "hole_ratio": hole_ratio,
                "overlap_ratio": overlap_ratio,
                "search_nodes": nodes,
                "search_backend": "python",
            }
            solutions.append((score, selected, metrics))
            solutions.sort(key=lambda item: item[0])
            del solutions[config.max_solutions_per_rectangle :]
            return

        viable_by_piece: dict[int, list[tuple[PoseCandidate, int]]] = {}
        possible_coverage = occupied
        maximum_new_cells = 0
        for piece_index in remaining:
            viable: list[tuple[PoseCandidate, int]] = []
            piece_union = 0
            piece_maximum = 0
            for pose in candidates[piece_index]:
                overlap = (occupied & pose.mask).bit_count()
                if overlap_cells + overlap > allowed_overlap:
                    continue
                viable.append((pose, overlap))
                piece_union |= pose.mask
                piece_maximum = max(piece_maximum, pose.cell_count - overlap)
            if not viable:
                return
            viable_by_piece[piece_index] = viable
            possible_coverage |= piece_union
            maximum_new_cells += piece_maximum

        if rectangle_cells - possible_coverage.bit_count() > allowed_holes:
            return
        if rectangle_cells - (occupied.bit_count() + maximum_new_cells) > allowed_holes:
            return

        piece_index = min(remaining, key=lambda index: len(viable_by_piece[index]))
        next_remaining = tuple(index for index in remaining if index != piece_index)
        viable_poses = sorted(
            viable_by_piece[piece_index],
            key=lambda item: (item[1], -item[0].cell_count),
        )
        for pose, overlap in viable_poses:
            if timed_out:
                return
            new_overlap = overlap_cells + overlap
            chosen[piece_index] = pose
            dfs(next_remaining, occupied | pose.mask, new_overlap)
            chosen[piece_index] = None

    dfs(tuple(range(len(polygons))), 0, 0)
    return solutions, nodes, timed_out


def _detect_edge_adjacencies(
    polygons: Sequence[np.ndarray],
    piece_ids: Sequence[str],
    config: SolverConfig,
) -> list[dict]:
    """Find paired target edges before the safety gap is applied.

    The raster solver does not explicitly connect edges.  Recovering the edge pairs
    here gives the caller concrete seams on which to perform image-pattern checks.
    """

    candidates: list[tuple[float, dict]] = []
    tolerance = min(
        config.adjacency_detection_tolerance_mm,
        config.max_adjacent_vertex_distance_mm,
    )
    for piece_a in range(len(polygons)):
        polygon_a = polygons[piece_a]
        for piece_b in range(piece_a + 1, len(polygons)):
            polygon_b = polygons[piece_b]
            for edge_a in range(len(polygon_a)):
                a0 = polygon_a[edge_a]
                a1 = polygon_a[(edge_a + 1) % len(polygon_a)]
                length_a = float(np.linalg.norm(a1 - a0))
                for edge_b in range(len(polygon_b)):
                    b0 = polygon_b[edge_b]
                    b1 = polygon_b[(edge_b + 1) % len(polygon_b)]
                    length_b = float(np.linalg.norm(b1 - b0))
                    if max(length_a, length_b) <= 1e-9:
                        continue
                    same = (float(np.linalg.norm(a0 - b0)), float(np.linalg.norm(a1 - b1)))
                    reverse = (
                        float(np.linalg.norm(a0 - b1)),
                        float(np.linalg.norm(a1 - b0)),
                    )
                    is_reversed = max(reverse) < max(same)
                    distances = reverse if is_reversed else same
                    maximum_distance = max(distances)
                    length_error = abs(length_a - length_b) / max(length_a, length_b)
                    if length_error <= 0.20 and maximum_distance <= tolerance:
                        interval_a = [0.0, 1.0]
                        interval_b = [1.0, 0.0] if is_reversed else [0.0, 1.0]
                        score = maximum_distance + 0.25 * sum(distances) + 5.0 * length_error
                    else:
                        direction_a = (a1 - a0) / length_a
                        direction_b = (b1 - b0) / length_b
                        parallel_error = abs(
                            float(
                                direction_a[0] * direction_b[1]
                                - direction_a[1] * direction_b[0]
                            )
                        )
                        if parallel_error > math.sin(math.radians(12.0)):
                            continue
                        normal_a = np.array([-direction_a[1], direction_a[0]])
                        line_error = max(
                            abs(float(np.dot(b0 - a0, normal_a))),
                            abs(float(np.dot(b1 - a0, normal_a))),
                        )
                        if line_error > tolerance:
                            continue
                        projected_b = [
                            float(np.dot(b0 - a0, direction_a)),
                            float(np.dot(b1 - a0, direction_a)),
                        ]
                        overlap_start = max(0.0, min(projected_b))
                        overlap_end = min(length_a, max(projected_b))
                        overlap_length = overlap_end - overlap_start
                        if overlap_length < max(10.0, 0.35 * min(length_a, length_b)):
                            continue
                        point_start = a0 + direction_a * overlap_start
                        point_end = a0 + direction_a * overlap_end
                        interval_a = [overlap_start / length_a, overlap_end / length_a]
                        interval_b = [
                            float(np.dot(point_start - b0, direction_b) / length_b),
                            float(np.dot(point_end - b0, direction_b) / length_b),
                        ]
                        interval_b = [float(np.clip(value, 0.0, 1.0)) for value in interval_b]
                        is_reversed = interval_b[1] < interval_b[0]
                        score = line_error + 10.0 * parallel_error + 3.0
                    candidates.append(
                        (
                            score,
                            {
                                "piece_a": piece_a,
                                "piece_b": piece_b,
                                "piece_a_id": piece_ids[piece_a],
                                "piece_b_id": piece_ids[piece_b],
                                "edge_a": edge_a,
                                "edge_b": edge_b,
                                "edge_b_reversed": is_reversed,
                                "edge_a_interval": interval_a,
                                "edge_b_interval": interval_b,
                                "edge_length_a_mm": length_a,
                                "edge_length_b_mm": length_b,
                            },
                        )
                    )

    # Exact seams sort ahead of incidental pairs.  Interval occupancy still permits
    # one long edge to pair with two disjoint shorter edges at a T junction.
    used_intervals: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def interval_conflicts(key: tuple[int, int], interval: Sequence[float]) -> bool:
        low, high = sorted((float(interval[0]), float(interval[1])))
        length = max(high - low, 1e-9)
        for used_low, used_high in used_intervals.get(key, []):
            overlap = max(0.0, min(high, used_high) - max(low, used_low))
            if overlap > 0.50 * length:
                return True
        return False

    result: list[dict] = []
    for _, adjacency in sorted(candidates, key=lambda item: item[0]):
        key_a = (adjacency["piece_a"], adjacency["edge_a"])
        key_b = (adjacency["piece_b"], adjacency["edge_b"])
        interval_a = adjacency["edge_a_interval"]
        interval_b = adjacency["edge_b_interval"]
        if interval_conflicts(key_a, interval_a) or interval_conflicts(key_b, interval_b):
            continue
        for key, interval in ((key_a, interval_a), (key_b, interval_b)):
            low, high = sorted((float(interval[0]), float(interval[1])))
            used_intervals.setdefault(key, []).append((low, high))
        result.append(adjacency)
    return result


def _placement_offsets(
    polygons: Sequence[np.ndarray],
    width: float,
    height: float,
    gap_mm: float,
    adjacencies: Sequence[dict] | None = None,
    maximum_offset_mm: float = math.inf,
) -> list[np.ndarray]:
    if len(polygons) <= 1 or gap_mm <= 0.0:
        return [np.zeros(2, dtype=float) for _ in polygons]

    if adjacencies:
        offsets = [np.zeros(2, dtype=float) for _ in polygons]
        centres = [polygon.mean(axis=0) for polygon in polygons]
        constraints: list[tuple[int, int, np.ndarray, float]] = []
        for adjacency in adjacencies:
            piece_a = int(adjacency["piece_a"])
            piece_b = int(adjacency["piece_b"])
            polygon_a = polygons[piece_a]
            polygon_b = polygons[piece_b]
            edge_a = int(adjacency["edge_a"])
            edge_b = int(adjacency["edge_b"])
            a0 = polygon_a[edge_a]
            a1 = polygon_a[(edge_a + 1) % len(polygon_a)]
            b0 = polygon_b[edge_b]
            b1 = polygon_b[(edge_b + 1) % len(polygon_b)]
            interval_a = adjacency.get("edge_a_interval", [0.0, 1.0])
            interval_b = adjacency.get(
                "edge_b_interval",
                [1.0, 0.0] if adjacency["edge_b_reversed"] else [0.0, 1.0],
            )
            midpoint_a = (
                a0 + (a1 - a0) * float(sum(interval_a)) * 0.5
            )
            midpoint_b = (
                b0 + (b1 - b0) * float(sum(interval_b)) * 0.5
            )
            edge_direction = a1 - a0
            edge_length = float(np.linalg.norm(edge_direction))
            if edge_length <= 1e-9:
                continue
            normal = np.asarray(
                [-edge_direction[1], edge_direction[0]], dtype=float
            ) / edge_length
            if float(np.dot(normal, centres[piece_b] - centres[piece_a])) < 0.0:
                normal *= -1.0
            base_separation = float(np.dot(midpoint_b - midpoint_a, normal))
            constraints.append((piece_a, piece_b, normal, base_separation))

        # Project pair constraints repeatedly. Shared pieces propagate the gap
        # through chains, producing cumulative offsets for strip-like layouts.
        for _ in range(64):
            largest_deficit = 0.0
            for piece_a, piece_b, normal, base_separation in constraints:
                separation = base_separation + float(
                    np.dot(offsets[piece_b] - offsets[piece_a], normal)
                )
                deficit = gap_mm - separation
                if deficit <= 1e-6:
                    continue
                largest_deficit = max(largest_deficit, deficit)
                adjustment = normal * (deficit * 0.5)
                offsets[piece_a] -= adjustment
                offsets[piece_b] += adjustment

            mean_offset = np.mean(offsets, axis=0)
            for index in range(len(offsets)):
                offsets[index] -= mean_offset
                norm = float(np.linalg.norm(offsets[index]))
                if norm > maximum_offset_mm > 0.0:
                    offsets[index] *= maximum_offset_mm / norm
            if largest_deficit <= 1e-4:
                break
        if constraints:
            return offsets

    rectangle_centre = np.array([width * 0.5, height * 0.5], dtype=float)
    offsets: list[np.ndarray] = []
    for index, polygon in enumerate(polygons):
        direction = polygon.mean(axis=0) - rectangle_centre
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            angle = 2.0 * math.pi * index / len(polygons)
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        else:
            direction /= norm
        offsets.append(direction * gap_mm)
    return offsets


def _raster_overlap_area_mm2(
    polygons: Sequence[np.ndarray], resolution_mm: float = 0.20
) -> float:
    if len(polygons) <= 1:
        return 0.0
    points = np.concatenate(polygons, axis=0)
    minimum = points.min(axis=0) - resolution_mm * 2.0
    maximum = points.max(axis=0) + resolution_mm * 2.0
    size = np.ceil((maximum - minimum) / resolution_mm).astype(int) + 1
    occupancy = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    overlap_pixels = 0
    for polygon in polygons:
        contour = np.rint((polygon - minimum) / resolution_mm).astype(np.int32)
        piece_mask = np.zeros_like(occupancy)
        cv2.fillPoly(piece_mask, [contour], 1)
        overlap_pixels += int(np.count_nonzero((occupancy > 0) & (piece_mask > 0)))
        occupancy |= piece_mask
    return overlap_pixels * resolution_mm * resolution_mm


def _resolve_placement_overlaps(
    polygons: Sequence[np.ndarray],
    initial_offsets: Sequence[np.ndarray],
    maximum_offset_mm: float,
    tolerance_mm2: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Repel overlapping pairs while keeping every safety offset bounded."""

    offsets = [np.asarray(offset, dtype=float).copy() for offset in initial_offsets]
    step_mm = 0.25
    maximum_iterations = max(1, int(math.ceil(maximum_offset_mm / step_mm)) * 4)
    for _ in range(maximum_iterations):
        placed = [polygon + offset for polygon, offset in zip(polygons, offsets)]
        conflicts: list[tuple[int, int]] = []
        for first in range(len(placed)):
            for second in range(first + 1, len(placed)):
                if (
                    _raster_overlap_area_mm2([placed[first], placed[second]])
                    > tolerance_mm2
                ):
                    conflicts.append((first, second))
        if not conflicts:
            overlap = _raster_overlap_area_mm2(placed)
            return placed, offsets, overlap

        changed = False
        for first, second in conflicts:
            first_centre = polygons[first].mean(axis=0) + offsets[first]
            second_centre = polygons[second].mean(axis=0) + offsets[second]
            direction = first_centre - second_centre
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                angle = 2.0 * math.pi * first / max(1, len(polygons))
                direction = np.asarray([math.cos(angle), math.sin(angle)])
            else:
                direction /= length
            for index, sign in ((first, 1.0), (second, -1.0)):
                proposed = offsets[index] + direction * (sign * step_mm * 0.5)
                norm = float(np.linalg.norm(proposed))
                if norm > maximum_offset_mm > 0.0:
                    proposed *= maximum_offset_mm / norm
                if not np.allclose(proposed, offsets[index]):
                    offsets[index] = proposed
                    changed = True
        if not changed:
            break

    placed = [polygon + offset for polygon, offset in zip(polygons, offsets)]
    return placed, offsets, _raster_overlap_area_mm2(placed)


def _adjacency_vertex_distances(
    polygons: Sequence[np.ndarray], adjacencies: list[dict]
) -> list[dict]:
    result: list[dict] = []
    for adjacency in adjacencies:
        polygon_a = polygons[adjacency["piece_a"]]
        polygon_b = polygons[adjacency["piece_b"]]
        edge_a = adjacency["edge_a"]
        edge_b = adjacency["edge_b"]
        edge_a_start = polygon_a[edge_a]
        edge_a_end = polygon_a[(edge_a + 1) % len(polygon_a)]
        edge_b_start = polygon_b[edge_b]
        edge_b_end = polygon_b[(edge_b + 1) % len(polygon_b)]
        interval_a = adjacency.get("edge_a_interval", [0.0, 1.0])
        interval_b = adjacency.get(
            "edge_b_interval", [1.0, 0.0] if adjacency["edge_b_reversed"] else [0.0, 1.0]
        )
        a0 = edge_a_start + (edge_a_end - edge_a_start) * interval_a[0]
        a1 = edge_a_start + (edge_a_end - edge_a_start) * interval_a[1]
        b0 = edge_b_start + (edge_b_end - edge_b_start) * interval_b[0]
        b1 = edge_b_start + (edge_b_end - edge_b_start) * interval_b[1]
        distances = [float(np.linalg.norm(a0 - b0)), float(np.linalg.norm(a1 - b1))]
        result.append(
            {
                **adjacency,
                "corresponding_vertex_distances_mm": distances,
                "max_corresponding_vertex_distance_mm": max(distances),
            }
        )
    return result


def _profile_interval(
    profile: np.ndarray, interval: Sequence[float], sample_count: int
) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(profile))
    target = np.linspace(float(interval[0]), float(interval[1]), sample_count)
    return np.column_stack(
        [np.interp(target, source, profile[:, channel]) for channel in range(profile.shape[1])]
    )


def _score_edge_patterns(
    adjacencies: list[dict],
    edge_profiles: Sequence[Sequence[np.ndarray]] | None,
) -> tuple[list[dict], float, float]:
    """Attach LAB edge-profile mismatch scores to recovered seams.

    A flat white edge carries no directional evidence and therefore receives zero
    weight.  This prevents the solver from claiming that an unprinted/overexposed
    piece has a verified pattern match.
    """

    if edge_profiles is None:
        return adjacencies, 0.0, 0.0

    scored: list[dict] = []
    weighted_error = 0.0
    total_evidence = 0.0
    for adjacency in adjacencies:
        profile_a = np.asarray(
            edge_profiles[adjacency["piece_a"]][adjacency["edge_a"]], dtype=float
        )
        profile_b = np.asarray(
            edge_profiles[adjacency["piece_b"]][adjacency["edge_b"]], dtype=float
        )
        sample_count = max(len(profile_a), len(profile_b))
        profile_a = _profile_interval(
            profile_a, adjacency.get("edge_a_interval", [0.0, 1.0]), sample_count
        )
        profile_b = _profile_interval(
            profile_b,
            adjacency.get(
                "edge_b_interval",
                [1.0, 0.0] if adjacency["edge_b_reversed"] else [0.0, 1.0],
            ),
            sample_count,
        )

        if profile_a.shape[1] % 3 != 0 or profile_b.shape[1] != profile_a.shape[1]:
            raise ValueError("Edge profiles must contain one or more LAB triplets")
        depth_count = profile_a.shape[1] // 3
        values_a = profile_a.reshape(sample_count, depth_count, 3)
        values_b = profile_b.reshape(sample_count, depth_count, 3)
        depth_weights = np.geomspace(1.0, 0.35, depth_count)
        depth_weights /= depth_weights.sum()
        colour_delta = np.linalg.norm(values_a - values_b, axis=2)
        mismatch = float(
            np.mean(colour_delta @ depth_weights) / (255.0 * math.sqrt(3.0))
        )
        variation = max(
            float(np.mean(np.std(profile_a, axis=0))),
            float(np.mean(np.std(profile_b, axis=0))),
        )
        gradient = max(
            float(np.mean(np.linalg.norm(np.diff(profile_a, axis=0), axis=1))),
            float(np.mean(np.linalg.norm(np.diff(profile_b, axis=0), axis=1))),
        )
        evidence = float(np.clip(max(variation / 15.0, gradient / 10.0), 0.0, 1.0))
        weighted_error += mismatch * evidence
        total_evidence += evidence
        scored.append(
            {
                **adjacency,
                "pattern_mismatch": mismatch,
                "pattern_evidence": evidence,
            }
        )

    mismatch = weighted_error / total_evidence if total_evidence > 1e-9 else 0.0
    evidence = total_evidence / len(scored) if scored else 0.0
    return scored, mismatch, evidence


def _score_card_features(
    poses: Sequence[PoseCandidate],
    width: float,
    height: float,
    piece_features: Sequence[dict] | None,
    config: SolverConfig | None = None,
) -> tuple[float, float, float, float, dict]:
    """Score original card corners and diagonal rank/suit corner marks."""

    if piece_features is None:
        return 0.0, 0.0, 0.0, 0.0, {}
    config = config or SolverConfig()

    rectangle_corners = np.asarray(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=float,
    )
    rounded_error = 0.0
    rounded_weight = 0.0
    corner_ink = np.zeros(4, dtype=float)
    corner_red = np.zeros(4, dtype=float)
    corner_black = np.zeros(4, dtype=float)
    corner_roundness = np.zeros(4, dtype=float)
    trusted_rounded_corners: list[dict] = []

    for pose, features in zip(poses, piece_features):
        roundness = np.asarray(features.get("corner_roundness_mm", []), dtype=float)
        ink = np.asarray(features.get("corner_ink_density", []), dtype=float)
        red = np.asarray(features.get("corner_red_density", []), dtype=float)
        black = np.asarray(features.get("corner_black_density", []), dtype=float)
        if len(roundness) != len(pose.polygon):
            continue
        for vertex_index, vertex in enumerate(pose.polygon):
            previous = pose.polygon[(vertex_index - 1) % len(pose.polygon)] - vertex
            following = pose.polygon[(vertex_index + 1) % len(pose.polygon)] - vertex
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
            if denominator <= 1e-9:
                continue
            vertex_angle = math.degrees(
                math.acos(
                    float(np.clip(np.dot(previous, following) / denominator, -1.0, 1.0))
                )
            )
            right_angle_weight = float(
                np.clip(1.0 - abs(vertex_angle - 90.0) / 25.0, 0.0, 1.0)
            )
            distances = np.linalg.norm(rectangle_corners - vertex, axis=1)
            target_corner = int(np.argmin(distances))
            distance = float(distances[target_corner])
            roundness_mm = float(roundness[vertex_index])
            evidence = float(
                np.clip((roundness_mm - 0.35) / 1.75, 0.0, 1.0)
                * right_angle_weight
            )
            rounded_error += evidence * float(np.clip(distance / 8.0, 0.0, 1.0))
            rounded_weight += evidence
            if distance <= 8.0 and evidence >= corner_roundness[target_corner]:
                corner_roundness[target_corner] = evidence
                if vertex_index < len(ink):
                    corner_ink[target_corner] = float(ink[vertex_index])
                if vertex_index < len(red):
                    corner_red[target_corner] = float(red[vertex_index])
                if vertex_index < len(black):
                    corner_black[target_corner] = float(black[vertex_index])
            if (
                roundness_mm >= config.min_trusted_corner_roundness_mm
                and right_angle_weight
                >= config.min_trusted_corner_right_angle_weight
            ):
                trusted_rounded_corners.append(
                    {
                        "piece_index": pose.piece_index,
                        "vertex_index": vertex_index,
                        "roundness_mm": roundness_mm,
                        "right_angle_weight": right_angle_weight,
                        "previous_edge_length_mm": float(np.linalg.norm(previous)),
                        "following_edge_length_mm": float(np.linalg.norm(following)),
                        "nearest_rectangle_corner": target_corner,
                        "distance_mm": distance,
                        "misplaced": distance
                        > config.max_trusted_corner_distance_mm,
                    }
                )

    rounded_mismatch = rounded_error / rounded_weight if rounded_weight > 1e-9 else 0.0
    rounded_evidence = float(np.clip(rounded_weight / 3.0, 0.0, 1.0))
    first_diagonal = 0.5 * (corner_ink[0] + corner_ink[2])
    second_diagonal = 0.5 * (corner_ink[1] + corner_ink[3])
    diagonal_contrast = abs(first_diagonal - second_diagonal)
    maximum_diagonal = max(first_diagonal, second_diagonal)
    mark_evidence = float(
        np.clip(maximum_diagonal / 0.045, 0.0, 1.0)
        * np.clip(corner_roundness.sum() / 2.0, 0.0, 1.0)
    )
    mark_mismatch = float(1.0 - np.clip(diagonal_contrast / 0.055, 0.0, 1.0))
    misplaced_rounded_corners = [
        item for item in trusted_rounded_corners if item["misplaced"]
    ]
    details = {
        "corner_ink_density": corner_ink.tolist(),
        "corner_red_density": corner_red.tolist(),
        "corner_black_density": corner_black.tolist(),
        "corner_roundness_evidence": corner_roundness.tolist(),
        "diagonal_corner_mark_contrast": float(diagonal_contrast),
        "trusted_rounded_corners": trusted_rounded_corners,
        "trusted_rounded_corner_count": len(trusted_rounded_corners),
        "misplaced_rounded_corner_count": len(misplaced_rounded_corners),
    }
    return rounded_mismatch, rounded_evidence, mark_mismatch, mark_evidence, details


def _violates_trusted_corner_constraint(
    details: dict, config: SolverConfig
) -> bool:
    minimum = int(config.min_misplaced_trusted_corner_count)
    return minimum > 0 and details.get("misplaced_rounded_corner_count", 0) >= minimum


def _score_card_symmetry(
    poses: Sequence[PoseCandidate],
    width: float,
    height: float,
    piece_features: Sequence[dict] | None,
    resolution_mm: float = 1.5,
) -> tuple[float, float, dict]:
    """Compare assembled red/black ink with its 180-degree rotation."""

    if piece_features is None:
        return 0.0, 0.0, {}
    columns = max(1, int(math.ceil(width / resolution_mm)))
    rows = max(1, int(math.ceil(height / resolution_mm)))
    masks = np.zeros((2, rows, columns), dtype=np.uint8)
    point_count = 0
    for pose, features in zip(poses, piece_features):
        points = np.asarray(features.get("ink_points_mm", []), dtype=float)
        colours = np.asarray(features.get("ink_point_colours", []), dtype=int)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) != len(colours):
            continue
        cosine, sine = math.cos(pose.rotation_rad), math.sin(pose.rotation_rad)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
        transformed = points @ rotation.T + np.asarray(pose.translation, dtype=float)
        x = np.floor(transformed[:, 0] / resolution_mm).astype(int)
        y = np.floor(transformed[:, 1] / resolution_mm).astype(int)
        valid = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
        x, y, colours = x[valid], y[valid], colours[valid]
        masks[np.clip(colours, 0, 1), y, x] = 1
        point_count += int(np.count_nonzero(valid))
    if point_count == 0:
        return 0.0, 0.0, {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    intersections = 0
    unions = 0
    colour_scores: list[float] = []
    for colour_mask in masks:
        expanded = cv2.dilate(colour_mask, kernel)
        rotated = expanded[::-1, ::-1]
        intersection = int(np.count_nonzero(expanded & rotated))
        union = int(np.count_nonzero(expanded | rotated))
        intersections += intersection
        unions += union
        colour_scores.append(intersection / union if union else 1.0)
    symmetry = intersections / unions if unions else 1.0
    evidence = float(np.clip(unions / 180.0, 0.0, 1.0))
    return (
        float(1.0 - symmetry),
        evidence,
        {
            "ink_point_count": point_count,
            "red_symmetry_iou": colour_scores[1],
            "black_symmetry_iou": colour_scores[0],
        },
    )


def _assembly_is_connected(piece_count: int, adjacencies: Sequence[dict]) -> bool:
    if piece_count <= 1:
        return True
    neighbours: list[set[int]] = [set() for _ in range(piece_count)]
    for adjacency in adjacencies:
        piece_a = int(adjacency["piece_a"])
        piece_b = int(adjacency["piece_b"])
        neighbours[piece_a].add(piece_b)
        neighbours[piece_b].add(piece_a)
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbour in neighbours[current] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    return len(visited) == piece_count


def _polygon_edge_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest physical distance between two simple polygon edges."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_start = first[:, None, :]
    first_end = np.roll(first, -1, axis=0)[:, None, :]
    second_start = second[None, :, :]
    second_end = np.roll(second, -1, axis=0)[None, :, :]
    first_delta = first_end - first_start
    second_delta = second_end - second_start

    def cross(first_vector: np.ndarray, second_vector: np.ndarray) -> np.ndarray:
        return (
            first_vector[..., 0] * second_vector[..., 1]
            - first_vector[..., 1] * second_vector[..., 0]
        )

    first_a = cross(first_delta, second_start - first_start)
    first_b = cross(first_delta, second_end - first_start)
    second_a = cross(second_delta, first_start - second_start)
    second_b = cross(second_delta, first_end - second_start)
    epsilon = 1e-9

    def on_segment(
        point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        return (
            (np.minimum(start[..., 0], end[..., 0]) - epsilon <= point[..., 0])
            & (point[..., 0] <= np.maximum(start[..., 0], end[..., 0]) + epsilon)
            & (np.minimum(start[..., 1], end[..., 1]) - epsilon <= point[..., 1])
            & (point[..., 1] <= np.maximum(start[..., 1], end[..., 1]) + epsilon)
        )

    intersects = (
        ((first_a * first_b < 0.0) & (second_a * second_b < 0.0))
        | ((np.abs(first_a) <= epsilon) & on_segment(second_start, first_start, first_end))
        | ((np.abs(first_b) <= epsilon) & on_segment(second_end, first_start, first_end))
        | ((np.abs(second_a) <= epsilon) & on_segment(first_start, second_start, second_end))
        | ((np.abs(second_b) <= epsilon) & on_segment(first_end, second_start, second_end))
    )
    if np.any(intersects):
        return 0.0

    def point_to_segments(
        point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        edge = end - start
        length_squared = np.sum(edge * edge, axis=-1)
        numerator = np.sum((point - start) * edge, axis=-1)
        fraction = np.zeros_like(numerator)
        np.divide(
            numerator,
            length_squared,
            out=fraction,
            where=length_squared > 1e-12,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = start + fraction[..., None] * edge
        return np.linalg.norm(point - projected, axis=-1)

    return float(
        min(
            np.min(point_to_segments(first_start, second_start, second_end)),
            np.min(point_to_segments(first_end, second_start, second_end)),
            np.min(point_to_segments(second_start, first_start, first_end)),
            np.min(point_to_segments(second_end, first_start, first_end)),
        )
    )


def _minimum_piece_clearance(polygons: Sequence[np.ndarray]) -> float:
    if len(polygons) < 2:
        return math.inf
    return min(
        _polygon_edge_distance(first, second)
        for first_index, first in enumerate(polygons)
        for second in polygons[first_index + 1 :]
    )


def _apply_safe_placement_gap(
    polygons: Sequence[np.ndarray],
    piece_ids: Sequence[str],
    width: float,
    height: float,
    config: SolverConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], float, float, float, bool]:
    adjacencies = _detect_edge_adjacencies(polygons, piece_ids, config)
    requested_gap = max(0.0, config.placement_gap_mm)
    maximum_gap = max(requested_gap, config.max_placement_gap_mm)
    gaps = np.arange(requested_gap, maximum_gap + 0.001, 0.25)
    if len(gaps) == 0:
        gaps = np.array([0.0])

    best_result = None
    best_quality = None
    for gap in gaps:
        offsets = _placement_offsets(
            polygons,
            width,
            height,
            float(gap),
            adjacencies,
            maximum_gap,
        )
        placed, offsets, overlap_area = _resolve_placement_overlaps(
            polygons,
            offsets,
            maximum_gap,
            config.final_overlap_tolerance_mm2,
        )
        if overlap_area > config.final_overlap_tolerance_mm2:
            continue
        checked_adjacencies = _adjacency_vertex_distances(placed, adjacencies)
        vertices_valid = all(
            adjacency["max_corresponding_vertex_distance_mm"]
            <= config.max_adjacent_vertex_distance_mm
            for adjacency in checked_adjacencies
        )
        if not vertices_valid:
            continue
        achieved_gap = _minimum_piece_clearance(placed)
        quality = (
            achieved_gap,
            -overlap_area,
            -sum(float(np.linalg.norm(offset)) for offset in offsets),
        )
        candidate = (
            placed,
            offsets,
            checked_adjacencies,
            float(gap),
            overlap_area,
            achieved_gap,
            achieved_gap + 1e-6 >= requested_gap,
        )
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_result = candidate
        if candidate[-1]:
            return candidate

    if best_result is not None:
        return best_result

    raise RuntimeError(
        "A geometric assembly was found, but no safe placement satisfied the "
        "clearance and adjacent-vertex distance limits"
    )


def solve_puzzle(
    polygons: Sequence[ArrayLike],
    piece_ids: Sequence[str] | None = None,
    target_origin_mm: Sequence[float] = (0.0, 0.0),
    config: SolverConfig | None = None,
    edge_profiles: Sequence[Sequence[ArrayLike]] | None = None,
    piece_features: Sequence[dict] | None = None,
) -> dict:
    """Solve polygons into a landscape rectangle and return rigid target poses.

    Each returned transform maps the corresponding input polygon directly into the
    target coordinate system: target_point = R(rotation) @ input_point + translation.
    """

    config = config or SolverConfig()
    arrays = _validated_polygons(polygons)
    ids = list(piece_ids) if piece_ids is not None else [str(i) for i in range(len(arrays))]
    if len(ids) != len(arrays):
        raise ValueError("piece_ids and polygons must have the same length")
    validated_profiles: list[list[np.ndarray]] | None = None
    if edge_profiles is not None:
        if len(edge_profiles) != len(arrays):
            raise ValueError("edge_profiles and polygons must have the same length")
        validated_profiles = []
        for piece_index, (piece_profiles, polygon) in enumerate(zip(edge_profiles, arrays)):
            if len(piece_profiles) != len(polygon):
                raise ValueError(
                    f"Piece {piece_index} must provide one edge profile per polygon edge"
                )
            checked_piece: list[np.ndarray] = []
            for profile in piece_profiles:
                array = np.asarray(profile, dtype=float)
                if (
                    array.ndim != 2
                    or array.shape[0] < 2
                    or array.shape[1] < 3
                    or array.shape[1] % 3 != 0
                ):
                    raise ValueError(
                        "Each edge profile must be an N x (3 * depth_count) LAB array"
                    )
                checked_piece.append(array)
            validated_profiles.append(checked_piece)
    validated_features: list[dict] | None = None
    if piece_features is not None:
        if len(piece_features) != len(arrays):
            raise ValueError("piece_features and polygons must have the same length")
        validated_features = [dict(features) for features in piece_features]
    appearance_ink_point_count = sum(
        len(features.get("ink_points_mm", []))
        for features in (validated_features or [])
    )
    appearance_search_enabled = validated_profiles is not None and (
        validated_features is None
        or appearance_ink_point_count >= 300
    )
    origin = np.asarray(target_origin_mm, dtype=float)
    if origin.shape != (2,):
        raise ValueError("target_origin_mm must contain exactly two values")

    all_solutions: list[tuple[float, float, float, list[PoseCandidate], dict]] = []
    total_nodes = 0
    solve_started = time.monotonic()
    deadline = (
        solve_started + config.max_solve_seconds
        if config.max_solve_seconds > 0.0
        else None
    )
    search_timed_out = False
    rectangle_candidates_found = False
    relaxed_search_used = False
    base_search_config = config
    if appearance_search_enabled:
        base_search_config = replace(
            config,
            max_solutions_per_rectangle=max(
                config.max_solutions_per_rectangle,
                config.pattern_max_solutions_per_rectangle,
            ),
        )
    search_configs: list[tuple[SolverConfig, bool]] = [(base_search_config, False)]
    if config.enable_relaxed_retry:
        relaxed_config = replace(
            base_search_config,
            dimension_area_tolerance=max(
                config.dimension_area_tolerance,
                config.retry_dimension_area_tolerance,
            ),
            inside_tolerance_mm=max(
                config.inside_tolerance_mm, config.retry_inside_tolerance_mm
            ),
            max_hole_ratio=max(config.max_hole_ratio, config.retry_max_hole_ratio),
            max_overlap_ratio=max(
                config.max_overlap_ratio, config.retry_max_overlap_ratio
            ),
            early_accept_score=max(
                config.early_accept_score, config.retry_early_accept_score
            ),
            enable_relaxed_retry=False,
        )
        relaxed_phase = (relaxed_config, True)
        if config.relaxed_search_first:
            search_configs.insert(0, relaxed_phase)
        else:
            search_configs.append(relaxed_phase)

    for search_config, phase_is_relaxed in search_configs:
        rectangles = candidate_rectangles(arrays, search_config)
        rectangle_candidates_found |= bool(rectangles)
        solved_rectangle_count = 0
        for width, height, dimension_error in rectangles:
            if deadline is not None and time.monotonic() >= deadline:
                search_timed_out = True
                break
            solutions, nodes, rectangle_timed_out = _search_rectangle(
                arrays, width, height, dimension_error, search_config, deadline
            )
            total_nodes += nodes
            for score, poses, metrics in solutions:
                all_solutions.append((score, width, height, poses, metrics))
            if solutions:
                solved_rectangle_count += 1
            all_solutions.sort(key=lambda item: item[0])
            if (
                appearance_search_enabled
                and solved_rectangle_count >= max(1, config.pattern_rectangle_candidates)
            ):
                break
            if (
                all_solutions
                and not appearance_search_enabled
                and search_config.early_accept_score > 0.0
                and all_solutions[0][0] <= search_config.early_accept_score
            ):
                break
            if rectangle_timed_out:
                search_timed_out = True
                break
        if all_solutions:
            relaxed_search_used = phase_is_relaxed
            break
        if search_timed_out:
            break

    if not all_solutions:
        if search_timed_out:
            raise RuntimeError(
                f"Puzzle search exceeded its {config.max_solve_seconds:.1f} second time budget"
            )
        if not rectangle_candidates_found:
            raise RuntimeError("No rectangle dimensions satisfy the configured ranges")
        raise RuntimeError(
            "No valid assembly found. Check polygon calibration, dimension ranges, "
            "or increase the hole/area tolerances."
        )

    evaluated_solutions = []
    for score, width, height, poses, metrics in sorted(all_solutions, key=lambda item: item[0]):
        card_symmetry_mismatch, card_symmetry_evidence, symmetry_details = (
            _score_card_symmetry(poses, width, height, validated_features)
        )
        if (
            appearance_search_enabled
            and card_symmetry_evidence >= config.min_card_symmetry_evidence
            and card_symmetry_mismatch > config.max_card_symmetry_mismatch
        ):
            continue
        assembled_polygons = [pose.polygon for pose in poses]
        try:
            (
                placed_polygons,
                placement_offsets,
                adjacencies,
                applied_gap_mm,
                final_overlap_area_mm2,
                achieved_gap_mm,
                gap_satisfied,
            ) = _apply_safe_placement_gap(
                assembled_polygons,
                ids,
                width,
                height,
                config,
            )
        except RuntimeError:
            continue

        connected = _assembly_is_connected(len(arrays), adjacencies)
        if config.require_connected_assembly and not connected:
            continue
        adjacencies, pattern_mismatch, pattern_evidence = _score_edge_patterns(
            adjacencies, validated_profiles
        )
        if (
            pattern_evidence >= config.min_pattern_evidence
            and pattern_mismatch > config.max_pattern_mismatch
        ):
            continue
        (
            rounded_corner_mismatch,
            rounded_corner_evidence,
            corner_mark_mismatch,
            corner_mark_evidence,
            card_feature_details,
        ) = _score_card_features(
            poses, width, height, validated_features, config=config
        )
        if _violates_trusted_corner_constraint(card_feature_details, config):
            continue
        ranked_score = (
            score
            + max(0.0, config.placement_gap_mm - achieved_gap_mm)
            / max(config.placement_gap_mm, 1e-6)
            + config.pattern_score_weight * pattern_mismatch * pattern_evidence
            + config.rounded_corner_score_weight
            * rounded_corner_mismatch
            * rounded_corner_evidence
            + config.corner_mark_score_weight
            * corner_mark_mismatch
            * corner_mark_evidence
            + config.card_symmetry_score_weight
            * card_symmetry_mismatch
            * card_symmetry_evidence
        )
        evaluated_solutions.append(
            (
                ranked_score,
                score,
                width,
                height,
                poses,
                metrics,
                placed_polygons,
                placement_offsets,
                adjacencies,
                applied_gap_mm,
                final_overlap_area_mm2,
                achieved_gap_mm,
                gap_satisfied,
                connected,
                pattern_mismatch,
                pattern_evidence,
                rounded_corner_mismatch,
                rounded_corner_evidence,
                corner_mark_mismatch,
                corner_mark_evidence,
                card_feature_details,
                card_symmetry_mismatch,
                card_symmetry_evidence,
                symmetry_details,
            )
        )

    if not evaluated_solutions:
        raise RuntimeError(
            "Geometric candidates were found, but none satisfied connectivity, "
            "non-overlap, adjacent-vertex, and visible-pattern constraints"
        )

    (
        _,
        score,
        width,
        height,
        poses,
        metrics,
        placed_polygons,
        placement_offsets,
        adjacencies,
        applied_gap_mm,
        final_overlap_area_mm2,
        achieved_gap_mm,
        gap_satisfied,
        connected,
        pattern_mismatch,
        pattern_evidence,
        rounded_corner_mismatch,
        rounded_corner_evidence,
        corner_mark_mismatch,
        corner_mark_evidence,
        card_feature_details,
        card_symmetry_mismatch,
        card_symmetry_evidence,
        symmetry_details,
    ) = min(evaluated_solutions, key=lambda item: item[0])
    output_pieces = []
    for piece_id, pose, placed_polygon, placement_offset in zip(
        ids, poses, placed_polygons, placement_offsets
    ):
        cosine, sine = math.cos(pose.rotation_rad), math.sin(pose.rotation_rad)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        absolute_translation = np.asarray(pose.translation) + placement_offset + origin
        absolute_polygon = placed_polygon + origin
        output_pieces.append(
            {
                "id": piece_id,
                "rotation_deg": math.degrees(pose.rotation_rad),
                "rotation_matrix": rotation.tolist(),
                "translation_mm": absolute_translation.tolist(),
                "target_polygon_mm": absolute_polygon.tolist(),
                "boundary_side": pose.boundary_side,
                "source_edge": pose.source_edge,
                "safety_offset_mm": placement_offset.tolist(),
            }
        )

    return {
        "rectangle": {
            "origin_mm": origin.tolist(),
            "width_mm": width,
            "height_mm": height,
        },
        "score": score,
        "metrics": {
            **metrics,
            "total_search_nodes": total_nodes,
            "applied_placement_gap_mm": applied_gap_mm,
            "requested_placement_gap_mm": config.placement_gap_mm,
            "achieved_placement_gap_mm": achieved_gap_mm,
            "placement_gap_satisfied": gap_satisfied,
            "final_overlap_area_mm2": final_overlap_area_mm2,
            "max_adjacent_vertex_distance_mm": max(
                (
                    adjacency["max_corresponding_vertex_distance_mm"]
                    for adjacency in adjacencies
                ),
                default=0.0,
            ),
            "assembly_connected": connected,
            "pattern_mismatch": pattern_mismatch,
            "pattern_evidence": pattern_evidence,
            "rounded_corner_mismatch": rounded_corner_mismatch,
            "rounded_corner_evidence": rounded_corner_evidence,
            "corner_mark_mismatch": corner_mark_mismatch,
            "corner_mark_evidence": corner_mark_evidence,
            "card_features": card_feature_details,
            "card_symmetry_mismatch": card_symmetry_mismatch,
            "card_symmetry_evidence": card_symmetry_evidence,
            "card_symmetry": symmetry_details,
            "solve_elapsed_seconds": time.monotonic() - solve_started,
            "search_timed_out": search_timed_out,
            "relaxed_search_used": relaxed_search_used,
            "appearance_search_enabled": appearance_search_enabled,
            "appearance_ink_point_count": appearance_ink_point_count,
        },
        "pieces": output_pieces,
        "adjacencies": adjacencies,
        "config": asdict(config),
    }


def _load_problem(path: Path) -> tuple[list, list[str], list[float], SolverConfig]:
    document = json.loads(path.read_text(encoding="utf-8"))
    pieces = document["pieces"]
    polygons = [piece["polygon_mm"] for piece in pieces]
    ids = [str(piece.get("id", index)) for index, piece in enumerate(pieces)]
    origin = document.get("target_origin_mm", [0.0, 0.0])
    config_values = document.get("solver", {})
    for key in ("width_range", "height_range"):
        if key in config_values:
            config_values[key] = tuple(config_values[key])
    return polygons, ids, origin, SolverConfig(**config_values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve up to four polygon pieces into a rectangular target"
    )
    parser.add_argument("input", type=Path, help="JSON problem file")
    parser.add_argument("-o", "--output", type=Path, help="Write result JSON here")
    arguments = parser.parse_args(argv)

    polygons, ids, origin, config = _load_problem(arguments.input)
    result = solve_puzzle(polygons, ids, origin, config)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
