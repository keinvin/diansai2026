from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


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
    max_hole_ratio: float = 0.025
    max_overlap_ratio: float = 0.002
    max_search_nodes_per_rectangle: int = 300_000
    max_solutions_per_rectangle: int = 8
    max_solve_seconds: float = 0.0
    placement_gap_mm: float = 1.5
    max_placement_gap_mm: float = 8.0
    adjacency_detection_tolerance_mm: float = 8.0
    max_adjacent_vertex_distance_mm: float = 20.0
    final_overlap_tolerance_mm2: float = 0.25
    pattern_score_weight: float = 0.15
    min_pattern_evidence: float = 0.12
    max_pattern_mismatch: float = 0.45
    require_connected_assembly: bool = True


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

    candidates.sort(key=lambda item: (item[2], item[0] * 2.0 + item[1]))
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
    candidates = [
        generate_piece_poses(index, polygon, width, height, config)
        for index, polygon in enumerate(polygons)
    ]
    if any(not piece_candidates for piece_candidates in candidates):
        return [], 0, False

    order = sorted(range(len(polygons)), key=lambda index: len(candidates[index]))
    rectangle_cells = int(round(width / config.grid_mm)) * int(
        round(height / config.grid_mm)
    )
    allowed_holes = math.ceil(rectangle_cells * config.max_hole_ratio)
    allowed_overlap = math.ceil(rectangle_cells * config.max_overlap_ratio)

    remaining_possible = [0] * (len(order) + 1)
    for depth in range(len(order) - 1, -1, -1):
        union = 0
        for pose in candidates[order[depth]]:
            union |= pose.mask
        remaining_possible[depth] = remaining_possible[depth + 1] | union

    chosen: list[PoseCandidate | None] = [None] * len(polygons)
    solutions: list[tuple[float, list[PoseCandidate], dict]] = []
    nodes = 0
    timed_out = False

    def dfs(depth: int, occupied: int, overlap_cells: int) -> None:
        nonlocal nodes, timed_out
        if nodes >= config.max_search_nodes_per_rectangle:
            return
        if deadline is not None and nodes % 256 == 0 and time.monotonic() >= deadline:
            timed_out = True
            return
        nodes += 1

        impossible_holes = rectangle_cells - (occupied | remaining_possible[depth]).bit_count()
        if impossible_holes > allowed_holes:
            return

        if depth == len(order):
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
            }
            solutions.append((score, selected, metrics))
            solutions.sort(key=lambda item: item[0])
            del solutions[config.max_solutions_per_rectangle :]
            return

        piece_index = order[depth]
        for pose in candidates[piece_index]:
            if timed_out:
                return
            overlap = (occupied & pose.mask).bit_count()
            new_overlap = overlap_cells + overlap
            if new_overlap > allowed_overlap:
                continue
            chosen[piece_index] = pose
            dfs(depth + 1, occupied | pose.mask, new_overlap)
            chosen[piece_index] = None

    dfs(0, 0, 0)
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
    polygons: Sequence[np.ndarray], width: float, height: float, gap_mm: float
) -> list[np.ndarray]:
    if len(polygons) <= 1 or gap_mm <= 0.0:
        return [np.zeros(2, dtype=float) for _ in polygons]

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

        colour_delta = np.linalg.norm(profile_a - profile_b, axis=1)
        mismatch = float(colour_delta.mean() / (255.0 * math.sqrt(3.0)))
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


def _apply_safe_placement_gap(
    polygons: Sequence[np.ndarray],
    piece_ids: Sequence[str],
    width: float,
    height: float,
    config: SolverConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], float, float]:
    adjacencies = _detect_edge_adjacencies(polygons, piece_ids, config)
    requested_gap = max(0.0, config.placement_gap_mm)
    maximum_gap = max(requested_gap, config.max_placement_gap_mm)
    gaps = np.arange(requested_gap, maximum_gap + 0.001, 0.25)
    if len(gaps) == 0:
        gaps = np.array([0.0])

    for gap in gaps:
        offsets = _placement_offsets(polygons, width, height, float(gap))
        placed = [polygon + offset for polygon, offset in zip(polygons, offsets)]
        overlap_area = _raster_overlap_area_mm2(placed)
        checked_adjacencies = _adjacency_vertex_distances(placed, adjacencies)
        vertices_valid = all(
            adjacency["max_corresponding_vertex_distance_mm"]
            <= config.max_adjacent_vertex_distance_mm
            for adjacency in checked_adjacencies
        )
        if overlap_area <= config.final_overlap_tolerance_mm2 and vertices_valid:
            return placed, offsets, checked_adjacencies, float(gap), overlap_area

    raise RuntimeError(
        "A geometric assembly was found, but no safe non-overlapping placement "
        "satisfied the adjacent-vertex distance limit"
    )


def solve_puzzle(
    polygons: Sequence[ArrayLike],
    piece_ids: Sequence[str] | None = None,
    target_origin_mm: Sequence[float] = (0.0, 0.0),
    config: SolverConfig | None = None,
    edge_profiles: Sequence[Sequence[ArrayLike]] | None = None,
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
                if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 3:
                    raise ValueError("Each edge profile must be an N x 3 LAB array")
                checked_piece.append(array)
            validated_profiles.append(checked_piece)
    origin = np.asarray(target_origin_mm, dtype=float)
    if origin.shape != (2,):
        raise ValueError("target_origin_mm must contain exactly two values")

    rectangles = candidate_rectangles(arrays, config)
    if not rectangles:
        raise RuntimeError("No rectangle dimensions satisfy the configured ranges")

    all_solutions: list[tuple[float, float, float, list[PoseCandidate], dict]] = []
    total_nodes = 0
    solve_started = time.monotonic()
    deadline = (
        solve_started + config.max_solve_seconds
        if config.max_solve_seconds > 0.0
        else None
    )
    search_timed_out = False
    for width, height, dimension_error in rectangles:
        if deadline is not None and time.monotonic() >= deadline:
            search_timed_out = True
            break
        solutions, nodes, rectangle_timed_out = _search_rectangle(
            arrays, width, height, dimension_error, config, deadline
        )
        total_nodes += nodes
        for score, poses, metrics in solutions:
            all_solutions.append((score, width, height, poses, metrics))
        if all_solutions and all_solutions[0][0] <= 0.005:
            break
        if rectangle_timed_out:
            search_timed_out = True
            break
        all_solutions.sort(key=lambda item: item[0])

    if not all_solutions:
        if search_timed_out:
            raise RuntimeError(
                f"Puzzle search exceeded its {config.max_solve_seconds:.1f} second time budget"
            )
        raise RuntimeError(
            "No valid assembly found. Check polygon calibration, dimension ranges, "
            "or increase the hole/area tolerances."
        )

    evaluated_solutions = []
    for score, width, height, poses, metrics in sorted(all_solutions, key=lambda item: item[0]):
        assembled_polygons = [pose.polygon for pose in poses]
        try:
            (
                placed_polygons,
                placement_offsets,
                adjacencies,
                applied_gap_mm,
                final_overlap_area_mm2,
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
        ranked_score = score + config.pattern_score_weight * pattern_mismatch
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
                connected,
                pattern_mismatch,
                pattern_evidence,
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
        connected,
        pattern_mismatch,
        pattern_evidence,
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
            "solve_elapsed_seconds": time.monotonic() - solve_started,
            "search_timed_out": search_timed_out,
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
