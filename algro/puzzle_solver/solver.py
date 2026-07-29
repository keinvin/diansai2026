from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
) -> tuple[list[tuple[float, list[PoseCandidate], dict]], int]:
    candidates = [
        generate_piece_poses(index, polygon, width, height, config)
        for index, polygon in enumerate(polygons)
    ]
    if any(not piece_candidates for piece_candidates in candidates):
        return [], 0

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

    def dfs(depth: int, occupied: int, overlap_cells: int) -> None:
        nonlocal nodes
        if nodes >= config.max_search_nodes_per_rectangle:
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
            overlap = (occupied & pose.mask).bit_count()
            new_overlap = overlap_cells + overlap
            if new_overlap > allowed_overlap:
                continue
            chosen[piece_index] = pose
            dfs(depth + 1, occupied | pose.mask, new_overlap)
            chosen[piece_index] = None

    dfs(0, 0, 0)
    return solutions, nodes


def solve_puzzle(
    polygons: Sequence[ArrayLike],
    piece_ids: Sequence[str] | None = None,
    target_origin_mm: Sequence[float] = (0.0, 0.0),
    config: SolverConfig | None = None,
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
    origin = np.asarray(target_origin_mm, dtype=float)
    if origin.shape != (2,):
        raise ValueError("target_origin_mm must contain exactly two values")

    rectangles = candidate_rectangles(arrays, config)
    if not rectangles:
        raise RuntimeError("No rectangle dimensions satisfy the configured ranges")

    all_solutions: list[tuple[float, float, float, list[PoseCandidate], dict]] = []
    total_nodes = 0
    for width, height, dimension_error in rectangles:
        solutions, nodes = _search_rectangle(
            arrays, width, height, dimension_error, config
        )
        total_nodes += nodes
        for score, poses, metrics in solutions:
            all_solutions.append((score, width, height, poses, metrics))
        if all_solutions and all_solutions[0][0] <= 0.005:
            break
        all_solutions.sort(key=lambda item: item[0])

    if not all_solutions:
        raise RuntimeError(
            "No valid assembly found. Check polygon calibration, dimension ranges, "
            "or increase the hole/area tolerances."
        )

    all_solutions.sort(key=lambda item: item[0])
    score, width, height, poses, metrics = all_solutions[0]
    output_pieces = []
    for piece_id, pose in zip(ids, poses):
        cosine, sine = math.cos(pose.rotation_rad), math.sin(pose.rotation_rad)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        absolute_translation = np.asarray(pose.translation) + origin
        absolute_polygon = pose.polygon + origin
        output_pieces.append(
            {
                "id": piece_id,
                "rotation_deg": math.degrees(pose.rotation_rad),
                "rotation_matrix": rotation.tolist(),
                "translation_mm": absolute_translation.tolist(),
                "target_polygon_mm": absolute_polygon.tolist(),
                "boundary_side": pose.boundary_side,
                "source_edge": pose.source_edge,
            }
        )

    return {
        "rectangle": {
            "origin_mm": origin.tolist(),
            "width_mm": width,
            "height_mm": height,
        },
        "score": score,
        "metrics": {**metrics, "total_search_nodes": total_nodes},
        "pieces": output_pieces,
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
