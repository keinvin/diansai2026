"""Main recognition -> solve -> coordinate transform -> motion pipeline."""

from __future__ import annotations

import logging
import itertools
import math
from dataclasses import dataclass, fields
from typing import Callable, Sequence

import numpy as np

from .config import load_config, project_path
from .timing import StageTimings, append_timing_log

from puzzle_solver.coordinates import A4ToGrblTransform
from puzzle_solver.solver import SolverConfig, solve_puzzle
from puzzle_solver.vision import (
    Calibration,
    DetectionResult,
    VisionConfig,
    detect_pieces,
    extract_edge_profiles,
    extract_piece_features,
)
from motion.motion_exec import MotionExecutor, PickPlaceConfig, build_pick_place_plan


LOGGER = logging.getLogger(__name__)


@dataclass
class RecognitionRun:
    result: DetectionResult
    solution: dict | None
    solve_error: str | None
    corrected_frame: np.ndarray
    calibration: Calibration
    timings: dict[str, float]
    puzzle_search_enabled: bool
    transport_only: bool


@dataclass
class MotionRun:
    plan: list[dict]
    timings: dict[str, float]


def solver_config_from_dict(document: dict) -> SolverConfig:
    values = dict(document.get("solver", {}))
    for key in ("width_range", "height_range"):
        if key in values:
            values[key] = tuple(values[key])
    valid_fields = {field.name for field in fields(SolverConfig)}
    return SolverConfig(**{key: value for key, value in values.items() if key in valid_fields})


def place_solution_in_opposite_half(
    solution: dict,
    source_region: str,
    a4_width: float,
    a4_height: float,
) -> dict:
    """Centre a zero-origin solver result in the unused half of the A4 sheet."""

    rectangle = solution["rectangle"]
    width = float(rectangle["width_mm"])
    height = float(rectangle["height_mm"])
    half_height = a4_height / 2.0
    if width > a4_width or height > half_height:
        raise RuntimeError(
            f"拼接矩形 {width:.1f}×{height:.1f} mm 放不进半张 A4"
        )
    target_half_y = half_height if source_region == "upper" else 0.0
    origin = np.asarray(
        [(a4_width - width) / 2.0, target_half_y + (half_height - height) / 2.0],
        dtype=float,
    )
    rectangle["origin_mm"] = origin.tolist()
    for piece in solution["pieces"]:
        piece["translation_mm"] = (
            np.asarray(piece["translation_mm"], dtype=float) + origin
        ).tolist()
        piece["target_polygon_mm"] = (
            np.asarray(piece["target_polygon_mm"], dtype=float) + origin
        ).tolist()
    return solution


def build_transport_solution(
    pieces: Sequence[object],
    source_region: str,
    a4_width: float,
    a4_height: float,
    *,
    gap_mm: float = 8.0,
) -> dict:
    """Pack detected pieces into the other A4 half without puzzle solving.

    The first-question workflow intentionally does not infer an assembly.  It
    first preserves every piece's direction, then also tries 90-degree turns
    when necessary to make the detected pieces fit into the available half.
    """

    if not pieces:
        raise RuntimeError("没有可搬运的碎片")
    half_height = a4_height / 2.0
    target_y0 = half_height if source_region == "upper" else 0.0
    ordered = sorted(pieces, key=lambda piece: (piece.centroid_mm[1], piece.centroid_mm[0]))

    def pack(
        candidate_gap_mm: float,
        placements: Sequence[tuple[object, float]],
    ) -> list[dict] | None:
        cursor_x = candidate_gap_mm
        cursor_y = target_y0 + candidate_gap_mm
        row_height = 0.0
        placed: list[dict] = []
        for piece, rotation_deg in placements:
            polygon = np.asarray(piece.polygon_mm, dtype=float)
            rotation = (
                np.asarray([[1.0, 0.0], [0.0, 1.0]])
                if rotation_deg == 0.0
                else np.asarray([[0.0, -1.0], [1.0, 0.0]])
            )
            rotated_polygon = polygon @ rotation.T
            minimum = rotated_polygon.min(axis=0)
            maximum = rotated_polygon.max(axis=0)
            width, height = maximum - minimum
            if (
                width + 2.0 * candidate_gap_mm > a4_width
                or height + 2.0 * candidate_gap_mm > half_height
            ):
                return None
            if cursor_x + width + candidate_gap_mm > a4_width:
                cursor_x = candidate_gap_mm
                cursor_y += row_height + candidate_gap_mm
                row_height = 0.0
            if cursor_y + height + candidate_gap_mm > target_y0 + half_height:
                return None
            translation = np.asarray([cursor_x, cursor_y], dtype=float) - minimum
            pickup_source = np.asarray(piece.pickup_point_mm, dtype=float)
            placed.append(
                {
                    "id": piece.id,
                    "rotation_deg": rotation_deg,
                    "rotation_matrix": rotation.tolist(),
                    "translation_mm": translation.tolist(),
                    "target_polygon_mm": (rotated_polygon + translation).tolist(),
                    "pickup_source_mm": pickup_source.tolist(),
                    "pickup_target_mm": (rotation @ pickup_source + translation).tolist(),
                }
            )
            cursor_x += width + candidate_gap_mm
            row_height = max(row_height, float(height))
        return placed

    placed: list[dict] | None = None
    applied_gap_mm = gap_mm
    # At most four pieces are detected, so 4! * 2^4 layouts remain cheap.
    # Keep the largest safe gap that fits, then choose the most compact
    # rectangle instead of accepting the first arbitrary shelf placement.
    target_aspect = a4_width / half_height
    for candidate_gap_mm in dict.fromkeys((gap_mm, 6.0, 4.0, 2.0, 1.0, 0.5)):
        best_layout: list[dict] | None = None
        best_score: tuple[float, float, int, tuple[str, ...]] | None = None
        for order in itertools.permutations(ordered):
            for rotations in itertools.product((0.0, 90.0), repeat=len(order)):
                candidate = pack(
                    float(candidate_gap_mm), tuple(zip(order, rotations))
                )
                if candidate is None:
                    continue
                points = np.vstack(
                    [np.asarray(piece["target_polygon_mm"], dtype=float) for piece in candidate]
                )
                size = points.max(axis=0) - points.min(axis=0)
                rectangle_area = float(size[0] * size[1])
                aspect_error = abs(
                    math.log(max(float(size[0]), 1e-6) / max(float(size[1]), 1e-6) / target_aspect)
                )
                score = (
                    rectangle_area,
                    aspect_error,
                    sum(rotation != 0.0 for rotation in rotations),
                    tuple(str(piece.id) for piece in order),
                )
                if best_score is None or score < best_score:
                    best_layout = candidate
                    best_score = score
        if best_layout is not None:
            placed = best_layout
            applied_gap_mm = float(candidate_gap_mm)
            break
    if placed is None:
        raise RuntimeError("碎片无法无重叠地放入另一半 A4")

    all_points = np.vstack(
        [np.asarray(piece["target_polygon_mm"], dtype=float) for piece in placed]
    )
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    size = upper - lower
    centering_offset = np.asarray(
        [
            (a4_width - size[0]) / 2.0 - lower[0],
            target_y0 + (half_height - size[1]) / 2.0 - lower[1],
        ],
        dtype=float,
    )
    for piece in placed:
        piece["translation_mm"] = (
            np.asarray(piece["translation_mm"], dtype=float) + centering_offset
        ).tolist()
        piece["target_polygon_mm"] = (
            np.asarray(piece["target_polygon_mm"], dtype=float) + centering_offset
        ).tolist()
        piece["pickup_target_mm"] = (
            np.asarray(piece["pickup_target_mm"], dtype=float) + centering_offset
        ).tolist()
    lower += centering_offset
    return {
        "pieces": placed,
        "rectangle": {
            "origin_mm": lower.tolist(),
            "width_mm": float(size[0]),
            "height_mm": float(size[1]),
        },
        "metrics": {
            "pattern_evidence": 0.0,
            "pattern_mismatch": 0.0,
            "hole_ratio": 0.0,
            "overlap_ratio": 0.0,
            "applied_placement_gap_mm": applied_gap_mm,
            "final_overlap_area_mm2": 0.0,
            "max_adjacent_vertex_distance_mm": 0.0,
        },
        "config": {"min_pattern_evidence": 1.0},
        "workflow": "transport",
    }


class MainPipeline:
    """Own the non-UI flow and its standalone configuration."""

    def __init__(self, document: dict | None = None) -> None:
        self.document = document if document is not None else load_config()
        self.vision_config = VisionConfig(**self.document["vision"])
        self.solver_config = solver_config_from_dict(self.document)
        self.motion_config = PickPlaceConfig(**self.document["motion"])
        self.timing_enabled = bool(self.document.get("timing", {}).get("enabled", True))

    def _log_timings(
        self,
        event: str,
        timings: StageTimings,
        **fields,
    ) -> None:
        config = self.document.get("timing", {})
        if not self.timing_enabled or not bool(config.get("log_enabled", True)):
            return
        configured_path = config.get("log_file", "data/logs/main_timing.jsonl")
        try:
            append_timing_log(
                project_path(configured_path),
                event,
                timings.to_dict(),
                **fields,
            )
        except OSError as exc:
            # Performance logging must never turn a successful hardware run into
            # a failed one, but the warning remains visible in terminal logs.
            LOGGER.warning("无法写入性能日志 %s: %s", configured_path, exc)

    def recognize(
        self,
        frame: np.ndarray,
        *,
        puzzle_search_enabled: bool | None = None,
        transport_only: bool = False,
        use_piece_features: bool = True,
        initial_timings: dict[str, float] | None = None,
    ) -> RecognitionRun:
        timings = StageTimings.from_dict(
            initial_timings, enabled=self.timing_enabled
        )
        try:
            with timings.measure("recognition"):
                calibration = Calibration.from_dict(self.document, frame.shape)
                result = detect_pieces(frame, calibration, self.vision_config)
                corrected_frame = calibration.undistort_image(frame)
        except Exception as exc:
            self._log_timings(
                "recognition_failed",
                timings,
                error=str(exc),
                a4_region=self.document.get("a4_region", "upper"),
            )
            raise

        enabled = (
            bool(self.document.get("puzzle_search_enabled", True))
            if puzzle_search_enabled is None
            else bool(puzzle_search_enabled)
        )
        enabled = enabled and not transport_only
        solution = None
        solve_error = None
        if transport_only:
            with timings.measure("solve"):
                try:
                    solution = build_transport_solution(
                        result.pieces,
                        self.document.get("a4_region", "upper"),
                        float(self.document["a4_width_mm"]),
                        float(self.document["a4_height_mm"]),
                    )
                except (RuntimeError, ValueError) as exc:
                    solve_error = str(exc)
        elif enabled:
            with timings.measure("solve"):
                try:
                    edge_profiles = extract_edge_profiles(corrected_frame, result.pieces)
                    piece_features = (
                        extract_piece_features(
                            corrected_frame, result.pieces, calibration
                        )
                        if use_piece_features
                        else None
                    )
                    solution = solve_puzzle(
                        [piece.polygon_mm for piece in result.pieces],
                        [piece.id for piece in result.pieces],
                        target_origin_mm=(0.0, 0.0),
                        config=self.solver_config,
                        edge_profiles=edge_profiles,
                        piece_features=piece_features,
                    )
                    solution = place_solution_in_opposite_half(
                        solution,
                        self.document.get("a4_region", "upper"),
                        float(self.document["a4_width_mm"]),
                        float(self.document["a4_height_mm"]),
                    )
                    pickup_by_id = {
                        piece.id: np.asarray(piece.pickup_point_mm, dtype=float)
                        for piece in result.pieces
                    }
                    for target_piece in solution["pieces"]:
                        pickup_source = pickup_by_id[target_piece["id"]]
                        rotation = np.asarray(
                            target_piece["rotation_matrix"], dtype=float
                        )
                        translation = np.asarray(
                            target_piece["translation_mm"], dtype=float
                        )
                        target_piece["pickup_source_mm"] = pickup_source.tolist()
                        target_piece["pickup_target_mm"] = (
                            rotation @ pickup_source + translation
                        ).tolist()
                except (RuntimeError, ValueError) as exc:
                    solve_error = str(exc)

        run = RecognitionRun(
            result=result,
            solution=solution,
            solve_error=solve_error,
            corrected_frame=corrected_frame,
            calibration=calibration,
            timings=timings.to_dict(),
            puzzle_search_enabled=enabled,
            transport_only=bool(transport_only),
        )
        self._log_timings(
            "recognition_completed",
            timings,
            a4_region=self.document.get("a4_region", "upper"),
            piece_count=len(result.pieces),
            puzzle_search_enabled=enabled,
            transport_only=bool(transport_only),
            use_piece_features=bool(use_piece_features),
            solution_found=solution is not None,
            solve_error=solve_error,
        )
        return run

    def _transform(self) -> A4ToGrblTransform:
        configured = self.document.get("paths", {}).get(
            "a4_grbl_calibration", "data/a4_grbl_calibration_samples.json"
        )
        return A4ToGrblTransform.load_initial_calibration(project_path(configured))

    def build_motion_plan(self, solution: dict) -> list[dict]:
        return build_pick_place_plan(solution, self._transform(), self.motion_config)

    def execute_solution(
        self,
        solution: dict,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        initial_timings: dict[str, float] | None = None,
    ) -> MotionRun:
        timings = StageTimings.from_dict(
            initial_timings, enabled=self.timing_enabled
        )
        hardware = self.document.get("hardware", {})
        try:
            with MotionExecutor(
                grbl_port=str(hardware.get("grbl_port", "/dev/diansai-grbl")),
                grbl_baudrate=int(hardware.get("grbl_baudrate", 115200)),
                grbl_step_idle_delay_ms=int(
                    hardware.get("grbl_step_idle_delay_ms", 255)
                ),
                grbl_set_work_origin_on_init=bool(
                    hardware.get("grbl_set_work_origin_on_init", True)
                ),
                grbl_release_on_close=bool(
                    hardware.get("grbl_release_on_close", True)
                ),
                servo_port=str(hardware.get("servo_port", "/dev/diansai-servo")),
                servo_baudrate=int(hardware.get("servo_baudrate", 115200)),
                servo_id=int(hardware.get("servo_id", 1)),
                mag_chip=str(hardware.get("mag_chip", "gpiochip1")),
                mag_line=int(hardware.get("mag_line", 7)),
            ) as executor:
                plan = executor.execute_solution(
                    solution,
                    self._transform(),
                    self.motion_config,
                    progress=progress,
                    timing_callback=timings.add,
                )
        except Exception as exc:
            self._log_timings(
                "motion_failed",
                timings,
                error=str(exc),
                piece_count=len(solution.get("pieces", [])),
            )
            raise
        self._log_timings(
            "motion_completed",
            timings,
            piece_count=len(plan),
        )
        return MotionRun(plan=plan, timings=timings.to_dict())
