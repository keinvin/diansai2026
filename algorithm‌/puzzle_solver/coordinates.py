"""Planar coordinate transforms between A4 millimetres and GRBL millimetres."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np


TransformModel = Literal["affine", "homography"]


@dataclass(frozen=True)
class A4ToGrblTransform:
    """Convert A4-plane points (mm) to the GRBL work coordinate system (mm).

    The matrix includes offset, axis reversal, and X/Y exchange.  Fit it from
    corresponding points obtained by jogging the magnet centre to known A4
    coordinates.
    """

    matrix: np.ndarray
    model: TransformModel = "affine"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("matrix must be a finite 3 x 3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("matrix must be invertible")
        if self.model not in ("affine", "homography"):
            raise ValueError("model must be 'affine' or 'homography'")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def fit(
        cls,
        a4_points_mm: Sequence[Sequence[float]],
        grbl_points_mm: Sequence[Sequence[float]],
        model: TransformModel = "affine",
        ransac_threshold_mm: float = 1.0,
    ) -> "A4ToGrblTransform":
        source = np.asarray(a4_points_mm, dtype=np.float32)
        target = np.asarray(grbl_points_mm, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape:
            raise ValueError("a4_points_mm and grbl_points_mm must be matching N x 2 arrays")
        required = 3 if model == "affine" else 4
        if len(source) < required:
            raise ValueError(f"{model} fitting needs at least {required} point pairs")

        if model == "affine":
            affine, _ = cv2.estimateAffine2D(
                source, target, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold_mm
            )
            if affine is None:
                raise ValueError("unable to fit affine A4-to-GRBL transform")
            matrix = np.vstack((affine, [0.0, 0.0, 1.0]))
        elif model == "homography":
            matrix, _ = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold_mm)
            if matrix is None:
                raise ValueError("unable to fit homography A4-to-GRBL transform")
        else:
            raise ValueError("model must be 'affine' or 'homography'")
        return cls(matrix, model)

    def to_grbl(self, a4_points_mm: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Map one or more ``[x, y]`` A4 points to GRBL work coordinates."""
        points = np.asarray(a4_points_mm, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(points, self.matrix)[0].astype(float)

    def to_a4(self, grbl_points_mm: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Map one or more GRBL work-coordinate points back to A4 millimetres."""
        points = np.asarray(grbl_points_mm, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(points, np.linalg.inv(self.matrix))[0].astype(float)

    def reprojection_error_mm(
        self, a4_points_mm: Sequence[Sequence[float]], grbl_points_mm: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Return per-point XY fitting error in millimetres."""
        expected = np.asarray(grbl_points_mm, dtype=float).reshape(-1, 2)
        actual = self.to_grbl(a4_points_mm)
        if expected.shape != actual.shape:
            raise ValueError("point arrays must have the same shape")
        return np.linalg.norm(actual - expected, axis=1)

    def to_dict(self) -> dict:
        return {"model": self.model, "a4_to_grbl_matrix": self.matrix.tolist()}

    @classmethod
    def from_dict(cls, document: dict) -> "A4ToGrblTransform":
        return cls(np.asarray(document["a4_to_grbl_matrix"], dtype=float), document.get("model", "affine"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "A4ToGrblTransform":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_initial_calibration(cls, path: str | Path) -> "A4ToGrblTransform":
        """Load the explicit one-point direction/scale prior from a sample file."""

        document = json.loads(Path(path).read_text(encoding="utf-8"))
        initial = document.get("initial_calibration")
        if not isinstance(initial, dict):
            raise ValueError("calibration sample file has no initial_calibration")
        matrix = initial.get("a4_to_grbl_affine_matrix")
        if matrix is None:
            raise ValueError("initial_calibration has no a4_to_grbl_affine_matrix")
        return cls(np.asarray(matrix, dtype=float), model="affine")
