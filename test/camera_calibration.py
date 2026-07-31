#!/usr/bin/env python3
"""Calibrate the USB camera from chessboard images saved in ``data/``.

The chessboard size is the number of *inner corners*, not the number of
black/white squares.  A calibration image must contain the complete inner
corner grid; images where the board is cut off are reported and skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_IMAGE_DIR / "camera_intrinsics.json"
DEFAULT_COMPARISON = DEFAULT_IMAGE_DIR / "camera_undistort_comparison.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用棋盘格图片标定摄像头")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="棋盘格图片目录")
    parser.add_argument("--cols", type=int, default=13, help="棋盘格横向内角点数，默认 10")
    parser.add_argument("--rows", type=int, default=9, help="棋盘格纵向内角点数，默认 7")
    parser.add_argument("--square-mm", type=float, default=18.0, help="单个格子的边长(mm)，默认 18")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="标定参数 JSON 输出路径")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON, help="标定前后对比图输出路径")
    return parser.parse_args()


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Find and refine one complete chessboard pattern."""
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    if found:
        return corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    return cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (170, 44), (0, 0, 0), thickness=-1)
    cv2.putText(result, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    if args.cols < 3 or args.rows < 3 or args.square_mm <= 0:
        raise SystemExit("--cols/--rows 至少为 3，--square-mm 必须大于 0")

    pattern_size = (args.cols, args.rows)
    image_paths = sorted(
        path for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for path in args.images.glob(extension)
    )
    if not image_paths:
        raise SystemExit(f"未在 {args.images} 找到图片")

    object_template = np.zeros((args.cols * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_mm
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[str] = []
    rejected: list[str] = []
    preview: np.ndarray | None = None
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            rejected.append(f"{path.name} (无法读取)")
            continue
        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        if (width, height) != image_size:
            rejected.append(f"{path.name} (分辨率不一致)")
            continue
        corners = find_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern_size)
        if corners is None:
            rejected.append(f"{path.name} (未找到完整 {args.cols}x{args.rows} 内角点)")
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        accepted.append(path.name)
        if preview is None:
            preview = image

    print(f"找到 {len(image_paths)} 张图片，接受 {len(accepted)} 张，跳过 {len(rejected)} 张")
    for item in rejected:
        print(f"跳过: {item}")
    if len(accepted) < 5 or image_size is None or preview is None:
        raise SystemExit("至少需要 5 张完整、清晰且分辨率一致的棋盘格图片")

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 1, image_size)
    undistorted = cv2.undistort(preview, camera_matrix, distortion, None, new_matrix)
    comparison = np.hstack((add_label(preview, "Before"), add_label(undistorted, "Undistorted")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "image_size_px": {"width": image_size[0], "height": image_size[1]},
                "chessboard": {"inner_corners": {"cols": args.cols, "rows": args.rows}, "square_mm": args.square_mm},
                "rms_reprojection_error_px": float(rms),
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": distortion.reshape(-1).tolist(),
                "optimal_new_camera_matrix": new_matrix.tolist(),
                "valid_roi_xywh": [int(value) for value in roi],
                "accepted_images": accepted,
                "rejected_images": rejected,
                "rotation_vectors": [value.reshape(-1).tolist() for value in rotations],
                "translation_vectors": [value.reshape(-1).tolist() for value in translations],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not cv2.imwrite(str(args.comparison), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise SystemExit(f"无法保存对比图：{args.comparison}")

    print(f"RMS 重投影误差: {rms:.4f} px")
    print(f"标定参数: {args.output}")
    print(f"前后对比图: {args.comparison}")


if __name__ == "__main__":
    main()
