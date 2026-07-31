#!/usr/bin/env python3
"""Locate a red A4 sheet in camera-image (screen) pixel coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCREEN_CORNER_NAMES = ("screen_top_left", "screen_top_right", "screen_bottom_right", "screen_bottom_left")
A4_CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
# The sheet is rotated relative to the camera.  Its short top edge is the
# right edge of the screen image: screen TR -> screen BR.
A4_FROM_SCREEN = (1, 2, 3, 0)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four image points as TL, TR, BR, BL."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def locate_red_a4(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red hue wraps around OpenCV's 0..179 hue range.
    low_red = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([12, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([165, 60, 45]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("未找到红色区域；请检查纸张是否入镜、光线及颜色阈值")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * 0.02:
        raise RuntimeError("红色区域过小，不能可靠定位 A4")

    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(polygon) != 4:
        # A perspective-distorted paper may have an imperfect contour; use its
        # minimum-area rectangle as a stable fallback and leave a visible note.
        polygon = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2)
    return order_corners(polygon), mask


def capture_frame(device: str) -> np.ndarray:
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头：{device}")
    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frame = None
        for _ in range(12):
            ok, frame = camera.read()
            if not ok:
                frame = None
        if frame is None:
            raise RuntimeError("摄像头读取失败")
        return frame
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="定位红色 A4 的摄像头像素坐标")
    parser.add_argument("--image", type=Path, help="使用已有图片；不填则读取摄像头")
    parser.add_argument("--device", default="/dev/video41")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "red_a4_corners.json")
    parser.add_argument("--preview", type=Path, default=DATA_DIR / "red_a4_detection.jpg")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image)) if args.image else capture_frame(args.device)
    if frame is None:
        raise SystemExit(f"无法读取图片：{args.image}")
    screen_corners, mask = locate_red_a4(frame)
    corners = screen_corners[list(A4_FROM_SCREEN)]

    preview = frame.copy()
    cv2.polylines(preview, [np.round(corners).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    for name, point in zip(A4_CORNER_NAMES, corners):
        xy = tuple(np.round(point).astype(int))
        cv2.circle(preview, xy, 8, (0, 255, 0), cv2.FILLED)
        cv2.putText(preview, name, (xy[0] + 10, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "image_size_px": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "corner_order": list(A4_CORNER_NAMES),
        "screen_corner_order": list(SCREEN_CORNER_NAMES),
        "a4_from_screen_indices": list(A4_FROM_SCREEN),
        "a4_corners_px": np.round(corners, 2).tolist(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cv2.imwrite(str(args.preview), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps({"a4_corners_px": np.round(corners, 2).tolist()}, ensure_ascii=False))
    print(f"坐标: {args.output}\n预览: {args.preview}")


if __name__ == "__main__":
    main()
