"""Fullscreen preview for the USB camera connected to the puzzle device."""

from __future__ import annotations

import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a live camera preview")
    parser.add_argument("--device", default="/dev/video41", help="V4L2 video device")
    parser.add_argument("--width", type=int, default=1920, help="Requested capture width")
    parser.add_argument("--height", type=int, default=1080, help="Requested capture height")
    parser.add_argument(
        "--windowed", action="store_true", help="Show a resizable window instead of fullscreen"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera device: {args.device}")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Camera preview"
    flags = cv2.WINDOW_NORMAL if args.windowed else cv2.WINDOW_FULLSCREEN
    cv2.namedWindow(window_name, flags)
    if not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame capture failed")
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
