"""Touch-friendly Qt page for A4-calibrated puzzle-piece recognition."""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# OpenCV bundles an incompatible Qt plugin directory. PySide6 must own the
# platform-plugin lookup before its first Qt import.
_pyside_plugins = Path(sysconfig.get_paths()["purelib"]) / "PySide6" / "Qt" / "plugins"
if _pyside_plugins.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_pyside_plugins)

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "data"
EXAMPLE_DIR = CAPTURE_DIR / "examples"
MOTION_CALIBRATION_PATH = PROJECT_ROOT / "data" / "a4_grbl_calibration_samples.json"
ALGORITHM_ROOT = next(PROJECT_ROOT.glob("algorithm*"))
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puzzle_solver.vision import (  # noqa: E402
    Calibration,
    VisionConfig,
    detect_pieces,
    draw_detection_overlay,
    extract_edge_profiles,
    extract_piece_features,
    render_assembled_image,
)
from puzzle_solver.solver import SolverConfig, solve_puzzle  # noqa: E402
from puzzle_solver.coordinates import A4ToGrblTransform  # noqa: E402
from motion.motion_exec import (  # noqa: E402
    MotionExecutor,
    PickPlaceConfig,
    build_pick_place_plan,
)


CONFIG_PATH = Path(__file__).with_name("vision_config.json")
RED_A4_CORNERS_PATH = PROJECT_ROOT / "data" / "red_a4_corners.json"
# Keep manual calibration identical to test/locate_red_a4.py.
SCREEN_CORNER_NAMES = ("屏幕左上", "屏幕右上（被遮挡）", "屏幕右下", "屏幕左下")
A4_FROM_SCREEN = (1, 2, 3, 0)  # A4 TL/TR/BR/BL = screen TR/BR/BL/TL


def _default_config() -> dict:
    return {
        "a4_corners_px": [],
        "a4_width_mm": 210.0,
        "a4_height_mm": 297.0,
        "a4_region": "upper",
        "use_a4_upper_half": True,
        "puzzle_search_enabled": True,
        "vision": asdict(VisionConfig()),
        "solver": asdict(SolverConfig()),
        "motion": asdict(PickPlaceConfig()),
    }


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return _default_config()
    try:
        document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_config()
    defaults = _default_config()
    defaults.update(document)
    if "a4_region" not in document:
        # Old configurations did not distinguish a lower-half ROI.
        defaults["a4_region"] = "upper"
    defaults["vision"] = {**_default_config()["vision"], **document.get("vision", {})}
    defaults["solver"] = {**_default_config()["solver"], **document.get("solver", {})}
    defaults["motion"] = {**_default_config()["motion"], **document.get("motion", {})}
    return defaults


def solver_config_from_dict(document: dict) -> SolverConfig:
    values = dict(document.get("solver", {}))
    for key in ("width_range", "height_range"):
        if key in values:
            values[key] = tuple(values[key])
    return SolverConfig(**values)


def place_solution_in_opposite_half(solution: dict, source_region: str, a4_width: float, a4_height: float) -> dict:
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


def draw_solution_overlay(image_bgr: np.ndarray, solution: dict, calibration: Calibration) -> np.ndarray:
    """Draw the assembled target rectangle in its physical A4 position."""
    overlay = image_bgr.copy()
    fill_layer = overlay.copy()
    palette = [(255, 130, 30), (40, 190, 70), (40, 80, 235), (190, 70, 190)]
    for index, piece in enumerate(solution["pieces"]):
        polygon_mm = np.asarray(piece["target_polygon_mm"], dtype=float)
        polygon_px = np.round(calibration.mm_to_pixels(polygon_mm)).astype(np.int32)
        colour = palette[index % len(palette)]
        cv2.fillPoly(fill_layer, [polygon_px], colour)
        cv2.polylines(overlay, [polygon_px], True, colour, 3, cv2.LINE_AA)
        centre = np.round(polygon_px.mean(axis=0)).astype(int)
        cv2.putText(
            overlay,
            f"target_{index}",
            (int(centre[0]) - 30, int(centre[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    overlay = cv2.addWeighted(fill_layer, 0.25, overlay, 0.75, 0.0)
    rectangle = solution["rectangle"]
    x0, y0 = rectangle["origin_mm"]
    x1 = x0 + rectangle["width_mm"]
    y1 = y0 + rectangle["height_mm"]
    rectangle_px = np.round(
        calibration.mm_to_pixels(np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    ).astype(np.int32)
    cv2.polylines(overlay, [rectangle_px], True, (255, 255, 0), 4, cv2.LINE_AA)
    return overlay


class RecognitionWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        frame: np.ndarray,
        calibration_document: dict,
        vision_config: VisionConfig,
        solver_config: SolverConfig,
        puzzle_search_enabled: bool,
    ) -> None:
        super().__init__()
        self._frame = frame
        self._calibration_document = calibration_document
        self._vision_config = vision_config
        self._solver_config = solver_config
        self._puzzle_search_enabled = puzzle_search_enabled

    @Slot()
    def run(self) -> None:
        try:
            calibration = Calibration.from_dict(
                self._calibration_document, self._frame.shape
            )
            result = detect_pieces(
                self._frame, calibration, self._vision_config
            )
            corrected_frame = calibration.undistort_image(self._frame)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        solution = None
        solve_error = None
        assembled_image = None
        if self._puzzle_search_enabled:
            try:
                edge_profiles = extract_edge_profiles(corrected_frame, result.pieces)
                piece_features = extract_piece_features(
                    corrected_frame, result.pieces, calibration
                )
                solution = solve_puzzle(
                    [piece.polygon_mm for piece in result.pieces],
                    [piece.id for piece in result.pieces],
                    target_origin_mm=(0.0, 0.0),
                    config=self._solver_config,
                    edge_profiles=edge_profiles,
                    piece_features=piece_features,
                )
                source_region = self._calibration_document.get("a4_region", "upper")
                solution = place_solution_in_opposite_half(
                    solution,
                    source_region,
                    float(self._calibration_document["a4_width_mm"]),
                    float(self._calibration_document["a4_height_mm"]),
                )
                pickup_by_id = {
                    piece.id: np.asarray(piece.pickup_point_mm, dtype=float)
                    for piece in result.pieces
                }
                for target_piece in solution["pieces"]:
                    pickup_source = pickup_by_id[target_piece["id"]]
                    rotation = np.asarray(target_piece["rotation_matrix"], dtype=float)
                    translation = np.asarray(target_piece["translation_mm"], dtype=float)
                    target_piece["pickup_source_mm"] = pickup_source.tolist()
                    target_piece["pickup_target_mm"] = (
                        rotation @ pickup_source + translation
                    ).tolist()
                assembled_image = render_assembled_image(
                    corrected_frame, result, solution, calibration
                )
            except (RuntimeError, ValueError) as exc:
                solve_error = str(exc)

        overlay = draw_detection_overlay(corrected_frame, result, calibration)
        if solution is not None:
            overlay = draw_solution_overlay(overlay, solution, calibration)
        self.completed.emit(
            {
                "result": result,
                "solution": solution,
                "solve_error": solve_error,
                "overlay": overlay,
                "assembled_image": assembled_image,
                "puzzle_search_enabled": self._puzzle_search_enabled,
            }
        )


class MotionWorker(QObject):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, solution: dict, config: PickPlaceConfig) -> None:
        super().__init__()
        self._solution = solution
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            transform = A4ToGrblTransform.load_initial_calibration(
                MOTION_CALIBRATION_PATH
            )
            with MotionExecutor() as executor:
                plan = executor.execute_solution(
                    self._solution,
                    transform,
                    self._config,
                    progress=self.progress.emit,
                )
            self.completed.emit(plan)
        except Exception as exc:
            self.failed.emit(str(exc))


def to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()


def save_recognition_example(
    frame_bgr: np.ndarray,
    region: str,
    directory: Path = EXAMPLE_DIR,
) -> Path:
    """Archive the untouched camera frame before running recognition."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        raise ValueError("待保存的相机画面不是有效的 BGR 图像")
    directory.mkdir(parents=True, exist_ok=True)
    safe_region = region if region in ("upper", "lower") else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"recognition_{timestamp}_{safe_region}.jpg"
    if not cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"无法写入识别样例：{path}")
    return path


class CameraView(QWidget):
    point_selected = Signal(QPointF)

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._target_rect = QRectF()
        self._corners: list[QPointF] = []
        self.setMinimumSize(900, 540)
        self.setCursor(Qt.CrossCursor)

    def set_image(self, frame: np.ndarray, corners: list[QPointF | None]) -> None:
        self._image = to_qimage(frame)
        self._corners = corners
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image is None or self._image.isNull():
            return QRectF()
        image_size = self._image.size()
        scale = min(self.width() / image_size.width(), self.height() / image_size.height())
        width = image_size.width() * scale
        height = image_size.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101318"))
        if self._image is None:
            return
        self._target_rect = self._image_rect()
        painter.drawImage(self._target_rect, self._image)
        if not self._corners:
            return

        scale_x = self._target_rect.width() / self._image.width()
        scale_y = self._target_rect.height() / self._image.height()
        points = [(index, QPointF(self._target_rect.left() + point.x() * scale_x,
                                  self._target_rect.top() + point.y() * scale_y))
                  for index, point in enumerate(self._corners) if point is not None]
        painter.setPen(QPen(QColor("#f4b942"), 3))
        for index, point in points:
            painter.drawEllipse(point, 7, 7)
            painter.drawText(point + QPointF(10, -10), SCREEN_CORNER_NAMES[index])

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton or self._image is None:
            return
        rect = self._target_rect
        if not rect.contains(event.position()):
            return
        x = (event.position().x() - rect.left()) * self._image.width() / rect.width()
        y = (event.position().y() - rect.top()) * self._image.height() / rect.height()
        self.point_selected.emit(QPointF(x, y))


class RecognitionWindow(QMainWindow):
    def __init__(self, device: str, fullscreen: bool) -> None:
        super().__init__()
        self._document = load_config()
        self._vision_config = VisionConfig(**self._document["vision"])
        self._solver_config = solver_config_from_dict(self._document)
        self._motion_config = PickPlaceConfig(**self._document["motion"])
        legacy_screen_order = "screen_a4_corners_px" not in self._document
        saved = self._document.get("screen_a4_corners_px", self._document.get("a4_corners_px", []))
        self._corners = [QPointF(*point) if point is not None else None for point in saved]
        if legacy_screen_order and len(self._corners) == 4 and all(point is not None for point in self._corners):
            self._document["screen_a4_corners_px"] = [[point.x(), point.y()] for point in self._corners]
            self._document["a4_corners_px"] = [
                [self._corners[index].x(), self._corners[index].y()] for index in A4_FROM_SCREEN
            ]
            self._document["a4_corner_indices"] = [0, 1, 2, 3]
            CONFIG_PATH.write_text(json.dumps(self._document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._latest_frame: np.ndarray | None = None
        self._latest_solution: dict | None = None
        self._frozen = False
        self._recognition_thread: QThread | None = None
        self._recognition_worker: RecognitionWorker | None = None
        self._recognition_example_path: Path | None = None
        self._recognition_example_error: str | None = None
        self._motion_thread: QThread | None = None
        self._motion_worker: MotionWorker | None = None

        self._camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            raise RuntimeError(f"无法打开摄像头：{device}")
        self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.setWindowTitle("拼图识别")
        self._build_ui()
        for shortcut in ("Escape", "Q", "Ctrl+Q"):
            QShortcut(QKeySequence(shortcut), self, activated=self.close)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._timer.start(33)
        if fullscreen:
            self.showFullScreen()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self._view = CameraView()
        self._view.setMinimumSize(640, 480)
        self._view.point_selected.connect(self._add_corner)
        layout.addWidget(self._view, 2)

        self._assembly_view = CameraView()
        self._assembly_view.setMinimumSize(320, 240)
        self._assembly_view.setCursor(Qt.ArrowCursor)
        self._assembly_view.setVisible(False)
        layout.addWidget(self._assembly_view, 1)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(300)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)

        title = QLabel("拼图识别")
        title.setObjectName("title")
        side_layout.addWidget(title)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setObjectName("status")
        side_layout.addWidget(self._status)

        self._recognize_button = QPushButton("识别当前画面")
        self._recognize_button.clicked.connect(self._recognize)
        side_layout.addWidget(self._recognize_button)

        self._execute_button = QPushButton("确认并执行拼图")
        self._execute_button.setObjectName("executeButton")
        self._execute_button.setEnabled(False)
        self._execute_button.clicked.connect(self._execute_solution)
        side_layout.addWidget(self._execute_button)

        self._region_selector = QComboBox()
        self._region_selector.addItem("识别 A4 上半", "upper")
        self._region_selector.addItem("识别 A4 下半", "lower")
        saved_region = self._document.get("a4_region", "upper")
        index = self._region_selector.findData(saved_region)
        self._region_selector.setCurrentIndex(index if index >= 0 else 0)
        self._region_selector.currentIndexChanged.connect(self._set_a4_region)
        side_layout.addWidget(self._region_selector)

        self._puzzle_search_checkbox = QCheckBox("启用拼图搜索")
        self._puzzle_search_checkbox.setChecked(
            bool(self._document.get("puzzle_search_enabled", True))
        )
        self._puzzle_search_checkbox.toggled.connect(
            self._set_puzzle_search_enabled
        )
        side_layout.addWidget(self._puzzle_search_checkbox)

        self._capture_button = QPushButton("截图保存（用于标定）")
        self._capture_button.clicked.connect(self._save_snapshot)
        side_layout.addWidget(self._capture_button)

        self._retake_button = QPushButton("继续预览")
        self._retake_button.clicked.connect(self._resume)
        side_layout.addWidget(self._retake_button)

        self._calibrate_button = QPushButton("重新标定 A4")
        self._calibrate_button.clicked.connect(self._start_calibration)
        side_layout.addWidget(self._calibrate_button)

        self._red_a4_button = QPushButton("载入红色 A4 自动定位")
        self._red_a4_button.clicked.connect(self._load_red_a4_corners)
        side_layout.addWidget(self._red_a4_button)

        self._skip_button = QPushButton("跳过当前角（三点标定）")
        self._skip_button.clicked.connect(self._skip_corner)
        side_layout.addWidget(self._skip_button)

        self._clear_button = QPushButton("清除标定")
        self._clear_button.clicked.connect(self._clear_calibration)
        side_layout.addWidget(self._clear_button)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setObjectName("details")
        side_layout.addWidget(self._details)
        side_layout.addStretch(1)

        exit_button = QPushButton("退出")
        exit_button.setObjectName("exitButton")
        exit_button.clicked.connect(self.close)
        side_layout.addWidget(exit_button)
        layout.addWidget(side)

        self.setStyleSheet(
            "QMainWindow { background: #171a20; color: #eef2f7; }"
            "#sidebar { background: #242933; border: 1px solid #394150; }"
            "#title { font-size: 26px; font-weight: 700; color: #ffffff; }"
            "#status { font-size: 16px; color: #d3dae5; min-height: 70px; }"
            "#details { font-size: 15px; color: #b8c2d1; padding-top: 12px; }"
            "QPushButton { background: #2d6a9f; border: 0; padding: 13px 12px;"
            " color: white; font-size: 16px; }"
            "QPushButton:disabled { background: #4a505a; color: #c0c4cb; }"
            "QPushButton:pressed { background: #1e4d77; }"
            "QCheckBox { color: #d3dae5; font-size: 16px; padding: 6px 2px; }"
            "#executeButton { background: #a66616; font-weight: 700; }"
            "#executeButton:pressed { background: #75480d; }"
            "#exitButton { background: #9b3a3a; }"
            "#exitButton:pressed { background: #702727; }"
        )
        self._update_status()

    def _read_frame(self) -> None:
        if self._frozen:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._status.setText("摄像头画面读取失败")
            return
        self._latest_frame = frame
        self._view.set_image(frame, self._corners)

    def _start_calibration(self) -> None:
        if self._recognition_thread is not None:
            return
        self._corners = []
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._frozen = False
        self._update_status()

    def _set_a4_region(self, _index: int) -> None:
        region = self._region_selector.currentData()
        if region not in ("upper", "lower"):
            return
        self._document["a4_region"] = region
        # Keep the legacy key for older consumers of vision_config.json.
        self._document["use_a4_upper_half"] = region == "upper"
        CONFIG_PATH.write_text(json.dumps(self._document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._frozen = False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._update_status()

    def _set_puzzle_search_enabled(self, enabled: bool) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._document["puzzle_search_enabled"] = bool(enabled)
        CONFIG_PATH.write_text(
            json.dumps(self._document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._frozen = False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._update_status()

    def _clear_calibration(self) -> None:
        if self._recognition_thread is not None:
            return
        self._corners = []
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._document["a4_corners_px"] = []
        self._document["a4_corner_indices"] = []
        self._document["screen_a4_corners_px"] = []
        CONFIG_PATH.write_text(json.dumps(self._document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._update_status()

    def _load_red_a4_corners(self) -> None:
        """Use the semantic A4 corner order generated by locate_red_a4.py."""
        if self._recognition_thread is not None:
            return
        try:
            result = json.loads(RED_A4_CORNERS_PATH.read_text(encoding="utf-8"))
            a4_corners = result["a4_corners_px"]
            if np.asarray(a4_corners, dtype=float).shape != (4, 2):
                raise ValueError("a4_corners_px 不是四个坐标")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self._status.setText(f"无法读取红色 A4 定位结果：{error}")
            return

        screen_corners: list[list[float] | None] = [None] * 4
        for a4_index, screen_index in enumerate(A4_FROM_SCREEN):
            screen_corners[screen_index] = a4_corners[a4_index]
        self._corners = [QPointF(*point) for point in screen_corners if point is not None]
        self._document["screen_a4_corners_px"] = screen_corners
        self._document["a4_corners_px"] = a4_corners
        self._document["a4_corner_indices"] = [0, 1, 2, 3]
        CONFIG_PATH.write_text(json.dumps(self._document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._frozen = False
        self._details.setText("已载入 red_a4_corners.json 的自动定位结果。")
        self._update_status()

    def _add_corner(self, point: QPointF) -> None:
        if self._frozen or len(self._corners) >= 4:
            return
        self._corners.append(point)
        self._save_corners_if_complete()
        self._update_status()

    def _skip_corner(self) -> None:
        if not self._frozen and len(self._corners) < 4:
            self._corners.append(None)
            self._save_corners_if_complete()
            self._update_status()

    def _save_corners_if_complete(self) -> None:
        if len(self._corners) != 4 or sum(point is not None for point in self._corners) < 3:
            return
        self._document["screen_a4_corners_px"] = [
            [round(point.x(), 2), round(point.y(), 2)] if point is not None else None
            for point in self._corners
        ]
        visible = [(a4_index, self._corners[screen_index]) for a4_index, screen_index in enumerate(A4_FROM_SCREEN) if self._corners[screen_index] is not None]
        self._document["a4_corners_px"] = [[round(point.x(), 2), round(point.y(), 2)] for _, point in visible]
        self._document["a4_corner_indices"] = [index for index, _ in visible]
        CONFIG_PATH.write_text(json.dumps(self._document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _update_status(self) -> None:
        complete = len(self._corners) == 4 and sum(point is not None for point in self._corners) >= 3
        if not complete:
            index = len(self._corners)
            self._status.setText(
                f"请点选 {SCREEN_CORNER_NAMES[index]}（{index}/4）"
            )
            self._recognize_button.setEnabled(False)
        else:
            region_name = "上半" if self._document.get("a4_region", "upper") == "upper" else "下半"
            self._status.setText(f"A4 标定完成，可以识别当前画面（当前：{region_name}）")
            self._recognize_button.setEnabled(True)
        if self._latest_frame is not None:
            self._view.set_image(self._latest_frame, self._corners)

    def _recognize(self) -> None:
        if (
            self._latest_frame is None
            or len(self._corners) != 4
            or sum(point is not None for point in self._corners) < 3
            or self._recognition_thread is not None
        ):
            return
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._assembly_view.setVisible(False)
        frame = self._latest_frame.copy()
        self._recognition_example_path = None
        self._recognition_example_error = None
        try:
            self._recognition_example_path = save_recognition_example(
                frame, self._document.get("a4_region", "upper")
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._recognition_example_error = str(error)
        self._frozen = False
        self._recognize_button.setEnabled(False)
        self._region_selector.setEnabled(False)
        self._puzzle_search_checkbox.setEnabled(False)
        puzzle_search_enabled = bool(
            self._document.get("puzzle_search_enabled", True)
        )
        if puzzle_search_enabled:
            self._status.setText("正在识别碎片并搜索拼接方案……")
        else:
            self._status.setText("正在识别碎片（拼图搜索已关闭）……")
        calibration_document = {
            "a4_corners_px": self._document["a4_corners_px"],
            "a4_corner_indices": self._document.get("a4_corner_indices"),
            "a4_width_mm": self._document["a4_width_mm"],
            "a4_height_mm": self._document["a4_height_mm"],
            "a4_region": self._document.get("a4_region", "upper"),
            "use_a4_upper_half": self._document.get("use_a4_upper_half", True),
        }
        thread = QThread(self)
        worker = RecognitionWorker(
            frame,
            calibration_document,
            self._vision_config,
            self._solver_config,
            puzzle_search_enabled,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._recognition_completed)
        worker.failed.connect(self._recognition_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._recognition_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._recognition_thread = thread
        self._recognition_worker = worker
        thread.start()

    @Slot(object)
    def _recognition_completed(self, payload: dict) -> None:
        result = payload["result"]
        solution = payload["solution"]
        solve_error = payload["solve_error"]
        puzzle_search_enabled = bool(payload["puzzle_search_enabled"])
        self._frozen = True
        self._latest_solution = solution if puzzle_search_enabled else None
        self._execute_button.setEnabled(
            puzzle_search_enabled and solution is not None
        )
        self._view.set_image(payload["overlay"], [])
        assembled_image = payload.get("assembled_image")
        if assembled_image is not None:
            self._assembly_view.set_image(assembled_image, [])
            self._assembly_view.setVisible(True)
        else:
            self._assembly_view.setVisible(False)
        if not puzzle_search_enabled:
            self._status.setText(
                f"识别到 {len(result.pieces)} 块拼图（拼图搜索已关闭）"
            )
        elif solution is None:
            self._status.setText(f"识别到 {len(result.pieces)} 块拼图，但矩形拼接失败")
        else:
            rectangle = solution["rectangle"]
            target_region = "下半" if self._document.get("a4_region", "upper") == "upper" else "上半"
            self._status.setText(
                f"识别到 {len(result.pieces)} 块，已在 A4 {target_region}显示 "
                f"{rectangle['width_mm']:.0f}×{rectangle['height_mm']:.0f} mm 矩形"
            )
        detail_lines = [
            f"{piece.id}  面积 {piece.area_mm2:.0f} mm2\n吸取点 ({piece.pickup_point_mm[0]:.1f}, {piece.pickup_point_mm[1]:.1f}) mm"
            for piece in result.pieces
        ]
        if self._recognition_example_path is not None:
            detail_lines.append(
                f"识别原图：data/examples/{self._recognition_example_path.name}"
            )
        elif self._recognition_example_error:
            detail_lines.append(
                f"识别原图保存失败：{self._recognition_example_error}"
            )
        if not puzzle_search_enabled:
            detail_lines.append("拼图搜索已关闭：仅显示碎片轮廓和顶点。")
        elif solution is not None:
            pattern_evidence = solution["metrics"]["pattern_evidence"]
            if pattern_evidence >= solution["config"]["min_pattern_evidence"]:
                pattern_line = "花纹不匹配度 {:.1%}（证据 {:.1%}）".format(
                    solution["metrics"]["pattern_mismatch"], pattern_evidence
                )
            else:
                pattern_line = "花纹：画面纹理不足，未作为有效判据"
            detail_lines.append(
                "拼接误差：孔洞 {:.2%}，搜索重叠 {:.2%}\n"
                "安全间隙 {:.1f} mm，最终重叠 {:.2f} mm2，邻边顶点最大距离 {:.1f} mm\n{}".format(
                    solution["metrics"]["hole_ratio"],
                    solution["metrics"]["overlap_ratio"],
                    solution["metrics"]["applied_placement_gap_mm"],
                    solution["metrics"]["final_overlap_area_mm2"],
                    solution["metrics"]["max_adjacent_vertex_distance_mm"],
                    pattern_line,
                )
            )
        elif solve_error:
            detail_lines.append(f"拼接失败：{solve_error}")
        self._details.setText("\n".join(detail_lines))

    @Slot(str)
    def _recognition_failed(self, error: str) -> None:
        suffix = (
            f"；样例保存失败：{self._recognition_example_error}"
            if self._recognition_example_error
            else ""
        )
        self._status.setText(f"识别失败：{error}{suffix}")
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._frozen = False

    @Slot()
    def _recognition_thread_finished(self) -> None:
        self._recognition_thread = None
        self._recognition_worker = None
        self._region_selector.setEnabled(True)
        self._puzzle_search_checkbox.setEnabled(True)
        self._recognize_button.setEnabled(True)

    def _execute_solution(self) -> None:
        if self._latest_solution is None or self._motion_thread is not None:
            return
        try:
            transform = A4ToGrblTransform.load_initial_calibration(
                MOTION_CALIBRATION_PATH
            )
            plan = build_pick_place_plan(
                self._latest_solution, transform, self._motion_config
            )
        except Exception as exc:
            self._status.setText(f"无法生成抓取计划：{exc}")
            return

        preview_lines = [
            "将使用单点方向/比例先验执行，尚不是三点精确标定。",
            f"共 {len(plan)} 片，Z 下压绝对坐标 {self._motion_config.z_down_mm:g} mm。",
            f"舵机方向系数 {self._motion_config.servo_direction:+g}。",
        ]
        for index, step in enumerate(plan, start=1):
            source = step["pickup_source_grbl_mm"]
            target = step["pickup_target_grbl_mm"]
            preview_lines.append(
                f"{index}. {step['id']}: GRBL ({source[0]:.1f}, {source[1]:.1f})"
                f" → ({target[0]:.1f}, {target[1]:.1f})，舵机相对旋转 {step['servo_delta_deg']:+.1f}°"
            )
        answer = QMessageBox.warning(
            self,
            "确认执行实际运动",
            "\n".join(preview_lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self._execute_button.setEnabled(False)
        self._recognize_button.setEnabled(False)
        self._region_selector.setEnabled(False)
        self._puzzle_search_checkbox.setEnabled(False)
        self._status.setText("正在初始化 GRBL、舵机和电磁铁……")
        thread = QThread(self)
        worker = MotionWorker(self._latest_solution, self._motion_config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._motion_progress)
        worker.completed.connect(self._motion_completed)
        worker.failed.connect(self._motion_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._motion_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._motion_thread = thread
        self._motion_worker = worker
        thread.start()

    @Slot(int, int, str)
    def _motion_progress(self, index: int, total: int, message: str) -> None:
        self._status.setText(f"{message}（{index}/{total}）")

    @Slot(object)
    def _motion_completed(self, plan: list[dict]) -> None:
        self._status.setText(f"拼图执行完成，共放置 {len(plan)} 片，机械机构已回到工作零点。")
        self._details.setText("实际动作已完成；继续识别前请确认所有碎片的位置。")
        self._latest_solution = None

    @Slot(str)
    def _motion_failed(self, error: str) -> None:
        self._status.setText(f"拼图动作失败：{error}")
        self._details.setText("已尝试执行 Z 收回、磁铁释放、舵机复位和 XY 回零。请先检查机构状态。")
        self._latest_solution = None

    @Slot()
    def _motion_thread_finished(self) -> None:
        self._motion_thread = None
        self._motion_worker = None
        self._region_selector.setEnabled(True)
        self._puzzle_search_checkbox.setEnabled(True)
        self._recognize_button.setEnabled(True)

    def _save_snapshot(self) -> None:
        """Save the raw camera frame, without recognition overlays, for calibration."""
        if self._latest_frame is None:
            self._status.setText("尚未收到摄像头画面，无法截图")
            return

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        filename = "camera_calibration_{}.jpg".format(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        path = CAPTURE_DIR / filename
        saved = cv2.imwrite(str(path), self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not saved:
            self._status.setText("截图保存失败")
            return
        self._status.setText(f"已保存原始截图：data/{filename}")
        self._details.setText("用于相机标定的原始图像，不含识别标注。")

    def _resume(self) -> None:
        if self._recognition_thread is not None:
            return
        self._frozen = False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._assembly_view.setVisible(False)
        self._update_status()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._recognition_thread is not None and self._recognition_thread.isRunning():
            self._status.setText("识别计算尚未结束，请稍候再关闭窗口。")
            event.ignore()
            return
        if self._motion_thread is not None and self._motion_thread.isRunning():
            self._status.setText("机械运动尚未结束，当前不能关闭窗口。")
            event.ignore()
            return
        self._timer.stop()
        self._camera.release()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="Puzzle-piece recognition touchscreen UI")
    parser.add_argument("--device", default="/dev/video41", help="V4L2 camera device")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    application = QApplication(sys.argv)
    window = RecognitionWindow(args.device, args.fullscreen)
    if not args.fullscreen:
        window.resize(1440, 850)
        window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
