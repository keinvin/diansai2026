"""Touch-friendly Qt page for A4-calibrated puzzle-piece recognition."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

# OpenCV bundles an incompatible Qt plugin directory. PySide6 must own the
# platform-plugin lookup before its first Qt import.
_pyside_plugins = Path(sysconfig.get_paths()["purelib"]) / "PySide6" / "Qt" / "plugins"
if _pyside_plugins.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_pyside_plugins)

from PySide6.QtCore import (
    QObject,
    QPointF,
    QProcess,
    QRectF,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "data"
EXAMPLE_DIR = CAPTURE_DIR / "examples"
ALGORITHM_ROOT = next(PROJECT_ROOT.glob("algorithm*"))
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puzzle_solver.vision import (  # noqa: E402
    Calibration,
    draw_detection_overlay,
)
from main.config import load_config, save_config  # noqa: E402
from main.pipeline import MainPipeline  # noqa: E402
from main.timing import StageTimings  # noqa: E402
from test.locate_red_a4 import locate_red_a4  # noqa: E402


RED_A4_CORNERS_PATH = PROJECT_ROOT / "data" / "red_a4_corners.json"
FINISHED_SOUND_PATH = Path(__file__).with_name("finished.wav")
HARDWARE_POLL_INTERVAL_MS = 500
# Keep manual calibration identical to test/locate_red_a4.py.
SCREEN_CORNER_NAMES = ("屏幕左上", "屏幕右上（被遮挡）", "屏幕右下", "屏幕左下")
A4_FROM_SCREEN = (1, 2, 3, 0)  # A4 TL/TR/BR/BL = screen TR/BR/BL/TL


def _warp_piece_into_target(
    source_image: np.ndarray,
    source_piece: object,
    target_polygon_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project one detected source piece into its solved target polygon."""

    source_polygon_px = np.asarray(source_piece.polygon_px, dtype=np.float32)
    target_polygon_px = np.asarray(target_polygon_px, dtype=np.float32)
    if source_polygon_px.shape != target_polygon_px.shape or len(source_polygon_px) < 3:
        return None
    source_mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(
        source_mask,
        [np.round(source_piece.contour_px).astype(np.int32)],
        255,
    )
    size = (source_image.shape[1], source_image.shape[0])
    if len(source_polygon_px) == 3:
        transform = cv2.getAffineTransform(source_polygon_px, target_polygon_px)
        warped_image = cv2.warpAffine(source_image, transform, size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(source_mask, transform, size, flags=cv2.INTER_NEAREST)
    else:
        transform, _ = cv2.findHomography(source_polygon_px, target_polygon_px, method=0)
        if transform is None:
            return None
        warped_image = cv2.warpPerspective(source_image, transform, size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpPerspective(source_mask, transform, size, flags=cv2.INTER_NEAREST)
    return warped_image, warped_mask


def draw_solution_overlay(
    image_bgr: np.ndarray,
    solution: dict,
    calibration: Calibration,
    *,
    source_image: np.ndarray | None = None,
    source_pieces: list[object] | None = None,
    show_piece_images: bool = False,
) -> np.ndarray:
    """Draw the solved target, optionally with each source piece projected into it."""

    overlay = image_bgr.copy()
    fill_layer = overlay.copy()
    palette = [(255, 130, 30), (40, 190, 70), (40, 80, 235), (190, 70, 190)]
    source_by_id = {
        piece.id: piece for piece in (source_pieces or [])
    }
    for index, piece in enumerate(solution["pieces"]):
        polygon_mm = np.asarray(piece["target_polygon_mm"], dtype=float)
        polygon_px = np.round(calibration.mm_to_pixels(polygon_mm)).astype(np.int32)
        colour = palette[index % len(palette)]
        source_piece = source_by_id.get(piece["id"])
        rendered_piece = (
            _warp_piece_into_target(source_image, source_piece, polygon_px)
            if show_piece_images and source_image is not None and source_piece is not None
            else None
        )
        if rendered_piece is None:
            cv2.fillPoly(fill_layer, [polygon_px], colour)
        else:
            warped_image, warped_mask = rendered_piece
            overlay[warped_mask != 0] = warped_image[warped_mask != 0]
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
    if not show_piece_images:
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
        config_document: dict,
        puzzle_search_enabled: bool,
        transport_only: bool,
        use_piece_features: bool,
        show_piece_images: bool,
        show_rejected_contours: bool,
    ) -> None:
        super().__init__()
        self._frame = frame
        self._config_document = config_document
        self._puzzle_search_enabled = puzzle_search_enabled
        self._transport_only = transport_only
        self._use_piece_features = use_piece_features
        self._show_piece_images = show_piece_images
        self._show_rejected_contours = show_rejected_contours

    @Slot()
    def run(self) -> None:
        try:
            run = MainPipeline(self._config_document).recognize(
                self._frame,
                puzzle_search_enabled=self._puzzle_search_enabled,
                transport_only=self._transport_only,
                use_piece_features=self._use_piece_features,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        overlay = draw_detection_overlay(
            run.corrected_frame,
            run.result,
            run.calibration,
            show_rejected_contours=self._show_rejected_contours,
        )
        if run.solution is not None:
            overlay = draw_solution_overlay(
                overlay,
                run.solution,
                run.calibration,
                source_image=run.corrected_frame,
                source_pieces=run.result.pieces,
                show_piece_images=self._show_piece_images,
            )
        self.completed.emit(
            {
                "result": run.result,
                "solution": run.solution,
                "solve_error": run.solve_error,
                "overlay": overlay,
                "corrected_frame": run.corrected_frame,
                "calibration": run.calibration,
                "puzzle_search_enabled": run.puzzle_search_enabled,
                "transport_only": run.transport_only,
                "timings": run.timings,
            }
        )


class MotionWorker(QObject):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        solution: dict,
        config_document: dict,
        initial_timings: dict[str, float],
    ) -> None:
        super().__init__()
        self._solution = solution
        self._config_document = config_document
        self._initial_timings = initial_timings

    @Slot()
    def run(self) -> None:
        try:
            run = MainPipeline(self._config_document).execute_solution(
                self._solution,
                progress=self.progress.emit,
                initial_timings=self._initial_timings,
            )
            self.completed.emit({"plan": run.plan, "timings": run.timings})
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


class HardwareWaitDialog(QDialog):
    """Block startup until every USB device needed by the main UI exists."""

    def __init__(self, devices: list[tuple[str, str]]) -> None:
        super().__init__()
        self._devices = devices
        self._device_labels: list[QLabel] = []
        self._ready = False
        self.setWindowTitle("设备连接检查")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet(
            "QDialog { background: #171a20; color: #eef2f7; }"
            "#hardwareTitle { font-size: 30px; font-weight: 700; color: #ffffff; }"
            "#hardwareSummary { font-size: 20px; color: #c7d0dc; }"
            "#hardwareDeviceState { background: #242933; border: 1px solid #394150;"
            " font-size: 20px; padding: 14px; }"
            "QPushButton { background: #9b3a3a; color: white; border: 0;"
            " font-size: 20px; padding: 12px 20px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(14)
        title = QLabel("正在检查 USB 设备")
        title.setObjectName("hardwareTitle")
        layout.addWidget(title)
        self._summary = QLabel()
        self._summary.setObjectName("hardwareSummary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        for _label, _path in self._devices:
            state = QLabel()
            state.setObjectName("hardwareDeviceState")
            state.setWordWrap(True)
            self._device_labels.append(state)
            layout.addWidget(state)
        layout.addStretch(1)
        exit_button = QPushButton("退出程序")
        exit_button.clicked.connect(self.reject)
        layout.addWidget(exit_button, 0, Qt.AlignRight)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(HARDWARE_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._refresh)
        self._refresh()
        self._poll_timer.start()

    def _refresh(self) -> None:
        missing: list[str] = []
        for (label, path), state in zip(self._devices, self._device_labels):
            connected = Path(path).exists()
            state.setText(
                f"{'已连接' if connected else '未检测到'}  {label}\n{path}"
            )
            state.setStyleSheet(
                "color: #51d59b;" if connected else "color: #ff8c8c;"
            )
            if not connected:
                missing.append(label)
        if missing:
            self._summary.setText("正在等待：" + "、".join(missing))
            return
        self._summary.setText("设备已就绪，正在进入主界面…")
        if not self._ready:
            self._ready = True
            self._poll_timer.stop()
            QTimer.singleShot(250, self.accept)

    def done(self, result: int) -> None:  # type: ignore[override]
        self._poll_timer.stop()
        super().done(result)


def wait_for_required_hardware(camera_device: str) -> bool:
    """Show a live device monitor before opening the camera or main window."""

    hardware = load_config().get("hardware", {})
    devices = [
        ("摄像头", camera_device),
        ("GRBL 控制器", str(hardware.get("grbl_port", "/dev/diansai-grbl"))),
        ("舵机总线", str(hardware.get("servo_port", "/dev/diansai-servo"))),
    ]
    return HardwareWaitDialog(devices).exec() == QDialog.DialogCode.Accepted


class CameraView(QWidget):
    point_selected = Signal(QPointF)

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._target_rect = QRectF()
        self._corners: list[QPointF] = []
        self.setMinimumSize(640, 360)
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
    def _save_document_fields(self, *keys: str) -> None:
        """Persist UI-owned fields without overwriting newer hardware settings."""
        latest = load_config()
        for key in keys:
            latest[key] = self._document[key]
        save_config(latest)
        self._document = latest

    def __init__(self, device: str, fullscreen: bool) -> None:
        super().__init__()
        self._document = load_config()
        self._pipeline = MainPipeline(self._document)
        self._latest_timings = StageTimings(
            enabled=self._pipeline.timing_enabled
        ).to_dict()
        legacy_screen_order = "screen_a4_corners_px" not in self._document
        saved = self._document.get("screen_a4_corners_px", self._document.get("a4_corners_px", []))
        self._corners = [QPointF(*point) if point is not None else None for point in saved]
        if legacy_screen_order and len(self._corners) == 4 and all(point is not None for point in self._corners):
            self._document["screen_a4_corners_px"] = [[point.x(), point.y()] for point in self._corners]
            self._document["a4_corners_px"] = [
                [self._corners[index].x(), self._corners[index].y()] for index in A4_FROM_SCREEN
            ]
            self._document["a4_corner_indices"] = [0, 1, 2, 3]
            self._save_document_fields(
                "screen_a4_corners_px", "a4_corners_px", "a4_corner_indices"
            )
        self._latest_frame: np.ndarray | None = None
        self._latest_solution: dict | None = None
        self._last_recognition_payload: dict | None = None
        self._frozen = False
        self._recognition_thread: QThread | None = None
        self._recognition_worker: RecognitionWorker | None = None
        self._recognition_example_path: Path | None = None
        self._recognition_example_error: str | None = None
        self._motion_thread: QThread | None = None
        self._motion_worker: MotionWorker | None = None
        self._task_mode: str | None = None
        self._one_click_execute_pending = False
        self._one_click_started_at: float | None = None
        self._finished_sound_process = QProcess(self)

        self._camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            raise RuntimeError(f"无法打开摄像头：{device}")
        self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self._pages = QStackedWidget()
        root_layout.addWidget(self._pages)

        menu = QWidget()
        menu_layout = QVBoxLayout(menu)
        menu_layout.setContentsMargins(56, 44, 56, 44)
        menu_layout.setSpacing(22)
        menu_title = QLabel("竞赛任务")
        menu_title.setObjectName("menuTitle")
        menu_title.setAlignment(Qt.AlignCenter)
        menu_layout.addWidget(menu_title)
        menu_hint = QLabel("选择题目后开始识别与执行")
        menu_hint.setObjectName("menuHint")
        menu_hint.setAlignment(Qt.AlignCenter)
        menu_layout.addWidget(menu_hint)
        menu_exit_button = QPushButton("退出")
        menu_exit_button.setObjectName("exitButton")
        menu_exit_button.clicked.connect(self.close)
        menu_layout.addWidget(menu_exit_button, 0, Qt.AlignRight)
        task_grid = QGridLayout()
        task_grid.setHorizontalSpacing(22)
        task_grid.setVerticalSpacing(22)
        task_buttons = (
            ("第一题", "识别碎片后直接搬运到 A4 另一半", "transport"),
            ("第二题", "识别白色碎片并搜索拼接方案", "white_puzzle"),
            ("第三题", "识别扑克牌并搜索拼接方案", "card_puzzle"),
            ("配置页面", "A4 定位、坐标校准与舵机校准", "configuration"),
        )
        for index, (title, hint, mode) in enumerate(task_buttons):
            button = QPushButton(f"{title}\n{hint}")
            button.setObjectName("taskButton")
            button.setMinimumHeight(180)
            button.clicked.connect(lambda _checked=False, value=mode: self._select_task(value))
            task_grid.addWidget(button, index // 2, index % 2)
        menu_layout.addLayout(task_grid, 1)
        self._pages.addWidget(menu)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(18, 14, 18, 18)
        workspace_layout.setSpacing(12)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setObjectName("status")
        self._status.setMaximumHeight(58)

        self._one_click_elapsed = QLabel("运行计时 0 s")
        self._one_click_elapsed.setObjectName("runTimer")
        self._one_click_elapsed.setAlignment(Qt.AlignCenter)
        self._one_click_elapsed.setVisible(False)
        self._one_click_timer = QTimer(self)
        self._one_click_timer.setInterval(1000)
        self._one_click_timer.timeout.connect(self._update_one_click_elapsed)

        header = QHBoxLayout()
        back_button = QPushButton("返回题目")
        back_button.setObjectName("backButton")
        back_button.clicked.connect(self._show_task_menu)
        header.addWidget(back_button)
        self._workspace_title = QLabel()
        self._workspace_title.setObjectName("workspaceTitle")
        header.addWidget(self._workspace_title, 1)
        header.addWidget(self._status, 2)
        header.addWidget(self._one_click_elapsed)
        exit_button = QPushButton("退出")
        exit_button.setObjectName("exitButton")
        exit_button.clicked.connect(self.close)
        header.addWidget(exit_button)
        workspace_layout.addLayout(header)

        layout = QHBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self._view = CameraView()
        self._view.point_selected.connect(self._add_corner)
        layout.addWidget(self._view, 1)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(360)
        self._configuration_sidebar = side
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)

        self._task_action_panel = QFrame()
        self._task_action_panel.setObjectName("taskActionPanel")
        task_action_layout = QHBoxLayout(self._task_action_panel)
        task_action_layout.setContentsMargins(0, 0, 0, 0)
        task_action_layout.setSpacing(14)

        self._one_click_button = QPushButton("一键启动")
        self._one_click_button.setObjectName("oneClickButton")
        self._one_click_button.clicked.connect(self._one_click_start)
        task_action_layout.addWidget(self._one_click_button, 4)

        self._recognize_button = QPushButton("仅识别并生成方案")
        self._recognize_button.setObjectName("taskActionButton")
        self._recognize_button.clicked.connect(self._recognize_manually)
        task_action_layout.addWidget(self._recognize_button, 3)

        self._execute_button = QPushButton("确认并执行拼图")
        self._execute_button.setObjectName("executeButton")
        self._execute_button.setEnabled(False)
        self._execute_button.clicked.connect(self._execute_solution)
        task_action_layout.addWidget(self._execute_button, 3)

        self._region_selector = QComboBox()
        self._region_selector.addItem("识别 A4 上半", "upper")
        self._region_selector.addItem("识别 A4 下半", "lower")
        saved_region = self._document.get("a4_region", "upper")
        index = self._region_selector.findData(saved_region)
        self._region_selector.setCurrentIndex(index if index >= 0 else 0)
        self._region_selector.currentIndexChanged.connect(self._set_a4_region)
        task_action_layout.addWidget(self._region_selector, 2)

        self._puzzle_search_checkbox = QCheckBox("启用拼图搜索")
        self._puzzle_search_checkbox.setChecked(
            bool(self._document.get("puzzle_search_enabled", True))
        )
        self._puzzle_search_checkbox.toggled.connect(
            self._set_puzzle_search_enabled
        )
        self._puzzle_search_checkbox.setVisible(False)
        side_layout.addWidget(self._puzzle_search_checkbox)

        self._capture_button = QPushButton("截图保存（用于标定）")
        self._capture_button.clicked.connect(self._save_snapshot)
        side_layout.addWidget(self._capture_button)

        self._retake_button = QPushButton("继续预览")
        self._retake_button.setObjectName("taskActionButton")
        self._retake_button.clicked.connect(self._resume)
        task_action_layout.addWidget(self._retake_button, 2)

        self._calibrate_button = QPushButton("重新标定 A4")
        self._calibrate_button.clicked.connect(self._start_calibration)
        side_layout.addWidget(self._calibrate_button)

        self._red_a4_button = QPushButton("载入红色 A4 自动定位")
        self._red_a4_button.clicked.connect(self._load_red_a4_corners)
        side_layout.addWidget(self._red_a4_button)

        self._locate_red_a4_button = QPushButton("从当前画面定位红色 A4")
        self._locate_red_a4_button.clicked.connect(self._locate_red_a4_current_frame)
        side_layout.addWidget(self._locate_red_a4_button)

        self._skip_button = QPushButton("跳过当前角（三点标定）")
        self._skip_button.clicked.connect(self._skip_corner)
        side_layout.addWidget(self._skip_button)

        self._clear_button = QPushButton("清除标定")
        self._clear_button.clicked.connect(self._clear_calibration)
        side_layout.addWidget(self._clear_button)

        self._grbl_calibration_button = QPushButton("打开 A4 ↔ GRBL 校准")
        self._grbl_calibration_button.clicked.connect(
            lambda: self._launch_tool("viz/grbl_a4_calibration_app.py")
        )
        side_layout.addWidget(self._grbl_calibration_button)

        self._servo_calibration_button = QPushButton("打开舵机校准")
        self._servo_calibration_button.clicked.connect(
            lambda: self._launch_tool("viz/servo_calibration_app.py")
        )
        side_layout.addWidget(self._servo_calibration_button)

        self._show_piece_images_checkbox = QCheckBox("拼接区显示原始图案")
        self._show_piece_images_checkbox.setChecked(
            bool(self._document.get("show_solution_piece_images", False))
        )
        self._show_piece_images_checkbox.toggled.connect(
            self._set_show_piece_images
        )
        side_layout.addWidget(self._show_piece_images_checkbox)

        self._show_rejected_contours_checkbox = QCheckBox("显示识别失败轮廓")
        self._show_rejected_contours_checkbox.setChecked(
            bool(self._document.get("show_rejected_contours", False))
        )
        self._show_rejected_contours_checkbox.toggled.connect(
            self._set_show_rejected_contours
        )
        side_layout.addWidget(self._show_rejected_contours_checkbox)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setObjectName("details")
        side_layout.addWidget(self._details)
        side_layout.addStretch(1)
        layout.addWidget(side)
        workspace_layout.addLayout(layout, 1)
        workspace_layout.addWidget(self._task_action_panel)
        self._pages.addWidget(workspace)

        self._task_controls = [
            self._task_action_panel,
            self._one_click_button,
            self._recognize_button,
            self._execute_button,
            self._region_selector,
            self._retake_button,
        ]
        self._configuration_controls = [
            self._capture_button,
            self._calibrate_button,
            self._red_a4_button,
            self._locate_red_a4_button,
            self._skip_button,
            self._clear_button,
            self._grbl_calibration_button,
            self._servo_calibration_button,
            self._show_piece_images_checkbox,
            self._show_rejected_contours_checkbox,
        ]
        # Only one control group is used at a time. Hiding the configuration
        # group initially keeps the inactive workspace from forcing the task
        # menu taller than the physical screen on first launch.
        for control in self._configuration_controls:
            control.setVisible(False)

        self.setStyleSheet(
            "QMainWindow { background: #171a20; color: #eef2f7; }"
            "#sidebar { background: #242933; border: 1px solid #394150; }"
            "#menuTitle { font-size: 50px; font-weight: 700; color: #ffffff; }"
            "#menuHint { font-size: 24px; color: #b8c2d1; }"
            "#workspaceTitle { font-size: 34px; font-weight: 700; color: #ffffff; }"
            "#status { font-size: 18px; color: #d3dae5; padding: 4px 10px; }"
            "#runTimer { background: #0f6b48; border: 2px solid #41d39a;"
            " color: #ffffff; font-size: 34px; font-weight: 700; padding: 14px 8px; }"
            "#details { font-size: 17px; color: #b8c2d1; padding-top: 12px; }"
            "QPushButton { background: #2d6a9f; border: 0; padding: 13px 12px;"
            " color: white; font-size: 20px; }"
            "#taskButton { background: #285b84; font-size: 28px; font-weight: 700; text-align: left; padding: 26px; }"
            "#taskButton:pressed { background: #1e4d77; }"
            "#backButton { background: #465465; font-size: 18px; }"
            "QPushButton:disabled { background: #4a505a; color: #c0c4cb; }"
            "QPushButton:pressed { background: #1e4d77; }"
            "QComboBox { font-size: 20px; padding: 8px; }"
            "QCheckBox { color: #d3dae5; font-size: 20px; padding: 6px 2px; }"
            "#taskActionPanel { background: transparent; }"
            "#oneClickButton { background: #16734a; font-size: 27px; font-weight: 700; min-height: 82px; padding: 18px 12px; }"
            "#oneClickButton:pressed { background: #0e5235; }"
            "#taskActionButton { font-size: 26px; font-weight: 700; min-height: 82px; }"
            "#executeButton { background: #a66616; font-size: 26px; font-weight: 700; min-height: 82px; }"
            "#taskActionPanel QComboBox { font-size: 22px; font-weight: 700; min-height: 64px; }"
            "#executeButton:pressed { background: #75480d; }"
            "#exitButton { background: #9b3a3a; }"
            "#exitButton:pressed { background: #702727; }"
        )
        self._show_task_menu()

    def _show_task_menu(self) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._frozen = False
        self._latest_solution = None
        self._last_recognition_payload = None
        self._one_click_execute_pending = False
        self._pages.setCurrentIndex(0)

    def _select_task(self, mode: str) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._task_mode = mode
        is_configuration = mode == "configuration"
        self._configuration_sidebar.setVisible(is_configuration)
        for control in self._task_controls:
            control.setVisible(not is_configuration)
        for control in self._configuration_controls:
            control.setVisible(is_configuration)
        self._reset_one_click_elapsed()
        self._puzzle_search_checkbox.setVisible(False)
        title_by_mode = {
            "transport": "第一题：碎片搬运",
            "white_puzzle": "第二题：白色碎片拼接",
            "card_puzzle": "第三题：扑克牌拼接",
            "configuration": "配置页面：标定与定位",
        }
        self._workspace_title.setText(title_by_mode[mode])
        self._execute_button.setText(
            "确认并执行搬运" if mode == "transport" else "确认并执行拼图"
        )
        self._one_click_button.setText(
            "一键启动：识别并搬运"
            if mode == "transport"
            else "一键启动：识别并拼图"
        )
        self._latest_solution = None
        self._one_click_execute_pending = False
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._frozen = False
        self._pages.setCurrentIndex(1)
        self._update_status()

    def _read_frame(self) -> None:
        if self._frozen:
            return
        frame = self._read_latest_camera_frame()
        if frame is None:
            self._status.setText("摄像头画面读取失败")
            return
        self._latest_frame = frame
        self._view.set_image(frame, self._corners)

    def _read_latest_camera_frame(self, discard_count: int = 0) -> np.ndarray | None:
        """Drain queued V4L2 frames so a new recognition never uses a frozen view."""
        frame: np.ndarray | None = None
        for _ in range(max(0, discard_count) + 1):
            ok, candidate = self._camera.read()
            if not ok or candidate is None:
                return frame
            frame = candidate
        return frame

    def _start_calibration(self) -> None:
        if self._recognition_thread is not None:
            return
        self._corners = []
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._frozen = False
        self._update_status()

    def _set_a4_region(self, _index: int) -> None:
        region = self._region_selector.currentData()
        if region not in ("upper", "lower"):
            return
        self._document["a4_region"] = region
        # Keep this derived key for calibration consumers that use the old flag.
        self._document["use_a4_upper_half"] = region == "upper"
        self._save_document_fields("a4_region", "use_a4_upper_half")
        self._frozen = False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._update_status()

    def _set_puzzle_search_enabled(self, enabled: bool) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._document["puzzle_search_enabled"] = bool(enabled)
        self._save_document_fields("puzzle_search_enabled")
        self._frozen = False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        self._details.clear()
        self._update_status()

    def _set_show_piece_images(self, enabled: bool) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._document["show_solution_piece_images"] = bool(enabled)
        self._save_document_fields("show_solution_piece_images")
        self._render_last_recognition_overlay()

    def _set_show_rejected_contours(self, enabled: bool) -> None:
        if self._recognition_thread is not None or self._motion_thread is not None:
            return
        self._document["show_rejected_contours"] = bool(enabled)
        self._save_document_fields("show_rejected_contours")
        self._render_last_recognition_overlay()

    def _render_last_recognition_overlay(self) -> None:
        payload = self._last_recognition_payload
        if payload is None:
            return
        overlay = draw_detection_overlay(
            payload["corrected_frame"],
            payload["result"],
            payload["calibration"],
            show_rejected_contours=self._show_rejected_contours_checkbox.isChecked(),
        )
        if payload["solution"] is not None:
            overlay = draw_solution_overlay(
                overlay,
                payload["solution"],
                payload["calibration"],
                source_image=payload["corrected_frame"],
                source_pieces=payload["result"].pieces,
                show_piece_images=self._show_piece_images_checkbox.isChecked(),
            )
        self._view.set_image(overlay, [])

    def _clear_calibration(self) -> None:
        if self._recognition_thread is not None:
            return
        self._corners = []
        self._latest_solution = None
        self._last_recognition_payload = None
        self._execute_button.setEnabled(False)
        self._document["a4_corners_px"] = []
        self._document["a4_corner_indices"] = []
        self._document["screen_a4_corners_px"] = []
        self._save_document_fields(
            "a4_corners_px", "a4_corner_indices", "screen_a4_corners_px"
        )
        self._details.clear()
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
        self._save_document_fields(
            "screen_a4_corners_px", "a4_corners_px", "a4_corner_indices"
        )
        self._frozen = False
        self._details.setText("已载入 red_a4_corners.json 的自动定位结果。")
        self._update_status()

    def _locate_red_a4_current_frame(self) -> None:
        """Run the existing red-paper locator against the live camera frame."""
        if self._latest_frame is None:
            self._status.setText("尚未收到摄像头画面，无法定位红色 A4")
            return
        try:
            screen_corners, _mask = locate_red_a4(self._latest_frame)
            a4_corners = screen_corners[list(A4_FROM_SCREEN)]
        except Exception as error:
            self._status.setText(f"红色 A4 自动定位失败：{error}")
            return
        self._corners = [QPointF(*point) for point in screen_corners]
        self._document["screen_a4_corners_px"] = np.round(screen_corners, 2).tolist()
        self._document["a4_corners_px"] = np.round(a4_corners, 2).tolist()
        self._document["a4_corner_indices"] = [0, 1, 2, 3]
        self._save_document_fields(
            "screen_a4_corners_px", "a4_corners_px", "a4_corner_indices"
        )
        self._frozen = False
        self._details.setText("已从当前画面定位并保存红色 A4 的四个角。")
        self._update_status()

    def _launch_tool(self, relative_path: str) -> None:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            self._status.setText(f"找不到校准工具：{relative_path}")
            return
        try:
            subprocess.Popen([sys.executable, str(path)], cwd=PROJECT_ROOT)
        except OSError as error:
            self._status.setText(f"无法启动校准工具：{error}")
            return
        self._status.setText(f"已打开 {path.stem}，关闭该窗口后可继续这里的操作。")

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
        self._save_document_fields(
            "screen_a4_corners_px", "a4_corners_px", "a4_corner_indices"
        )

    def _update_status(self) -> None:
        complete = len(self._corners) == 4 and sum(point is not None for point in self._corners) >= 3
        if self._task_mode == "configuration":
            if complete:
                self._status.setText("A4 标定已完成；可重新点选四角，或使用红色 A4 自动定位。")
            else:
                index = len(self._corners)
                self._status.setText(f"请点选 {SCREEN_CORNER_NAMES[index]}（{index}/4）")
            if self._latest_frame is not None:
                self._view.set_image(self._latest_frame, self._corners)
            return
        if not complete:
            index = len(self._corners)
            self._status.setText(
                f"请点选 {SCREEN_CORNER_NAMES[index]}（{index}/4）"
            )
            self._recognize_button.setEnabled(False)
        else:
            region_name = "上半" if self._document.get("a4_region", "upper") == "upper" else "下半"
            task_hint = {
                "transport": "可以识别碎片并搬运到另一半 A4",
                "white_puzzle": "可以识别白色碎片并搜索拼接方案",
                "card_puzzle": "可以识别扑克牌并搜索拼接方案",
            }.get(self._task_mode, "可以识别当前画面")
            self._status.setText(f"A4 标定完成，{task_hint}（当前：{region_name}）")
            self._recognize_button.setEnabled(True)
        if self._latest_frame is not None:
            self._view.set_image(self._latest_frame, self._corners)

    def _recognize(self) -> bool:
        if (
            len(self._corners) != 4
            or sum(point is not None for point in self._corners) < 3
            or self._recognition_thread is not None
        ):
            return False
        self._latest_solution = None
        self._execute_button.setEnabled(False)
        fresh_frame = self._read_latest_camera_frame(discard_count=7)
        if fresh_frame is None:
            self._status.setText("摄像头画面读取失败，未开始识别")
            self._one_click_execute_pending = False
            return False
        self._latest_frame = fresh_frame
        self._view.set_image(fresh_frame, self._corners)
        frame = fresh_frame.copy()
        self._recognition_example_path = None
        self._recognition_example_error = None
        try:
            self._recognition_example_path = save_recognition_example(
                frame, self._document.get("a4_region", "upper")
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._recognition_example_error = str(error)
        self._frozen = False
        self._one_click_button.setEnabled(False)
        self._recognize_button.setEnabled(False)
        self._region_selector.setEnabled(False)
        self._show_piece_images_checkbox.setEnabled(False)
        self._puzzle_search_checkbox.setEnabled(False)
        transport_only = self._task_mode == "transport"
        puzzle_search_enabled = not transport_only
        use_piece_features = self._task_mode == "card_puzzle"
        show_piece_images = self._show_piece_images_checkbox.isChecked()
        show_rejected_contours = self._show_rejected_contours_checkbox.isChecked()
        if transport_only:
            self._status.setText("正在识别碎片并生成搬运位置……")
        else:
            self._status.setText("正在识别碎片并搜索拼接方案……")
        thread = QThread(self)
        worker = RecognitionWorker(
            frame,
            self._document,
            puzzle_search_enabled,
            transport_only,
            use_piece_features,
            show_piece_images,
            show_rejected_contours,
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
        return True

    def _recognize_manually(self) -> None:
        self._one_click_execute_pending = False
        self._recognize()

    def _one_click_start(self) -> None:
        self._one_click_execute_pending = True
        self._start_one_click_elapsed()
        if not self._recognize():
            self._one_click_execute_pending = False
            self._reset_one_click_elapsed()

    def _start_one_click_elapsed(self) -> None:
        self._one_click_timer.stop()
        self._one_click_started_at = monotonic()
        self._one_click_elapsed.setText("运行计时 0 s")
        self._one_click_elapsed.setVisible(True)
        self._one_click_timer.start()

    @Slot()
    def _update_one_click_elapsed(self) -> None:
        if self._one_click_started_at is None:
            return
        elapsed = max(0, int(monotonic() - self._one_click_started_at))
        self._one_click_elapsed.setText(f"运行计时 {elapsed} s")

    def _stop_one_click_elapsed(self, prefix: str) -> None:
        if self._one_click_started_at is None:
            return
        elapsed = max(0, int(monotonic() - self._one_click_started_at))
        self._one_click_timer.stop()
        self._one_click_started_at = None
        self._one_click_elapsed.setText(f"{prefix} {elapsed} s")

    def _reset_one_click_elapsed(self) -> None:
        self._one_click_timer.stop()
        self._one_click_started_at = None
        self._one_click_elapsed.setVisible(False)

    @Slot(object)
    def _recognition_completed(self, payload: dict) -> None:
        result = payload["result"]
        solution = payload["solution"]
        solve_error = payload["solve_error"]
        puzzle_search_enabled = bool(payload["puzzle_search_enabled"])
        transport_only = bool(payload["transport_only"])
        self._latest_timings = dict(payload["timings"])
        self._frozen = True
        self._latest_solution = solution
        self._last_recognition_payload = payload
        self._execute_button.setEnabled(solution is not None)
        self._render_last_recognition_overlay()
        if transport_only and solution is None:
            self._status.setText(f"识别到 {len(result.pieces)} 块碎片，但无法生成搬运位置")
        elif transport_only:
            self._status.setText(
                f"识别到 {len(result.pieces)} 块碎片，已生成另一半 A4 的搬运位置"
            )
        elif not puzzle_search_enabled:
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
        if transport_only and solution is not None:
            detail_lines.append("第一题：不做拼接；自动使用 0°/90° 放置，将碎片依次搬运到 A4 另一半。")
        elif not puzzle_search_enabled:
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
        detail_lines.extend(
            StageTimings.from_dict(
                self._latest_timings,
                enabled=self._pipeline.timing_enabled,
            ).format_lines()
        )
        self._details.setText("\n".join(detail_lines))

    @Slot(str)
    def _recognition_failed(self, error: str) -> None:
        suffix = (
            f"；样例保存失败：{self._recognition_example_error}"
            if self._recognition_example_error
            else ""
        )
        self._status.setText(f"识别或规划失败：{error}{suffix}")
        self._details.clear()
        self._frozen = False
        self._one_click_execute_pending = False
        self._last_recognition_payload = None
        self._stop_one_click_elapsed("结束用时")

    @Slot()
    def _recognition_thread_finished(self) -> None:
        self._recognition_thread = None
        self._recognition_worker = None
        self._region_selector.setEnabled(True)
        self._show_piece_images_checkbox.setEnabled(True)
        self._puzzle_search_checkbox.setEnabled(True)
        self._one_click_button.setEnabled(True)
        self._recognize_button.setEnabled(True)
        one_click_requested = self._one_click_execute_pending
        execute_after_recognition = one_click_requested and self._latest_solution is not None
        self._one_click_execute_pending = False
        if execute_after_recognition:
            self._execute_solution(skip_confirmation=True)
        elif one_click_requested:
            self._stop_one_click_elapsed("结束用时")

    def _execute_solution(self, *, skip_confirmation: bool = False) -> None:
        if self._latest_solution is None or self._motion_thread is not None:
            if skip_confirmation:
                self._stop_one_click_elapsed("结束用时")
            return
        try:
            pipeline = MainPipeline(self._document)
            plan = pipeline.build_motion_plan(self._latest_solution)
        except Exception as exc:
            self._status.setText(f"无法生成抓取计划：{exc}")
            if skip_confirmation:
                self._stop_one_click_elapsed("结束用时")
            return

        motion_config = pipeline.motion_config
        preview_lines = [
            f"共 {len(plan)} 片，Z 下压绝对坐标 {motion_config.z_down_mm:g} mm。",
            f"舵机方向系数 {motion_config.servo_direction:+g}。",
        ]
        for index, step in enumerate(plan, start=1):
            source = step["pickup_source_grbl_mm"]
            target = step["pickup_target_grbl_mm"]
            preview_lines.append(
                f"{index}. {step['id']}: GRBL ({source[0]:.1f}, {source[1]:.1f})"
                f" → ({target[0]:.1f}, {target[1]:.1f})，舵机相对旋转 {step['servo_delta_deg']:+.1f}°"
            )
        if not skip_confirmation:
            answer = QMessageBox.question(
                self,
                "确认执行实际运动",
                "\n".join(preview_lines),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._execute_button.setEnabled(False)
        self._one_click_button.setEnabled(False)
        self._recognize_button.setEnabled(False)
        self._region_selector.setEnabled(False)
        self._show_piece_images_checkbox.setEnabled(False)
        self._puzzle_search_checkbox.setEnabled(False)
        self._status.setText("正在初始化 GRBL、舵机和电磁铁……")
        thread = QThread(self)
        worker = MotionWorker(
            self._latest_solution,
            self._document,
            self._latest_timings,
        )
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
    def _motion_completed(self, payload: dict) -> None:
        plan = payload["plan"]
        self._latest_timings = dict(payload["timings"])
        action = "搬运" if self._task_mode == "transport" else "拼图"
        self._status.setText(f"{action}执行完成，共放置 {len(plan)} 片，机械机构已回到工作零点。")
        timing_lines = StageTimings.from_dict(
            self._latest_timings,
            enabled=self._pipeline.timing_enabled,
        ).format_lines()
        self._details.setText(
            "实际动作已完成；继续识别前请确认所有碎片的位置。\n"
            + "\n".join(timing_lines)
        )
        self._latest_solution = None
        self._last_recognition_payload = None
        self._stop_one_click_elapsed("完成用时")
        self._play_finished_sound()

    def _play_finished_sound(self) -> None:
        if (
            not FINISHED_SOUND_PATH.is_file()
            or self._finished_sound_process.state()
            != QProcess.ProcessState.NotRunning
        ):
            return
        self._finished_sound_process.start(
            "/usr/bin/aplay", ["-q", str(FINISHED_SOUND_PATH)]
        )

    @Slot(str)
    def _motion_failed(self, error: str) -> None:
        action = "搬运动作" if self._task_mode == "transport" else "拼图动作"
        self._status.setText(f"{action}失败：{error}")
        self._details.setText("已尝试执行 Z 收回、磁铁释放、舵机复位和 XY 回零。请先检查机构状态。")
        self._latest_solution = None
        self._stop_one_click_elapsed("结束用时")

    @Slot()
    def _motion_thread_finished(self) -> None:
        self._motion_thread = None
        self._motion_worker = None
        self._region_selector.setEnabled(True)
        self._show_piece_images_checkbox.setEnabled(True)
        self._puzzle_search_checkbox.setEnabled(True)
        self._one_click_button.setEnabled(True)
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
        self._one_click_execute_pending = False
        self._reset_one_click_elapsed()
        self._execute_button.setEnabled(False)
        self._details.clear()
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
        self._one_click_timer.stop()
        self._camera.release()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="Puzzle-piece recognition touchscreen UI")
    parser.add_argument("--device", default="/dev/diansai-camera", help="V4L2 camera device")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    application = QApplication(sys.argv)
    if not wait_for_required_hardware(args.device):
        return 0
    window = RecognitionWindow(args.device, args.fullscreen)
    if not args.fullscreen:
        window.resize(1440, 850)
        window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
