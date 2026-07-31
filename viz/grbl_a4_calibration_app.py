#!/usr/bin/env python3
"""Qt tool for collecting A4(mm) <-> GRBL work-coordinate calibration pairs.

Workflow: jog the magnet to the shown A4 point, refresh/read its GRBL WPos,
then save the pair.  Three or more saved pairs produce data/a4_to_grbl.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
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
ALGORITHM_ROOT = next(PROJECT_ROOT.glob("algorithm*"))
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion.core_xy import CoreXY  # noqa: E402
from puzzle_solver.coordinates import A4ToGrblTransform  # noqa: E402


SAMPLES_PATH = PROJECT_ROOT / "data" / "a4_grbl_calibration_samples.json"
TRANSFORM_PATH = PROJECT_ROOT / "data" / "a4_to_grbl.json"
WPOS_PATTERN = re.compile(r"WPos:([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)")


def parse_work_position(status: str) -> tuple[float, float, float]:
    """Extract GRBL work coordinates, never confuse MPos with WPos."""
    match = WPOS_PATTERN.search(status)
    if match is None:
        raise ValueError(
            "GRBL 当前只报告 MPos（机床坐标），未报告 WPos（工作坐标）；"
            "请先点击“将当前点设为原点”以配置 $10=0"
        )
    return tuple(float(value) for value in match.groups())


class CalibrationWindow(QMainWindow):
    def __init__(self, port: str, baudrate: int, feed_mm_min: float) -> None:
        super().__init__()
        self._machine = CoreXY(port=port, baudrate=baudrate)
        self._feed_mm_min = float(feed_mm_min)
        self._calibration_document = self._load_calibration_document()
        self._samples = self._calibration_document.get("samples", [])
        self._initial_calibration = self._calibration_document.get("initial_calibration")
        self._a4_point = self._random_a4_point()

        self.setWindowTitle("A4 ↔ GRBL 坐标校准")
        self.setMinimumSize(620, 480)
        self._build_ui()
        self._show_a4_point()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)

        title = QLabel("A4 ↔ GRBL 坐标校准")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        layout.addWidget(title)

        help_text = QLabel(
            "将磁铁中心手动移动到下方指定的 A4 点，再保存当前 GRBL 工作坐标。"
            "随机点会避开机构死区，且均为 1 cm 的整数网格；至少保存 3 组后自动生成转换矩阵。"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        target = QFrame()
        target.setFrameShape(QFrame.StyledPanel)
        target_layout = QVBoxLayout(target)
        self._a4_label = QLabel()
        self._a4_label.setAlignment(Qt.AlignCenter)
        self._a4_label.setStyleSheet("font-size: 30px; font-weight: 700; color: #136f3a;")
        target_layout.addWidget(self._a4_label)
        self._new_point_button = QPushButton("换一个 A4 随机点（未保存时慎用）")
        self._new_point_button.clicked.connect(self._new_a4_point)
        target_layout.addWidget(self._new_point_button)
        layout.addWidget(target)

        fields = QFormLayout()
        self._x = self._coordinate_box()
        self._y = self._coordinate_box()
        self._z = self._coordinate_box()
        fields.addRow("GRBL 工作坐标 X (mm)", self._x)
        fields.addRow("GRBL 工作坐标 Y (mm)", self._y)
        fields.addRow("GRBL 工作坐标 Z (mm)", self._z)
        layout.addLayout(fields)

        self._orientation_prior = QCheckBox("使用方向先验：GRBL X = A4 Y，GRBL Y = -A4 X")
        self._orientation_prior.setChecked(True)
        self._orientation_prior.setToolTip(
            "当前机构中：GRBL X 正方向向下，GRBL Y 正方向向左；"
            "A4 X 正方向向右，A4 Y 正方向向下。"
        )
        self._orientation_prior.toggled.connect(lambda _checked: self._fill_prior_guess())
        layout.addWidget(self._orientation_prior)

        action_row = QHBoxLayout()
        self._origin_button = QPushButton("将当前点设为原点")
        self._origin_button.clicked.connect(self._set_origin)
        action_row.addWidget(self._origin_button)
        self._zero_button = QPushButton("低速回零点")
        self._zero_button.clicked.connect(self._go_to_zero)
        action_row.addWidget(self._zero_button)
        self._refresh_button = QPushButton("读取当前坐标")
        self._refresh_button.clicked.connect(self._refresh_position)
        action_row.addWidget(self._refresh_button)
        layout.addLayout(action_row)

        self._move_button = QPushButton("低速移动到输入的绝对 GRBL 坐标")
        self._move_button.clicked.connect(self._move_to_entered_position)
        layout.addWidget(self._move_button)

        self._save_button = QPushButton("保存当前 A4 ↔ GRBL 对应点")
        self._save_button.setStyleSheet("font-size: 18px; font-weight: 700; padding: 10px;")
        self._save_button.clicked.connect(self._save_sample)
        layout.addWidget(self._save_button)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    @staticmethod
    def _coordinate_box() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-10000.0, 10000.0)
        box.setDecimals(3)
        box.setSingleStep(0.1)
        box.setSuffix(" mm")
        return box

    def _connect(self) -> None:
        if self._machine.uart is None or not self._machine.uart.is_open:
            self._machine.open()

    def _random_a4_point(self) -> tuple[float, float]:
        used = {(item["a4_mm"][0], item["a4_mm"][1]) for item in self._samples}
        # A4 is 21 x 29.7 cm, so a 1 cm integer grid reaches X=21, Y=29.
        min_x_mm, min_y_mm = (0.0, 0.0)
        if self._initial_calibration is not None:
            min_x_mm, min_y_mm = self._initial_calibration.get("reachable_a4_min_mm", (0.0, 0.0))
        min_x_cm = max(0, math.ceil(float(min_x_mm) / 10.0))
        min_y_cm = max(0, math.ceil(float(min_y_mm) / 10.0))
        choices = [
            (float(x), float(y))
            for x in range(min_x_cm, 22)
            for y in range(min_y_cm, 30)
            if (float(x * 10), float(y * 10)) not in used
        ]
        x_cm, y_cm = random.choice(choices or [(10.0, 15.0)])
        return x_cm * 10, y_cm * 10

    def _show_a4_point(self) -> None:
        self._a4_label.setText(
            f"本次 A4 目标点：X={self._a4_point[0] / 10:g} cm，Y={self._a4_point[1] / 10:g} cm"
        )
        self._fill_prior_guess()
        self._status.setText(f"已保存 {len(self._samples)} 组对应点。")

    def _new_a4_point(self) -> None:
        self._a4_point = self._random_a4_point()
        self._show_a4_point()

    def _fill_prior_guess(self) -> None:
        """Pre-fill GRBL XY from the known axis orientation and saved samples.

        The explicit ``initial_calibration`` point is always the reference.
        Later samples are retained for final transform fitting, but deliberately
        do not alter this convenient movement prior.
        """
        if not self._orientation_prior.isChecked():
            return
        if self._initial_calibration is None:
            return
        matrix = self._initial_calibration.get("a4_to_grbl_affine_matrix")
        if matrix is None:
            # Compatibility with old calibration files: construct the same
            # affine matrix from one base point plus the direction prior.
            base_a4_x, base_a4_y = self._initial_calibration["a4_mm"]
            base_grbl_x, base_grbl_y, _ = self._initial_calibration["grbl_wpos_mm"]
            x_scale = float(self._initial_calibration.get("grbl_x_per_a4_y", 1.0))
            y_scale = float(self._initial_calibration.get("grbl_y_per_a4_x", -1.0))
            matrix = [
                [0.0, x_scale, base_grbl_x - x_scale * base_a4_y],
                [y_scale, 0.0, base_grbl_y - y_scale * base_a4_x],
                [0.0, 0.0, 1.0],
            ]
        grbl_xy = A4ToGrblTransform(np.asarray(matrix), model="affine").to_grbl([self._a4_point])[0]
        self._x.setValue(grbl_xy[0])
        self._y.setValue(grbl_xy[1])
        self._z.setValue(self._initial_calibration["grbl_wpos_mm"][2])

    def _refresh_position(self, silent: bool = False) -> bool:
        try:
            self._connect()
            x, y, z = parse_work_position(self._machine.status())
            self._x.setValue(x)
            self._y.setValue(y)
            self._z.setValue(z)
            if not silent:
                self._status.setText("已读取当前 GRBL 工作坐标（WPos）。")
            return True
        except Exception as exc:
            if not silent:
                self._status.setText(f"读取失败：{exc}")
            return False

    def _set_origin(self) -> None:
        try:
            self._connect()
            # This GRBL 1.3a firmware uses $10=0 for WPos status reports.
            # It is persistent and does not move the machine. G92 below then
            # makes this WPos zero.
            self._machine.command("$10=0")
            self._machine.set_work_position(0, 0, 0)  # G92 X0 Y0 Z0
            self._x.setValue(0)
            self._y.setValue(0)
            self._z.setValue(0)
            self._status.setText("已配置 $10=0 并发送 G92 X0 Y0 Z0：当前点已设为 GRBL 工作原点。")
        except Exception as exc:
            self._status.setText(f"设原点失败：{exc}")

    def _go_to_zero(self) -> None:
        try:
            self._connect()
            self._machine.move_to(
                0, 0, 0, feed=self._feed_mm_min, rapid=False
            )
            position = self._machine.wait_until_position(x=0, y=0, z=0)
            self._refresh_position(silent=True)
            self._status.setText(
                f"已低速回到工作零点（F{self._feed_mm_min:g}）：{position}"
            )
        except Exception as exc:
            self._status.setText(f"回零失败：{exc}")

    def _move_to_entered_position(self) -> None:
        """Move absolutely to the X/Y/Z values currently shown in the form."""
        x, y, z = self._x.value(), self._y.value(), self._z.value()
        try:
            self._connect()
            self._machine.move_to(
                x, y, z, feed=self._feed_mm_min, rapid=False
            )
            position = self._machine.wait_until_position(x=x, y=y, z=z)
            self._refresh_position(silent=True)
            self._status.setText(
                f"已低速到达绝对坐标 X{x:g} Y{y:g} Z{z:g} "
                f"（F{self._feed_mm_min:g}）：{position}"
            )
        except Exception as exc:
            self._status.setText(f"移动失败：{exc}")

    def _save_sample(self) -> None:
        # Never save the prior/entered target as if it were measured feedback.
        # The actual GRBL work position is the calibration observation.
        if not self._refresh_position():
            return
        grbl = [self._x.value(), self._y.value(), self._z.value()]
        sample = {
            "a4_mm": list(self._a4_point),
            "grbl_wpos_mm": grbl,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._samples.append(sample)
        self._write_samples()
        try:
            if len(self._samples) >= 3:
                a4_points = [item["a4_mm"] for item in self._samples]
                grbl_points = [item["grbl_wpos_mm"][:2] for item in self._samples]
                transform = self._fit_transform(a4_points, grbl_points)
                transform.save(TRANSFORM_PATH)
                max_error = max(transform.reprojection_error_mm(a4_points, grbl_points))
                result = f"已保存第 {len(self._samples)} 组；转换矩阵已更新，最大误差 {max_error:.3f} mm。"
            else:
                result = f"已保存第 {len(self._samples)} 组；再保存 {3 - len(self._samples)} 组即可生成转换矩阵。"
            self._a4_point = self._random_a4_point()
            self._show_a4_point()
            self._status.setText(result)
        except Exception as exc:
            self._status.setText(f"对应点已保存，但矩阵拟合失败：{exc}")

    def _fit_transform(self, a4_points: list[list[float]], grbl_points: list[list[float]]) -> A4ToGrblTransform:
        """Fit either the known axis orientation or a general affine transform."""
        if not self._orientation_prior.isChecked():
            return A4ToGrblTransform.fit(a4_points, grbl_points)

        a4 = np.asarray(a4_points, dtype=float)
        grbl = np.asarray(grbl_points, dtype=float)
        # Prior: GRBL +X is down (A4 +Y); GRBL +Y is left (A4 -X).
        if np.ptp(a4[:, 0]) < 1e-6 or np.ptp(a4[:, 1]) < 1e-6:
            raise ValueError("启用方向先验时，保存的 A4 点必须同时覆盖不同的 X 和 Y")
        x_scale, x_offset = np.polyfit(a4[:, 1], grbl[:, 0], 1)
        y_scale, y_offset = np.polyfit(a4[:, 0], grbl[:, 1], 1)
        if x_scale <= 0 or y_scale >= 0:
            raise ValueError(
                "样本与方向先验不一致：应满足 GRBL X 随 A4 Y 增大、GRBL Y 随 A4 X 减小；"
                "请检查方向，或取消方向先验后再拟合"
            )
        matrix = np.array(
            [[0.0, x_scale, x_offset], [y_scale, 0.0, y_offset], [0.0, 0.0, 1.0]]
        )
        return A4ToGrblTransform(matrix, model="affine")

    def _load_calibration_document(self) -> dict:
        if not SAMPLES_PATH.exists():
            return {"samples": []}
        try:
            document = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
            document.setdefault("samples", [])
            return document
        except (OSError, json.JSONDecodeError):
            return {"samples": []}

    def _write_samples(self) -> None:
        SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._calibration_document["samples"] = self._samples
        SAMPLES_PATH.write_text(
            json.dumps(self._calibration_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._machine.close()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 ↔ GRBL 坐标校准页面")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--feed",
        type=float,
        default=1000.0,
        help="校准移动速度，单位 mm/min（默认 1000）",
    )
    args = parser.parse_args()
    if args.feed <= 0:
        parser.error("--feed 必须大于 0")
    app = QApplication(sys.argv)
    window = CalibrationWindow(args.port, args.baudrate, args.feed)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
