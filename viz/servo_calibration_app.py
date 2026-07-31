#!/usr/bin/env python3
"""Touch-friendly Qt page for Fashion Star bus-servo calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion.servo import DEFAULT_BAUDRATE, DEFAULT_PORT, FashionStarServo  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "data" / "servo_calibration.json"


class ServoCalibrationWindow(QMainWindow):
    def __init__(self, port: str, baudrate: int, servo_id: int) -> None:
        super().__init__()
        self._servo: FashionStarServo | None = None
        self._records = self._load_records()

        self.setWindowTitle("舵机校准")
        self.setMinimumSize(580, 560)
        self._build_ui(port, baudrate, servo_id)
        self._set_status("请先连接舵机；“扫描 ID”不会使舵机运动。")

    def _build_ui(self, port: str, baudrate: int, servo_id: int) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        title = QLabel("舵机校准")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        layout.addWidget(title)

        connection = QFrame()
        connection.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(connection)
        self._port = QLineEdit(port)
        self._baudrate = QSpinBox()
        self._baudrate.setRange(1200, 3_000_000)
        self._baudrate.setValue(baudrate)
        self._servo_id = QSpinBox()
        self._servo_id.setRange(0, 253)
        self._servo_id.setValue(servo_id)
        form.addRow("串口", self._port)
        form.addRow("波特率", self._baudrate)
        form.addRow("舵机 ID", self._servo_id)
        layout.addWidget(connection)

        connection_actions = QHBoxLayout()
        connect = QPushButton("连接并读取")
        connect.clicked.connect(self._connect_and_read)
        connection_actions.addWidget(connect)
        scan = QPushButton("扫描 ID (0–10)")
        scan.clicked.connect(self._scan)
        connection_actions.addWidget(scan)
        layout.addLayout(connection_actions)

        current = QFrame()
        current.setFrameShape(QFrame.StyledPanel)
        current_layout = QVBoxLayout(current)
        self._angle_label = QLabel("当前反馈角度：未读取")
        self._angle_label.setAlignment(Qt.AlignCenter)
        self._angle_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        current_layout.addWidget(self._angle_label)
        read = QPushButton("读取当前角度")
        read.clicked.connect(self._read_angle)
        current_layout.addWidget(read)
        layout.addWidget(current)

        fields = QFormLayout()
        self._target = self._angle_box()
        self._target.setValue(0.0)
        self._physical = self._angle_box()
        self._physical.setValue(0.0)
        fields.addRow("目标舵机角度 (°)", self._target)
        fields.addRow("当前机械参考角度 (°)", self._physical)
        layout.addLayout(fields)

        self._multi_turn = QCheckBox("多圈位置模式（360° 舵机建议用于大范围定位）")
        self._multi_turn.setToolTip("启用后发送多圈定位命令；不是持续转动的轮式模式。")
        self._multi_turn.toggled.connect(self._set_multi_turn)
        layout.addWidget(self._multi_turn)

        movement = QHBoxLayout()
        move = QPushButton("移动到目标角度")
        move.clicked.connect(self._move)
        movement.addWidget(move)
        release = QPushButton("释放扭矩（可手动对位）")
        release.clicked.connect(self._release)
        movement.addWidget(release)
        hold = QPushButton("保持当前位置")
        hold.clicked.connect(self._hold)
        movement.addWidget(hold)
        layout.addLayout(movement)

        origin = QPushButton("将当前机械位置设为舵机零位")
        origin.setStyleSheet("font-weight: 700;")
        origin.clicked.connect(self._set_origin)
        layout.addWidget(origin)

        save = QPushButton("保存当前参考点")
        save.setStyleSheet("font-size: 18px; font-weight: 700; padding: 9px;")
        save.clicked.connect(self._save_record)
        layout.addWidget(save)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    @staticmethod
    def _angle_box() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-180.0, 180.0)
        box.setDecimals(2)
        box.setSingleStep(1.0)
        box.setSuffix(" °")
        return box

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _set_multi_turn(self, enabled: bool) -> None:
        # SDK multi-turn command accepts ±368640°, while the normal position
        # command is deliberately limited to one turn, ±180°.
        self._target.setRange(-368_640.0 if enabled else -180.0, 368_640.0 if enabled else 180.0)

    def _connect(self, verify: bool = True) -> FashionStarServo:
        requested = (self._port.text().strip(), self._baudrate.value(), self._servo_id.value())
        if self._servo is not None:
            current = (self._servo.port, self._servo.baudrate, self._servo.servo_id)
            if current == requested and self._servo.uart is not None and self._servo.uart.is_open:
                return self._servo
            self._servo.close()
        self._servo = FashionStarServo(port=requested[0], baudrate=requested[1], servo_id=requested[2])
        return self._servo.open(verify=verify)

    def _scan(self) -> None:
        try:
            servo = self._connect(verify=False)
            ids = servo.scan(0, 10)
            if ids:
                self._servo_id.setValue(ids[0])
                self._set_status("发现在线舵机 ID：{}；已填入第一个 ID。".format(", ".join(map(str, ids))))
            else:
                self._set_status("未扫描到 ID 0–10 的舵机。")
        except Exception as exc:
            self._set_status(f"扫描失败：{exc}")

    def _connect_and_read(self) -> None:
        try:
            self._connect()
            self._read_angle()
            self._set_status("舵机已连接，当前反馈角度已读取。")
        except Exception as exc:
            self._set_status(f"连接失败：{exc}")

    def _read_angle(self) -> float | None:
        try:
            angle = self._connect().angle(multi_turn=self._multi_turn.isChecked())
            if angle is None:
                raise RuntimeError("未收到角度反馈")
            self._angle_label.setText(f"当前反馈角度：{angle:.2f}°")
            return angle
        except Exception as exc:
            self._set_status(f"读取角度失败：{exc}")
            return None

    def _move(self) -> None:
        try:
            multi_turn = self._multi_turn.isChecked()
            self._connect().move(
                self._target.value(), interval_ms=800, multi_turn=multi_turn, wait=False
            )
            mode = "多圈" if multi_turn else "单圈"
            self._set_status(f"已发送{mode}位置命令：{self._target.value():.2f}°。")
        except Exception as exc:
            self._set_status(f"移动失败：{exc}")

    def _release(self) -> None:
        try:
            self._connect().release_torque()
            self._set_status("已释放扭矩，请扶住负载后再手动对位。")
        except Exception as exc:
            self._set_status(f"释放扭矩失败：{exc}")

    def _hold(self) -> None:
        try:
            self._connect().stop("hold")
            self._set_status("已命令舵机保持当前位置。")
        except Exception as exc:
            self._set_status(f"保持失败：{exc}")

    def _set_origin(self) -> None:
        try:
            self._connect().set_origin()
            self._target.setValue(0.0)
            self._angle_label.setText("当前反馈角度：零位已写入，请重新读取确认")
            self._set_status("已将当前机械位置写为舵机硬件零位。")
        except Exception as exc:
            self._set_status(f"设零位失败：{exc}")

    def _save_record(self) -> None:
        angle = self._read_angle()
        if angle is None:
            return
        self._records.append(
            {
                "physical_reference_deg": self._physical.value(),
                "servo_feedback_deg": angle,
                "target_deg": self._target.value(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "port": self._port.text().strip(),
                    "baudrate": self._baudrate.value(),
                    "servo_id": self._servo_id.value(),
                    "reference_points": self._records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._set_status(f"已保存第 {len(self._records)} 个参考点：{OUTPUT_PATH}")

    @staticmethod
    def _load_records() -> list[dict]:
        if not OUTPUT_PATH.exists():
            return []
        try:
            return json.loads(OUTPUT_PATH.read_text(encoding="utf-8")).get("reference_points", [])
        except (OSError, json.JSONDecodeError):
            return []

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._servo is not None:
            self._servo.close()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion Star 舵机校准页面")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--id", type=int, default=0, dest="servo_id")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = ServoCalibrationWindow(args.port, args.baudrate, args.servo_id)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
