#!/usr/bin/env python3
"""Minimal GRBL serial controller for the CoreXY writing machine.

The controller firmware performs CoreXY motor kinematics. This module sends
standard Cartesian X/Y/Z G-code over the CH340 serial adapter.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import serial


DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200


class GrblError(RuntimeError):
    """The controller rejected a G-code command."""


class GrblTimeout(TimeoutError):
    """The controller did not return the expected response in time."""


class CoreXY:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 0.1,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.uart: serial.Serial | None = None
        self._work_coordinate_offset: tuple[float, ...] | None = None

    def open(self) -> "CoreXY":
        self.uart = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=2,
        )
        self.uart.reset_input_buffer()
        return self

    def close(self) -> None:
        if self.uart is not None:
            self.uart.close()
            self.uart = None

    def __enter__(self) -> "CoreXY":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_uart(self) -> serial.Serial:
        if self.uart is None or not self.uart.is_open:
            raise RuntimeError("串口未打开，请先调用 open()")
        return self.uart

    @staticmethod
    def _validate_line(line: str) -> str:
        line = line.strip()
        if not line:
            raise ValueError("G-code 不能为空")
        if "\r" in line or "\n" in line:
            raise ValueError("一次只能发送一行 G-code")
        return line

    def command(self, line: str, timeout: float = 5.0) -> list[str]:
        """Send one G-code line and wait for its ``ok`` or ``error`` response."""
        line = self._validate_line(line)
        uart = self._require_uart()
        uart.write((line + "\n").encode("ascii"))
        uart.flush()

        deadline = time.monotonic() + timeout
        responses: list[str] = []
        while time.monotonic() < deadline:
            raw = uart.readline()
            if not raw:
                continue
            response = raw.decode("ascii", errors="replace").strip()
            if not response:
                continue
            responses.append(response)
            if response == "ok":
                return responses
            if response.startswith(("error:", "ALARM:")):
                raise GrblError("{} -> {}".format(line, response))
        raise GrblTimeout("等待控制器响应超时: {}".format(line))

    def status(self, timeout: float = 1.0) -> str:
        """Return the latest real-time GRBL status report without moving."""
        uart = self._require_uart()
        uart.write(b"?")
        uart.flush()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = uart.readline()
            if not raw:
                continue
            response = raw.decode("ascii", errors="replace").strip()
            if response.startswith("<") and response.endswith(">"):
                return response
        raise GrblTimeout("等待状态报告超时")

    def soft_reset(self, timeout: float = 2.0) -> list[str]:
        """Reset GRBL with realtime Ctrl-X and wait for its startup banner.

        When invoked while Idle this does not move an axis. It also makes a
        newly-written ``$1=0`` take effect immediately, releasing the stepper
        enable output during hardware deinitialization.
        """
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        uart = self._require_uart()
        uart.write(b"\x18")
        uart.flush()

        deadline = time.monotonic() + timeout
        responses: list[str] = []
        while time.monotonic() < deadline:
            raw = uart.readline()
            if not raw:
                continue
            response = raw.decode("ascii", errors="replace").strip()
            if not response:
                continue
            responses.append(response)
            if response.startswith("Grbl "):
                return responses
        raise GrblTimeout("等待 GRBL 软复位启动信息超时")

    @staticmethod
    def _axis_words(x: float | None, y: float | None, z: float | None) -> str:
        words = []
        for axis, value in (("X", x), ("Y", y), ("Z", z)):
            if value is not None:
                words.append("{}{:g}".format(axis, value))
        if not words:
            raise ValueError("至少指定 X、Y、Z 中的一个坐标")
        return " ".join(words)

    def move_to(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        feed: float | None = None,
        rapid: bool = False,
    ) -> list[str]:
        """Move to an absolute work coordinate. ``rapid=True`` uses G0."""
        words = self._axis_words(x, y, z)
        self.command("G90")
        motion = "G0" if rapid else "G1"
        if feed is not None:
            if feed <= 0:
                raise ValueError("进给速度必须大于 0")
            words += " F{:g}".format(feed)
        return self.command("{} {}".format(motion, words))

    def move_by(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        feed: float | None = None,
        rapid: bool = False,
    ) -> list[str]:
        """Move by a relative X/Y/Z offset and restore absolute mode."""
        words = self._axis_words(x, y, z)
        self.command("G91")
        motion = "G0" if rapid else "G1"
        if feed is not None:
            if feed <= 0:
                raise ValueError("进给速度必须大于 0")
            words += " F{:g}".format(feed)
        try:
            return self.command("{} {}".format(motion, words))
        finally:
            self.command("G90")

    def set_work_position(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> list[str]:
        """Set the current work coordinate with G92; this does not move axes."""
        return self.command("G92 {}".format(self._axis_words(x, y, z)))

    def wait_until_idle(self, timeout: float = 30.0, interval: float = 0.1) -> str:
        """Wait until GRBL reports Idle. ``ok`` only means a move was queued."""
        deadline = time.monotonic() + timeout
        last_status = ""
        while time.monotonic() < deadline:
            last_status = self.status()
            state = last_status[1:].split("|", 1)[0]
            if state == "Idle":
                return last_status
            if state.startswith("Alarm"):
                raise GrblError(last_status)
            time.sleep(interval)
        raise GrblTimeout("等待运动完成超时，最后状态: {}".format(last_status))

    @staticmethod
    def _parse_status_report(report: str) -> tuple[str, dict[str, str]]:
        """Parse ``<State|Key:Value|...>`` without assuming a fixed field set."""

        if not report.startswith("<") or not report.endswith(">"):
            raise ValueError("不是有效的 GRBL 状态报告: {}".format(report))
        parts = report[1:-1].split("|")
        state = parts[0]
        fields: dict[str, str] = {}
        for part in parts[1:]:
            if ":" in part:
                key, value = part.split(":", 1)
                fields[key] = value
        return state, fields

    def wait_until_position(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        timeout: float = 30.0,
        interval: float = 0.05,
        tolerance: float = 0.05,
    ) -> str:
        """Wait for both planner Idle and the requested work position.

        A status query immediately after a queued G-code command can briefly
        report ``Idle`` before execution changes to ``Run``.  Checking ``WPos``
        as well prevents that transient state from releasing the caller early.
        """

        targets = {index: value for index, value in enumerate((x, y, z)) if value is not None}
        if not targets:
            raise ValueError("至少需要一个目标轴坐标")
        if timeout <= 0.0 or interval < 0.0 or tolerance < 0.0:
            raise ValueError("timeout 必须大于 0，interval/tolerance 不能小于 0")

        deadline = time.monotonic() + timeout
        last_status = ""
        last_position: tuple[float, ...] | None = None
        while time.monotonic() < deadline:
            last_status = self.status()
            state, fields = self._parse_status_report(last_status)
            if state.startswith("Alarm"):
                raise GrblError(last_status)

            encoded_offset = fields.get("WCO")
            if encoded_offset is not None:
                try:
                    offset = tuple(float(value) for value in encoded_offset.split(","))
                    if len(offset) >= 3:
                        self._work_coordinate_offset = offset
                except ValueError:
                    pass

            encoded_position = fields.get("WPos")
            if encoded_position is not None:
                try:
                    last_position = tuple(float(value) for value in encoded_position.split(","))
                except ValueError:
                    last_position = None
            elif fields.get("MPos") is not None:
                try:
                    machine_position = tuple(
                        float(value) for value in fields["MPos"].split(",")
                    )
                except ValueError:
                    last_position = None
                else:
                    offset = getattr(self, "_work_coordinate_offset", None)
                    if offset is not None and len(machine_position) >= 3:
                        last_position = tuple(
                            machine - origin
                            for machine, origin in zip(machine_position, offset)
                        )
            reached = (
                last_position is not None
                and len(last_position) >= 3
                and all(
                    abs(last_position[index] - float(target)) <= tolerance
                    for index, target in targets.items()
                )
            )
            if state == "Idle" and reached:
                return last_status
            time.sleep(interval)

        expected = ", ".join(
            "{}={:g}".format(axis, target)
            for axis, target in zip("XYZ", (x, y, z))
            if target is not None
        )
        raise GrblTimeout(
            "等待到达目标位置超时（{}），最后位置: {}，最后状态: {}".format(
                expected, last_position, last_status
            )
        )

    def run_file(self, path: str | Path, timeout_per_line: float = 5.0) -> None:
        """Stream a G-code file using GRBL's reliable send-response protocol."""
        for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith((";", "(")):
                continue
            try:
                self.command(line, timeout=timeout_per_line)
            except Exception as exc:
                raise GrblError("第 {} 行失败: {}".format(line_number, exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRBL CoreXY serial controller")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status", help="读取控制器状态，不运动")

    move = subparsers.add_parser("move", help="移动 XYZ 坐标")
    move.add_argument("--x", type=float)
    move.add_argument("--y", type=float)
    move.add_argument("--z", type=float)
    move.add_argument("--feed", type=float, default=1000)
    move.add_argument("--relative", action="store_true")
    move.add_argument("--rapid", action="store_true")
    move.add_argument("--wait", action="store_true", help="等待物理运动结束")

    set_position = subparsers.add_parser("set-position", help="用 G92 设置当前工作坐标")
    set_position.add_argument("--x", type=float)
    set_position.add_argument("--y", type=float)
    set_position.add_argument("--z", type=float)

    program = subparsers.add_parser("run", help="逐行执行 G-code 文件")
    program.add_argument("file")
    program.add_argument("--wait", action="store_true", help="等待全部运动完成")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with CoreXY(args.port, args.baudrate) as machine:
        if args.action == "status":
            print(machine.status())
        elif args.action == "move":
            move = machine.move_by if args.relative else machine.move_to
            print(*move(args.x, args.y, args.z, args.feed, args.rapid), sep="\n")
            if args.wait:
                if args.relative:
                    print(machine.wait_until_idle())
                else:
                    print(machine.wait_until_position(x=args.x, y=args.y, z=args.z))
        elif args.action == "set-position":
            print(*machine.set_work_position(args.x, args.y, args.z), sep="\n")
        elif args.action == "run":
            machine.run_file(args.file)
            if args.wait:
                print(machine.wait_until_idle())


if __name__ == "__main__":
    main()
