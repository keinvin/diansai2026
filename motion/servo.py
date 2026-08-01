#!/usr/bin/env python3
"""Fashion Star UART bus-servo wrapper for the local CH340 adapter.

The class uses the APIs documented at:
https://fashionstar.com.hk/wiki/zh/sdk/servo/python-sdk/
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import asdict, dataclass
from typing import Callable

import fashionstar_uart_sdk as uservo_sdk
import serial


DEFAULT_PORT = "/dev/diansai-servo"
DEFAULT_BAUDRATE = 115200


class ServoError(RuntimeError):
    """Base exception for Fashion Star servo operations."""


class ServoTimeout(ServoError):
    """The servo did not provide the expected feedback in time."""


@dataclass
class ServoTelemetry:
    online: bool
    angle: float | None
    voltage_v: float | None
    current_a: float | None
    power_w: float | None
    temperature_c: float | None
    status: int | None
    multi_turn_angle: float | None
    monitor_supported: bool


class FashionStarServo:
    """High-level wrapper around ``fashionstar_uart_sdk.UartServoManager``.

    The default configuration matches the confirmed local setup: ID 0 at
    115200 baud on ``/dev/diansai-servo``.
    """

    def __init__(
        self,
        servo_id: int = 0,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 0.1,
        debug: bool = False,
    ) -> None:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id 必须在 0 到 253 之间")
        self.servo_id = servo_id
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug
        self.uart: serial.Serial | None = None
        self.manager: uservo_sdk.UartServoManager | None = None

    def open(self, verify: bool = True) -> "FashionStarServo":
        self.uart = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )
        self.manager = uservo_sdk.UartServoManager(self.uart, is_debug=self.debug)
        if verify and not self.ping():
            self.close()
            raise ServoError(
                "未检测到 ID={} 的舵机，请检查串口、波特率、供电和总线接线".format(
                    self.servo_id
                )
            )
        return self

    def close(self) -> None:
        if self.uart is not None:
            self.uart.close()
        self.uart = None
        self.manager = None

    def __enter__(self) -> "FashionStarServo":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_manager(self) -> uservo_sdk.UartServoManager:
        if self.manager is None:
            raise ServoError("串口未打开，请先调用 open()")
        return self.manager

    def ping(self) -> bool:
        """Return whether the configured servo ID responds to the SDK ping."""
        return self._require_manager().ping(self.servo_id)

    def scan(self, start_id: int = 0, end_id: int = 253) -> list[int]:
        """Return every online servo ID in the inclusive scan range.

        This only sends SDK ping packets and does not command any movement.
        """
        if not 0 <= start_id <= end_id <= 253:
            raise ValueError("扫描范围必须在 0 到 253 之间")
        manager = self._require_manager()
        return [servo_id for servo_id in range(start_id, end_id + 1) if manager.ping(servo_id)]

    def angle(self, multi_turn: bool = False) -> float | None:
        """Read the current angle in degrees, or ``None`` if no reply arrives."""
        manager = self._require_manager()
        if multi_turn:
            return manager.query_servo_mturn_angle(self.servo_id, realtime=True)
        return manager.query_servo_angle(self.servo_id, realtime=True)

    def telemetry(self) -> ServoTelemetry:
        """Read the standard voltage/current/power/temperature/status registers."""
        manager = self._require_manager()
        readers: dict[str, Callable[..., object]] = {
            "voltage_v": manager.query_voltage,
            "current_a": manager.query_current,
            "power_w": manager.query_power,
            "temperature_c": manager.query_temperature,
            "status": manager.query_status,
        }
        values: dict[str, object | None] = {}
        for name, reader in readers.items():
            try:
                values[name] = reader(self.servo_id, realtime=True)
            except (ArithmeticError, KeyError, struct.error):
                values[name] = None

        monitor = manager.query_servo_monitor(self.servo_id, realtime=True)
        monitor_supported = monitor.voltage is not None
        return ServoTelemetry(
            online=self.ping(),
            angle=self.angle(),
            voltage_v=values["voltage_v"],  # type: ignore[arg-type]
            current_a=values["current_a"],  # type: ignore[arg-type]
            power_w=values["power_w"],  # type: ignore[arg-type]
            temperature_c=values["temperature_c"],  # type: ignore[arg-type]
            status=values["status"],  # type: ignore[arg-type]
            multi_turn_angle=self.angle(multi_turn=True),
            monitor_supported=monitor_supported,
        )

    def move(
        self,
        angle: float,
        *,
        interval_ms: int | None = None,
        velocity_dps: float | None = None,
        acceleration_ms: int = 20,
        deceleration_ms: int = 20,
        power_mw: int = 0,
        multi_turn: bool = False,
        wait: bool = True,
        timeout: float = 15.0,
    ) -> float | None:
        """Move to an angle using interval, velocity, or SDK-estimated timing.

        ``angle`` is degrees. Single-turn range is -180 to 180. Multi-turn
        feedback is firmware-dependent, so waiting for it requires a response
        to the SDK's multi-turn query command.
        """
        if interval_ms is not None and velocity_dps is not None:
            raise ValueError("interval_ms 与 velocity_dps 只能指定一个")
        if not multi_turn and not -180.0 <= angle <= 180.0:
            raise ValueError("单圈角度必须在 -180 到 180 度之间")
        manager = self._require_manager()
        if not manager.set_servo_angle(
            self.servo_id,
            angle,
            is_mturn=multi_turn,
            interval=interval_ms,
            velocity=velocity_dps,
            t_acc=acceleration_ms,
            t_dec=deceleration_ms,
            power=power_mw,
        ):
            raise ServoError("舵机拒绝角度控制请求")
        if not wait:
            return None
        return self.wait_for_angle(angle, multi_turn=multi_turn, timeout=timeout)

    def wait_for_angle(
        self,
        target_angle: float,
        *,
        multi_turn: bool = False,
        timeout: float = 15.0,
        tolerance: float = 0.5,
        interval: float = 0.05,
    ) -> float:
        """Wait for two consecutive position samples within ``tolerance``.

        The connected servo reports status bit 0 as set even when stationary,
        so position feedback is more reliable than the SDK's removed ``wait``
        method or status-bit polling.
        """
        deadline = time.monotonic() + timeout
        settled = 0
        last_angle: float | None = None
        while time.monotonic() < deadline:
            last_angle = self.angle(multi_turn=multi_turn)
            if last_angle is None:
                raise ServoError("等待舵机位置时未收到响应")
            if abs(last_angle - target_angle) <= tolerance:
                settled += 1
                if settled >= 2:
                    return last_angle
            else:
                settled = 0
            time.sleep(interval)
        raise ServoTimeout(
            "{:.1f} 秒内未到达目标 {:.1f}°，最后反馈 {}".format(
                timeout, target_angle, last_angle
            )
        )

    def set_damping(self, power_mw: int = 0) -> None:
        """Enter damping mode. This changes holding behaviour."""
        self._require_manager().set_damping(self.servo_id, power_mw)

    def release_torque(self) -> None:
        """Release holding torque; the mechanical load may drop or rotate."""
        self._require_manager().disable_torque(self.servo_id)

    def stop(self, mode: str = "hold", power_mw: int = 0) -> None:
        """Stop in ``release``, ``hold``, or ``damping`` mode."""
        methods = {"release": 0x10, "hold": 0x11, "damping": 0x12}
        if mode not in methods:
            raise ValueError("mode 必须是 release、hold 或 damping")
        self._require_manager().stop_on_control_mode(
            self.servo_id, methods[mode], power_mw
        )

    def set_origin(self) -> None:
        """Set the current position as origin for supported absolute encoders."""
        self._require_manager().set_origin_point(self.servo_id)

    def reset_multi_turn_count(self) -> None:
        """Reset multi-turn count; only use while torque is released."""
        self._require_manager().reset_multi_turn_angle(self.servo_id)

    def read_data(self, address: int) -> bytes | None:
        """Read a raw SDK memory-table entry."""
        return self._require_manager().read_data(self.servo_id, address, realtime=True)

    def write_data(self, address: int, content: bytes) -> bool:
        """Write a raw SDK memory-table entry. This may persist across reboot."""
        return self._require_manager().write_data(
            self.servo_id, address, content, realtime=True
        )

    def reset_user_data(self) -> bool:
        """Restore user data defaults. The SDK documents that this resets ID to 0."""
        return self._require_manager().reset_user_data(self.servo_id)

    def begin_async(self) -> None:
        self._require_manager().begin_async()

    def end_async(self, cancel: bool = False) -> None:
        self._require_manager().end_async(1 if cancel else 0)

    def sync_move_by_interval(
        self, targets: dict[int, float], interval_ms: int, power_mw: int = 0
    ) -> None:
        """Send a V316+ synchronous single-turn move to multiple servo IDs."""
        if not targets:
            raise ValueError("targets 不能为空")
        if not 0 < interval_ms <= 65535:
            raise ValueError("interval_ms 必须在 1 到 65535 之间")
        commands = []
        for servo_id, angle in targets.items():
            if not 0 <= servo_id <= 253 or not -180.0 <= angle <= 180.0:
                raise ValueError("同步单圈目标必须使用有效 ID 和 -180 到 180 度")
            commands.append(
                struct.pack("<BhHH", servo_id, int(angle * 10), interval_ms, power_mw)
            )
        self._require_manager().send_sync_angle(8, len(commands), commands)

    def diagnose(self) -> ServoTelemetry:
        """Run only non-destructive communication and feedback checks."""
        return self.telemetry()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fashion Star UART servo controller")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--id", type=int, default=0, dest="servo_id")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("diagnose", help="只读诊断：通信、角度与遥测")

    scan = subparsers.add_parser("scan", help="扫描在线舵机 ID，不运动")
    scan.add_argument("--start-id", type=int, default=0)
    scan.add_argument("--end-id", type=int, default=253)

    move = subparsers.add_parser("move", help="执行单圈或多圈角度运动")
    move.add_argument("angle", type=float)
    move.add_argument("--interval-ms", type=int)
    move.add_argument("--velocity-dps", type=float)
    move.add_argument("--power-mw", type=int, default=0)
    move.add_argument("--multi-turn", action="store_true")
    move.add_argument("--no-wait", action="store_true")

    damping = subparsers.add_parser("damping", help="进入阻尼模式")
    damping.add_argument("--power-mw", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    servo = FashionStarServo(
        servo_id=args.servo_id, port=args.port, baudrate=args.baudrate
    )
    try:
        servo.open(verify=args.action != "scan")
        if args.action == "diagnose":
            print(asdict(servo.diagnose()))
        elif args.action == "scan":
            print(servo.scan(args.start_id, args.end_id))
        elif args.action == "move":
            print(
                servo.move(
                    args.angle,
                    interval_ms=args.interval_ms,
                    velocity_dps=args.velocity_dps,
                    power_mw=args.power_mw,
                    multi_turn=args.multi_turn,
                    wait=not args.no_wait,
                )
            )
        elif args.action == "damping":
            servo.set_damping(args.power_mw)
    finally:
        servo.close()


if __name__ == "__main__":
    main()
