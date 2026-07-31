#!/usr/bin/env python3
"""Coordinated CoreXY, Z-axis, and servo action executor."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

try:  # Works both as ``python motion/motion_exec.py`` and package import.
    from .core_xy import CoreXY, DEFAULT_BAUDRATE as GRBL_BAUDRATE, DEFAULT_PORT as GRBL_PORT
    from .mag import MagExecutor
    from .servo import FashionStarServo, DEFAULT_BAUDRATE as SERVO_BAUDRATE, DEFAULT_PORT as SERVO_PORT
except ImportError:
    from core_xy import CoreXY, DEFAULT_BAUDRATE as GRBL_BAUDRATE, DEFAULT_PORT as GRBL_PORT
    from mag import MagExecutor
    from servo import FashionStarServo, DEFAULT_BAUDRATE as SERVO_BAUDRATE, DEFAULT_PORT as SERVO_PORT


@dataclass(frozen=True)
class MotionTestConfig:
    """Physical targets for one pick/actuate/retract test.

    ``z_down_mm`` is an *absolute GRBL work coordinate*. Set its sign to suit
    the installed Z axis; it is deliberately required instead of guessed.
    """

    target_x_mm: float
    target_y_mm: float
    z_down_mm: float
    xy_feed_mm_min: float = 1500.0
    z_feed_mm_min: float = 100.0
    servo_angle_deg: float = 90.0
    servo_interval_ms: int = 800
    servo_multi_turn: bool = True


@dataclass(frozen=True)
class PickPlaceConfig:
    """Motion parameters for executing a complete puzzle solution."""

    z_down_mm: float = 30.0
    z_retracted_mm: float = 0.0
    xy_feed_mm_min: float = 1500.0
    z_feed_mm_min: float = 100.0
    servo_direction: float = 1.0
    servo_zero_deg: float = 0.0
    servo_interval_ms: int = 800
    magnet_settle_s: float = 0.20
    return_xy_zero: bool = True
    move_xy_zero_before_start: bool = False
    optimize_piece_order: bool = True
    reset_servo_between_pieces: bool = False
    servo_soft_limit_deg: float = 360.0


class A4ToGrblLike(Protocol):
    def to_grbl(self, a4_points_mm: Sequence[Sequence[float]]): ...


def _shortest_angle_deg(angle: float) -> float:
    normalized = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 and angle > 0.0 else normalized


def build_pick_place_plan(
    solution: dict,
    transform: A4ToGrblLike,
    config: PickPlaceConfig | None = None,
) -> list[dict]:
    """Convert solver A4 pickup points into an executable GRBL plan."""

    config = config or PickPlaceConfig()
    if config.servo_direction not in (-1.0, 1.0):
        raise ValueError("servo_direction 必须是 +1 或 -1")
    plan: list[dict] = []
    for index, piece in enumerate(solution.get("pieces", [])):
        try:
            source_a4 = [float(value) for value in piece["pickup_source_mm"]]
            target_a4 = [float(value) for value in piece["pickup_target_mm"]]
            rotation_deg = float(piece["rotation_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 片缺少有效的吸取点或旋转角") from exc
        if len(source_a4) != 2 or len(target_a4) != 2:
            raise ValueError(f"第 {index + 1} 片吸取点必须是二维 A4 坐标")
        source_grbl = transform.to_grbl([source_a4])[0]
        target_grbl = transform.to_grbl([target_a4])[0]
        servo_delta = _shortest_angle_deg(rotation_deg * config.servo_direction)
        plan.append(
            {
                "id": str(piece.get("id", index)),
                "pickup_source_a4_mm": source_a4,
                "pickup_target_a4_mm": target_a4,
                "pickup_source_grbl_mm": [float(source_grbl[0]), float(source_grbl[1])],
                "pickup_target_grbl_mm": [float(target_grbl[0]), float(target_grbl[1])],
                "algorithm_rotation_deg": rotation_deg,
                "servo_delta_deg": servo_delta,
                "servo_target_deg": config.servo_zero_deg + servo_delta,
            }
        )
    if not plan:
        raise ValueError("拼图解中没有可执行碎片")
    if config.optimize_piece_order and len(plan) > 1:
        origin = (0.0, 0.0)

        def distance(first: Sequence[float], second: Sequence[float]) -> float:
            return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))

        def route_length(route: Sequence[dict]) -> float:
            total = 0.0
            current: Sequence[float] = origin
            for step in route:
                source = step["pickup_source_grbl_mm"]
                target = step["pickup_target_grbl_mm"]
                total += distance(current, source) + distance(source, target)
                current = target
            if config.return_xy_zero:
                total += distance(current, origin)
            return total

        plan = list(min(itertools.permutations(plan), key=route_length))
    return plan


class MotionExecutor:
    """Execute the coordinated test sequence using GRBL work coordinates.

    By default the current position becomes the GRBL work origin during open().
    This class does not run ``$H`` or move an axis while initializing.
    """

    def __init__(
        self,
        *,
        grbl_port: str = GRBL_PORT,
        grbl_baudrate: int = GRBL_BAUDRATE,
        grbl_step_idle_delay_ms: int = 255,
        grbl_set_work_origin_on_init: bool = True,
        grbl_release_on_close: bool = True,
        servo_port: str = SERVO_PORT,
        servo_baudrate: int = SERVO_BAUDRATE,
        servo_id: int = 1,
        mag_chip: str = "gpiochip1",
        mag_line: int = 7,
        corexy: CoreXY | None = None,
        servo: FashionStarServo | None = None,
        mag: MagExecutor | None = None,
    ) -> None:
        self.corexy = corexy or CoreXY(grbl_port, grbl_baudrate)
        if not 0 <= int(grbl_step_idle_delay_ms) <= 255:
            raise ValueError("grbl_step_idle_delay_ms 必须在 0 到 255 之间")
        self._grbl_step_idle_delay_ms = int(grbl_step_idle_delay_ms)
        self._grbl_set_work_origin_on_init = bool(grbl_set_work_origin_on_init)
        self._grbl_release_on_close = bool(grbl_release_on_close)
        self._grbl_initialized = False
        # ID 1 is the bus servo currently detected on this machine. It remains
        # configurable for a future wiring/ID change.
        self.servo = servo or FashionStarServo(servo_id, servo_port, servo_baudrate)
        self.mag = mag
        self._mag_chip = mag_chip
        self._mag_line = mag_line
        self._timing_callback: Callable[[str, float], None] | None = None

    def open(self) -> "MotionExecutor":
        if self.corexy.uart is None:
            self.corexy.open()
        if not self._grbl_initialized:
            self.corexy.command(f"$1={self._grbl_step_idle_delay_ms}")
            if self._grbl_set_work_origin_on_init:
                self.corexy.set_work_position(0.0, 0.0, 0.0)
            self._grbl_initialized = True
        if self.servo.uart is None:
            self.servo.open()
        if self.mag is None:
            self.mag = MagExecutor(self._mag_chip, self._mag_line)
        return self

    def close(self) -> None:
        first_error: Exception | None = None

        def cleanup(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if self.mag is not None:
            cleanup(self.mag.close)
        cleanup(self.servo.close)
        if (
            self._grbl_initialized
            and self._grbl_release_on_close
            and self.corexy.uart is not None
        ):
            cleanup(lambda: self.corexy.command("$1=0"))
            cleanup(self.corexy.soft_reset)
        cleanup(self.corexy.close)
        self._grbl_initialized = False

        if first_error is not None:
            raise first_error

    def __enter__(self) -> "MotionExecutor":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _move_xy(self, x: float, y: float, feed: float) -> None:
        started = time.perf_counter()
        try:
            self.corexy.move_to(x=x, y=y, feed=feed, rapid=False)
            self.corexy.wait_until_position(x=x, y=y)
        finally:
            self._record_timing("xy", started)

    def _move_z(self, z: float, feed: float) -> None:
        started = time.perf_counter()
        try:
            self.corexy.move_to(z=z, feed=feed, rapid=False)
            self.corexy.wait_until_position(z=z)
        finally:
            self._record_timing("z", started)

    def _record_timing(self, stage: str, started: float) -> None:
        if self._timing_callback is not None:
            self._timing_callback(stage, time.perf_counter() - started)

    def _move_servo(self, angle: float, **kwargs) -> float | None:
        started = time.perf_counter()
        try:
            return self.servo.move(angle, **kwargs)
        finally:
            self._record_timing("servo", started)

    def _wait_for_magnet(self, seconds: float) -> None:
        started = time.perf_counter()
        try:
            time.sleep(seconds)
        finally:
            self._record_timing("magnet_wait", started)

    def _magnet(self) -> MagExecutor:
        if self.mag is None:
            raise RuntimeError("电磁铁 GPIO 尚未初始化")
        return self.mag

    def _safe_return(self, config: MotionTestConfig) -> list[Exception]:
        """Best-effort recovery in the mechanically safest order."""
        errors: list[Exception] = []
        for action in (
            lambda: self._move_z(0.0, config.z_feed_mm_min),
            self._magnet().off,
            lambda: self._move_servo(
                0.0,
                interval_ms=config.servo_interval_ms,
                multi_turn=config.servo_multi_turn,
                wait=True,
            ),
            lambda: self._move_xy(0.0, 0.0, config.xy_feed_mm_min),
        ):
            try:
                action()
            except Exception as exc:  # Preserve the original failure.
                errors.append(exc)
        return errors

    def test(self, config: MotionTestConfig) -> None:
        """Run a full pick/place test with magnet, Z, servo, and CoreXY.

        Sequence: zero → XY target → Z down → magnet on → Z up → servo +90
        → XY zero → Z down → magnet off → Z up → servo 0. The machine starts
        by retracting Z and moving XY to work zero. On any failure, the same
        safe-return sequence is attempted before re-raising.
        """
        if config.xy_feed_mm_min <= 0 or config.z_feed_mm_min <= 0:
            raise ValueError("XY 与 Z 的进给速度必须大于 0")
        self.open()
        try:
            # Start at the configured work origin. Retract before XY motion.
            self._move_z(0.0, config.z_feed_mm_min)
            self._move_xy(0.0, 0.0, config.xy_feed_mm_min)

            self._move_xy(config.target_x_mm, config.target_y_mm, config.xy_feed_mm_min)
            self._move_z(config.z_down_mm, config.z_feed_mm_min)
            self._magnet().on()
            self._move_z(0.0, config.z_feed_mm_min)
            self._move_servo(
                config.servo_angle_deg,
                interval_ms=config.servo_interval_ms,
                multi_turn=config.servo_multi_turn,
                wait=True,
            )

            self._move_xy(0.0, 0.0, config.xy_feed_mm_min)
            self._move_z(config.z_down_mm, config.z_feed_mm_min)
            self._magnet().off()
            self._move_z(0.0, config.z_feed_mm_min)
            self._move_servo(
                0.0,
                interval_ms=config.servo_interval_ms,
                multi_turn=config.servo_multi_turn,
                wait=True,
            )
        except Exception:
            self._safe_return(config)
            raise

    def _safe_return_pick_place(self, config: PickPlaceConfig) -> list[Exception]:
        errors: list[Exception] = []
        for action in (
            lambda: self._move_z(config.z_retracted_mm, config.z_feed_mm_min),
            self._magnet().off,
            lambda: self._move_servo(
                config.servo_zero_deg,
                interval_ms=config.servo_interval_ms,
                multi_turn=True,
                wait=True,
            ),
            lambda: self._move_xy(0.0, 0.0, config.xy_feed_mm_min),
        ):
            try:
                action()
            except Exception as exc:
                errors.append(exc)
        return errors

    def execute_solution(
        self,
        solution: dict,
        transform: A4ToGrblLike,
        config: PickPlaceConfig | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        timing_callback: Callable[[str, float], None] | None = None,
    ) -> list[dict]:
        """Pick, rotate, and place every piece in a solver result."""

        config = config or PickPlaceConfig()
        if config.xy_feed_mm_min <= 0 or config.z_feed_mm_min <= 0:
            raise ValueError("XY 与 Z 的进给速度必须大于 0")
        if config.magnet_settle_s < 0:
            raise ValueError("magnet_settle_s 不能小于 0")
        plan = build_pick_place_plan(solution, transform, config)
        total = len(plan)
        self.open()
        previous_timing_callback = self._timing_callback
        self._timing_callback = timing_callback
        try:
            self._move_z(config.z_retracted_mm, config.z_feed_mm_min)
            if config.move_xy_zero_before_start:
                self._move_xy(0.0, 0.0, config.xy_feed_mm_min)
            current_servo_angle = self.servo.angle(multi_turn=True)
            if current_servo_angle is None:
                self._move_servo(
                    config.servo_zero_deg,
                    interval_ms=config.servo_interval_ms,
                    multi_turn=True,
                    wait=True,
                )
                current_servo_angle = config.servo_zero_deg
            for index, step in enumerate(plan, start=1):
                servo_target = current_servo_angle + step["servo_delta_deg"]
                exceeds_soft_limit = (
                    abs(servo_target - config.servo_zero_deg)
                    > config.servo_soft_limit_deg
                )
                if exceeds_soft_limit and abs(current_servo_angle - config.servo_zero_deg) > 0.5:
                    self._move_servo(
                        config.servo_zero_deg,
                        interval_ms=config.servo_interval_ms,
                        multi_turn=True,
                        wait=True,
                    )
                    current_servo_angle = config.servo_zero_deg
                    servo_target = current_servo_angle + step["servo_delta_deg"]
                if progress is not None:
                    progress(index, total, f"正在抓取 {step['id']}")
                source_x, source_y = step["pickup_source_grbl_mm"]
                self._move_xy(source_x, source_y, config.xy_feed_mm_min)
                self._move_z(config.z_down_mm, config.z_feed_mm_min)
                self._magnet().on()
                self._wait_for_magnet(config.magnet_settle_s)
                self._move_z(config.z_retracted_mm, config.z_feed_mm_min)

                if progress is not None:
                    progress(index, total, f"正在搬运 {step['id']} 到放置点")
                target_x, target_y = step["pickup_target_grbl_mm"]
                self._move_xy(target_x, target_y, config.xy_feed_mm_min)
                if abs(servo_target - current_servo_angle) > 0.05:
                    if progress is not None:
                        progress(index, total, f"正在放置点旋转 {step['id']}")
                    self._move_servo(
                        servo_target,
                        interval_ms=config.servo_interval_ms,
                        multi_turn=True,
                        wait=True,
                    )
                    current_servo_angle = servo_target
                self._move_z(config.z_down_mm, config.z_feed_mm_min)
                self._magnet().off()
                self._wait_for_magnet(config.magnet_settle_s)
                self._move_z(config.z_retracted_mm, config.z_feed_mm_min)
                if config.reset_servo_between_pieces:
                    self._move_servo(
                        config.servo_zero_deg,
                        interval_ms=config.servo_interval_ms,
                        multi_turn=True,
                        wait=True,
                    )
                    current_servo_angle = config.servo_zero_deg
            if abs(current_servo_angle - config.servo_zero_deg) > 0.5:
                self._move_servo(
                    config.servo_zero_deg,
                    interval_ms=config.servo_interval_ms,
                    multi_turn=True,
                    wait=True,
                )
            if config.return_xy_zero:
                self._move_xy(0.0, 0.0, config.xy_feed_mm_min)
            if progress is not None:
                progress(total, total, "拼图动作完成")
            return plan
        except Exception:
            self._safe_return_pick_place(config)
            raise
        finally:
            self._timing_callback = previous_timing_callback


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 CoreXY + Z + 舵机完整动作测试")
    parser.add_argument("--x", type=float, required=True, help="目标 GRBL X (mm)")
    parser.add_argument("--y", type=float, required=True, help="目标 GRBL Y (mm)")
    parser.add_argument("--z-down", type=float, required=True, help="Z 下压后的绝对 GRBL 坐标 (mm)")
    parser.add_argument("--xy-feed", type=float, default=15000.0)
    parser.add_argument("--z-feed", type=float, default=5000.0)
    parser.add_argument("--servo-angle", type=float, default=90.0)
    parser.add_argument("--servo-interval", type=int, default=800, help="舵机动作周期 (ms)")
    parser.add_argument("--grbl-port", default=GRBL_PORT)
    parser.add_argument("--grbl-baudrate", type=int, default=GRBL_BAUDRATE)
    parser.add_argument("--servo-port", default=SERVO_PORT)
    parser.add_argument("--servo-baudrate", type=int, default=SERVO_BAUDRATE)
    parser.add_argument("--servo-id", type=int, default=1)
    parser.add_argument("--mag-chip", default="gpiochip1")
    parser.add_argument("--mag-line", type=int, default=7)
    args = parser.parse_args()

    config = MotionTestConfig(
        args.x,
        args.y,
        args.z_down,
        args.xy_feed,
        args.z_feed,
        args.servo_angle,
        args.servo_interval,
    )
    with MotionExecutor(
        grbl_port=args.grbl_port,
        grbl_baudrate=args.grbl_baudrate,
        servo_port=args.servo_port,
        servo_baudrate=args.servo_baudrate,
        servo_id=args.servo_id,
        mag_chip=args.mag_chip,
        mag_line=args.mag_line,
    ) as executor:
        executor.test(config)
    print("动作测试完成：已回到 GRBL 工作零点，舵机已复位 0°。")


if __name__ == "__main__":
    main()
