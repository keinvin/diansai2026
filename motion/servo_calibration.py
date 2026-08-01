#!/usr/bin/env python3
"""Measure Fashion Star servo position error and direction-dependent backlash."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from servo import FashionStarServo, ServoError


@dataclass
class Measurement:
    direction: str
    target_deg: float
    actual_deg: float
    error_deg: float
    settle_seconds: float
    sample_span_deg: float


def wait_for_settle(
    servo: FashionStarServo,
    *,
    min_wait: float,
    timeout: float,
    samples_required: int,
    max_span: float,
) -> tuple[float, float, float]:
    """Return stable position, elapsed time, and final sample span.

    This measures actual feedback stability instead of treating the SDK's
    status bit as a reliable completion signal.
    """
    started = time.monotonic()
    readings: list[float] = []
    while time.monotonic() - started < timeout:
        angle = servo.angle()
        if angle is None:
            raise ServoError("读取舵机位置失败")
        readings.append(angle)
        if len(readings) > samples_required:
            readings.pop(0)
        elapsed = time.monotonic() - started
        span = max(readings) - min(readings)
        if (
            elapsed >= min_wait
            and len(readings) == samples_required
            and span <= max_span
        ):
            return statistics.mean(readings), elapsed, span
        time.sleep(0.1)
    raise ServoError("{:.1f} 秒内未获得稳定的位置反馈".format(timeout))


def measure_path(
    servo: FashionStarServo,
    points: list[float],
    direction: str,
    interval_ms: int,
    power_mw: int,
    settle_samples: int,
    settle_span: float,
) -> list[Measurement]:
    results: list[Measurement] = []
    for target in points:
        servo.move(target, interval_ms=interval_ms, power_mw=power_mw, wait=False)
        actual, elapsed, span = wait_for_settle(
            servo,
            min_wait=interval_ms / 1000.0 + 0.3,
            timeout=interval_ms / 1000.0 + 8.0,
            samples_required=settle_samples,
            max_span=settle_span,
        )
        result = Measurement(
            direction=direction,
            target_deg=target,
            actual_deg=actual,
            error_deg=actual - target,
            settle_seconds=elapsed,
            sample_span_deg=span,
        )
        results.append(result)
        print(
            "{:<7} target={:7.1f} actual={:7.2f} error={:+6.2f} "
            "settle={:4.1f}s".format(
                direction, target, actual, result.error_deg, elapsed
            )
        )
    return results


def analyze(results: list[Measurement], constant_threshold: float) -> dict[str, object]:
    errors = [item.error_deg for item in results]
    grouped: dict[float, dict[str, Measurement]] = defaultdict(dict)
    for item in results:
        grouped[item.target_deg][item.direction] = item

    hysteresis = {
        target: values["reverse"].actual_deg - values["forward"].actual_deg
        for target, values in grouped.items()
        if "forward" in values and "reverse" in values
    }
    error_range = max(errors) - min(errors)
    max_hysteresis = max((abs(value) for value in hysteresis.values()), default=0.0)
    mean_error = statistics.mean(errors)
    is_constant = (
        error_range <= constant_threshold and max_hysteresis <= constant_threshold
    )
    return {
        "mean_error_deg": mean_error,
        "error_range_deg": error_range,
        "error_stdev_deg": statistics.pstdev(errors),
        "direction_hysteresis_deg": hysteresis,
        "max_hysteresis_deg": max_hysteresis,
        "constant_bias": is_constant,
        "suggested_command_offset_deg": -mean_error if is_constant else None,
    }


def save_results(path: Path, results: list[Measurement], analysis: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=Measurement.__dataclass_fields__)
            writer.writeheader()
            writer.writerows(asdict(item) for item in results)
        return
    path.write_text(
        json.dumps(
            {"measurements": [asdict(item) for item in results], "analysis": analysis},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fashion Star servo multi-point error test")
    parser.add_argument("--port", default="/dev/diansai-servo")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--id", type=int, default=1, dest="servo_id")
    parser.add_argument(
        "--points",
        type=float,
        nargs="+",
        default=[-80.0, -40.0, 0.0, 40.0, 80.0],
        help="单圈测试角度，默认 -80 -40 0 40 80",
    )
    parser.add_argument("--interval-ms", type=int, default=1500)
    parser.add_argument("--power-mw", type=int, default=0)
    parser.add_argument("--settle-samples", type=int, default=4)
    parser.add_argument("--settle-span", type=float, default=0.3)
    parser.add_argument("--constant-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("servo_error_report.json"))
    parser.add_argument("--no-restore", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    points = sorted(set(args.points))
    if len(points) < 2 or any(point < -180 or point > 180 for point in points):
        raise ValueError("至少提供两个 -180 到 180 度之间的测试点")
    if args.interval_ms < 100:
        raise ValueError("interval-ms 至少为 100")

    with FashionStarServo(args.servo_id, args.port, args.baudrate) as servo:
        initial_angle = servo.angle()
        if initial_angle is None:
            raise ServoError("无法读取初始位置")
        print("initial_angle={:.2f}".format(initial_angle))
        try:
            results = measure_path(
                servo,
                points,
                "forward",
                args.interval_ms,
                args.power_mw,
                args.settle_samples,
                args.settle_span,
            )
            results.extend(
                measure_path(
                    servo,
                    list(reversed(points)),
                    "reverse",
                    args.interval_ms,
                    args.power_mw,
                    args.settle_samples,
                    args.settle_span,
                )
            )
        finally:
            if not args.no_restore:
                servo.move(
                    initial_angle,
                    interval_ms=args.interval_ms,
                    power_mw=args.power_mw,
                    wait=False,
                )
                print("restore_command={:.2f}".format(initial_angle))

    analysis = analyze(results, args.constant_threshold)
    save_results(args.output, results, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("report={}".format(args.output))


if __name__ == "__main__":
    main()
