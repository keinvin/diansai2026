"""Shared phase timing for one recognition-to-motion run."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping


STAGE_ORDER = (
    "recognition",
    "solve",
    "xy",
    "z",
    "servo",
    "magnet_wait",
)
STAGE_LABELS = {
    "recognition": "识别",
    "solve": "求解",
    "xy": "XY",
    "z": "Z",
    "servo": "舵机",
    "magnet_wait": "磁铁等待",
}


@dataclass
class StageTimings:
    """Accumulate wall-clock seconds under stable stage names."""

    enabled: bool = True
    seconds: dict[str, float] = field(
        default_factory=lambda: {stage: 0.0 for stage in STAGE_ORDER}
    )

    @classmethod
    def from_dict(
        cls, values: Mapping[str, float] | None, *, enabled: bool = True
    ) -> "StageTimings":
        timings = cls(enabled=enabled)
        if values:
            for stage in STAGE_ORDER:
                timings.seconds[stage] = max(0.0, float(values.get(stage, 0.0)))
        return timings

    def add(self, stage: str, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        if stage not in STAGE_LABELS:
            raise ValueError(f"未知计时阶段：{stage}")
        self.seconds[stage] = self.seconds.get(stage, 0.0) + max(
            0.0, float(elapsed_seconds)
        )

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def to_dict(self) -> dict[str, float]:
        return {stage: float(self.seconds.get(stage, 0.0)) for stage in STAGE_ORDER}

    def total_seconds(self) -> float:
        return sum(self.seconds.get(stage, 0.0) for stage in STAGE_ORDER)

    def format_lines(self, *, include_zero: bool = True) -> list[str]:
        parts = []
        for stage in STAGE_ORDER:
            elapsed = self.seconds.get(stage, 0.0)
            if include_zero or elapsed > 0.0:
                parts.append(f"{STAGE_LABELS[stage]} {elapsed:.3f}s")
        return ["分阶段计时：" + "  |  ".join(parts), f"阶段合计：{self.total_seconds():.3f}s"]


def append_timing_log(
    path: Path,
    event: str,
    timings: Mapping[str, float],
    **fields,
) -> None:
    """Append one self-contained JSON record for later performance analysis."""

    normalized = StageTimings.from_dict(timings).to_dict()
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": str(event),
        "timings_seconds": normalized,
        "stage_total_seconds": sum(normalized.values()),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
