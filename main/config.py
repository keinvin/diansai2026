"""Load and save the standalone main-flow configuration."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")
LEGACY_CONFIG_PATH = PROJECT_ROOT / "viz" / "vision_config.json"
ALGORITHM_ROOT = next(PROJECT_ROOT.glob("algorithm*"))
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puzzle_solver.solver import SolverConfig  # noqa: E402
from puzzle_solver.vision import VisionConfig  # noqa: E402
from motion.motion_exec import PickPlaceConfig  # noqa: E402


def default_config() -> dict:
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
        "hardware": {
            "grbl_port": "/dev/ttyUSB0",
            "grbl_baudrate": 115200,
            "grbl_step_idle_delay_ms": 255,
            "grbl_set_work_origin_on_init": True,
            "grbl_release_on_close": True,
            "servo_port": "/dev/ttyUSB1",
            "servo_baudrate": 115200,
            "servo_id": 1,
            "mag_chip": "gpiochip1",
            "mag_line": 7,
        },
        "paths": {
            "a4_grbl_calibration": "data/a4_grbl_calibration_samples.json"
        },
        "timing": {
            "enabled": True,
            "log_enabled": True,
            "log_file": "data/logs/main_timing.jsonl",
        },
    }


def load_config(path: Path = CONFIG_PATH) -> dict:
    source = path
    if not source.exists() and path == CONFIG_PATH and LEGACY_CONFIG_PATH.exists():
        source = LEGACY_CONFIG_PATH
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        document = {}

    defaults = default_config()
    defaults.update(document)
    if "a4_region" not in document:
        defaults["a4_region"] = "upper"
    for section in ("vision", "solver", "motion", "hardware", "paths", "timing"):
        defaults[section] = {
            **default_config()[section],
            **document.get(section, {}),
        }
    return defaults


def save_config(document: dict, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
