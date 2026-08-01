from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


workspace = Path(__file__).parent
project = Path(r"C:\Users\user\Desktop\diansai2026")
sys.path.insert(0, str(workspace / "round_corner_work"))

from puzzle_solver.solver import SolverConfig, _score_card_features
from puzzle_solver.vision import Calibration, VisionConfig, detect_pieces, extract_piece_features


reports = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in (
        workspace / "examples_old134_card_05_v2.json",
        workspace / "examples_new84_card_5s.json",
    )
]
config_document = json.loads((project / "viz" / "vision_config.json").read_text(encoding="utf-8"))
vision_config = VisionConfig(**config_document["vision"])
selected = {
    "104855", "120348",
    "133320", "135025", "153126", "162227", "171541",
    "172048", "173405", "173858", "174415", "174628",
}

for record in (record for report in reports for record in report["records"]):
    if not record.get("solved") or not any(token in record["name"] for token in selected):
        continue
    image = cv2.imread(str(project / "examples" / record["name"]))
    calibration = Calibration.from_dict(config_document, image.shape)
    detection = detect_pieces(image, calibration, vision_config)
    corrected = calibration.undistort_image(image)
    features = extract_piece_features(corrected, detection.pieces, calibration)
    poses = [
        SimpleNamespace(piece_index=index, polygon=np.asarray(piece["target_polygon_mm"], dtype=float))
        for index, piece in enumerate(record["solution_pieces"])
    ]
    width, height = record["rectangle"]
    *_, details = _score_card_features(
        poses, width, height, features, config=SolverConfig()
    )
    rounded = details["trusted_rounded_corners"]
    distances = [round(item["distance_mm"], 2) for item in rounded]
    roundness = [round(item["roundness_mm"], 2) for item in rounded]
    edge_lengths = [
        [round(item["previous_edge_length_mm"], 1), round(item["following_edge_length_mm"], 1)]
        for item in rounded
    ]
    misplaced = sum(item["misplaced"] for item in rounded)
    print(record["name"], "count", len(rounded), "misplaced", misplaced,
          "distances", distances, "roundness", roundness, "edges", edge_lengths)
