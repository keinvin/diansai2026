import json

import numpy as np

from puzzle_solver.coordinates import A4ToGrblTransform


def test_affine_fit_supports_xy_swap_and_offset():
    a4 = np.asarray([[0, 0], [210, 0], [0, 297], [210, 297]], dtype=float)
    # GRBL X follows A4 Y, GRBL Y follows A4 X.
    grbl = np.column_stack((100 + a4[:, 1], 200 + a4[:, 0]))
    transform = A4ToGrblTransform.fit(a4, grbl)

    assert np.allclose(transform.to_grbl([[20, 30]]), [[130, 220]], atol=1e-4)
    assert np.allclose(transform.to_a4([[130, 220]]), [[20, 30]], atol=1e-4)
    assert np.all(transform.reprojection_error_mm(a4, grbl) < 1e-4)


def test_load_initial_calibration(tmp_path):
    path = tmp_path / "samples.json"
    matrix = [[0.0, 1.0, -29.0], [-1.0, 0.0, 9.5], [0.0, 0.0, 1.0]]
    path.write_text(
        json.dumps({"initial_calibration": {"a4_to_grbl_affine_matrix": matrix}}),
        encoding="utf-8",
    )
    transform = A4ToGrblTransform.load_initial_calibration(path)
    assert np.allclose(transform.to_grbl([[120.0, 130.0]]), [[101.0, -110.5]])
