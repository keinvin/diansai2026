# Rectangle puzzle geometry solver

This module implements the geometry stage for the 2026 E-problem puzzle device.
It accepts one to four measured polygons in millimetres and returns a rectangular
target plus one rigid target pose for every piece.

The search uses the competition constraint that every piece owns at least one
edge of the target rectangle. It does not require internal seams to have a
one-to-one edge match, so a long edge may contact several short edges.

## Input

Each polygon must list its vertices in boundary order. Clockwise and
counter-clockwise polygons are both accepted.

```json
{
  "target_origin_mm": [55.0, 190.0],
  "pieces": [
    {"id": "piece_0", "polygon_mm": [[10, 10], [30, 10], [20, 30]]}
  ]
}
```

`target_origin_mm` is the desired top-left corner of the result in writing-machine
coordinates. The output transform already includes this origin.

## Run the fixed-piece example

From the directory containing `puzzle_solver`:

```powershell
python -m puzzle_solver puzzle_solver/examples/fixed_four.json -o solution.json
```

The important output fields are:

- `rectangle`: target origin, width, and height.
- `rotation_deg`: commanded planar rotation relative to the observed piece.
- `translation_mm`: rigid-transform translation after rotation.
- `target_polygon_mm`: target vertices in writing-machine coordinates.

For an observed pickup point `g`, compute its release location with the returned
matrix and translation:

```text
g_target = rotation_matrix * g + translation_mm
```

## Tests

```powershell
python -m unittest puzzle_solver.tests.test_solver
```

The current solver is geometry-only. Card-face texture scoring should rank the
best geometric candidates in a later stage rather than changing this interface.

For camera-derived contours with roughly 0.5 mm or less vertex error, start with:

```json
{
  "solver": {
    "grid_mm": 1.0,
    "dimension_area_tolerance": 0.06,
    "max_hole_ratio": 0.04,
    "max_overlap_ratio": 0.003
  }
}
```

Tighten these tolerances after camera-to-machine calibration is stable. Loose
tolerances increase the number of geometrically ambiguous assemblies.
