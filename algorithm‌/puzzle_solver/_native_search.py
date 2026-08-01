"""ctypes bridge for the optional native bit-mask puzzle search."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class NativeSearchError(RuntimeError):
    """The native solver rejected a valid-looking request or failed internally."""


class NativeSearchUnavailable(RuntimeError):
    """The native shared library cannot be loaded on this system."""


@dataclass(frozen=True)
class NativeSolution:
    candidate_indices: tuple[int, ...]
    hole_cells: int
    overlap_cells: int


@dataclass(frozen=True)
class NativeSearchResult:
    solutions: tuple[NativeSolution, ...]
    nodes: int
    timed_out: bool


_UINT64_PTR = ctypes.POINTER(ctypes.c_uint64)
_INT32_PTR = ctypes.POINTER(ctypes.c_int32)
_INT64_PTR = ctypes.POINTER(ctypes.c_int64)
_LIBRARY: ctypes.CDLL | None = None
_LOAD_ERROR: Exception | None = None


def _library_candidates() -> list[Path]:
    configured = os.environ.get("PUZZLE_SOLVER_NATIVE_LIBRARY")
    package_dir = Path(__file__).resolve().parent
    names = ("libpuzzle_solver_native.so", "puzzle_solver_native.dll")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    for directory in (package_dir / "native" / "build", package_dir / "native" / "build" / "Release", package_dir / "native"):
        candidates.extend(directory / name for name in names)
    return candidates


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_ERROR is not None:
        raise NativeSearchUnavailable(str(_LOAD_ERROR)) from _LOAD_ERROR

    errors: list[str] = []
    for path in _library_candidates():
        if not path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(path))
            library.puzzle_search_abi_version.argtypes = []
            library.puzzle_search_abi_version.restype = ctypes.c_int32
            if library.puzzle_search_abi_version() != 1:
                raise NativeSearchUnavailable(f"Unsupported native search ABI: {path}")
            library.puzzle_search.argtypes = [
                _UINT64_PTR,
                _INT32_PTR,
                _INT32_PTR,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int64,
                ctypes.c_double,
                ctypes.c_int32,
                _INT32_PTR,
                _INT64_PTR,
                _INT32_PTR,
                _INT32_PTR,
                _INT32_PTR,
                _INT32_PTR,
            ]
            library.puzzle_search.restype = ctypes.c_int32
            _LIBRARY = library
            return library
        except (OSError, AttributeError, NativeSearchUnavailable) as error:
            errors.append(f"{path}: {error}")
    _LOAD_ERROR = RuntimeError("; ".join(errors) or "native library not found")
    raise NativeSearchUnavailable(str(_LOAD_ERROR)) from _LOAD_ERROR


def _as_pointer(values: np.ndarray, pointer_type):
    return values.ctypes.data_as(pointer_type)


def search_candidate_masks(
    candidates_by_piece: Sequence[Sequence[int]],
    *,
    rectangle_cells: int,
    allowed_holes: int,
    allowed_overlap: int,
    max_nodes: int,
    max_seconds: float,
    max_solutions: int,
) -> NativeSearchResult:
    """Search raster masks with the C++ DFS implementation."""

    if not 1 <= len(candidates_by_piece) <= 4:
        raise ValueError("native search supports one to four pieces")
    if rectangle_cells <= 0 or max_nodes <= 0 or max_solutions <= 0:
        raise ValueError("native search limits must be positive")
    word_count = (int(rectangle_cells) + 63) // 64
    flat_masks: list[int] = []
    offsets = [0]
    cell_counts: list[int] = []
    limit = (1 << int(rectangle_cells)) - 1
    for piece in candidates_by_piece:
        if not piece:
            raise ValueError("every piece must have at least one candidate")
        for mask in piece:
            if mask < 0 or mask & ~limit:
                raise ValueError("candidate mask exceeds the rectangle grid")
            cell_counts.append(int(mask).bit_count())
            flat_masks.append(int(mask))
        offsets.append(len(flat_masks))

    masks = np.zeros((len(flat_masks), word_count), dtype=np.uint64)
    for index, mask in enumerate(flat_masks):
        for word in range(word_count):
            masks[index, word] = (mask >> (word * 64)) & ((1 << 64) - 1)
    offsets_array = np.asarray(offsets, dtype=np.int32)
    counts_array = np.asarray(cell_counts, dtype=np.int32)
    output_count = ctypes.c_int32()
    output_nodes = ctypes.c_int64()
    output_timed_out = ctypes.c_int32()
    output_indices = np.full(
        (int(max_solutions), len(candidates_by_piece)), -1, dtype=np.int32
    )
    output_holes = np.zeros(int(max_solutions), dtype=np.int32)
    output_overlaps = np.zeros(int(max_solutions), dtype=np.int32)

    status = _load_library().puzzle_search(
        _as_pointer(masks, _UINT64_PTR),
        _as_pointer(offsets_array, _INT32_PTR),
        _as_pointer(counts_array, _INT32_PTR),
        len(candidates_by_piece),
        word_count,
        int(rectangle_cells),
        int(allowed_holes),
        int(allowed_overlap),
        int(max_nodes),
        float(max_seconds),
        int(max_solutions),
        ctypes.byref(output_count),
        ctypes.byref(output_nodes),
        ctypes.byref(output_timed_out),
        _as_pointer(output_indices, _INT32_PTR),
        _as_pointer(output_holes, _INT32_PTR),
        _as_pointer(output_overlaps, _INT32_PTR),
    )
    if status != 0:
        raise NativeSearchError(f"native puzzle search failed with status {status}")
    count = int(output_count.value)
    if count < 0 or count > max_solutions:
        raise NativeSearchError("native puzzle search returned an invalid solution count")
    solutions = tuple(
        NativeSolution(
            tuple(int(value) for value in output_indices[index]),
            int(output_holes[index]),
            int(output_overlaps[index]),
        )
        for index in range(count)
    )
    return NativeSearchResult(solutions, int(output_nodes.value), bool(output_timed_out.value))
