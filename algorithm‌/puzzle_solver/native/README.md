# Native rectangle search

This optional C++17 library accelerates only the bit-mask MRV/DFS stage. Vision,
pose generation, scoring, safety offsets, and the public Python API remain in
Python. When the library is absent or fails to load, the solver automatically
uses the Python implementation.

The repository includes `libpuzzle_solver_native.so` for Linux AArch64, matching
the checked-in `.venv` architecture, plus `puzzle_solver_native.dll` for Windows
x86-64 development. Rebuild on the target when the OS image, compiler ABI, or CPU
architecture changes.

Build in place on Linux or Windows:

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j
```

Run these commands from the `puzzle_solver` directory. The Python loader searches
`native/build`, `native/build/Release`, and the package directory. A library built
elsewhere can be selected with `PUZZLE_SOLVER_NATIVE_LIBRARY=/absolute/path/to/lib`.

On the Linux ARM target, install `cmake`, `g++`, and `make` or `ninja` first. The
library has no Python, NumPy, or OpenCV build dependency.

Set `native_search_required` to `true` in the solver configuration during
deployment checks. Successful results expose `metrics.search_backend` as `cpp`;
otherwise the normal automatic fallback reports `python`.
