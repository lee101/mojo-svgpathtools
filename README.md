# mojo-svgpathtools

`mojo-svgpathtools` is a standalone Mojo port of the compute-heavy geometry in
Python's [`svgpathtools`](https://github.com/mathandy/svgpathtools). It keeps
the familiar `Line`, `QuadraticBezier`, `CubicBezier`, `Arc`, and `Path`
interfaces while moving bulk evaluation and adaptive arclength integration
into a compiled Mojo shared library.

The Python package does not import `svgpathtools` at runtime. The upstream
package is installed in the development environment only for parity tests and
benchmarks.

## Coverage

Covered:

- SVG line, quadratic Bézier, cubic Bézier, and elliptical-arc construction
- scalar `point`, `derivative`, `unit_tangent`, `normal`, and `curvature`
- native vectorized `points` and `derivatives`
- `length` and `ilength`, including partial intervals
- exact segment bounding boxes
- `split`, `cropped`, and `reversed`
- arclength-parameterized `Path` point and differential geometry
- path length, bounding box, cropping, reversal, and continuous subpaths
- all SVG path-data commands: `M/L/H/V/C/S/Q/T/A/Z`, absolute and relative
- line-line and Bézier-line intersections
- native `bezier_lengths` and `arc_lengths` batch APIs

Not covered:

- general Bézier-Bézier, arc, or path intersections
- SVG/XML document reading and writing
- geometric transforms, offsets, area, and radial-range queries
- degenerate SVG arcs with a zero radius

The covered class methods use upstream names and signatures. The plural
`derivatives` method and the two batch-length functions are additions for
workloads large enough to benefit from Mojo.

## Install and run

The repository pins the tested Mojo nightly:

```bash
pixi install
pixi run build
```

`pixi` sets `PYTHONPATH=python` for all tasks. A complete example:

```bash
pixi run python - <<'PY'
import numpy as np
from mojo_svgpathtools import CubicBezier, parse_path

curve = CubicBezier(0j, 1 + 3j, 4 - 2j, 6 + 1j)
print(round(curve.length(), 6))
print(curve.points(np.linspace(0, 1, 4)))

path = parse_path("M 0,0 C 1,3 4,-2 6,1 L 8,1")
print(round(path.length(), 6), path.bbox())
PY
```

Build and verification commands are:

```bash
pixi run build
pixi run test
pixi run bench
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of three runs. Each result is numerically
checked against `svgpathtools` 1.7.2 before timing is reported.

| Operation | Size | Mojo port | svgpathtools | Speedup |
|---|---:|---:|---:|---:|
| CubicBezier.points | 2,000,000 t values | 10.50 ms | 68.91 ms | 6.56x |
| CubicBezier first derivatives | 2,000,000 t values | 8.32 ms | 64.17 ms | 7.71x |
| Arc point evaluation | 500,000 t values | 11.46 ms | 2615.31 ms | 228.13x |
| CubicBezier lengths | 2,000 curves | 25.07 ms | 450.86 ms | 17.99x |
| Arc lengths | 2,000 curves | 26.10 ms | 1380.15 ms | 52.89x |
| Path.length | 5,000 cubic segments | 73.00 ms | 1069.78 ms | 14.66x |

These figures measure bulk geometry. Scalar calls remain in Python because a
sub-microsecond formula does not benefit from crossing a foreign-function
boundary.

No GPU path is included. Point evaluation has too little arithmetic intensity to
justify host/device transfers, while the higher-intensity length kernels are
already 14--53x faster than upstream on CPU and are outside the optimization
target set. These benchmarks cover the CPU implementation only.

## How it works

`src/kernels.mojo` is one compilation unit exported as
`dist/libmojo-svgpathtools.so`. The Python layer loads it with `ctypes`.
Because exported Mojo functions cannot be parametric, NumPy buffers cross the
C ABI as integer addresses and are reconstructed as mutable `Float64`
pointers inside Mojo.

Complex coordinates use interleaved row-major float64 storage:
`[x0, y0, x1, y1, ...]`. Python and NumPy own every allocation; Mojo only
reads input buffers and writes caller-provided output buffers. Bulk Bézier
evaluation writes directly into the final NumPy `complex128` result instead of
allocating and converting an intermediate coordinate matrix.

Curve length uses adaptive 16-point Gauss-Legendre quadrature. A coarse
interval is compared with its two half-intervals and recursively refined,
which handles cusps and high-eccentricity ellipses without imposing the
per-function overhead of SciPy integration. `Path.length` groups segments by
kind and integrates each group in a single FFI call.
