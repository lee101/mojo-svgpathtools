from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get(
    "MOJO_SVGPATHTOOLS_LIB",
    os.path.join(ROOT, "dist", "libmojo-svgpathtools.so"),
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msp_bezier_points": ([I, I, I, I, I], None),
    "msp_bezier_derivatives": ([I, I, I, I, I, I], None),
    "msp_bezier_lengths": ([I, I, I, F, F, I, I], None),
    "msp_arc_points": ([F, F, F, F, F, F, F, I, I, I], None),
    "msp_arc_derivatives": ([F, F, F, F, F, I, I, I, I], None),
    "msp_arc_lengths": ([I, I, F, F, I, I], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if os.path.exists(LIB):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        raise RuntimeError(
            f"compiled library not found at {LIB}; set MOJO_SVGPATHTOOLS_LIB"
        )
    subprocess.run(["bash", script], check=True, cwd=ROOT)
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        raise TypeError("parameter values must be real, not complex")
    if source.dtype.kind not in "biuf":
        raise TypeError("parameter values must be a real numeric array")
    result = np.ascontiguousarray(source, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("parameter values must be finite float64 values")
    return result


def addr(array: np.ndarray) -> int:
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError("FFI buffers must use float64 or complex128 storage")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a nonempty buffer")
    return address
