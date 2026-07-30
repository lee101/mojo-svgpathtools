"""Benchmarks against svgpathtools 1.7.2 on identical geometry."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import svgpathtools as upstream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_svgpathtools as mojo  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def main():
    rng = np.random.default_rng(42)
    cases = []

    ts = np.linspace(0, 1, 2_000_000)
    own_cubic = mojo.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j)
    up_cubic = upstream.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j)
    cases.append(
        (
            "CubicBezier.points",
            "2,000,000 t values",
            lambda: own_cubic.points(ts),
            lambda: up_cubic.points(ts),
        )
    )
    cases.append(
        (
            "CubicBezier first derivatives",
            "2,000,000 t values",
            lambda: own_cubic.derivatives(ts),
            lambda: up_cubic.derivative(ts),
        )
    )

    arc_ts = np.linspace(0, 1, 500_000)
    own_arc = mojo.Arc(0j, 100 + 3j, 33, True, True, 30 + 40j)
    up_arc = upstream.Arc(0j, 100 + 3j, 33, True, True, 30 + 40j)
    cases.append(
        (
            "Arc point evaluation",
            "500,000 t values",
            lambda: own_arc.points(arc_ts),
            lambda: np.array([up_arc.point(t) for t in arc_ts]),
        )
    )

    cubic_controls = rng.normal(size=(2_000, 4, 2))

    def own_cubic_lengths():
        segments = [
            mojo.CubicBezier(*(row[:, 0] + 1j * row[:, 1]))
            for row in cubic_controls
        ]
        return mojo.bezier_lengths(segments)

    def up_cubic_lengths():
        return np.array(
            [
                upstream.CubicBezier(
                    *(row[:, 0] + 1j * row[:, 1])
                ).length()
                for row in cubic_controls
            ]
        )

    cases.append(
        (
            "CubicBezier lengths",
            "2,000 curves",
            own_cubic_lengths,
            up_cubic_lengths,
        )
    )

    arc_specs = [
        (
            complex(*rng.normal(size=2)),
            complex(*rng.uniform(1, 20, size=2)),
            float(rng.uniform(-90, 90)),
            bool(i % 2),
            bool((i // 2) % 2),
            complex(*rng.normal(size=2) + np.array([20, 20])),
        )
        for i in range(2_000)
    ]

    def own_arc_lengths():
        return mojo.arc_lengths([mojo.Arc(*spec) for spec in arc_specs])

    def up_arc_lengths():
        return np.array([upstream.Arc(*spec).length() for spec in arc_specs])

    cases.append(
        ("Arc lengths", "2,000 curves", own_arc_lengths, up_arc_lengths)
    )

    path_controls = rng.normal(size=(5_000, 4, 2)).cumsum(axis=0)

    def own_path_length():
        return mojo.Path(
            *[
                mojo.CubicBezier(*(row[:, 0] + 1j * row[:, 1]))
                for row in path_controls
            ]
        ).length()

    def up_path_length():
        return upstream.Path(
            *[
                upstream.CubicBezier(*(row[:, 0] + 1j * row[:, 1]))
                for row in path_controls
            ]
        ).length()

    cases.append(
        ("Path.length", "5,000 cubic segments", own_path_length, up_path_length)
    )

    own_cubic.points(np.array([0.5]))
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Operation | Size | Mojo port | svgpathtools | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, size, own_fn, upstream_fn in cases:
        own_result = own_fn()
        upstream_result = upstream_fn()
        if isinstance(own_result, np.ndarray):
            np.testing.assert_allclose(own_result, upstream_result, rtol=1e-8, atol=5e-10)
        else:
            np.testing.assert_allclose(own_result, upstream_result, rtol=1e-8, atol=5e-10)
        own_time = timeit(own_fn)
        upstream_time = timeit(upstream_fn)
        print(
            f"| {name} | {size} | {format_time(own_time)} | "
            f"{format_time(upstream_time)} | {upstream_time / own_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
