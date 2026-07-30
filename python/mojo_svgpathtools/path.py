from __future__ import annotations

import math
import re
from collections.abc import MutableSequence

import numpy as np

from ._lib import addr, f64, lib

_COMMAND = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]")
_TOKEN = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _complex_array(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] + 1j * xy[:, 1]


def _bezier_coefficients(points: tuple[complex, ...]) -> np.ndarray:
    if len(points) == 2:
        return np.array([points[0], points[1] - points[0]], dtype=complex)
    if len(points) == 3:
        p0, p1, p2 = points
        return np.array([p0, 2 * (p1 - p0), p0 - 2 * p1 + p2], dtype=complex)
    p0, p1, p2, p3 = points
    return np.array(
        [
            p0,
            3 * (p1 - p0),
            3 * (p0 - 2 * p1 + p2),
            -p0 + 3 * p1 - 3 * p2 + p3,
        ],
        dtype=complex,
    )


def _split_points(points: tuple[complex, ...], t: float):
    levels = [list(points)]
    while len(levels[-1]) > 1:
        row = levels[-1]
        levels.append([(1 - t) * row[i] + t * row[i + 1] for i in range(len(row) - 1)])
    left = tuple(row[0] for row in levels)
    right = tuple(row[-1] for row in reversed(levels))
    return left, right


def _unit_tangent(segment, t):
    value = segment.derivative(t)
    if value:
        return value / abs(value)
    for order in range(2, 5):
        value = segment.derivative(t, order)
        if value:
            return value / abs(value)
    raise ValueError("tangent is undefined for a constant segment")


def _curvature(segment, t):
    first = segment.derivative(t)
    second = segment.derivative(t, 2)
    speed2 = first.real * first.real + first.imag * first.imag
    if speed2 == 0:
        return float("inf")
    return abs(first.real * second.imag - first.imag * second.real) / speed2**1.5


def _inverse_length(segment, s, s_tol, maxits, error, min_depth):
    total = segment.length(error=error, min_depth=min_depth)
    if s < 0 or s > total:
        raise ValueError("s must satisfy 0 <= s <= segment.length()")
    if s == 0:
        return 0.0
    if s == total:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(maxits):
        mid = (lo + hi) * 0.5
        value = segment.length(0, mid, error=error, min_depth=min_depth)
        if abs(value - s) <= s_tol:
            return mid
        if value < s:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _bezier_bbox(segment):
    coeffs = _bezier_coefficients(segment.bpoints())
    candidates = [0.0, 1.0]
    for values in (coeffs.real, coeffs.imag):
        derivative = np.arange(1, len(values)) * values[1:]
        if np.any(derivative):
            for root in np.roots(derivative[::-1]):
                if abs(root.imag) < 1e-12 and 0 < root.real < 1:
                    candidates.append(float(root.real))
    points = [segment.point(t) for t in candidates]
    return (
        min(z.real for z in points),
        max(z.real for z in points),
        min(z.imag for z in points),
        max(z.imag for z in points),
    )


def _bezier_line_intersections(segment, line):
    direction = line.end - line.start
    coeffs = _bezier_coefficients(segment.bpoints())
    cross = np.array(
        [(z * direction.conjugate()).imag for z in coeffs], dtype=float
    )
    cross[0] -= (line.start * direction.conjugate()).imag
    roots = np.roots(np.trim_zeros(cross[::-1], "f"))
    found = []
    denom = abs(direction) ** 2
    for root in roots:
        if abs(root.imag) > 1e-10 or not -1e-12 <= root.real <= 1 + 1e-12:
            continue
        t = min(1.0, max(0.0, float(root.real)))
        u = ((segment.point(t) - line.start) * direction.conjugate()).real / denom
        if -1e-12 <= u <= 1 + 1e-12:
            pair = (t, min(1.0, max(0.0, u)))
            if not any(abs(pair[0] - old[0]) < 1e-9 for old in found):
                found.append(pair)
    return found


class _Bezier:
    degree: int

    def __len__(self):
        return self.degree + 1

    def __getitem__(self, item):
        return self.bpoints()[item]

    def __hash__(self):
        return hash(self.bpoints())

    def points(self, ts):
        values = f64(ts).reshape(-1)
        if not values.size:
            return np.empty(0, dtype=np.complex128)
        controls = np.array(
            [[point.real, point.imag] for point in self.bpoints()], dtype=np.float64
        )
        result = np.empty(values.size, dtype=np.complex128)
        lib().msp_bezier_points(
            addr(controls), self.degree, addr(values), values.size, addr(result)
        )
        return result

    def derivatives(self, ts, n=1):
        if n < 1:
            raise ValueError("n should be a positive integer.")
        values = f64(ts).reshape(-1)
        if not values.size:
            return np.empty(0, dtype=np.complex128)
        controls = np.array(
            [[point.real, point.imag] for point in self.bpoints()], dtype=np.float64
        )
        result = np.empty(values.size, dtype=np.complex128)
        lib().msp_bezier_derivatives(
            addr(controls), self.degree, addr(values), values.size, n, addr(result)
        )
        return result

    def length(self, t0=0, t1=1, error=1e-12, min_depth=5):
        controls = np.array(
            [[point.real, point.imag] for point in self.bpoints()], dtype=np.float64
        )
        result = np.empty(1, dtype=np.float64)
        subdivisions = 1 << min(max(int(min_depth or 0) - 3, 0), 4)
        lib().msp_bezier_lengths(
            addr(controls), self.degree, 1, t0, t1, subdivisions, addr(result)
        )
        return float(result[0])

    def ilength(
        self, s, s_tol=1e-12, maxits=10000, error=1e-12, min_depth=5
    ):
        return _inverse_length(self, s, s_tol, maxits, error, min_depth)

    def unit_tangent(self, t):
        return _unit_tangent(self, t)

    def normal(self, t):
        return -1j * self.unit_tangent(t)

    def curvature(self, t):
        return _curvature(self, t)

    def bbox(self):
        return _bezier_bbox(self)

    def poly(self, return_coeffs=False):
        coeffs = tuple(_bezier_coefficients(self.bpoints())[::-1])
        return coeffs if return_coeffs else np.poly1d(coeffs)

    def cropped(self, t0, t1):
        if not 0 <= t0 <= 1 or not 0 <= t1 <= 1 or t0 == t1:
            raise ValueError("crop parameters must be distinct values in [0, 1]")
        if t0 > t1:
            return self.cropped(t1, t0).reversed()
        left, _ = self.split(t1)
        if t0 == 0:
            return left
        _, middle = left.split(t0 / t1)
        return middle

    def intersect(self, other_seg, tol=1e-12):
        if isinstance(other_seg, Line):
            return _bezier_line_intersections(self, other_seg)
        raise NotImplementedError("covered intersections are Line-Line and Bezier-Line")

    def joins_smoothly_with(self, previous, wrt_parameterization=False, error=0):
        if self.start != previous.end:
            return False
        if wrt_parameterization:
            return abs(self.derivative(0) - previous.derivative(1)) <= error
        return abs(self.unit_tangent(0) - previous.unit_tangent(1)) <= error


class Line(_Bezier):
    degree = 1

    def __init__(self, start, end):
        self.start = complex(start)
        self.end = complex(end)

    def __repr__(self):
        return f"Line(start={self.start}, end={self.end})"

    def __eq__(self, other):
        return isinstance(other, Line) and self.bpoints() == other.bpoints()

    def bpoints(self):
        return self.start, self.end

    def point(self, t):
        return self.start + (self.end - self.start) * t

    def derivative(self, t=None, n=1):
        if n < 1:
            raise ValueError("n should be a positive integer.")
        if self.start == self.end:
            raise AssertionError("line endpoints must differ")
        return self.end - self.start if n == 1 else 0

    def length(self, t0=0, t1=1, error=None, min_depth=None):
        return abs(self.end - self.start) * (t1 - t0)

    def unit_tangent(self, t=None):
        value = self.derivative()
        return value / abs(value)

    def curvature(self, t):
        return 0

    def bbox(self):
        return (
            min(self.start.real, self.end.real),
            max(self.start.real, self.end.real),
            min(self.start.imag, self.end.imag),
            max(self.start.imag, self.end.imag),
        )

    def split(self, t):
        point = self.point(t)
        return Line(self.start, point), Line(point, self.end)

    def cropped(self, t0, t1):
        return Line(self.point(t0), self.point(t1))

    def reversed(self):
        return Line(self.end, self.start)

    def intersect(self, other_seg, tol=None):
        if isinstance(other_seg, (QuadraticBezier, CubicBezier)):
            return [(b, a) for a, b in other_seg.intersect(self, tol or 1e-12)]
        if not isinstance(other_seg, Line):
            raise TypeError("other_seg must be a covered path segment")
        p, r = self.start, self.end - self.start
        q, s = other_seg.start, other_seg.end - other_seg.start
        denominator = (r * s.conjugate()).imag
        if math.isclose(denominator, 0):
            return []
        t = ((q - p) * s.conjugate()).imag / denominator
        u = ((q - p) * r.conjugate()).imag / denominator
        return [(t, u)] if 0 <= t <= 1 and 0 <= u <= 1 else []


class QuadraticBezier(_Bezier):
    degree = 2

    def __init__(self, start, control, end):
        self.start = complex(start)
        self.control = complex(control)
        self.end = complex(end)

    def __repr__(self):
        return (
            f"QuadraticBezier(start={self.start}, control={self.control}, "
            f"end={self.end})"
        )

    def __eq__(self, other):
        return isinstance(other, QuadraticBezier) and self.bpoints() == other.bpoints()

    def bpoints(self):
        return self.start, self.control, self.end

    def point(self, t):
        u = 1 - t
        return u * u * self.start + 2 * u * t * self.control + t * t * self.end

    def derivative(self, t, n=1):
        if n == 1:
            return 2 * (
                (self.control - self.start) * (1 - t)
                + (self.end - self.control) * t
            )
        if n == 2:
            return 2 * (self.end - 2 * self.control + self.start)
        if n > 2:
            return 0
        raise ValueError("n should be a positive integer.")

    def split(self, t):
        left, right = _split_points(self.bpoints(), t)
        return QuadraticBezier(*left), QuadraticBezier(*right)

    def reversed(self):
        return QuadraticBezier(self.end, self.control, self.start)


class CubicBezier(_Bezier):
    degree = 3

    def __init__(self, start, control1, control2, end):
        self.start = complex(start)
        self.control1 = complex(control1)
        self.control2 = complex(control2)
        self.end = complex(end)

    def __repr__(self):
        return (
            f"CubicBezier(start={self.start}, control1={self.control1}, "
            f"control2={self.control2}, end={self.end})"
        )

    def __eq__(self, other):
        return isinstance(other, CubicBezier) and self.bpoints() == other.bpoints()

    def bpoints(self):
        return self.start, self.control1, self.control2, self.end

    def point(self, t):
        return self.start + t * (
            3 * (self.control1 - self.start)
            + t
            * (
                3 * (self.start + self.control2)
                - 6 * self.control1
                + t
                * (
                    -self.start
                    + 3 * (self.control1 - self.control2)
                    + self.end
                )
            )
        )

    def derivative(self, t, n=1):
        p0, p1, p2, p3 = self.bpoints()
        if n == 1:
            return (
                3 * (p1 - p0) * (1 - t) ** 2
                + 6 * (p2 - p1) * (1 - t) * t
                + 3 * (p3 - p2) * t**2
            )
        if n == 2:
            return 6 * ((1 - t) * (p2 - 2 * p1 + p0) + t * (p3 - 2 * p2 + p1))
        if n == 3:
            return 6 * (p3 - 3 * p2 + 3 * p1 - p0)
        if n > 3:
            return 0
        raise ValueError("n should be a positive integer.")

    def split(self, t):
        left, right = _split_points(self.bpoints(), t)
        return CubicBezier(*left), CubicBezier(*right)

    def reversed(self):
        return CubicBezier(self.end, self.control2, self.control1, self.start)


class Arc:
    def __init__(
        self,
        start,
        radius,
        rotation,
        large_arc,
        sweep,
        end,
        autoscale_radius=True,
    ):
        self.start = complex(start)
        self.end = complex(end)
        if self.start == self.end:
            raise AssertionError("arc endpoints must differ")
        radius = complex(radius)
        if radius.real == 0 or radius.imag == 0:
            raise AssertionError("arc radii must be nonzero")
        self.radius = abs(radius.real) + 1j * abs(radius.imag)
        self.rotation = rotation
        self.large_arc = bool(large_arc)
        self.sweep = bool(sweep)
        self.autoscale_radius = autoscale_radius
        self.phi = math.radians(rotation)
        self.rot_matrix = complex(math.cos(self.phi), math.sin(self.phi))
        self._parameterize()

    def _parameterize(self):
        rx, ry = self.radius.real, self.radius.imag
        zp = (self.start - self.end) / (2 * self.rot_matrix)
        xp, yp = zp.real, zp.imag
        check = xp * xp / (rx * rx) + yp * yp / (ry * ry)
        if check > 1:
            if not self.autoscale_radius:
                raise ValueError("No such elliptic arc exists.")
            scale = math.sqrt(check)
            rx *= scale
            ry *= scale
            self.radius = rx + 1j * ry
        numerator = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp
        denominator = rx * rx * yp * yp + ry * ry * xp * xp
        radicand = numerator / denominator
        radical = 0.0 if math.isclose(radicand, 0.0, abs_tol=1e-12) else math.sqrt(max(0.0, radicand))
        if self.large_arc == self.sweep:
            radical = -radical
        cp = radical * complex(rx * yp / ry, -ry * xp / rx)
        self.center = self.rot_matrix * cp + (self.start + self.end) / 2
        u1 = complex((xp - cp.real) / rx, (yp - cp.imag) / ry)
        u2 = complex((-xp - cp.real) / rx, (-yp - cp.imag) / ry)
        self.theta = math.degrees(math.atan2(u1.imag, u1.real))
        delta = math.degrees(math.atan2((u1.conjugate() * u2).imag, (u1.conjugate() * u2).real))
        if not self.sweep and delta >= 0:
            delta -= 360
        elif self.sweep and delta <= 0:
            delta += 360
        self.delta = delta

    def __repr__(self):
        return (
            f"Arc(start={self.start}, radius={self.radius}, rotation={self.rotation}, "
            f"large_arc={self.large_arc}, sweep={self.sweep}, end={self.end})"
        )

    def __eq__(self, other):
        return isinstance(other, Arc) and self.apoints() == other.apoints()

    def __hash__(self):
        return hash(self.apoints())

    def apoints(self):
        return (
            self.start,
            self.radius,
            self.rotation,
            self.large_arc,
            self.sweep,
            self.end,
        )

    def point(self, t):
        angle = math.radians(self.theta + t * self.delta)
        cp, sp = self.rot_matrix.real, self.rot_matrix.imag
        rx, ry = self.radius.real, self.radius.imag
        return complex(
            rx * cp * math.cos(angle) - ry * sp * math.sin(angle) + self.center.real,
            rx * sp * math.cos(angle) + ry * cp * math.sin(angle) + self.center.imag,
        )

    def points(self, ts):
        values = f64(ts).reshape(-1)
        if not values.size:
            return np.empty(0, dtype=np.complex128)
        result = np.empty((values.size, 2), dtype=np.float64)
        lib().msp_arc_points(
            self.center.real,
            self.center.imag,
            self.radius.real,
            self.radius.imag,
            self.phi,
            math.radians(self.theta),
            math.radians(self.delta),
            addr(values),
            values.size,
            addr(result),
        )
        return _complex_array(result)

    def derivative(self, t, n=1):
        if n < 1:
            raise ValueError("n should be a positive integer.")
        angle = math.radians(self.theta + t * self.delta)
        rx, ry = self.radius.real, self.radius.imag
        cp, sp = self.rot_matrix.real, self.rot_matrix.imag
        scale = math.radians(self.delta) ** n
        mode = n % 4
        if mode == 1:
            z = complex(-rx * cp * math.sin(angle) - ry * sp * math.cos(angle), -rx * sp * math.sin(angle) + ry * cp * math.cos(angle))
        elif mode == 2:
            z = complex(-rx * cp * math.cos(angle) + ry * sp * math.sin(angle), -rx * sp * math.cos(angle) - ry * cp * math.sin(angle))
        elif mode == 3:
            z = complex(rx * cp * math.sin(angle) + ry * sp * math.cos(angle), rx * sp * math.sin(angle) - ry * cp * math.cos(angle))
        else:
            z = complex(rx * cp * math.cos(angle) - ry * sp * math.sin(angle), rx * sp * math.cos(angle) + ry * cp * math.sin(angle))
        return scale * z

    def derivatives(self, ts, n=1):
        if n < 1:
            raise ValueError("n should be a positive integer.")
        values = f64(ts).reshape(-1)
        if not values.size:
            return np.empty(0, dtype=np.complex128)
        result = np.empty((values.size, 2), dtype=np.float64)
        lib().msp_arc_derivatives(
            self.radius.real,
            self.radius.imag,
            self.phi,
            math.radians(self.theta),
            math.radians(self.delta),
            addr(values),
            values.size,
            n,
            addr(result),
        )
        return _complex_array(result)

    def length(self, t0=0, t1=1, error=1e-12, min_depth=5):
        if not 0 <= t0 <= 1 or not 0 <= t1 <= 1:
            raise AssertionError("arc parameters must be in [0, 1]")
        params = np.array(
            [[self.radius.real, self.radius.imag, math.radians(self.theta), math.radians(self.delta)]],
            dtype=np.float64,
        )
        result = np.empty(1, dtype=np.float64)
        subdivisions = 1 << min(max(int(min_depth or 0) - 3, 0), 4)
        lib().msp_arc_lengths(addr(params), 1, t0, t1, subdivisions, addr(result))
        return float(result[0])

    def ilength(self, s, s_tol=1e-12, maxits=10000, error=1e-12, min_depth=5):
        return _inverse_length(self, s, s_tol, maxits, error, min_depth)

    def unit_tangent(self, t):
        value = self.derivative(t)
        return value / abs(value)

    def normal(self, t):
        return -1j * self.unit_tangent(t)

    def curvature(self, t):
        return _curvature(self, t)

    def bbox(self):
        candidates = [0.0, 1.0]
        rx, ry = self.radius.real, self.radius.imag
        cp, sp = self.rot_matrix.real, self.rot_matrix.imag
        bases = (
            math.atan2(-ry * sp, rx * cp),
            math.atan2(ry * cp, rx * sp),
        )
        theta = math.radians(self.theta)
        delta = math.radians(self.delta)
        for base in bases:
            for half_turn in range(-4, 5):
                t = (base + half_turn * math.pi - theta) / delta
                if 0 < t < 1:
                    candidates.append(t)
        points = [self.point(t) for t in candidates]
        return (
            min(z.real for z in points),
            max(z.real for z in points),
            min(z.imag for z in points),
            max(z.imag for z in points),
        )

    def cropped(self, t0, t1):
        if not 0 <= t0 <= 1 or not 0 <= t1 <= 1 or t0 == t1:
            raise ValueError("crop parameters must be distinct values in [0, 1]")
        span = self.delta * (t1 - t0)
        return Arc(
            self.point(t0),
            self.radius,
            self.rotation,
            abs(span) > 180,
            span > 0,
            self.point(t1),
        )

    def split(self, t):
        return self.cropped(0, t), self.cropped(t, 1)

    def reversed(self):
        return Arc(
            self.end,
            self.radius,
            self.rotation,
            self.large_arc,
            not self.sweep,
            self.start,
        )

    def intersect(self, other_seg, tol=1e-12):
        raise NotImplementedError("elliptical-arc intersections are not covered")

    def joins_smoothly_with(self, previous, wrt_parameterization=False, error=0):
        if self.start != previous.end:
            return False
        if wrt_parameterization:
            return abs(self.derivative(0) - previous.derivative(1)) <= error
        return abs(self.unit_tangent(0) - previous.unit_tangent(1)) <= error


def bezier_lengths(segments, t0=0, t1=1, min_depth=5):
    segments = list(segments)
    if not segments:
        return np.empty(0)
    degree = segments[0].degree
    if degree not in (1, 2, 3) or any(seg.degree != degree for seg in segments):
        raise TypeError("all segments must be Beziers of the same degree")
    controls = np.array(
        [
            [[point.real, point.imag] for point in segment.bpoints()]
            for segment in segments
        ],
        dtype=np.float64,
    )
    result = np.empty(len(segments), dtype=np.float64)
    subdivisions = 1 << min(max(int(min_depth) - 3, 0), 4)
    lib().msp_bezier_lengths(
        addr(controls), degree, len(segments), t0, t1, subdivisions, addr(result)
    )
    return result


def arc_lengths(segments, t0=0, t1=1, min_depth=5):
    segments = list(segments)
    if any(not isinstance(seg, Arc) for seg in segments):
        raise TypeError("all segments must be Arc instances")
    params = np.array(
        [
            [
                seg.radius.real,
                seg.radius.imag,
                math.radians(seg.theta),
                math.radians(seg.delta),
            ]
            for seg in segments
        ],
        dtype=np.float64,
    )
    result = np.empty(len(segments), dtype=np.float64)
    if segments:
        subdivisions = 1 << min(max(int(min_depth) - 3, 0), 4)
        lib().msp_arc_lengths(
            addr(params), len(segments), t0, t1, subdivisions, addr(result)
        )
    return result


class Path(MutableSequence):
    def __init__(self, *segments, **kw):
        if segments and isinstance(segments[0], str):
            parsed = parse_path(
                segments[0], segments[1] if len(segments) > 1 else kw.get("current_pos", 0j)
            )
            self._segments = list(parsed)
        else:
            self._segments = list(segments)
        self._length = None
        self._lengths = None

    def _invalidate(self):
        self._length = self._lengths = None

    def __getitem__(self, index):
        return self._segments[index]

    def __setitem__(self, index, value):
        self._segments[index] = value
        self._invalidate()

    def __delitem__(self, index):
        del self._segments[index]
        self._invalidate()

    def __len__(self):
        return len(self._segments)

    def insert(self, index, value):
        self._segments.insert(index, value)
        self._invalidate()

    def __repr__(self):
        return "Path({})".format(",\n     ".join(repr(x) for x in self))

    def __eq__(self, other):
        return isinstance(other, Path) and self._segments == other._segments

    @property
    def start(self):
        return self[0].start if self else None

    @property
    def end(self):
        return self[-1].end if self else None

    def _calc_lengths(self, error=1e-12, min_depth=5):
        if self._length is not None:
            return
        lengths = np.empty(len(self), dtype=float)
        for degree in (1, 2, 3):
            indices = [i for i, seg in enumerate(self) if getattr(seg, "degree", 0) == degree]
            if indices:
                lengths[indices] = bezier_lengths(
                    [self[i] for i in indices], min_depth=min_depth
                )
        indices = [i for i, seg in enumerate(self) if isinstance(seg, Arc)]
        if indices:
            lengths[indices] = arc_lengths([self[i] for i in indices], min_depth=min_depth)
        self._length = float(lengths.sum())
        self._lengths = lengths / self._length if self._length else lengths

    def point(self, pos):
        index, t = self.T2t(pos)
        return self[index].point(t)

    def T2t(self, T):
        if not self:
            raise ValueError("This path contains no segments!")
        if T == 0:
            return 0, 0.0
        if T == 1:
            return len(self) - 1, 1.0
        self._calc_lengths()
        start = 0.0
        for index, fraction in enumerate(self._lengths):
            end = start + fraction
            if end >= T:
                return index, (T - start) / (end - start)
            start = end
        raise ValueError("T must lie in [0, 1]")

    def t2T(self, seg, t):
        self._calc_lengths()
        index = seg if isinstance(seg, int) else self.index(seg)
        return float(np.sum(self._lengths[:index]) + t * self._lengths[index])

    def length(self, T0=0, T1=1, error=1e-12, min_depth=5):
        self._calc_lengths(error, min_depth)
        if T0 == 0 and T1 == 1:
            return self._length
        i0, t0 = self.T2t(T0)
        i1, t1 = self.T2t(T1)
        if i0 == i1:
            return self[i0].length(t0, t1, error, min_depth)
        return (
            self[i0].length(t0, 1, error, min_depth)
            + sum(seg.length(error=error, min_depth=min_depth) for seg in self[i0 + 1 : i1])
            + self[i1].length(0, t1, error, min_depth)
        )

    def ilength(self, s, s_tol=1e-12, maxits=10000, error=1e-12, min_depth=5):
        return _inverse_length(self, s, s_tol, maxits, error, min_depth)

    def derivative(self, T, n=1):
        index, t = self.T2t(T)
        return self[index].derivative(t, n) / self[index].length() ** n

    def unit_tangent(self, T):
        index, t = self.T2t(T)
        return self[index].unit_tangent(t)

    def normal(self, t):
        return -1j * self.unit_tangent(t)

    def curvature(self, T):
        index, t = self.T2t(T)
        return self[index].curvature(t)

    def bbox(self):
        boxes = [segment.bbox() for segment in self]
        if not boxes:
            raise ValueError("empty path has no bounding box")
        return (
            min(box[0] for box in boxes),
            max(box[1] for box in boxes),
            min(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def reversed(self):
        return Path(*(segment.reversed() for segment in reversed(self)))

    def iscontinuous(self):
        return all(self[i].end == self[i + 1].start for i in range(len(self) - 1))

    def isclosed(self):
        if not self or not self.iscontinuous():
            raise AssertionError("path must be nonempty and continuous")
        return self.start == self.end

    def continuous_subpaths(self):
        if not self:
            return []
        paths, start = [], 0
        for i in range(len(self) - 1):
            if self[i].end != self[i + 1].start:
                paths.append(Path(*self[start : i + 1]))
                start = i + 1
        paths.append(Path(*self[start:]))
        return paths

    def cropped(self, T0, T1):
        if not 0 <= T0 <= 1 or not 0 <= T1 <= 1 or T0 >= T1:
            raise ValueError("covered Path.cropped requires 0 <= T0 < T1 <= 1")
        i0, t0 = self.T2t(T0)
        i1, t1 = self.T2t(T1)
        if i0 == i1:
            return Path(self[i0].cropped(t0, t1))
        segments = [self[i0].cropped(t0, 1)]
        segments.extend(self[i0 + 1 : i1])
        segments.append(self[i1].cropped(0, t1))
        return Path(*segments)


def parse_path(pathdef, current_pos=0j, tree_element=None):
    tokens = _TOKEN.findall(pathdef.replace(",", " "))
    path = []
    i = 0
    command = None
    current = complex(current_pos)
    subpath_start = current
    previous_control = None
    previous_command = None

    def number():
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    def point(relative):
        value = complex(number(), number())
        return current + value if relative else value

    while i < len(tokens):
        if _COMMAND.fullmatch(tokens[i]):
            command = tokens[i]
            i += 1
        elif command is None:
            raise ValueError("path data must begin with a command")
        upper = command.upper()
        relative = command.islower()
        if upper == "Z":
            if current != subpath_start:
                path.append(Line(current, subpath_start))
            current = subpath_start
            previous_control = None
            previous_command = upper
            command = None
            continue
        if upper == "M":
            target = point(relative)
            current = target
            subpath_start = target
            command = "l" if relative else "L"
            previous_control = None
            previous_command = "M"
            continue
        if upper == "L":
            target = point(relative)
            path.append(Line(current, target))
            current = target
            previous_control = None
        elif upper == "H":
            x = number() + (current.real if relative else 0)
            target = complex(x, current.imag)
            path.append(Line(current, target))
            current = target
            previous_control = None
        elif upper == "V":
            y = number() + (current.imag if relative else 0)
            target = complex(current.real, y)
            path.append(Line(current, target))
            current = target
            previous_control = None
        elif upper == "C":
            control1 = point(relative)
            control2 = point(relative)
            target = point(relative)
            path.append(CubicBezier(current, control1, control2, target))
            current, previous_control = target, control2
        elif upper == "S":
            control1 = (
                2 * current - previous_control
                if previous_command in ("C", "S") and previous_control is not None
                else current
            )
            control2 = point(relative)
            target = point(relative)
            path.append(CubicBezier(current, control1, control2, target))
            current, previous_control = target, control2
        elif upper == "Q":
            control = point(relative)
            target = point(relative)
            path.append(QuadraticBezier(current, control, target))
            current, previous_control = target, control
        elif upper == "T":
            control = (
                2 * current - previous_control
                if previous_command in ("Q", "T") and previous_control is not None
                else current
            )
            target = point(relative)
            path.append(QuadraticBezier(current, control, target))
            current, previous_control = target, control
        elif upper == "A":
            radius = complex(number(), number())
            rotation = number()
            large_arc = bool(number())
            sweep = bool(number())
            target = point(relative)
            path.append(Arc(current, radius, rotation, large_arc, sweep, target))
            current = target
            previous_control = None
        else:
            raise ValueError(f"unsupported SVG path command {command}")
        previous_command = upper
    result = Path(*path)
    if tree_element is not None:
        result.element = tree_element
    return result
