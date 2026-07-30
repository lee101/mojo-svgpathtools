from std.algorithm import parallelize
from std.math import cos, sin, sqrt
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_GRAIN = 131072
comptime MAX_PARALLEL_TASKS = 16


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def bezier_value(ctrl: Ptr, degree: Int, t: Float64, axis: Int) -> Float64:
    if degree == 1:
        return ctrl[axis] + t * (ctrl[2 + axis] - ctrl[axis])
    if degree == 2:
        var a = ctrl[axis]
        var b = ctrl[2 + axis]
        var c = ctrl[4 + axis]
        var u = 1.0 - t
        return u * u * a + 2.0 * u * t * b + t * t * c
    var a = ctrl[axis]
    var b = ctrl[2 + axis]
    var c = ctrl[4 + axis]
    var d = ctrl[6 + axis]
    return a + t * (
        3.0 * (b - a) + t * (
            3.0 * (a + c) - 6.0 * b + t * (-a + 3.0 * (b - c) + d)
        )
    )


def bezier_derivative(
    ctrl: Ptr, degree: Int, t: Float64, order: Int, axis: Int
) -> Float64:
    if degree == 1:
        if order == 1:
            return ctrl[2 + axis] - ctrl[axis]
        return 0.0
    if degree == 2:
        var a = ctrl[axis]
        var b = ctrl[2 + axis]
        var c = ctrl[4 + axis]
        if order == 1:
            return 2.0 * ((b - a) * (1.0 - t) + (c - b) * t)
        if order == 2:
            return 2.0 * (c - 2.0 * b + a)
        return 0.0
    var a = ctrl[axis]
    var b = ctrl[2 + axis]
    var c = ctrl[4 + axis]
    var d = ctrl[6 + axis]
    if order == 1:
        var u = 1.0 - t
        return 3.0 * (b - a) * u * u + 6.0 * (c - b) * u * t + 3.0 * (d - c) * t * t
    if order == 2:
        return 6.0 * ((1.0 - t) * (c - 2.0 * b + a) + t * (d - 2.0 * c + b))
    if order == 3:
        return 6.0 * (d - 3.0 * c + 3.0 * b - a)
    return 0.0


def bezier_speed(ctrl: Ptr, degree: Int, t: Float64) -> Float64:
    var dx = bezier_derivative(ctrl, degree, t, 1, 0)
    var dy = bezier_derivative(ctrl, degree, t, 1, 1)
    return sqrt(dx * dx + dy * dy)


def bezier_points_range(
    ctrl: Ptr, degree: Int, ts: Ptr, dst: Ptr, start: Int, end: Int
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    if degree == 1:
        var ax = ctrl[0]
        var ay = ctrl[1]
        var bx = ctrl[2] - ax
        var by = ctrl[3] - ay
        while i + W <= end:
            var tv = ts.load[width=W](i)
            var x = ax + tv * bx
            var y = ay + tv * by
            dst.store(2 * i, x.interleave(y))
            i += W
    elif degree == 2:
        var ax = ctrl[0]
        var ay = ctrl[1]
        var bx = ctrl[2]
        var by = ctrl[3]
        var cx = ctrl[4]
        var cy = ctrl[5]
        while i + W <= end:
            var tv = ts.load[width=W](i)
            var uv = 1.0 - tv
            var x = uv * uv * ax + 2.0 * uv * tv * bx + tv * tv * cx
            var y = uv * uv * ay + 2.0 * uv * tv * by + tv * tv * cy
            dst.store(2 * i, x.interleave(y))
            i += W
    else:
        var ax = ctrl[0]
        var ay = ctrl[1]
        var bx = ctrl[2]
        var by = ctrl[3]
        var cx = ctrl[4]
        var cy = ctrl[5]
        var dx = ctrl[6]
        var dy = ctrl[7]
        while i + W <= end:
            var tv = ts.load[width=W](i)
            var x = ax + tv * (
                3.0 * (bx - ax)
                + tv
                * (
                    3.0 * (ax + cx)
                    - 6.0 * bx
                    + tv * (-ax + 3.0 * (bx - cx) + dx)
                )
            )
            var y = ay + tv * (
                3.0 * (by - ay)
                + tv
                * (
                    3.0 * (ay + cy)
                    - 6.0 * by
                    + tv * (-ay + 3.0 * (by - cy) + dy)
                )
            )
            dst.store(2 * i, x.interleave(y))
            i += W
    for j in range(i, end):
        dst[2 * j] = bezier_value(ctrl, degree, ts[j], 0)
        dst[2 * j + 1] = bezier_value(ctrl, degree, ts[j], 1)


def bezier_derivatives_range(
    ctrl: Ptr,
    degree: Int,
    ts: Ptr,
    dst: Ptr,
    start: Int,
    end: Int,
    order: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    if order == 1:
        if degree == 1:
            var dx = ctrl[2] - ctrl[0]
            var dy = ctrl[3] - ctrl[1]
            while i + W <= end:
                var tv = ts.load[width=W](i)
                var x = tv * 0.0 + dx
                var y = tv * 0.0 + dy
                dst.store(2 * i, x.interleave(y))
                i += W
        elif degree == 2:
            var ax = ctrl[0]
            var ay = ctrl[1]
            var bx = ctrl[2]
            var by = ctrl[3]
            var cx = ctrl[4]
            var cy = ctrl[5]
            while i + W <= end:
                var tv = ts.load[width=W](i)
                var uv = 1.0 - tv
                var x = 2.0 * ((bx - ax) * uv + (cx - bx) * tv)
                var y = 2.0 * ((by - ay) * uv + (cy - by) * tv)
                dst.store(2 * i, x.interleave(y))
                i += W
        else:
            var ax = ctrl[0]
            var ay = ctrl[1]
            var bx = ctrl[2]
            var by = ctrl[3]
            var cx = ctrl[4]
            var cy = ctrl[5]
            var dx = ctrl[6]
            var dy = ctrl[7]
            while i + W <= end:
                var tv = ts.load[width=W](i)
                var uv = 1.0 - tv
                var x = (
                    3.0 * (bx - ax) * uv * uv
                    + 6.0 * (cx - bx) * uv * tv
                    + 3.0 * (dx - cx) * tv * tv
                )
                var y = (
                    3.0 * (by - ay) * uv * uv
                    + 6.0 * (cy - by) * uv * tv
                    + 3.0 * (dy - cy) * tv * tv
                )
                dst.store(2 * i, x.interleave(y))
                i += W
    for j in range(i, end):
        dst[2 * j] = bezier_derivative(ctrl, degree, ts[j], order, 0)
        dst[2 * j + 1] = bezier_derivative(ctrl, degree, ts[j], order, 1)


def gl_x(i: Int) -> Float64:
    if i == 0:
        return 0.09501250983763744
    if i == 1:
        return 0.2816035507792589
    if i == 2:
        return 0.4580167776572274
    if i == 3:
        return 0.6178762444026438
    if i == 4:
        return 0.755404408355003
    if i == 5:
        return 0.8656312023878318
    if i == 6:
        return 0.9445750230732326
    return 0.9894009349916499


def gl_w(i: Int) -> Float64:
    if i == 0:
        return 0.1894506104550685
    if i == 1:
        return 0.1826034150449236
    if i == 2:
        return 0.16915651939500254
    if i == 3:
        return 0.14959598881657673
    if i == 4:
        return 0.12462897125553387
    if i == 5:
        return 0.09515851168249278
    if i == 6:
        return 0.06225352393864789
    return 0.027152459411754096


def gl_bezier_interval(
    ctrl: Ptr, degree: Int, left: Float64, right: Float64
) -> Float64:
    var mid = 0.5 * (left + right)
    var half = 0.5 * (right - left)
    var acc = 0.0
    for i in range(8):
        var offset = half * gl_x(i)
        acc += gl_w(i) * (
            bezier_speed(ctrl, degree, mid - offset)
            + bezier_speed(ctrl, degree, mid + offset)
        )
    return half * acc


def adaptive_bezier(
    ctrl: Ptr,
    degree: Int,
    left: Float64,
    right: Float64,
    tolerance: Float64,
    depth: Int,
) -> Float64:
    var mid = 0.5 * (left + right)
    var coarse = gl_bezier_interval(ctrl, degree, left, right)
    var fine = gl_bezier_interval(ctrl, degree, left, mid) + gl_bezier_interval(
        ctrl, degree, mid, right
    )
    if depth <= 0 or abs(fine - coarse) <= tolerance:
        return fine
    return adaptive_bezier(
        ctrl, degree, left, mid, 0.5 * tolerance, depth - 1
    ) + adaptive_bezier(ctrl, degree, mid, right, 0.5 * tolerance, depth - 1)


def integrate_bezier(
    ctrl: Ptr, degree: Int, t0: Float64, t1: Float64, subdivisions: Int
) -> Float64:
    var total = 0.0
    var step = (t1 - t0) / Float64(subdivisions)
    for part in range(subdivisions):
        var left = t0 + Float64(part) * step
        total += adaptive_bezier(
            ctrl, degree, left, left + step, 1e-13 / Float64(subdivisions), 14
        )
    return total


def arc_speed(
    rx: Float64, ry: Float64, theta: Float64, delta: Float64, t: Float64
) -> Float64:
    var angle = theta + delta * t
    var sx = rx * sin(angle)
    var cy = ry * cos(angle)
    return abs(delta) * sqrt(sx * sx + cy * cy)


def gl_arc_interval(
    rx: Float64,
    ry: Float64,
    theta: Float64,
    delta: Float64,
    left: Float64,
    right: Float64,
) -> Float64:
    var mid = 0.5 * (left + right)
    var half = 0.5 * (right - left)
    var acc = 0.0
    for i in range(8):
        var offset = half * gl_x(i)
        acc += gl_w(i) * (
            arc_speed(rx, ry, theta, delta, mid - offset)
            + arc_speed(rx, ry, theta, delta, mid + offset)
        )
    return half * acc


def adaptive_arc(
    rx: Float64,
    ry: Float64,
    theta: Float64,
    delta: Float64,
    left: Float64,
    right: Float64,
    tolerance: Float64,
    depth: Int,
) -> Float64:
    var mid = 0.5 * (left + right)
    var coarse = gl_arc_interval(rx, ry, theta, delta, left, right)
    var fine = gl_arc_interval(rx, ry, theta, delta, left, mid) + gl_arc_interval(
        rx, ry, theta, delta, mid, right
    )
    if depth <= 0 or abs(fine - coarse) <= tolerance:
        return fine
    return adaptive_arc(
        rx, ry, theta, delta, left, mid, 0.5 * tolerance, depth - 1
    ) + adaptive_arc(
        rx, ry, theta, delta, mid, right, 0.5 * tolerance, depth - 1
    )


def integrate_arc(
    rx: Float64,
    ry: Float64,
    theta: Float64,
    delta: Float64,
    t0: Float64,
    t1: Float64,
    subdivisions: Int,
) -> Float64:
    var total = 0.0
    var step = (t1 - t0) / Float64(subdivisions)
    for part in range(subdivisions):
        var left = t0 + Float64(part) * step
        total += adaptive_arc(
            rx,
            ry,
            theta,
            delta,
            left,
            left + step,
            1e-13 / Float64(subdivisions),
            14,
        )
    return total


@export("msp_bezier_points")
def msp_bezier_points(
    ctrl_addr: Int, degree: Int, ts_addr: Int, n: Int, dst_addr: Int
) abi("C"):
    if n <= 0:
        return
    if ctrl_addr == 0 or ts_addr == 0 or dst_addr == 0:
        return
    if degree < 1 or degree > 3:
        return
    var ctrl = p(ctrl_addr)
    var ts = p(ts_addr)
    var dst = p(dst_addr)
    if n < PARALLEL_THRESHOLD:
        bezier_points_range(ctrl, degree, ts, dst, 0, n)
        return

    comptime W = simd_width_of[DType.float64]()
    var tasks = min(
        MAX_PARALLEL_TASKS, (n + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN
    )
    var chunk = (n + tasks - 1) // tasks
    chunk = ((chunk + W - 1) // W) * W

    @always_inline
    def worker(
        task: Int
    ) {imm ctrl, imm degree, imm ts, imm dst, imm n, imm chunk}:
        var start = task * chunk
        var end = min(start + chunk, n)
        bezier_points_range(ctrl, degree, ts, dst, start, end)

    parallelize(worker, tasks, tasks)


@export("msp_bezier_derivatives")
def msp_bezier_derivatives(
    ctrl_addr: Int, degree: Int, ts_addr: Int, n: Int, order: Int, dst_addr: Int
) abi("C"):
    if n <= 0:
        return
    if ctrl_addr == 0 or ts_addr == 0 or dst_addr == 0:
        return
    if degree < 1 or degree > 3 or order < 1:
        return
    var ctrl = p(ctrl_addr)
    var ts = p(ts_addr)
    var dst = p(dst_addr)
    bezier_derivatives_range(ctrl, degree, ts, dst, 0, n, order)


@export("msp_bezier_lengths")
def msp_bezier_lengths(
    ctrls_addr: Int,
    degree: Int,
    count: Int,
    t0: Float64,
    t1: Float64,
    subdivisions: Int,
    dst_addr: Int,
) abi("C"):
    if count <= 0:
        return
    if ctrls_addr == 0 or dst_addr == 0:
        return
    if degree < 1 or degree > 3 or subdivisions <= 0:
        return
    var ctrls = p(ctrls_addr)
    var dst = p(dst_addr)
    var stride = 2 * (degree + 1)
    for i in range(count):
        dst[i] = integrate_bezier(
            ctrls + i * stride, degree, t0, t1, subdivisions
        )


@export("msp_arc_points")
def msp_arc_points(
    cx: Float64,
    cy: Float64,
    rx: Float64,
    ry: Float64,
    phi: Float64,
    theta: Float64,
    delta: Float64,
    ts_addr: Int,
    n: Int,
    dst_addr: Int,
) abi("C"):
    if n <= 0:
        return
    if ts_addr == 0 or dst_addr == 0:
        return
    var ts = p(ts_addr)
    var dst = p(dst_addr)
    var cp = cos(phi)
    var sp = sin(phi)
    for i in range(n):
        var angle = theta + delta * ts[i]
        var ca = cos(angle)
        var sa = sin(angle)
        dst[2 * i] = rx * cp * ca - ry * sp * sa + cx
        dst[2 * i + 1] = rx * sp * ca + ry * cp * sa + cy


@export("msp_arc_derivatives")
def msp_arc_derivatives(
    rx: Float64,
    ry: Float64,
    phi: Float64,
    theta: Float64,
    delta: Float64,
    ts_addr: Int,
    n: Int,
    order: Int,
    dst_addr: Int,
) abi("C"):
    if n <= 0:
        return
    if ts_addr == 0 or dst_addr == 0 or order < 1:
        return
    var ts = p(ts_addr)
    var dst = p(dst_addr)
    var cp = cos(phi)
    var sp = sin(phi)
    for i in range(n):
        var angle = theta + delta * ts[i]
        var ca = cos(angle)
        var sa = sin(angle)
        var scale = 1.0
        for _ in range(order):
            scale *= delta
        var mode = order % 4
        if mode == 1:
            dst[2 * i] = scale * (-rx * cp * sa - ry * sp * ca)
            dst[2 * i + 1] = scale * (-rx * sp * sa + ry * cp * ca)
        elif mode == 2:
            dst[2 * i] = scale * (-rx * cp * ca + ry * sp * sa)
            dst[2 * i + 1] = scale * (-rx * sp * ca - ry * cp * sa)
        elif mode == 3:
            dst[2 * i] = scale * (rx * cp * sa + ry * sp * ca)
            dst[2 * i + 1] = scale * (rx * sp * sa - ry * cp * ca)
        else:
            dst[2 * i] = scale * (rx * cp * ca - ry * sp * sa)
            dst[2 * i + 1] = scale * (rx * sp * ca + ry * cp * sa)


@export("msp_arc_lengths")
def msp_arc_lengths(
    params_addr: Int,
    count: Int,
    t0: Float64,
    t1: Float64,
    subdivisions: Int,
    dst_addr: Int,
) abi("C"):
    if count <= 0:
        return
    if params_addr == 0 or dst_addr == 0 or subdivisions <= 0:
        return
    var params = p(params_addr)
    var dst = p(dst_addr)
    for i in range(count):
        var row = params + 4 * i
        dst[i] = integrate_arc(
            row[0], row[1], row[2], row[3], t0, t1, subdivisions
        )
