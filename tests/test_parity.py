import numpy as np
import pytest
import svgpathtools as upstream

import mojo_svgpathtools as mojo


SEGMENT_CASES = [
    (
        mojo.Line(1 + 2j, 4 - 3j),
        upstream.Line(1 + 2j, 4 - 3j),
    ),
    (
        mojo.QuadraticBezier(0j, 2 + 3j, 4 - 1j),
        upstream.QuadraticBezier(0j, 2 + 3j, 4 - 1j),
    ),
    (
        mojo.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j),
        upstream.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j),
    ),
    (
        mojo.Arc(1 + 2j, 4 + 2j, 35, True, False, 8 - 1j),
        upstream.Arc(1 + 2j, 4 + 2j, 35, True, False, 8 - 1j),
    ),
]


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_point_parity(ours, theirs):
    for t in np.linspace(0, 1, 31):
        assert ours.point(float(t)) == pytest.approx(theirs.point(float(t)), abs=2e-12)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_first_derivative_parity(ours, theirs):
    for t in np.linspace(0, 1, 17):
        assert ours.derivative(float(t)) == pytest.approx(
            theirs.derivative(float(t)), abs=2e-11
        )


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES[1:])
def test_higher_derivative_parity(ours, theirs):
    for order in (2, 3):
        for t in (0.0, 0.2, 0.7, 1.0):
            assert ours.derivative(t, order) == pytest.approx(
                theirs.derivative(t, order), abs=2e-10
            )


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_vectorized_points_parity(ours, theirs):
    ts = np.linspace(-0.25, 1.25, 10_003)
    expected = np.array([theirs.point(t) for t in ts])
    assert ours.points(ts) == pytest.approx(expected, abs=3e-12)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_vectorized_derivatives_parity(ours, theirs):
    ts = np.linspace(-0.25, 1.25, 10_003)
    expected = np.array([theirs.derivative(t) for t in ts])
    assert ours.derivatives(ts) == pytest.approx(expected, abs=3e-11)


@pytest.mark.parametrize("ours,_", SEGMENT_CASES)
def test_vectorized_empty_inputs(ours, _):
    assert ours.points([]).shape == (0,)
    assert ours.points([]).dtype == np.complex128
    assert ours.derivatives([]).shape == (0,)
    assert ours.derivatives([]).dtype == np.complex128


@pytest.mark.parametrize("values", [[0.5 + 1j], ["0.5"], [np.inf], [np.nan]])
def test_vectorized_inputs_reject_unsafe_float64_coercions(values):
    curve = mojo.CubicBezier(0j, 1 + 1j, 2 + 1j, 3j)
    with pytest.raises((TypeError, ValueError)):
        curve.points(values)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES[:3])
@pytest.mark.parametrize("size", [1, 3, 5, 7])
def test_bezier_simd_tail_parity(ours, theirs, size):
    ts = np.linspace(-0.25, 1.25, size)
    expected_points = np.array([theirs.point(t) for t in ts])
    expected_derivatives = np.array([theirs.derivative(t) for t in ts])
    assert ours.points(ts) == pytest.approx(expected_points, abs=3e-12)
    assert ours.derivatives(ts) == pytest.approx(expected_derivatives, abs=3e-11)


@pytest.mark.parametrize("size", [262_143, 262_145])
def test_bezier_parallel_threshold_parity(size):
    ours = mojo.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j)
    theirs = upstream.CubicBezier(0j, 1 + 4j, 5 - 2j, 7 + 1j)
    ts = np.linspace(-0.25, 1.25, size)
    assert ours.points(ts) == pytest.approx(theirs.point(ts), abs=3e-12)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_length_parity(ours, theirs):
    assert ours.length() == pytest.approx(theirs.length(), rel=2e-11, abs=2e-11)
    assert ours.length(0.13, 0.83) == pytest.approx(
        theirs.length(0.13, 0.83), rel=2e-11, abs=2e-11
    )


def test_eccentric_arc_length_parity():
    ours = mojo.Arc(0j, 100 + 1j, 60, True, True, 20 + 30j)
    theirs = upstream.Arc(0j, 100 + 1j, 60, True, True, 20 + 30j)
    assert ours.length() == pytest.approx(theirs.length(), rel=2e-11)


def test_batched_arc_lengths_match_upstream():
    specs = [
        (0j, 3 + 2j, 0, False, True, 4 + 1j),
        (1 - 2j, 8 + 1j, 47, True, False, 7 + 5j),
        (-4 + 3j, 2 + 9j, -31, False, False, 5 - 6j),
        (2 + 1j, 20 + 3j, 80, True, True, 15 + 17j),
    ]
    ours = [mojo.Arc(*spec) for spec in specs]
    theirs = [upstream.Arc(*spec) for spec in specs]
    assert mojo.arc_lengths(ours) == pytest.approx(
        [segment.length() for segment in theirs], rel=2e-10, abs=2e-11
    )


def test_empty_and_wrong_type_batch_lengths():
    assert mojo.bezier_lengths([]).shape == (0,)
    assert mojo.arc_lengths([]).shape == (0,)
    with pytest.raises(TypeError):
        mojo.arc_lengths([mojo.Line(0j, 1j)])


def test_random_batched_bezier_lengths_match_upstream():
    rng = np.random.default_rng(7)
    controls = rng.normal(size=(80, 4, 2))
    ours = [mojo.CubicBezier(*(row[:, 0] + 1j * row[:, 1])) for row in controls]
    theirs = [
        upstream.CubicBezier(*(row[:, 0] + 1j * row[:, 1])) for row in controls
    ]
    got = mojo.bezier_lengths(ours)
    expected = np.array([segment.length() for segment in theirs])
    assert got == pytest.approx(expected, rel=5e-10, abs=2e-11)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_unit_tangent_normal_and_curvature(ours, theirs):
    for t in (0.1, 0.4, 0.9):
        assert ours.unit_tangent(t) == pytest.approx(theirs.unit_tangent(t), abs=2e-11)
        assert ours.normal(t) == pytest.approx(theirs.normal(t), abs=2e-11)
        assert ours.curvature(t) == pytest.approx(theirs.curvature(t), rel=2e-10)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_bbox_parity(ours, theirs):
    assert ours.bbox() == pytest.approx(theirs.bbox(), abs=2e-11)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_split_parity(ours, theirs):
    left, right = ours.split(0.37)
    expected_left, expected_right = theirs.split(0.37)
    for t in np.linspace(0, 1, 9):
        assert left.point(t) == pytest.approx(expected_left.point(t), abs=2e-11)
        assert right.point(t) == pytest.approx(expected_right.point(t), abs=2e-11)


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_crop_and_reverse_parity(ours, theirs):
    cropped = ours.cropped(0.2, 0.8)
    expected = theirs.cropped(0.2, 0.8)
    reversed_segment = ours.reversed()
    for t in np.linspace(0, 1, 9):
        assert cropped.point(t) == pytest.approx(expected.point(t), abs=3e-11)
        assert reversed_segment.point(t) == pytest.approx(
            theirs.reversed().point(t), abs=3e-11
        )


@pytest.mark.parametrize("ours,theirs", SEGMENT_CASES)
def test_inverse_length_parity(ours, theirs):
    s = 0.42 * theirs.length()
    assert ours.ilength(s) == pytest.approx(theirs.ilength(s), abs=2e-10)


def test_line_line_intersection_parity():
    a = mojo.Line(0j, 2 + 2j)
    b = mojo.Line(0 + 2j, 2 + 0j)
    expected = upstream.Line(0j, 2 + 2j).intersect(
        upstream.Line(0 + 2j, 2 + 0j)
    )
    assert a.intersect(b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "ours,theirs",
    [
        (
            mojo.QuadraticBezier(0j, 1 + 3j, 2 + 0j),
            upstream.QuadraticBezier(0j, 1 + 3j, 2 + 0j),
        ),
        (
            mojo.CubicBezier(0j, 0 + 3j, 2 - 3j, 2 + 0j),
            upstream.CubicBezier(0j, 0 + 3j, 2 - 3j, 2 + 0j),
        ),
    ],
)
def test_bezier_line_intersection_parity(ours, theirs):
    line = mojo.Line(-1 + 0.5j, 3 + 0.5j)
    expected_line = upstream.Line(-1 + 0.5j, 3 + 0.5j)
    assert ours.intersect(line) == pytest.approx(theirs.intersect(expected_line))


def _paths():
    ours = mojo.Path(
        mojo.Line(0j, 2 + 0j),
        mojo.QuadraticBezier(2 + 0j, 3 + 2j, 4 + 0j),
        mojo.CubicBezier(4 + 0j, 5 - 2j, 6 + 2j, 7 + 0j),
        mojo.Arc(7 + 0j, 3 + 2j, 20, False, True, 10 + 2j),
    )
    theirs = upstream.Path(
        upstream.Line(0j, 2 + 0j),
        upstream.QuadraticBezier(2 + 0j, 3 + 2j, 4 + 0j),
        upstream.CubicBezier(4 + 0j, 5 - 2j, 6 + 2j, 7 + 0j),
        upstream.Arc(7 + 0j, 3 + 2j, 20, False, True, 10 + 2j),
    )
    return ours, theirs


def test_path_length_point_and_bbox_parity():
    ours, theirs = _paths()
    assert ours.length() == pytest.approx(theirs.length(), rel=2e-11)
    assert ours.bbox() == pytest.approx(theirs.bbox(), abs=3e-11)
    for t in np.linspace(0, 1, 31):
        assert ours.point(t) == pytest.approx(theirs.point(t), abs=4e-11)


def test_path_differential_geometry_parity():
    ours, theirs = _paths()
    for t in (0.05, 0.3, 0.57, 0.9):
        assert ours.derivative(t) == pytest.approx(theirs.derivative(t), rel=2e-10)
        assert ours.unit_tangent(t) == pytest.approx(theirs.unit_tangent(t), abs=2e-11)
        assert ours.normal(t) == pytest.approx(theirs.normal(t), abs=2e-11)
        assert ours.curvature(t) == pytest.approx(theirs.curvature(t), rel=2e-10)


def test_path_crop_reverse_and_inverse_length_parity():
    ours, theirs = _paths()
    cropped, expected = ours.cropped(0.17, 0.86), theirs.cropped(0.17, 0.86)
    assert cropped.length() == pytest.approx(expected.length(), rel=3e-10)
    for t in np.linspace(0, 1, 15):
        assert cropped.point(t) == pytest.approx(expected.point(t), abs=5e-10)
        assert ours.reversed().point(t) == pytest.approx(
            theirs.reversed().point(t), abs=5e-10
        )
    s = 0.62 * theirs.length()
    assert ours.ilength(s) == pytest.approx(theirs.ilength(s), abs=2e-10)


def test_continuous_subpaths_match_upstream_behavior():
    path = mojo.Path(
        mojo.Line(0j, 1j),
        mojo.Line(1j, 2j),
        mojo.Line(5j, 6j),
    )
    parts = path.continuous_subpaths()
    assert [len(part) for part in parts] == [2, 1]
    assert all(part.iscontinuous() for part in parts)


def test_parse_absolute_commands_parity():
    data = (
        "M 0,0 L 2,0 H 3 V 1 C 4,2 5,2 6,1 "
        "S 8,0 9,1 Q 10,3 11,1 T 13,1 A 2,3 25 0 1 16,4 Z"
    )
    ours, theirs = mojo.parse_path(data), upstream.parse_path(data)
    assert len(ours) == len(theirs)
    for own_segment, upstream_segment in zip(ours, theirs):
        for t in np.linspace(0, 1, 7):
            assert own_segment.point(t) == pytest.approx(
                upstream_segment.point(t), abs=5e-11
            )


def test_parse_relative_and_repeated_commands_parity():
    data = "m 1,2 2,0 0,2 c 1,0 2,1 3,1 1,0 2,-1 3,-1 q 1,2 2,0 1,-2 2,0 z"
    ours, theirs = mojo.parse_path(data), upstream.parse_path(data)
    assert len(ours) == len(theirs)
    for own_segment, upstream_segment in zip(ours, theirs):
        for t in (0.0, 0.25, 0.75, 1.0):
            assert own_segment.point(t) == pytest.approx(
                upstream_segment.point(t), abs=5e-11
            )


def test_parse_all_relative_commands_parity():
    data = (
        "m 1,1 l 2,0 h 1 v 1 c 1,1 2,1 3,0 "
        "s 2,-1 3,0 q 1,2 2,0 t 2,0 a 2,3 20 0 1 3,2 z"
    )
    ours, theirs = mojo.parse_path(data), upstream.parse_path(data)
    assert len(ours) == len(theirs)
    for own_segment, upstream_segment in zip(ours, theirs):
        for t in np.linspace(0, 1, 7):
            assert own_segment.point(t) == pytest.approx(
                upstream_segment.point(t), abs=5e-11
            )
