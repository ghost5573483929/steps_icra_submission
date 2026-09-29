"""Convention under test: interior rows use ``xy[i+1] - xy[i-1]``, endpoints use
their one-sided difference, and displacements at or below ``1e-8`` are ``NaN``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from heading import HEADING_SPEED_EPSILON, central_difference_heading


PI = math.pi


def test_speed_floor_is_the_pinned_convention():
    assert HEADING_SPEED_EPSILON == 1e-8


def test_single_position_track_has_no_heading():
    result = central_difference_heading(np.asarray([[3.25, -7.5]]))
    assert result.shape == (1,)
    assert result.dtype == np.float64
    assert math.isnan(float(result[0]))


def test_two_positions_use_the_same_one_sided_difference_twice():
    # delta[0] = delta[1] = (1, 1); atan2(1, 1) = pi/4 for both rows.
    result = central_difference_heading(np.asarray([[0.0, 0.0], [1.0, 1.0]]))
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [PI / 4, PI / 4], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize(
    ("direction", "expected_angle"),
    [
        ((1.0, 0.0), 0.0),
        ((0.0, 1.0), PI / 2),
        # dy of the stencil is +0.0, so atan2(+0, -) is +pi, not -pi.
        ((-1.0, 0.0), PI),
    ],
    ids=["positive_x", "positive_y", "negative_x"],
)
def test_axis_aligned_straight_line_holds_one_exact_angle(direction, expected_angle):
    # Every stencil, one-sided or central, is a positive multiple of `direction`.
    xy = np.arange(4.0).reshape(4, 1) * np.asarray(direction)
    np.testing.assert_array_equal(central_difference_heading(xy), [expected_angle] * 4)


def test_interior_row_uses_the_central_difference_not_a_one_sided_one():
    # An L-shaped turn distinguishes the stencils:
    #   row 0: p1 - p0 = (1, 0)  -> 0
    #   row 1: p2 - p0 = (1, 1)  -> pi/4   (a one-sided difference would give 0 or pi/2)
    #   row 2: p2 - p1 = (0, 1)  -> pi/2
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        central_difference_heading(xy), [0.0, PI / 4, PI / 2], rtol=1e-15, atol=0.0
    )


def test_negative_coordinates_and_third_quadrant_angle():
    # delta = (-3, -4) for both rows; the 3-4-5 triangle in the third quadrant.
    expected = math.atan2(-4.0, -3.0)
    assert -PI < expected < -PI / 2
    xy = np.asarray([[-1.0, -1.0], [-4.0, -5.0]])
    np.testing.assert_allclose(central_difference_heading(xy), [expected, expected], rtol=1e-15, atol=0.0)


def test_fully_stationary_track_is_all_nan():
    # Every difference is exactly (0, 0), so speed 0 <= 1e-8 and heading is undefined.
    xy = np.asarray([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    result = central_difference_heading(xy)
    assert result.shape == (4,)
    assert np.isnan(result).all()


def test_track_stationary_only_in_the_interior():
    # rows 0,1,3,4 have a (1, 0) difference; row 2's central difference p3 - p1 is
    # exactly (0, 0) because the track pauses for three samples.
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = central_difference_heading(xy)
    np.testing.assert_array_equal(result[[0, 1, 3, 4]], [0.0, 0.0, 0.0, 0.0])
    assert math.isnan(float(result[2]))


def test_coincident_consecutive_pair_keeps_neighbours_defined():
    # rows 1 and 2 coincide, yet their central differences p2 - p0 = (1, 1) and
    # p3 - p1 = (2, 1) are nonzero, so only a genuinely zero stencil yields NaN.
    xy = np.asarray([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [3.0, 2.0]])
    result = central_difference_heading(xy)
    assert not np.isnan(result).any()
    np.testing.assert_allclose(
        result,
        [PI / 4, PI / 4, math.atan2(1.0, 2.0), math.atan2(1.0, 2.0)],
        rtol=1e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("displacement", "expected"),
    [
        (1e-9, np.nan),
        # At the floor exactly: a `>` guard and a `>=` guard differ only here.
        (HEADING_SPEED_EPSILON, np.nan),
        (HEADING_SPEED_EPSILON * 1.5, 0.0),
        (1e-7, 0.0),
    ],
    ids=["well_below", "exactly_at", "just_above", "well_above"],
)
def test_speed_floor_decides_whether_a_displacement_has_a_heading(displacement, expected):
    # The displacement is along +x, so a kept heading is exactly 0.0.
    xy = np.asarray([[0.0, 0.0], [displacement, 0.0]])
    np.testing.assert_array_equal(central_difference_heading(xy), [expected, expected])


def test_headings_are_wrapped_into_the_closed_pi_interval():
    angles = np.linspace(0.0, 2 * PI, 37)
    xy = np.c_[np.cos(angles) * 4.0, np.sin(angles) * 4.0]
    result = central_difference_heading(xy)
    assert np.isfinite(result).all()
    assert np.all(result >= -PI) and np.all(result <= PI)


def test_translation_leaves_heading_unchanged():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.5, 0.75], [2.0, -1.0], [4.0, -3.0]])
    shifted = xy + np.asarray([1234.5, -987.25])
    np.testing.assert_allclose(
        central_difference_heading(shifted), central_difference_heading(xy), rtol=1e-12, atol=0.0
    )


def test_rotating_the_track_rotates_every_heading_by_the_same_angle():
    theta = 0.7
    xy = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.5, 0.75], [2.0, -1.0], [4.0, -3.0]])
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    rotated = xy @ rotation.T
    before = central_difference_heading(xy)
    after = central_difference_heading(rotated)
    difference = (after - before - theta + PI) % (2 * PI) - PI
    np.testing.assert_allclose(difference, np.zeros(len(xy)), rtol=0.0, atol=1e-12)


def test_reversing_the_track_flips_every_heading_by_pi():
    # Reversal negates every stencil difference, so heading -> heading + pi.
    xy = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.5, 0.75], [2.0, -1.0], [4.0, -3.0]])
    forward = central_difference_heading(xy)
    backward = central_difference_heading(xy[::-1])
    wrapped = (backward - forward[::-1]) % (2 * PI) - PI
    np.testing.assert_allclose(wrapped, np.zeros(len(xy)), rtol=0.0, atol=1e-12)


def test_integer_input_is_promoted_to_float():
    xy = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    result = central_difference_heading(xy)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [PI / 4, PI / 4, PI / 4], rtol=1e-15, atol=0.0)


def test_input_is_not_modified():
    xy = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    before = xy.copy()
    central_difference_heading(xy)
    np.testing.assert_array_equal(xy, before)


def test_empty_track_is_rejected():
    with pytest.raises(ValueError, match="no positions"):
        central_difference_heading(np.empty((0, 2)))


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((3, 3)), np.zeros((2, 2, 2)), np.zeros((3, 1))])
def test_non_two_column_input_is_rejected(bad):
    with pytest.raises(ValueError, match=r"\(n, 2\) array"):
        central_difference_heading(bad)


def test_interior_central_difference_magnitude_is_not_halved():
    # arctan2 is scale-invariant, so a halved central difference shows up only in
    # the epsilon band: the two-step span is above the floor, each one-step gap below.
    step = HEADING_SPEED_EPSILON * 0.75
    xy = np.array([[0.0, 0.0], [step, 0.0], [2.0 * step, 0.0]])
    heading = central_difference_heading(xy)
    assert not np.isnan(heading[1]), "interior sample must survive at full stencil scale"
    np.testing.assert_allclose(heading[1], 0.0)
