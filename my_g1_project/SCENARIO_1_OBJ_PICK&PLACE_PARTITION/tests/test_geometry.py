import numpy as np
import pytest

from g1_cbf.model_utils import minjerk, point_box_margin

CENTER = np.zeros(3)
HALF = np.ones(3)


def test_minjerk_endpoints_and_monotone():
    y = minjerk(np.linspace(0, 1, 101))
    assert y[0] == pytest.approx(0) and y[-1] == pytest.approx(1)
    assert np.all(np.diff(y) >= 0)


def test_margin_outside_is_euclidean_distance():
    assert point_box_margin(np.array([4.0, 0, 0]), CENTER, HALF) == pytest.approx(3.0)
    assert point_box_margin(np.array([2.0, 2.0, 2.0]), CENTER, HALF) == pytest.approx(np.sqrt(3))


def test_margin_inside_is_negative_depth():
    assert point_box_margin(np.zeros(3), CENTER, HALF) == pytest.approx(-1.0)


def test_margin_radius_inflates_box():
    assert point_box_margin(np.array([1.5, 0, 0]), CENTER, HALF, r=0.5) == pytest.approx(0.0)
