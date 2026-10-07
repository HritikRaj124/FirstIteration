import numpy as np
import pytest

pytest.importorskip("proxsuite")
from g1_cbf.safety_filter import solve_safety_qp  # noqa: E402


def test_passthrough_when_constraint_inactive():
    qd = np.full(7, 0.5)
    out = solve_safety_qp(qd, [10.0], [np.ones(7)], alpha=1.0, qdot_max=2.0)
    assert out == pytest.approx(qd, abs=1e-4)


def test_cbf_constraint_is_enforced():
    # h = 0 (on the boundary) requires grad_h . qdot >= 0; the nominal command violates it
    grad = np.zeros(7)
    grad[0] = 1.0
    qd = np.zeros(7)
    qd[0] = -1.0
    out = solve_safety_qp(qd, [0.0], [grad], alpha=1.0, qdot_max=2.0)
    assert grad @ out >= -1e-4


def test_velocity_limit_respected():
    out = solve_safety_qp(np.full(7, 9.0), [10.0], [np.ones(7)], alpha=1.0, qdot_max=2.0)
    assert np.all(np.abs(out) <= 2.0 + 1e-4)
