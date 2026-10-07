import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("mujoco.mjx")

from g1_cbf.barrier import point_box_margin_jax, segment_box_margin_jax  # noqa: E402
from g1_cbf.model_utils import point_box_margin  # noqa: E402

CENTER = jnp.zeros(3)
HALF = jnp.ones(3)


@pytest.mark.parametrize("p", [[4, 0, 0], [2, 2, 2], [0.2, 0.1, 0.0], [0.5, 1.5, 0.0]])
def test_jax_margin_matches_numpy(p):
    p = np.array(p, float)
    got = float(point_box_margin_jax(jnp.array(p), CENTER, HALF))
    assert got == pytest.approx(point_box_margin(p, np.zeros(3), np.ones(3)), abs=1e-4)


@pytest.mark.parametrize("p", [[0.1, 0.2, 0.3], [3.0, 0.5, 0.0]])
def test_gradient_is_finite_inside_and_outside(p):
    g = jax.grad(point_box_margin_jax)(jnp.array(p), CENTER, HALF)
    assert np.all(np.isfinite(np.array(g)))


def test_segment_margin_finds_closest_point():
    # segment passes above the box; closest approach is directly over it
    p0, p1 = jnp.array([-3.0, 0, 2.0]), jnp.array([3.0, 0, 2.0])
    assert float(segment_box_margin_jax(p0, p1, CENTER, HALF)) == pytest.approx(1.0, abs=1e-3)
