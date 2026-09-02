""" JAX native forward kinematics along with the capsule margin and the
    differentiation of the barrier function h(q)

    This file consists of the math to derive the marginal distance between the
    obstacle and the arm. This is done using the Signed Distance Function.
    This file is built entirely on JAX operations (mjx + jnp) so jax.grad can
    differentiate through it. The closest-point search runs on numpy, only the
    gradient at the fixed point needs to be JAX native."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

LEFT_PALM_SITE = "left_palm"
LEFT_SHOULDER_BODY = "left_shoulder_pitch_link"
LEFT_ELBOW_BODY = "left_elbow_link"

_FN_CACHE = {}

def _get_capsule_fns(mjx_model, qpos_ids, shoulder_body_id, elbow_body_id, palm_site_id):
    key = (id(mjx_model), tuple(qpos_ids.tolist()), shoulder_body_id, elbow_body_id, palm_site_id)
    if key in _FN_CACHE:
        return _FN_CACHE[key]

    qpos_ids_j = jnp.array(qpos_ids)
    template = mjx.make_data(mjx_model)

    def capsule_points(arm_q, base_qpos):
        qpos = base_qpos.at[qpos_ids_j].set(arm_q)
        d = template.replace(qpos=qpos)
        d = mjx.forward(mjx_model, d)
        shoulder = d.xpos[shoulder_body_id]
        elbow = d.xpos[elbow_body_id]
        palm = d.site_xpos[palm_site_id]
        return jnp.stack([shoulder, elbow, palm])

    fn = jax.jit(capsule_points)
    _FN_CACHE[key] = fn
    return fn

def get_capsule_points(model, mjx_model, qpos_ids, base_qpos, arm_q):
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_SHOULDER_BODY)
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_ELBOW_BODY)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)
    if shoulder_id < 0 or elbow_id < 0 or palm_id < 0:
        raise ValueError("Unknown shoulder/elbow body or palm site")

    fn = _get_capsule_fns(mjx_model, qpos_ids, shoulder_id, elbow_id, palm_id)
    return fn(jnp.array(arm_q), jnp.array(base_qpos))

# Signed Distance Function to calculate the margin distance of a point in space and a box (in this case robot arm and partition wall)
def point_box_margin_jax(p, center, half_size, radius=0.0):
    diff = jnp.abs(p - center) - half_size - radius                                                                     # measures how far the point is from the box walls along each axis. Negative means inside the wall; positive means outside
    inside_val = jnp.max(diff)                                                                                          # it finds the largest number in our axis list
    is_inside = jnp.all(diff < 0)                                                                                       # check whether all the axis in the point vector are inside the box, all negative --> point inside box
    clamped = jnp.maximum(diff, 0)                                                                                   # It sets any negative numbers to 0
    safe_clamped = jnp.where(is_inside, jnp.ones_like(clamped), clamped)                                                # we replace the negative or zero with 1 to avoid the Nan error occurred by 0 at the gradient
    outside_val = jnp.sqrt(jnp.sum(safe_clamped ** 2) + 1e-12)                                                          # calculates the straight-line diagonal distance using the Pythagorean theorem (\(A^2 + B^2 + C^2\))
    return jnp.where(jnp.all(diff < 0), inside_val, outside_val)                                                        # sees whether the point is inside the box or outside, inside --> return inside_val and outside --> return outside_val

# Ternary search algorithm to find point on line-segment closest to the box ( line segment: shoulder to elbow --> upper arm, elbow to palm --> forearm)
def segment_box_margin_jax(p0, p1, center, half_size, radius=0.0, n_iters=20):

    # This helper function picks a spot on the line segment (ARM) based on t, calculates its 3D coordinate p, and feeds it into the point_box_margin_jax
    def margin_at_t(t):
        p = p0 +t*(p1 - p0)                                                                                             # standard linear interpolation
        return point_box_margin_jax(p, center, half_size, radius)

    def body(i, bounds):
        lo, hi = bounds
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3

        f1 = margin_at_t(m1)
        f2 = margin_at_t(m2)

        lo_new = jnp.where(f1 <= f2, lo,m1)
        hi_new = jnp.where(f1 <= f2, m2, hi)

        return (lo_new, hi_new)

    lo, hi = jax.lax.fori_loop(0, n_iters, body, (0,1))
    t_star = (lo + hi) / 2
    return margin_at_t(t_star)

def make_segment_h_fn(model, mjx_model, qpos_ids, base_qpos, obstacle_center, obstacle_half_size, capsule_radius, segment = "forearm"):
    """segment: 'upper' = shoulder->elbow and 'forearm' = elbow->palm """

    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_SHOULDER_BODY)
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_ELBOW_BODY)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)

    points_fn = _get_capsule_fns(mjx_model, qpos_ids, shoulder_id, elbow_id, palm_id)
    base_qpos_j = jnp.array(base_qpos)
    center_j = jnp.array(obstacle_center)
    half_size_j = jnp.array(obstacle_half_size)

    def h(arm_q):
        pts = points_fn(arm_q, base_qpos_j)
        shoulder, elbow, palm = pts[0], pts[1], pts[2]
        p0, p1 = (shoulder, elbow) if segment == "upper" else (elbow, palm)
        return segment_box_margin_jax(p0, p1, center_j, half_size_j, capsule_radius)

    return jax.jit(h)
