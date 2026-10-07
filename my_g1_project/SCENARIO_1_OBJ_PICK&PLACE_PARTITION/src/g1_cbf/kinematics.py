"""Damped-least-squares position IK for the arm, differentiated with JAX/MJX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from .config import LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE, IKConfig
from .model_utils import get_joint_limits, get_qpos_indices, name2id

_FN_CACHE = {}


def _get_fns(mjx_model, qpos_ids: np.ndarray, site_id: int):
    key = (id(mjx_model), tuple(qpos_ids.tolist()), site_id)
    if key in _FN_CACHE:
        return _FN_CACHE[key]

    qpos_ids_j = jnp.array(qpos_ids)
    template = mjx.make_data(mjx_model)

    def site_pos(arm_q, base_qpos):
        qpos = base_qpos.at[qpos_ids_j].set(arm_q)
        d = template.replace(qpos=qpos)
        d = mjx.forward(mjx_model, d)
        return d.site_xpos[site_id]

    fns = (jax.jit(site_pos), jax.jit(jax.jacfwd(site_pos, argnums=0)))
    _FN_CACHE[key] = fns
    return fns


def solve_ik_jax(
    model: mujoco.MjModel,
    mjx_model,
    base_qpos: np.ndarray,
    target_pos: np.ndarray,
    joint_names: list[str] | None = None,
    site_name: str = LEFT_PALM_SITE,
    max_iters: int = IKConfig.max_iters,
    tol: float = IKConfig.tol,
    damping: float = IKConfig.damping,
    max_step: float = IKConfig.max_step,
) -> tuple[np.ndarray, bool, int, float]:

    joint_names = joint_names or list(LEFT_ARM_JOINT_NAMES)

    site_id = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    qpos_ids = get_qpos_indices(model, joint_names)
    lo, hi = get_joint_limits(model, joint_names)
    lo_j, hi_j = jnp.array(lo), jnp.array(hi)

    site_pos_fn, jac_fn = _get_fns(mjx_model, qpos_ids, site_id)

    base_j = jnp.array(base_qpos)
    arm_q = base_j[jnp.array(qpos_ids)]
    target = jnp.array(target_pos)
    eye3 = jnp.eye(3)

    converged = False
    it = 0

    for it in range(max_iters):  # noqa: B007 (it is reported after the loop)
        err = target - site_pos_fn(arm_q, base_j)
        if float(jnp.linalg.norm(err)) < tol:
            converged = True
            break

        J = jac_fn(arm_q, base_j)
        dq = J.T @ jnp.linalg.solve(
            J @ J.T + (damping**2) * eye3, err
        )  # Levenberg-Marquardt Solver (Damped Pseudo-Inverse)

        n = jnp.linalg.norm(dq)
        dq = jnp.where(n > max_step, dq * max_step / n, dq)
        arm_q = jnp.clip(arm_q + dq, lo_j, hi_j)

    resid = float(jnp.linalg.norm(target - site_pos_fn(arm_q, base_j)))
    return np.array(arm_q), converged, it + 1, resid
