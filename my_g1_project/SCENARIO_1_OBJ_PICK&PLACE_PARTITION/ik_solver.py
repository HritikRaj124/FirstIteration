from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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

_FN_CACHE = {}

def get_qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    #jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]

    jnt_ids = []
    for n in joint_names:
        jnt_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n))
        bad = [n for n, j in zip(joint_names, jnt_ids) if j < 0]
        if bad:
            raise ValueError(f"Unknown joint: {bad}")
    return  np.array([model.jnt_qposadr[j] for j in jnt_ids])

def get_joint_limits(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []

    for n in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        a, b = model.jnt_range[j]
        lo.append(a if b > a else -np.pi)
        hi.append(b if b > a else np.pi)

    return np.array(lo), np.array(hi)

def _get_fns(mjx_model, qpos_ids : np.ndarray, site_id : int):
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
    max_iters: int = 100,
    tol: float = 1e-3,
    damping: float = 3e-3,
    max_step: float = 0.2,
) -> tuple[np.ndarray, bool, int, float]:

    joint_names = joint_names or LEFT_ARM_JOINT_NAMES

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_id}")

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

    for it in range(max_iters):
        err = target - site_pos_fn(arm_q, base_j)
        if float(jnp.linalg.norm(err)) < tol:
            converged = True
            break

        J = jac_fn(arm_q, base_j)
        dq = J.T @jnp.linalg.solve(J @ J.T + (damping ** 2) * eye3, err) # Levenberg-Marquardt Solver (Damped Pseudo-Inverse)

        n = jnp.linalg.norm(dq)
        dq = jnp.where(n > max_step, dq * max_step/ n, dq)
        arm_q = jnp.clip(arm_q + dq, lo_j, hi_j)

    resid = float(jnp.linalg.norm(target - site_pos_fn(arm_q, base_j)))
    return np.array(arm_q), converged, it + 1, resid

