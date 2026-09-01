import ik_solver
import mujoco
from mujoco import mjx
import numpy as np

from ik_solver import solve_ik_jax, LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE

model = mujoco.MjModel.from_xml_path("scenarios/partition_task/scene.xml")
model.geom_contype[:] = 0
model.geom_conaffinity[:] = 0
mjx_model = mjx.put_model(model)

qpos_ids = ik_solver.get_qpos_indices(model, ik_solver.LEFT_ARM_JOINT_NAMES)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ik_solver.LEFT_PALM_SITE)

site_pos_fn, jac_fn = ik_solver._get_fns(mjx_model, qpos_ids, site_id)

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
base_qpos = np.array(data.qpos)
arm_q = base_qpos[qpos_ids]

import jax.numpy as jnp
pos = site_pos_fn((arm_q), jnp.array(base_qpos))
print("palm position: ", pos)

J = jac_fn(jnp.array(arm_q), jnp.array(base_qpos))
print("Jacobian shape:", J.shape)
print(J)

print("qpos_ids:", qpos_ids)
print("site_id:", site_id)
print("arm_q:", arm_q)

eps = 0.1
arm_q_pert = jnp.array(arm_q).at[0].add(eps)   # nudge the first arm joint
pos0 = site_pos_fn(jnp.array(arm_q), jnp.array(base_qpos))
pos1 = site_pos_fn(arm_q_pert, jnp.array(base_qpos))
print("pos before nudge:", pos0)
print("pos after nudge: ", pos1)
print("difference:      ", pos1 - pos0)

obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "movable_object")
obj_pos = data.xpos[obj_id].copy()

import time

target = obj_pos + np.array([0.0, 0.0, 0.10])   # 10cm above the object, matching earlier runs

t0 = time.time()
q_des, converged, n_iters, resid = solve_ik_jax(
    model, mjx_model, data.qpos.copy(), target,
    joint_names=LEFT_ARM_JOINT_NAMES, site_name=LEFT_PALM_SITE,
)
print(f"\nsolve_ik_jax result:")
print(f"  converged: {converged}")
print(f"  iterations: {n_iters}")
print(f"  residual: {resid:.5f} m")
print(f"  q_des: {q_des}")
print(f"  time: {time.time() - t0:.2f}s")