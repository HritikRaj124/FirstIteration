import time
import numpy as np
import mujoco
import mujoco.viewer

from reacher_policy import (
    load_reacher_policy, get_dof_indices, get_actuator_ids, get_pelvis_pose,
    quat_apply_inverse, build_observation, apply_action,
    RIGHT_ARM_JOINT_NAMES, RIGHT_PALM_SITE, DECIMATION,
)
from ik_solver import get_qpos_indices
from capsule_geometry import get_arm_capsule_endpoints, arm_min_margin

SCENE_PATH = "scenarios/partition_task/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
model.opt.timestep = 0.005
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

session = load_reacher_policy("right_reacher.onnx")

PARTITION_CENTER = np.array([0.475, 0.0, 0.808])
PARTITION_HALF   = np.array([0.35, 0.01, 0.075])
ARM_RADIUS = 0.03
HAND_RADIUS = 0.06

target = np.array([0.3, 0.15, 0.80], dtype=np.float32)
print("target:", target)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_PALM_SITE)
qpos_ids = get_qpos_indices(model, RIGHT_ARM_JOINT_NAMES)
dof_ids = get_dof_indices(model, RIGHT_ARM_JOINT_NAMES)
arm_acts = get_actuator_ids(model, RIGHT_ARM_JOINT_NAMES)
pelvis_pos, pelvis_quat = get_pelvis_pose(model)

reach_target = quat_apply_inverse(pelvis_quat, target - pelvis_pos).astype(np.float32)
reach_orientation = np.zeros(3, dtype=np.float32)
last_arm_action = np.zeros(7, dtype=np.float32)
last_arm_target = data.qpos[qpos_ids].copy()

traj = {"shoulder": [], "elbow": [], "palm": [], "margin": []}
MAX_TICKS, TOL = 1000, 0.02

with mujoco.viewer.launch_passive(model, data) as viewer:
    reached, dist, tick = False, float("inf"), 0
    for tick in range(MAX_TICKS):
        if not viewer.is_running():
            break

        if tick % DECIMATION == 0:
            obs = build_observation(data, qpos_ids, dof_ids, site_id,
                                     reach_target, reach_orientation, last_arm_action,
                                     pelvis_pos, pelvis_quat)
            raw_action = session.run(None, {"observation": obs.reshape(1, -1)})[0][0]
            last_arm_target = apply_action(raw_action, last_arm_target)
            last_arm_action = raw_action.copy()
            data.ctrl[arm_acts] = last_arm_target

        mujoco.mj_step(model, data)

        # real per-tick capsule state, RIGHT arm, captured while it's actually moving
        upper_arm, palm_pos, forearm = get_arm_capsule_endpoints(model, data, side="right")
        margin = arm_min_margin(model, data, PARTITION_CENTER, PARTITION_HALF,
                                 ARM_RADIUS, HAND_RADIUS, side="right")
        traj["shoulder"].append(upper_arm[0].copy())
        traj["elbow"].append(upper_arm[1].copy())
        traj["palm"].append(palm_pos.copy())
        traj["margin"].append(margin)

        viewer.sync()
        time.sleep(0.02)

        dist = float(np.linalg.norm(target - data.site_xpos[site_id]))
        if dist < TOL:
            reached = True
            break

    print(f"reached={reached} ticks={len(traj['margin'])} final_dist={dist:.4f}")

    while viewer.is_running():   # hold the final pose until you close it yourself
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.02)

np.savez("trajectory_log.npz",
         shoulder=np.array(traj["shoulder"]),
         elbow=np.array(traj["elbow"]),
         palm=np.array(traj["palm"]),
         margin=np.array(traj["margin"]))
print(f"saved trajectory_log.npz with {len(traj['margin'])} ticks")