"""ONNX right-arm reacher policy integration for the G1.

Observation layout and action interpretation reverse-engineered from the
vendor's run.py (G1Controller.step(), reach-mode branch) and
model_config.json (default_joint_pos / action_scales, right-arm subset).

IMPORTANT: run.py assumes a FLOATING pelvis (7-slot freejoint qpos offset,
6-slot qvel offset). Our model WELDS the pelvis (dofnum=0) -- so all index
lookups here use mj_name2id/jnt_qposadr dynamically, like ik_solver.py
already does, instead of the vendor's hardcoded 7+i / 6+i. Pelvis pose is
therefore a fixed constant read from the XML, not from data.qpos.
"""

from __future__ import annotations

import time
import numpy as np
import mujoco
import onnxruntime as ort

from ik_solver import get_qpos_indices

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_PALM_SITE = "right_palm"

# From model_config.json's default_joint_pos / action_scales, right-arm
# subset (there's no "right_reacher" override block in the config, so
# per run.py's fallback logic, these global defaults ARE what training used).
ARM_DEFAULT_POS = np.array([0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_ACTION_SCALES = np.array(
    [0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501],
    dtype=np.float32,
)

ARM_MAX_DELTA = 0.012      # rad per policy call -- rate limit, from run.py
DECIMATION = 4              # policy runs every 4 physics steps (50 Hz @ 200 Hz)
PHYSICS_TIMESTEP = 0.005    # must match training


def get_dof_indices(model, joint_names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    return np.array([model.jnt_dofadr[j] for j in ids])


def get_actuator_ids(model, joint_names):
    jids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names}
    return np.array([a for a in range(model.nu)
                      if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                      and model.actuator_trnid[a, 0] in jids])


def load_reacher_policy(onnx_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(onnx_path)
    (obs_input,) = session.get_inputs()
    (action_output,) = session.get_outputs()
    print(f"Loaded reacher policy: {obs_input.name} {obs_input.shape} "
          f"-> {action_output.name} {action_output.shape}")
    return session


def get_pelvis_pose(model):
    """Our pelvis is welded -- its world pose is fixed, defined in the XML,
    not something read from data.qpos (which has no slots for it here)."""
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return model.body_pos[pid].copy(), model.body_quat[pid].copy()


def quat_apply_inverse(quat, vec):
    """Rotate vec by the inverse of quat: world frame -> local frame."""
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def get_projected_gravity(pelvis_quat):
    return quat_apply_inverse(pelvis_quat, np.array([0.0, 0.0, -1.0]))


def get_palm_pos_in_pelvis(data, site_id, pelvis_pos, pelvis_quat):
    palm_world = data.site_xpos[site_id].copy()
    return quat_apply_inverse(pelvis_quat, palm_world - pelvis_pos)


def get_palm_orientation_in_pelvis(data, site_id, pelvis_quat):
    mat = data.site_xmat[site_id].reshape(3, 3)
    palm_q = np.zeros(4)
    mujoco.mju_mat2Quat(palm_q, mat.flatten())
    w1, x1, y1, z1 = pelvis_quat[0], -pelvis_quat[1], -pelvis_quat[2], -pelvis_quat[3]
    w2, x2, y2, z2 = palm_q
    rel = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    w, x, y, z = rel
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def build_observation(data, qpos_ids, dof_ids, site_id,
                       reach_target, reach_orientation, last_arm_action,
                       pelvis_pos, pelvis_quat):
    """36 raw (unnormalized) values -- the ONNX graph normalizes internally."""
    arm_pos = data.qpos[qpos_ids] - ARM_DEFAULT_POS
    arm_vel = data.qvel[dof_ids]
    palm_pos = get_palm_pos_in_pelvis(data, site_id, pelvis_pos, pelvis_quat)
    palm_ori = get_palm_orientation_in_pelvis(data, site_id, pelvis_quat)
    proj_grav = get_projected_gravity(pelvis_quat)

    return np.concatenate([
        reach_target, reach_orientation,
        palm_pos, palm_ori,
        arm_pos, arm_vel,
        last_arm_action,
        proj_grav,
    ]).astype(np.float32)


def apply_action(raw_action, last_arm_target):
    """Scale + default offset, then rate-limit relative to the previous
    commanded target. Stateful -- pass the previous return value back in."""
    arm_target = ARM_DEFAULT_POS + raw_action * ARM_ACTION_SCALES
    if last_arm_target is not None:
        delta = np.clip(arm_target - last_arm_target, -ARM_MAX_DELTA, ARM_MAX_DELTA)
        arm_target = last_arm_target + delta
    return arm_target


def run_reacher_to_target(
    model, data, session, target_world_pos, target_orientation=None,
    joint_names=RIGHT_ARM_JOINT_NAMES, site_name=RIGHT_PALM_SITE,
    max_ticks=2000, tol=0.02, viewer=None, viz_sleep=0.03, log=None,
):
    """Closed-loop reach, policy called every DECIMATION physics steps."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")

    qpos_ids = get_qpos_indices(model, joint_names)
    dof_ids = get_dof_indices(model, joint_names)
    arm_acts = get_actuator_ids(model, joint_names)

    pelvis_pos, pelvis_quat = get_pelvis_pose(model)

    target_world_pos = np.array(target_world_pos, dtype=np.float32)
    reach_target = quat_apply_inverse(pelvis_quat, target_world_pos - pelvis_pos).astype(np.float32)
    reach_orientation = (np.zeros(3, dtype=np.float32) if target_orientation is None
                          else np.array(target_orientation, dtype=np.float32))

    last_arm_action = np.zeros(7, dtype=np.float32)
    last_arm_target = data.qpos[qpos_ids].copy()  # start from current pose, like run.py

    reached, dist, tick = False, float("inf"), 0
    for tick in range(max_ticks):
        if tick % DECIMATION == 0:
            obs = build_observation(data, qpos_ids, dof_ids, site_id,
                                     reach_target, reach_orientation, last_arm_action,
                                     pelvis_pos, pelvis_quat)
            raw_action = session.run(None, {"observation": obs.reshape(1, -1)})[0][0]
            last_arm_target = apply_action(raw_action, last_arm_target)
            last_arm_action = raw_action.copy()
            data.ctrl[arm_acts] = last_arm_target

        mujoco.mj_step(model, data)

        if log is not None:
            log.append({"tick": tick, "t": tick * model.opt.timestep,
                         "pos": data.site_xpos[site_id].copy()})

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(viz_sleep)

        dist = float(np.linalg.norm(target_world_pos - data.site_xpos[site_id]))
        if dist < tol:
            reached = True
            break

    return reached, tick + 1, dist