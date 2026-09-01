"""Numerical inverse kinematics for the G1 right arm (position-servo targets).

Damped least-squares (Levenberg-Marquardt style) Jacobian IK. Solves for a
joint-angle target q_des such that the end-effector site reaches a desired
pose (position, and optionally orientation). Only moves the joints you pass
in via `joint_names` -- everything else (legs, torso, other arm) is left
untouched, matching the fixed-legs scenario setup.

Usage pattern:
    1. Run `list_candidate_arm_joints(model)` once to find your right-arm
       joint names, then fill in RIGHT_ARM_JOINT_NAMES below.
    2. Call `solve_ik(...)` per waypoint (grasp pose, transit pose, place
       pose, etc.) to get q_des for that phase.
    3. Feed q_des to your position actuators / phase controller.

This function is side-effect-free on `data` -- it snapshots qpos before
iterating and restores everything outside the solved joints when done.
"""

from __future__ import annotations

import numpy as np
import mujoco

RIGHT_PALM_SITE = "right_palm"


# ---------------------------------------------------------------------------
# TODO: fill this in after running list_candidate_arm_joints() on your model.
# Order matters -- it should match the kinematic chain from shoulder to wrist.
# ---------------------------------------------------------------------------
RIGHT_ARM_JOINT_NAMES: list[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def list_candidate_arm_joints(model: mujoco.MjModel, substrings=("right", "arm", "shoulder", "elbow", "wrist")) -> list[str]:
    """Print + return joint names that look like they belong to the right arm.

    Run this once interactively against your loaded model to figure out the
    exact names/order for RIGHT_ARM_JOINT_NAMES.
    """
    names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and any(s in name.lower() for s in substrings):
            names.append(name)
    print("Candidate joints found (verify order matches the kinematic chain):")
    for n in names:
        print(" ", n)
    return names


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    site_name: str = RIGHT_PALM_SITE,
    joint_names: list[str] | None = None,
    max_iters: int = 150,
    tol: float = 1e-4,
    damping: float = 1e-4,
    max_step: float = 0.2,
) -> tuple[np.ndarray, bool, int]:
    """Solve for joint angles that bring `site_name` to `target_pos`/`target_quat`.

    Args:
        target_pos: (3,) desired world-frame position of the site.
        target_quat: (4,) desired world-frame orientation (w,x,y,z), or None
            to solve position-only (orientation left free).
        joint_names: ordered list of joint names forming the arm chain.
            Defaults to RIGHT_ARM_JOINT_NAMES.
        max_iters: solver iteration cap.
        tol: stop when ||error|| (position [m] stacked with orientation
            [rad]) falls below this.
        damping: Levenberg-Marquardt damping factor (higher = more stable,
            less accurate near singularities).
        max_step: per-iteration cap on ||dq|| (rad) to prevent instability
            on large initial errors.

    Returns:
        (q_des, converged, n_iters) where q_des has the same length/order as
        joint_names.
    """
    joint_names = joint_names or RIGHT_ARM_JOINT_NAMES
    if not joint_names:
        raise ValueError(
            "No joint_names provided. Run list_candidate_arm_joints(model) "
            "and fill in RIGHT_ARM_JOINT_NAMES first."
        )

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")

    jnt_ids = []
    for n in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise ValueError(f"Unknown joint: {n}")
        jnt_ids.append(jid)

    dof_ids = np.array([model.jnt_dofadr[j] for j in jnt_ids])
    qpos_ids = np.array([model.jnt_qposadr[j] for j in jnt_ids])

    # Snapshot so this function has no side effects beyond the returned array.
    qpos_snapshot = data.qpos.copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    converged = False
    it = 0
    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        site_pos = data.site_xpos[site_id].copy()
        pos_err = target_pos - site_pos

        if target_quat is not None:
            site_mat = data.site_xmat[site_id].reshape(3, 3).copy()
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

            neg_site_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_site_quat, site_quat)

            quat_err = np.zeros(4)
            mujoco.mju_mulQuat(quat_err, target_quat, neg_site_quat)

            # Rotation-vector approximation of the quaternion error.
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)

            err = np.concatenate([pos_err, rot_err])
        else:
            err = pos_err

        if np.linalg.norm(err) < tol:
            converged = True
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        if target_quat is not None:
            J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])  # 6 x n_arm_dof
        else:
            J = jacp[:, dof_ids]  # 3 x n_arm_dof

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * err
        n_task = J.shape[0]
        JJt = J @ J.T
        lam2 = (damping**2) * np.eye(n_task)
        dq = J.T @ np.linalg.solve(JJt + lam2, err)

        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq *= max_step / step_norm

        data.qpos[qpos_ids] += dq

        # Respect joint limits.
        for k, j in enumerate(jnt_ids):
            lo, hi = model.jnt_range[j]
            if hi > lo:  # (0,0) means unlimited in MuJoCo convention
                data.qpos[qpos_ids[k]] = np.clip(data.qpos[qpos_ids[k]], lo, hi)

    q_des = data.qpos[qpos_ids].copy()

    # Restore original state -- this function only *computes*, doesn't mutate.
    data.qpos[:] = qpos_snapshot
    mujoco.mj_forward(model, data)

    return q_des, converged, it + 1


if __name__ == "__main__":
    # Minimal smoke-test scaffold. Point this at your scenario scene and a
    # real target to sanity-check convergence before wiring into the phase
    # controller.
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python ik_solver.py <path_to_scene.xml>")
        sys.exit(1)

    scene_path = Path(sys.argv[1])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not RIGHT_ARM_JOINT_NAMES:
        list_candidate_arm_joints(model)
        print("\nFill in RIGHT_ARM_JOINT_NAMES at the top of this file, then re-run.")
        sys.exit(0)

    # Example: solve toward the movable_object position, offset up a bit
    # for an approach pose. Replace with your real grasp target.
    obj_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "movable_object")]
    target = obj_pos + np.array([0.0, 0.0, 0.10])

    q_des, ok, n_iters = solve_ik(model, data, target_pos=target)
    print(f"converged={ok} in {n_iters} iters")
    print("q_des:", q_des)