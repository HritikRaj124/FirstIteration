import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjx

from ik_solver import solve_ik_jax, LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE, get_qpos_indices

SCENE_PATH = "scenarios/partition_task/scene.xml"
OBJECT_BODY = "movable_object"
PARTITION_GEOM = "partition_wall"

PLACE_OFFSET = np.array([0.0, -0.30, 0.0])   # mirror to the far side of the partition
APPROACH_H = 0.10
GRASP_H = 0.03
LIFT_H = 0.15
TRANSIT_H = 0.12   # deliberately intersects the partition -- this is what guarantees the violation
RETREAT_H = 0.15

PHASE_DURATION = 1.5   # seconds
CONTROL_DT = 0.01
ARM_SAFETY_RADIUS = 0.03   # rough allowance for the wrist/palm's physical size


def get_dof_indices(model, joint_names):
    jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    return np.array([model.jnt_dofadr[j] for j in jnt_ids])


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def point_box_margin(p, center, half_size, safety_radius=0.0):
    """Signed distance from point p to a box. Negative = inside (violation)."""
    diff = np.abs(p - center) - half_size - safety_radius
    if np.all(diff < 0):
        return float(np.max(diff))       # inside: how deep (negative)
    return float(np.linalg.norm(np.maximum(diff, 0)))   # outside: distance (positive)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Separate model just for MJX/IK: collisions off, since MJX can't handle
    # this scene's cylinder-vs-mesh pairs. The real `model`/`data` above keep
    # collisions ON for actual simulation and violation logging.
    ik_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    ik_model.geom_contype[:] = 0
    ik_model.geom_conaffinity[:] = 0
    mjx_model = mjx.put_model(ik_model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)
    qpos_ids = get_qpos_indices(model, LEFT_ARM_JOINT_NAMES)
    dof_ids = get_dof_indices(model, LEFT_ARM_JOINT_NAMES)

    partition_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PARTITION_GEOM)
    partition_center = data.geom_xpos[partition_geom_id].copy()
    partition_half_size = model.geom_size[partition_geom_id].copy()

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    obj_pos = data.xpos[obj_id].copy()
    place_pos = obj_pos + PLACE_OFFSET

    waypoints = [
        ("approach",      obj_pos + [0, 0, APPROACH_H]),
        ("grasp",         obj_pos + [0, 0, GRASP_H]),
        ("lift",          obj_pos + [0, 0, LIFT_H]),
        ("transit_start", obj_pos + [0, 0, TRANSIT_H]),
        ("transit_end",   place_pos + [0, 0, TRANSIT_H]),
        ("place",         place_pos + [0, 0, GRASP_H]),
        ("retreat",       place_pos + [0, 0, RETREAT_H]),
    ]

    log = []
    t = 0.0
    current_q = data.qpos[qpos_ids].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for phase_name, target_pos in waypoints:
            q_des, ok, n_iters = solve_ik_jax(model, mjx_model, data.qpos.copy(), np.array(target_pos))
            if not ok:
                print(f"WARNING: IK did not converge for phase '{phase_name}' ({n_iters} iters)")

            q_start = current_q.copy()
            q_end = np.array(q_des)
            n_steps = int(PHASE_DURATION / CONTROL_DT)

            for i in range(n_steps):
                s = minjerk((i + 1) / n_steps)
                q_interp = q_start + s * (q_end - q_start)

                data.qpos[qpos_ids] = q_interp
                data.qvel[dof_ids] = 0.0
                mujoco.mj_step(model, data)

                site_pos = data.site_xpos[site_id].copy()
                margin = point_box_margin(site_pos, partition_center, partition_half_size, ARM_SAFETY_RADIUS)
                log.append({"t": t, "phase": phase_name, "margin": margin})
                t += CONTROL_DT

                if viewer.is_running():
                    viewer.sync()
                time.sleep(CONTROL_DT)

            current_q = q_end

    margins = [e["margin"] for e in log]
    violations = [e for e in log if e["margin"] < 0]
    print(f"\nTotal ticks: {len(log)}")
    print(f"Violating ticks: {len(violations)}")
    print(f"Minimum margin: {min(margins):.4f}  (negative = penetration)")


if __name__ == "__main__":
    main()