import os

USE_GPU = False
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
if not USE_GPU:
    os.environ["JAX_PLATFORMS"] = 'cpu'
else:
    os.environ.pop("JAX_PLATFORMS", None)

import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjx

from ik_solver import solve_ik_jax, get_qpos_indices, LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE

SCENE_PATH = "scenarios/partition_task/scene.xml"
OBJECT_BODY = "movable_object"
PARTITION_GEOM = "partition_wall"

TARGET_PLACE_POS = np.array([0.3, -0.05, 0.80])

APPROACH_H, GRASP_H, TRANSIT_H, RETREAT_H = 0.10, 0.03, 0.05, 0.15

PHASE_DURATION = 2.5
CONTROL_DT = 0.01
ARM_SAFETY_RADIUS = 0.03
GRASP_RADIUS = 0.08
REALTIME = True
SHOW_VIEWER = True

GRASP_PHASES = {"descend", "transit_start", "transit_end", "place"}


def get_dof_indices(model, joint_names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    return np.array([model.jnt_dofadr[j] for j in ids])


def build_actuator_map(model):
    return {int(model.actuator_trnid[a, 0]): a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# Signed Distance Fucntion for the point to obs distance
def point_box_margin(p, center, half, r=0.0):
    diff = np.abs(p - center) - half - r
    if np.all(diff < 0):
        return float(np.max(diff))
    return float(np.linalg.norm(np.maximum(diff, 0)))


def update_grasp(data, palm_pos, obj_qadr, obj_dadr, active, grasped, obj_rest_z):
    if not active:
        return False
    obj = data.qpos[obj_qadr:obj_qadr + 3]
    if not grasped and float(np.linalg.norm(palm_pos - obj)) < GRASP_RADIUS:
        grasped = True
    if grasped:
        held = palm_pos.copy()
        held[2] = max(held[2], obj_rest_z)
        data.qpos[obj_qadr:obj_qadr + 3] = held
        data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
        data.qvel[obj_dadr:obj_dadr + 6] = 0.0
    return grasped


def main():
    import jax
    print("JAX devices:", jax.devices())

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)

    # Disabling the collision of the arm in the partition and letting it penetrate it
    pg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PARTITION_GEOM)
    model.geom_contype[pg] = 0
    model.geom_conaffinity[pg] = 0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"pelvis dofnum = {model.body_dofnum[pelvis]} (0 = welded)")

    ik_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    ik_model.geom_contype[:] = 0
    ik_model.geom_conaffinity[:] = 0
    mjx_model = mjx.put_model(ik_model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)
    if site_id < 0:
        raise ValueError(f"Site: '{LEFT_PALM_SITE}' missing")

    qpos_ids = get_qpos_indices(model, LEFT_ARM_JOINT_NAMES)
    dof_ids = get_dof_indices(model, LEFT_ARM_JOINT_NAMES)

    amap = build_actuator_map(model)
    arm_acts = np.array([amap[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEFT_ARM_JOINT_NAMES])

    servo = model.actuator_biastype[arm_acts[0]] == mujoco.mjtBias.mjBIAS_AFFINE
    print(f"Actuator Mode: {'Position servo' if servo else 'Torque'}")
    if not servo:
        raise RuntimeError("Torque actuators need the PD path -- non enabled in this build")

    arm_set = set(arm_acts.tolist())
    hold = [(aid, float(data.qpos[model.jnt_qposadr[jid]])) for jid, aid in amap.items() if aid not in arm_set]

    p_center = data.geom_xpos[pg].copy()
    p_half = model.geom_size[pg].copy()

    obj_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)].copy()
    place_pos = TARGET_PLACE_POS.copy()

    obj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "movable_object_joint")
    if obj_joint_id < 0:
        raise ValueError("Unknown joint: movable_object_joint")
    obj_qadr = model.jnt_qposadr[obj_joint_id]
    obj_dadr = model.jnt_dofadr[obj_joint_id]
    obj_rest_z = obj_pos[2]

    print("\n--- geometry ---")
    print(f"partition x[{p_center[0]-p_half[0]:.3f},{p_center[0]+p_half[0]:.3f}] "
          f"y[{p_center[1]-p_half[1]:.3f},{p_center[1]+p_half[1]:.3f}] "
          f"z[{p_center[2]-p_half[2]:.3f},{p_center[2]+p_half[2]:.3f}]")
    print(f"object {np.round(obj_pos, 3)}   place {np.round(place_pos, 3)}")

    waypoints = [
        ("approach",      obj_pos + [0, 0, APPROACH_H]),
        ("descend",       obj_pos + [0, 0, GRASP_H]),
        ("transit_start", obj_pos + [0, 0, TRANSIT_H]),
        ("transit_end",   place_pos + [0, 0, TRANSIT_H]),
        ("place",         place_pos + [0, 0, GRASP_H]),
        ("retreat",       place_pos + [0, 0, RETREAT_H]),
    ]

    print("\n --- Solving Inverse Kinematics ---")
    t0 = time.time()
    plan = []
    q_seed = data.qpos.copy()

    for phase, tgt in waypoints:
        q, ok, iters, resid = solve_ik_jax(model, mjx_model, q_seed, np.array(tgt),
                                           joint_names=LEFT_ARM_JOINT_NAMES, site_name=LEFT_PALM_SITE,
                                           max_iters=100)

        flag = "ok" if ok else "Did not Converge"
        print(f" {phase:14s} {flag:14s} {iters:3d} iters residual {resid:.4f} m")
        plan.append((phase, np.array(q)))
        q_seed = q_seed.copy()
        q_seed[qpos_ids] = q

    print(f"Inverse Kinematics done in {time.time() - t0:.1f} seconds")

    n_sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    n_steps = int(PHASE_DURATION / CONTROL_DT)

    log = []
    t = 0.0
    q_start = data.qpos[qpos_ids].copy()
    grasped = False
    dists_during_descend = []

    viewer = mujoco.viewer.launch_passive(model, data) if SHOW_VIEWER else None
    try:
        for phase, q_end in plan:
            for i in range(n_steps):
                q_ref = q_start + minjerk((i+1) / n_steps) * (q_end - q_start)

                for _ in range(n_sub):
                    data.ctrl[arm_acts] = q_ref
                    for aid, q0 in hold:
                        data.ctrl[aid] = q0
                    mujoco.mj_step(model, data)

                    grasped = update_grasp(data, data.site_xpos[site_id].copy(),
                                            obj_qadr, obj_dadr,
                                            phase in GRASP_PHASES, grasped, obj_rest_z)
                    if phase == "descend":
                        dist = float(np.linalg.norm(data.site_xpos[site_id] - data.qpos[obj_qadr:obj_qadr+3]))
                        dists_during_descend.append(dist)

                p = data.site_xpos[site_id].copy()
                log.append({"t": t, "phase": phase, "pos": p,
                            "margin": point_box_margin(p, p_center, p_half, ARM_SAFETY_RADIUS),
                            "err": float(np.linalg.norm(data.qpos[qpos_ids] - q_ref))})
                t += CONTROL_DT

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    if REALTIME:
                        time.sleep(CONTROL_DT)

            q_start = data.qpos[qpos_ids].copy()
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if viewer is not None:
            viewer.close()

    if not log:
        print("No data logged")
        return

    m = [e["margin"] for e in log]

    print(f"\n--- results ---")
    print(f"ticks {len(log)}   violating {sum(x < 0 for x in m)}")

    print(f"min margin {min(m):+.4f} m")
    print(f"max tracking error {max(e['err'] for e in log):.4f} rad")

    if dists_during_descend:
        print(f"descend: min dist = {min(dists_during_descend):.4f}   "
              f"max dist = {max(dists_during_descend):.4f}   "
              f"grasp radius = {GRASP_RADIUS}")

    per = {}
    for e in log:
        per.setdefault(e["phase"], []).append(e)

    print("\nby phase:")
    for k, es in per.items():
        worst = min(es, key=lambda e: e["margin"])
        print(f"  {k:14s} min {worst['margin']:+.4f} at palm {np.round(worst['pos'], 3)}")


if __name__ == "__main__":
    main()