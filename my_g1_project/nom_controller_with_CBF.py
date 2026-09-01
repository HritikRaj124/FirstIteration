import os

USE_GPU = False
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
if not USE_GPU:
    os.environ["JAX_PLATFORMS"] = 'cpu'
else:
    os.environ.pop("JAX_PLATFORMS", None)

# imports
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from pathlib import Path
from mujoco import mjx

# Funcs import
from ik_solver import solve_ik_jax, get_qpos_indices, LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE
from capsule_jax import make_segment_h_fn
from CBF_QP import solve_safety_qp  
from plots import plot_filter_comparison



# Variables and Constants
SCENE_PATH = "scenarios/partition_task/scene.xml"
OBJECT_BODY = "movable_object"
PARTITION_GEOM = "partition_wall"

TARGET_PLACE_POS = np.array([0.3, -0.05, 0.80]) # Placing target position

APPROACH, GRASP, TRANSIT, RETREAT = 0.10, 0.03, 0.05, 0.15 # Waypoints determining movements (used to update the z-axis of the obj_pos)

PHASE_DURATION = 2.5  # time for every phase of the movement for the waypoints
CONTROL_DT = 0.01
ARM_SAFETY_RADIUS = 0.03  # Safety net layer radius
GRASP_RADIUS = 0.08
REALTIME = True

# Grasp mechanism phase list
GRASP_PHASES = {"descend", "transit_start", "transit_end", "place", "retreat"}

# CBF filter settings
USE_SAFETY_FILTER = False
SHOW_VIEWER = False
ALPHA = 3.5       # CBF class-K gain
QDOT_MAX = 2.0      # joint velocity limit, rad/s


# DOF indices
def get_dof_indices(model, joint_names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    return np.array([model.jnt_dofadr[j] for j in ids])


# writes a directory of joint actuator transmission IDs to their corresponding actuator indices.
def build_actuator_map(model):
    return {int(model.actuator_trnid[a, 0]): a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}


# Minimum-jerk polynomial for smooth trajectory interpolation between 0 and 1.
def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# Signed Distance Function for the point-to-obstacle distance
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


def run_scenario(use_filter: bool, show_viewer: bool = True) -> list[dict]:
    """Run one full pick-place rollout, either with the CBF safety filter
    active or not, and return the per-tick log."""
    global USE_SAFETY_FILTER, SHOW_VIEWER
    USE_SAFETY_FILTER = use_filter
    SHOW_VIEWER = show_viewer

    print(f"\n=== Running scenario: filter {'ON' if use_filter else 'OFF'} ===")
    print("JAX devices:", jax.devices())
    print(f"Safety Filter: {'ON' if USE_SAFETY_FILTER else 'OFF'}  alpha={ALPHA}  qdot_max={QDOT_MAX}")

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)

    # Disable arm-vs-partition collision so the unfiltered baseline can
    # genuinely penetrate it (intentional, to demonstrate the violation).
    pg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PARTITION_GEOM)
    model.geom_contype[pg] = 0
    model.geom_conaffinity[pg] = 0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    print(f"Pelvis DOFNUM: {model.body_dofnum[pelvis]} (0 = Welded)")

    # kinematics-only model for IK / CBF FK, no contacts
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
        raise RuntimeError("Torque actuators need the PD path -- not enabled in this build")

    arm_set = set(arm_acts.tolist())
    hold = [(aid, float(data.qpos[model.jnt_qposadr[jid]])) for jid, aid in amap.items() if aid not in arm_set]

    p_center = data.geom_xpos[pg].copy()
    p_half = model.geom_size[pg].copy()

    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if tg < 0:
        raise ValueError("Unknown geom: table_top")
    t_center = data.geom_xpos[tg].copy()
    t_half = model.geom_size[tg].copy()

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

    # --- build the CBF barrier functions once, reusing the live obstacle geometry above ---
    base_qpos_for_h = data.qpos.copy()   # pelvis is welded, so this stays valid throughout
    h_forearm = make_segment_h_fn(model, mjx_model, qpos_ids, base_qpos_for_h,
                                   p_center, p_half, ARM_SAFETY_RADIUS, segment="forearm")
    h_forearm_table = make_segment_h_fn(model, mjx_model, qpos_ids, base_qpos_for_h,
                                         t_center, t_half, ARM_SAFETY_RADIUS, segment="forearm")

    waypoints = [
        ("approach",      obj_pos + [0, 0, APPROACH]),
        ("descend",       obj_pos + [0, 0, GRASP]),
        ("transit_start", obj_pos + [0, 0, TRANSIT]),
        ("transit_end",   place_pos + [0, 0, TRANSIT]),
        ("place",         place_pos + [0, 0, GRASP]),
        ("retreat",       place_pos + [0, 0, RETREAT]),
    ]

    print("\n--- Solving Inverse Kinematics ---")
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

                q_current = data.qpos[qpos_ids].copy()
                arm_q = jnp.array(q_current)

                # always compute h(q) for both obstacles -- cheap (no QP solve),
                # and needed for a fair baseline-vs-filtered comparison later
                h_val = float(h_forearm(arm_q))
                h_val_table = float(h_forearm_table(arm_q))

                q_command = q_ref  # default: unfiltered
                if USE_SAFETY_FILTER:
                    qdot_des = (q_ref - q_current) / CONTROL_DT
                    grad_h = np.array(jax.grad(h_forearm)(arm_q))
                    grad_h_table = np.array(jax.grad(h_forearm_table)(arm_q))
                    qdot_safe = solve_safety_qp(
                        qdot_des,
                        [h_val, h_val_table],
                        [grad_h, grad_h_table],
                        ALPHA, QDOT_MAX,
                    )
                    q_command = q_current + qdot_safe * CONTROL_DT

                for _ in range(n_sub):
                    data.ctrl[arm_acts] = q_command
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
                            "h_forearm": h_val,
                            "h_forearm_table": h_val_table,
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
        return log

    m = [e["margin"] for e in log]
    hvals = [e["h_forearm"] for e in log]
    hvals_table = [e["h_forearm_table"] for e in log]

    print(f"\n--- results ({'filtered' if use_filter else 'baseline'}) ---")
    print(f"ticks {len(log)}   violating (palm margin) {sum(x < 0 for x in m)}")
    print(f"min margin {min(m):+.4f} m")
    print(f"violating (forearm h, partition) {sum(x < 0 for x in hvals)}   min h_forearm {min(hvals):+.4f} m")
    print(f"violating (forearm h, table)     {sum(x < 0 for x in hvals_table)}   min h_forearm_table {min(hvals_table):+.4f} m")
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

    return log


def main():
    filter_settings = [False, True]   # the switch: baseline, then filtered
    logs = {}

    for use_filter in filter_settings:
        label = "filtered" if use_filter else "baseline"
        logs[label] = run_scenario(use_filter=use_filter, show_viewer=False)

    out_dir = Path("outputs/filter_comparison")
    paths = plot_filter_comparison(logs["baseline"], logs["filtered"], out_dir)

    print("\nwrote:")
    for name, p in paths.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    main()