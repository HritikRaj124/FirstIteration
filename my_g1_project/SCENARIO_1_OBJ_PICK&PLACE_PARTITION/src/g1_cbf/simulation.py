"""Pick-and-place rollout with an optional CBF safety filter."""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from .barrier import make_segment_h_fn
from .config import LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE, RunConfig
from .kinematics import solve_ik_jax
from .model_utils import (
    build_actuator_map,
    get_qpos_indices,
    minjerk,
    name2id,
    point_box_margin,
)
from .safety_filter import solve_safety_qp

log = logging.getLogger(__name__)

LogEntry = dict


def update_grasp(data, palm_pos, obj_qadr, obj_dadr, active, grasped, object_z, grasp_radius):
    """Attach the object to the palm once it is within grasp_radius (kinematic grasp)."""
    if not active:
        return False
    obj = data.qpos[obj_qadr : obj_qadr + 3]
    if not grasped and float(np.linalg.norm(palm_pos - obj)) < grasp_radius:
        grasped = True
    if grasped:
        held = palm_pos.copy()
        held[2] = max(held[2], object_z)
        data.qpos[obj_qadr : obj_qadr + 3] = held
        data.qpos[obj_qadr + 3 : obj_qadr + 7] = [1, 0, 0, 0]
        data.qvel[obj_dadr : obj_dadr + 6] = 0.0
    return grasped


def run_scenario(cfg: RunConfig) -> list[LogEntry]:
    """Run one full pick-and-place rollout and return the per-tick log.

    cfg.safety.enabled switches the CBF filter on or off.
    """
    scene, task, safety = cfg.scene, cfg.task, cfg.safety
    scene_path = str(scene.scene_path)

    log.info(
        "Scenario: filter %s  alpha=%s  qdot_max=%s",
        "ON" if safety.enabled else "OFF",
        safety.alpha,
        safety.qdot_max,
    )
    log.info("JAX devices: %s", jax.devices())

    model = mujoco.MjModel.from_xml_path(scene_path)

    # Disable arm-vs-partition collision so the unfiltered baseline can
    # genuinely penetrate it (intentional, to demonstrate the violation).
    partition_geom = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, scene.partition_geom)
    model.geom_contype[partition_geom] = 0
    model.geom_conaffinity[partition_geom] = 0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pelvis = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    log.info("Pelvis DOF count: %d (0 = welded)", model.body_dofnum[pelvis])

    # kinematics-only model for IK / CBF FK, no contacts
    ik_model = mujoco.MjModel.from_xml_path(scene_path)
    ik_model.geom_contype[:] = 0
    ik_model.geom_conaffinity[:] = 0
    mjx_model = mjx.put_model(ik_model)

    site_id = name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)
    qpos_ids = get_qpos_indices(model, LEFT_ARM_JOINT_NAMES)

    actuators_map = build_actuator_map(model)
    arm_joint_ids = [name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_ARM_JOINT_NAMES]
    arm_actuators = np.array([actuators_map[j] for j in arm_joint_ids])

    servo = model.actuator_biastype[arm_actuators[0]] == mujoco.mjtBias.mjBIAS_AFFINE
    log.info("Actuator mode: %s", "position servo" if servo else "torque")
    if not servo:
        raise RuntimeError("Torque actuators need the PD path -- not enabled in this build")

    arm_set = set(arm_actuators.tolist())
    hold = [
        (aid, float(data.qpos[model.jnt_qposadr[jid]]))
        for jid, aid in actuators_map.items()
        if aid not in arm_set
    ]

    p_center = data.geom_xpos[partition_geom].copy()
    p_half = model.geom_size[partition_geom].copy()

    table_top = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, scene.table_geom)
    t_center = data.geom_xpos[table_top].copy()
    t_half = model.geom_size[table_top].copy()

    object_pos = data.xpos[name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.object_body)].copy()
    place_pos = task.place_array

    obj_joint_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.object_joint)
    obj_qadr = model.jnt_qposadr[obj_joint_id]
    obj_dadr = model.jnt_dofadr[obj_joint_id]
    object_z = object_pos[2]

    log.info(
        "partition x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]",
        p_center[0] - p_half[0],
        p_center[0] + p_half[0],
        p_center[1] - p_half[1],
        p_center[1] + p_half[1],
        p_center[2] - p_half[2],
        p_center[2] + p_half[2],
    )
    log.info("object %s  place %s", np.round(object_pos, 3), np.round(place_pos, 3))

    # Barrier functions are built once from the live obstacle geometry.
    base_qpos = data.qpos.copy()  # pelvis is welded, so this stays valid throughout
    h_partition = make_segment_h_fn(
        model, mjx_model, qpos_ids, base_qpos, p_center, p_half, safety.arm_radius, segment="forearm"
    )
    h_table = make_segment_h_fn(
        model, mjx_model, qpos_ids, base_qpos, t_center, t_half, safety.arm_radius, segment="forearm"
    )

    waypoints = [
        ("approach", object_pos + [0, 0, task.approach_h]),
        ("descend", object_pos + [0, 0, task.grasp_h]),
        ("transit_start", object_pos + [0, 0, task.transit_h]),
        ("transit_end", place_pos + [0, 0, task.transit_h]),
        ("place", place_pos + [0, 0, task.grasp_h]),
        ("retreat", place_pos + [0, 0, task.retreat_h]),
    ]

    log.info("Solving inverse kinematics")
    t0 = time.time()
    plan = []
    q_seed = data.qpos.copy()
    for phase, tgt in waypoints:
        q, ok, iters, resid = solve_ik_jax(
            model,
            mjx_model,
            q_seed,
            np.array(tgt),
            joint_names=list(LEFT_ARM_JOINT_NAMES),
            site_name=LEFT_PALM_SITE,
            max_iters=cfg.ik.max_iters,
            tol=cfg.ik.tol,
            damping=cfg.ik.damping,
            max_step=cfg.ik.max_step,
        )
        (log.info if ok else log.warning)(
            "  %-14s %-16s %3d iters  residual %.4f m",
            phase,
            "ok" if ok else "did not converge",
            iters,
            resid,
        )
        plan.append((phase, np.array(q)))
        q_seed = q_seed.copy()
        q_seed[qpos_ids] = q
    log.info("IK done in %.1f s", time.time() - t0)

    n_sub = max(1, int(round(task.control_dt / model.opt.timestep)))
    n_steps = int(task.phase_duration / task.control_dt)

    trace: list[LogEntry] = []
    t = 0.0
    q_start = data.qpos[qpos_ids].copy()
    grasped = False
    descend_dist: list[float] = []

    viewer = mujoco.viewer.launch_passive(model, data) if cfg.show_viewer else None
    try:
        for phase, q_end in plan:
            for i in range(n_steps):
                q_ref = q_start + minjerk((i + 1) / n_steps) * (q_end - q_start)

                q_current = data.qpos[qpos_ids].copy()
                arm_q = jnp.array(q_current)

                h_val_partition = float(h_partition(arm_q))
                h_val_table = float(h_table(arm_q))

                q_command = q_ref  # default: unfiltered
                if safety.enabled:
                    qdot_des = (q_ref - q_current) / task.control_dt
                    grad_p = np.array(jax.grad(h_partition)(arm_q))
                    grad_t = np.array(jax.grad(h_table)(arm_q))
                    qdot_safe = solve_safety_qp(
                        qdot_des,
                        [h_val_partition, h_val_table],
                        [grad_p, grad_t],
                        safety.alpha,
                        safety.qdot_max,
                    )
                    q_command = q_current + qdot_safe * task.control_dt

                for _ in range(n_sub):
                    data.ctrl[arm_actuators] = q_command
                    for aid, q0 in hold:
                        data.ctrl[aid] = q0
                    mujoco.mj_step(model, data)

                    grasped = update_grasp(
                        data,
                        data.site_xpos[site_id].copy(),
                        obj_qadr,
                        obj_dadr,
                        phase in task.grasp_phases,
                        grasped,
                        object_z,
                        task.grasp_radius,
                    )
                    if phase == "descend":
                        descend_dist.append(
                            float(
                                np.linalg.norm(data.site_xpos[site_id] - data.qpos[obj_qadr : obj_qadr + 3])
                            )
                        )

                p = data.site_xpos[site_id].copy()
                trace.append(
                    {
                        "t": t,
                        "phase": phase,
                        "pos": p,
                        "margin": point_box_margin(p, p_center, p_half, safety.arm_radius),
                        "h_forearm": h_val_partition,
                        "h_forearm_table": h_val_table,
                        "err": float(np.linalg.norm(data.qpos[qpos_ids] - q_ref)),
                    }
                )
                t += task.control_dt

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    if cfg.realtime:
                        time.sleep(task.control_dt)

            q_start = data.qpos[qpos_ids].copy()
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if viewer is not None:
            viewer.close()

    _report(trace, safety.enabled, descend_dist, task.grasp_radius)
    return trace


def summarize(trace: list[LogEntry]) -> dict:
    """Scalar metrics for a rollout (also used by tests and CI)."""
    if not trace:
        return {}
    m = [e["margin"] for e in trace]
    hp = [e["h_forearm"] for e in trace]
    ht = [e["h_forearm_table"] for e in trace]
    return {
        "ticks": len(trace),
        "violations_palm": sum(x < 0 for x in m),
        "violations_partition": sum(x < 0 for x in hp),
        "violations_table": sum(x < 0 for x in ht),
        "min_margin": min(m),
        "min_h_partition": min(hp),
        "min_h_table": min(ht),
        "max_tracking_error": max(e["err"] for e in trace),
    }


def _report(trace, filtered, descend_dist, grasp_radius) -> None:
    if not trace:
        log.warning("No data logged")
        return
    s = summarize(trace)
    log.info("--- results (%s) ---", "filtered" if filtered else "baseline")
    log.info(
        "ticks %d   violating (palm margin) %d   min margin %+.4f m",
        s["ticks"],
        s["violations_palm"],
        s["min_margin"],
    )
    log.info(
        "violating (forearm h, partition) %d   min %+.4f m", s["violations_partition"], s["min_h_partition"]
    )
    log.info("violating (forearm h, table)     %d   min %+.4f m", s["violations_table"], s["min_h_table"])
    log.info("max tracking error %.4f rad", s["max_tracking_error"])
    if descend_dist:
        log.info(
            "descend: min dist %.4f  max dist %.4f  grasp radius %s",
            min(descend_dist),
            max(descend_dist),
            grasp_radius,
        )

    per: dict[str, list[LogEntry]] = {}
    for e in trace:
        per.setdefault(e["phase"], []).append(e)
    for k, es in per.items():
        worst = min(es, key=lambda e: e["margin"])
        log.info("  %-14s min %+.4f at palm %s", k, worst["margin"], np.round(worst["pos"], 3))
