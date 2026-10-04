"""
Position-target to torque conversion, per the PD law confirmed necessary
for this robot's actuators (mjGAIN_FIXED / mjBIAS_NONE -- pure torque
actuators, no implicit servo).

tau = Kp * (target - q) - Kd * qdot

Sources: Wang et al., "General Humanoid Whole-Body Control via Pretraining
and Fast Adaptation" (arXiv 2602.11929), Appendix A.3, Eq. 7-8; Shi, Lyu &
Wang, "Discovering Self-Protective Falling Policy..." (arXiv 2512.01336),
Section III-B, Eq. 2.

Per-joint Kp/Kd values match Unitree's real actuator classes (5020, 7520-14,
7520-22, 4010), as used in the LuckyRobots demo's gain table.
"""

import numpy as np


def build_pd_gains(joint_names):
    """Per-joint (Kp, Kd, effort_limit) arrays, assigned by joint name."""
    S5020, D5020, E5020 = 14.2506, 0.9072, 25.0
    S7520_14, D7520_14, E7520_14 = 40.1792, 2.5579, 88.0
    S7520_22, D7520_22, E7520_22 = 99.0984, 6.3088, 139.0
    S4010, D4010, E4010 = 16.7783, 1.0681, 5.0

    n = len(joint_names)
    kp = np.zeros(n, dtype=np.float32)
    kd = np.zeros(n, dtype=np.float32)
    effort_limit = np.zeros(n, dtype=np.float32)

    for i, name in enumerate(joint_names):
        if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
            kp[i], kd[i], effort_limit[i] = S5020, D5020, E5020
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            kp[i], kd[i], effort_limit[i] = S7520_14, D7520_14, E7520_14
        elif "hip_roll" in name or "knee" in name:
            kp[i], kd[i], effort_limit[i] = S7520_22, D7520_22, E7520_22
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            kp[i], kd[i], effort_limit[i] = S4010, D4010, E4010
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            kp[i], kd[i], effort_limit[i] = S5020 * 2, D5020 * 2, E5020 * 2
        else:
            kp[i], kd[i], effort_limit[i] = S5020, D5020, E5020

    return kp, kd, effort_limit

def set_armature(model, joint_names, qvel_offsets):
    """Sets each joint's reflected rotor inertia to match the real actuator's
    physical class. Without this, joints behave as unrealistically light and
    responsive, which destabilizes policies trained against the real inertia."""
    ARM_5020 = 0.00360972
    ARM_7520_14 = 0.01017752
    ARM_7520_22 = 0.02510192
    ARM_4010 = 0.00425000
    ARM_2x5020 = 0.00721945

    for i, name in enumerate(joint_names):
        dof = 6 + qvel_offsets[i]
        if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
            model.dof_armature[dof] = ARM_5020
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            model.dof_armature[dof] = ARM_7520_14
        elif "hip_roll" in name or "knee" in name:
            model.dof_armature[dof] = ARM_7520_22
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            model.dof_armature[dof] = ARM_4010
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            model.dof_armature[dof] = ARM_2x5020
        else:
            model.dof_armature[dof] = ARM_5020


def pd_to_torque(target_pos, q, qdot, kp, kd, effort_limit=None):
    """tau = Kp*(target - q) - Kd*qdot, optionally clipped to each joint's
    torque limit."""
    tau = kp * (target_pos - q) - kd * qdot
    if effort_limit is not None:
        tau = np.clip(tau, -effort_limit, effort_limit)
    return tau


if __name__ == "__main__":
    print("=== build_pd_gains: spot-check a few joint categories ===")
    test_names = [
        "left_elbow_joint", "left_hip_yaw_joint", "left_hip_roll_joint",
        "left_wrist_pitch_joint", "left_ankle_roll_joint", "waist_yaw_joint",
        "waist_pitch_joint",
    ]
    kp, kd, eff = build_pd_gains(test_names)
    for i, name in enumerate(test_names):
        print(f"  {name:<26} Kp={kp[i]:8.4f}  Kd={kd[i]:7.4f}  effort_limit={eff[i]:6.1f}")

    print("\n=== pd_to_torque: simple hand-checked cases ===")
    # A) pure position error, no velocity
    tau = pd_to_torque(target_pos=np.array([0.1]), q=np.array([0.0]),
                        qdot=np.array([0.0]), kp=np.array([10.0]), kd=np.array([1.0]))
    print(f"A) target=0.1, q=0, qdot=0, Kp=10, Kd=1 -> tau={tau[0]:.2f} (expect 1.00)")

    # B) pure velocity damping, no position error
    tau = pd_to_torque(target_pos=np.array([0.0]), q=np.array([0.0]),
                        qdot=np.array([2.0]), kp=np.array([10.0]), kd=np.array([1.0]))
    print(f"B) target=0, q=0, qdot=2, Kp=10, Kd=1 -> tau={tau[0]:.2f} (expect -2.00)")

    # C) large error, clipped to effort limit
    tau = pd_to_torque(target_pos=np.array([5.0]), q=np.array([0.0]),
                        qdot=np.array([0.0]), kp=np.array([100.0]), kd=np.array([1.0]),
                        effort_limit=np.array([25.0]))
    print(f"C) target=5, q=0, Kp=100, effort_limit=25 -> tau={tau[0]:.2f} (expect 25.00, clipped)")