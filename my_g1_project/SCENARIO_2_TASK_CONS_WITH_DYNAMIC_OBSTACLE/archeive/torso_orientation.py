"""
Torso orientation overlay for Scenario 2 (task-consistent manipulation with
a dynamic human-hand obstacle).

Given a target position (a cup or a shelf slot) and the robot's current
pelvis pose, this computes the waist_yaw joint target that would rotate the
torso to face that target -- and rate-limits how fast that target is allowed
to change, so the walker policy is never handed an instantaneous jump it
never saw during training.
"""

import numpy as np


def _quat_apply_inverse(quat, vec):
    """Rotate a world-frame vector into the pelvis (base) frame.
    quat is (w, x, y, z), matching MuJoCo's convention (data.qpos[3:7])."""
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def compute_target_waist_yaw(target_world_pos, pelvis_pos, pelvis_quat):
    """
    target_world_pos : (3,) target position in world frame (a cup or shelf slot)
    pelvis_pos        : (3,) pelvis position in world frame  (data.qpos[:3])
    pelvis_quat       : (4,) pelvis orientation, wxyz         (data.qpos[3:7])

    Returns the absolute waist_yaw target (radians) that rotates the torso
    to face the target, measured about the pelvis's local +x (forward) axis.
    """
    vec_world = np.asarray(target_world_pos) - np.asarray(pelvis_pos)
    vec_pelvis = _quat_apply_inverse(pelvis_quat, vec_world)
    return np.arctan2(vec_pelvis[1], vec_pelvis[0])


def rate_limit_yaw(current_yaw_target, desired_yaw, max_delta_per_step):
    """Move current_yaw_target toward desired_yaw, capped at max_delta_per_step
    radians, always turning the short way around the circle."""
    delta = desired_yaw - current_yaw_target
    delta = np.arctan2(np.sin(delta), np.cos(delta))  # wrap to [-pi, pi]
    delta = np.clip(delta, -max_delta_per_step, max_delta_per_step)
    return current_yaw_target + delta


class TorsoOrientationController:
    """Stateful wrapper: remembers the last commanded waist_yaw so it can
    take the next bounded step from there, tick after tick."""

    def __init__(self, max_deg_per_sec, control_dt):
        self.max_delta_per_step = np.radians(max_deg_per_sec) * control_dt
        self.current_yaw_target = 0.0

    def update(self, target_world_pos, pelvis_pos, pelvis_quat):
        desired = compute_target_waist_yaw(target_world_pos, pelvis_pos, pelvis_quat)
        self.current_yaw_target = rate_limit_yaw(
            self.current_yaw_target, desired, self.max_delta_per_step
        )
        return self.current_yaw_target


def apply_torso_overlay(target_pos, controller, target_world_pos, pelvis_pos, pelvis_quat, waist_yaw_index=12):
    """Overwrite target_pos[waist_yaw_index] in-place with the rate-limited
    waist_yaw command from the torso orientation controller."""
    target_pos[waist_yaw_index] = controller.update(target_world_pos, pelvis_pos, pelvis_quat)
    return target_pos


if __name__ == "__main__":
    pelvis_pos = np.array([0.0, 0.0, 0.75])
    pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity: pelvis facing +x

    cup_3 = np.array([0.5, 0.4, 0.76])
    shelf_slot = np.array([-0.25, -0.55, 0.9])

    print("=== compute_target_waist_yaw ===")
    yaw_cup3 = compute_target_waist_yaw(cup_3, pelvis_pos, pelvis_quat)
    yaw_shelf = compute_target_waist_yaw(shelf_slot, pelvis_pos, pelvis_quat)
    print(f"yaw to face cup_3:      {np.degrees(yaw_cup3):.1f} deg")
    print(f"yaw to face shelf slot: {np.degrees(yaw_shelf):.1f} deg")

    print("\n=== rate_limit_yaw ===")
    max_step = np.radians(5)  # 5 deg per tick, just for this test
    step = rate_limit_yaw(np.radians(0), np.radians(38.7), max_step)
    print(f"A) 0 -> 38.7 deg, one step: {np.degrees(step):.2f} deg (expect 5.00)")
    step = rate_limit_yaw(np.radians(38.7), np.radians(-114.4), max_step)
    print(f"B) 38.7 -> -114.4 deg, one step: {np.degrees(step):.2f} deg (expect 33.70)")
    step = rate_limit_yaw(np.radians(170), np.radians(-170), max_step)
    print(f"C) 170 -> -170 deg, one step: {np.degrees(step):.2f} deg (expect 175.00)")

    print("\n=== TorsoOrientationController (tick loop) ===")
    controller = TorsoOrientationController(max_deg_per_sec=60, control_dt=1 / 50)
    for tick in range(150):
        yaw = controller.update(shelf_slot, pelvis_pos, pelvis_quat)
        if tick % 15 == 0 or tick == 149:
            print(f"tick {tick:3d}: waist_yaw = {np.degrees(yaw):.2f} deg")

    print("\n=== apply_torso_overlay (scoped to one index) ===")
    controller = TorsoOrientationController(max_deg_per_sec=60, control_dt=1 / 50)
    fake_walker_output = np.arange(29, dtype=np.float32) * 0.01  # dummy, distinguishable values

    for tick in [0, 1, 50]:
        target_pos = fake_walker_output.copy()
        for _ in range(tick + 1):
            result = apply_torso_overlay(target_pos, controller, shelf_slot, pelvis_pos, pelvis_quat)
        unchanged = np.array_equal(np.delete(result, 12), np.delete(fake_walker_output, 12))
        print(f"tick {tick:3d}: waist_yaw={np.degrees(result[12]):.2f} deg | other 28 entries unchanged: {unchanged}")