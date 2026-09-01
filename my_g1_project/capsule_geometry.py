"""Capsule and sphere geometry for the safety filter"""

import numpy as np
import mujoco

def point_box_margin(p, center, half_size, radius=0.0):
    """Signed distance from a point to box"""
    diff = np.abs(p - center) - half_size - radius
    if np.all(diff < 0):
        return float(np.max(diff))
    return float(np.linalg.norm(np.maximum(diff, 0)))

def capsule_box_margin(p1, p2, center, half_size, radius=0.0, n_iters = 25):
    """Signed distance margin for a capsule against a box.
    This finds the closest point on the segment via a ternary search"""

    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    lo, hi = 0.0, 1.0
    for i in range(n_iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3

        d1 = point_box_margin(p1 + m1 * (p2 - p1), center, half_size, radius)
        d2 = point_box_margin(p1 + m2 * (p2 - p2), center, half_size, radius)

        if d1 < d2:
            hi = m2
        else:
            lo = m1

    s = (lo + hi) / 2
    return point_box_margin(p1 + s * (p2 - p1), center, half_size, radius)

def get_arm_capsule_endpoints(model, data, side="left"):
    shoulder_name = f"{side}_shoulder_pitch_link"
    elbow_name = f"{side}_elbow_link"
    palm_site_name = f"{side}_palm"

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name)
    if sid < 0:
        raise ValueError(f"Unknown body: {shoulder_name}")
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, elbow_name)
    if eid < 0:
        raise ValueError(f"Unknown body: {elbow_name}")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_site_name)
    if pid < 0:
        raise ValueError(f"Unknown site: {palm_site_name}")

    shoulder_pos = data.xpos[sid].copy()
    elbow_pos = data.xpos[eid].copy()
    palm_pos = data.site_xpos[pid].copy()

    return (shoulder_pos, elbow_pos), palm_pos, (elbow_pos, palm_pos)

def arm_min_margin(model, data, center, half_size, r_arm, r_hand, side="left"):
    """Overall margin for the whole arm: the worst (smallest) of the three
    pieces -- upper arm, forearm, hand."""
    upper_arm, palm_pos, forearm = get_arm_capsule_endpoints(model, data, side)

    m_upper = capsule_box_margin(*upper_arm, center, half_size, r_arm)
    m_fore = capsule_box_margin(*forearm, center, half_size, r_arm)
    m_hand = point_box_margin(palm_pos, center, half_size, r_hand)

    return min(m_upper, m_fore, m_hand)




