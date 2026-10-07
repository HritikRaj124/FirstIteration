"""Small MuJoCo model-lookup helpers and geometry/trajectory primitives."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np


def name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    """Look up a named object, raising a clear error instead of returning -1."""
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise ValueError(f"Unknown {obj.name[6:].lower()}: {name!r}")
    return idx


def joint_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> list[int]:
    return [name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]


def get_qpos_indices(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    return np.array([model.jnt_qposadr[j] for j in joint_ids(model, joint_names)])


def get_dof_indices(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    return np.array([model.jnt_dofadr[j] for j in joint_ids(model, joint_names)])


def get_joint_limits(model: mujoco.MjModel, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Joint range; unlimited joints fall back to [-pi, pi]."""
    lo, hi = [], []
    for j in joint_ids(model, joint_names):
        a, b = model.jnt_range[j]
        lo.append(a if b > a else -np.pi)
        hi.append(b if b > a else np.pi)
    return np.array(lo), np.array(hi)


def build_actuator_map(model: mujoco.MjModel) -> dict[int, int]:
    """Map joint id -> actuator index for joint-transmission actuators."""
    return {
        int(model.actuator_trnid[a, 0]): a
        for a in range(model.nu)
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
    }


def minjerk(s):
    """Minimum-jerk blend, 0 -> 1 for s in [0, 1]."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def point_box_margin(p, center, half, r: float = 0.0) -> float:
    """Signed distance from point p to an axis-aligned box inflated by r.

    Negative inside the (inflated) box, positive outside.
    """
    diff = np.abs(p - center) - half - r
    if np.all(diff < 0):
        return float(np.max(diff))
    return float(np.linalg.norm(np.maximum(diff, 0)))
