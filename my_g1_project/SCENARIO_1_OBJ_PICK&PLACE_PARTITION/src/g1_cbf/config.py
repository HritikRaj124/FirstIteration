"""Typed configuration for the pick-and-place + CBF scenario."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
LEFT_PALM_SITE = "left_palm"
LEFT_SHOULDER_BODY = "left_shoulder_pitch_link"
LEFT_ELBOW_BODY = "left_elbow_link"


@dataclass(frozen=True)
class SceneConfig:
    scene_path: Path = Path("scenarios/partition_task/scene.xml")
    object_body: str = "movable_object"
    object_joint: str = "movable_object_joint"
    partition_geom: str = "partition_wall"
    table_geom: str = "table_top"


@dataclass(frozen=True)
class TaskConfig:
    place_pos: tuple[float, float, float] = (0.3, -0.05, 0.80)
    # Waypoint heights above the object / place position [m]
    approach_h: float = 0.10
    grasp_h: float = 0.03
    transit_h: float = 0.05
    retreat_h: float = 0.15
    phase_duration: float = 2.5  # [s] per waypoint
    control_dt: float = 0.01  # [s]
    grasp_radius: float = 0.08  # [m] palm-object distance that triggers a grasp
    grasp_phases: frozenset[str] = frozenset({"descend", "transit_start", "transit_end", "place", "retreat"})

    @property
    def place_array(self) -> np.ndarray:
        return np.asarray(self.place_pos, dtype=float)


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    alpha: float = 3.5  # CBF class-K gain, alpha(h) = alpha * h
    qdot_max: float = 2.0  # joint velocity limit [rad/s]
    arm_radius: float = 0.03  # capsule radius around the arm [m]


@dataclass(frozen=True)
class IKConfig:
    max_iters: int = 100
    tol: float = 1e-3
    damping: float = 3e-3
    max_step: float = 0.2


@dataclass(frozen=True)
class RunConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    show_viewer: bool = False
    realtime: bool = True
    use_gpu: bool = False
    out_dir: Path = Path("outputs/filter_comparison")

    def with_filter(self, enabled: bool) -> RunConfig:
        return replace(self, safety=replace(self.safety, enabled=enabled))
