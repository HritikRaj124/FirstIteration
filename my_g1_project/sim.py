"""Full-body G1 MuJoCo simulation factory and helpers for G1 MuJoCo standard.

This module builds the full-body Unitree G1 simulation by reusing the external
vendor module (``ONNXPolicy``, ``G1Controller``, ``CameraRenderer``,
``set_armature``) and replicating its headless setup. It exposes a small, explicit
API so smoke scripts do not duplicate simulation plumbing:

- :func:`build_sim` constructs model, data, controller, and policies.
- :func:`reset_sim` restores the canonical spawn pose.
- :func:`control_step` advances one control tick (decimated physics).
- :class:`OffscreenRecorder` renders named cameras and writes MP4/GIF.

The pretrained ONNX policies act as the nominal locomotion + balance layer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config


@dataclass
class FullBodySim:
    """Loaded full-body G1 simulation bundle."""

    mujoco: Any
    model: Any
    data: Any
    controller: Any
    vendor: Any
    model_config: dict
    joint_names: list[str]
    timestep: float
    decimation: int
    target_pos: np.ndarray = field(default=None, repr=False)

    # --- scene queries -------------------------------------------------- #
    def body_pos(self, name: str) -> np.ndarray:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body not found: {name}")
        return self.data.xpos[bid].copy()

    def site_pos(self, name: str) -> np.ndarray:
        sid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"site not found: {name}")
        return self.data.site_xpos[sid].copy()

    def camera_names(self) -> list[str]:
        names = []
        for i in range(self.model.ncam):
            nm = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, i)
            if nm:
                names.append(nm)
        return names

    def palm_pos(self) -> np.ndarray:
        return self.site_pos(config.RIGHT_PALM_SITE)


def _spawn_pose(sim: FullBodySim) -> None:
    """Place the robot behind the source table in its default joint posture."""

    data = sim.data
    cfg = sim.model_config
    data.qpos[0] = -0.6   # x: stand back from the table
    data.qpos[2] = 0.76   # z: pelvis height
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, value in cfg["default_joint_pos"].items():
        if name in sim.joint_names:
            data.qpos[7 + sim.joint_names.index(name)] = value


def build_sim(
    timestep: float = config.TRAIN_TIMESTEP,
    decimation: int = config.CONTROL_DECIMATION,
    load_reacher: bool = True,
    scene_xml: "Path | None" = None,
) -> FullBodySim:
    """Construct the full-body G1 simulation (headless).

    Mirrors the vendor ``main()`` setup but without any viewer or OpenCV windows.
    Raises a clear error when MuJoCo or the external assets are unavailable.

    ``scene_xml`` overrides the vendor scene (e.g. a temporary scene that adds a
    bimanual workspace); it must still include the vendor ``g1.xml``.
    """

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco  # local import keeps non-MuJoCo tooling able to import this file

    if not config.assets_available():
        raise FileNotFoundError(config.missing_assets_message())

    vendor = config.load_vendor_module()
    base = config.resolve_g1_challenge_dir()

    with open(config.vendor_model_config()) as f:
        model_config = json.load(f)
    joint_names = list(model_config["joint_names"])

    scene_path = Path(scene_xml) if scene_xml is not None else config.vendor_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = timestep
    vendor.set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Policies: walker keeps the legs/waist stable; the right reacher overlays the
    # right arm when reach mode is active. Croucher/rotator are loaded for parity.
    walker = vendor.ONNXPolicy(str(base / "walker.onnx"))
    croucher = vendor.ONNXPolicy(str(base / "croucher.onnx"))
    rotator = vendor.ONNXPolicy(str(base / "rotator.onnx"))
    right_reacher = None
    rr_path = base / "right_reacher.onnx"
    if load_reacher and rr_path.exists():
        right_reacher = vendor.ONNXPolicy(str(rr_path))

    controller = vendor.G1Controller(
        model, data, walker, croucher, rotator, model_config,
        right_reacher=right_reacher,
    )

    sim = FullBodySim(
        mujoco=mujoco,
        model=model,
        data=data,
        controller=controller,
        vendor=vendor,
        model_config=model_config,
        joint_names=joint_names,
        timestep=timestep,
        decimation=decimation,
    )
    reset_sim(sim)
    _warmup_policies(sim)
    return sim


def _warmup_policies(sim: FullBodySim) -> None:
    """Trigger ONNX session JIT compilation with dummy observations."""

    ctrl = sim.controller
    ctrl.walker_policy(np.zeros((1, 99), dtype=np.float32))
    ctrl.croucher_policy(np.zeros((1, 101), dtype=np.float32))
    ctrl.rotator_policy(np.zeros((1, 99), dtype=np.float32))
    if ctrl.right_reacher_policy is not None:
        ctrl.right_reacher_policy(np.zeros((1, 36), dtype=np.float32))


def reset_sim(sim: FullBodySim) -> None:
    """Reset physics and controller state to the canonical spawn."""

    mujoco = sim.mujoco
    ctrl = sim.controller
    mujoco.mj_resetData(sim.model, sim.data)
    _spawn_pose(sim)
    mujoco.mj_forward(sim.model, sim.data)

    ctrl.last_action[:] = 0
    ctrl.last_arm_action[:] = 0
    ctrl.lin_vel_x = ctrl.lin_vel_y = ctrl.ang_vel_z = 0.0
    ctrl.reach_active = False
    ctrl.last_arm_target = None
    ctrl.frozen_arm_pos = None
    ctrl.grip_closed = False
    ctrl.input_mode = "walk"
    sim.target_pos = ctrl.default_joint_pos.copy()


def control_step(sim: FullBodySim) -> None:
    """Advance one control tick: one policy update plus ``decimation`` physics steps."""

    mujoco = sim.mujoco
    ctrl = sim.controller
    for k in range(sim.decimation):
        if k == 0:
            sim.target_pos = ctrl.step()
        ctrl.apply_pd_control(sim.target_pos)
        mujoco.mj_step(sim.model, sim.data)


def control_dt(sim: FullBodySim) -> float:
    """Wall-clock duration of one control tick."""

    return sim.timestep * sim.decimation


class OffscreenRecorder:
    """Render named cameras offscreen and write MP4/GIF videos.

    One :class:`mujoco.Renderer` is created per requested resolution. Frames are
    captured on demand and flushed to disk with imageio.
    """

    def __init__(self, sim: FullBodySim, width: int = 640, height: int = 480):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.sim = sim
        self.width = width
        self.height = height
        # The default offscreen framebuffer is 640x480; grow it if needed so the
        # requested render size is always valid.
        glob = sim.model.vis.global_
        glob.offwidth = max(int(glob.offwidth), width)
        glob.offheight = max(int(glob.offheight), height)
        self._renderer = sim.mujoco.Renderer(sim.model, height, width)
        self._frames: dict[str, list[np.ndarray]] = {}

    def capture(self, cameras: list[str]) -> None:
        for cam in cameras:
            self._renderer.update_scene(self.sim.data, camera=cam)
            frame = self._renderer.render().copy()
            self._frames.setdefault(cam, []).append(frame)

    def n_frames(self, camera: str) -> int:
        return len(self._frames.get(camera, []))

    def save(self, out_dir: Path, fps: int = 20, prefix: str = "") -> dict[str, Path]:
        """Write one video per captured camera. Returns {camera: path}.

        Prefers MP4 (imageio-ffmpeg); falls back to GIF when ffmpeg is missing.
        """

        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for cam, frames in self._frames.items():
            if not frames:
                continue
            stem = f"{prefix}{cam}" if prefix else cam
            mp4_path = out_dir / f"{stem}.mp4"
            try:
                imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=None)
                written[cam] = mp4_path
            except Exception:
                gif_path = out_dir / f"{stem}.gif"
                imageio.mimsave(gif_path, frames, duration=1.0 / fps)
                written[cam] = gif_path
        return written

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass
