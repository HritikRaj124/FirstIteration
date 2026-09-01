"""Bimanual tabletop scenarios for the G1 MuJoCo standard full-body G1 sim.

This module owns the fixed-base dual-arm tabletop manipulation setup used by the
G1 MuJoCo standard nominal scenarios. The original scenario is a deliberately more complex
cross-swap: two objects rest on a workspace shelf in front of the robot, the left
object must go to the right drop spot, and the right object to the left drop
spot, so the arms have to cross. A simpler sequential baseline is also provided:
left hand pick-place first, then right hand pick-place, with each object staying
on its own side of the shelf.

Why a dedicated module (not the vendor reacher path):
- the pretrained ONNX policies only include a *right-arm* reacher, so a faithful
  bimanual task cannot be driven by them. Both arms are instead driven by
  kinematic damped-least-squares (DLS) inverse kinematics on the full-body model
  while the walker policy keeps the lower body balanced.
- grasping uses a deterministic kinematic "magnetic" attach (the held object
  tracks the palm) instead of frictional finger grasping, so the demonstration is
  reproducible and focuses on the crossing arm motion. The attach is gated on the
  palm actually reaching the object, so objects are never teleported.
- the pelvis is pinned to its standing pose and the lower body is held at the
  default posture (fixed-base tabletop manipulation) so reaching both arms forward
  cannot tip the robot, and the arm position actuators are stiffened so the palms
  track their Cartesian waypoints instead of drooping under gravity.
- a small temporary workspace shelf with two free cylinders and four visual
  markers is injected into the vendor scene so both sources and both swapped
  targets are simultaneously within arm reach (no locomotion needed).

The external vendor scene/meshes are not modified; a temporary XML is compiled
beside the vendor ``g1.xml`` with a relative include so ``meshdir`` resolves.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .sim import FullBodySim

# Right/left arm joint names, in the model's joint order within each arm.
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_JOINTS = tuple(n.replace("right_", "left_") for n in RIGHT_ARM_JOINTS)

# --- Workspace geometry (world frame; robot pelvis ~(-0.6, 0, 0.76), faces +x).
# The shelf is intentionally low (top ~0.80, just below the pelvis) so the
# objects sit inside the arms' comfortable reach envelope. Measured reach shows
# that reaching down to ~0.78 hits arm joint limits, while ~0.99 (an earlier
# draft) sat well above the shoulders. Source and swapped-target lanes are both
# inside the reachable band at x=-0.32.
SHELF_TOP_Z = 0.80
OBJ_HALF_H = 0.04
OBJ_REST_Z = SHELF_TOP_Z + OBJ_HALF_H            # 0.84
WORKSPACE_X = -0.32                              # objects ~0.28 m in front
SRC_Y = 0.18                                     # source lane lateral offset
TGT_Y = 0.06                                     # swapped-target lateral offset
# (each hand can only *place* ~0.05 m past body centre at shelf height before the
#  elbow meets the torso; the cross itself sweeps higher and further, but the
#  drop-off lane is kept inside the reachable band so both placements are clean.)

# Object start positions (sources).
LEFT_OBJ_START = np.array([WORKSPACE_X, +SRC_Y, OBJ_REST_Z])
RIGHT_OBJ_START = np.array([WORKSPACE_X, -SRC_Y, OBJ_REST_Z])

# Swapped drop targets: left object -> right spot, right object -> left spot.
LEFT_OBJ_TARGET = np.array([WORKSPACE_X, -TGT_Y, OBJ_REST_Z])   # crosses to -y
RIGHT_OBJ_TARGET = np.array([WORKSPACE_X, +TGT_Y, OBJ_REST_Z])  # crosses to +y

# Same-side drop targets for the simpler sequential baseline.
LEFT_SEQ_TARGET = np.array([WORKSPACE_X, +TGT_Y, OBJ_REST_Z])
RIGHT_SEQ_TARGET = np.array([WORKSPACE_X, -TGT_Y, OBJ_REST_Z])

# Vertical schedule: hover above, descend to grasp, then cross HIGH. A G1 arm
# cannot bring its hand deep across to the opposite side at chest height -- the
# elbow/shoulder jam into the torso (measured: the right hand stalls at y~+0.02
# at z=0.84). Lifting the carry to shoulder/chin height lets the arm extend over
# the torso (measured: the right hand reaches y=+0.10 at z>=0.95). The two arms
# also cross at clearly different heights so their wrists never collide.
APPROACH_Z = OBJ_REST_Z + 0.06                   # 0.90 hover before descent
CARRY_Z_RIGHT = 0.97                             # right arm crosses high
CARRY_Z_LEFT = 1.09                              # left arm crosses higher (clears right)
RETRACT_Z = OBJ_REST_Z + 0.08                    # 0.92 lift-off after release
SEQUENTIAL_CARRY_Z = 0.97                        # no crossing; one comfortable carry height

# Depth (x) at the crossing point. Reaching slightly closer to the body
# (x ~ -0.28) lets the high arm sweep further across; both arms use the same
# depth and rely on the height stagger to avoid each other.
CROSS_X_LEFT = WORKSPACE_X + 0.04                # -0.28
CROSS_X_RIGHT = WORKSPACE_X + 0.04               # -0.28


def _scene_injection() -> str:
    """XML fragment: a static shelf, two free cylinders, and four visual markers."""

    return f"""
    <!-- G1 MuJoCo standard bimanual workspace (temporary; not part of the vendor scene).
         The shelf and the two free objects live on collision group 2 only, so
         they rest on the shelf but pass through the robot. The kinematic
         magnetic grasp (not contact) attaches an object to a palm, so the arm
         must never knock the light cylinders off the shelf. -->
    <geom name="work_shelf" type="box" pos="{WORKSPACE_X} 0 {SHELF_TOP_Z - 0.02}"
          size="0.12 0.28 0.02"
          rgba="0.45 0.30 0.18 1" contype="2" conaffinity="2"/>

    <body name="bi_left_obj" pos="{LEFT_OBJ_START[0]} {LEFT_OBJ_START[1]} {LEFT_OBJ_START[2]}">
      <freejoint name="bi_left_obj_joint"/>
      <geom name="bi_left_obj_geom" type="cylinder" size="0.025 {OBJ_HALF_H}"
            rgba="0.15 0.65 0.20 1" density="200"
            contype="2" conaffinity="2" friction="2 0.1 0.01"/>
    </body>
    <body name="bi_right_obj" pos="{RIGHT_OBJ_START[0]} {RIGHT_OBJ_START[1]} {RIGHT_OBJ_START[2]}">
      <freejoint name="bi_right_obj_joint"/>
      <geom name="bi_right_obj_geom" type="cylinder" size="0.025 {OBJ_HALF_H}"
            rgba="0.20 0.30 0.80 1" density="200"
            contype="2" conaffinity="2" friction="2 0.1 0.01"/>
    </body>

    <!-- Swapped target markers (visual only) -->
    <geom name="bi_target_for_left" type="cylinder"
          pos="{LEFT_OBJ_TARGET[0]} {LEFT_OBJ_TARGET[1]} {SHELF_TOP_Z + 0.002}"
          size="0.03 0.002" rgba="0.15 0.65 0.20 0.4"
          contype="0" conaffinity="0"/>
    <geom name="bi_target_for_right" type="cylinder"
          pos="{RIGHT_OBJ_TARGET[0]} {RIGHT_OBJ_TARGET[1]} {SHELF_TOP_Z + 0.002}"
          size="0.03 0.002" rgba="0.20 0.30 0.80 0.4"
          contype="0" conaffinity="0"/>

    <!-- Same-side target markers for the sequential baseline (visual only). -->
    <geom name="bi_seq_target_for_left" type="cylinder"
          pos="{LEFT_SEQ_TARGET[0]} {LEFT_SEQ_TARGET[1]} {SHELF_TOP_Z + 0.006}"
          size="0.022 0.002" rgba="0.15 0.65 0.20 0.75"
          contype="0" conaffinity="0"/>
    <geom name="bi_seq_target_for_right" type="cylinder"
          pos="{RIGHT_SEQ_TARGET[0]} {RIGHT_SEQ_TARGET[1]} {SHELF_TOP_Z + 0.006}"
          size="0.022 0.002" rgba="0.20 0.30 0.80 0.75"
          contype="0" conaffinity="0"/>
"""


def build_bimanual_scene_xml() -> Path:
    """Compose a temp scene XML: vendor scene + temporary bimanual workspace.

    The vendor ``red_block`` body is removed and the workspace fragment is
    injected before the fixed cameras. The temp file is written under the system
    temp directory with an absolute include to the vendor ``g1.xml`` so external
    asset checkouts can remain read-only. The caller should delete it after the
    model is compiled (see :func:`cleanup_scene_xml`).
    """

    base = config.resolve_g1_challenge_dir()
    scene_path = base / "scene.xml"
    if not scene_path.is_file():
        raise FileNotFoundError(config.missing_assets_message())

    text = scene_path.read_text()

    # Drop the vendor red block body (leave its comments; harmless). Anchor on
    # the body element itself so the earlier "Red block material" comment in the
    # <asset> section is never matched.
    text = re.sub(
        r'\s*<body name="red_block".*?</body>',
        "", text, flags=re.DOTALL,
    )

    # Inject the workspace just before the fixed cameras.
    marker = "<!-- Fixed cameras -->"
    text = text.replace(marker, _scene_injection() + "\n    " + marker, 1)

    temp_dir = Path(tempfile.mkdtemp(prefix="g1_mujoco_standard_bimanual_scene_"))
    (temp_dir / "g1.xml").symlink_to(base / "g1.xml")
    (temp_dir / "assets").symlink_to(base / "assets", target_is_directory=True)
    scene_out = temp_dir / "scene.xml"
    scene_out.write_text(text, encoding="utf-8")
    return scene_out


def cleanup_scene_xml(path: Path) -> None:
    """Remove a temporary scene file produced by :func:`build_bimanual_scene_xml`."""

    p = Path(path)
    if p.parent.name.startswith("g1_mujoco_standard_bimanual_scene_"):
        shutil.rmtree(p.parent, ignore_errors=True)
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


@dataclass
class ArmIndex:
    """Resolved per-arm joint addressing for IK and target scatter."""

    joint_indices: list[int]   # index into the 29-joint controller ordering
    qpos_adr: list[int]        # qpos address of each arm joint
    dof_adr: list[int]         # dof address of each arm joint
    site_id: int               # palm site id


def _resolve_arm(sim: FullBodySim, arm_joints: tuple[str, ...], palm_site: str) -> ArmIndex:
    mj = sim.mujoco
    model = sim.model
    joint_indices, qpos_adr, dof_adr = [], [], []
    for name in arm_joints:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint not found: {name}")
        joint_indices.append(sim.joint_names.index(name))
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, palm_site)
    if site_id < 0:
        raise KeyError(f"site not found: {palm_site}")
    return ArmIndex(joint_indices, qpos_adr, dof_adr, site_id)


@dataclass
class BimanualController:
    """Fixed-base dual-arm DLS-IK overlay with proximity-gated magnetic grasping.

    For this nominal tabletop scenario the pelvis is pinned to its standing pose
    every substep (fixed-base manipulation) and the non-arm joints are held at the
    default standing posture, so balance is a non-issue and reaching both arms
    forward cannot tip the robot. The two arms are driven by accumulated
    damped-least-squares inverse kinematics; the arm position actuators are
    stiffened so the palms track the Cartesian waypoints closely instead of
    drooping under gravity. Grasping attaches a free object to the palm only once
    the palm is genuinely within ``grasp_radius`` of it (no teleport), and the
    held object then rides exactly with the palm until release.
    """

    sim: FullBodySim
    stiffen_factor: float = 8.0
    ik_damping: float = 0.08
    max_joint_step: float = 0.03
    max_cart_err: float = 0.05
    palm_speed: float = 0.012      # palm waypoint approach speed, m per control tick
    grasp_radius: float = 0.08

    left: ArmIndex = field(init=False)
    right: ArmIndex = field(init=False)
    left_obj_qadr: int = field(init=False)
    right_obj_qadr: int = field(init=False)
    left_obj_dadr: int = field(init=False)
    right_obj_dadr: int = field(init=False)

    def __post_init__(self) -> None:
        self.left = _resolve_arm(self.sim, LEFT_ARM_JOINTS, "left_palm")
        self.right = _resolve_arm(self.sim, RIGHT_ARM_JOINTS, "right_palm")
        self.left_obj_qadr, self.left_obj_dadr = self._free_obj("bi_left_obj_joint")
        self.right_obj_qadr, self.right_obj_dadr = self._free_obj("bi_right_obj_joint")
        self._jacp = np.zeros((3, self.sim.model.nv))

        self._stiffen_arms()
        self._lo_left, self._hi_left = self._joint_ranges(LEFT_ARM_JOINTS)
        self._lo_right, self._hi_right = self._joint_ranges(RIGHT_ARM_JOINTS)

        data = self.sim.data
        self.default_pose = np.asarray(self.sim.controller.default_joint_pos, float).copy()
        self.base_qpos = data.qpos[0:7].copy()
        self.cmd_left = data.qpos[self.left.qpos_adr].copy()
        self.cmd_right = data.qpos[self.right.qpos_adr].copy()
        self.cur_left = self.palm_pos(self.left)
        self.cur_right = self.palm_pos(self.right)
        self.grasped_left = False
        self.grasped_right = False

    def settle(self, n_ticks: int = 25) -> None:
        """Hold the arms at their rest pose for a few ticks to damp startup jerk.

        The freshly built model carries small residual joint velocities and the
        stiffened arm actuators react sharply on the first step; settling with the
        palms held in place and the base pinned removes the initial transient
        before the scenario waypoints start moving.
        """

        sim = self.sim
        mj = sim.mujoco
        ctrl = sim.controller
        sim.data.qvel[:] = 0.0
        mj.mj_forward(sim.model, sim.data)
        target_pos = self.default_pose.copy()
        for i, idx in enumerate(self.left.joint_indices):
            target_pos[idx] = self.cmd_left[i]
        for i, idx in enumerate(self.right.joint_indices):
            target_pos[idx] = self.cmd_right[i]
        for _ in range(n_ticks):
            for _ in range(sim.decimation):
                ctrl.apply_pd_control(target_pos)
                mj.mj_step(sim.model, sim.data)
                sim.data.qpos[0:7] = self.base_qpos
                sim.data.qvel[0:6] = 0.0
                mj.mj_forward(sim.model, sim.data)
        # Re-sync command/cursor to the settled pose.
        self.cmd_left = sim.data.qpos[self.left.qpos_adr].copy()
        self.cmd_right = sim.data.qpos[self.right.qpos_adr].copy()
        self.cur_left = self.palm_pos(self.left)
        self.cur_right = self.palm_pos(self.right)

    # -- setup helpers -------------------------------------------------------- #
    def _free_obj(self, joint_name: str) -> tuple[int, int]:
        mj = self.sim.mujoco
        jid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise KeyError(f"free joint not found: {joint_name}")
        return int(self.sim.model.jnt_qposadr[jid]), int(self.sim.model.jnt_dofadr[jid])

    def _joint_ranges(self, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        mj = self.sim.mujoco
        model = self.sim.model
        lo, hi = [], []
        for n in names:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)
            lo.append(float(model.jnt_range[jid][0]))
            hi.append(float(model.jnt_range[jid][1]))
        return np.array(lo), np.array(hi)

    def _stiffen_arms(self) -> None:
        """Scale up the arm position-actuator gains so the palms track IK targets.

        The vendor arm actuators are tuned for locomotion and droop badly when the
        arm is extended; without this the palm lags ~0.1 m behind the command and
        grasping looks like a teleport. Only the 14 arm actuators are touched.
        """

        mj = self.sim.mujoco
        model = self.sim.model
        f = self.stiffen_factor
        for name in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS):
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                continue
            model.actuator_gainprm[aid][0] *= f
            model.actuator_biasprm[aid][1] *= f
            model.actuator_biasprm[aid][2] *= np.sqrt(f)

    # -- kinematics ----------------------------------------------------------- #
    def palm_pos(self, arm: ArmIndex) -> np.ndarray:
        return self.sim.data.site_xpos[arm.site_id].copy()

    def object_pos(self, which: str) -> np.ndarray:
        qadr = self.left_obj_qadr if which == "left" else self.right_obj_qadr
        return self.sim.data.qpos[qadr:qadr + 3].copy()

    @staticmethod
    def _advance(cur: np.ndarray, target: np.ndarray, speed: float) -> np.ndarray:
        d = target - cur
        n = float(np.linalg.norm(d))
        if n <= speed or n < 1e-9:
            return target.copy()
        return cur + d * (speed / n)

    def _ik_command(self, arm: ArmIndex, cmd: np.ndarray, lo: np.ndarray,
                    hi: np.ndarray, cursor: np.ndarray, iters: int = 8) -> np.ndarray:
        """Converge the *commanded* joint angles so the palm reaches ``cursor``.

        Inverse kinematics is solved on a kinematic shadow of the model: the arm
        joints are temporarily set to the running command and forward kinematics
        (``mj_kinematics`` + ``mj_comPos``, no constraint solve) is evaluated to
        read the commanded palm and Jacobian. This decouples the joint command
        from the dynamic lag of the stiff actuators, so the command is a smooth
        kinematic trajectory the PD controller can track without oscillating
        (feeding the *actual* lagging palm error back into the command made the
        arms jitter). The model state is restored before returning.
        """

        mj = self.sim.mujoco
        model = self.sim.model
        data = self.sim.data

        saved = data.qpos[arm.qpos_adr].copy()
        cmd = cmd.copy()
        data.qpos[arm.qpos_adr] = cmd
        lam2 = self.ik_damping ** 2
        for _ in range(iters):
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_jacSite(model, data, self._jacp, None, arm.site_id)
            j_arm = self._jacp[:, arm.dof_adr]                  # 3 x 7
            err = cursor - data.site_xpos[arm.site_id]
            n = float(np.linalg.norm(err))
            if n < 1e-4:
                break
            if n > self.max_cart_err:
                err = err * (self.max_cart_err / n)
            dq = j_arm.T @ np.linalg.solve(j_arm @ j_arm.T + lam2 * np.eye(3), err)
            dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
            cmd = np.clip(cmd + dq, lo, hi)
            data.qpos[arm.qpos_adr] = cmd
        data.qpos[arm.qpos_adr] = saved
        return cmd

    # -- one control tick ----------------------------------------------------- #
    def control_tick(self, phase: "BiPhase") -> None:
        """Advance the palms toward the phase waypoints and step the dynamics."""

        sim = self.sim
        mj = sim.mujoco
        ctrl = sim.controller

        self.cur_left = self._advance(self.cur_left, phase.left_palm, self.palm_speed)
        self.cur_right = self._advance(self.cur_right, phase.right_palm, self.palm_speed)

        # Smooth kinematic joint command (decoupled from dynamic lag).
        self.cmd_left = self._ik_command(
            self.left, self.cmd_left, self._lo_left, self._hi_left, self.cur_left)
        self.cmd_right = self._ik_command(
            self.right, self.cmd_right, self._lo_right, self._hi_right, self.cur_right)
        # Restore kinematics to the actual configuration before stepping.
        mj.mj_kinematics(sim.model, sim.data)
        mj.mj_comPos(sim.model, sim.data)

        target_pos = self.default_pose.copy()
        for i, idx in enumerate(self.left.joint_indices):
            target_pos[idx] = self.cmd_left[i]
        for i, idx in enumerate(self.right.joint_indices):
            target_pos[idx] = self.cmd_right[i]

        for _ in range(sim.decimation):
            ctrl.apply_pd_control(target_pos)
            mj.mj_step(sim.model, sim.data)
            # Pin the base to the standing pose (fixed-base manipulation).
            sim.data.qpos[0:7] = self.base_qpos
            sim.data.qvel[0:6] = 0.0
            self._update_grasp(phase)
            mj.mj_forward(sim.model, sim.data)

    def _update_grasp(self, phase: "BiPhase") -> None:
        """Latch/unlatch each magnetic grasp and carry held objects with the palm.

        The carried object tracks the palm but its height is clamped to at least
        its shelf-resting height, so a slightly drooping arm at full extension can
        never press the object down into the shelf (which would store contact
        energy and eject the object on release).
        """

        data = self.sim.data
        # Left.
        if phase.grasp_left:
            palm = self.palm_pos(self.left)
            if not self.grasped_left:
                obj = data.qpos[self.left_obj_qadr:self.left_obj_qadr + 3]
                if float(np.linalg.norm(palm - obj)) < self.grasp_radius:
                    self.grasped_left = True
            if self.grasped_left:
                palm[2] = max(palm[2], OBJ_REST_Z)
                data.qpos[self.left_obj_qadr:self.left_obj_qadr + 3] = palm
                data.qpos[self.left_obj_qadr + 3:self.left_obj_qadr + 7] = [1, 0, 0, 0]
                data.qvel[self.left_obj_dadr:self.left_obj_dadr + 6] = 0.0
        else:
            self.grasped_left = False
        # Right.
        if phase.grasp_right:
            palm = self.palm_pos(self.right)
            if not self.grasped_right:
                obj = data.qpos[self.right_obj_qadr:self.right_obj_qadr + 3]
                if float(np.linalg.norm(palm - obj)) < self.grasp_radius:
                    self.grasped_right = True
            if self.grasped_right:
                palm[2] = max(palm[2], OBJ_REST_Z)
                data.qpos[self.right_obj_qadr:self.right_obj_qadr + 3] = palm
                data.qpos[self.right_obj_qadr + 3:self.right_obj_qadr + 7] = [1, 0, 0, 0]
                data.qvel[self.right_obj_dadr:self.right_obj_dadr + 6] = 0.0
        else:
            self.grasped_right = False


@dataclass(frozen=True)
class BiPhase:
    name: str
    duration_s: float
    left_palm: np.ndarray
    right_palm: np.ndarray
    grasp_left: bool
    grasp_right: bool


def _palm(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z])


def cross_swap_phases() -> tuple[BiPhase, ...]:
    """Deterministic schedule producing the staggered-height crossing motion."""

    ls, rs = LEFT_OBJ_START, RIGHT_OBJ_START
    lt, rt = LEFT_OBJ_TARGET, RIGHT_OBJ_TARGET
    return (
        # Hover above each source.
        BiPhase("ready", 1.0,
                _palm(ls[0], ls[1], APPROACH_Z), _palm(rs[0], rs[1], APPROACH_Z),
                False, False),
        # Descend onto the objects.
        BiPhase("descend", 1.0, ls.copy(), rs.copy(), False, False),
        # Close the magnetic grasp.
        BiPhase("grasp", 0.6, ls.copy(), rs.copy(), True, True),
        # Lift to staggered carry heights, splaying to staggered depths.
        BiPhase("lift", 1.0,
                _palm(CROSS_X_LEFT, ls[1], CARRY_Z_LEFT),
                _palm(CROSS_X_RIGHT, rs[1], CARRY_Z_RIGHT),
                True, True),
        # Cross: arms pass at different heights and different depths.
        BiPhase("cross", 2.4,
                _palm(CROSS_X_LEFT, lt[1], CARRY_Z_LEFT),
                _palm(CROSS_X_RIGHT, rt[1], CARRY_Z_RIGHT),
                True, True),
        # Lower onto the swapped targets (dwell lets the arms settle on target).
        BiPhase("lower", 1.6, lt.copy(), rt.copy(), True, True),
        # Release.
        BiPhase("release", 0.6, lt.copy(), rt.copy(), False, False),
        # Retract upward, clear of the objects.
        BiPhase("retract", 1.0,
                _palm(lt[0], lt[1], RETRACT_Z), _palm(rt[0], rt[1], RETRACT_Z),
                False, False),
    )


def _single_arm_pickplace_phases(
    arm: str,
    pick: np.ndarray,
    place: np.ndarray,
    park_other: np.ndarray,
) -> tuple[BiPhase, ...]:
    """One active arm moves one object while the other arm holds its park pose."""

    if arm not in {"left", "right"}:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

    pick = np.asarray(pick, dtype=float)
    place = np.asarray(place, dtype=float)
    park = np.asarray(park_other, dtype=float)
    hover = _palm(pick[0], pick[1], APPROACH_Z)
    lift = _palm(pick[0], pick[1], SEQUENTIAL_CARRY_Z)
    carry = _palm(place[0], place[1], SEQUENTIAL_CARRY_Z)
    retract = _palm(place[0], place[1], RETRACT_Z)
    steps = (
        ("ready", 1.0, hover, False),
        ("descend", 1.0, pick, False),
        ("grasp", 0.6, pick, True),
        ("lift", 1.0, lift, True),
        ("carry", 1.6, carry, True),
        ("lower", 1.2, place, True),
        ("release", 0.6, place, False),
        ("retract", 1.0, retract, False),
    )

    phases = []
    for name, dur, palm, grasp in steps:
        if arm == "left":
            phases.append(BiPhase(f"L_{name}", dur, palm, park.copy(), grasp, False))
        else:
            phases.append(BiPhase(f"R_{name}", dur, park.copy(), palm, False, grasp))
    return tuple(phases)


def sequential_pickplace_phases(
    park_left: np.ndarray,
    park_right: np.ndarray,
) -> tuple[BiPhase, ...]:
    """Left-hand pick-place, then right-hand pick-place, never simultaneously.

    The active hand closes the temporary-scene magnetic grasp during ``grasp``,
    ``lift``, ``carry`` and ``lower``. The object is released only after the hand
    reaches the same-side target lane.
    """

    left_block = _single_arm_pickplace_phases(
        "left", LEFT_OBJ_START, LEFT_SEQ_TARGET, park_right)
    right_block = _single_arm_pickplace_phases(
        "right", RIGHT_OBJ_START, RIGHT_SEQ_TARGET, park_left)
    return left_block + right_block


def phase_at(phases: tuple[BiPhase, ...], t: float) -> BiPhase:
    acc = 0.0
    for ph in phases:
        acc += ph.duration_s
        if t < acc:
            return ph
    return phases[-1]


def total_duration(phases: tuple[BiPhase, ...]) -> float:
    return float(sum(p.duration_s for p in phases))


def evaluate_success(ctrl: BimanualController) -> dict:
    """Both objects ended near their swapped targets and stayed up on the shelf."""

    left_final = ctrl.object_pos("left")
    right_final = ctrl.object_pos("right")
    left_err = float(np.linalg.norm(left_final[:2] - LEFT_OBJ_TARGET[:2]))
    right_err = float(np.linalg.norm(right_final[:2] - RIGHT_OBJ_TARGET[:2]))
    on_shelf = bool(left_final[2] > SHELF_TOP_Z - 0.03 and right_final[2] > SHELF_TOP_Z - 0.03)
    placed = bool(left_err < 0.12 and right_err < 0.12 and on_shelf)
    return {
        "success": placed,
        "left_obj_final": left_final.tolist(),
        "right_obj_final": right_final.tolist(),
        "left_target": LEFT_OBJ_TARGET.tolist(),
        "right_target": RIGHT_OBJ_TARGET.tolist(),
        "left_placement_error_xy": left_err,
        "right_placement_error_xy": right_err,
        "objects_on_shelf": on_shelf,
    }


def evaluate_sequential_success(ctrl: BimanualController) -> dict:
    """Both objects ended near same-side sequential targets and stayed on shelf."""

    left_final = ctrl.object_pos("left")
    right_final = ctrl.object_pos("right")
    left_err = float(np.linalg.norm(left_final[:2] - LEFT_SEQ_TARGET[:2]))
    right_err = float(np.linalg.norm(right_final[:2] - RIGHT_SEQ_TARGET[:2]))
    on_shelf = bool(left_final[2] > SHELF_TOP_Z - 0.03 and right_final[2] > SHELF_TOP_Z - 0.03)
    placed = bool(left_err < 0.12 and right_err < 0.12 and on_shelf)
    return {
        "success": placed,
        "left_obj_final": left_final.tolist(),
        "right_obj_final": right_final.tolist(),
        "left_target": LEFT_SEQ_TARGET.tolist(),
        "right_target": RIGHT_SEQ_TARGET.tolist(),
        "left_placement_error_xy": left_err,
        "right_placement_error_xy": right_err,
        "objects_on_shelf": on_shelf,
    }
