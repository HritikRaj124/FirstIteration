import mujoco
from pathlib import Path
import time
import mujoco.viewer
from dataclasses import dataclass
from dataclasses import asdict
import json
import numpy as np

g1_dir = Path(r"D:\FirstIteration\mujoco_menagerie\unitree_g1")

model = mujoco.MjModel.from_xml_path(str(g1_dir / "scene.xml"))
data = mujoco.MjData(model)

right_arm_joint = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

right_arm_joint_id, qpos_addr, actuator_id= [], [], []

for name in right_arm_joint:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    right_arm_joint_id.append(jid)
    qpos_addr.append(model.jnt_qposadr[jid])
    actuator_id.append(aid)

hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
hazard_point = np.array([0.12, -0.10, 0.83])
safe_raduis= 0.15
alpha = 3.0

joint_target = np.array([0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3])
joint_step_size = 0.01

def nominal_controller(current_cmd, target, step_size):
    delta = target - current_cmd
    dist = np.linalg.norm(delta)

    if dist <= step_size:
        return target.copy()
    return current_cmd + delta * (step_size / dist)

cmd = np.array([data.ctrl[aid] for aid in actuator_id])

@dataclass
class Tick:
    step: int
    margin: float
    filter_active: bool

log = []
FILTER_ON = False


palm_positions = []

base_qpos = data.qpos[0:7].copy()

with mujoco.viewer.launch_passive(model,data) as viewer:
    for step in range(2000):

        cmd = nominal_controller(cmd, joint_target, joint_step_size)

        palm_pos = data.xpos[hand_body_id].copy()
        palm_positions.append(palm_pos.copy())
        dist = np.linalg.norm(palm_pos - hazard_point)
        margin = dist - safe_raduis

        filter_active = False

        if FILTER_ON and margin < 0.05:
            scale = min(1.0, max(0.1, alpha*margin))
            filter_active = True

        else:
            scale = 1.0

        for i, aid in enumerate(actuator_id):
            previous = data.ctrl[aid]
            data.ctrl[aid] = previous + (cmd[i] - previous) * scale

        if step == 0 or step == 1999:
            current_angles = [data.qpos[addr] for addr in qpos_addr]
            print("step", step, "joint angles:", current_angles)

        if step == 0 or step == 1999:
            print("step", step, "cmd:", cmd)
        mujoco.mj_step(model,data)

        data.qpos[0:7] = base_qpos
        data.qvel[0:6] = 0.0
        mujoco.mj_forward(model, data)

        log.append(Tick(step=step, margin=float(margin), filter_active=filter_active))
        viewer.sync()
        time.sleep(0.001)


# after the loop:
print("Palm start:", palm_positions[0])
print("Palm end:", palm_positions[-1])

margins = [t.margin for t in log]
summary = {"filter_on": FILTER_ON, "min_margin": min(margins), "final_margin": margins[-1]}
with open("summary_filter_on.json" if FILTER_ON else "summary_filter_off.json", "w") as f:
    json.dump(summary, f, indent=2)

print("min margin:", min(margins))