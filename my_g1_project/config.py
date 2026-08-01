import mujoco
from pathlib import Path
import time
import mujoco.viewer
from dataclasses import dataclass
from dataclasses import asdict
import json
from mujoco._enums import mjtObj

@dataclass
class Summary:
    joint_name: str
    joint_id: int
    actuator_id: int
    angle_before: float
    angle_after: float

g1_dir = Path(r"C:\Users\HRITIK~1\AppData\Local\Temp\my_g1_no_gravity_h7p621ol")

model = mujoco.MjModel.from_xml_path(str(g1_dir / "g1.xml"))
print("njnt:", model.njnt)
print("nu:", model.nu)
data = mujoco.MjData(model)

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(i, name)

right_arm_joint = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

right_arm_joint_id = []
qpos_addr = []
actuator_id = []

for name in right_arm_joint:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    right_arm_joint_id.append(jid)
    qpos_addr.append(model.jnt_qposadr[jid])
    actuator_id.append(aid)

print(right_arm_joint)
print(right_arm_joint_id)
print(qpos_addr)

control_inp = [5,5,7,8,5,6,9]

for aid, value in zip(actuator_id, control_inp):
    data.ctrl[aid] = value

current_angle = [data.qpos[addr] for addr in qpos_addr]

log = []

with mujoco.viewer.launch_passive(model,data) as viewer:
    for step in range(2000000):
        mujoco.mj_step(model,data)
        current_angle = [data.qpos[addr] for addr in qpos_addr]
        log.append(current_angle)
        viewer.sync()
        time.sleep(0.002)

summaries = []
for name, jid, aid, before, after in zip(
    right_arm_joint, right_arm_joint_id, actuator_id, log[0], log[-1]):
    s = Summary(joint_name=name, joint_id=int(jid), actuator_id=int(aid), angle_before=float(before), angle_after=float(after))
    summaries.append(s)

payload = [asdict(s) for s in summaries]

with open("summary.json", "w") as f:
    json.dump(payload, f, indent=2)