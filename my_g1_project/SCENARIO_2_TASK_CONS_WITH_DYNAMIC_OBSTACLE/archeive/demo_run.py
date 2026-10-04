import json
import numpy as np
import mujoco

SCENE_PATH = r"D:\FirstIteration\my_g1_project\SCENARIO_2_TASK_CONS_WITH_DYNAMIC_OBSTACLE\scene.xml"
CONFIG_PATH = "model_config.json"

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with open(CONFIG_PATH) as f:
    config = json.load(f)

joint_names = config["joint_names"]

# No floating base to skip past anymore -- no subtraction
qpos_offsets = np.array([model.joint(n).qposadr[0] for n in joint_names])
qvel_offsets = np.array([model.joint(n).dofadr[0] for n in joint_names])
actuator_ids = np.array([model.actuator(n).id for n in joint_names])

print(f"{'action_idx':>10}  {'joint_name':<28}  {'qpos_off':>8}  {'qvel_off':>8}")
for i, name in enumerate(joint_names):
    print(f"{i:10d}  {name:<28}  {qpos_offsets[i]:8d}  {qvel_offsets[i]:8d}")

pelvis_id = model.body("pelvis").id
pelvis_pos = model.body_pos[pelvis_id].copy()
pelvis_quat = model.body_quat[pelvis_id].copy()
print(f"\npelvis pos (static):  {pelvis_pos}")
print(f"pelvis quat (static): {pelvis_quat}")

site_id = model.site("right_palm").id
print(f"\nright_palm site world pos: {data.site_xpos[site_id]}")

print("\nCup world positions:")
for i in range(1, 6):
    body_id = model.body(f"cup_{i}").id
    print(f"  cup_{i}: {data.xpos[body_id]}")