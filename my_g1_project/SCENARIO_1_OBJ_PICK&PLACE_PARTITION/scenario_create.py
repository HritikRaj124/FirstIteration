import mujoco
import mujoco.viewer
import time
import tempfile
from pathlib import Path

g1_dir = Path(r"D:\FirstIteration\mujoco_menagerie\unitree_g1")
text = (g1_dir / "scene.xml").read_text()

scene_fragment = """
    <body name="table" pos="0.5 0 0.713">
        <geom name="table_top" type="box" size="0.4 0.7 0.02" />
        <!--  Table legs  -->
        <geom name="table_leg_1" type="cylinder" size="0.025 0.345" pos="0.35 0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_2" type="cylinder" size="0.025 0.345" pos="-0.35 0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_3" type="cylinder" size="0.025 0.345" pos="0.35 -0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_4" type="cylinder" size="0.025 0.345" pos="-0.35 -0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
    </body>

    <body name="partition" pos="0.475 0.05 0.808">
      <geom name="partition_wall" type="box" size="0.35 0.01 0.075"
            rgba="0.8 0.1 0.1 1" contype="1" conaffinity="1"/>
    </body>

    <body name="movable_object" pos="0.3 0.25 0.80">
      <freejoint name="movable_object_joint"/>
      <geom name="movable_object_geom" type="cylinder" size="0.025 0.03"
            rgba="0.1 0.6 0.8 1" density="200"
            contype="0" conaffinity="1" friction="2 0.1 0.01"/>
    </body>
"""

marker = "</worldbody>"
final_text = text.replace(marker, scene_fragment + "\n  " + marker, 1)

my_scenario_dir = Path(r"D:\FirstIteration\my_g1_project\SCENARIO_1_OBJ_PICK&PLACE_PARTITION\scenarios\partition_task")

scene_out = my_scenario_dir / "scene.xml"
scene_out.write_text(final_text, encoding="utf-8")

model = mujoco.MjModel.from_xml_path(str(scene_out))
data = mujoco.MjData(model)
print("Loaded successfully. njnt:", model.njnt, "| nbody:", model.nbody)

with mujoco.viewer.launch_passive(model, data) as viewer:
    base_qpos = data.qpos[0:7].copy()
    for step in range(2000):
        data.qpos[0:7] = base_qpos
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.001)


aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_joint")
print("gaintype:", model.actuator_gaintype[aid])
print("biastype:", model.actuator_biastype[aid])