import mujoco
from pathlib import Path



DIR = r"D:\unitree_ros\robots\g1_description\g1_29dof_mode_15_with_dex1_1.urdf"

# 1. Load and compile the URDF file
model = mujoco.MjModel.from_xml_path(DIR)

# 2. Save the compiled model directly into native MJCF format
mujoco.mj_saveLastXML(r"D:\FirstIteration\mujoco_menagerie\unitree_g1\g1_29dof_mode_15_with_dex1_1.xml", model)






