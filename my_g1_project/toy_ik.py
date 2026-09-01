from ik_solver import solve_ik_jax, get_qpos_indices, LEFT_ARM_JOINT_NAMES, LEFT_PALM_SITE
import numpy as np
import mujoco
from pathlib import Path
from mujoco import mjx

SCENE_PATH = "scenarios/partition_task/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)




x = np.array((1, 2, 3))
y = np.array((4, 5, 6))

z = x-y


print(z)

