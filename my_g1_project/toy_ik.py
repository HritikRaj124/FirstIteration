import numpy as np
import mujoco
from pathlib import Path
from mujoco import mjx
import jax
import jax.numpy as jnp
from time import time
from mujoco import viewer


SCENE_PATH = "scenarios/partition_task/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

model.geom_contype[:] = 0
model.geom_conaffinity[:] = 0

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.0000005)




