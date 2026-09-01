import time
import mujoco
import mujoco.viewer
import numpy as np
from reacher_policy import load_reacher_policy, run_reacher_to_target

SCENE_PATH = "scenarios/partition_task/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
model.opt.timestep = 0.005
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

session = load_reacher_policy("right_reacher.onnx")

obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "movable_object")
target = np.array([0.3, 0.15, 0.80])
print("target:", target)

with mujoco.viewer.launch_passive(model, data) as viewer:
    reached, ticks, dist = run_reacher_to_target(
        model, data, session, target, viewer=viewer, viz_sleep=0.08
    )
    print(f"reached={reached} ticks={ticks} final_dist={dist:.4f}")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.02)
