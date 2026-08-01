import mujoco
import mujoco.viewer
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BallState:
    step: int
    H: float

xml = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="ball" pos="0 0 0">
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
      <freejoint/>
    </body>
    <body name="obstacle" pos="3 0 0">
      <geom type="cylinder" size="0.1 1.5" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

nominal_vel= 1.5
obs_x = 3
safe_radi= 0.3
alpha = 0.5

state = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(5000):
        viewer.cam.lookat[:] = [1.5, 0, 0]
        viewer.cam.distance = 6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        mujoco.mj_step(model, data)

        ball_x = data.qpos[0]
        h = np.linalg.norm(ball_x - obs_x) - safe_radi
        v_max = alpha*h
        vx_used = min(nominal_vel, v_max)
        data.qvel[0] = vx_used
        time.sleep(0.001)

        s = BallState(step = step, H= h)
        state.append(s)

        viewer.sync()


all_steps = [s.step for s in state]
all_h = [s.H for s in state]
print(min(all_h))

plt.plot(all_steps, all_h)
plt.axhline(y=0, color='red')
plt.show()
