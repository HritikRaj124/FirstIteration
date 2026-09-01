import mujoco
import numpy as np
import time
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0  1 0 0" size="0.05"/>
      <body name="link2" pos="1 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0  0.8 0 0" size="0.05"/>
        <body name="hand" pos="0.8 0 0">
          <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

target = np.array([1.2, 0.9, 0.0])   # note: 3D now, since MuJoCo always works in 3D

hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
print("Starting hand position:", data.xpos[hand_body_id])


for i in range(10):
    mujoco.mj_forward(model, data)
    hand_pos = data.xpos[hand_body_id].copy()
    error = target - hand_pos
    print(f"iter {i}: hand at {hand_pos}, error norm = {np.linalg.norm(error):.4f}")

    if np.linalg.norm(error) < 0.001:
        break

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, hand_body_id)

    d_theta = np.linalg.solve(jacp[:, :2].T @ jacp[:, :2] + 1e-6*np.eye(2),
                              jacp[:, :2].T @ error)
    data.qpos[0] += d_theta[0]
    data.qpos[1] += d_theta[1]

print("Final angles:", data.qpos[:2])
print("Final hand position:", data.xpos[hand_body_id])


with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(2000):

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)


