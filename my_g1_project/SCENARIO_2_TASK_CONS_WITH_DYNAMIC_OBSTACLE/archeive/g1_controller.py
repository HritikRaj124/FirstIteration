'''
g1_controller.py

Stage 2 of scenario 2 (Task-consistent manipulation and with dynamic human-hand obstacle)

This script loads two pretrained RL policies: `walker.onnx` (locomotion and
balance) and `right_reacher.onnx` (arm reaching) and runs them as the
NOMINAL controller for the G1 humanoid. It does not yet include the OSCBF/HOCBF
safety filter that is a separate, later layer that will sit downstream
of this file's torque output and correct it only when needed.

What this file produces, per control tick:
    target_pos : (29,) array -- joint position targets from the policies
    torque     : (29,) array -- the corresponding torque command, via the
                 explicit PD conversion tau = Kp(target - q) - Kd*qdot,
                 required because this robot's actuators are pure torque
                 actuators with no built-in position servo.

'''

import json
import numpy as np
import mujoco
import onnxruntime as ort

from torso_orientation import _quat_apply_inverse, TorsoOrientationController
from pd_to_torque import build_pd_gains, pd_to_torque

SCENE_PATH = r"D:\FirstIteration\my_g1_project\SCENARIO_2_TASK_CONS_WITH_DYNAMIC_OBSTACLE\scene.xml"
CONFIG_PATH = r"D:\FirstIteration\my_g1_project\SCENARIO_2_TASK_CONS_WITH_DYNAMIC_OBSTACLE\model_config.json"
WALKER_ONNX = "walker.onnx"
RIGHT_REACHER_ONNX = "right_reacher.onnx"

class ONNXPolicy:
    """ONNX Policy wrapper for CPU interface"""

    def __init__(self, model_path):

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs):
        obs = obs.reshape(1,-1).astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: obs})[0][0]

class G1Controller:
    def __init__(self, model, data, config, walker_policy, right_reacher_policy):
        self.model = model
        self.data = data
        self.walker_policy = walker_policy
        self.right_reacher_policy = right_reacher_policy

        self.joint_names = config["joint_names"]
        self.num_joints = len(self.joint_names)

        self.qpos_offsets = np.array([model.joint(n).qposadr[0] - 7 for n in self.joint_names])
        self.qvel_offsets = np.array([model.joint(n).dofadr[0] - 6 for n in self.joint_names])
        self.actuator_ids = np.array([model.actuator(n).id for n in self.joint_names])

        

