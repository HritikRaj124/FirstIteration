import numpy as np
from reacher_policy import apply_action, ARM_DEFAULT_POS

raw_action = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
last_arm_target = ARM_DEFAULT_POS.copy()

result = apply_action(raw_action, last_arm_target)
print(result)
print(result[0])  
