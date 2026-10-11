"""

inverse_kniematics.py

inverse kinematics for one arm
"""

import numpy as np
import mujoco

def site_jacobian(model, data, site_name, dof_addrs):
    site_id = model.site(site_name).id
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    return data.site_xpos[site_id].copy(), jacp[:, dof_addrs]