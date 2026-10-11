import mujoco
import model_utils as mu
import actuation as a

m, d = mu.load_model()
checked, bad = 0, []
for j in range(m.njnt):
    name = m.joint(j).name
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or "hand" in name:
        continue
    kp, kd, lim = a.gains_for(name)
    xml_lim = m.jnt_actfrcrange[j, 1]
    checked += 1
    if abs(lim - xml_lim) > 1e-6:
        bad.append((name, lim, xml_lim))
print("checked", checked, "mismatches", bad)