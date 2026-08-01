import re
from pathlib import Path
import mujoco
import tempfile

g1_dir = Path(r"C:\Users\Hritik Raj\robot_assets\external_baselines\g1-manipulation-challenge")

# READ from the vendor file -- fine
text = (g1_dir / "g1.xml").read_text()

marker = "<worldbody>"
my_line = '<option gravity="0 0 0"/>\n  '
final_text = text.replace(marker, my_line + marker, 1)

print(marker in final_text)
print(final_text.count("<option"))   # <- should print exactly 1 now

# Create a NEW, separate temp folder -- nothing here touches g1_dir
temp_dir = Path(tempfile.mkdtemp(prefix="my_g1_no_gravity_"))
(temp_dir / "assets").symlink_to(g1_dir / "assets", target_is_directory=True)

# WRITE only inside temp_dir -- never back into g1_dir
scene_out = temp_dir / "g1.xml"
scene_out.write_text(final_text, encoding="utf-8")

model = mujoco.MjModel.from_xml_path(str(scene_out))
data = mujoco.MjData(model)
print("Loaded successfully. njnt:", model.njnt)
print(temp_dir)