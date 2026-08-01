import re
from pathlib import Path
import mujoco

g1_dir = Path(r"C:\Users\Hritik Raj\robot_assets\external_baselines\g1-manipulation-challenge")
scene_path = g1_dir/"scene.xml"
text = scene_path.read_text()

text_without_block = re.sub(r'\s*<body name="red_block".*?</body>', "", text, flags=re.DOTALL)


my_obj_xml = """
    <body name="my_block" pos="0.351 0 0.75">
        <freejoint name="my_block_joint"/>
        <geom name="my_block_geom" type="cylinder" size="0.03 0.05" rgba="0.1 0.8 0.1 1" density="200" contype="1" conaffinity="1" friction="2 0.1 0.01"/>
    </body>    
"""

marker = "<!-- Fixed cameras -->"
final_text = text_without_block.replace(marker, my_obj_xml + "\n " + marker, 1)

print(marker in final_text)
print("my_block" in final_text)
print(final_text)