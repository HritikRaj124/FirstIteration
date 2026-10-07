"""Generate the partition pick-and-place scene from the MuJoCo Menagerie G1 model."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

MENAGERIE_ENV = "G1_MENAGERIE_DIR"

SCENE_FRAGMENT = """
    <body name="table" pos="0.5 0 0.713">
        <geom name="table_top" type="box" size="0.4 0.7 0.02" />
        <!--  Table legs  -->
        <geom name="table_leg_1" type="cylinder" size="0.025 0.345" pos="0.35 0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_2" type="cylinder" size="0.025 0.345" pos="-0.35 0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_3" type="cylinder" size="0.025 0.345" pos="0.35 -0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
        <geom name="table_leg_4" type="cylinder" size="0.025 0.345" pos="-0.35 -0.2 -0.365" rgba="0.4 0.25 0.15 1" contype="1" conaffinity="1" mass="1"/>
    </body>

    <body name="partition" pos="0.475 0.05 0.808">
      <geom name="partition_wall" type="box" size="0.35 0.01 0.075"
            rgba="0.8 0.1 0.1 1" contype="1" conaffinity="1"/>
    </body>

    <body name="movable_object" pos="0.3 0.25 0.80">
      <freejoint name="movable_object_joint"/>
      <geom name="movable_object_geom" type="cylinder" size="0.025 0.03"
            rgba="0.1 0.6 0.8 1" density="200"
            contype="0" conaffinity="1" friction="2 0.1 0.01"/>
    </body>
"""


def build_scene(menagerie_g1_dir: Path | None, out_path: Path) -> Path:
    """Write out_path = Menagerie scene.xml + table, partition and object.

    menagerie_g1_dir defaults to $G1_MENAGERIE_DIR (the `unitree_g1` folder).
    """
    g1_dir = Path(menagerie_g1_dir or os.environ.get(MENAGERIE_ENV, ""))
    src = g1_dir / "scene.xml"
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found. Pass --menagerie-dir or set ${MENAGERIE_ENV} "
            "to the mujoco_menagerie/unitree_g1 folder."
        )

    marker = "</worldbody>"
    text = src.read_text()
    if marker not in text:
        raise ValueError(f"No {marker} in {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patched = text.replace(marker, SCENE_FRAGMENT + "\n  " + marker, 1)
    out_path.write_text(patched, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out_path))  # validate it loads
    print(f"Wrote {out_path} (njnt={model.njnt}, nbody={model.nbody})")
    return out_path
