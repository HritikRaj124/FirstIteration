"""Configuration and external-asset resolution for G1 MuJoCo standard.

This module owns:
- repository-local paths and the output root,
- resolution of the external ``g1-manipulation-challenge`` asset directory,
- import of the vendor standalone module (``run.py``) as a reusable library.

The vendor repository lives *outside* this repository (assets are not committed
here). It is resolved via the ``G1_MANIPULATION_CHALLENGE_DIR`` environment
variable, falling back to ``~/robot_assets/external_baselines/g1-manipulation-challenge``
which matches the local robot-asset convention used during extraction.

The vendor ``run.py`` is import-safe (its entry point is guarded by
``if __name__ == "__main__"``). We load it under a unique module name so that we
reuse its ``ONNXPolicy``, ``G1Controller``, ``CameraRenderer`` and ``set_armature``
implementations instead of duplicating the simulation logic.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

PACKAGE_NAME = "g1_mujoco_standard"
STUDY_NAME = PACKAGE_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs"

# External baseline asset (cloned outside the repo).
G1_CHALLENGE_ENV = "G1_MANIPULATION_CHALLENGE_DIR"
DEFAULT_G1_CHALLENGE_DIR = (
    Path.home() / "robot_assets" / "external_baselines" / "g1-manipulation-challenge"
)

# Unique module name used when importing the vendor run.py as a library.
_VENDOR_MODULE_NAME = "g1_mujoco_standard_g1_challenge_vendor"

# Scene-level body / site names (from scene.xml and g1.xml).
RED_BLOCK_BODY = "red_block"
SOURCE_TABLE_BODY = "table"          # brown source table
TARGET_TABLE_BODY = "table_white"    # drop-off table
RIGHT_PALM_SITE = "right_palm"

# Cameras available in the scene (scene.xml + g1.xml).
SCENE_CAMERAS: tuple[str, ...] = (
    "overhead",
    "side_view",
    "tracking",
    "head_cam",
    "wrist_cam",
)

# Physics timestep used during policy training (must match for stable behavior).
TRAIN_TIMESTEP = 0.005  # 200 Hz
# Control decimation: policy runs every ``CONTROL_DECIMATION`` physics steps.
CONTROL_DECIMATION = 4


def resolve_g1_challenge_dir() -> Path:
    """Resolve the external g1-manipulation-challenge directory."""

    raw = os.environ.get(G1_CHALLENGE_ENV, "").strip()
    base = Path(raw) if raw else DEFAULT_G1_CHALLENGE_DIR
    return base.expanduser().resolve()


def vendor_scene_xml() -> Path:
    return resolve_g1_challenge_dir() / "scene.xml"


def vendor_model_config() -> Path:
    return resolve_g1_challenge_dir() / "model_config.json"


def assets_available() -> bool:
    """True when the external scene and the walker policy are present."""

    base = resolve_g1_challenge_dir()
    return (base / "scene.xml").is_file() and (base / "walker.onnx").is_file()


def missing_assets_message() -> str:
    base = resolve_g1_challenge_dir()
    return (
        "external g1-manipulation-challenge assets missing under "
        f"{base}. Clone the repo there or set {G1_CHALLENGE_ENV}."
    )


def ensure_output_dir(run_name: str) -> Path:
    out = OUTPUT_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_vendor_module() -> ModuleType:
    """Import the external ``run.py`` as a reusable library module.

    The module is cached in ``sys.modules`` under a unique name so repeated calls
    are cheap and the vendor's ``SCRIPT_DIR`` keeps pointing at its real on-disk
    location (which is how it resolves ``scene.xml``, meshes, and ``*.onnx``).
    """

    if _VENDOR_MODULE_NAME in sys.modules:
        return sys.modules[_VENDOR_MODULE_NAME]

    base = resolve_g1_challenge_dir()
    run_py = base / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(missing_assets_message())

    spec = importlib.util.spec_from_file_location(_VENDOR_MODULE_NAME, str(run_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for vendor module: {run_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VENDOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
