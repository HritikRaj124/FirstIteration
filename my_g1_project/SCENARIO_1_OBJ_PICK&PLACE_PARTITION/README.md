# g1-cbf

Control-barrier-function (CBF) safety filtering for Unitree G1 left-arm pick-and-place,
built on MuJoCo, MJX/JAX and ProxSuite. A nominal min-jerk joint trajectory is filtered
by a QP so the forearm capsule stays clear of a partition wall and the table.

## Pipeline

```
build-scene → IK waypoints → min-jerk reference → CBF-QP filter → MuJoCo step → log → plots/metrics
```

| Module | Role |
| --- | --- |
| `config.py` | Frozen dataclasses for scene, task, safety and IK settings |
| `scene_builder.py` | Generates `scene.xml` from Menagerie's `unitree_g1` plus table, partition, object |
| `kinematics.py` | Damped-least-squares IK via `jax.jacfwd` through MJX |
| `barrier.py` | Differentiable capsule-vs-box signed distance, h(q) |
| `safety_filter.py` | CBF-QP: `min ‖q̇ − q̇_des‖²  s.t.  ∇h·q̇ ≥ −α h,  |q̇| ≤ q̇_max` |
| `simulation.py` | Rollout loop, returns a per-tick log; `summarize()` gives scalar metrics |
| `plotting.py` | Baseline-vs-filtered comparison figures |
| `cli.py` | `g1-cbf` entry point |

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Use

```bash
export G1_MENAGERIE_DIR=/path/to/mujoco_menagerie/unitree_g1
g1-cbf build-scene                       # writes scenarios/partition_task/scene.xml
g1-cbf run                               # baseline + filtered, writes outputs/filter_comparison/
g1-cbf run --mode filtered --alpha 5 --viewer
```

Outputs: three comparison PNGs and `metrics.json` (violation counts, minimum margins,
tracking error) in `--out-dir`.

## Development

```bash
ruff check . && ruff format --check .
pytest
```

CI (`.github/workflows/scenario1-ci.yml` at the repository root) runs lint and the unit tests
on every change to this folder.

`legacy/` holds the earlier controller without the CBF filter; it is not part of the package.
