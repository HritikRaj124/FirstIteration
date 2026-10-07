"""Command line interface: `g1-cbf build-scene` and `g1-cbf run`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import __version__
from .config import RunConfig, SafetyConfig, SceneConfig, TaskConfig
from .runtime import configure_jax


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="g1-cbf", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-scene", help="generate scene.xml from the Menagerie unitree_g1 model")
    b.add_argument(
        "--menagerie-dir", type=Path, help="path to mujoco_menagerie/unitree_g1 (default: $G1_MENAGERIE_DIR)"
    )
    b.add_argument("--out", type=Path, default=SceneConfig.scene_path)

    r = sub.add_parser("run", help="run baseline vs CBF-filtered rollouts and plot the comparison")
    r.add_argument("--scene", type=Path, default=SceneConfig.scene_path)
    r.add_argument("--out-dir", type=Path, default=RunConfig.out_dir)
    r.add_argument("--mode", choices=["both", "baseline", "filtered"], default="both")
    r.add_argument("--alpha", type=float, default=SafetyConfig.alpha, help="CBF class-K gain")
    r.add_argument("--qdot-max", type=float, default=SafetyConfig.qdot_max, help="rad/s")
    r.add_argument("--arm-radius", type=float, default=SafetyConfig.arm_radius, help="m")
    r.add_argument("--phase-duration", type=float, default=TaskConfig.phase_duration, help="s")
    r.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    r.add_argument("--gpu", action="store_true", help="run JAX on GPU")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.command == "build-scene":
        from .scene_builder import build_scene

        build_scene(args.menagerie_dir, args.out)
        return 0

    configure_jax(use_gpu=args.gpu)  # must precede the first jax import
    from .plotting import plot_filter_comparison
    from .simulation import run_scenario, summarize

    cfg = RunConfig(
        scene=SceneConfig(scene_path=args.scene),
        task=TaskConfig(phase_duration=args.phase_duration),
        safety=SafetyConfig(alpha=args.alpha, qdot_max=args.qdot_max, arm_radius=args.arm_radius),
        show_viewer=args.viewer,
        use_gpu=args.gpu,
        out_dir=args.out_dir,
    )

    modes = {"both": ["baseline", "filtered"], "baseline": ["baseline"], "filtered": ["filtered"]}[args.mode]
    traces = {m: run_scenario(cfg.with_filter(m == "filtered")) for m in modes}

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {m: summarize(t) for m, t in traces.items()}
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if len(traces) == 2:
        paths = plot_filter_comparison(traces["baseline"], traces["filtered"], cfg.out_dir)
        for name, path in paths.items():
            print(f"{name}: {path}")
    print(f"metrics: {cfg.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
