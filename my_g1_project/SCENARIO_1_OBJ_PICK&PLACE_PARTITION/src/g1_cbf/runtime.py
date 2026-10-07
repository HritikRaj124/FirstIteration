"""JAX runtime setup. Call configure_jax() before the first `import jax`."""

from __future__ import annotations

import os


def configure_jax(use_gpu: bool = False) -> None:
    """Select the JAX backend. Must run before jax is imported anywhere."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if use_gpu:
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
