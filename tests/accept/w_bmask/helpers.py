"""W-BMASK fixtures. Nothing here imports torch at import time."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SIDECAR = REPO / "benchmarks" / "cascade_benchmark"
OBSERVED = SIDECAR / "observed.py"


def load_observed():
    """The sidecar's observed.py, loaded by path (the sidecar is its own
    project; observed.py imports only numpy at module level)."""
    assert OBSERVED.is_file(), "benchmarks/cascade_benchmark/observed.py is missing"
    spec = importlib.util.spec_from_file_location("w_bmask_observed", OBSERVED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pyramid_contract():
    """The smallest pyramid size (``cascade-moe-v2-2m``)."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w_bmask_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_checkpoint(directory: Path, *, seed: int = 31) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code
    (model.py and forecast_wrapper.py are byte copies of the repo's)."""
    import torch

    from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    contract = pyramid_contract()
    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    state = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    save_checkpoint(directory, state, config, contract)
    return directory


def load_pinned_wrapper(checkpoint_dir: Path):
    """The checkpoint's forecast_wrapper.Wrapper, imported the way the
    sidecar's predictor imports it."""
    spec = importlib.util.spec_from_file_location(
        "w_bmask_bench_wrapper", Path(checkpoint_dir) / "forecast_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Wrapper(str(checkpoint_dir), device="cpu")


def series(n_points: int, seed: int, level: float = 20.0):
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64)
    return level + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n_points)
