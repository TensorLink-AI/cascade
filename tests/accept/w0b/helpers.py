"""Shared spec and fixture builders for W0b (context equivalence).

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W0b row, §1.1 (context is
a scorer-side setting, config stays 4096, no padding, the weights fit any
context because the positional buffers are non-persistent), and B7 (quantile
head at the 9-level grid, the lane A scoring path under
``[v2] quantile_head_after``). README.md states the contract.

Nothing here imports torch or a lane D module at import time: the torch-only
builders import inside their bodies, and only tests marked ``torch`` call
them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# Every context variant W0b iterates: the trained 4096 (128 patches of 32)
# and its 2x..32x extensions (§1.1; launch arms 4k/8k/16k, S8 adds 32k-128k).
W0B_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)
TRAINED_CONTEXT = 4096
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# The pyramid head's Q21 (pyramid_model.QUANTILE_LEVELS), written out so the
# non-torch tests never import the model module.
HEAD_LEVELS = (0.01, *(i / 20 for i in range(1, 20)), 0.99)
HORIZON = 64
SEASONAL_PERIOD = 24

# Max abs difference allowed between the lane D scorer and the pinned wrapper
# on the same windows, in data units. Both run the same fp32 forward with the
# same weights on the same last-4096 points; the series level is ~20, where
# one fp32 ulp is ~2e-6, so 1e-4 is ~50 ulp. That covers only reduction-order
# differences between batch shapes (lane A's quantile path decodes one window
# per forward, the lane D scorer one forward per input length). Any real
# divergence (padding, a different context, a wrong level pick, a different
# scale) moves these quantiles by orders of magnitude more; the edge-pad
# control test shows that on the same checkpoints.
ATOL = 1e-4


def series(n_points: int, seed: int, level: float = 20.0) -> np.ndarray:
    """A positive hourly-looking float64 series of ``n_points`` points."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64)
    return level + 3.0 * np.sin(2 * np.pi * t / SEASONAL_PERIOD) + rng.normal(0, 0.5, n_points)


def long_histories() -> list[np.ndarray]:
    """Twelve windows with >= 4096 history: exact 4096, patch multiples above
    it, and lengths that are not patch multiples (the wrapper and the scorer
    both keep the last 4096 points at context 4096)."""
    lengths = [TRAINED_CONTEXT + 37 * i for i in range(10)] + [TRAINED_CONTEXT + 32, 6000]
    return [series(n, seed=200 + i) for i, n in enumerate(lengths)]


# ── torch-only builders (call only from tests marked ``torch``) ──────────────


def pyramid_contract():
    """The smallest pyramid size (``cascade-moe-v2-2m``), via the repo's own
    ``scripts/pyramid_config.py`` (loaded by path)."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w0b_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_checkpoint(directory: Path, contract, *, seed: int, median_shift: float,
                    bf16: bool) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code
    (``save_checkpoint``: config.json at context 4096, model.py and
    forecast_wrapper.py byte copies of the repo's). ``median_shift`` raises
    the head's median-channel bias; ``bf16`` casts floating tensors to BF16
    (D14: lane D entries are BF16, upcast to fp32 at load)."""
    import torch

    from cascade.trainer.pyramid_model import HEAD_CHANNELS, CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    if median_shift:
        with torch.no_grad():
            model.head.bias.view(config.patch_size, HEAD_CHANNELS)[:, 0] += median_shift
    state = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach()
        if bf16 and t.is_floating_point():
            t = t.to(torch.bfloat16)
        state[name] = t.contiguous().clone()
    save_checkpoint(directory, state, config, contract)
    return directory
