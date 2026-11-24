"""The whole of a lane D entry's ``config.json``, pinned at intake.

Ingest (W4, :mod:`cascade.eval.lane_d_ingest` step 6) compares only the
``"moe"`` object with the contract's. Every other top-level key rode along
unchecked, and :meth:`CascadeMoEConfig.from_dict` silently drops keys it does
not know, so a key nothing reads today would become miner-controlled the day
some code starts reading it. The intake therefore pins the WHOLE file at
``ready``: an explicit allowlist of top-level keys (exactly what
``cascade.trainer.pyramid_trainer.save_checkpoint`` writes), the ``"moe"`` key
set (exactly :class:`CascadeMoEConfig`'s fields) and a pinned value or shape
for every other key. Anything else is a typed refusal.

Stdlib only: the intake box never imports torch, so the field names are
pinned here and a test holds them equal to the trainer's sources.
"""

from __future__ import annotations

import json
import math
import re

from .intake import IntakeError

ENTRY_CONFIG_FILE = "config.json"
# A real lane D config.json is ~1 KB; anything far larger is not one.
MAX_ENTRY_CONFIG_BYTES = 64 * 1024
LANE_D_BASE_ARCH = "cascade_pyramid"
# cascade.trainer.pyramid_model.QUANTILE_LEVELS (same expression).
PYRAMID_QUANTILE_LEVELS = (0.01, *(i / 20 for i in range(1, 20)), 0.99)
# cascade.trainer.pyramid_model.CascadeMoEConfig fields, in declaration order.
MOE_KEYS = (
    "d_model", "num_layers", "num_heads", "head_dim", "patch_size", "num_quantiles",
    "context_length", "horizon", "max_patches", "variate_blocks", "moe_blocks", "dense_ffn",
    "num_experts", "experts_per_layer", "top_k", "expert_hidden", "shared_hidden",
    "conv_channels", "conv_kernel", "conv_dilations", "copula_rank", "residual_mult",
    "cpm_c_max", "cpm_p_max", "bias_update_rate", "balance_loss_coef", "router_z_loss_coef",
)
# The top-level keys pyramid_trainer.save_checkpoint writes ("power_ema" only
# on an EMA-scored checkpoint, POWER_EMA_MARKER).
ENTRY_CONFIG_KEYS = ("arch", "base_arch", "input_transform", "moe", "power_ema",
                     "quantile_levels")
REQUIRED_ENTRY_CONFIG_KEYS = ("base_arch", "moe")
POWER_EMA_KEYS = ("scored_sigma_rel", "sigma_rels")

# The typed refusals (also in cascade.lane_d.service.REFUSAL_CODES).
CONFIG_UNREADABLE = "config_unreadable"
CONFIG_KEY_NOT_ALLOWED = "config_key_not_allowed"
CONFIG_VALUE_NOT_ALLOWED = "config_value_not_allowed"
ENTRY_CONFIG_CODES = (CONFIG_UNREADABLE, CONFIG_KEY_NOT_ALLOWED, CONFIG_VALUE_NOT_ALLOWED)

_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _bad(detail: str) -> IntakeError:
    return IntakeError(CONFIG_VALUE_NOT_ALLOWED, detail)


def _check_moe(moe) -> None:
    if not isinstance(moe, dict):
        raise _bad('"moe" is not an object')
    extra = sorted(str(k)[:64] for k in moe if k not in MOE_KEYS)
    if extra:
        raise IntakeError(CONFIG_KEY_NOT_ALLOWED, f"moe keys {extra[:10]} are not "
                                                  "CascadeMoEConfig fields")
    missing = [k for k in MOE_KEYS if k not in moe]
    if missing:
        raise _bad(f"moe lacks {missing[:10]}")
    for k in MOE_KEYS:
        v = moe[k]
        if isinstance(v, list):
            if not all(_int(x) for x in v):
                raise _bad(f"moe.{k} is not a list of integers")
        elif not _number(v):
            raise _bad(f"moe.{k} is not a finite number")


def _check_power_ema(p) -> None:
    if not isinstance(p, dict) or sorted(p) != sorted(POWER_EMA_KEYS):
        raise _bad(f'"power_ema" must be an object with exactly {list(POWER_EMA_KEYS)}')
    sig = p["sigma_rels"]
    if not isinstance(sig, list) or not sig or not all(_number(s) for s in sig):
        raise _bad("power_ema.sigma_rels is not a non-empty list of numbers")
    if not _number(p["scored_sigma_rel"]) or p["scored_sigma_rel"] not in sig:
        raise _bad("power_ema.scored_sigma_rel is not one of sigma_rels")


def check_entry_config(raw: bytes) -> dict:
    """The parsed ``config.json`` when the whole file is allowed, else a typed
    :class:`IntakeError` (``config_unreadable`` / ``config_key_not_allowed`` /
    ``config_value_not_allowed``). Never executes anything: plain JSON."""
    if len(raw) > MAX_ENTRY_CONFIG_BYTES:
        raise IntakeError(CONFIG_UNREADABLE, f"{ENTRY_CONFIG_FILE} is {len(raw)} bytes "
                                             f"(> {MAX_ENTRY_CONFIG_BYTES})")
    try:
        obj = json.loads(bytes(raw).decode("utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise IntakeError(CONFIG_UNREADABLE, f"{ENTRY_CONFIG_FILE} is not JSON") from None
    if not isinstance(obj, dict):
        raise IntakeError(CONFIG_UNREADABLE, f"{ENTRY_CONFIG_FILE} is not a JSON object")
    extra = sorted(str(k)[:64] for k in obj if k not in ENTRY_CONFIG_KEYS)
    if extra:
        raise IntakeError(CONFIG_KEY_NOT_ALLOWED,
                          f"{ENTRY_CONFIG_FILE} keys {extra[:10]} are outside the allowlist "
                          f"{list(ENTRY_CONFIG_KEYS)}")
    missing = [k for k in REQUIRED_ENTRY_CONFIG_KEYS if k not in obj]
    if missing:
        raise _bad(f"{ENTRY_CONFIG_FILE} lacks {missing}")
    if obj["base_arch"] != LANE_D_BASE_ARCH:
        raise _bad(f"base_arch must be {LANE_D_BASE_ARCH!r}")
    _check_moe(obj["moe"])
    if "quantile_levels" in obj and obj["quantile_levels"] != list(PYRAMID_QUANTILE_LEVELS):
        raise _bad("quantile_levels differ from the pyramid's")
    for k in ("arch", "input_transform"):
        if k in obj and not (isinstance(obj[k], str) and _NAME.fullmatch(obj[k])):
            raise _bad(f"{k} is not a plain name")
    if "power_ema" in obj:
        _check_power_ema(obj["power_ema"])
    return obj


__all__ = [
    "CONFIG_KEY_NOT_ALLOWED", "CONFIG_UNREADABLE", "CONFIG_VALUE_NOT_ALLOWED",
    "ENTRY_CONFIG_CODES", "ENTRY_CONFIG_FILE", "ENTRY_CONFIG_KEYS", "LANE_D_BASE_ARCH",
    "MAX_ENTRY_CONFIG_BYTES", "MOE_KEYS", "POWER_EMA_KEYS", "PYRAMID_QUANTILE_LEVELS",
    "REQUIRED_ENTRY_CONFIG_KEYS", "check_entry_config",
]
