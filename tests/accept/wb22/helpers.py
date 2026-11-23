"""Shared spec and helpers for the WB22 acceptance tests (lane B GPU list).

Everything here comes from docs/plans/LANE_D_PLAN.md (WB22 row, §12b.6 D42,
rev6.8 WB25 ``gpu_n``, TB23 / TB24) and the owner decisions of 2026-11-23
(README.md). Nothing from a PATHS module is imported at module level; the
WB22 module is imported inside test bodies (``gpus()``), with torch blocked.
"""

from __future__ import annotations

import importlib
import importlib.abc
import json
import re
import sys
import tomllib
from contextlib import contextmanager
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOML = REPO / "chain.toml"
TESTNET_TOML = REPO / "chain.testnet.toml"
MODULE = "cascade.funding.lane_b_gpus"

# The owner's list (2026-11-23): lane A's data-centre multi-GPU ladder plus
# 8xRTX4090 (lane B only); never a single-GPU layout.
OWNER_LANE_B_SKUS = ("8xH200", "8xH100", "4xH200", "4xH100", "8xRTX4090")
# [round] funded_pod_skus on main (lane A's list); WB22 never changes it.
MAIN_ROUND_SKUS = ("8xH200", "8xH100", "4xH200", "4xH100")
UNMEASURED = "gpu_rate_unmeasured"
STEP_RATES_ENV = "CASCADE_LANE_B_STEP_RATES"

_SHA = re.compile(r"sha256:[0-9a-f]{64}")
GIB = 1024 ** 3


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"WB22: '{name}' must not be imported", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def gpus():
    """The WB22 module, imported with torch blocked."""
    with torch_blocked():
        return importlib.import_module(MODULE)


@cache
def _raw(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def raw_toml(path: Path = CHAIN_TOML) -> dict:
    return tomllib.loads(_raw(path))


def chain_cfg(path: Path = CHAIN_TOML):
    from cascade.shared.config import load_chain_config

    return load_chain_config(str(path))


def cfg_with_lane_b(**lane_b):
    """The shipped mainnet config with ``[lane_b]`` keys overwritten."""
    from cascade.shared.config import parse_chain_config

    raw = raw_toml()
    raw["lane_b"] = {**raw.get("lane_b", {}), **lane_b}
    return parse_chain_config(raw)


def pinned_image(cfg=None) -> str:
    """Independent derivation: the image of the contract at [v2] from_block."""
    cfg = cfg or chain_cfg()
    block = int(cfg.training.v2_from_block)
    m = _SHA.search(str(cfg.throne_contracts_at(block)[0].train_image_digest))
    assert m, "the shipped contract carries no sha256 image pin"
    return m.group(0)


GPU_ROWS = {
    "8xH200": ("NVIDIA H200", 8, 141 * GIB),
    "8xH100": ("NVIDIA H100 80GB HBM3", 8, 80 * GIB),
    "4xH200": ("NVIDIA H200", 4, 141 * GIB),
    "4xH100": ("NVIDIA H100 80GB HBM3", 4, 80 * GIB),
    "8xRTX4090": ("NVIDIA GeForce RTX 4090", 8, 24 * GIB),
}


def table(skus, *, image: str | None = None, rate: float = 0.44) -> dict:
    """A ``laned_wb19_table/1`` with one measured row per SKU."""
    rows = {}
    for i, sku in enumerate(skus):
        name, count, mem = GPU_ROWS[sku]
        r = rate + 0.01 * i
        rows[sku] = {"gpu_name": name, "gpu_count": count, "driver": "580.65", "cuda": "12.4",
                     "steps_per_s": r, "tokens_per_s": r * 512 * 4096,
                     "overhead_seconds": 600.0, "peak_memory_bytes": mem // 2,
                     "memory_total_bytes": mem, "run_body_sha256": "ab" * 32}
    return {"schema": "laned_wb19_table/1", "image_digest": image or pinned_image(),
            "contract_digest": "cd" * 32, "contract_block": 9_399_600, "rows": rows}


def write_table(path: Path, skus, **kw) -> Path:
    path.write_text(json.dumps(table(skus, **kw)), encoding="utf-8")
    return path


def rates_env(tmp_path: Path, skus, **kw) -> dict:
    """An env whose STEP_RATES_ENV names a table measuring ``skus``."""
    return {STEP_RATES_ENV: str(write_table(tmp_path / "wb19-table.json", skus, **kw))}


def norm(v) -> tuple:
    return tuple(v or ())
