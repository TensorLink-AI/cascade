"""Shared spec and helpers for the WB23 acceptance tests (lane B leg length).

Everything here comes from docs/plans/LANE_D_PLAN.md (WB23 row, §12b.6 D42
item 2 and item 5, D42b = (a), TB25-TB27) and README.md. Nothing from a
PATHS module is imported at module level; the WB23 module is imported inside
test bodies (``leg()``), with torch blocked.
"""

from __future__ import annotations

import importlib
import importlib.abc
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOML = REPO / "chain.toml"
MODULE = "cascade.shared.lane_b_leg"
STEP_RATES_ENV = "CASCADE_LANE_B_STEP_RATES"
TRAIN_HOURS_HEADER = "X-Lane-B-Train-Hours"

BAD_TRAIN_HOURS = "bad_train_hours"
NEEDS_GPU = "train_hours_needs_gpu"
WITH_FALLBACK = "train_hours_with_fallback"
UNAVAILABLE = "train_hours_unavailable"

# A type both lane A's and lane B's lists carry on main and after WB22.
SKU = "8xH100"
RATE = 0.44            # steps/s of the fake WB19 row
OVERHEAD = 600.0       # s, the fake WB19 row's leg overhead

GPU_ROWS = {
    "8xH100": ("NVIDIA H100 80GB HBM3", 8, 80 * 1024 ** 3),
    "8xH200": ("NVIDIA H200", 8, 141 * 1024 ** 3),
}
_SHA = re.compile(r"sha256:[0-9a-f]{64}")


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"WB23: '{name}' must not be imported", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def leg():
    """The WB23 module, imported with torch blocked."""
    with torch_blocked():
        return importlib.import_module(MODULE)


def chain_cfg():
    from cascade.shared.config import load_chain_config

    return load_chain_config(str(CHAIN_TOML))


def lane_b_cfg():
    return chain_cfg().lane_b


def base_contract(cfg=None):
    """The contract a lane B leg derives from: the throne contract at [v2] from_block."""
    cfg = cfg or chain_cfg()
    return cfg.throne_contracts_at(int(cfg.training.v2_from_block))[0]


def pinned_image(cfg=None) -> str:
    cfg = cfg or chain_cfg()
    m = _SHA.search(str(base_contract(cfg).train_image_digest))
    assert m, "the shipped contract carries no sha256 image pin"
    return m.group(0)


# ── independent derivations of the README formulas ───────────────────────────

def want_reference_steps(contract, hours: float) -> int:
    return max(1, round(contract.training_steps * hours / contract.target_train_hours))


def want_steps(hours: float, rate: float, cap: int) -> int:
    return min(math.floor(hours * 3600 * rate), cap)


# owner 2026-11-24 (plan rev6.21; tests-correction): max_leg_hours / max_leg_seconds
# ship at 12 h / 54000 s and bind as configured (lane B is not rolled out: no
# activation block).


def caps_in_force(lb) -> tuple[float, int]:
    return float(lb.max_leg_hours), int(lb.max_leg_seconds)


def want_guard(hours: float, lb, overhead: float = OVERHEAD) -> int:
    return min(math.ceil(hours * 3600 * float(lb.guard_factor) + overhead),
               caps_in_force(lb)[1])


def want_guard_floor(hours: float, lb) -> int:
    return min(math.ceil(hours * 3600 * float(lb.guard_factor)), caps_in_force(lb)[1])


def step_rate(sku: str = SKU, rate: float = RATE, overhead: float = OVERHEAD):
    from cascade.funding.step_rates import StepRate

    name, count, mem = GPU_ROWS[sku]
    return StepRate(sku=sku, gpu_name=name, gpu_count=count, steps_per_s=rate,
                    tokens_per_s=rate * 512 * 4096, overhead_seconds=overhead,
                    peak_memory_bytes=mem // 2, memory_total_bytes=mem)


def table(skus, *, rate: float = RATE) -> dict:
    """A ``laned_wb19_table/1`` measuring ``skus`` on the pinned image."""
    rows = {}
    for sku in skus:
        name, count, mem = GPU_ROWS[sku]
        rows[sku] = {"gpu_name": name, "gpu_count": count, "driver": "580.65", "cuda": "12.4",
                     "steps_per_s": rate, "tokens_per_s": rate * 512 * 4096,
                     "overhead_seconds": OVERHEAD, "peak_memory_bytes": mem // 2,
                     "memory_total_bytes": mem, "run_body_sha256": "ab" * 32}
    return {"schema": "laned_wb19_table/1", "image_digest": pinned_image(),
            "contract_digest": "cd" * 32, "contract_block": 9_399_600, "rows": rows}


def rates_env(tmp_path: Path, skus=(SKU,)) -> dict:
    p = tmp_path / "wb19-table.json"
    p.write_text(json.dumps(table(list(skus))), encoding="utf-8")
    return {STEP_RATES_ENV: str(p)}
