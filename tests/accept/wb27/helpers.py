"""Shared spec and helpers for the WB27 acceptance tests (plan rev6.21, owner
2026-11-24): the miner-chosen lane B leg levers, 12 h legs, token limits and
own-risk GPUs. Nothing from a PATHS module is imported at module level; the
modules are imported inside test bodies with torch blocked."""

from __future__ import annotations
import __future__

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOML = REPO / "chain.toml"
LEVERS = "cascade.shared.lane_b_levers"
LEVERS_HEADER = "X-Lane-B-Levers"
HOURS_HEADER = "X-Lane-B-Train-Hours"
TOKENS_HEADER = "X-Lane-B-Train-Tokens"
# an era boundary well after [v2] from_block on the shipped grid (WB10's)
SKU = "8xH100"
RATE = 0.44
OVERHEAD = 600.0
UNMEASURED_SKU = "4xL40S"
HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/gen@sha256:" + "a" * 64
CODES = ("bad_levers", "lr_scale_out_of_range", "bad_curriculum",
         "context_length_out_of_range", "context_length_unfit", "ema_out_of_range")


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"WB27: '{name}' must not be imported", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def levers():
    with torch_blocked():
        return importlib.import_module(LEVERS)


def leg():
    with torch_blocked():
        return importlib.import_module("cascade.shared.lane_b_leg")


def chain_cfg():
    from cascade.shared.config import load_chain_config

    return load_chain_config(str(CHAIN_TOML))


def effective_block(cfg=None) -> int:
    """An era boundary after ``[v2] from_block`` on the shipped grid."""
    cfg = cfg or chain_cfg()
    from cascade.shared.era import era_length_blocks

    v2 = int(cfg.training.v2_from_block)
    era = era_length_blocks(cfg.round, v2)
    return (v2 // era + 4) * era


def armed_cfg(*, ema: bool = True, block: int | None = None):
    """The shipped config and an era boundary after [v2] from_block: no
    activation block anywhere in lane B (owner 2026-11-24), so the levers and
    rev6.20's EMA / no-cooldown rule are the shipped behaviour."""
    del ema
    cfg = chain_cfg()
    return cfg, block or effective_block(cfg)


def base_contract(cfg=None):
    cfg = cfg or chain_cfg()
    return cfg.throne_contracts_at(int(cfg.training.v2_from_block))[0]


def want_tokens_per_step(c) -> int:
    return int(c.batch_size) * (int(c.context_length) // int(c.patch_size)) * int(c.patch_size)


def want_reference_steps(c, hours: float) -> int:
    return max(1, round(c.training_steps * hours / c.target_train_hours))


def table(skus) -> dict:
    from cascade.funding.lane_b_gpus import lane_b_image_pin

    rows = {}
    for sku in skus:
        rows[sku] = {"gpu_name": "NVIDIA H100 80GB HBM3", "gpu_count": 8, "driver": "580.65",
                     "cuda": "12.4", "steps_per_s": RATE, "tokens_per_s": RATE * 512 * 4096,
                     "overhead_seconds": OVERHEAD, "peak_memory_bytes": 40 * 1024 ** 3,
                     "memory_total_bytes": 80 * 1024 ** 3, "run_body_sha256": "ab" * 32}
    return {"schema": "laned_wb19_table/1", "image_digest": lane_b_image_pin(chain_cfg()),
            "contract_digest": "cd" * 32, "contract_block": 9_399_600, "rows": rows}


def rates_env(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "wb19-table.json"
    p.write_text(json.dumps(table([SKU])), encoding="utf-8")
    return {"CASCADE_LANE_B_STEP_RATES": str(p)}


def intake(tmp_path: Path, cfg):
    from cascade.funding.intake import FundingIntake
    from cascade.funding.main import gpu_catalogs
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    cat_a, gpus_b = gpu_catalogs(cfg, env=rates_env(tmp_path))
    lane_b = LaneBQueue(lane_b_queue_path(tmp_path / "queue.json"))
    it = FundingIntake(
        FundedQueue(tmp_path / "queue.json"), PayerKeyVault(dir=None),
        resolve_reveal=lambda hk, ref: 700, require_signature=False,
        verify=lambda hk, msg, s: False, gpu_catalog=cat_a,
        lane_b_queue=lane_b, resolve_lane_b_reveal=lambda hk, t, r: 700,
        lane_b_targets=None, lane_b_max_targets=2, lane_b_gpus=gpus_b,
        lane_b_cfg=cfg.lane_b, lane_b_contract=base_contract(cfg))
    return it, lane_b


def fund_headers(**kw) -> dict:
    from cascade.funding.intake import LANE_B_TARGETS_HEADER

    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, LANE_B_TARGETS_HEADER: "energy",
         "X-Lium-Api-Key": "sk-live", "X-Gpu-Type": SKU}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def nothing_charged(it, lane_b) -> None:
    assert lane_b.get(HK) is None, "a refused fund must queue nothing"
    assert it.vault.get(HK) is None, "a refused fund must vault no key"


# ── pyramid_trainer.py pure defs, read from source (this box never loads torch)

def trainer_path() -> Path:
    spec = importlib.util.find_spec("cascade.trainer.pyramid_trainer")
    assert spec is not None and spec.origin
    return Path(spec.origin)


def trainer_namespace() -> dict:
    tree = ast.parse(trainer_path().read_text(encoding="utf-8"))
    ns: dict = {"__name__": "wb27_source_view", "json": json, "math": math,
                "hashlib": hashlib, "time": time, "Path": Path}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(trainer_path()),
                           "exec", flags=__future__.annotations.compiler_flag,
                           dont_inherit=True)
            try:
                exec(code, ns)  # noqa: S102 — repo source, not miner bytes
            except NameError:
                continue
    return ns


def require_built() -> None:
    """WB27's surface exists (an AssertionError, never an import or attribute
    error, on a tree without it): every test calls this first."""
    import dataclasses

    assert importlib.util.find_spec(LEVERS) is not None, f"{LEVERS} is missing (WB27)"
    from cascade.shared.config.v2 import LaneBConfig

    names = {f.name for f in dataclasses.fields(LaneBConfig)}
    for k in ("lr_scale_min", "lr_scale_max", "sigma_rel_choices", "max_context_length"):
        assert k in names, f"LaneBConfig has no {k} (WB27)"
    from cascade.shared.expert_manifest import ExpertEntry

    entry = {f.name for f in dataclasses.fields(ExpertEntry)}
    for k in ("levers", "train_tokens", "gpu_measured", "trained_steps", "trained_tokens",
              "stop"):
        assert k in entry, f"ExpertEntry has no {k} (WB27)"
    from cascade.funding.intake import FundingIntake

    assert hasattr(FundingIntake, "_lane_b_levers"), "the intake parses no levers (WB27)"
    src = trainer_path().read_text(encoding="utf-8")
    for name in ("lane_b_lever_terms", "lane_b_context_patches", "lane_b_leg_record"):
        assert f"def {name}(" in src, f"pyramid_trainer.{name} is missing (WB27)"
    cli = (REPO / "cascade" / "generation" / "cli.py").read_text(encoding="utf-8")
    assert "--curriculum-stage" in cli, "the generation CLI takes no --curriculum-stage (WB27)"
    import cascade.shared.lane_b_leg as leg_mod

    for name in ("admit_leg_limits", "tokens_per_step", "TRAIN_TOKENS_HEADER"):
        assert hasattr(leg_mod, name), f"lane_b_leg.{name} is missing (WB27)"
