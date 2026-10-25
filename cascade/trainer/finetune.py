"""Generic lane-C recipe transforms retained for compatibility (DEC-CA-0056).

The generic recipe.json path stages a weights-only base, validates bounded
learning-rate/step/schedule settings, and disables lineage averaging and
finished-form branches for the exported fine-tune. Architecture fields stay
unchanged and the staged optimizer starts fresh.

Current MarketCast admission uses marketcast.json, the pinned Timestral base,
and marketcast_runner instead. RecipeError is shared by both paths; the generic
helpers remain available for existing workers and historical contracts.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

RECIPE_FILE = "recipe.json"
RECIPE_SCHEDULES = ("cosine", "constant", "linear")
# recipe schedule -> the trainer's lr_schedule (cascade.trainer.toto2_trainer)
SCHEDULE_MAP = {"cosine": "warmup_cosine", "constant": "warmup_flat",
                "linear": "warmup_linear"}
# the files a weights-only warm start keeps (the leg checkpoint layout)
_BASE_FILES = ("weights.safetensors", "config.json", "model.py", "forecast_wrapper.py")
# repo / pod-dir suffix prefix of a lane-C leg (``-c<target>-u<uid>``)
SUFFIX_PREFIX = "-c"


class RecipeError(ValueError):
    """The miner's ``recipe.json`` is missing, unreadable or out of bounds —
    a miner fault: the lane-C entry ends (never retried on the miner's key)."""


@dataclass(frozen=True)
class LegRecipe:
    """The fields a fine-tune leg trains under (the shared ``Recipe``'s)."""

    target: str
    lr_scale: float
    steps_fraction: float
    warmup_fraction: float
    schedule: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))


def _lane_c_cfg(cfg):
    """``[lane_c]`` whether handed the ChainConfig or the section itself."""
    return getattr(cfg, "lane_c", cfg)


def parse_leg_recipe(obj, cfg, *, target: str | None = None) -> LegRecipe:
    """Validate a recipe object with the SHARED rule
    (:func:`cascade.shared.lane_c.parse_recipe` against ``[lane_c]``) and
    return the leg's view of it. ``cfg`` is the ChainConfig (or its
    ``[lane_c]``); ``target`` (the committed target) must match the recipe's
    when given. Raises :class:`RecipeError` on anything the rule refuses."""
    from ..shared.lane_c import parse_recipe

    try:
        r = parse_recipe(obj, _lane_c_cfg(cfg), target=target)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise RecipeError(f"invalid recipe.json: {e}") from e
    if r.schedule not in SCHEDULE_MAP:
        raise RecipeError(f"invalid recipe.json: schedule {r.schedule!r} must be one of "
                          f"{RECIPE_SCHEDULES}")
    return LegRecipe(target=str(r.target), lr_scale=float(r.lr_scale),
                     steps_fraction=float(r.steps_fraction),
                     warmup_fraction=float(r.warmup_fraction), schedule=str(r.schedule))


def read_recipe(tree_dir: Path | str, cfg, *, target: str | None = None) -> LegRecipe:
    """``recipe.json`` at the root of a fetched ref tree, validated
    (:func:`parse_leg_recipe`, against the committed ``target`` when given).
    Raises :class:`RecipeError` (miner fault)."""
    path = Path(tree_dir) / RECIPE_FILE
    if not path.is_file():
        raise RecipeError(f"the ref's tree has no {RECIPE_FILE} at its root")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as e:
        raise RecipeError(f"{RECIPE_FILE} is not valid JSON: {e}") from e
    return parse_leg_recipe(obj, cfg, target=target)


def recipe_from_json(text: str, cfg) -> LegRecipe:
    """The worker's side: the recipe JSON the orchestrator passed, re-validated
    (defence in depth — the pod enforces the same bounds)."""
    try:
        obj = json.loads(text)
    except ValueError as e:
        raise RecipeError(f"--recipe-json is not valid JSON: {e}") from e
    return parse_leg_recipe(obj, cfg)


def _set(contract, **kw):
    names = {f.name for f in dataclasses.fields(contract)}
    return dataclasses.replace(contract, **{k: v for k, v in kw.items() if k in names})


def finetune_contract(contract, recipe: LegRecipe, *, leg_hours: float,
                      guard_factor: float = 1.0, guard_floor_seconds: int = 0,
                      max_lr_scale: float | None = None,
                      max_steps_fraction: float | None = None):
    """The fine-tune leg's contract (see the module docstring). The budget is
    ``leg_hours`` of wall (the guard) and ``steps_fraction × leg_hours`` of
    reference-throughput tokens; architecture fields are left alone."""
    if recipe.schedule not in SCHEDULE_MAP:
        raise ValueError(f"unknown recipe schedule {recipe.schedule!r}")
    if max_lr_scale is not None and not 0.0 < recipe.lr_scale <= float(max_lr_scale):
        raise ValueError(f"lr_scale {recipe.lr_scale} outside (0, {max_lr_scale}]")
    if max_steps_fraction is not None and not (
            0.0 < recipe.steps_fraction <= float(max_steps_fraction)):
        raise ValueError(f"steps_fraction {recipe.steps_fraction} outside "
                         f"(0, {max_steps_fraction}]")
    if not 0.0 <= recipe.warmup_fraction < 1.0:
        raise ValueError(f"warmup_fraction {recipe.warmup_fraction} outside [0, 1)")
    c = contract.for_hours(float(leg_hours), guard_factor=guard_factor,
                           guard_floor_seconds=int(guard_floor_seconds))
    return _set(
        c,
        target_train_hours=float(leg_hours) * float(recipe.steps_fraction),
        base_lr=float(contract.base_lr) * float(recipe.lr_scale),
        lr_schedule=SCHEDULE_MAP[recipe.schedule],
        warmup_fraction=float(recipe.warmup_fraction),
        # the exported weights ARE the fine-tune: no finished-form branch,
        # no lineage continuity knobs
        anneal_fraction=0.0, ema_decay=0.0, rewarmup_fraction=0.0, warm_lr_scale=1.0,
        power_ema_sigma_rels=(), power_ema_eval=-1,
    )


def leg_contract(cfg, contract, recipe: LegRecipe):
    """:func:`finetune_contract` with the chain config's ``[lane_c]`` budget
    and bounds and ``[round]`` guard knobs — the ONE place both the
    orchestrator (corpus sizing, harvest verify) and the worker derive it."""
    lc = _lane_c_cfg(cfg)
    rnd = getattr(cfg, "round", None)
    return finetune_contract(
        contract, recipe, leg_hours=float(getattr(lc, "leg_hours", 1.0)),
        guard_factor=float(getattr(rnd, "heat_guard_factor", 1.0) or 1.0),
        guard_floor_seconds=int(getattr(rnd, "heat_guard_floor_seconds", 0) or 0),
        max_lr_scale=float(getattr(lc, "max_lr_scale", 1.0)),
        max_steps_fraction=float(getattr(lc, "max_steps_fraction", 1.0)))


def stage_finetune_base(base_dir: Path | str, out_dir: Path | str) -> Path:
    """Stage the base checkpoint as a weights-only warm start in ``out_dir``
    (see the module docstring). Raises FileNotFoundError when the base has
    no scored weights."""
    base, out = Path(base_dir), Path(out_dir)
    if not (base / "weights.safetensors").is_file():
        raise FileNotFoundError(f"fine-tune base {base} has no weights.safetensors")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in _BASE_FILES:
        if (base / name).is_file():
            shutil.copyfile(base / name, out / name)
    return out


def leg_suffix(target: str, uid: int) -> str:
    """The repo / pod-dir suffix of a lane-C leg: ``-c<target>-u<uid>`` with
    the target's ``:`` made path-safe."""
    safe = str(target).replace(":", "_")
    return f"{SUFFIX_PREFIX}{safe}-u{int(uid)}"


__all__ = [
    "LegRecipe", "RECIPE_FILE", "RECIPE_SCHEDULES", "RecipeError", "SCHEDULE_MAP",
    "SUFFIX_PREFIX", "finetune_contract", "leg_contract", "leg_suffix", "parse_leg_recipe",
    "read_recipe", "recipe_from_json", "stage_finetune_base",
]
