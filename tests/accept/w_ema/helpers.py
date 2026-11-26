"""Shared spec and loaders for the W-EMA acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev5 W-EMA row, §8 "WP W-EMA"):

- "Score the raw cooled endpoint on every leg (lanes A and B):
  `power_ema_active` → raw for warm-start legs; keep writing
  `weights_ema005`/`weights_ema010` unscored; main-validator EMA telemetry
  off the critical path (§8)."
- §8: "`power_ema_active` is changed so warm-start legs behave like
  random-init legs do today: `weights.safetensors` is the raw endpoint, and
  `weights_ema005` / `weights_ema010` are written alongside, unscored."
- §8 Tests: "a warm-start leg's scored file is byte-identical to the raw
  endpoint; both EMA files are present; a lane B leg from an EMA-marked king
  still scores raw."

``pyramid_trainer`` imports torch at module level, and this production box
never loads torch. So the box-side tests evaluate the module's pure gate
(``power_ema_active``) and its constants FROM SOURCE: the module's literal
constants and its function definitions are executed in a namespace with only
the stdlib (no import of the module, no torch). A def that is never called
runs no torch code. The torch-marked tests (CI only) train real tiny legs.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import time
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAINER_MODULE = "cascade.trainer.pyramid_trainer"

EMA005 = "weights_ema005.safetensors"
EMA010 = "weights_ema010.safetensors"
RAW = "weights_raw.safetensors"
MARKER_KEY = "power_ema"
# The marker an EMA-scored checkpoint carries in config.json today
# (save_checkpoint, pyramid_trainer.py:203-205).
MARKER = {"sigma_rels": [0.05, 0.10], "scored_sigma_rel": 0.10}
# The validator's scored checkpoint files (private_worker.CHECKPOINT_FILES).
SCORED_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")


@cache
def trainer_path() -> Path:
    """The source file ``import cascade.trainer.pyramid_trainer`` would load
    (located without executing it; ``cascade.trainer`` imports no torch)."""
    spec = importlib.util.find_spec(TRAINER_MODULE)
    assert spec is not None and spec.origin, f"{TRAINER_MODULE} not found"
    return Path(spec.origin)


@cache
def trainer_tree() -> ast.Module:
    path = trainer_path()
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal(node: ast.AST):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False, None


@cache
def constants() -> dict:
    """Every module-level ``NAME = <literal>`` of pyramid_trainer.py."""
    out = {}
    for node in trainer_tree().body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            ok, value = _literal(node.value)
            if ok:
                out[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            ok, value = _literal(node.value)
            if ok:
                out[node.target.id] = value
    return out


def namespace() -> dict:
    """The module's literal constants and every module-level def, executed
    with the stdlib names the module imports (json, math, hashlib, time,
    Path). Nothing is imported from the module itself and torch is never
    loaded; calling a def that needs torch would raise NameError."""
    ns: dict = {"__name__": "w_ema_source_view", "json": json, "math": math,
                "hashlib": hashlib, "time": time, "Path": Path}
    ns.update(constants())
    # the future import above binds `annotations` to the __future__ feature:
    # its flag keeps every annotation in the source view unevaluated
    flags = annotations.compiler_flag
    for node in trainer_tree().body:
        if not isinstance(node, ast.FunctionDef):
            continue
        code = compile(ast.Module(body=[node], type_ignores=[]), str(trainer_path()), "exec",
                       flags=flags, dont_inherit=True)
        try:
            exec(code, ns)  # noqa: S102 — repo source, not miner bytes
        except NameError:
            # a decorator or default value that needs torch: that def is not
            # part of the pure surface these tests read
            continue
    return ns


def function_names() -> set[str]:
    return {n.name for n in trainer_tree().body if isinstance(n, ast.FunctionDef)}


def write_king(path: Path, *, marked: bool) -> Path:
    """A king checkpoint directory as the gate reads it (config.json only):
    ``marked`` = it carries today's EMA-scored marker."""
    path.mkdir(parents=True, exist_ok=True)
    meta = {"moe": {}, "base_arch": "cascade_pyramid"}
    if marked:
        meta[MARKER_KEY] = dict(MARKER)
    (path / "config.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return path


def mark_config(ckpt: Path) -> None:
    """Make a trained checkpoint an EMA-marked king (the config.json bytes
    save_checkpoint writes on an EMA-scored leg: sorted keys plus the
    marker)."""
    cfg_path = Path(ckpt) / "config.json"
    meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    meta[MARKER_KEY] = dict(MARKER)
    cfg_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def file_bytes(path: Path) -> dict[str, bytes]:
    return {f.name: f.read_bytes() for f in sorted(Path(path).iterdir()) if f.is_file()}


def config_of(path: Path) -> dict:
    return json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
