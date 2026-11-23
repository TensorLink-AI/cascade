"""Shared spec and helpers for the WB25 acceptance tests (multi-GPU lane B
expert legs).

From docs/plans/LANE_D_PLAN.md: the WB25 row (§3.0 lane B track, rev6.8,
owner 2026-11-23) and the rev6.8 changelog. The contract is in README.md.

Nothing from a PATHS module is imported at module level, and torch is never
imported here: the AST helpers run on the 7 GB box, and the torch helpers
import torch inside their bodies (they are called only from tests marked
``torch``, which run in CI).
"""

from __future__ import annotations

import ast
import sys
import types
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"
TRAINER = REPO / "cascade" / "trainer" / "pyramid_trainer.py"
DP = REPO / "cascade" / "trainer" / "pyramid_dp.py"
PYPROJECT = REPO / "pyproject.toml"
UV_LOCK = REPO / "uv.lock"

# main @ cbe4c79, byte for byte: the lane A oracle (README "Lane A unchanged")
SNAPSHOT_DIR = "main_snapshot"
SNAPSHOT_TRAINER = "pyramid_trainer.py.txt"
SNAPSHOT_DP = "pyramid_dp.py.txt"
SNAPSHOT_SHA256 = {
    SNAPSHOT_TRAINER: "4521dda15fccc06020c5c4cc0fd7efd26f660f073ee535b02d207d8d2f0dad79",
    SNAPSHOT_DP: "6842716ee9725b96b09942387d482d801011a5585fcab613dcd84b4ddc3afe8b",
}
ORACLE_TRAINER = "cascade.trainer._wb25_main_pyramid_trainer"
ORACLE_DP = "cascade.trainer._wb25_main_pyramid_dp"

GPU_N = "gpu_n"
# metrics that measure wall time, not numerics
TIMING_METRICS = frozenset({"train_seconds", "throughput_tokens_per_s", "data_wait_s",
                            "deadline_hit", "corpus_exhausted"})

# the 2-rank vs 1-rank tolerance on the selected rows (README)
RTOL, ATOL = 1e-4, 1e-6


# ── AST (box-safe) ──────────────────────────────────────────────────────────


@cache
def module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found at module level")


def callee(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and callee(n) == name]


def names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def assignments(fn: ast.AST, target: str) -> list[ast.AST]:
    """The value of every plain or annotated assignment to ``target``."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == target for t in n.targets):
                out.append(n.value)
        elif (isinstance(n, ast.AnnAssign | ast.AugAssign) and isinstance(n.target, ast.Name)
              and n.target.id == target and n.value is not None):
            out.append(n.value)
    return out


def parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def is_conditional(node: ast.AST, up: dict[ast.AST, ast.AST], stop: ast.AST) -> bool:
    """Does ``node`` sit inside an ``if`` / conditional expression below ``stop``?"""
    cur = up.get(node)
    while cur is not None and cur is not stop:
        if isinstance(cur, ast.If | ast.IfExp):
            return True
        cur = up.get(cur)
    return False


def guarded_by_slices_none(node: ast.AST, up: dict[ast.AST, ast.AST], stop: ast.AST) -> bool:
    """Is ``node`` in the body of an ``if slices is None:`` (lane A) branch?"""
    cur, child = up.get(node), node
    while cur is not None and cur is not stop:
        if isinstance(cur, ast.If) and child in cur.body:
            t = cur.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "slices" and len(t.ops) == 1
                    and isinstance(t.ops[0], ast.Is)
                    and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value is None):
                return True
        child, cur = cur, up.get(cur)
    return False


def plan_row(wp: str) -> list[str] | None:
    """The cells of ``| <wp> | ... |`` in the plan, or None."""
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| {wp} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")]
    return None


# ── torch helpers (called only from tests marked torch) ─────────────────────


def tiny_contract(**overrides):
    from dataclasses import replace

    from cascade.shared.config import parse_chain_config
    from scripts.pyramid_config import local_config

    c = parse_chain_config(local_config("2m")).training
    # micro_batches=4 so both ranks of a 2-rank gang own micro-batches
    return replace(c, **{"training_steps": 3, "batch_size": 8, "micro_batches": 4,
                         "max_train_seconds": 300, **overrides})


def data(rng_seed: int = 84):
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def trainer():
    from cascade.trainer.toto2_trainer import Toto2Trainer

    return Toto2Trainer(device="cpu")


def train(path, contract, *, seed: int = 1, rng_seed: int = 84, **kwargs):
    """A leg through the production entry point (Toto2Trainer.train)."""
    return trainer().train(iter(data(rng_seed)), contract, training_seed=seed,
                           token_budget=contract.train_tokens, out_dir=path, **kwargs)


def selection_of(king_dir) -> dict[int, list[int]]:
    """The four most-loaded experts per MoE layer (they receive tokens)."""
    import torch

    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(data(84)[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    return {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}


def weights(d, name: str = "weights.safetensors") -> dict:
    from safetensors.torch import load_file

    return load_file(str(Path(d) / name))


def moe_names(king_dir) -> list[tuple[str, str]]:
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_model import moe_param_names

    return moe_param_names(load_king(king_dir))


def router_keys(state: dict) -> list[str]:
    return sorted(k for k in state if ".router." in k or k.endswith("route_bias"))


def leaf_shapes(king_state: dict, selection: dict, names) -> list[tuple[int, ...]]:
    """Shapes of the compact trainable leaves ExpertSlices builds."""
    out = []
    for li, pair in enumerate(names):
        rows = selection.get(li, [])
        if rows:
            for n in pair:
                out.append((len(rows), *king_state[n].shape[1:]))
    return out


def file_bytes(d) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir()) if p.is_file()}


def numeric_metrics(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in TIMING_METRICS}


def load_main_oracle():
    """main's pyramid_trainer + pyramid_dp (cbe4c79) as two modules inside
    the live ``cascade.trainer`` package, wired to each other. Every other
    import (model, optimizer, fast kernels, ExpertSlices, ...) is the live
    one: those files are not WB25's PATHS."""
    import hashlib
    import importlib

    pkg = importlib.import_module("cascade.trainer")
    texts = {}
    for name, digest in SNAPSHOT_SHA256.items():
        raw = (HERE / SNAPSHOT_DIR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, f"{name}: snapshot changed"
        texts[name] = raw.decode("utf-8")
    trainer_src = texts[SNAPSHOT_TRAINER].replace("from .pyramid_dp import",
                                                  "from ._wb25_main_pyramid_dp import")
    dp_src = texts[SNAPSHOT_DP].replace("from .pyramid_trainer import",
                                        "from ._wb25_main_pyramid_trainer import")
    assert "from ._wb25_main_pyramid_dp import" in trainer_src
    assert "from ._wb25_main_pyramid_trainer import" in dp_src
    # register both before running either (each imports the other lazily)
    pending = []
    for full, src, fname in ((ORACLE_DP, dp_src, SNAPSHOT_DP),
                             (ORACLE_TRAINER, trainer_src, SNAPSHOT_TRAINER)):
        mod = types.ModuleType(full)
        mod.__package__ = pkg.__name__
        mod.__file__ = str(HERE / SNAPSHOT_DIR / fname)
        sys.modules[full] = mod
        pending.append((mod, src))
    for mod, src in pending:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return pending[1][0]


def unload_main_oracle() -> None:
    for full in (ORACLE_TRAINER, ORACLE_DP):
        sys.modules.pop(full, None)
