"""Shared helpers for the WB16 acceptance tests (D40 option (a): router rows
frozen in a lane B expert leg).

From docs/plans/LANE_D_PLAN.md: the WB16 row (§3.0 lane B track), §12b.2
LB2 "Router rows [DECIDE D40]", the D40 row of the decision table (decided
(a), owner default 2026-11-21) and §3.2 "Services touched" ((b) is a shared
worker-image repin; (a) needs no image change).

Box-side helpers read source by AST or text and never import torch.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPERT_TRAINING = REPO / "cascade" / "trainer" / "expert_training.py"
WORKER = REPO / "cascade" / "trainer" / "worker.py"


@cache
def module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Every call whose callee is ``func_name`` or ``<x>.func_name``."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == func_name) or \
                    (isinstance(f, ast.Attribute) and f.attr == func_name):
                out.append(n)
    return out


def worker_flags() -> list[str]:
    """The ``--flags`` worker.py registers with argparse."""
    flags = []
    for call in calls(module_tree(WORKER), "add_argument"):
        for a in call.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                flags.append(a.value)
    return flags


def grep_cascade(needle: str) -> list[str]:
    """Files under cascade/ (code and JSON schemas) containing ``needle``."""
    hits = []
    for p in sorted((REPO / "cascade").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".json", ".toml") \
                and needle in p.read_text(encoding="utf-8", errors="replace"):
            hits.append(p.relative_to(REPO).as_posix())
    return hits


def router_keys(state: dict) -> list[str]:
    """State-dict keys of the router weights and the routing biases."""
    return sorted(k for k in state if ".router." in k or k.endswith("route_bias"))
