"""Shared spec and helpers for the WB14 acceptance tests (declared rows, LB2).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB14 row (§3.0
lane B track), §12b.2 LB2, the T-rows TB13 / TB15 / TB15b / TB21 (§4), the
build rules (§3.1) and the DEC-CA-0071 amendment rev4 (Appendix C).

Nothing from the PATHS modules is imported at module level. The model's
expert geometry is read from ``cascade/trainer/pyramid_model.py`` by AST,
because that module imports torch at top level and torch must never load in
these tests (production box rule; the implementation must not need it either).
"""

from __future__ import annotations

import ast
import importlib.abc
import json
import math
import sys
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
PYRAMID = REPO / "cascade" / "trainer" / "pyramid_model.py"
WORKER = REPO / "cascade" / "trainer" / "worker.py"
FUNDING_MAIN = REPO / "cascade" / "funding" / "main.py"
CHAIN_TOML = REPO / "chain.toml"

MODULE = "cascade.trainer.declared_rows"
HEADER = "X-Lane-B-Declared-Rows"

OUT_OF_RANGE = "declared_rows_out_of_range"
BAD_LAYER = "declared_rows_bad_layer"
COUNT = "declared_rows_count"
MALFORMED = "declared_rows_malformed"
CODES = (OUT_OF_RANGE, BAD_LAYER, COUNT, MALFORMED)

# D39 recommended fractions (WB10 defaults); the tests read the live values
# from chain.toml and only use these to cross-check the plan's worked clamps.
PLAN_MIN_FRAC = 0.03125
PLAN_MAX_FRAC = 0.1484375
PLAN_CLAMPS = {128: (4, 19), 256: (8, 38), 512: (16, 76)}


# ── torch guard ──────────────────────────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB14: '{name}' must not be imported (declared-rows parsing is "
                "pure Python; pyramid_model.py is read from source)", name=name)
        return None


@contextmanager
def torch_blocked():
    """Refuse any torch import while the block runs (a lazy import inside a
    call is caught too). A torch already in ``sys.modules`` is not unloaded."""
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def dr():
    """The WB14 module, imported with torch blocked."""
    import importlib

    with torch_blocked():
        return importlib.import_module(MODULE)


# ── model geometry, from pyramid_model.py source ─────────────────────────────


@cache
def _config_defaults() -> dict:
    tree = ast.parse(PYRAMID.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CascadeMoEConfig":
            out = {}
            for st in node.body:
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                        and st.value is not None):
                    try:
                        out[st.target.id] = ast.literal_eval(st.value)
                    except ValueError:
                        continue
            return out
    raise AssertionError("CascadeMoEConfig not found in pyramid_model.py")


def experts_per_layer() -> tuple[int, ...]:
    return tuple(_config_defaults()["experts_per_layer"])


def moe_blocks() -> tuple[int, ...]:
    return tuple(_config_defaults()["moe_blocks"])


# ── clamps ───────────────────────────────────────────────────────────────────


@cache
def live_fracs() -> tuple[float, float]:
    """``[lane_b] min_experts_frac`` / ``max_experts_frac`` from the shipped
    chain.toml through the real loader (WB10)."""
    from cascade.shared.config import load_chain_config

    lb = load_chain_config(CHAIN_TOML).lane_b
    return float(lb.min_experts_frac), float(lb.max_experts_frac)


def live_lane_b():
    from cascade.shared.config import load_chain_config

    return load_chain_config(CHAIN_TOML).lane_b


def clamp(n_experts: int, min_frac: float, max_frac: float) -> tuple[int, int]:
    """LB2: ``min = ceil(frac_min * E)``, ``max = floor(frac_max * E)``."""
    return math.ceil(min_frac * n_experts), math.floor(max_frac * n_experts)


def rows(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def parse(raw, *, epl=None, fracs=None):
    """``parse_declared_rows`` with the model geometry and the live fractions
    unless overridden."""
    lo, hi = fracs if fracs is not None else live_fracs()
    mod = dr()
    with torch_blocked():
        return mod.parse_declared_rows(
            raw, experts_per_layer=tuple(epl if epl is not None else experts_per_layer()),
            min_frac=lo, max_frac=hi)


def refusal(raw, *, epl=None, fracs=None) -> str:
    """The typed refusal code of ``raw`` (AssertionError if accepted or if
    the error is not the module's ``DeclaredRowsError``)."""
    mod = dr()
    err = getattr(mod, "DeclaredRowsError", None)
    assert isinstance(err, type) and issubclass(err, ValueError), (
        "declared_rows.DeclaredRowsError must be a ValueError subclass")
    try:
        got = parse(raw, epl=epl, fracs=fracs)
    except err as e:
        code = getattr(e, "code", None)
        assert code in CODES, f"unknown refusal code {code!r}"
        return code
    raise AssertionError(f"declared rows {raw!r} were accepted as {got!r}")


def nonempty(sel: dict) -> dict:
    """The layers with at least one row; every other present layer must be []."""
    for k, v in sel.items():
        assert isinstance(k, int) and not isinstance(k, bool), f"key {k!r} is not an int"
        assert isinstance(v, list), f"layer {k} value {v!r} is not a list"
    return {k: v for k, v in sel.items() if v}


# ── the worker's --selection-json parse (unchanged on-pod code) ──────────────


@cache
def _worker_selection_expr() -> ast.Expression:
    """The exact expression ``worker.main`` assigns to ``expert_selection``
    (worker.py:226-227). worker.py runs on the pod, so WB14 must not change
    it (plan §3: "declared rows need no image change"); the tests evaluate
    the deployed expression itself."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for st in ast.walk(node):
                if (isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                        and st.targets[0].id == "expert_selection"
                        and isinstance(st.value, ast.DictComp)):
                    return ast.Expression(body=st.value)
    raise AssertionError("worker.main no longer parses --selection-json into expert_selection")


def worker_parse_selection(text: str) -> dict[int, list[int]]:
    code = compile(_worker_selection_expr(), str(WORKER), "eval")
    return eval(code, {"json": json, "int": int},  # noqa: S307 — repo source, not input
                {"args": SimpleNamespace(selection_json=text)})


# ── funding intake ───────────────────────────────────────────────────────────

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/gen@sha256:" + "a" * 64
SLOTS = ("energy", "web", "horizon:64")
API_KEY = "sk-live"


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def intake(tmp_path, *, lane_b_cfg=None, sig=False, verify=None):
    """A real :class:`FundingIntake` with a real lane B queue on disk. Only
    the chain (the reveal oracle) and the signature verifier are stand-ins."""
    import inspect

    from cascade.funding.intake import FundingIntake
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    assert "lane_b_cfg" in inspect.signature(FundingIntake.__init__).parameters, (
        "FundingIntake takes lane_b_cfg (the loaded [lane_b] config: the row clamp fractions)")
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    qpath = lane_b_queue_path(tmp_path / "queue.json")
    lane_b = LaneBQueue(qpath, clock=clock)
    it = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: None, require_signature=sig,
        verify=verify or (lambda hk, msg, s: False), clock=clock,
        lane_b_queue=lane_b, resolve_lane_b_reveal=lambda hk, t, r: 700,
        lane_b_targets=SLOTS, lane_b_max_targets=2,
        lane_b_cfg=lane_b_cfg if lane_b_cfg is not None else live_lane_b())
    return SimpleNamespace(intake=it, lane_a=lane_a, lane_b=lane_b, clock=clock, qpath=qpath)


def headers(**kw) -> dict:
    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lane-B-Targets": "energy",
         "X-Lium-Api-Key": API_KEY}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def with_rows(sel, **kw) -> dict:
    return headers(**{HEADER: sel if isinstance(sel, str) else json.dumps(sel)}, **kw)


def queue_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
