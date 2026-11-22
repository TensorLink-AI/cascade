"""W8: ``lane_pool_shares`` returns lane D's share too (C3/D48), and every
caller uses the new signature (a consensus signature at every caller).

The new signature (README "Module contract"):

    lane_pool_shares(lane_b_cfg, lane_c_cfg=None, lane_d_cfg=None) -> LanePoolShares

``LanePoolShares`` is a ``typing.NamedTuple`` exported by
``cascade.shared.lane_c`` with the fields ``("b", "c", "d")``, each a fraction
of the total; lane A's pool is ``1 - b - c - d`` (before lane B's
``remainder = "lane_a"`` refund). On main it returns a ``(B, C)`` pair.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from .helpers import REPO, lane_b_cfg, lane_c_cfg, lane_d_cfg

# Where callers are enumerated: the package, the scripts and the unit tests.
CALLER_ROOTS = ("cascade", "scripts", "tests/unit")
# The callers on main (plan W8 row: emission.py:361, lane_c.py:605).
KNOWN_CASCADE_CALLERS = {"cascade/shared/emission.py", "cascade/shared/lane_c.py"}


def _shares(*args, **kw):
    from cascade.shared.lane_c import lane_pool_shares

    params = list(inspect.signature(lane_pool_shares).parameters)
    assert params[:3] == ["lane_b_cfg", "lane_c_cfg", "lane_d_cfg"], (
        f"lane_pool_shares{inspect.signature(lane_pool_shares)} takes no lane_d_cfg")
    return lane_pool_shares(*args, **kw)


def _approx(got, want):
    assert len(got) == len(want), f"{tuple(got)!r} is not a (B, C, D) triple"
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-15), (tuple(got), want)


def test_lane_pool_shares_returns_named_b_c_d():
    from cascade.shared import lane_c
    from cascade.shared.lane_c import LanePoolShares

    assert issubclass(LanePoolShares, tuple)
    assert LanePoolShares._fields == ("b", "c", "d")
    assert "LanePoolShares" in lane_c.__all__
    got = _shares(lane_b_cfg(), lane_c_cfg(), lane_d_cfg())
    assert isinstance(got, LanePoolShares) and len(got) == 3
    assert (got.b, got.c, got.d) == tuple(got)


def test_lane_pool_shares_takes_lane_d_as_third_argument():
    d = lane_d_cfg()
    assert tuple(_shares(None, None, d)) == tuple(_shares(None, None, lane_d_cfg=d))
    assert tuple(_shares(lane_b_cfg(), lane_c_cfg(), d)) == tuple(
        _shares(lane_b_cfg(), lane_c_cfg(), lane_d_cfg=d))


@pytest.mark.parametrize("case, want", [
    # without lane D: today's (B, C), D = 0
    ("nothing", (0.0, 0.0, 0.0)),
    ("b50", (0.5, 0.0, 0.0)),
    ("b25_c25_from_a", (0.25, 0.25, 0.0)),
    ("b50_c25_from_b", (0.25, 0.25, 0.0)),
    # D1/D48: lane D is 0.25 of the total, from lane A's pot
    ("d25", (0.0, 0.0, 0.25)),
    ("b25_d25", (0.25, 0.0, 0.25)),
    ("b25_c25_d25", (0.25, 0.25, 0.25)),          # A 25 / B 25 / C 25 / D 25
    ("b50_c25_from_b_d25", (0.25, 0.25, 0.25)),   # legacy C-from-B; D still from A
    ("d0", (0.25, 0.25, 0.0)),                     # armed, share 0 (inv. 9)
    # capped at what lane A holds
    ("b50_c50_d25", (0.5, 0.5, 0.0)),
    ("b50_c25_d50", (0.5, 0.25, 0.25)),
    # legacy takes_from = "lane_b" (the deployed default until the S6 overlay)
    ("b50_d25_from_b", (0.25, 0.0, 0.25)),
])
def test_lane_pool_shares_golden(case, want):
    cfgs = {
        "nothing": (None, None, None),
        "b50": (lane_b_cfg(0.5), None, None),
        "b25_c25_from_a": (lane_b_cfg(), lane_c_cfg(), None),
        "b50_c25_from_b": (lane_b_cfg(0.5), lane_c_cfg(takes_from="lane_b"), None),
        "d25": (None, None, lane_d_cfg()),
        "b25_d25": (lane_b_cfg(), None, lane_d_cfg()),
        "b25_c25_d25": (lane_b_cfg(), lane_c_cfg(), lane_d_cfg()),
        "b50_c25_from_b_d25": (lane_b_cfg(0.5), lane_c_cfg(takes_from="lane_b"), lane_d_cfg()),
        "d0": (lane_b_cfg(), lane_c_cfg(), lane_d_cfg(share=0.0)),
        "b50_c50_d25": (lane_b_cfg(0.5), lane_c_cfg(0.5), lane_d_cfg()),
        "b50_c25_d50": (lane_b_cfg(0.5), lane_c_cfg(0.25), lane_d_cfg(share=0.5)),
        "b50_d25_from_b": (lane_b_cfg(0.5), None, lane_d_cfg(takes_from="lane_b")),
    }
    _approx(_shares(*cfgs[case]), want)


def test_lane_pool_shares_without_lane_d_keeps_todays_b_c_exactly():
    """Invariant 1 at the signature: no lane D => D is exactly 0.0 and (B, C)
    are bit-identical to today's (values recorded on main a6204c0)."""
    for args, b, c in (((None, None), 0.0, 0.0),
                       ((lane_b_cfg(0.5), None), 0.5, 0.0),
                       ((lane_b_cfg(), lane_c_cfg()), 0.25, 0.25),
                       ((lane_b_cfg(0.5), lane_c_cfg(takes_from="lane_b")), 0.25, 0.25)):
        got = _shares(*args)
        assert len(got) == 3, f"{tuple(got)!r} is not a (B, C, D) triple"
        assert got[0] == b and got[1] == c and got[2] == 0.0
        assert tuple(_shares(*args, None)) == tuple(got)


def test_lane_c_shares_keeps_its_b_c_pair():
    """``lane_c_shares`` (a caller, ``lane_c.py:605``) keeps returning the
    plain ``(B, C)`` pair its callers (validator/lane_b.py, audit/checks.py)
    and the receipt's share records use: goldens copied from
    tests/unit/test_lane_c_emission.py."""
    from cascade.shared.lane_c import lane_c_shares

    lb = lane_b_cfg(0.5)
    lc = replace(lane_c_cfg(takes_from="lane_b"), from_block=7200)
    assert lane_c_shares(lb, lc, 7200) == (0.5, 0.0)
    assert lane_c_shares(lb, lc, 7201) == (0.25, 0.25)
    assert lane_c_shares(lb, replace(lc, share=0.9), 10**9) == (0.0, 0.5)
    got = lane_c_shares(lane_b_cfg(), replace(lc, takes_from="lane_a"), 7201)
    assert type(got) is tuple and got == (0.25, 0.25)


# ── every caller (AST) ──────────────────────────────────────────────────────


def _calls(tree: ast.AST):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name == "lane_pool_shares":
                yield node, parents


def _enclosing_function(node, parents) -> str:
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
    return ""


def _caller_sites():
    sites = []
    for root in CALLER_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for call, parents in _calls(tree):
                sites.append((rel, call, parents))
    return sites


def test_every_lane_pool_shares_caller_uses_the_b_c_d_signature():
    sites = _caller_sites()
    files = {rel for rel, _, _ in sites}
    assert files >= KNOWN_CASCADE_CALLERS, f"callers not found: {KNOWN_CASCADE_CALLERS - files}"
    bad = []
    for rel, call, parents in sites:
        parent = parents.get(call)
        targets = []
        if isinstance(parent, ast.Assign):
            targets = parent.targets
        elif isinstance(parent, ast.AnnAssign | ast.NamedExpr):
            targets = [parent.target]
        elif isinstance(parent, ast.comprehension) and parent.iter is call:
            targets = []
        for t in targets:
            if isinstance(t, ast.Tuple | ast.List) and len(t.elts) != 3:
                bad.append(f"{rel}:{call.lineno} unpacks lane_pool_shares into {len(t.elts)}")
        if (isinstance(parent, ast.Return) and rel == "cascade/shared/lane_c.py"
                and _enclosing_function(call, parents) == "lane_c_shares"):
            bad.append(f"{rel}:{call.lineno} lane_c_shares returns lane_pool_shares(...) "
                       "as is; it must return its (B, C) pair")
    assert not bad, "\n".join(bad)


def test_lane_parts_passes_lane_d_to_lane_pool_shares():
    """``lane_parts`` takes lane D's share from ``lane_pool_shares`` (the one
    allocation rule), passing its ``lane_d_cfg`` as the third argument."""
    rel = "cascade/shared/emission.py"
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    found = []
    for call, parents in _calls(tree):
        if _enclosing_function(call, parents) != "lane_parts":
            continue
        names = [a.id for a in call.args if isinstance(a, ast.Name)]
        names += [k.value.id for k in call.keywords
                  if k.arg == "lane_d_cfg" and isinstance(k.value, ast.Name)]
        found.append((call.lineno, len(call.args) >= 3 or any(
            k.arg == "lane_d_cfg" for k in call.keywords), "lane_d_cfg" in names))
    assert found, "lane_parts no longer calls lane_pool_shares"
    assert all(n_ok and named for _, n_ok, named in found), (
        f"lane_parts calls lane_pool_shares without lane_d_cfg: {found}")


def test_caller_roots_exist():
    for root in CALLER_ROOTS:
        assert Path(REPO / root).is_dir(), root
