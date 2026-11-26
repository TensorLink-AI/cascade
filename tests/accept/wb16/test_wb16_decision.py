"""WB16 on the box (no torch): D40 decided option (a), router rows frozen,
is the only router mode the release carries, and nothing on the worker image
offers an unfreeze. README.md has the contract.

Torch is never imported here: expert_training.py, worker.py and the trainers
are read by AST/text only. The torch-marked behaviour tests (CI only) are in
test_wb16_frozen_torch.py.
"""

from __future__ import annotations

import ast
import tomllib

import pytest

from . import helpers as h


def test_d40_option_a_is_the_only_router_rows_value():
    from cascade.shared.config.lane_b import LANE_B_ROUTER_ROWS

    assert tuple(LANE_B_ROUTER_ROWS) == ("frozen",)


@pytest.mark.parametrize("name", ["chain.toml", "chain.testnet.toml"])
def test_shipped_configs_load_router_rows_frozen(name):
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(h.REPO / name)
    assert cfg.lane_b.router_rows == "frozen"


@pytest.mark.parametrize("value", ["unfrozen", "flexolmo", "b", "declared"])
def test_an_unfreeze_mode_is_refused_at_load(value):
    """Option (b) is not built: the loader refuses any other mode, so an
    overlay can't switch a leg to an unfrozen router."""
    from cascade.shared.config import load_chain_config

    raw = tomllib.loads((h.REPO / "chain.toml").read_text(encoding="utf-8"))
    raw.setdefault("lane_b", {})["router_rows"] = value
    with pytest.raises(ValueError, match="router_rows"):
        load_chain_config(raw=raw)


def test_expert_slices_makes_only_expert_rows_trainable():
    """``ExpertSlices`` freezes every model parameter and creates trainable
    leaves only from ``w_in`` / ``w_out`` rows: never from the router weight
    or the routing bias (expert_training.py, the D40 (a) backend)."""
    tree = h.module_tree(h.EXPERT_TRAINING)
    init = h.method(tree, "ExpertSlices", "__init__")
    src = ast.unparse(init)
    assert "requires_grad_(False)" in src, "ExpertSlices must freeze every parameter"
    leaf_tuples = [n for n in ast.walk(init) if isinstance(n, ast.Tuple)
                   and all(isinstance(e, ast.Constant) for e in n.elts) and n.elts]
    names = {e.value for t in leaf_tuples for e in t.elts}
    assert {"w_in", "w_out"} <= names, names
    assert not {n for n in names if "router" in str(n) or "bias" in str(n)}, names
    for call in h.calls(tree, "Parameter"):
        text = ast.unparse(call)
        assert "w_in" in text or "w_out" in text, text
        assert "router" not in text and "route_bias" not in text, text


def test_nothing_in_the_backend_unfreezes():
    for path in (h.EXPERT_TRAINING,):
        src = path.read_text(encoding="utf-8")
        assert "requires_grad_(True)" not in src, path.name
        assert "requires_grad = True" not in src, path.name


def test_the_worker_image_offers_no_router_unfreeze():
    """D40 (a): no image change. The on-pod CLI has no flag for an unfrozen
    router or a traffic-stealing guard ((b) would add those with a repin)."""
    flags = h.worker_flags()
    assert flags, "no --flags found in worker.py"
    bad = [f for f in flags if any(w in f for w in ("unfreeze", "router-rows", "router_rows",
                                                    "traffic"))]
    assert not bad, bad


def test_no_traffic_share_delta_receipt_field_under_option_a():
    """The row: receipt field ``traffic_share_delta`` is (b) only."""
    hits = h.grep_cascade("traffic_share_delta")
    assert not hits, hits
