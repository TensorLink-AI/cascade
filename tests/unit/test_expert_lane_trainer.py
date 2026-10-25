"""DEC-CA-0050 lane B trainer policy: gate, selection, queue, settle + merge."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.trainer.expert_lane import (  # noqa: E402
    LaneBQueue,
    LaneBSubmission,
    _moe_names,
    apply_merges,
    gate_open,
    probe_routing,
    selection_jaccard_layers,
    settle_leg,
)
from cascade.trainer.moe_model import MergeError  # noqa: E402

from .test_moe_trainer import _C, _train  # noqa: E402

CFG = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "min_experts": 2,
                    "max_experts": 6, "esft_p": 0.2, "affinity_windows": 16})


def test_jaccard_and_gate():
    a = {0: [1, 2, 3], 1: [4]}
    assert selection_jaccard_layers(a, a) == {0: 1.0, 1: 1.0}
    assert selection_jaccard_layers({0: [1, 2]}, {0: [2, 3]}) == {0: pytest.approx(1 / 3)}
    def m(overlap, jaccard):
        return {"overlap": overlap, "jaccard": jaccard, "tokens": 100}

    good = m([0.9, 0.85], [0.8, 0.75])
    ok, why = gate_open(10, {"energy": good, "web": good}, CFG)
    assert not ok and "lineage" in why
    ok, why = gate_open(CFG.open_tokens, {"energy": good}, CFG)
    assert not ok and "no previous king" in why and "web" in why
    # ONE layer below the Jaccard bar closes it, even with a high mean
    ok, why = gate_open(CFG.open_tokens, {"energy": good, "web": m([1.0, 1.0], [1.0, 0.5])},
                        CFG)
    assert not ok and "web" in why and "[1]" in why
    # ONE layer below the token-overlap bar closes it
    ok, why = gate_open(CFG.open_tokens, {"energy": m([0.79, 1.0], [1.0, 1.0]), "web": good},
                        CFG)
    assert not ok and "energy" in why
    assert gate_open(CFG.open_tokens, {"energy": good, "web": good}, CFG)[0]
    empty = {"overlap": [1.0], "jaccard": [1.0], "tokens": 0}
    assert not gate_open(CFG.open_tokens, {"energy": good, "web": empty}, CFG)[0]


def test_queue_fifo_one_per_domain_and_disjoint():
    q = LaneBQueue()
    s1 = LaneBSubmission("a", 1, "g1", "energy", 100)
    s2 = LaneBSubmission("b", 2, "g2", "energy", 90)
    s3 = LaneBSubmission("c", 3, "g3", "web", 120)
    for s in (s1, s2, s3):
        q.submit(s, CFG)
    nxt = q.next_dispatchable(CFG)
    assert [s.hotkey for s in nxt] == ["b", "c"]          # earliest commit per domain
    q.start(s2, {0: [1, 2]}, "king@1", base_era=10)
    assert [s.hotkey for s in q.next_dispatchable(CFG)] == ["c"]
    with pytest.raises(ValueError, match="overlaps"):
        q.start(s3, {0: [2, 9]}, "king@1", base_era=10)
    leg = q.start(s3, {0: [3, 9]}, "king@1", base_era=10)
    assert q.owned_rows() == {0: {1, 2, 3, 9}}
    assert q.stale(leg, 13, CFG) and not q.stale(leg, 12, CFG)
    back = LaneBQueue.from_json(q.to_json())
    assert back.to_json() == q.to_json()
    with pytest.raises(ValueError):
        q.submit(LaneBSubmission("d", 4, "g", "sales", 1), CFG)


def test_selection_settle_and_merge(tmp_path):
    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp_path / "king0", c, budget=600)
    rng = np.random.default_rng(0)
    ctx = [np.cumsum(rng.normal(size=32)) for _ in range(20)]
    sel = probe_routing(tmp_path / "king0", ctx, CFG).select(CFG)
    assert sel == probe_routing(tmp_path / "king0", ctx, CFG).select(CFG)  # deterministic
    assert 2 <= len(sel[0]) <= 6
    # leg trains against king0; lane A moves the king on to king1 meanwhile
    _train(tmp_path / "leg", c, budget=600, warm=tmp_path / "king0", sel=sel, seed=3)
    _train(tmp_path / "king1", c, budget=600, warm=tmp_path / "king0", seed=4)
    settle_leg(tmp_path / "king1", tmp_path / "king0", tmp_path / "leg", sel,
            tmp_path / "rebased")
    from safetensors.torch import load_file

    def load(d):
        return load_file(str(tmp_path / d / "weights.safetensors"))

    k1 = load("king1")
    mg = apply_merges(k1, [{"king_pointer": "king1", "rebased_pointer": "rebased",
                            "selection": sel}], load, _moe_names(tmp_path / "king1"))
    assert torch.equal(k1["head.weight"], mg["head.weight"])
    rows = sel[0]
    assert not torch.equal(k1["blocks.1.ffn.w_in"][rows], mg["blocks.1.ffn.w_in"][rows])
    # a leg whose weights moved outside its rows is refused at settlement
    with pytest.raises(MergeError):
        settle_leg(tmp_path / "king1", tmp_path / "king0", tmp_path / "king1", sel,
                   tmp_path / "bad")
