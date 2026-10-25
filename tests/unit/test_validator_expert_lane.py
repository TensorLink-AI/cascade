"""DEC-CA-0050: validator-side lane-B judging on real tiny MoE checkpoints."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.eval.scoring import WindowScore  # noqa: E402
from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest  # noqa: E402
from cascade.shared.lane_b import LaneBState  # noqa: E402
from cascade.shared.manifest import format_trained_pointer  # noqa: E402
from cascade.trainer.expert_lane import settle_leg  # noqa: E402
from cascade.validator.expert_lane import (  # noqa: E402
    apply_judgements,
    judge_manifest,
    judgements_record,
)

from .test_moe_trainer import _C, _train  # noqa: E402

CFG = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "bootstrap_B": 500,
                    "min_target_windows": 20, "min_guard_windows": 20})
GEN = "alice/metro-gen@sha256:" + "a" * 64


def _ptr(tag):
    import hashlib

    return format_trained_pointer(
        f"cascade/ckpt-{tag}@sha256:" + hashlib.sha256(tag.encode()).hexdigest())


def _scores(effect, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, d in enumerate(["energy"] * 60 + ["web"] * 60):
        q = rng.lognormal(0, 0.3, 9)
        m = rng.lognormal(0, 0.3)
        f = effect.get(d, 1.0)
        out.append((d, i, q, m, f))
    return out


def _ws(rows, chal):
    res = []
    for d, i, q, m, f in rows:
        g = f if chal else 1.0
        res.append(WindowScore(series_id=f"s{i}", mase=float(m * g), qloss_per_q=q * g,
                               abs_target=50.0, quantile_levels=tuple(np.linspace(.1, .9, 9)),
                               channel=0, domain=d, source=f"feed{i % 15}"))
    return res


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lb")
    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp / "king0", c, budget=600)
    sel = {0: [1, 2, 3]}
    _train(tmp / "leg", c, budget=600, warm=tmp / "king0", sel=sel, seed=3)
    settle_leg(tmp / "king0", tmp / "king0", tmp / "leg", sel, tmp / "rebased")
    return {_ptr("king"): tmp / "king0", _ptr("rebased"): tmp / "rebased"}, sel


def _manifest(sel, rebased_tag="rebased", king_tag="king"):
    e = ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref=GEN, domain="energy",
                    selection=sel, exclude={}, base_king_pointer=_ptr("king"),
                    rebased_pointer=_ptr(rebased_tag), corpus_digest="d", train_block=5,
                    commit_block=4)
    return ExpertManifest(round_id="r9", created_block=100, king_pointer=_ptr(king_tag),
                          lineage_tokens=10, entries=(e,))


def _judge(ckpts, effect, manifest=None, recompute=None):
    dirs, sel = ckpts
    rows = _scores(effect)
    m = manifest or _manifest(sel)
    return judge_manifest(m, king_pointer=_ptr("king"), king_scores=_ws(rows, False),
                          evaluate=lambda p: _ws(rows, True), fetch_dir=dirs.__getitem__,
                          cfg=CFG, seed="blk", recompute_selection=recompute)


def test_passing_leg_becomes_a_domain_win(ckpts):
    js = _judge(ckpts, {"energy": 0.9})
    assert js[0].passed, js[0].reason
    st = LaneBState()
    wins = apply_judgements(st, js, round_id="r9", settle_block=1000)
    assert [w.domain for w in wins] == ["energy"] and st.wins["energy"].uid == 7
    assert judgements_record(js)[0]["passed"] is True


def test_regression_on_other_domain_fails(ckpts):
    js = _judge(ckpts, {"energy": 0.9, "web": 1.06})
    assert not js[0].passed and "web" in js[0].reason


def test_stale_and_tampered_and_selection_mismatch(ckpts):
    _, sel = ckpts
    assert _judge(ckpts, {"energy": 0.9}, _manifest(sel, king_tag="other"))[0].stage == "stale"
    # a rebased checkpoint that moved rows outside its claimed selection
    j = _judge(ckpts, {"energy": 0.9}, _manifest({0: [5]}))[0]
    assert j.stage == "merge" and not j.passed
    j = _judge(ckpts, {"energy": 0.9}, recompute=lambda k, d, ex: {0: [9]})[0]
    assert j.stage == "selection"


def test_unconfigured_domain_ignored(ckpts):
    _, sel = ckpts
    m = _manifest(sel)
    m = dataclasses.replace(m, entries=(dataclasses.replace(m.entries[0], domain="sales"),))
    assert _judge(ckpts, {"energy": 0.9}, m) == []
