"""DEC-CA-0050 lane B routing stability (trainer policy): the opening gate
measures OLMoE-style router saturation between the PREVIOUS era's king and
the current one (never a king against itself), per MoE layer, and latches
open; settlement drops a leg as ``routing_stale`` when the current king routes
its domain to different experts than the leg's base king."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.shared.expert_manifest import format_expert_commit  # noqa: E402
from cascade.trainer import expert_lane as tel  # noqa: E402
from cascade.trainer.expert_lane import (  # noqa: E402
    LaneBQueue,
    LaneBSubmission,
    OpenLeg,
    RoutingProbe,
    routing_metrics,
    routing_stale,
    topk_overlap_layers,
)
from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver  # noqa: E402
from cascade.trainer.moe_model import (  # noqa: E402
    CascadeMoEConfig,
    CascadeMoEModel,
    domain_expert_affinity,
    routing_topk,
    select_domain_experts,
)

GEN = "alice/metro-gen@sha256:" + "a" * 64
E = 16


# ── routing_topk ──────────────────────────────────────────────────────────────


def _tiny_model() -> CascadeMoEModel:
    torch.manual_seed(0)
    cfg = CascadeMoEConfig(d_model=32, num_layers=4, num_heads=2, head_dim=16, patch_size=4,
                           context_length=32, horizon=8, max_patches=16, variate_blocks=(),
                           moe_blocks=(1, 2, 3), dense_ffn=40, num_experts=E, top_k=4,
                           expert_hidden=8, shared_hidden=8, conv_channels=8)
    return CascadeMoEModel(cfg).eval()


def _batches(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return [(torch.as_tensor(rng.normal(size=(2, 8, 4)), dtype=torch.float32), None)
            for _ in range(n)]


def test_routing_topk_is_what_the_forward_dispatched():
    model = _tiny_model()
    layers = model.moe_layers()
    # a large routing bias on expert 5 of the middle layer: the SELECTION
    # includes the bias, so every token of that layer must route to it
    layers[1].route_bias[5] = 100.0
    seen: list[list[torch.Tensor]] = [[] for _ in layers]
    hooks = [layer.register_forward_pre_hook(
        lambda mod, args, i=i: seen[i].append(mod.route(args[0])[0].clone()))
        for i, layer in enumerate(layers)]
    try:
        tk = routing_topk(model, _batches())
    finally:
        for h in hooks:
            h.remove()
    assert len(tk) == 3 and all(t.shape == (3 * 2 * 8, 4) for t in tk)
    for i, t in enumerate(tk):
        assert torch.equal(t, torch.cat(seen[i]))      # the dispatch's own top-k
    assert bool((tk[1] == 5).any(dim=1).all())         # bias included
    assert not bool((tk[0] == 5).any(dim=1).all())
    # its bincount is the ESFT affinity, row for row
    aff = domain_expert_affinity(model, _batches())
    for i, t in enumerate(tk):
        assert torch.equal(torch.bincount(t.reshape(-1), minlength=E), aff[i])
    assert not model.training


def test_probe_selection_matches_esft_selection(tmp_path):
    from .test_moe_trainer import _C, _train

    c = dataclasses.replace(_C(), copula_loss_cap=0.0)
    _train(tmp_path / "king", c, budget=600)
    cfg = parse_lane_b({"from_block": 1, "domains": ["energy"], "min_experts": 2,
                        "max_experts": 6, "affinity_windows": 16})
    rng = np.random.default_rng(0)
    ctx = [np.cumsum(rng.normal(size=32)) for _ in range(20)]
    model = tel.load_king(tmp_path / "king")
    aff = domain_expert_affinity(model, tel.affinity_batches(
        ctx[:16], patch_size=model.cfg.patch_size,
        max_patches=model.cfg.context_length // model.cfg.patch_size))
    want = select_domain_experts(aff, p=cfg.esft_p, min_n=2, max_n=6)
    probe = tel.probe_routing(tmp_path / "king", ctx, cfg)
    assert probe.select(cfg) == want
    assert np.array_equal(probe.affinity, aff.numpy()) and probe.topk[0].dtype == np.uint8


# ── metrics ───────────────────────────────────────────────────────────────────


def test_topk_overlap_layers():
    a = [np.array([[0, 1], [2, 3]]), np.array([[0, 1], [0, 1]])]
    b = [np.array([[1, 0], [2, 9]]), np.array([[5, 6], [0, 7]])]
    assert topk_overlap_layers(a, b) == [pytest.approx(0.75), pytest.approx(0.25)]
    assert topk_overlap_layers(a, a) == [1.0, 1.0]
    with pytest.raises(ValueError, match="same tokens"):
        topk_overlap_layers(a, [b[0][:1], b[1]])


def _probe(*layers) -> RoutingProbe:
    topk = [np.asarray(t, dtype=np.uint8) for t in layers]
    return RoutingProbe(topk=topk, affinity=np.stack(
        [np.bincount(t.ravel(), minlength=E) for t in topk]).astype(np.int64))


# 40 tokens, top-2, two MoE layers: the domain lives on {0,1} / {4,5}
L0 = [[0, 1]] * 30 + [[2, 3]] * 10
L1 = [[4, 5]] * 30 + [[6, 7]] * 10
L1_MOVED = [[8, 9]] * 30 + [[6, 7]] * 10            # layer 1 re-routed elsewhere
ROUTES = {"king-a": _probe(L0, L1), "king-b": _probe(L0, L1), "king-c": _probe(L0, L1_MOVED)}


def test_routing_metrics_are_per_layer():
    cfg = parse_lane_b({"min_experts": 2, "max_experts": 4})
    m = routing_metrics(ROUTES["king-a"], ROUTES["king-c"], cfg)
    assert m["overlap"] == [1.0, 0.25] and m["jaccard"] == [1.0, 0.0] and m["tokens"] == 40


# ── the driver's gate (synthetic probes) ──────────────────────────────────────


class _Ops:
    def __init__(self, subs=(("alice", "energy"),)):
        self.subs, self.dispatched, self.published = list(subs), [], []

    def commitments(self):
        return [SimpleNamespace(hotkey=hk, uid=7 + i, commit_block=5 + i,
                                payload=format_expert_commit(d, GEN))
                for i, (hk, d) in enumerate(self.subs)]

    def contexts(self, domain, block):
        return [np.arange(8.0)]

    def fetch(self, p):
        return Path("/nonexistent") / p

    def upload(self, d, repo):
        raise OSError("hub down")

    def dispatch_leg(self, leg, era, block, done):
        self.dispatched.append(leg)

    def bench_rows(self, d):
        return None

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block):
        return []


@pytest.fixture
def routes(monkeypatch):
    calls = []

    def probe(king_dir, contexts, cfg):
        calls.append(Path(king_dir).name)
        return ROUTES[Path(king_dir).name]

    monkeypatch.setattr(tel, "probe_routing", probe)
    return calls


def _driver(tmp_path, ops=None, **lb):
    cfg = SimpleNamespace(lane_b=parse_lane_b({
        "from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
        "min_experts": 2, "max_experts": 4, **lb}))
    ops = ops or _Ops()
    return LaneBDriver(cfg, ops, tmp_path / "lane_b.json"), ops


def test_gate_never_compares_a_king_with_itself(tmp_path, routes):
    """Regression: the old gate compared the king with ITSELF on the previous
    tick (1.0 on the second look), so lane B opened on tokens alone."""
    drv, ops = _driver(tmp_path)
    drv.add_lineage_tokens(10**6)
    for b in range(10, 15):
        assert drv.tick(b, 0, "king-a") == []
    assert drv.tick(15, 1, "king-a") == []             # a new era, the SAME king
    assert drv.st.gate_opened_block == 0 and not ops.dispatched
    assert "no previous king" in drv.st.routing["reason"]
    assert routes == []                                # nothing to compare → no probe


def test_gate_opens_when_consecutive_kings_route_alike(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    assert drv.tick(10, 0, "king-a") == []
    assert drv.tick(11, 1, "king-b") == []             # saturated, but lineage < open_tokens
    assert "lineage" in drv.st.routing["reason"]
    drv.add_lineage_tokens(100)
    legs = drv.tick(12, 1, "king-b")
    assert [leg.sub.hotkey for leg in legs] == ["alice"] and drv.st.gate_opened_block == 12
    r = drv.st.routing
    assert (r["king"], r["prev_king"], r["open"]) == ("king-b", "king-a", True)
    assert r["domains"]["energy"] == {"overlap": [1.0, 1.0], "jaccard": [1.0, 1.0],
                                      "tokens": 40}
    # one saturation-curve row per (previous king, king) pair, not per tick
    assert [(h["prev_king"], h["king"]) for h in drv.st.routing_history] == [
        ("king-a", "king-b")]
    assert drv.st.jaccard == {"energy": 1.0, "web": 1.0}
    assert legs[0].selection == {0: [0, 1], 1: [4, 5]}
    # probed once per (king, domain), never per tick
    assert sorted(routes) == ["king-a", "king-a", "king-b", "king-b"]
    # state (gate numbers + latch) survives a restart
    again = LaneBDriver(drv.cfg, ops, tmp_path / "lane_b.json")
    assert again.st.gate_opened_block == 12 and again.st.routing == drv.st.routing


def test_gate_stays_closed_when_one_layer_moved(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    drv.add_lineage_tokens(100)
    drv.tick(10, 0, "king-a")
    assert drv.tick(11, 1, "king-c") == []
    assert drv.st.gate_opened_block == 0 and not ops.dispatched
    d = drv.st.routing["domains"]["energy"]
    assert d["overlap"] == [1.0, 0.25] and d["jaccard"] == [1.0, 0.0]
    assert "routing unsaturated" in drv.st.routing["reason"]
    assert "[1]" in drv.st.routing["reason"]           # names the layer


def test_gate_is_a_one_way_latch(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    drv.add_lineage_tokens(100)
    drv.tick(10, 0, "king-a")
    assert len(drv.tick(11, 1, "king-b")) == 1
    drv._leg_done(ops.dispatched[0], None)             # the leg failed; alice re-queues
    legs = drv.tick(12, 2, "king-c")                   # the router moved — still open
    assert drv.st.gate_opened_block == 11 and len(legs) == 1
    assert drv.st.routing["domains"]["energy"]["overlap"] == [1.0, 0.25]   # still measured
    assert len(drv.st.routing_history) == 2


def test_concurrent_legs_record_their_exclusions(tmp_path, routes):
    drv, ops = _driver(tmp_path, ops=_Ops([("alice", "energy"), ("bob", "web")]))
    drv.add_lineage_tokens(100)
    drv.tick(10, 0, "king-a")
    a, b = drv.tick(11, 1, "king-b")
    assert a.exclude == {} and b.exclude == {0: [0, 1], 1: [4, 5]}
    assert not set(b.selection[0]) & set(a.selection[0])
    back = LaneBQueue.from_json(drv.queue.to_json())
    assert back.open[1].exclude == b.exclude


# ── rebase staleness ──────────────────────────────────────────────────────────


def test_routing_stale_rule():
    cfg = parse_lane_b({"min_experts": 2, "max_experts": 4})
    sel = {0: [0, 1], 1: [4, 5]}
    assert routing_stale(ROUTES["king-a"], ROUTES["king-b"], sel, cfg)[0] is False
    stale, detail = routing_stale(ROUTES["king-a"], ROUTES["king-c"], sel, cfg)
    assert stale and detail == {"jaccard": [1.0, 0.0], "overlap": [1.0, 0.25]}
    # loose bars keep it
    loose = parse_lane_b({"min_experts": 2, "max_experts": 4, "rebase_jaccard": 0.0,
                          "rebase_overlap": 0.2})
    assert routing_stale(ROUTES["king-a"], ROUTES["king-c"], sel, loose)[0] is False
    # the re-plan honours the leg's exclusions
    stale, detail = routing_stale(ROUTES["king-a"], ROUTES["king-b"], {0: [2, 3], 1: [6, 7]},
                                  cfg, exclude={0: {0, 1}, 1: {4, 5}})
    assert not stale and detail["jaccard"] == [1.0, 1.0]


def test_settle_drops_a_routing_stale_leg(tmp_path, routes):
    drv, ops = _driver(tmp_path)
    leg = OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5),
                  selection={0: [0, 1], 1: [4, 5]}, base_king_ref="king-a", base_era=1)
    drv.st.finished = [asdict(FinishedExpertLeg(leg=asdict(leg), leg_pointer="legptr",
                                                corpus_digest="c", train_block=3))]
    assert drv.settle(round_id="r1", block=20, era_index=2, king_pointer="king-c",
                      work_dir=tmp_path / "w") == []
    assert not ops.published and drv.st.finished == []
    assert drv.expert_manifest_digest("r1") == ""
    (o,) = drv.st.outcomes
    assert (o["hotkey"], o["domain"], o["outcome"]) == ("alice", "energy", "routing_stale")
    assert o["detail"] == {"jaccard": [1.0, 0.0], "overlap": [1.0, 0.25]}
    assert o["base_king_ref"] == "king-a" and o["king_pointer"] == "king-c"
