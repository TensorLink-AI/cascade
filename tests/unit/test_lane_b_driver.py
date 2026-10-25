"""DEC-CA-0050: the trainer's lane-B driver through one full cycle on tiny
real MoE checkpoints (fake chain/storage/dispatch)."""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.shared.config import parse_lane_b  # noqa: E402
from cascade.shared.expert_manifest import format_expert_commit  # noqa: E402
from cascade.shared.manifest import format_trained_pointer  # noqa: E402
from cascade.trainer.lane_b_driver import LaneBDriver  # noqa: E402
from cascade.validator.expert_lane import judge_manifest  # noqa: E402

from .test_moe_trainer import _C, _train  # noqa: E402
from .test_validator_expert_lane import _scores, _ws  # noqa: E402

GEN = "alice/metro-gen@sha256:" + "a" * 64
C = dataclasses.replace(_C(), copula_loss_cap=0.0)


class Ops:
    def __init__(self, tmp: Path, effect):
        self.tmp, self.store, self.published, self.effect = tmp, {}, [], effect

    def _ptr(self, d: Path) -> str:
        h = hashlib.sha256(str(d).encode()).hexdigest()
        p = format_trained_pointer(f"cascade/{d.name}@sha256:{h}")
        self.store[p] = d
        return p

    def commitments(self):
        return [SimpleNamespace(hotkey="alice", uid=7, commit_block=5,
                                payload=format_expert_commit("energy", GEN)),
                SimpleNamespace(hotkey="bob", uid=8, commit_block=6,
                                payload="metro-v1:gen:hippius:" + GEN)]   # lane A: ignored

    def contexts(self, domain, block):
        rng = np.random.default_rng(len(domain))
        return [np.cumsum(rng.normal(size=32)) for _ in range(16)]

    def fetch(self, p):
        return self.store[p]

    def upload(self, d, repo):
        dst = self.tmp / repo
        if Path(d) != dst:
            shutil.copytree(d, dst, dirs_exist_ok=True)
        return self._ptr(dst)

    def dispatch_leg(self, leg, era, block, done):
        out = self.tmp / f"leg-{leg.sub.hotkey}"
        _train(out, C, budget=600, warm=self.store[leg.base_king_ref], sel=leg.selection, seed=3)
        done(leg, self._ptr(out), "digest", block, "cpu")

    def bench_rows(self, d):
        return None

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block):
        rows = _scores(self.effect)
        return judge_manifest(m, king_pointer=m.king_pointer, king_scores=_ws(rows, False),
                              evaluate=lambda p: _ws(rows, True), fetch_dir=self.fetch,
                              cfg=self.cfg.lane_b, seed=round_id)


def _driver(tmp, effect):
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
                       "routing_jaccard": 0.7, "min_experts": 2, "max_experts": 4,
                       "bootstrap_B": 300, "min_target_windows": 20, "min_guard_windows": 20})
    ops = Ops(tmp, effect)
    cfg = SimpleNamespace(lane_b=lb)
    ops.cfg = cfg
    return LaneBDriver(cfg, ops, tmp / "lane_b.json"), ops


def _kings(tmp, ops):
    """Two consecutive era kings that route identically: era 0's king is a
    byte copy of era 1's under its own pointer."""
    shutil.copytree(tmp / "king", tmp / "king-prev")
    return ops._ptr(tmp / "king-prev"), ops._ptr(tmp / "king")


def test_full_cycle_merges_winner_into_next_era_init(tmp_path):
    _train(tmp_path / "king", C, budget=600)
    drv, ops = _driver(tmp_path, {"energy": 0.9})
    prev, king = _kings(tmp_path, ops)
    assert drv.tick(10, 0, prev) == []                 # lineage below open_tokens
    drv.add_lineage_tokens(1000)
    assert drv.tick(11, 0, prev) == []                 # same king: never compared with itself
    assert drv.st.gate_opened_block == 0 and "no previous king" in drv.st.routing["reason"]
    legs = drv.tick(12, 1, king)                       # previous era's king routes alike → open
    assert drv.st.gate_opened_block == 12
    r = drv.st.routing["domains"]["energy"]
    assert r["overlap"] == [1.0] and r["jaccard"] == [1.0] and r["tokens"] > 0
    assert [leg.sub.hotkey for leg in legs] == ["alice"]
    assert drv.st.finished and not drv.queue.open
    js = drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
                    work_dir=tmp_path / "w")
    assert ops.published and js[0].passed, js[0].reason
    assert len(drv.st.pending_merges) == 1
    merged, merges = drv.init_for(2, king, tmp_path / "w")
    assert merged != king and drv.init_for(2, king, tmp_path / "w")[0] == merged  # idempotent
    assert merges[0]["king_pointer"] == king
    from safetensors.torch import load_file

    a = load_file(str(ops.fetch(king) / "weights.safetensors"))
    b = load_file(str(ops.fetch(merged) / "weights.safetensors"))
    rows = js[0].entry.selection[0]
    assert not torch.equal(a["blocks.1.ffn.w_in"][rows], b["blocks.1.ffn.w_in"][rows])
    assert torch.equal(a["head.weight"], b["head.weight"])
    # state survives a restart
    drv2, _ = _driver(tmp_path, {})
    assert drv2.st.merged_inits == drv.st.merged_inits


def test_failing_leg_is_published_but_not_merged(tmp_path):
    _train(tmp_path / "king", C, budget=600)
    drv, ops = _driver(tmp_path, {"energy": 0.9, "web": 1.08})
    prev, king = _kings(tmp_path, ops)
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    drv.tick(2, 1, king)
    js = drv.settle(round_id="r1", block=20, era_index=1, king_pointer=king,
                    work_dir=tmp_path / "w")
    assert not js[0].passed and drv.st.pending_merges == []
    assert drv.init_for(2, king, tmp_path / "w") == (king, [])


def test_stale_leg_dropped(tmp_path):
    _train(tmp_path / "king", C, budget=600)
    drv, ops = _driver(tmp_path, {"energy": 0.9})
    prev, king = _kings(tmp_path, ops)
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    drv.tick(2, 1, king)
    assert drv.settle(round_id="r9", block=20, era_index=5, king_pointer=king,
                      work_dir=tmp_path / "w") == []
    assert [(o["hotkey"], o["outcome"]) for o in drv.st.outcomes] == [("alice", "stale")]


def _rerouted_copy(src: Path, dst: Path, away_from: list[int]) -> None:
    """A king that routes EVERY token of its MoE layer to 8 experts outside
    ``away_from`` (a routing-bias shift — the bias is part of the selection)."""
    from safetensors.torch import load_file

    from cascade.trainer.ckpt_digest import save_tensors_hashed

    shutil.copytree(src, dst)
    w = load_file(str(dst / "weights.safetensors"))
    elsewhere = [e for e in range(w["blocks.1.ffn.route_bias"].numel()) if e not in away_from][:8]
    w["blocks.1.ffn.route_bias"][elsewhere] += 100.0
    save_tensors_hashed(w, dst / "weights.safetensors")


def _one_leg(tmp_path, effect):
    _train(tmp_path / "king", C, budget=600)
    drv, ops = _driver(tmp_path, effect)
    prev, king = _kings(tmp_path, ops)
    drv.add_lineage_tokens(1000)
    drv.tick(1, 0, prev)
    (leg,) = drv.tick(2, 1, king)
    return drv, ops, king, leg


def test_settle_drops_leg_when_current_king_routes_domain_elsewhere(tmp_path):
    drv, ops, king, leg = _one_leg(tmp_path, {"energy": 0.9})
    _rerouted_copy(tmp_path / "king", tmp_path / "king-moved", leg.selection[0])
    moved = ops._ptr(tmp_path / "king-moved")
    assert drv.settle(round_id="r1", block=20, era_index=1, king_pointer=moved,
                      work_dir=tmp_path / "w") == []
    assert not ops.published and drv.st.finished == [] and drv.st.pending_merges == []
    (o,) = drv.st.outcomes
    assert (o["hotkey"], o["outcome"]) == ("alice", "routing_stale")
    assert o["detail"]["jaccard"] == [0.0] and o["detail"]["overlap"][0] < 0.7


def test_settle_keeps_leg_when_current_king_routes_domain_alike(tmp_path):
    drv, ops, king, leg = _one_leg(tmp_path, {"energy": 0.9})
    shutil.copytree(tmp_path / "king", tmp_path / "king-next")       # new pointer, same routing
    nxt = ops._ptr(tmp_path / "king-next")
    js = drv.settle(round_id="r1", block=20, era_index=1, king_pointer=nxt,
                    work_dir=tmp_path / "w")
    assert ops.published and js and js[0].passed, js and js[0].reason
    assert drv.st.outcomes == []
    results = drv.status()["submission_progress"]
    assert any(r["ref"] == GEN and r["stage"] == "qualified" for r in results)
    restored = LaneBDriver(drv.cfg, ops, drv.path)
    assert restored.status()["submission_progress"] == results


def test_two_passing_legs_distinguish_slot_competition_from_failed_qualification(tmp_path):
    import copy

    drv, ops, king, _ = _one_leg(tmp_path, {"energy": 0.9})
    other = copy.deepcopy(drv.st.finished[0])
    other["leg"]["sub"].update(hotkey="bob", uid=8, commit_block=6)
    drv.st.finished.append(other)
    judgements = drv.settle(round_id="both", block=20, era_index=1,
                            king_pointer=king, work_dir=tmp_path / "settle")
    assert len(judgements) == 2 and all(j.passed for j in judgements)
    progress = drv.status()["submission_progress"]
    # Each hotkey has its own bootstrap seed: equal point gains need not
    # produce equal confidence bounds. The best LCB wins before tie-breaking.
    best = min(judgements, key=lambda j: (-j.verdict.target_lcb,
                                         j.entry.commit_block, j.entry.miner_hotkey))
    assert {r["hotkey"]: r["stage"] for r in progress} == {
        j.entry.miner_hotkey: ("qualified" if j is best else "qualification_not_selected")
        for j in judgements}
    assert len(drv.st.pending_merges) == 1
