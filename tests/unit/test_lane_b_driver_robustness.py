"""Audit finding 8 (DEC-CA-0050): the trainer's lane-B driver never loses a
finished leg or a pending merge to a transient I/O failure (entries leave the
queue only after success), settles idempotently per round, and plans each
domain's ESFT selection once per king instead of every scheduler tick."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.shared.config import parse_lane_b
from cascade.shared.expert_manifest import format_expert_commit
from cascade.trainer import expert_lane as tel
from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver

GEN = "alice/metro-gen@sha256:" + "a" * 64


class _Ops:
    def __init__(self):
        self.fail_fetch: set[str] = set()
        self.published = []

    def commitments(self):
        return [SimpleNamespace(hotkey="alice", uid=7, commit_block=5,
                                payload=format_expert_commit("energy", GEN))]

    def contexts(self, domain, block):
        return []

    def fetch(self, p):
        if p in self.fail_fetch or "*" in self.fail_fetch:
            raise OSError(f"registry 503 for {p}")
        return Path("/nonexistent") / p

    def upload(self, d, repo):
        raise OSError("hub down")

    def dispatch_leg(self, leg, era, block, done):
        pass

    def bench_rows(self, d):
        return None

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block):
        return []


def _driver(tmp_path, **lb):
    cfg = SimpleNamespace(lane_b=parse_lane_b({"from_block": 1, "domains": ["energy", "web"],
                                               "open_tokens": 0, **lb}))
    ops = _Ops()
    return LaneBDriver(cfg, ops, tmp_path / "lane_b.json"), ops


def _finished(hk="alice"):
    leg = OpenLeg(sub=LaneBSubmission(hk, 7, GEN, "energy", 5), selection={0: [1]},
                  base_king_ref="kingref", base_era=0)
    return asdict(FinishedExpertLeg(leg=asdict(leg), leg_pointer="legptr",
                                    corpus_digest="c", train_block=3))


def test_settle_failure_keeps_finished_legs(tmp_path):
    drv, ops = _driver(tmp_path)
    drv.st.finished = [_finished()]
    drv.save()
    ops.fail_fetch.add("legptr")
    with pytest.raises(OSError):
        drv.settle(round_id="r1", block=20, era_index=0, king_pointer="kingref",
                   work_dir=tmp_path / "w")
    assert drv.st.finished == [_finished()]
    assert "r1" not in drv.st.settled and drv.expert_manifest_digest("r1") == ""
    # persisted too: a restart still holds the leg
    again = LaneBDriver(drv.cfg, ops, tmp_path / "lane_b.json")
    assert [f["leg_pointer"] for f in again.st.finished] == ["legptr"]


def test_settle_is_idempotent_per_round(tmp_path):
    drv, ops = _driver(tmp_path)
    drv.st.finished = [_finished()]
    drv.st.finished[0]["leg"]["base_era"] = -10        # stale → dropped, nothing published
    assert drv.settle(round_id="r1", block=20, era_index=0, king_pointer="k",
                      work_dir=tmp_path / "w") == []
    assert drv.st.finished == [] and drv.st.settled == {"r1": {"digest": ""}}
    drv.st.finished = [_finished()]
    ops.fail_fetch.add("*")                            # a re-call must not touch I/O
    assert drv.settle(round_id="r1", block=20, era_index=0, king_pointer="k",
                      work_dir=tmp_path / "w") == []
    assert drv.st.finished == [_finished()]            # left for the NEXT settlement


def test_init_for_failure_keeps_pending_merges(tmp_path):
    drv, ops = _driver(tmp_path)
    merge = {"rebased_pointer": "rb", "king_pointer": "k", "selection": {"0": [1]},
             "round_id": "r1"}
    drv.st.pending_merges = [merge]
    ops.fail_fetch.add("*")
    with pytest.raises(OSError):
        drv.init_for(1, "member", tmp_path / "w")
    assert drv.st.pending_merges == [merge] and drv.st.merged_inits == {}


def test_tick_probes_once_per_king_and_domain(tmp_path, monkeypatch):
    calls = []

    def probe(king_dir, contexts, cfg):
        calls.append(Path(king_dir).name)
        topk = [np.array([[1, 2]] * 4, dtype=np.uint8)]
        return tel.RoutingProbe(topk=topk, affinity=np.bincount(topk[0].ravel(),
                                                                 minlength=8)[None])

    monkeypatch.setattr(tel, "probe_routing", probe)
    drv, _ = _driver(tmp_path, min_experts=2, max_experts=2)
    drv.tick(9, 0, "king-0")
    for b in range(10, 15):
        drv.tick(b, 1, "king-a")
    # era 0: king-0 alone (no previous king → no comparison); era 1: king-a vs
    # king-0, each probed once per domain however many ticks run
    assert sorted(calls) == ["king-0", "king-0", "king-a", "king-a"]
    drv.tick(16, 2, "king-b")
    assert calls[4:] == ["king-b", "king-b"]         # a new king re-probes (king-a cached)
