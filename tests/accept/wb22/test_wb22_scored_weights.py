"""WB22, plan rev6.20 (owner 2026-11-24, lane B only: EMA, no cooldown): the
signed expert manifest entry records which weights the leg scored
(``scored_weights``: "ema010" from ``[lane_b] ema_no_cooldown_from_block``,
"" = the raw cooled endpoint, dropped from the body so an entry without it
keeps its bytes), the trainer applies the leg-contract rule at dispatch, and
settlement refuses a leg whose checkpoint marker disagrees with the rule at
its train_block. Contract: README.md "rev6.20"."""

from __future__ import annotations

import dataclasses
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

GEN = "alice/metro-gen@sha256:" + "a" * 64
KING = "metro-v1:trained:hippius:org/king@sha256:" + "b" * 64
LEG = "metro-v1:trained:hippius:org/leg@sha256:" + "c" * 64
REB = "metro-v1:trained:hippius:org/rebased@sha256:" + "d" * 64
MARKER = {"sigma_rels": [0.05, 0.10], "scored_sigma_rel": 0.10,
          "schedule": "warmup_constant_ema"}


def _has_field():
    from cascade.shared.expert_manifest import ExpertEntry

    assert "scored_weights" in {f.name for f in dataclasses.fields(ExpertEntry)}, (
        "ExpertEntry has no scored_weights (plan rev6.20)")


def _entry(**kw):
    from cascade.shared.expert_manifest import ExpertEntry

    return ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=GEN, domain="energy",
                       selection={0: [1, 2]}, exclude={}, base_king_pointer=KING,
                       rebased_pointer=REB, corpus_digest="cd", train_block=5,
                       commit_block=4, **kw)


def _manifest(entry):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id="r1", created_block=10, king_pointer=KING,
                          lineage_tokens=1, entries=(entry,))


def test_scored_weights_round_trips_and_is_dropped_when_raw():
    _has_field()
    from cascade.shared.expert_manifest import load_expert_manifest

    raw = _manifest(_entry())
    assert "scored_weights" not in json.loads(raw.canonical_body())["entries"][0]
    m = _manifest(_entry(scored_weights="ema010"))
    assert json.loads(m.canonical_body())["entries"][0]["scored_weights"] == "ema010"
    assert m.digest() != raw.digest()          # signed: the record is in the digest
    back = load_expert_manifest(m.to_json())
    assert back.entries[0].scored_weights == "ema010" and back.digest() == m.digest()
    assert load_expert_manifest(raw.to_json()).entries[0].scored_weights == ""


# rev6.21 (owner 2026-11-24; tests-correction): "ema005" etc. are the miner's
# EMA choices (WB27, widths 0.01-0.30) — valid labels now; "ema999" stays unknown
@pytest.mark.parametrize("bad", ["raw", "ema999", "EMA010", None])
def test_scored_weights_refuses_unknown_labels(bad):
    _has_field()
    with pytest.raises(ValueError):
        _entry(scored_weights=bad)


def _ckpt(path: Path, marker=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    meta = {"moe": {}, "base_arch": "cascade_pyramid"}
    if marker is not None:
        meta["power_ema"] = marker
    (path / "config.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return path


def test_leg_scored_weights_reads_the_checkpoint_marker(tmp_path):
    from cascade.trainer import lane_b_driver

    assert hasattr(lane_b_driver, "leg_scored_weights"), "lane_b_driver.leg_scored_weights"
    read = lane_b_driver.leg_scored_weights
    assert read(_ckpt(tmp_path / "raw")) == ""
    assert read(_ckpt(tmp_path / "ema", MARKER)) == "ema010"
    # an EMA scored under any other rule never matches an expected value
    assert read(_ckpt(tmp_path / "w_ema", {"sigma_rels": [0.05, 0.10],
                                            "scored_sigma_rel": 0.10})) == "?"
    # rev6.21 (WB27; tests-correction): the miner's EMA choice 0.05
    assert read(_ckpt(tmp_path / "s005", {**MARKER, "scored_sigma_rel": 0.05})) == "ema005"
    assert read(_ckpt(tmp_path / "s050", {**MARKER, "scored_sigma_rel": 0.50})) == "?"


def test_dispatch_paths_apply_the_leg_contract_rule():
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    for fn in (RunnerLaneBOps._dispatch_funded, RunnerLaneBOps._dispatch_operator):
        assert "lane_b_leg_schedule(self.r.cfg.lane_b" in inspect.getsource(fn), fn.__name__


class _Ops:
    def __init__(self, dirs):
        self.dirs = dirs
        self.published = []

    def fetch(self, p):
        return self.dirs[p]

    def upload(self, d, repo):
        return REB

    def bench_rows(self, d):
        return None

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block):
        return []


def _settle(tmp_path, monkeypatch, *, marker, from_block):
    from cascade.shared.config import parse_lane_b
    from cascade.trainer import lane_b_driver as d
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    king = _ckpt(tmp_path / "king")
    leg_dir = _ckpt(tmp_path / "leg", marker)
    rebuilt = []

    def fake_settle_leg(king_now, base, leg, sel, out, **kw):
        rebuilt.append(Path(leg))
        return _ckpt(Path(out))

    monkeypatch.setattr(d, "settle_leg", fake_settle_leg)
    cfg = SimpleNamespace(lane_b=parse_lane_b({
        "from_block": 1, "domains": ["energy"], "open_tokens": 0,
        "ema_no_cooldown_from_block": from_block}))
    ops = _Ops({KING: king, LEG: leg_dir})
    drv = d.LaneBDriver(cfg, ops, tmp_path / "lane_b.json")
    leg = OpenLeg(sub=LaneBSubmission("alice", 7, GEN, "energy", 5), selection={0: [1]},
                  base_king_ref=KING, base_era=0)
    drv.st.finished = [asdict(d.FinishedExpertLeg(leg=asdict(leg), leg_pointer=LEG,
                                                  corpus_digest="c", train_block=30))]
    drv.settle(round_id="r1", block=40, era_index=0, king_pointer=KING,
               work_dir=tmp_path / "w")
    return drv, ops, rebuilt


def test_settle_records_ema010_for_an_armed_ema_leg(tmp_path, monkeypatch):
    _has_field()
    drv, ops, rebuilt = _settle(tmp_path, monkeypatch, marker=MARKER, from_block=20)
    assert rebuilt == [tmp_path / "leg"]                  # rebased from that checkpoint
    (m,) = ops.published
    assert m.entries[0].scored_weights == "ema010" and m.entries[0].train_block == 30


def test_settle_drops_a_raw_leg_once_the_rule_is_armed(tmp_path, monkeypatch):
    _has_field()
    drv, ops, rebuilt = _settle(tmp_path, monkeypatch, marker=None, from_block=20)
    assert rebuilt == [] and ops.published == []
    assert [o["outcome"] for o in drv.st.outcomes] == ["scored_weights_mismatch"]


def test_settle_drops_an_ema_leg_before_the_rule(tmp_path, monkeypatch):
    _has_field()
    drv, ops, rebuilt = _settle(tmp_path, monkeypatch, marker=MARKER, from_block=31)
    assert rebuilt == [] and ops.published == []
    assert [o["outcome"] for o in drv.st.outcomes] == ["scored_weights_mismatch"]
