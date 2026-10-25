"""DEC-CA-0056 lane C, trainer side of the rolling scheduler: the lane-C
manifest settles BEFORE the lane-A settlement manifest, which binds its
digest (signed, drop-when-default); a failed lane-C settlement binds nothing
and lane A proceeds; a lane-A publish replayed after a crash republishes the
identical lane-C bytes first; lane-C legs keep their payer pod through the
lane-A sweep and hold back a same-hotkey lane-A leg."""

from __future__ import annotations

from dataclasses import replace

from cascade.shared.config.lane_c import LaneCConfig
from cascade.shared.manifest import TrainingManifest, dump_manifest, load_manifest

from .test_rolling import EB, REF, Clock, FakeClient, _advance, _armed, _commit, _join, _sched

DIGEST = "cd" * 32


class _LaneC:
    def __init__(self, events, *, fail=False, live=()):
        self.events, self.fail, self.live = events, fail, set(live)
        self.ticks: list = []
        self.settled: dict = {}

    def tick(self, block, era_index, king_pointer, era_start):
        self.ticks.append((block, era_index, king_pointer, era_start))
        return []

    def settle(self, *, round_id, block, era_index, base_pointer, created_block,
               base_round_id, pool_block):
        self.events.append(("lane_c_settle", round_id, base_pointer))
        if self.fail:
            raise OSError("bucket 503")
        self.settled[round_id] = DIGEST
        return DIGEST

    def published(self, round_id):
        return {"key": f"lane_c/round-{round_id}.json", "text": "LANE-C-TEXT"}

    def live_hotkeys(self):
        return set(self.live)

    def status(self):
        return {}


def _run(cfg, tmp_path, *, fail=False, publish_fails=0, live=()):
    armed = _armed(cfg)
    armed = replace(armed, lane_c=LaneCConfig(from_block=1, targets=("mkt:returns",)))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    events: list = []
    sched.lane_c = _LaneC(events, fail=fail, live=live)
    orig = ops.publish_manifest
    left = {"n": publish_fails}

    def publish(m):
        if left["n"]:
            left["n"] -= 1
            raise OSError("manifest bucket 503")
        events.append(("lane_a_publish", m.round_id))
        orig(m)

    ops.publish_manifest = publish
    ops.publish_text = lambda key, text: events.append(("republish", key, text))
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    try:
        _advance(clock, ops, sched, client, from_block=b0, to_block=era_start + EB + 3)
    except OSError:
        assert publish_fails                 # the tick fails; the write-ahead stays
        assert sched.state.pending_publish["lane_c"]["text"] == "LANE-C-TEXT"
    if publish_fails:
        _advance(clock, ops, sched, client, from_block=era_start + EB + 3,
                 to_block=era_start + EB + 5)
    return sched, events, ops


def test_lane_c_settles_first_and_is_bound_into_the_lane_a_manifest(cfg, tmp_path):
    sched, events, ops = _run(cfg, tmp_path)
    (m,) = ops.manifests
    king = m.entries[0].trained_pointer
    assert events == [("lane_c_settle", m.round_id, king), ("lane_a_publish", m.round_id)]
    assert m.lane_c_digest == DIGEST
    assert b"lane_c_digest" in m.canonical_body()
    assert load_manifest(dump_manifest(m)).lane_c_digest == DIGEST
    assert sched.lane_c.ticks, "the driver ticks every scheduler tick of a lane-C era"


def test_failed_lane_c_settlement_binds_nothing_and_lane_a_proceeds(cfg, tmp_path):
    _, _, ops = _run(cfg, tmp_path, fail=True)
    (m,) = ops.manifests
    assert m.lane_c_digest == ""
    assert b"lane_c_digest" not in m.canonical_body()


def test_replayed_lane_a_publish_republishes_the_lane_c_bytes_first(cfg, tmp_path):
    _, events, ops = _run(cfg, tmp_path, publish_fails=1)
    (m,) = ops.manifests
    assert m.lane_c_digest == DIGEST
    i = events.index(("republish", f"lane_c/round-{m.round_id}.json", "LANE-C-TEXT"))
    assert events.index(("lane_a_publish", m.round_id)) > i
    # settled once: the replay never re-derives the lane-C manifest
    assert [e for e in events if e[0] == "lane_c_settle"] == [
        ("lane_c_settle", m.round_id, m.entries[0].trained_pointer)]


def test_lane_c_legs_keep_their_pod_and_hold_lane_a(cfg, tmp_path):
    armed = _armed(cfg)
    armed = replace(armed, lane_c=LaneCConfig(from_block=1, targets=("mkt:returns",)))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    sched.lane_c = _LaneC([], live={"ALFA"})
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    assert ops.queue().get("ALFA").status == "queued"        # held while lane C runs
    sched._reconciled_at = -1e12
    sched._maybe_reconcile(client, b0 + 1, clock())
    assert "ALFA" in ops.sweeps[-1][1]


def test_binding_is_drop_when_default():
    m = TrainingManifest(round_id="1", created_block=2, contract_digest="1" * 64,
                         base_arch_digest="2" * 64, eval_dataset="d", entries=[])
    assert b"lane_c_digest" not in m.canonical_body()
