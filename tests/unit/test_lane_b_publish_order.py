"""Audit finding 4 (DEC-CA-0050), trainer side: lane B settles and its expert
manifest is published BEFORE the lane-A settlement manifest, which binds the
expert manifest's digest (signed, drop-when-default). A validator polling
between the two publishes can therefore never handle a settlement without the
lane-B legs the next era init will merge."""

from __future__ import annotations

from dataclasses import replace

from cascade.shared.config import parse_lane_b
from cascade.shared.manifest import TrainingManifest, dump_manifest, load_manifest

from .test_rolling import (
    EB,
    REF,
    Clock,
    FakeClient,
    _advance,
    _armed,
    _commit,
    _join,
    _sched,
)


class _Driver:
    def __init__(self, events, digest="ab" * 32, fail=False):
        self.events, self.digest, self.fail = events, digest, fail
        self.settled = {}

    def tick(self, block, era_index, king_pointer):
        return []

    def add_lineage_tokens(self, n):
        pass

    def init_for(self, era_index, init_pointer, work_dir, era_start=None):
        return init_pointer, []

    def settle(self, *, round_id, block, era_index, king_pointer, work_dir):
        self.events.append(("lane_b_settle", round_id))
        if self.fail:
            raise OSError("registry 503")
        self.settled[round_id] = self.digest
        return []

    def expert_manifest_digest(self, round_id):
        return self.settled.get(round_id, "")


def _run(cfg, tmp_path, driver_kw):
    armed = _armed(cfg)
    armed = replace(armed, lane_b=parse_lane_b({"from_block": 1, "domains": ["energy"]}))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    events = []
    sched.lane_b = _Driver(events, **driver_kw)
    orig = ops.publish_manifest
    ops.publish_manifest = lambda m: (events.append(("lane_a_publish", m.round_id)), orig(m))
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    _advance(clock, ops, sched, client, from_block=b0, to_block=era_start + EB + 3)
    return events, ops.manifests


def test_expert_manifest_settles_before_and_is_bound_into_the_lane_a_manifest(cfg, tmp_path):
    events, manifests = _run(cfg, tmp_path, {})
    (m,) = manifests
    assert events == [("lane_b_settle", m.round_id), ("lane_a_publish", m.round_id)]
    assert m.expert_manifest_sha256 == "ab" * 32
    # the binding is SIGNED (in the canonical body) and round-trips
    assert b"expert_manifest_sha256" in m.canonical_body()
    assert load_manifest(dump_manifest(m)).expert_manifest_sha256 == m.expert_manifest_sha256


def test_a_failed_lane_b_settlement_binds_nothing_and_lane_a_proceeds(cfg, tmp_path):
    events, manifests = _run(cfg, tmp_path, {"fail": True})
    (m,) = manifests
    assert m.expert_manifest_sha256 == ""
    assert b"expert_manifest_sha256" not in m.canonical_body()


def test_binding_is_drop_when_default():
    m = TrainingManifest(round_id="1", created_block=2, contract_digest="1" * 64,
                         base_arch_digest="2" * 64, eval_dataset="d", entries=[])
    assert b"expert_manifest_sha256" not in m.canonical_body()
    assert replace(m, expert_manifest_sha256="").canonical_body() == m.canonical_body()
