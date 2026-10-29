"""The scheduler tick never wedges on one slow step (T9): chain reads run
under the runner's deadline and a timeout skips only that step; lane-B
settlement runs off the tick thread, still strictly before its lane-A
manifest, and is abandoned past a bound."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config import parse_lane_b
from cascade.trainer.rolling import LegOps

from .test_rolling import (
    EB,
    REF,
    Clock,
    FakeClient,
    FakeOps,
    _advance,
    _armed,
    _commit,
    _join,
    _sched,
)


def test_chain_reads_run_under_the_runner_deadline():
    seen = []

    def with_deadline(fn, seconds):
        seen.append(seconds)
        raise TimeoutError("hung websocket")

    ops = LegOps(SimpleNamespace(_with_deadline=with_deadline, chain_read_timeout_s=5.0))
    client = SimpleNamespace(block_seed=lambda b: 1, poll_commitments=lambda **kw: [],
                             highest_incentive_hotkey=lambda **kw: "K")
    with pytest.raises(TimeoutError):
        ops.block_seed(client, 10)
    with pytest.raises(TimeoutError):
        ops.commitments(client)
    assert ops.metagraph_king(client) is None           # a cross-check: None on timeout
    assert seen == [5.0, 5.0, 5.0]
    # no runner helper (unit fakes): a plain call
    assert LegOps(SimpleNamespace()).block_seed(client, 10) == 1


class _HangOps(FakeOps):
    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.hang = False

    def block_seed(self, client, block):
        if self.hang:
            raise TimeoutError("chain read deadline")
        return super().block_seed(client, block)


def test_a_timed_out_chain_read_skips_the_settlement_not_the_tick(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock, ops=_HangOps(tmp_path, Clock()))
    ops.clock = clock
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100), _commit("BRAV", REF["BRAV"], b0 - 50)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    ops.hang = True
    ops.queue().add("BRAV", REF["BRAV"], reveal_block=b0 - 50)
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)    # does not raise
    assert ops.manifests == [] and sched.state.last_settled_boundary < b1
    assert "BRAV" in [c[0] for c in ops.leg_calls]     # intake still ran
    ops.hang = False
    _advance(clock, ops, sched, client, from_block=b1 + 3, to_block=b1 + 4)
    assert len(ops.manifests) == 1 and sched.state.last_settled_boundary == b1


class _SlowDriver:
    def __init__(self, digest="cd" * 32):
        self.release = threading.Event()
        self.settled: dict[str, str] = {}
        self.abandoned: list[str] = []
        self.digest = digest
        self.calls = 0

    def tick(self, block, era_index, king_pointer):
        return []

    def add_lineage_tokens(self, n):
        pass

    def init_for(self, era_index, init_pointer, work_dir, era_start=None):
        return init_pointer, []

    def settle(self, *, round_id, block, era_index, king_pointer, work_dir):
        self.calls += 1
        self.release.wait(10)
        if round_id not in self.abandoned:
            self.settled[round_id] = self.digest
        return []

    def expert_manifest_digest(self, round_id):
        return self.settled.get(round_id, "")

    def abandon(self, round_id):
        self.abandoned.append(round_id)
        return self.settled.get(round_id, "")


def _lane_b_setup(cfg, tmp_path):
    armed = _armed(cfg)
    armed = replace(armed, lane_b=parse_lane_b({"from_block": 1, "domains": ["energy"]}))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    sched.lane_b = _SlowDriver()
    sched.LANE_B_SETTLE_WAIT_SECONDS = 0.05
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100), _commit("BRAV", REF["BRAV"], b0 - 50)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    return sched, ops, clock, client, era_start, b0


def test_a_slow_lane_b_settlement_holds_only_its_lane_a_manifest(cfg, tmp_path):
    sched, ops, clock, client, era_start, b0 = _lane_b_setup(cfg, tmp_path)
    b1 = era_start + EB
    ops.queue().add("BRAV", REF["BRAV"], reveal_block=b0 - 50)
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert ops.manifests == [] and sched.state.last_settled_boundary < b1
    assert "BRAV" in [c[0] for c in ops.leg_calls]     # intake ran meanwhile
    _advance(clock, ops, sched, client, from_block=b1 + 3, to_block=b1 + 4)
    assert ops.manifests == [] and sched.lane_b.calls == 1    # one settlement, never two
    sched.lane_b.release.set()
    sched._lane_b_slot["thread"].join(5)
    _advance(clock, ops, sched, client, from_block=b1 + 4, to_block=b1 + 5)
    (m,) = ops.manifests
    assert m.expert_manifest_sha256 == "cd" * 32       # bound, and published AFTER lane B
    # legs that finished while the boundary waited ride it (like a late settlement)
    assert [e.miner_hotkey for e in m.entries][:2] == ["KING", "ALFA"]
    assert sched._lane_b_slot is None


def test_a_lane_b_settlement_past_its_bound_is_abandoned(cfg, tmp_path):
    sched, ops, clock, client, era_start, b0 = _lane_b_setup(cfg, tmp_path)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert ops.manifests == []
    clock.t += sched.LANE_B_SETTLE_MAX_SECONDS
    sched.tick(client, b1 + 4)
    _join(sched)
    (m,) = ops.manifests
    assert m.expert_manifest_sha256 == ""              # nothing the driver could still change
    assert sched.lane_b.abandoned == [m.round_id]
    sched.lane_b.release.set()
    sched._lane_b_slot["thread"].join(5)
    assert m.round_id not in sched.lane_b.settled      # the driver committed nothing after


def test_the_king_is_pinned_only_while_its_lane_b_boundary_is_open(cfg, tmp_path):
    sched, ops, clock, client, era_start, b0 = _lane_b_setup(cfg, tmp_path)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert sched._lane_b_pinned()                      # dethrone adoption waits
    rid = sched._lane_b_slot["rid"]
    sched.state.last_settled_boundary = b1             # the boundary closed some other way
    assert not sched._lane_b_pinned()
    assert sched.lane_b.abandoned == [rid]             # it can no longer commit
    sched.lane_b.release.set()
    sched._lane_b_slot["thread"].join(5)
    assert not sched._lane_b_pinned() and sched._lane_b_slot is not None


def test_metagraph_king_never_names_the_burn_uid():
    """While no king is paid the burn slot out-earns every miner; the
    fallback king must skip it, or the era waits on a hotkey with no generator."""
    seen = {}

    def highest(exclude_uids=()):
        seen["exclude"] = exclude_uids
        return "K"

    r = SimpleNamespace(cfg=SimpleNamespace(scoring=SimpleNamespace(burn_uid=0)))
    assert LegOps(r).metagraph_king(SimpleNamespace(highest_incentive_hotkey=highest)) == "K"
    assert seen["exclude"] == (0,)
