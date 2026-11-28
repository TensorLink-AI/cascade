"""W-E2E lane D: one full cycle (submit -> intake -> pool build/pin -> settle
-> publish -> verify -> miner status) on fake infrastructure, and the cycle
catching this week's lane D failure classes.

The bugs below are fixed on main, so each regression test applies a
revert-patch (the pre-fix behaviour) and shows the SAME cycle then fails by
assertion; without the patch the cycle passes (``test_lane_d_full_cycle``).
Lane D trains nothing, so the leg classes (re-attach with no pod, GPU OOM,
NaN loss) belong to the lane B / C cycles. tests/e2e_lanes/lane_d.py
documents every stage."""

from __future__ import annotations

import json

import pytest

from tests.e2e_lanes import lane_d as cyc
from tests.e2e_lanes.harness import BoxBudgetExceeded, box_guard


@pytest.fixture
def world(tmp_path, monkeypatch, capsys, caplog):
    return cyc.make_world(tmp_path, monkeypatch, capsys, caplog)


def _until_settle(w):
    for stage in (cyc.stage_submit, cyc.stage_intake, cyc.stage_pool_build, cyc.stage_pool_pin):
        stage(w)
    cyc._alias_entry(w)


def test_lane_d_full_cycle(world):
    from cascade.validator.lane_d_wiring import report_key

    w = world
    cycle = cyc.run_cycle(w)
    assert cycle.names() == list(cyc.STAGES)
    # the crowning report is the one the outside audit passed
    assert json.loads(w.ns.v.store.objects[report_key(cyc.B1)]) == w.ns.report
    # the box never loaded a pinned snapshot; the eval host settled both boundaries
    assert w.ns.v.pool_reads == []
    assert [c[0] for c in w.ns.v.host._run.calls].count("settle") == 2
    # the eval host's replies carried download chatter on stdout
    assert w.ns.v.host._run.chatter.startswith(b"Downloading")


# ── #1000 (W-OFFBOX): the pool phase loaded in the validator process ──────────


def _pre_offbox(monkeypatch):
    """Pre-#1000: the inputs file's eval_pool_root was not read; every pin
    loaded in the validator process with no box budget."""
    from cascade.validator import lane_d_inputs

    real = lane_d_inputs.load_inputs_services

    def pre_fix(cfg, path):
        out = real(cfg, path)
        out.update(settle_on_host=False, box_pool_budget_bytes=1 << 50)
        return out

    monkeypatch.setattr(lane_d_inputs, "load_inputs_services", pre_fix)


def test_in_process_pool_load_trips_the_box_tripwire(world, monkeypatch):
    _pre_offbox(monkeypatch)
    with box_guard():
        _until_settle(world)
        world.ns.v = cyc.validator(world, "validator")
        with pytest.raises(AssertionError, match="loaded pinned pool_d file"):
            cyc.run_settlements(world, world.ns.v)
    assert world.ns.v.pool_reads, "the tripwire saw no pool_d read"


def test_in_process_pool_load_blows_the_box_budget(world, monkeypatch):
    """The same revert, judged by memory alone (no tripwire)."""
    _pre_offbox(monkeypatch)
    with box_guard():
        _until_settle(world)
        world.ns.v = cyc.validator(world, "validator")
        with pytest.raises(BoxBudgetExceeded):
            cyc.run_settlements(world, world.ns.v, tripwire=False)


# ── #976 / #997: a frozen builder mirror re-cut the same data ────────────────


def _frozen_pool(w):
    return [cyc.make_pool_d1(cyc.Path(w.fx["chain"]["pools"][1])), cyc.frozen_snapshot(w)]


def test_frozen_mirror_snapshot_is_refused_before_the_validator(world):
    w = world
    with box_guard():
        cyc.stage_submit(w)
        cyc.stage_intake(w)
        with pytest.raises(cyc.CycleFailed, match="stale_data"):
            cyc.stage_pool_pin(w, snapshots=_frozen_pool(w))
    assert [s for s, _ in cyc.pinned_first_seen(w.ns.pins_path)] == ["w7a-1"]


def test_frozen_mirror_without_the_guard_is_caught_by_the_cycle(world, monkeypatch):
    """Pre-#976/#997: publish and pin sync had no data-advance guard; the
    cycle's own check refuses the pinned stale snapshot."""
    from cascade.pool import pool_d_publish as pub

    monkeypatch.setattr(pub, "_judge_freshness",
                        lambda name, as_of, newest, *, previous, max_age_seconds=0:
                        {"max_first_seen_ts": newest, "data_age_seconds": 0})
    w = world
    with box_guard():
        cyc.stage_submit(w)
        cyc.stage_intake(w)
        with pytest.raises(AssertionError, match="carries no newer data"):
            cyc.stage_pool_pin(w, snapshots=_frozen_pool(w))


# ── #994 R13: the builder synced its release only at bootstrap ───────────────


def test_builder_without_per_build_release_sync_is_caught(world, monkeypatch):
    from cascade.laned import elastic

    real = elastic.builder_run_steps
    monkeypatch.setattr(elastic, "builder_run_steps",
                        lambda cfg, inst, *, now_ts: real(cfg, inst, now_ts=now_ts)[1:])
    with box_guard(), pytest.raises(AssertionError, match="release sync"):
        cyc.stage_pool_build(world)


def test_builder_sync_that_ignores_a_release_change_is_caught(world, monkeypatch):
    """Pre-#994: the sync ran once (bootstrap); a later release commit was
    never picked up."""
    from cascade.laned import elastic

    real = elastic.release_sync_script
    seen = []

    def once(cfg, inst, *, force=False):
        seen.append(1)
        return real(cfg, inst, force=force) if len(seen) == 1 else "true"

    monkeypatch.setattr(elastic, "release_sync_script", once)
    with box_guard(), pytest.raises(AssertionError, match="stale code"):
        cyc.stage_pool_build(world)


# ── #983 class: an eval-host reply with download chatter on stdout ───────────


def test_eval_host_reply_parsed_as_whole_stdout_is_caught(world, monkeypatch):
    """Pre-#983 (lane B's copy of this parser): the whole stdout was the JSON
    reply. Real eval hosts print download progress first."""
    import subprocess

    from cascade.validator.lane_d_inputs import SSHEvalHost

    def whole_stdout(self, host, args, payload, timeout_s=None):
        cmd = self.remote_command(host, args, timeout_s=timeout_s)
        proc = self._run(["ssh", cmd], input=payload, capture_output=True,
                         timeout=timeout_s, check=False)
        assert isinstance(proc, subprocess.CompletedProcess)
        return proc.stdout.decode("utf-8")

    monkeypatch.setattr(SSHEvalHost, "_ssh", whole_stdout)
    with box_guard():
        _until_settle(world)
        world.ns.v = cyc.validator(world, "validator")
        with pytest.raises(cyc.CycleFailed, match="Expecting value|JSONDecodeError"):
            cyc.run_settlements(world, world.ns.v)
