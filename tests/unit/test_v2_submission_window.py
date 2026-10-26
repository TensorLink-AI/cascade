"""The opening queues A entries; only v2 may train them, without early burns."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from cascade.shared.calendar_launch import (
    effective_commit_floor,
    lane_a_launch_gate,
    launch_timestamp,
    resolve_calendar_launch,
)
from cascade.shared.config import load_chain_config


def test_opening_and_launch_resolve_independently_for_live_and_late_nodes():
    from tests.unit.test_calendar_launch import _chain

    cfg = load_chain_config()
    launch = launch_timestamp(cfg.v2.launch_at_utc)
    assert launch - launch_timestamp(cfg.v2.commit_open_at_utc) == 3 * 3600
    before = resolve_calendar_launch(cfg, _chain(launch, 29100))
    assert before is cfg
    opened = resolve_calendar_launch(cfg, _chain(launch, 29101))
    assert opened.v2.commit_from_block == 29101 and opened.v2.from_block == 0
    assert lane_a_launch_gate(before, 29100)[1]["code"] == "v2_submissions_not_open"
    assert lane_a_launch_gate(opened, 29101, 29101) is None
    assert lane_a_launch_gate(opened, 29101, 29100)[1]["code"] == "v2_commit_before_open"
    first = resolve_calendar_launch(opened, _chain(launch, 30001))
    late = resolve_calendar_launch(cfg, _chain(launch, 100000))
    assert first.v2.commit_from_block == late.v2.commit_from_block == 29101
    assert first.v2.from_block == late.v2.from_block == 43200
    assert lane_a_launch_gate(late, 100000, 99999) is None  # Never closes again.
    assert effective_commit_floor(first, 1) == cfg.round.commit_floor_block
    assert resolve_calendar_launch(first, object()) is first


@pytest.mark.parametrize("state", ["closed", "old_reveal", "unavailable"])
def test_paid_intake_refuses_before_queue_and_credential_changes(tmp_path, cfg, state):
    from tests.unit.test_funding_intake import HK, REF, make_intake

    intake, _ = make_intake(tmp_path, resolve=lambda hk, ref: 499)
    view = replace(cfg, v2=replace(cfg.v2, from_block=1000,
        commit_open_at_utc="2026-10-30T09:00:00Z", commit_from_block=0 if state == "closed" else 500))

    def admission(block):
        if state == "unavailable":
            raise OSError("chain unavailable")
        return lane_a_launch_gate(view, 1001, block)

    intake.lane_a_admission = admission
    status, body = intake.fund({"X-Miner-Hotkey": HK, "X-Commit-Ref": REF,
                                "X-Lium-Api-Key": "fixture-only"})
    assert status == {"closed": 409, "old_reveal": 403, "unavailable": 503}[state]
    assert body["code"] in {"v2_submissions_not_open", "v2_commit_before_open", "launch_rules_unavailable"}
    assert intake.queue.get(HK) is None and intake.vault.get(HK) is None
    if state == "closed":
        data = b"not even parsed as an upload"
        code, _ = intake.submit({"X-Lium-Api-Key": "fixture-only"}, data,
            gate_ctx={"hotkey": HK, "declared": "sha256:" + hashlib.sha256(data).hexdigest()})
        assert code == 409 and intake.queue.get(HK) is None


def test_queued_work_survives_restart_and_only_dispatches_after_v2(cfg, tmp_path, monkeypatch):
    from cascade.trainer.rolling import EraState
    from tests.unit.test_dec55_trainer_epochs import _v2
    from tests.unit.test_rolling import REF, Clock, _commit, _sched

    cfg, start, _ = _v2(cfg)
    cfg = replace(cfg, v2=replace(cfg.v2, commit_from_block=start - 900,
                                  commit_open_at_utc="2026-10-30T09:00:00Z"))
    clock = Clock()
    sched, ops = _sched(cfg, tmp_path, clock)
    queue = ops.queue()
    queue.add("ALFA", REF["ALFA"], start - 800)
    sched.state.current = EraState(start // 3600 - 1, start - 3600, start - 3600, 7)
    sched._intake(object(), start - 1, clock())
    assert queue.get("ALFA").status == "queued"
    assert queue.get("ALFA").phase == "waiting_launch"
    assert queue.get("ALFA").attempts == 0 and not ops.leg_calls and not ops.burnt
    resumed, ops2 = _sched(cfg, tmp_path, clock)
    resumed.state.current = EraState(start // 3600, start, start, 7)
    ops2.commits.append(_commit("ALFA", REF["ALFA"], start - 800))
    launched = []
    monkeypatch.setattr(resumed, "_launch_leg", lambda *a: launched.append(a))
    resumed._intake(SimpleNamespace(), start + 1, clock())
    assert len(launched) == 1 and ops2.queue().get("ALFA").status == "in_flight"
    assert not ops2.burnt


def test_trainer_refuses_old_reveals_without_burning(cfg, tmp_path):
    from cascade.trainer.rolling import EraState
    from tests.unit.test_dec55_trainer_epochs import _v2
    from tests.unit.test_rolling import REF, Clock, _commit, _sched

    cfg, start, _ = _v2(cfg)
    cfg = replace(cfg, v2=replace(cfg.v2, commit_from_block=start - 900))
    clock = Clock()
    sched, ops = _sched(cfg, tmp_path, clock)
    sched.state.current = EraState(start // 3600, start, start, 7)
    ops.queue().add("ALFA", REF["ALFA"], start - 901)
    ops.commits.append(_commit("ALFA", REF["ALFA"], start - 901))
    sched._intake(object(), start + 1, clock())
    entry = ops.queue().get("ALFA")
    assert entry.status == "failed" and entry.last_error_class == "commit_before_open"
    assert entry.attempts == 0 and not ops.leg_calls and not ops.burnt


def test_launch_hold_refreshes_queue_liveness_without_resetting_stage_or_attempts(tmp_path):
    from cascade.funding.queue import FundedQueue
    from cascade.miner.submission_status import submission_view
    from tests.unit.test_submission_tracking import HK, REF, snapshot

    clock = [1_000_000.0]
    queue = FundedQueue(tmp_path / "queued.json", clock=lambda: clock[0])
    queue.add(HK, REF, 500)
    queue.hold_for_launch()
    first = queue.get(HK)
    clock[0] += 3 * 3600
    queue.hold_for_launch()
    held = queue.get(HK)
    assert held.last_active == clock[0] and held.phase_at == first.phase_at
    assert held.attempts == 0 and held.status == "queued"
    view = submission_view(snapshot(), queue=queue.public_view(), hotkey=HK, lane="a", ref=REF,
                           now=datetime.fromtimestamp(clock[0], UTC))
    assert view["status"] == "waiting_launch"
    assert "v2 activation" in view["next_action"]


def test_fresh_recommit_of_the_same_ref_uses_chain_order_and_eligible_seniority(tmp_path):
    from cascade.funding.queue import FundedQueue
    from cascade.trainer.rolling import RollingScheduler
    from tests.unit.test_rolling import REF, _commit

    history = [_commit("ALFA", REF["ALFA"], 499), _commit("ALFA", REF["ALFA"], 501),
               _commit("ALFA", REF["ALFA"], 700)]
    assert RollingScheduler._reveal_block(history, "ALFA", REF["ALFA"], 600) == 501
    queue = FundedQueue(tmp_path / "q.json")
    queue.add("ALFA", REF["ALFA"], 499)
    first = queue.get("ALFA")
    queue.rebase_preopening_reveals(500,
        lambda hk, ref: RollingScheduler._reveal_block(history, hk, ref, 600))
    updated = queue.get("ALFA")
    assert updated.reveal_block == 501 and updated.attempts == 0
    assert updated.funded_at == first.funded_at and updated.ref == first.ref


def test_cli_treats_unknown_launch_rules_as_retryable_without_hiding_the_code(monkeypatch):
    from cascade.miner.cli import post_intake

    data = json.dumps({"code": "launch_rules_unavailable", "message": "Chain read unavailable"}).encode()

    def unavailable(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", None, io.BytesIO(data))

    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    out = SimpleNamespace(fail=lambda *args, **kw: calls.append((args, kw)))
    assert post_intake(urllib.request.Request("https://fixture.invalid/v1/fund"), out, timeout=30) is None
    assert calls[0][0][0] == "launch_rules_unavailable" and calls[0][0][2] == 3


def test_running_intake_remembers_opening_when_the_node_later_prunes_it():
    from cascade.funding.main import ChainLaunchAdmission
    from tests.unit.test_calendar_launch import _chain

    cfg = load_chain_config()
    target = launch_timestamp(cfg.v2.launch_at_utc)
    chain = _chain(target, 29101)
    admission = ChainLaunchAdmission(cfg, chain)
    assert admission(29101) is None
    assert admission.cfg.v2.commit_from_block == 29101
    chain.finalized_block = lambda: 30001
    timestamp = chain.block_timestamp

    def pruned(block):
        if block < 29900:
            raise RuntimeError("old state pruned")
        return timestamp(block)

    chain.block_timestamp = pruned
    # Re-reading the startup cfg loses the observed opening and fails here.
    with pytest.raises(RuntimeError, match="archive"):
        resolve_calendar_launch(cfg, chain)
    assert admission(30001) is None
    assert admission.cfg.v2.from_block == 43200
    chain.finalized_block = lambda: 43201
    chain.block_timestamp = lambda _: pytest.fail("resolved pins must not be read again")
    assert admission(43201) is None
