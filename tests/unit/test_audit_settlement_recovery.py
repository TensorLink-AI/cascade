"""Failure injection at the A/B/C settlement persistence boundaries."""

from dataclasses import replace

import pytest

from cascade.shared.lane_b import DomainWin
from cascade.shared.lane_c import TargetWin
from cascade.shared.settlement_state import state_digest
from cascade.validator import settlement_transaction as tx
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis


def runner(cfg, tmp_path):
    cfg = replace(cfg, validator=replace(cfg.validator,
                                        state_db_path=str(tmp_path / "state.json")),
                  lane_b=replace(cfg.lane_b, from_block=1),
                  lane_c=replace(cfg.lane_c, from_block=1),
                  netconfig=replace(cfg.netconfig, owner_hotkey="owner"))
    return ValidatorRunner(cfg=cfg, state=genesis("king", 1), verify_signatures=False)


def record_win(r):
    r._lane_b().record(DomainWin("energy", "miner", 2, "100", 100,
                                 target_lcb=0.02, tier=0.02, full_eras=5))
    r._lane_b().mark_applied("100", 100)


def test_rejudgement_restores_lane_b_prior_digest(cfg, tmp_path):
    r = runner(cfg, tmp_path)
    before = state_digest(r._satellite_state())
    pre = r._snapshot("100")
    record_win(r)
    r.state = replace(r.state, pre_round=pre, last_handled_round_id="100",
                      followup_round_id="100")
    assert r._restore_pre_round("100")
    assert state_digest(r._satellite_state()) == before
    assert not r._lane_b().applied


def test_hold_rolls_back_lane_b_acceptance_ledger(cfg, tmp_path):
    r = runner(cfg, tmp_path)
    before = r._lane_b().to_json()
    point = r._rollback_point()
    record_win(r)
    r._lane_b().accepted["delta"] = 100
    r._apply_rollback(point)
    assert r._lane_b().to_json() == before


@pytest.mark.parametrize("lane", ["b", "c", "champion"])
def test_persistence_failure_is_not_success(cfg, tmp_path, monkeypatch, lane):
    r = runner(cfg, tmp_path)

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    if lane == "champion":
        monkeypatch.setattr("cascade.shared.atomic.write_text_atomic", disk_full)
        persist = r._persist_state
    else:
        monkeypatch.setattr(f"cascade.shared.lane_{lane}.save_state_atomic", disk_full)
        persist = getattr(r, f"_persist_lane_{lane}")
    with pytest.raises(OSError, match="disk full"):
        persist()


@pytest.mark.parametrize("boundary", ["lane_b", "lane_c", "handled"])
def test_restart_between_state_writes_restores_all_lanes(cfg, tmp_path, boundary):
    r = runner(cfg, tmp_path)
    r._persist_state()
    prior = state_digest(r._satellite_state())
    tx.begin(r)
    record_win(r)
    r._persist_lane_b()
    if boundary in ("lane_c", "handled"):
        r._lane_c().slots["mkt:returns"] = TargetWin(
            "mkt:returns", "other", 3, "100", 100, 0, "ptr", "recipe", "base", 0.03,
            tier=0.02, full_eras=5)
        r._persist_lane_c()
    if boundary == "handled":
        r.state = replace(r.state, rounds_seen=1, last_handled_round_id="100")
        r._persist_state()
    # Discard all in-memory objects as at a process kill, before journal commit.
    from cascade.validator.loop import _load_state

    restarted = runner(cfg, tmp_path)
    restarted.state = _load_state(restarted.cfg.validator.state_db_path)
    assert tx.recover(restarted)
    assert state_digest(restarted._satellite_state()) == prior
    assert not tx.recover(restarted)
    again = runner(cfg, tmp_path)
    again.state = _load_state(again.cfg.validator.state_db_path)
    assert state_digest(again._satellite_state()) == prior


def test_committed_journal_preserves_followup_and_rewards(cfg, tmp_path):
    r = runner(cfg, tmp_path)
    tx.begin(r)
    record_win(r)
    r._persist_lane_b()
    r.state = replace(r.state, last_handled_round_id="100", followup_round_id="100")
    r._persist_state()
    tx.commit(r)
    assert not tx.recover(r)
    assert r.state.followup_round_id == "100"
    assert r._lane_b().wins["energy"].hotkey == "miner"


def test_recovery_disk_failure_keeps_undo_for_retry(cfg, tmp_path, monkeypatch):
    r = runner(cfg, tmp_path)
    tx.begin(r)
    record_win(r)
    r._persist_lane_b()
    original = r._persist_lane_b
    monkeypatch.setattr(r, "_persist_lane_b", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        tx.recover(r)
    assert '"prepared"' in tx.journal_path(r).read_text()
    monkeypatch.setattr(r, "_persist_lane_b", original)
    assert tx.recover(r)
    assert not r._lane_b().wins


def test_only_one_validator_can_write_a_state_bundle(tmp_path):
    from cascade.shared.atomic import acquire_state_lock

    path = tmp_path / "validator.json.lock"
    first = acquire_state_lock(path)
    try:
        with pytest.raises(RuntimeError, match="already holds"):
            acquire_state_lock(path)
    finally:
        first.close()
    with acquire_state_lock(path):
        pass  # a stopped/crashed process releases the lock


def test_promotion_persistence_failure_is_not_acknowledged(tmp_path, monkeypatch):
    from cascade.validator import cascade

    controller = cascade.CascadeController(reign_days=7, state_path=tmp_path / "cascade.json")

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cascade, "write_text_atomic", disk_full)
    with pytest.raises(OSError, match="disk full"):
        controller._persist()


def test_directory_fsync_failure_is_not_a_durable_success(tmp_path, monkeypatch):
    import os
    import stat

    from cascade.shared.atomic import write_text_atomic

    fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory fsync failed"):
        write_text_atomic(tmp_path / "state.json", '{}')
