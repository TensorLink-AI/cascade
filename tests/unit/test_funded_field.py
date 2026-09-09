"""[round] funded_mode: field selection, settle marking, and the cadence floor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue
from cascade.shared.config import RoundConfig, validate_funded_mode
from cascade.trainer.loop import TrainerRunner

REF = "ns/gen@sha256:" + "a" * 64


def _challenger(hotkey: str, ref: str = REF, reveal_block: int = 0) -> SimpleNamespace:
    return SimpleNamespace(hotkey=hotkey, uid=1, ref=ref, reveal_block=reveal_block)


def _runner(tmp_path, **round_kw):
    """A minimal stand-in: the funded methods touch only cfg.round + work_root."""
    import threading
    rnd = RoundConfig(funded_queue_path="funded_queue.json", **round_kw)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd,
                                               subnet=SimpleNamespace(netuid=91)),
                           work_root=tmp_path,
                           _funded_field={}, _funded_leg_failures={},
                           _funded_admission_info={},
                           _funded_ledger_lock=threading.Lock(),
                           _storage_dropped={},
                           _funded_roster={"seated": [], "waiting": [],
                                           "terminal": [], "outcomes": []})
    for name in ("_funded_gate_open", "_effective_funded_mode", "_burn_hotkeys",
                 "_effective_funded_pods", "_funded_queue", "_filter_funded_challengers", "_settle_funded",
                 "_skip_unfunded_round", "_submissions_path", "_payer_vault",
                 "_reconcile_funded_pods", "_record_funded_failure",
                 "_funded_admission_cap", "_probe_funded_capacity"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    return fake


def _queue(tmp_path) -> FundedQueue:
    return FundedQueue(tmp_path / "funded_queue.json")


def test_off_mode_touches_nothing(tmp_path):
    runner = _runner(tmp_path, funded_mode="off")
    field = [_challenger("hkA"), _challenger("hkB")]
    assert runner._filter_funded_challengers(field) == field
    assert not (tmp_path / "funded_queue.json").exists()
    assert not runner._skip_unfunded_round("r1")


def test_shadow_mode_reports_but_never_selects_or_mutates(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    q.add("hkC", REF, reveal_block=20)
    q.mark_in_round(["hkC"])                 # e.g. left over from a required era
    runner = _runner(tmp_path, funded_mode="shadow")
    field = [_challenger("hkA"), _challenger("hkB")]
    assert runner._filter_funded_challengers(field) == field
    assert _queue(tmp_path).get("hkA").status == "queued"   # untouched
    # Strictly read-only: shadow must not recover/expire — the observer never
    # changes the observed (audit 2026-08-29).
    assert _queue(tmp_path).get("hkC").status == "in_round"


def test_required_selects_by_queue_seniority_capped(tmp_path):
    q = _queue(tmp_path)
    q.add("hkLate", REF, reveal_block=300)
    q.add("hkEarly", REF, reveal_block=100)
    q.add("hkMid", REF, reveal_block=200)
    runner = _runner(tmp_path, funded_mode="required", finalists=1, max_finalists=2)
    field = [_challenger(hk) for hk in ("hkLate", "hkEarly", "hkMid", "hkUnfunded")]
    kept = runner._filter_funded_challengers(field)
    # finalist_cap = max(1, 2) = 2: two earliest-revealed funded entries enter.
    assert [c.hotkey for c in kept] == ["hkEarly", "hkMid"]
    reread = _queue(tmp_path)
    assert reread.get("hkEarly").status == "in_round"
    assert reread.get("hkMid").status == "in_round"
    assert reread.get("hkLate").status == "queued"          # waits, unburned


def test_required_ref_mismatch_is_terminal(tmp_path):
    # A re-revealed ref can never match again — leaving it queued would hold
    # the skip-floor open and bill a king leg per boundary forever.
    q = _queue(tmp_path)
    q.add("hkA", "ns/old@sha256:" + "b" * 64, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    kept = runner._filter_funded_challengers([_challenger("hkA", ref=REF)])
    assert kept == []
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.last_error_class) == ("failed", "ref_mismatch")
    # …and the miner can immediately fund the new ref.
    assert _queue(tmp_path).add("hkA", REF, 11) == "queued"


def test_required_burned_hotkey_is_terminal(tmp_path):
    import json

    q = _queue(tmp_path)
    q.add("hkBurned", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    (tmp_path / "trainer_submissions.json").write_text(json.dumps(["hkBurned"]))
    # The burn filter already removed the challenger from the field; the
    # funded entry must die too, not squat in the depth count.
    assert runner._filter_funded_challengers([_challenger("hkOther")]) == []
    entry = _queue(tmp_path).get("hkBurned")
    assert (entry.status, entry.last_error_class) == ("failed", "burned")


def test_required_funded_but_not_revealed_waits(tmp_path):
    q = _queue(tmp_path)
    q.add("hkGhost", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    assert runner._filter_funded_challengers([_challenger("hkOther")]) == []
    assert _queue(tmp_path).get("hkGhost").status == "queued"


def test_torn_round_recovers_then_settle_marks_done(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = [_challenger("hkA")]
    assert [c.hotkey for c in runner._filter_funded_challengers(field)] == ["hkA"]
    # Crash before settle: entry is stuck in_round. The NEXT round's filter
    # recovers it and re-selects — never burned, never dropped.
    kept = runner._filter_funded_challengers(field)
    assert [c.hotkey for c in kept] == ["hkA"]
    # Settle from the duel outcome: a trained (judged) entry goes done; a
    # re-fund then starts fresh.
    runner._settle_funded([(field[0], "challenger")],
                          [SimpleNamespace(miner_hotkey="hkA", role="challenger")])
    assert _queue(tmp_path).get("hkA").status == "done"
    assert _queue(tmp_path).add("hkA", REF, 10) == "queued"


def test_settle_guards_on_ref_and_settles_restart_recovered_entries(tmp_path):
    # A restart between heat settle and the settled-retry passes through
    # recover_in_round (in_round → queued) BEFORE settle runs, so settle
    # accepts queued entries too — guarded by ref equality: a re-fund under a
    # NEW ref since selection is a different entry and must not be settled by
    # this round (review 2026-09-02).
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    q.add("hkB", REF, reveal_block=11)
    runner = _runner(tmp_path, funded_mode="required")
    runner._funded_field = {"hkA": REF,
                            "hkB": "other/repo@sha256:" + "d" * 64}
    runner._settle_funded(
        [(_challenger("hkA"), "challenger"), (_challenger("hkB"), "challenger")],
        [SimpleNamespace(miner_hotkey="hkA", role="challenger"),
         SimpleNamespace(miner_hotkey="hkB", role="challenger")])
    # Same ref → the restart-recovered queued entry settles done.
    assert _queue(tmp_path).get("hkA").status == "done"
    # Ref changed since selection → untouched.
    assert _queue(tmp_path).get("hkB").status == "queued"


def test_settle_requeues_infra_failed_leg_unburning_the_entry(tmp_path):
    # The 2026-09-01 live finding: a duel leg lost to operator infra must NOT
    # consume the paid entry — it re-queues with one bounded attempt burned.
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._record_funded_failure("hkA", "remote challenger failed (rc=255)",
                                  miner_fault=False, error_class="infra", burn=True)
    runner._settle_funded([(field[0], "challenger")], [])    # no manifest entry
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.attempts, entry.last_error_class) == ("queued", 1, "infra")


def test_settle_fails_miner_fault_leg_terminally(tmp_path):
    # Worker rc=3 = the miner's own submission was rejected: their entry fails
    # visibly (fund again after fixing), never silently re-queues forever.
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._record_funded_failure("hkA", "miner submission rejected: corpus",
                                  miner_fault=True, error_class="generator", burn=False)
    runner._settle_funded([(field[0], "challenger")], [])
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.last_error_class) == ("failed", "generator")


def test_settle_unknown_failure_defaults_to_bounded_infra_requeue(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required")
    field = runner._filter_funded_challengers([_challenger("hkA")])
    runner._settle_funded([(field[0], "challenger")], [])    # no record at all
    entry = _queue(tmp_path).get("hkA")
    assert (entry.status, entry.attempts) == ("queued", 1)


def test_skip_unfunded_round_floor(tmp_path):
    import time

    runner = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=True)
    assert runner._skip_unfunded_round("r1")                 # empty queue: skip
    FundedQueue(tmp_path / "funded_queue.json", clock=time.time).add(
        "hkA", REF, reveal_block=10)
    assert not runner._skip_unfunded_round("r1")
    # Without the flag a required round still runs (king-only if unfunded).
    runner2 = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=False)
    assert not runner2._skip_unfunded_round("r1")


def test_skip_floor_expires_dead_entries(tmp_path):
    # An entry far older than the payer-key TTL cannot hold the boundary open:
    # the skip path expires it terminally and then skips — no perpetual
    # king-leg drain from one never-enterable fund (audit 2026-08-29).
    FundedQueue(tmp_path / "funded_queue.json", clock=lambda: 1000.0).add(
        "hkStale", REF, reveal_block=10)     # funded_at ≈ 1970 vs the real now
    runner = _runner(tmp_path, funded_mode="required", skip_unfunded_rounds=True)
    assert runner._skip_unfunded_round("r1")
    entry = _queue(tmp_path).get("hkStale")
    assert (entry.status, entry.last_error_class) == ("failed", "funding_expired")


def test_validate_funded_mode():
    assert validate_funded_mode("shadow") == "shadow"
    with pytest.raises(ValueError):
        validate_funded_mode("on")


def test_round_config_defaults_are_inert():
    rnd = RoundConfig()
    assert rnd.funded_mode == "off"
    assert rnd.skip_unfunded_rounds is False
    assert rnd.max_rounds_per_day == 1


# ── review 2026-09-02: held-back touch, no-heat cohort, settled-retry state ──


def test_held_back_entries_get_a_proof_of_life_touch(tmp_path):
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=100)
    q.add("hkB", REF, reveal_block=200)
    q.add("hkC", REF, reveal_block=300)
    runner = _runner(tmp_path, funded_mode="required", funded_field_cap=1)
    before = {e.hotkey: e.active_at for e in q.entries()}
    kept = runner._filter_funded_challengers(
        [_challenger(hk) for hk in ("hkA", "hkB", "hkC")])
    assert [c.hotkey for c in kept] == ["hkA"]
    after = {e.hotkey: e.active_at for e in _queue(tmp_path).entries()}
    # The seat holders that admission held back never rent, so nothing else
    # refreshes them — the filter itself must, or they TTL-expire while
    # actively waiting.
    assert after["hkB"] > before["hkB"]
    assert after["hkC"] > before["hkC"]


def test_no_heat_every_seated_funded_challenger_advances(tmp_path):
    from cascade.trainer.loop import TrainerRunner

    field = [_challenger(f"hk{i}") for i in range(5)]
    fake = SimpleNamespace(
        cfg=SimpleNamespace(round=RoundConfig(
            funded_mode="required", finalists=1, max_finalists=3)),
        _funded_field={c.hotkey: c.ref for c in field},
        screen_fn=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the screen must never run for an all-funded field")),
    )
    for name in ("_run_heat", "_funded_gate_open", "_effective_funded_mode"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    # finalist_cap (max(1, max_finalists)=3) < seated 5: without the funded
    # fast-path the overflow would be screened out and burned unjudged.
    finalists, heat = fake._run_heat(field, None, 100)
    assert [c.hotkey for c in finalists] == [c.hotkey for c in field]
    assert heat is None


def test_settled_retry_restores_funded_state_from_the_marker(tmp_path):
    from cascade.trainer.loop import TrainerRunner

    runner = _runner(tmp_path, funded_mode="required")
    runner._mark_heat_complete = TrainerRunner._mark_heat_complete.__get__(runner)
    runner._settled_finalists = TrainerRunner._settled_finalists.__get__(runner)
    finalists = [_challenger("hkA"), _challenger("hkB")]
    runner._funded_field = {"hkA": REF, "hkB": REF}
    runner._funded_round_sku = "L40S"
    runner._funded_admission_info = {"cap": 2, "sku": "L40S"}
    runner._funded_roster["seated"] = [{"hotkey": "hkA", "reveal_block": 5},
                                      {"hotkey": "hkB", "reveal_block": 6}]
    runner._mark_heat_complete(777, finalists, finalists)
    # The settled-retry path re-enters run_round, which resets everything —
    # the marker must hand it all back or the retry's legs bill the OPERATOR
    # and the multi-SKU king rent aborts on sku="".
    runner._funded_field = {}
    runner._funded_round_sku = ""
    runner._funded_admission_info = {}
    runner._funded_roster = {"seated": [], "waiting": [], "terminal": [],
                             "outcomes": []}
    reused = runner._settled_finalists(777, finalists)
    assert [c.hotkey for c in reused] == ["hkA", "hkB"]
    assert runner._funded_field == {"hkA": REF, "hkB": REF}
    assert runner._funded_round_sku == "L40S"
    assert runner._funded_admission_info == {"cap": 2, "sku": "L40S"}
    assert runner._funded_roster["seated"][0]["hotkey"] == "hkA"


def test_funded_config_validators_fail_loud():
    import pytest

    from cascade.shared.config import (validate_funded_field_cap,
                                       validate_funded_pod_skus)

    assert validate_funded_field_cap(0) == 0
    assert validate_funded_field_cap("8") == 8
    with pytest.raises(ValueError):
        validate_funded_field_cap(-1)            # truthy → would seat NOBODY
    assert validate_funded_pod_skus(["RTX4090", "A6000"]) == ("RTX4090", "A6000")
    with pytest.raises(ValueError):
        validate_funded_pod_skus("RTX4090")      # would iterate as characters


def test_funded_activation_block_gates_everything(tmp_path):
    # Release-then-activate: the armed config ships early and stays inert
    # until the chain reaches the announced block (2026-09-04 go-live).
    q = _queue(tmp_path)
    q.add("hkA", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required", funded_pods="rent",
                     skip_unfunded_rounds=True, funded_activation_block=1000)
    field = [_challenger("hkA"), _challenger("hkB")]
    # No block seen yet → gate CLOSED (armed config must never leak early):
    # legacy rounds keep running, nothing is skipped, nothing rents.
    assert runner._filter_funded_challengers(field) == field
    assert not runner._skip_unfunded_round("r1")
    assert runner._effective_funded_pods() == "off"
    # Pre-activation block → still closed.
    runner._funded_gate_block = 999
    assert runner._filter_funded_challengers(field) == field
    assert _queue(tmp_path).get("hkA").status == "queued"    # untouched
    # At the block → configured modes apply: queue becomes the field.
    runner._funded_gate_block = 1000
    kept = runner._filter_funded_challengers(field)
    assert [c.hotkey for c in kept] == ["hkA"]
    assert runner._effective_funded_pods() == "rent"
    # No gate configured (0) → open immediately, block or not (compat).
    open_runner = _runner(tmp_path, funded_mode="required")
    assert open_runner._effective_funded_mode() == "required"


def test_funded_burn_happens_at_settle_per_outcome(tmp_path):
    # one_submission_per_hotkey = true (mainnet): a hotkey's submission is
    # spent only when its funded leg was actually JUDGED (trained) or its own
    # generator failed — never by a requeue (sold-out / rate limit / infra)
    # or an auth-class key fault, which stay re-fundable. Burning at the heat
    # settle (the legacy shape) would terminally fail the requeued entry as
    # "burned" at the next round's filter (review 2026-09-02).
    from cascade.trainer.loop import _load_seen_hotkeys

    q = _queue(tmp_path)
    for hk in ("hkTrained", "hkGen", "hkAuth", "hkInfra", "hkRate"):
        q.add(hk, REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required",
                     one_submission_per_hotkey=True, funded_field_cap=8)
    field = [_challenger(hk) for hk in ("hkTrained", "hkGen", "hkAuth",
                                        "hkInfra", "hkRate")]
    assert len(runner._filter_funded_challengers(field)) == 5
    # Nothing is burned at admission / heat time.
    assert _load_seen_hotkeys(tmp_path / "trainer_submissions.json") == set()
    runner._record_funded_failure("hkGen", "rc=3", miner_fault=True,
                                  error_class="generator", burn=False)
    runner._record_funded_failure("hkAuth", "bad key", miner_fault=True,
                                  error_class="auth", burn=False)
    runner._record_funded_failure("hkInfra", "pod died", miner_fault=False,
                                  error_class="infra", burn=True)
    runner._record_funded_failure("hkRate", "429", miner_fault=False,
                                  error_class="rate_limited", burn=False)
    runner._settle_funded([(c, "challenger") for c in field],
                          [SimpleNamespace(miner_hotkey="hkTrained", role="challenger")])
    burned = _load_seen_hotkeys(tmp_path / "trainer_submissions.json")
    assert burned == {"hkTrained", "hkGen"}
    statuses = {hk: _queue(tmp_path).get(hk).status for hk in
                ("hkTrained", "hkGen", "hkAuth", "hkInfra", "hkRate")}
    assert statuses == {"hkTrained": "done", "hkGen": "failed",
                        "hkAuth": "failed", "hkInfra": "queued",
                        "hkRate": "queued"}
    # The requeued ones re-enter the next round instead of dying as "burned".
    nxt = runner._filter_funded_challengers([_challenger("hkInfra"),
                                             _challenger("hkRate")])
    assert sorted(c.hotkey for c in nxt) == ["hkInfra", "hkRate"]


def test_settle_burns_a_tampered_leg(tmp_path):
    from cascade.trainer.loop import _load_seen_hotkeys

    q = _queue(tmp_path)
    q.add("hkT", REF, reveal_block=10)
    runner = _runner(tmp_path, funded_mode="required", one_submission_per_hotkey=True)
    assert runner._filter_funded_challengers([_challenger("hkT")])
    runner._record_funded_failure("hkT", "pod identity: pod id changed", miner_fault=True,
                                  error_class="tamper", burn=False)
    runner._settle_funded([(_challenger("hkT"), "challenger")], [])
    assert _queue(tmp_path).get("hkT").status == "failed"
    assert _queue(tmp_path).get("hkT").last_error_class == "tamper"
    assert "hkT" in _load_seen_hotkeys(tmp_path / "trainer_submissions.json")
