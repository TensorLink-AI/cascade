"""The miner waste cap (owner 2026-11-03) and the payer-balance pre-rent check.

Pods on a miner's key that produced nothing are summed per submission; at
``[round] miner_waste_cap_usd`` the entry pauses (``paused_spend_cap``) —
never failed, burned, expired or rejected — and infra waste files a
``spend_cap_make_good`` owner approval to finish the leg on the operator's
account."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace

import pytest

from cascade.funding.miner_spend import MinerSpend
from cascade.funding.operator_approvals import OperatorApprovals
from cascade.funding.queue import PAUSED_SPEND_STATUS, FundedQueue
from cascade.provision import funded as funded_mod
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.operator_legs_cli import main as cli
from tests.unit.test_funded_pod_wiring import REF, _challenger, _rent_ok, _runner, _vault
from tests.unit.test_money_safety import _edited

HK = "hkA"


def _setup(tmp_path, monkeypatch, **round_kw):
    r = _runner(tmp_path, **round_kw)
    _vault(tmp_path, HK)
    q = FundedQueue(tmp_path / "funded_queue.json")
    q.add(HK, REF, 10)
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    r._funded_queue = lambda: q
    r._leg_wall_seconds = lambda sku=None: 4 * 3600.0
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: rents.append(kw) or _rent_ok(HK))
    return r, q, rents


def _seed(tmp_path, q, pods):
    """Write the spend ledger as the rent/teardown sites would have: pods
    ``(pod_id, $/h, hours, outcome, kind)`` for the CURRENT funding."""
    e = q.get(HK)
    now = time.time()
    rows = [{"pod_id": pid, "rate": rate, "start": now - 3600 * h - 1, "end": now - 1,
             "outcome": outcome, "kind": kind} for pid, rate, h, outcome, kind in pods]
    (tmp_path / "miner_spend.json").write_text(json.dumps({"records": {HK: {
        "ref": e.ref, "funded_at": e.funded_at, "pods": rows, "capped": False}}}))


def _settle(r, q):
    msg, miner_fault, cls, burn = r._funded_leg_failures[HK]
    q.requeue(HK, error=msg, error_class=cls, burn_attempt=burn)
    return cls


def test_knob_round_trips_with_an_integral_float(tmp_path):
    assert RoundConfig().miner_waste_cap_usd == 30.0
    assert load_chain_config(_edited(tmp_path, "miner_waste_cap_usd = 45\n")
                             ).round.miner_waste_cap_usd == 45.0
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="miner_waste_cap_usd"):
        load_chain_config(_edited(tmp_path / "b", "miner_waste_cap_usd = -1\n"))


def test_replay_uid145_pauses_after_the_first_loss(tmp_path, monkeypatch, caplog):
    """Tonight: an $86 our-side loss, then a $4 lemon and a $60 unreachable
    host would have followed. At cap $30 the next rent never happens."""
    r, q, rents = _setup(tmp_path, monkeypatch)
    _seed(tmp_path, q, [("p1", 21.5, 4.0, "waste", "infra")])          # $86, our side
    with caplog.at_level(logging.ERROR), pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    assert rents == []
    assert _settle(r, q) == "spend_cap"
    e = q.get(HK)
    assert e.status == PAUSED_SPEND_STATUS and e.attempts == 0
    assert "your submission is kept" in e.last_error
    (req,) = OperatorApprovals.for_work_root(tmp_path).pending()
    assert req["kind"] == "spend_cap_make_good" and req["miner_lost_usd"] == pytest.approx(86, abs=1)
    view = q.public_view()["entries"][0]
    assert view["miner_waste_usd"] == 86 and view["status"] == PAUSED_SPEND_STATUS
    # never failed, expired or rejected while paused
    assert q.expire_stale(max_age_seconds=0) == 0
    q.hold_for_launch()
    q.rebase_preopening_reveals(10 ** 9, lambda hk, ref: 10 ** 9)
    assert q.get(HK).status == PAUSED_SPEND_STATUS and q.get(HK).reveal_block == 10


def test_a_successful_attempt_is_never_waste(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    _seed(tmp_path, q, [("p1", 40.0, 4.0, "result", "infra")])         # $160, the result
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1


def test_the_accounting_closes_at_the_real_sites(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    host, pod = r._rent_funded_host("777", _challenger(HK))
    led = MinerSpend(tmp_path / "miner_spend.json")
    s = led.summary(HK, ref=q.get(HK).ref, funded_at=q.get(HK).funded_at)
    assert [p["pod_id"] for p in s.pods] == [pod.instance_id] and s.pods[0]["end"] is None
    from cascade.trainer.miner_spend_guard import spend_end

    spend_end(r, HK, pod.instance_id, outcome="result")
    spend_end(r, HK, pod.instance_id, outcome="waste")                 # a later teardown: no-op
    s = led.summary(HK, ref=q.get(HK).ref, funded_at=q.get(HK).funded_at)
    assert s.pods[0]["outcome"] == "result" and s.waste_usd == 0


def test_a_refund_resets_the_count(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    _seed(tmp_path, q, [("p1", 30.0, 2.0, "waste", "infra")])
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    _settle(r, q)
    assert q.add(HK, REF, 10) == "queued"                              # the miner re-funds
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1 and not q.get(HK).operator_billed


def test_the_owner_resume_resets_the_count(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    _seed(tmp_path, q, [("p1", 30.0, 2.0, "waste", "infra")])
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    _settle(r, q)
    assert cli(["resume-operator-leg", HK, "--work-root", str(tmp_path)]) == 0
    assert q.get(HK).status == "queued"
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1


def test_an_unreadable_ledger_fails_closed(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    (tmp_path / "miner_spend.json").write_text("{torn")
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    assert rents == [] and _settle(r, q) == "spend_cap"
    assert "unreadable" in q.get(HK).last_error


def test_cap_zero_is_off(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch, miner_waste_cap_usd=0.0)
    _seed(tmp_path, q, [("p1", 50.0, 4.0, "waste", "infra")])
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1


def test_miner_caused_waste_pauses_without_a_make_good(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch)
    _seed(tmp_path, q, [("p1", 30.0, 2.0, "waste", "miner")])         # their own doing
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    assert _settle(r, q) == "spend_cap" and "re-fund" in q.get(HK).last_error
    assert OperatorApprovals.for_work_root(tmp_path).pending() == []


def test_a_generator_fault_fails_by_the_normal_rules():
    from cascade.trainer.faults import classify_funded_worker_failure

    assert classify_funded_worker_failure(3, "CorpusError: bad generator",
                                          stalled_before=False) == (True, "generator", False)


def _make_good_runner(tmp_path, monkeypatch):
    from tests.unit.test_funded_expired_key_operator_rent import _expired_runner

    r, q = _expired_runner(tmp_path)
    _vault(tmp_path, "hkA")                                            # a live miner key
    r.cfg.round = replace(r.cfg.round, operator_daily_spend_cap_usd=0.0)
    r._operator_balance_reader = lambda: 10_000.0
    r._operator_offer_rate = lambda skus: 1.0
    q.mark_in_flight("hkA", q.get("hkA").ref, target_boundary=1, era_index=1, started_block=1)
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(kw) or _rent_ok())
    e = q.get("hkA")
    now = time.time()
    (tmp_path / "miner_spend.json").write_text(json.dumps({"records": {"hkA": {
        "ref": e.ref, "funded_at": e.funded_at, "capped": False, "pods": [
            {"pod_id": "p1", "rate": 20.0, "start": now - 7201, "end": now - 1,
             "outcome": "waste", "kind": "infra"}]}}}))
    return r, q, rents


def test_approve_finishes_the_leg_operator_billed(tmp_path, monkeypatch):
    from cascade.trainer.faults import _FundedOperatorRent
    from tests.unit.test_funded_pod_wiring import _challenger as ch

    r, q, rents = _make_good_runner(tmp_path, monkeypatch)
    gen = ch("hkA", ref=q.get("hkA").ref)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", gen)
    msg, mf, cls, burn = r._funded_leg_failures["hkA"]
    q.requeue("hkA", error=msg, error_class=cls, burn_attempt=burn)
    assert q.get("hkA").status == PAUSED_SPEND_STATUS
    assert cli(["approve-operator-leg", "hkA", "--work-root", str(tmp_path)]) == 0
    e = q.get("hkA")
    assert e.status == "queued" and e.operator_billed
    q.mark_in_flight("hkA", e.ref, target_boundary=1, era_index=1, started_block=2)
    r.__dict__.setdefault("_leg_local_obj", threading.local()).operator_rent_ok = True
    with pytest.raises(_FundedOperatorRent):                           # the operator's account
        r._rent_funded_host("777", gen)
    assert rents == []                                                 # never the miner's key


def test_deny_keeps_it_paused_and_a_refund_resumes_on_the_miner_key(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _challenger as ch

    r, q, rents = _make_good_runner(tmp_path, monkeypatch)
    gen = ch("hkA", ref=q.get("hkA").ref)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", gen)
    msg, mf, cls, burn = r._funded_leg_failures["hkA"]
    q.requeue("hkA", error=msg, error_class=cls, burn_attempt=burn)
    assert cli(["deny-operator-leg", "hkA", "--work-root", str(tmp_path)]) == 0
    assert q.get("hkA").status == PAUSED_SPEND_STATUS                  # never rejected
    assert q.add("hkA", gen.ref, 1) == "queued"                        # the miner re-funds
    q.mark_in_flight("hkA", gen.ref, target_boundary=1, era_index=1, started_block=3)
    r._rent_funded_host("777", gen)
    assert len(rents) == 1 and not q.get("hkA").operator_billed


# ── the payer-balance pre-rent check ────────────────────────────────────────


def test_a_payer_shortfall_requeues_unburned_with_a_top_up_reason(tmp_path, monkeypatch, caplog):
    r, q, rents = _setup(tmp_path, monkeypatch, payer_projected_rate_per_hour=15.0)
    r._payer_balance_reader = lambda key: 5.0
    with caplog.at_level(logging.DEBUG), pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    msg, miner_fault, cls, burn = r._funded_leg_failures[HK]
    assert (miner_fault, cls, burn) == (False, "payer_balance", False)
    assert msg.startswith("top up: need") and "balance $5.00" in msg and rents == []
    assert "sk_test123" not in caplog.text                             # the key never logged
    q.requeue(HK, error=msg, error_class=cls, burn_attempt=burn)
    e = q.get(HK)
    assert e.status == "queued" and e.attempts == 0 and e.operator_requeues == 0


def test_an_unknown_payer_balance_is_retried_then_proceeds(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch, payer_projected_rate_per_hour=15.0)
    calls = []

    def flaky(key):
        calls.append(1)
        raise RuntimeError("lium API 502")

    r._payer_balance_reader = flaky
    r._rent_funded_host("777", _challenger(HK))
    assert len(calls) == 3 and len(rents) == 1


def test_a_chosen_h200_is_projected_at_its_own_cap(tmp_path, monkeypatch):
    from cascade.trainer import money_guard

    r, _q, _rents = _setup(tmp_path, monkeypatch, payer_projected_rate_per_hour=15.0)
    r._payer_balance_reader = lambda key: 100.0
    why = money_guard.payer_balance_refusal(r, "sk", hours=4.0, rate=max(15.0, 32.0))
    assert "≈$128.00" in why
    assert money_guard.payer_balance_refusal(r, "sk", hours=4.0, rate=15.0) == ""


# ── the miner's own TOTAL budget (--max-total-spend) ────────────────────────


def _budget_runner(tmp_path, monkeypatch, budget, **kw):
    r, q, rents = _setup(tmp_path, monkeypatch, miner_waste_cap_usd=0.0, **kw)
    q.add(HK, REF, 10, max_total_spend_usd=budget)               # (a re-fund may change it)
    return r, q, rents


def _leg_need(r):
    from cascade.trainer.miner_spend_guard import fallback_rate

    return fallback_rate(r) * (4 * 3600.0 + r.cfg.round.funded_ready_timeout_seconds) / 3600.0


def test_a_budget_too_small_for_one_leg_pauses_before_any_rent(tmp_path, monkeypatch):
    r, q, rents = _budget_runner(tmp_path, monkeypatch, 1.0)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    assert rents == [] and _settle(r, q) == "budget"
    e = q.get(HK)
    assert e.status == "paused_budget" and e.attempts == 0
    assert "budget $1 insufficient" in e.last_error and "your submission is kept" in e.last_error
    assert q.expire_stale(max_age_seconds=0) == 0 and q.get(HK).status == "paused_budget"
    assert q.public_view()["entries"][0]["budget_usd"] == 1.0


def test_a_budget_for_one_leg_blocks_the_retry_after_a_lost_run(tmp_path, monkeypatch):
    r, q, rents = _budget_runner(tmp_path, monkeypatch, None)
    budget = _leg_need(r) + 1.0                                   # exactly one leg
    q.add(HK, REF, 10, max_total_spend_usd=budget)
    _seed(tmp_path, q, [("p1", 1.5, 4.25, "waste", "infra")])     # the lost run
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(HK))
    assert rents == [] and _settle(r, q) == "budget"
    # raising the budget (a re-fund) resumes it on a fresh count
    assert q.add(HK, REF, 10, max_total_spend_usd=budget * 3) == "queued"
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1


def test_no_budget_is_unchanged_and_old_rows_load(tmp_path, monkeypatch):
    r, q, rents = _setup(tmp_path, monkeypatch, miner_waste_cap_usd=0.0)
    r._rent_funded_host("777", _challenger(HK))
    assert len(rents) == 1
    doc = json.loads(q.path.read_text())
    assert "max_total_spend_usd" not in doc["entries"][0]          # unset ⇒ not written
    assert FundedQueue(q.path).get(HK).max_total_spend_usd is None


def test_the_cli_flag_round_trips_through_the_intake(tmp_path):
    from cascade.funding.gpu_choice import BUDGET_HEADER
    from cascade.miner.cli import build_fund_headers, build_parser
    from tests.unit.test_miner_chosen_gpu import HK as GHK
    from tests.unit.test_miner_chosen_gpu import REF as GREF
    from tests.unit.test_miner_chosen_gpu import _clock, _intake, _sign

    args = build_parser().parse_args(["fund", "https://intake.example", "--ref", GREF,
                                      "--max-total-spend", "75"])
    assert args.max_total_spend == 75.0
    h = build_fund_headers("fund", GHK, GREF, "sk", _sign, now=_clock, max_total_spend=75.0)
    assert h[BUDGET_HEADER] == "75.0"
    intake = _intake(tmp_path)
    status, body = intake.fund(h)
    assert status == 202 and intake.queue.get(GHK).max_total_spend_usd == 75.0
    # the budget is SIGNED: a header swapped in transit fails the signature
    assert intake.fund({**h, BUDGET_HEADER: "9999"})[0] in (401, 403)


@pytest.mark.parametrize("raw", ["-5", "0", "nan", "inf", "abc"])
def test_the_intake_rejects_a_bad_budget(tmp_path, raw):
    from cascade.funding.gpu_choice import BUDGET_HEADER
    from cascade.miner.cli import build_fund_headers
    from tests.unit.test_miner_chosen_gpu import HK as GHK
    from tests.unit.test_miner_chosen_gpu import REF as GREF
    from tests.unit.test_miner_chosen_gpu import _clock, _intake, _sign

    h = build_fund_headers("fund", GHK, GREF, "sk", _sign, now=_clock)
    status, body = _intake(tmp_path).fund({**h, BUDGET_HEADER: raw})
    assert status == 400 and body["code"] == "bad_max_total_spend"
