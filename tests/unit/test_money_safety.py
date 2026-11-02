"""2026-11-02 money safety: owner approval per operator-billed leg, the
operator-fault cap, the operator-account circuit breaker, the balance gate,
the daily spend cap, the payer-balance check and their config knobs.

The incident: an expired-key make-good rented an 8xH100 on the operator's
Lium account unasked, drained it to zero (Lium then killed every pod on the
account, eval pods included), and the 402 re-dispatched one leg every ~2.5
min. None of these tests use the ``operator_legs_preapproved`` fixture.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.funding.operator_approvals import OperatorApprovals
from cascade.funding.queue import (
    APPROVAL_CLASS,
    AWAITING_APPROVAL_STATUS,
    DENIED_CLASS,
    OPERATOR_BLOCKED_CLASS,
    OPERATOR_FAULT_CLASS,
    PAUSED_STATUS,
    FundedQueue,
    select_field,
)
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.shared.config.money import MONEY_KNOBS, VALIDATOR_CACHE_KNOBS
from cascade.trainer import money_guard
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.operator_legs_cli import main as cli

REPO_TOML = Path(__file__).resolve().parents[2] / "chain.toml"
HK = "5DpyoY8pxp5xCXvErnuUDDygBT6rXQcTEBYJHBYo4rpPpGRT"
REF = "vault/direct@sha256:" + "a" * 64


# ── config knobs: defaults + loader parse + round trip (the 3x defect class) ──


def test_round_knob_defaults_match_the_dataclass_and_the_shipped_toml():
    rnd, shipped = RoundConfig(), load_chain_config(REPO_TOML).round
    for name, (default, _kind) in MONEY_KNOBS.items():
        assert getattr(rnd, name) == default, name
        assert getattr(shipped, name) == default, name
    v = load_chain_config(REPO_TOML).validator
    for name, (default, _kind) in VALIDATOR_CACHE_KNOBS.items():
        assert getattr(v, name) == default, name


def _edited(tmp_path, round_lines: str = "", validator_lines: str = "") -> Path:
    src = REPO_TOML.read_text(encoding="utf-8")
    anchor = "funded_operator_fallback = false\n"
    assert src.count(anchor) == 1 and src.count("[validator]\n") == 1
    src = src.replace(anchor, anchor + round_lines, 1)
    src = src.replace("[validator]\n", "[validator]\n" + validator_lines, 1)
    p = tmp_path / "chain.toml"
    p.write_text(src, encoding="utf-8")
    return p


def test_every_knob_round_trips_through_the_loader(tmp_path):
    p = _edited(tmp_path,
                "operator_fault_cap = 5\noperator_breaker_cooldown_seconds = 900\n"
                "operator_cost_safety_factor = 1.5\noperator_balance_reserve_usd = 75.0\n"
                "operator_daily_spend_cap_usd = 120.0\npayer_projected_rate_per_hour = 12.5\n"
                "expected_checkpoint_gb = 16.0\nharvest_disk_retries = 4\n"
                "harvest_disk_retry_seconds = 30\n",
                'eval_ckpt_cache_dir = "/mnt/cache"\neval_ckpt_cache_max_gb = 200.0\n'
                "eval_ckpt_cache_min_free_gb = 80\neval_ckpt_cache_grace_seconds = 600\n")
    cfg = load_chain_config(p)
    r, v = cfg.round, cfg.validator
    assert (r.operator_fault_cap, r.operator_breaker_cooldown_seconds) == (5, 900)
    assert (r.operator_cost_safety_factor, r.operator_balance_reserve_usd) == (1.5, 75.0)
    assert (r.operator_daily_spend_cap_usd, r.payer_projected_rate_per_hour) == (120.0, 12.5)
    assert (r.expected_checkpoint_gb, r.harvest_disk_retries,
            r.harvest_disk_retry_seconds) == (16.0, 4, 30)
    assert (v.eval_ckpt_cache_dir, v.eval_ckpt_cache_max_gb, v.eval_ckpt_cache_min_free_gb,
            v.eval_ckpt_cache_grace_seconds) == ("/mnt/cache", 200.0, 80.0, 600)


@pytest.mark.parametrize("line", ["operator_fault_cap = -1", 'operator_fault_cap = "3"',
                                  "operator_cost_safety_factor = 0.5",
                                  "operator_daily_spend_cap_usd = -5.0",
                                  "harvest_disk_retries = 1.5"])
def test_bad_round_knobs_fail_loud(tmp_path, line):
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        load_chain_config(_edited(tmp_path, line + "\n"))


@pytest.mark.parametrize("line", ["eval_ckpt_cache_max_gb = -1", 'eval_ckpt_cache_dir = ""',
                                  "eval_ckpt_cache_grace_seconds = -2"])
def test_bad_validator_knobs_fail_loud(tmp_path, line):
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        load_chain_config(_edited(tmp_path, validator_lines=line + "\n"))


# ── the queue: parked states, the operator-fault cap ────────────────────────


def _flight(tmp_path, t=None, **kw):
    clock = (lambda: t[0]) if t is not None else None
    q = FundedQueue(tmp_path / "funded_queue.json", **({"clock": clock} if clock else {}), **kw)
    q.add(HK, REF, 10)
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    return q


def test_awaiting_approval_parks_unburned_and_is_never_selected(tmp_path):
    q = _flight(tmp_path)
    assert q.requeue(HK, error="awaiting", error_class=APPROVAL_CLASS, burn_attempt=False)
    e = q.get(HK)
    assert e.status == AWAITING_APPROVAL_STATUS and e.attempts == 0 and e.operator_requeues == 0
    assert select_field(q.entries(), 0) == []
    assert not q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
    assert q.expire_stale(max_age_seconds=0) == 0            # never TTL-expired
    assert q.public_view()["entries"][0]["phase"] == AWAITING_APPROVAL_STATUS


def test_every_operator_fault_counts_and_the_cap_pauses(tmp_path):
    t = [1000.0]
    q = _flight(tmp_path, t)
    q.requeue(HK, error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    # interleaved non-operator outcomes never reset the count (2026-11-02 loop)
    for cls in ("no_capacity", OPERATOR_BLOCKED_CLASS, "operator_disk"):
        t[0] = q.get(HK).not_before + 1
        assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
        q.requeue(HK, error="y", error_class=cls, burn_attempt=False)
        assert q.get(HK).operator_requeues == 1
    t[0] = q.get(HK).not_before + 1
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=3)
    q.requeue(HK, error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    t[0] = q.get(HK).not_before + 1
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=4)
    q.requeue(HK, error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    e = q.get(HK)
    assert e.status == PAUSED_STATUS and e.attempts == 0 and e.operator_requeues == 3
    t[0] += 10 ** 7
    assert not q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=5)
    assert q.unpark(HK, expect_status=PAUSED_STATUS)
    assert (q.get(HK).status, q.get(HK).operator_requeues) == ("queued", 0)


def test_blocked_classes_back_off_without_counting(tmp_path):
    t = [1000.0]
    q = _flight(tmp_path, t)
    q.requeue(HK, error="breaker", error_class=OPERATOR_BLOCKED_CLASS, burn_attempt=False)
    e = q.get(HK)
    assert e.not_before > 1000.0 and e.operator_requeues == 0 and e.status == "queued"


def test_cap_zero_never_pauses(tmp_path):
    q = _flight(tmp_path, operator_fault_cap=0)
    for _ in range(6):
        q.requeue(HK, error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
        assert q.get(HK).status == "queued"


def test_refund_of_a_parked_entry_returns_it_on_the_miners_key(tmp_path):
    q = _flight(tmp_path)
    q.set_operator_billed(HK, True)
    q.requeue(HK, error="awaiting", error_class=APPROVAL_CLASS, burn_attempt=False)
    assert q.add(HK, REF, 10) == "queued"
    e = q.get(HK)
    assert e.status == "queued" and not e.operator_billed and e.operator_requeues == 0


def test_deny_clears_operator_billing(tmp_path):
    q = _flight(tmp_path)
    q.set_operator_billed(HK, True)
    q.requeue(HK, error="awaiting", error_class=APPROVAL_CLASS, burn_attempt=False)
    assert q.unpark(HK, expect_status=AWAITING_APPROVAL_STATUS, deny=True)
    e = q.get(HK)
    assert e.status == "queued" and not e.operator_billed and e.last_error_class == DENIED_CLASS


def test_parked_entries_can_be_withdrawn(tmp_path):
    q = _flight(tmp_path)
    q.requeue(HK, error="awaiting", error_class=APPROVAL_CLASS, burn_attempt=False)
    assert q.withdraw(HK) and q.get(HK).status == "withdrawn"


# ── the approval store ──────────────────────────────────────────────────────


def test_approvals_are_single_use_and_bound_to_the_request(tmp_path):
    s = OperatorApprovals(tmp_path / "a.json")
    assert s.request("funded_queue.json", HK, ref=REF, attempt_id="A1", uid=146,
                     projected_cost_usd=65.0, kind="expired_key")
    assert not s.request("funded_queue.json", HK, ref=REF, attempt_id="A2")   # refresh only
    assert s.pending()[0]["request_attempt_id"] == "A1"
    with pytest.raises(ValueError):
        s.decide(HK, approve=True, attempt_id="A2")
    with pytest.raises(LookupError):
        s.decide("5Other", approve=True)
    s.decide(HK, approve=True, attempt_id="A1")
    assert s.consume("funded_queue.json", HK, ref="other@x", attempt_id="B",
                     projected_cost_usd=1.0) == "none"                       # other ref
    assert s.consume("funded_queue.json", HK, ref=REF, attempt_id="B",
                     projected_cost_usd=65.0) == "approved"
    assert s.consume("funded_queue.json", HK, ref=REF, attempt_id="C",
                     projected_cost_usd=65.0) == "none"                      # used up


def test_approval_ceiling_refuses_a_costlier_dispatch(tmp_path):
    s = OperatorApprovals(tmp_path / "a.json")
    s.request("q", HK, ref=REF, attempt_id="A1", projected_cost_usd=65.0)
    s.decide(HK, approve=True, max_cost_usd=50.0)
    assert s.consume("q", HK, ref=REF, attempt_id="A1", projected_cost_usd=65.0) == "over_ceiling"
    assert s.pending()[0]["state"] == "pending"


# ── the gate on a fake runner ───────────────────────────────────────────────


def _gate_runner(tmp_path, balance=1000.0, **round_kw):
    rnd = RoundConfig(funded_mode="required", funded_pod_sku="8xH100",
                      operator_max_price_per_hour=20.0, funded_ready_timeout_seconds=900.0,
                      **round_kw)
    q = _flight(tmp_path)
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path,
                        _funded_leg_failures={})

    def record(hk, msg, *, miner_fault, error_class, burn):
        r._funded_leg_failures[hk] = (msg, miner_fault, error_class, burn)

    r._record_funded_failure = record
    r._funded_queue = lambda: q
    r._funded_skus_for_rent = lambda: ("8xH100",)
    r._leg_wall_seconds = lambda sku=None: 3 * 3600.0
    r._operator_balance_reader = lambda: balance
    return r, q


GEN = SimpleNamespace(hotkey=HK, ref=REF, uid=146)


def _park(r, q, kind="expired_key", rent=True):
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind=kind, rent=rent)
    msg, miner_fault, cls, burn = r._funded_leg_failures[HK]
    q.requeue(HK, error=msg, error_class=cls, burn_attempt=burn)
    return msg, miner_fault, cls, burn


def _redispatch(q):
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=7)


def test_unapproved_operator_leg_parks_with_cost_and_alert(tmp_path):
    r, q = _gate_runner(tmp_path)
    msg, miner_fault, cls, burn = _park(r, q)
    assert (miner_fault, cls, burn) == (False, APPROVAL_CLASS, False)
    assert q.get(HK).status == AWAITING_APPROVAL_STATUS
    (req,) = OperatorApprovals.for_work_root(tmp_path).pending()
    assert req["uid"] == 146 and req["sku"] == "8xH100" and req["kind"] == "expired_key"
    assert req["projected_cost_usd"] == pytest.approx(20.0 * 3.25)
    codes = {c["code"] for c in money_guard.health_conditions(r, {"a": q.entries()})}
    assert "operator_leg_awaiting_approval" in codes


def test_approved_leg_dispatches_exactly_once_and_books_the_spend(tmp_path, capsys):
    r, q = _gate_runner(tmp_path)
    _park(r, q)
    assert cli(["approve-operator-leg", HK, "--work-root", str(tmp_path)]) == 0
    assert "APPROVED" in capsys.readouterr().out
    assert q.get(HK).status == "queued"
    _redispatch(q)
    assert money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True) is None
    assert money_guard.OperatorSpend.for_runner(r).total() == pytest.approx(65.0)
    # a later failure's retry needs a NEW approval
    _park(r, q)
    assert q.get(HK).status == AWAITING_APPROVAL_STATUS


def test_denied_expired_key_fails_unburned_and_make_good_returns_to_the_miner(tmp_path):
    r, q = _gate_runner(tmp_path)
    _park(r, q, kind="make_good")
    assert cli(["deny-operator-leg", HK, "--work-root", str(tmp_path)]) == 0
    assert not q.get(HK).operator_billed and q.get(HK).status == "queued"
    _redispatch(q)
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind="make_good", rent=True)
    assert r._funded_leg_failures[HK][1:] == (False, DENIED_CLASS, False)
    # expired key + denial: terminal "re-fund needed", miner-side, never burned
    _park(r, q)
    cli(["deny-operator-leg", HK, "--work-root", str(tmp_path)])
    _redispatch(q)
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True)
    msg, miner_fault, cls, burn = r._funded_leg_failures[HK]
    assert (miner_fault, cls, burn) == (True, "auth", False) and "fresh key" in msg


def test_king_legs_are_not_gated(tmp_path):
    r, _q = _gate_runner(tmp_path)
    r._funded_queue = lambda: None
    assert money_guard.operator_leg_gate(r, GEN, kind="make_good", rent=True) is None


def test_cli_lists_pending_and_paused(tmp_path, capsys):
    r, q = _gate_runner(tmp_path)
    _park(r, q)
    assert cli(["list-operator-legs", "--work-root", str(tmp_path), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["awaiting_approval"][0]["hotkey"] == HK and doc["paused"] == []
    assert cli(["resume-operator-leg", HK, "--work-root", str(tmp_path)]) == 1


def test_trainer_main_dispatches_the_subcommands(tmp_path, capsys):
    from cascade.trainer.main import main

    assert main(["list-operator-legs", "--work-root", str(tmp_path)]) == 0
    assert "no operator-billed legs" in capsys.readouterr().out


# ── money preflight: breaker, caps, balance ─────────────────────────────────


def _approve(r, q):
    _park(r, q)
    cli(["approve-operator-leg", HK, "--work-root", str(r.work_root)])
    _redispatch(q)


def test_open_breaker_blocks_and_keeps_the_approval(tmp_path):
    r, q = _gate_runner(tmp_path)
    _approve(r, q)
    money_guard.OperatorBreaker.for_runner(r).trip("402 BALANCE_REQUIRED")
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True)
    assert r._funded_leg_failures[HK][2] == OPERATOR_BLOCKED_CLASS
    assert OperatorApprovals.for_work_root(tmp_path).peek_approved("funded_queue.json", HK, REF)
    codes = {c["code"] for c in money_guard.health_conditions(r, {})}
    assert "operator_breaker_open" in codes


def test_daily_cap_blocks_with_one_alert(tmp_path):
    r, q = _gate_runner(tmp_path, operator_daily_spend_cap_usd=50.0)
    _approve(r, q)
    for _ in range(3):
        with pytest.raises(_FundedLegSkip):
            money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True)
    caps = [c for c in money_guard.health_conditions(r, {}) if c["code"] == "operator_daily_cap"]
    assert len(caps) == 1
    assert money_guard.daily_cap_usd(RoundConfig(operator_daily_spend_cap_usd=200.0,
                                                 ops_operator_daily_cap_usd=80.0)) == 80.0


def test_unknown_balance_refuses_and_low_balance_trips_the_breaker(tmp_path):
    r, q = _gate_runner(tmp_path, balance=None)
    _approve(r, q)
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True)
    assert money_guard.OperatorBreaker.for_runner(r).open_record() is None
    r._operator_balance_reader = lambda: 100.0              # < 65 * 1.25 + 50
    with pytest.raises(_FundedLegSkip):
        money_guard.operator_leg_gate(r, GEN, kind="expired_key", rent=True)
    assert money_guard.OperatorBreaker.for_runner(r).open_record() is not None


def test_balance_refusal_on_any_operator_launch_trips_the_breaker(tmp_path):
    from cascade.provision.core import ProvisionError

    r, _q = _gate_runner(tmp_path)
    calls = []

    class _P:
        name = "lium"

        def launch(self, spec):
            calls.append(spec)
            raise ProvisionError("lium up failed: API error 402 BALANCE_REQUIRED")

    p = money_guard.guard_operator_provider(r, _P())
    assert p.name == "lium"
    with pytest.raises(ProvisionError, match="402"):
        p.launch("spec")
    with pytest.raises(ProvisionError, match="circuit breaker open"):
        p.launch("spec")
    assert len(calls) == 1                                   # never retried against it
    assert money_guard.is_balance_refusal("Insufficient balance")


def test_corrupt_breaker_state_fails_closed(tmp_path):
    b = money_guard.OperatorBreaker(tmp_path / "operator_breaker.json")
    (tmp_path / "operator_breaker.json").write_text("{not json")
    assert b.open_record() is not None


def test_payer_balance_check(tmp_path):
    r, _q = _gate_runner(tmp_path, payer_projected_rate_per_hour=10.0)
    r._payer_balance_reader = lambda key: 20.0
    why = money_guard.payer_balance_refusal(r, "sk", hours=3.0)
    assert "cannot finish" in why and "$30.00" in why
    r._payer_balance_reader = lambda key: None              # unknown never blocks a miner
    assert money_guard.payer_balance_refusal(r, "sk", hours=3.0) == ""
    r2, _ = _gate_runner(tmp_path / "x")
    assert money_guard.payer_balance_refusal(r2, "sk", hours=3.0) == ""   # knob off


def test_paused_entries_raise_one_condition(tmp_path):
    r, q = _gate_runner(tmp_path)
    for _ in range(3):
        q.requeue(HK, error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    conds = [c for c in money_guard.health_conditions(r, {"a": q.entries()})
             if c["code"] == "entry_paused"]
    assert len(conds) == 1 and re.search("resume-operator-leg", conds[0]["message"])
