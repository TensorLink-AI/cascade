"""Owner 2026-11-28 07:30 "add make goods" (lane C): a MarketCast leg lost on
OUR side (5Cu2sxndK8Dw, requeued ``operator_billed``) was re-dispatched on the
MINER's key ("operator-side failure 1/3 and no operator make-good configured")
because the lane C leg never opted in to the operator make-good. Now it does,
like a rolling lane A leg and the lane B make-good (5GMzsP, 2026-11-27):

* an operator fault on lane C PROPOSES an operator make-good: the approval
  request is filed (``make_good``, priced on the miner's GPU type x the leg
  wall) and the payer key is never used;
* until approved the entry waits ``awaiting_operator_approval`` — unburned,
  uncharged — with the miner-visible reason "our fault: waiting for an
  operator re-run, no cost to you";
* an approval on file → the re-run rents on the OPERATOR account (Lium, the
  miner's chosen GPU type), within the money guards (daily cap, reserve);
* ``approve-operator-leg <hotkey> --lane c`` approves it.

Torch-free: the real rent decision, gate, approval store, queue and lane C
driver; the pods / provider are fakes.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

HK = "5Cu2sxndK8Dw"
REF = "alice/mkt-recipe@sha256:" + "a" * 64
TARGET = "mkt:market"
GPU = "1xRTX4090"
QNAME = "lane_c_queue.json"


class _PayerRent(Exception):
    """The payer-key rent was reached (the bug)."""


class _OperatorRented(BaseException):  # not swallowed by the leg's except Exception
    """The operator-account rent was reached (the fix, once approved)."""


def _queue(tmp_path):
    from cascade.funding.queue import LaneCQueue

    q = LaneCQueue(tmp_path / QNAME, entry_ttl_seconds=86400.0)
    q.add_lane_c(HK, TARGET, REF, 1)
    assert q.mark_in_flight(HK, q.get(HK).ref, target_boundary=10, era_index=1,
                            started_block=1)
    # tonight's requeue: an operator-side failure, operator-billed, unburned
    q.requeue(HK, error="marketcast leg failed (operator side): boom",
              error_class="operator_transient", burn_attempt=False, operator_billed=True)
    e = q.get(HK)
    assert e.operator_billed and e.status == "queued"
    return q


def _runner(tmp_path, monkeypatch, q, **round_kw):
    from cascade.provision import funded as funded_mod
    from cascade.trainer import money_guard
    from tests.unit.test_funded_expired_key_operator_rent import _bind
    from tests.unit.test_funded_pod_wiring import _runner as wiring_runner
    from tests.unit.test_funded_pod_wiring import _vault

    round_kw.setdefault("operator_max_price_per_hour", 2.0)
    round_kw.setdefault("funded_ready_timeout_seconds", 0.0)
    r = _bind(wiring_runner(tmp_path, funded_operator_fallback=True, **round_kw))
    r.remote_hosts_path = None
    r.remote_hosts = [replace(r.remote_hosts[0], profile_only=True)]   # no operator lane
    r._funded_queue = lambda: q
    r._operator_balance_reader = lambda: 1000.0
    _vault(tmp_path, HK)                                   # a LIVE payer key is on file
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: (_ for _ in ()).throw(_PayerRent("payer rent")))
    # the live offer of the miner's chosen type (the provider listing is a fake)
    monkeypatch.setattr(money_guard, "offer_rates_by_sku",
                        lambda runner, skus: {s: 0.5 for s in skus})
    return r


def _ops(tmp_path, monkeypatch, r, q):
    """The lane C ops the driver calls, over the REAL ``train_mkt_leg``: its
    ``_run_funded_leg`` is the rent decision only (no corpus, no pod)."""
    from cascade.funding import mkt_gpus
    from cascade.trainer import marketcast_runner as mr
    from cascade.trainer.faults import _FundedOperatorRent

    monkeypatch.setattr(mr, "mkt_leg_contract", lambda cfg, base, hours=None: base)
    monkeypatch.setattr(mr, "mkt_contract_block_kw", lambda cfg, era: {})
    monkeypatch.setattr(mr, "mkt_leg_hours", lambda cfg, hours=0.0: 3.0)
    monkeypatch.setattr(mkt_gpus, "mkt_rent_skus", lambda cfg, choice: (choice.gpu,))
    base = SimpleNamespace(arch_preset="timestral-28m", max_train_seconds=3 * 3600,
                           train_image_digest="")
    r.cfg.throne_contracts_at = lambda block: [base]
    r.cfg.marketcast = SimpleNamespace(base_ref="hf://base@" + "b" * 40, pod_image="")
    r._mkt_harvested = {}
    r._dispatcher = lambda: None
    r.seen = []

    def run_funded_leg(disp, gen, seeds, block, contract, suffix, **kw):
        loc = r.__dict__.get("_leg_local_obj")
        loc.funded_contract = contract          # what funded_leg_scope sets
        r.seen.append({"opt_in": bool(getattr(loc, "operator_rent_ok", False)),
                       "skus": tuple(r._funded_skus_for_rent()), "role": kw.get("role")})
        try:
            r._rent_funded_host("777", gen)
        except _FundedOperatorRent:
            # what _run_funded_leg does for a mkt_finetune leg: the OPERATOR rent
            raise _OperatorRented(gen.hotkey) from None
        raise AssertionError("the rent decision returned a payer pod")

    r._run_funded_leg = run_funded_leg
    era = SimpleNamespace(start_block=1, index=1)
    return SimpleNamespace(r=r, _seeds=lambda: (SimpleNamespace(base_seed=777), era),
                           queue=lambda: q)


def _leg():
    return {"hotkey": HK, "uid": 7, "ref": REF, "target": TARGET, "entry_ref": f"{TARGET}|{REF}",
            "started_block": 1, "gpu": GPU, "gpu_allow_fallback": False,
            "gpu_max_price_per_hour": 0.6}


def _recipe():
    from cascade.trainer.marketcast_runner import MktLegRecipe

    return MktLegRecipe(target=TARGET, payload="{}", has_generator=False)


def _train(ops):
    from cascade.trainer.marketcast_runner import train_mkt_leg

    return train_mkt_leg(ops, _leg(), _recipe())


def _driver_done(tmp_path, ops, failure):
    from cascade.trainer.lane_c_driver import LaneCDriver

    d = LaneCDriver(SimpleNamespace(lane_c=None), ops, tmp_path / "lane_c_state.json")
    leg = _leg()
    d.st.open.append(leg)
    d._leg_done(leg, failure=failure)


def _approvals(tmp_path):
    from cascade.funding.operator_approvals import OperatorApprovals

    return OperatorApprovals.for_work_root(tmp_path)


# ── 1. an operator fault on lane C proposes a make-good; payer key untouched ─


def test_operator_fault_on_lane_c_requests_approval_and_never_uses_the_payer_key(
        tmp_path, monkeypatch):
    from cascade.trainer.lane_c_driver import LegFailure

    q = _queue(tmp_path)
    r = _runner(tmp_path, monkeypatch, q)
    ops = _ops(tmp_path, monkeypatch, r, q)
    try:
        _train(ops)
    except _PayerRent:
        pytest.fail("the lane C make-good fell back to the MINER's key (no opt-in)")
    except LegFailure as f:
        failure = f
    else:
        pytest.fail("the unapproved make-good trained")
    assert r.seen and r.seen[0]["opt_in"] is True, "the lane C leg does not opt in"
    assert r.seen[0]["role"] == "mkt_finetune" and r.seen[0]["skus"] == (GPU,)
    assert failure.error_class == "awaiting_operator_approval", failure.error_class
    assert failure.miner_fault is False and failure.burn is False
    # the request: a make-good, priced on the miner's GPU type x the leg wall
    req = _approvals(tmp_path).get(QNAME, HK)
    assert req is not None and req["kind"] == "make_good", req
    assert req["sku"] == GPU and float(req["rate_per_hour"]) == 0.5
    assert float(req["expected_hours"]) == pytest.approx(3.0)
    assert float(req["projected_cost_usd"]) == pytest.approx(1.5)
    # nothing on the payer's key: no payer pod row on the ledger
    assert [p for p in r._load_funded_ledger() if p.payer_hotkey] == []
    assert q.get(HK).operator_billed is True


# ── 2. unapproved → it waits, unburned, with the miner-visible reason ───────


def test_unapproved_lane_c_make_good_waits_unburned_with_the_our_fault_reason(
        tmp_path, monkeypatch):
    from cascade.trainer.lane_c_driver import LegFailure

    q = _queue(tmp_path)
    r = _runner(tmp_path, monkeypatch, q)
    ops = _ops(tmp_path, monkeypatch, r, q)
    attempts = q.get(HK).attempts
    with pytest.raises(LegFailure) as info:
        _train(ops)
    _driver_done(tmp_path, ops, info.value)
    e = q.get(HK)
    assert e.status == "awaiting_operator_approval", e.status
    assert e.attempts == attempts and e.operator_billed is True       # uncharged, our bill
    assert q.queued() == []                                           # never re-dispatched
    rows = [x for x in q.public_view().get("entries", []) if x.get("hotkey") == HK]
    from cascade.funding.public_reason import public_reason

    code, text = public_reason(e)
    assert text == "our fault: waiting for an operator re-run, no cost to you", text
    assert code == "operator_rerun_pending"
    assert rows and rows[0]["reason"] == text, rows      # the queue view the miner reads
    # a second dispatch before the owner acts files nothing new on the payer key
    with pytest.raises(LegFailure):
        _train(ops)
    assert [p for p in r._load_funded_ledger() if p.payer_hotkey] == []


# ── 3. approved → the re-run rents on the OPERATOR account ──────────────────


def _approve_cli(tmp_path, *extra):
    from cascade.trainer.operator_legs_cli import main

    return main(["approve-operator-leg", HK, "--work-root", str(tmp_path), *extra])


def test_approved_lane_c_make_good_rents_on_the_operator_account(tmp_path, monkeypatch, capsys):
    from cascade.trainer.lane_c_driver import LegFailure

    q = _queue(tmp_path)
    r = _runner(tmp_path, monkeypatch, q)
    ops = _ops(tmp_path, monkeypatch, r, q)
    with pytest.raises(LegFailure) as info:
        _train(ops)
    _driver_done(tmp_path, ops, info.value)
    assert q.get(HK).status == "awaiting_operator_approval"
    # the owner's command, lane C by name
    assert _approve_cli(tmp_path, "--lane", "c") == 0
    out = capsys.readouterr().out
    assert "APPROVED" in out and "entry returned to the queue" in out, out
    e = q.get(HK)
    assert e.status == "queued" and e.operator_billed is True
    with pytest.raises(_OperatorRented):
        _train(ops)                       # the operator rent, never the payer's
    assert r.seen[-1]["opt_in"] is True and r.seen[-1]["skus"] == (GPU,)
    assert q.get(HK).operator_billed is True


def test_the_approved_lane_c_rent_still_respects_the_daily_cap(tmp_path, monkeypatch):
    from cascade.trainer.lane_c_driver import LegFailure

    q = _queue(tmp_path)
    r = _runner(tmp_path, monkeypatch, q, ops_operator_daily_cap_usd=1.0)
    ops = _ops(tmp_path, monkeypatch, r, q)
    with pytest.raises(LegFailure) as info:
        _train(ops)
    _driver_done(tmp_path, ops, info.value)
    assert q.get(HK).status == "awaiting_operator_approval"
    assert _approve_cli(tmp_path, "--lane", "c", "--max-cost", "5") == 0
    with pytest.raises(LegFailure) as info:
        _train(ops)                       # $1.50 projected > the $1 daily cap
    assert info.value.error_class == "operator_blocked" and info.value.burn is False
    assert "daily spend cap" in info.value.msg


# ── 4. the CLI's --lane is the lane's queue ─────────────────────────────────


def test_lane_flag_names_the_lane_queue_and_refuses_a_contradiction(tmp_path):
    from cascade.trainer import operator_legs_cli as cli

    assert getattr(cli, "LANE_QUEUES", {}).get("c") == QNAME
    _queue(tmp_path)
    assert _approve_cli(tmp_path, "--lane", "c", "--queue", "lane_b_queue.json") == 2
    from cascade.funding.queue import LaneCQueue

    assert cli._queue_for(tmp_path, {"queue_path": str(tmp_path / QNAME)}, None).__class__ \
        is LaneCQueue
