"""An EMPTY ACCOUNT is not an empty MARKET (2026-11-04, 5D2KVsje).

A funded leg whose payer key listed nothing on ``lium ls`` polled "no
capacity" every 90 s for five days and re-dispatched every era, while two
other payers rented the same 8xH100 type on the same days: the capacity probe
could not tell the miner's account from the marketplace, the wait swallowed
probe exceptions, and an empty listing logged nothing. Now the payer's
listing is checked against the OPERATOR's listing under the same caps; an
account that lists nothing while the market has the type ends the wait with
a terminal, unburned, miner-readable ``payer_account`` verdict.
"""
from __future__ import annotations

import logging
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from cascade.funding.queue import PAYER_ACCOUNT_CLASS
from cascade.miner.submission_status import ERROR_TEXT
from cascade.provision import funded as funded_mod
from cascade.provision.core import LiumProvider, ProvisionError
from cascade.provision.payer_pool import PayerAccountSuspect, PayerCapacityWatch
from cascade.trainer import money_guard
from cascade.trainer.faults import king_key_failed
from cascade.trainer.loop import _FundedLegSkip
from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault
from tests.unit.test_funded_rent_wait import _arm_wait

SKUS = ("8xH100",)


def _watch(payer, control, **kw):
    clock = {"t": 1000.0}
    w = PayerCapacityWatch(payer_probe=payer, control_probe=control,
                           now=lambda: clock["t"], **kw)
    return w, clock


def _poll(w, clock, n, step=90.0):
    out = []
    for _ in range(n):
        out.append(w(SKUS))
        clock["t"] += step
    return out


# ── the watch (pure) ─────────────────────────────────────────────────────────

def test_empty_payer_listing_with_a_full_operator_listing_convicts_after_two_spaced_controls():
    controls = []

    def control(skus):
        controls.append(skus)
        return 5

    w, clock = _watch(lambda skus: 0, control)
    with pytest.raises(PayerAccountSuspect) as info:
        _poll(w, clock, 12)
    e = info.value
    # control after 3 empty polls (t=1180), the second one >= 300 s later (poll 7, t=1540)
    assert len(controls) == 2 and controls == [SKUS, SKUS]
    assert (e.kind, e.control, e.zero_polls) == ("empty_listing", 5, 7)
    assert "lists no 8xH100 executor" in str(e) and "marketplace shows 5" in str(e)
    assert w.verdict is e


def test_an_empty_market_stays_a_capacity_wait():
    w, clock = _watch(lambda skus: 0, lambda skus: 0)
    assert _poll(w, clock, 12) == [0] * 12            # no verdict, ever
    assert w.verdict is None and w.control_hits == 0 and w.last_control == 0


def test_payer_capacity_reappearing_resets_the_count():
    seq = [0, 0, 0, 2, 0, 0]
    controls = []
    w, clock = _watch(lambda skus: seq.pop(0) if seq else 0,
                      lambda skus: controls.append(1) or 5)
    assert _poll(w, clock, 4) == [0, 0, 0, 2]
    assert len(controls) == 1                         # one control before capacity came back
    assert w.zero_polls == 0 and w.control_hits == 0 and w.verdict is None


def test_a_failing_control_listing_never_convicts(caplog):
    def control(skus):
        raise RuntimeError("operator listing down")

    w, clock = _watch(lambda skus: 0, control)
    with caplog.at_level(logging.WARNING, logger="cascade.provision.payer_pool"):
        assert _poll(w, clock, 12) == [0] * 12
    assert w.verdict is None and w.control_hits == 0
    assert "not counted" in caplog.text and "RuntimeError" in caplog.text


def test_auth_text_from_the_payer_listing_convicts_at_once():
    def payer(skus):
        raise ProvisionError("`lium ls --gpu H100 --format json` failed (rc=1): Invalid API key")

    w, clock = _watch(payer, lambda skus: 5)
    with pytest.raises(PayerAccountSuspect) as info:
        w(SKUS)
    assert info.value.kind == "auth" and "refused the key" in str(info.value)
    assert "Invalid API key" in str(info.value)


def test_other_payer_probe_failures_are_logged_and_answer_unknown(caplog):
    def payer(skus):
        raise OSError("lium CLI timed out")

    w, clock = _watch(payer, lambda skus: 5)
    with caplog.at_level(logging.WARNING, logger="cascade.provision.payer_pool"):
        assert _poll(w, clock, 5) == [None] * 5
    assert w.verdict is None and "no answer this poll" in caplog.text
    assert "OSError" in caplog.text


# ── the wait loop ────────────────────────────────────────────────────────────

def test_the_wait_ends_payer_account_on_the_verdict(tmp_path, caplog):
    runner = _runner(tmp_path)
    _arm_wait(runner, deadline_offsets=3600, capacity_seq=[0, 0])

    def probe(skus):
        raise PayerAccountSuspect("empty_listing", skus=skus, control=4, zero_polls=7,
                                  span_seconds=540.0)

    with caplog.at_level(logging.WARNING, logger="cascade.trainer"):
        assert runner._wait_for_funded_capacity("8xH100", describe="funded leg x", hotkey="hk",
                                                capacity_probe=probe) == "payer_account"
    assert "payer_account verdict" in caplog.text and "marketplace shows 4" in caplog.text


def test_a_probe_exception_is_logged_and_the_wait_keeps_polling(tmp_path, caplog):
    runner = _runner(tmp_path)
    clock = _arm_wait(runner, deadline_offsets=200, capacity_seq=[0])
    calls = []

    def probe(skus):
        calls.append(clock["t"])
        raise ValueError("boom")

    with caplog.at_level(logging.WARNING, logger="cascade.trainer"):
        assert runner._wait_for_funded_capacity("8xH100", describe="funded leg x", hotkey="hk",
                                                capacity_probe=probe) is False
    assert len(calls) >= 2                                   # kept polling to the deadline
    assert "capacity probe failed" in caplog.text and "ValueError: boom" in caplog.text


# ── the funded rent: terminal, unburned, readable ────────────────────────────

def test_funded_rent_turns_an_account_verdict_terminal_and_unburned(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: (
        rents.append(1) or funded_mod.FundedRentResult(
            hotkey="hkA", ok=False, error="sold out", error_class="no_capacity",
            burn_attempt=False)))
    _arm_wait(runner, deadline_offsets=7200, capacity_seq=[0] * 40)
    # the payer's listing is empty on every poll; the operator's shows three
    monkeypatch.setattr("cascade.provision.payer_pool.payer_capacity", lambda *a, **kw: 0)
    controls = []
    runner._probe_funded_capacity = (
        lambda sku, exclude_ids=(), price_caps=None: controls.append(price_caps) or 3)
    runner._payer_balance_reader = lambda key: None          # the key cannot read /users/me
    runner.PAYER_CONTROL_INTERVAL_SECONDS = 300.0
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (True, PAYER_ACCOUNT_CLASS, False)   # terminal, unburned
    assert "marketplace shows 3" in msg and "fund again" in msg
    assert "cannot read its own account balance" in msg
    assert "sk_test123" not in msg                           # never the key
    assert rents == [1]                                      # one sold-out rent, then the wait
    assert len(controls) == 2 and all(isinstance(c, dict) for c in controls)  # miner caps
    assert runner._load_funded_ledger() == []                # the write-ahead row is gone


def test_a_market_drought_still_requeues_no_capacity(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: funded_mod.FundedRentResult(
        hotkey="hkA", ok=False, error="sold out", error_class="no_capacity", burn_attempt=False))
    _arm_wait(runner, deadline_offsets=1200, capacity_seq=[0] * 40)
    monkeypatch.setattr("cascade.provision.payer_pool.payer_capacity", lambda *a, **kw: 0)
    runner._probe_funded_capacity = lambda sku, exclude_ids=(), price_caps=None: 0
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)   # unchanged: the market


def test_king_key_failed_treats_the_account_verdict_as_a_key_failure():
    assert king_key_failed(("lists nothing", True, PAYER_ACCOUNT_CLASS, False))
    assert not king_key_failed(("sold out", False, "no_capacity", False))


# ── the balance signal and the miner-facing text ─────────────────────────────

def test_payer_balance_note():
    assert money_guard.payer_balance_note(
        SimpleNamespace(_payer_balance_reader=lambda k: 12.5), "sk") == "balance $12.50"
    note = money_guard.payer_balance_note(SimpleNamespace(_payer_balance_reader=lambda k: None), "sk")
    assert note.startswith("the key cannot read its own account balance")
    assert money_guard.payer_balance_note(SimpleNamespace(), "") == ""


def test_unknown_payer_balance_is_logged_against_the_leg(caplog):
    r = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(payer_projected_rate_per_hour=10.0)),
                        _payer_balance_reader=lambda k: None)
    with caplog.at_level(logging.WARNING, logger="cascade.trainer"):
        assert money_guard.payer_balance_refusal(r, "sk", hours=3.0, who="5D2KVsje") == ""
    assert "funded leg 5D2KVsje: payer balance unknown" in caplog.text


def test_miner_cli_explains_the_class():
    assert "fund again" in ERROR_TEXT[PAYER_ACCOUNT_CLASS]
    assert "no attempt was consumed" in ERROR_TEXT[PAYER_ACCOUNT_CLASS]


# ── the listing log line ─────────────────────────────────────────────────────

def _lium(stdout: str, api_key: str = "sk-very-secret") -> LiumProvider:
    return LiumProvider(bin="lium", api_key=api_key,
                        _run=lambda argv: CompletedProcess(argv, 0, stdout=stdout, stderr=""),
                        _spawn=lambda argv: None)


def test_an_empty_payer_listing_logs_the_raw_row_count_without_the_key(caplog):
    with caplog.at_level(logging.INFO, logger="cascade.provision.core"):
        assert _lium("[]").capacity("8xH100") == 0
    assert "lium (payer key): 8xH100 listing — 0 raw row(s), 0 of the 8-GPU shape" in caplog.text
    assert "sk-very-secret" not in caplog.text


def test_a_listing_emptied_by_the_shape_filter_says_so(caplog):
    rows = '[{"id": "ex-1", "gpu_type": "H100", "gpu_count": 1, "price_per_hour": 2.0}]'
    with caplog.at_level(logging.INFO, logger="cascade.provision.core"):
        assert _lium(rows, api_key="").capacity("8xH100") == 0
    assert "lium: 8xH100 listing — 1 raw row(s), 0 of the 8-GPU shape" in caplog.text
