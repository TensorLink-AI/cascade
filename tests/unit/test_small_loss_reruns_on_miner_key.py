"""Owner 2026-11-04: a small our-side loss (a $3.81 / $4.03 lemon) must not
turn a miner's leg into a ~$60 operator make-good waiting for approval — it
reruns on the miner's own key. Above ``[round] make_good_rerun_below_usd``,
after an owner approval, past the waste cap or the operator-fault pause, the
approval path is unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cascade.trainer.miner_spend_guard as msg
from cascade.funding.operator_approvals import OperatorApprovals
from cascade.shared.config import load_chain_config
from cascade.shared.config.loader import DEFAULT_CHAIN_TOML
from cascade.shared.config.rounds import RoundConfig
from cascade.trainer.runner_funded_pods import _FundedPodMixin

HK = "5Grq7ZmW8ZQtkw8o5iy6T2LoyEegPFoV1Ky9mvCvYpfhjRnN"


class _Queue:
    def __init__(self, path):
        self.path = path
        self.billed = True

    def set_operator_billed(self, hotkey, flag):
        self.billed = flag
        return True


def _runner(tmp_path, cap=20.0):
    r = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(make_good_rerun_below_usd=cap)),
                        OPERATOR_FAULT_RETRIES=3, work_root=tmp_path)
    return r, _FundedPodMixin._small_loss_rerun


def _run(tmp_path, monkeypatch, *, lost=3.81, cap=20.0, faults=1, is_capped=False):
    spend = SimpleNamespace(make_good_usd=lost, infra_waste_usd=0.0)
    monkeypatch.setattr(msg, "_ledger", lambda runner: SimpleNamespace(
        summary=lambda hk, ref, funded_at: spend))
    monkeypatch.setattr(msg, "capped", lambda runner, gen: is_capped)
    r, fn = _runner(tmp_path, cap)
    q = _Queue(tmp_path / "funded_queue.json")
    entry = SimpleNamespace(operator_faults=faults, ref="r", funded_at=1.0)
    return fn(r, SimpleNamespace(hotkey=HK), entry, q), q


def test_a_small_loss_reruns_on_the_miners_key(tmp_path, monkeypatch):
    cleared, q = _run(tmp_path, monkeypatch, lost=3.81)
    assert cleared and q.billed is False


@pytest.mark.parametrize("kw", [dict(lost=60.0), dict(cap=0.0), dict(faults=3),
                                dict(is_capped=True), dict(lost=0.0)])
def test_the_approval_path_is_kept_otherwise(tmp_path, monkeypatch, kw):
    cleared, q = _run(tmp_path, monkeypatch, **kw)
    assert not cleared and q.billed is True


def test_an_owner_approved_operator_leg_is_never_moved_back(tmp_path, monkeypatch):
    store = OperatorApprovals.for_work_root(tmp_path)
    store.request("funded_queue.json", HK, ref="r", attempt_id="a", queue_path="q",
                  uid=78, sku="8xH100", rate_per_hour=17.6, expected_hours=3.4,
                  projected_cost_usd=60.0, kind="make_good", reason="x", funded_at=1.0)
    store.decide(HK, approve=True)
    cleared, q = _run(tmp_path, monkeypatch, lost=3.81)
    assert not cleared and q.billed is True


def test_knob_and_timeout_defaults_and_parse(tmp_path):
    assert RoundConfig.make_good_rerun_below_usd == 20.0
    assert RoundConfig.funded_ready_timeout_seconds == 600.0
    src = DEFAULT_CHAIN_TOML.read_text()
    assert "\n[round]\n" in src
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("\n[round]\n", "\n[round]\nmake_good_rerun_below_usd = 7.5\n", 1))
    assert load_chain_config(p).round.make_good_rerun_below_usd == 7.5
    p.write_text(src.replace("\n[round]\n", "\n[round]\nmake_good_rerun_below_usd = -3\n", 1))
    with pytest.raises(ValueError):
        load_chain_config(p)
    assert load_chain_config(DEFAULT_CHAIN_TOML).round.make_good_rerun_below_usd == 20.0
