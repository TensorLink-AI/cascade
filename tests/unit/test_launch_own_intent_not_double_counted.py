"""2026-11-05: an approved $89 make-good (5G6jNi6Y) was refused at a $205
balance — "balance $205.48 - committed $98.31 - this launch ≈$89.38 < $50".
The leg's own write-ahead ledger row (no payer ⇒ operator) was counted as
committed AND its projection was added again. The launch check now leaves
out the intent row of the pod it is about to create; every other operator
pod still counts."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from cascade.provision.core import OperatorRentsHeld
from cascade.provision.state import PodInstance
from cascade.shared.config import RoundConfig
from cascade.trainer import money_guard

LEG = "cascade-n91-777-funded-abc-op"


def _runner(tmp_path, *, balance, ledger):
    rnd = RoundConfig(funded_mode="required", funded_pod_sku="8xH100", epoch_blocks=900,
                      era_settlements=4, operator_max_price_per_hour=20.0,
                      funded_ready_timeout_seconds=0.0, operator_daily_spend_cap_usd=0.0)
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd, subnet=SimpleNamespace(netuid=91)),
                        work_root=tmp_path, _funded_leg_failures={})
    r._leg_wall_seconds = lambda sku=None: 3 * 3600.0
    r._funded_skus_for_rent = lambda: ("8xH100",)
    r._operator_balance_reader = lambda: balance
    rows = list(ledger)
    r._load_funded_ledger = lambda: rows
    return r


def _intent(pid):
    return PodInstance(provider="lium", instance_id=pid, stage="funded",
                       rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), payer_hotkey="")


class _Prov:
    name = "lium"

    def __init__(self, ids):
        self.ids, self.launched = list(ids), 0

    def launch(self, spec):
        self.launched += 1
        return list(self.ids)

    def price_of(self, pod_id):
        return None


def test_the_launching_pods_own_intent_row_is_not_committed_twice(tmp_path):
    # 150 - (own intent 60) - (projection 60) = 30 < 50 was the bug; 150 - 60 = 90 fits
    r = _runner(tmp_path, balance=150.0, ledger=[_intent(LEG + "-0")])
    prov = money_guard.guard_operator_provider(r, _Prov([LEG + "-0"]))
    assert prov.launch(SimpleNamespace(name_prefix=LEG)) == [LEG + "-0"]
    assert prov._provider.launched == 1


def test_other_operator_pods_still_count(tmp_path):
    other = "cascade-n91-777-funded-abc-op2"                 # a prefix lookalike counts
    r = _runner(tmp_path, balance=150.0, ledger=[_intent(LEG + "-0"), _intent(other + "-0")])
    prov = money_guard.guard_operator_provider(r, _Prov([LEG + "-0"]))
    with pytest.raises(OperatorRentsHeld, match="NO NEW OPERATOR RENTS"):
        prov.launch(SimpleNamespace(name_prefix=LEG))        # 150 - 60 - 60 = 30 < 50
    assert prov._provider.launched == 0


def test_committed_spend_without_launching_is_unchanged(tmp_path):
    r = _runner(tmp_path, balance=150.0, ledger=[_intent(LEG + "-0")])
    assert money_guard.committed_spend(r)["committed_usd"] > 0
    assert money_guard.committed_spend(r, launching=LEG)["committed_usd"] == 0
