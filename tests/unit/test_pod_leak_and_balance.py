"""2026-10-29 testnet: a released king pod came back BROKEN off-ledger and
billed for 6 h; Lium's balance refusal was read as a lemon host."""

from __future__ import annotations

import json
import logging
import types

import pytest

from cascade.funding.faults import classify_rent_failure
from cascade.provision.core import LiumProvider, ProvisionError

LIUM_403 = ('{"detail":"Renting needs a balance above '
            '15 minutes of the node\'s hourly price plus the same for each pod already running, '
            'starting or awaiting an automatic retry. Top up at https://lium.io/billing.",'
            '"status_code":403}')


def _never_listed(tmp_path, monkeypatch, up_output):
    monkeypatch.setattr(LiumProvider, "_up_log_path", staticmethod(lambda name: tmp_path / f"{name}.log"))
    (tmp_path / "pod-0.log").write_text(up_output)
    clock = {"t": 0.0}
    return LiumProvider(bin="lium", _run=lambda argv: types.SimpleNamespace(returncode=0, stdout="[]", stderr=""),
                        _sleep=lambda s: clock.__setitem__("t", clock["t"] + s), _now=lambda: clock["t"])


def test_balance_refusal_raises_as_a_funding_failure(tmp_path, monkeypatch):
    prov = _never_listed(tmp_path, monkeypatch, LIUM_403)
    with pytest.raises(ProvisionError) as err:
        prov.wait_ready("pod-0", timeout=900.0)
    assert classify_rent_failure(str(err.value)) == "auth"     # the payer tops up; no lemon retry


def test_a_plain_failed_up_is_still_a_lemon(tmp_path, monkeypatch):
    prov = _never_listed(tmp_path, monkeypatch, "executor in cooldown")
    assert prov.wait_ready("pod-0", timeout=900.0) is False


class _Runner:
    """Just enough of the trainer for the operator orphan sweep."""

    from cascade.trainer.runner_funded_pods import _FundedPodMixin as _M  # noqa: N814

    OPERATOR_ORPHAN_GRACE_SECONDS = _M.OPERATOR_ORPHAN_GRACE_SECONDS
    _sweep_operator_orphans = _M._sweep_operator_orphans
    _is_king_pod_of = _M._is_king_pod_of

    def __init__(self, ledger):
        self.cfg = types.SimpleNamespace(subnet=types.SimpleNamespace(netuid=259))
        self._ledger = ledger

    def _load_funded_ledger(self):
        return [types.SimpleNamespace(instance_id=x) for x in self._ledger]

    def _is_operator_leg_pod_of(self, name, hotkeys):
        return False


def test_operator_orphans_are_swept_from_the_listing(monkeypatch, caplog):
    pods = [
        {"name": "cascade-n259-3551-funded-king-0", "status": "BROKEN", "created_at": "2026-10-29T14:32:09"},
        {"name": "cascade-n259-7777-funded-king-0", "status": "RUNNING", "created_at": "2026-10-29T14:32:09"},  # current era
        {"name": "cascade-n259-8888-funded-king-0", "status": "RUNNING", "created_at": "2026-10-29T14:32:09"},  # ledgered
        {"name": "cascade-n91-3551-funded-king-0", "status": "RUNNING", "created_at": "2026-10-29T14:32:09"},   # mainnet
        {"name": "my-own-pod", "status": "RUNNING", "created_at": "2026-10-29T14:32:09"},
    ]
    removed = []
    monkeypatch.setattr(LiumProvider, "_list_pods", lambda self: list(pods))
    monkeypatch.setattr("cascade.provision.funded.terminate_verified",
                        lambda prov, name: removed.append(name) or True)
    runner = _Runner(["cascade-n259-8888-funded-king-0"])
    with caplog.at_level(logging.ERROR):
        runner._sweep_operator_orphans(("7777",), frozenset())
    assert removed == ["cascade-n259-3551-funded-king-0"]
    assert "was on no ledger" in caplog.text


def test_a_fresh_unledgered_pod_gets_its_grace(monkeypatch):
    import datetime as dt

    fresh = dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()
    monkeypatch.setattr(LiumProvider, "_list_pods",
                        lambda self: [{"name": "cascade-n259-1-funded-king-0", "created_at": fresh}])
    removed = []
    monkeypatch.setattr("cascade.provision.funded.terminate_verified",
                        lambda prov, name: removed.append(name) or True)
    _Runner([])._sweep_operator_orphans((), frozenset())
    assert removed == []


def test_json_fixture_is_valid():
    assert json.loads(LIUM_403)["status_code"] == 403
