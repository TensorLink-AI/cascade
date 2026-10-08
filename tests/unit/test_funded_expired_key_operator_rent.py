"""Owner 2026-10-08: "make sure for backlog pods that we fund round if their TTL expires".

A seated rolling challenger whose payer key aged out of the vault (36 h TTL) while the
marketplace was sold out used to die with an ``auth`` failure whenever no operator lane
was on file (mainnet has none). With ``[round] funded_operator_fallback`` on, the rolling
leg now RENTS a pod on the operator's Lium account instead — operator-billed, unburned,
same guards as every other rent. Live keys, the flag off, and non-rolling callers are
unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue
from cascade.provision import funded as funded_mod
from cascade.trainer.loop import (
    TrainerRunner,
    _FundedLegSkip,
    _FundedOperatorFallback,
    _FundedOperatorRent,
)
from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault
from tests.unit.test_funded_rent_wait import _arm_wait

REF_A = "alice/gen-a@sha256:" + "a" * 64
VAULT_REF = "vault/direct@sha256:" + "d" * 64


def _bind(r):
    for name in ("_operator_rent_allowed", "_operator_leg_pod_prefix",
                 "_is_operator_leg_pod_of", "_rent_operator_leg_host",
                 "_run_operator_rented_leg"):
        setattr(r, name, getattr(TrainerRunner, name).__get__(r))
    r.OPERATOR_LEG_POD_TAG = TrainerRunner.OPERATOR_LEG_POD_TAG
    return r


def _expired_runner(tmp_path, *, on=True, opted_in=True, queue=True):
    r = _bind(_runner(tmp_path, funded_operator_fallback=on))
    r.remote_hosts_path = None
    r.remote_hosts = []                                    # no operator lane on file
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    r._leg_local_obj = SimpleNamespace(operator_rent_ok=opted_in)
    (tmp_path / "pv").mkdir(exist_ok=True)                 # vault exists, key does not
    q = None
    if queue:
        q = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0)
        q.add("hkA", REF_A, reveal_block=1)
        r._funded_queue = lambda: q
    return r, q


def _reached_payer_rent(monkeypatch):
    class _Reached(Exception):
        pass

    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: (_ for _ in ()).throw(_Reached("payer rent")))
    return _Reached


# ── the decision in _rent_funded_host ───────────────────────────────────────


def test_expired_key_with_no_lane_rents_on_the_operator_and_flags_the_entry(tmp_path):
    r, q = _expired_runner(tmp_path)
    with pytest.raises(_FundedOperatorRent):
        r._rent_funded_host("777", _challenger("hkA"))
    assert q.get("hkA").operator_billed is True           # every retry stays on our bill
    assert "hkA" not in r._funded_leg_failures             # no auth verdict, no burn
    assert r._load_funded_ledger() == []                   # no payer write-ahead row


def test_flag_off_keeps_the_old_auth_verdict(tmp_path):
    r, q = _expired_runner(tmp_path, on=False)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, cls, burn = r._funded_leg_failures["hkA"]
    assert cls == "auth" and burn is False and "TTL expired" in msg
    assert q.get("hkA").operator_billed is False


def test_callers_that_did_not_opt_in_keep_the_old_behaviour(tmp_path):
    # The legacy round path never sets the thread-local opt-in.
    r, _q = _expired_runner(tmp_path, opted_in=False)
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    assert r._funded_leg_failures["hkA"][2] == "auth"


def test_a_live_key_still_rents_on_the_payers_account(tmp_path, monkeypatch):
    r, _q = _expired_runner(tmp_path)
    _vault(tmp_path, "hkA")
    reached = _reached_payer_rent(monkeypatch)
    with pytest.raises(reached):
        r._rent_funded_host("777", _challenger("hkA"))


def test_an_operator_billed_entry_with_no_lane_rents_on_the_operator(tmp_path, monkeypatch):
    r, q = _expired_runner(tmp_path)
    _vault(tmp_path, "hkA")                                 # even with a live key
    _reached_payer_rent(monkeypatch)
    q.set_operator_billed("hkA", True)
    with pytest.raises(_FundedOperatorRent):
        r._rent_funded_host("777", _challenger("hkA"))
    # not opted in ⇒ the plain lane fallback, exactly as before
    r._leg_local_obj = SimpleNamespace(operator_rent_ok=False)
    with pytest.raises(_FundedOperatorFallback) as ei:
        r._rent_funded_host("777", _challenger("hkA"))
    assert not isinstance(ei.value, _FundedOperatorRent)


# ── the operator rent itself ─────────────────────────────────────────────────


def _fake_lium(monkeypatch, *, ready_seq=(), live=None):
    import cascade.provision.core as core_mod
    import cascade.provision.funded as pf
    from cascade.provision.core import PodAddress

    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda host, **kw: "")
    torn, launched = [], []
    monkeypatch.setattr(pf, "terminate_verified", lambda prov, pod_id: torn.append(pod_id) or True)
    seq = list(ready_seq)

    class _Prov:
        name = "lium"
        def capacity(self, sku, *, gpus=1, exclude_ids=()):
            return 1
        def launch(self, spec):
            launched.append(spec)
            return [f"{spec.name_prefix}-0"]
        def wait_ready(self, pod_id, *, timeout):
            return seq.pop(0) if seq else True
        def get_ip(self, pod_id):
            return PodAddress(ip="9.9.9.9", ssh_port=41000)
        def machine_of(self, pod_id):
            return f"exec-{len(launched)}"
        def live_pod_address(self, pod_id):
            return live
        def terminate(self, pod_id):
            torn.append(pod_id)

    monkeypatch.setattr(core_mod, "LiumProvider", _Prov)
    return launched, torn


def _rent_runner(tmp_path, monkeypatch):
    r, _q = _expired_runner(tmp_path)
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[1] * 10)
    r._push_deployed_chain_toml = lambda host: host
    r._host_bench_below_floor = lambda host, sku, label: ""
    return r


def test_operator_rent_names_ledgers_and_retries_a_lemon(tmp_path, monkeypatch):
    launched, torn = _fake_lium(monkeypatch, ready_seq=[False, True])
    r = _rent_runner(tmp_path, monkeypatch)
    host, pod_id = r._rent_operator_leg_host("777", _challenger("hkA"))
    assert pod_id == "cascade-n91-777-funded-hka-op-0"
    assert (host.host, host.port) == ("9.9.9.9", 41000) and host.isolated
    assert len(launched) == 2 and torn == [pod_id]          # lemon torn down, rented again
    assert "exec-1" in launched[1].exclude_ids              # …its executor excluded
    rows = r._load_funded_ledger()
    assert [(x.instance_id, x.payer_hotkey) for x in rows] == [(pod_id, "")]   # operator marker


def test_operator_rent_requeues_unburned_when_nothing_appears(tmp_path, monkeypatch):
    _fake_lium(monkeypatch)
    r = _rent_runner(tmp_path, monkeypatch)
    import cascade.provision.core as core_mod
    core_mod.LiumProvider.capacity = lambda self, sku, *, gpus=1, exclude_ids=(): 0
    _arm_wait(r, deadline_offsets=-1, capacity_seq=[])       # already past the safe start
    with pytest.raises(_FundedLegSkip):
        r._rent_operator_leg_host("777", _challenger("hkA"))
    _msg, miner_fault, cls, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)


def test_operator_rent_adopts_a_live_pod_after_a_restart(tmp_path, monkeypatch):
    from cascade.provision.core import PodAddress

    launched, torn = _fake_lium(monkeypatch, live=PodAddress(ip="7.7.7.7", ssh_port=40001))
    r = _rent_runner(tmp_path, monkeypatch)
    host, pod_id = r._rent_operator_leg_host("777", _challenger("hkA"))
    assert (host.host, launched, torn) == ("7.7.7.7", [], [])


def test_operator_leg_stages_the_vault_zip_and_always_tears_down(tmp_path, monkeypatch):
    _fake_lium(monkeypatch)
    r = _rent_runner(tmp_path, monkeypatch)
    staged, torn = [], []
    r._stage_vault_zip_on = lambda host, digest: staged.append(digest) or host
    r._teardown_operator_pod = lambda pod: torn.append(pod.instance_id)

    class _Disp:
        def dispatch(self, host, **kw):
            assert kw["role"] == "challenger" and kw["hotkey"] == "hkA"
            raise RuntimeError("leg crashed")

    seeds = SimpleNamespace(base_seed=777)
    contract = SimpleNamespace(arch_preset="toto2-4m")
    with pytest.raises(RuntimeError, match="leg crashed"):
        r._run_operator_rented_leg(_Disp(), _challenger("hkA", ref=VAULT_REF), seeds, 5,
                                   contract, "-u1", warm_start_ref=None)
    assert staged == ["d" * 64]
    assert torn == ["cascade-n91-777-funded-hka-op-0"]


def test_the_sweep_recognises_in_flight_operator_leg_pods(tmp_path):
    r, _q = _expired_runner(tmp_path)
    pod = "cascade-n91-777-funded-hka-op-0"
    assert r._is_operator_leg_pod_of(pod, {"hkA"})
    assert not r._is_operator_leg_pod_of(pod, {"hkB"})
    assert not r._is_operator_leg_pod_of("cascade-n91-777-funded-hka-0", {"hkA"})   # payer pod
    assert not r._is_operator_leg_pod_of("cascade-n91-777-funded-king-0", {"hkA"})
    assert not r._is_operator_leg_pod_of("cascade-n259-777-funded-hka-op-0", {"hkA"})  # testnet
