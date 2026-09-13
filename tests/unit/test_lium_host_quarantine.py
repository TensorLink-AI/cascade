"""Lium host quarantine (2026-09-13): one physical host lists several executor
ids; 91.224.44.222 booted a stale cached worker under the pinned tag, then sat
RUNNING with no ports for the full 900 s on every later rent — each time under
a different id, so the per-id cooldown never kept us off it. A lemon pod now
quarantines its HOST, spends none of the miner's attempts, and the leg rents
again elsewhere."""
from __future__ import annotations

import json
import types
from dataclasses import replace

import pytest

from cascade.provision import core as core_mod
from cascade.provision import funded as funded_mod
from cascade.provision.core import (
    LIUM_HOST_QUARANTINE_SECONDS,
    LIUM_QUARANTINE_FILE_ENV,
    LiumProvider,
    quarantined_hosts,
    record_host_quarantine,
)
from cascade.provision.funded import LEMON_CLASS, quarantine_lemon_host
from tests.unit.test_funded_provision import HK, FakeProvider, _rent


def _use_file(monkeypatch, tmp_path):
    p = tmp_path / "quarantine.json"
    monkeypatch.setenv(LIUM_QUARANTINE_FILE_ENV, str(p))
    monkeypatch.setenv(core_mod.LIUM_COOLDOWN_FILE_ENV, str(tmp_path / "cooldown.json"))
    return p


# ── storage ─────────────────────────────────────────────────────────────────

def test_record_expiry_and_prune(monkeypatch, tmp_path):
    p = _use_file(monkeypatch, tmp_path)
    record_host_quarantine("10.0.0.1", "stale image", now=1000.0)
    record_host_quarantine("10.0.0.2", "never ready", now=2000.0, seconds=100.0)
    assert quarantined_hosts(now=2050.0) == {"10.0.0.1": "stale image", "10.0.0.2": "never ready"}
    assert quarantined_hosts(now=2150.0) == {"10.0.0.1": "stale image"}
    assert quarantined_hosts(now=1000.0 + LIUM_HOST_QUARANTINE_SECONDS + 1) == {}
    # A later write prunes expired entries; a re-record never SHORTENS a quarantine.
    record_host_quarantine("10.0.0.1", "again", now=1500.0, seconds=10.0)
    d = json.loads(p.read_text())
    assert set(d) == {"10.0.0.1", "10.0.0.2"}
    assert d["10.0.0.1"]["until"] == 1000.0 + LIUM_HOST_QUARANTINE_SECONDS
    record_host_quarantine("", "nothing", now=0.0)                 # no-op, never raises


def test_missing_or_corrupt_file_means_no_quarantine(monkeypatch, tmp_path):
    p = _use_file(monkeypatch, tmp_path)
    assert quarantined_hosts() == {}
    p.write_text("{not json", encoding="utf-8")
    assert quarantined_hosts() == {}
    p.write_text('{"10.0.0.9": "bare string"}', encoding="utf-8")
    assert quarantined_hosts() == {}


# ── listings skip every executor on a quarantined host ──────────────────────

def _provider(monkeypatch, hosts: dict[str, str]):
    def _run(argv):
        out = ('[{"id": "ex-a", "gpu_count": 1}, {"id": "ex-b", "gpu_count": 1}, '
               '{"id": "ex-c", "gpu_count": 1}]') if "ls" in argv else "[]"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    prov = LiumProvider(bin="lium", _run=_run)
    monkeypatch.setattr(prov, "_executor_hosts", lambda: hosts)
    return prov


def test_listing_skips_all_executors_of_a_quarantined_host(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    hosts = {"ex-a": "91.224.44.222", "ex-b": "91.224.44.222", "ex-c": "146.120.227.147"}
    prov = _provider(monkeypatch, hosts)
    assert [e["id"] for e in prov._list_executors("RTX4090")] == ["ex-a", "ex-b", "ex-c"]
    record_host_quarantine("91.224.44.222", "stale image")
    assert [e["id"] for e in prov._list_executors("RTX4090")] == ["ex-c"]
    assert prov.capacity("RTX4090") == 1
    # A fresh instance (another payer's provider, after a restart) sees it too.
    assert [e["id"] for e in _provider(monkeypatch, hosts)._list_executors("RTX4090")] == ["ex-c"]


def test_listing_without_a_host_map_keeps_the_market(monkeypatch, tmp_path):
    # The API is best-effort: an unmapped executor is NOT assumed bad (the
    # fingerprint check still guards the leg); a mapped one is.
    _use_file(monkeypatch, tmp_path)
    record_host_quarantine("91.224.44.222", "stale image")
    prov = _provider(monkeypatch, {"ex-b": "91.224.44.222"})
    assert [e["id"] for e in prov._list_executors("RTX4090")] == ["ex-a", "ex-c"]
    assert [e["id"] for e in _provider(monkeypatch, {})._list_executors("RTX4090")] == \
        ["ex-a", "ex-b", "ex-c"]


def test_executor_hosts_and_host_of_pod_read_the_api(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    monkeypatch.setenv("LIUM_API_KEY", "sk_test")
    calls = []

    class _R:
        ok = True

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    def get(url, headers=None, timeout=None):
        calls.append(url.rsplit("/", 1)[1])
        assert headers == {"X-API-KEY": "sk_test"}
        if url.endswith("/executors"):
            return _R([{"id": "ex-a", "executor_ip_address": "91.224.44.222"},
                       {"id": "ex-c", "executor_ip_address": "146.120.227.147"},
                       {"id": "ex-noip"}])
        return _R([{"pod_name": "pod-1", "executor": {"executor_ip_address": "91.224.44.81"}},
                   {"pod_name": "pod-2", "executor": {}}])

    monkeypatch.setitem(__import__("sys").modules, "requests",
                        types.SimpleNamespace(get=get))
    clock = [100.0]
    prov = LiumProvider(bin="lium", _run=lambda argv: types.SimpleNamespace(
        returncode=0, stdout="[]", stderr=""), _now=lambda: clock[0])
    assert prov._executor_hosts() == {"ex-a": "91.224.44.222", "ex-c": "146.120.227.147"}
    assert prov._executor_hosts() == {"ex-a": "91.224.44.222", "ex-c": "146.120.227.147"}
    assert calls.count("executors") == 1                           # cached
    clock[0] += LiumProvider._EXECUTOR_HOSTS_TTL + 1
    prov._executor_hosts()
    assert calls.count("executors") == 2                           # refreshed
    assert prov.host_of_pod("pod-1") == "91.224.44.81"             # the pod's own record
    prov._executor_by_name["pod-2"] = "ex-a"
    assert prov.host_of_pod("pod-2") == "91.224.44.222"            # launch record + map
    assert prov.host_of_pod("pod-unknown") == ""


def test_host_lookups_fail_soft_without_a_key_or_api(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    prov = LiumProvider(bin="lium", _run=lambda argv: types.SimpleNamespace(
        returncode=0, stdout="[]", stderr=""))
    assert prov._executor_hosts() == {}
    assert prov.host_of_pod("pod-1") == ""
    assert quarantine_lemon_host(prov, "pod-1", "never ready") == ""
    assert quarantined_hosts() == {}


# ── a never-ready rent is a lemon: host quarantined, no attempt burned ───────

class _LemonProvider(FakeProvider):
    def host_of_pod(self, pod_id: str) -> str:
        assert pod_id not in self.terminated                       # looked up BEFORE teardown
        return "91.224.44.222"


def test_never_ready_rent_quarantines_the_host_and_burns_nothing(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    provider = _LemonProvider(ready=False)
    res = _rent(provider, ready_timeout=1.0)
    assert not res.ok and res.error_class == LEMON_CLASS and res.burn_attempt is False
    assert "not ready within" in res.error
    assert provider.terminated == [f"{funded_mod.funded_pod_name('777', HK, 91)}-0"]
    q = quarantined_hosts()
    assert set(q) == {"91.224.44.222"} and "not ready within" in q["91.224.44.222"]


def test_other_rent_failures_keep_their_classes(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    res = _rent(_LemonProvider(fail_launch="lium: only 0 × 1xRTX4090 available, need 1"))
    assert res.error_class != LEMON_CLASS
    assert quarantined_hosts() == {}


# ── the funded leg rents again elsewhere, then settles infra unburned ────────

def _lemon(hotkey="hkA"):
    return funded_mod.FundedRentResult(hotkey=hotkey, ok=False, error="funded pod x not ready "
                                       "within 900s", error_class=LEMON_CLASS,
                                       burn_attempt=False)


def test_lemon_pod_rents_again_on_another_host(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

    _use_file(monkeypatch, tmp_path)
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    outcomes = [_lemon(), _rent_ok()]
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: outcomes.pop(0))
    runner._funded_pod_code_mismatch = lambda result, profile: ""
    runner._rent_wait_now = lambda: 0.0
    runner._funded_rent_wait_deadline = lambda: 1.0                # still inside the window
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert outcomes == [] and host.host == "10.9.9.9"
    assert "hkA" not in runner._funded_leg_failures              # nothing against the miner


def test_lemons_past_the_budget_or_deadline_settle_infra_unburned(tmp_path, monkeypatch):
    from cascade.trainer.loop import TrainerRunner, _FundedLegSkip
    from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault

    _use_file(monkeypatch, tmp_path)
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    rents = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _lemon())
    runner._rent_wait_now = lambda: 0.0
    runner._funded_rent_wait_deadline = lambda: 1.0
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == TrainerRunner.FUNDED_MAX_STALE_PODS
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "infra", False)
    assert "bad pod(s)" in msg and "not ready" in msg

    # Past the round's latest safe start: no second rent at all.
    runner2 = _runner(tmp_path)
    rents.clear()
    runner2._rent_wait_now = lambda: 5.0
    runner2._funded_rent_wait_deadline = lambda: 1.0
    with pytest.raises(_FundedLegSkip):
        runner2._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == 1
    assert runner2._funded_leg_failures["hkA"][1:] == (False, "infra", False)


def test_stale_image_pod_quarantines_its_host(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

    _use_file(monkeypatch, tmp_path)
    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    outcomes = [replace(_rent_ok(), address=core_mod.PodAddress(ip="91.224.44.222", ssh_port=1)),
                _rent_ok()]
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: outcomes.pop(0))
    monkeypatch.setattr(funded_mod, "teardown_funded", lambda pods, vault, **kw: [])
    verdicts = ["stale worker image: pod code aaaa… != pinned bbbb…", ""]
    runner._funded_pod_code_mismatch = lambda result, profile: verdicts.pop(0)
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert host.host == "10.9.9.9"
    assert set(quarantined_hosts()) == {"91.224.44.222"}
