"""The funded pod sweep never cuts a pod that still matters, and a restart's
re-attach is never refused for a cold coldkey identity cache.

Failure class (2026-10-31, restarts at 20:52, 23:13 and 23:34): the startup /
reconcile sweep tore down live funded pods mid-training — one holding a
FINISHED but unharvested checkpoint — because it trusted the queue's keep-set
alone. The restart came up with an empty coldkey identity cache, so every
in-flight re-attach failed ``[no_capacity] coldkey identity is unavailable``,
was requeued, and its pod then fell out of the keep-set and was swept.

Fixed in layers, each pinned here:
- the sweep asks each candidate pod itself (``_sweep_keeps_pod``): a live
  detached worker, an exit-0 run not harvested since it ended, or an
  unreachable pod inside its grace window keeps the pod;
- the restart sweep runs AFTER the re-attach pass;
- the identity cache persists beside the trainer state, and a re-attached
  (already admitted) leg passes owner admission with an empty cache.
"""
from __future__ import annotations

import json
import threading
import types
from dataclasses import replace

import pytest

from cascade.funding.vault import PayerKeyVault
from cascade.provision import funded as funded_mod
from cascade.provision.core import PodAddress
from cascade.provision.state import PodInstance
from cascade.shared.config import RoundConfig
from cascade.trainer import admission as A
from cascade.trainer.remote import PodRunsProbe, RemoteHost, probe_pod_runs
from cascade.trainer.runner_funded_pods import _FundedPodMixin

NOW = 2_000_000.0
PROBE_OK = "__CASCADE_POD_PROBE__"


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _SSH:
    """Scripted ssh: answers per pod ip (the probe's ``user@ip`` target)."""

    def __init__(self, by_ip):
        self.by_ip = dict(by_ip)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, argv, timeout, stdin=None):
        target = next(a for a in argv if "@" in a)
        ip = target.split("@", 1)[1]
        self.calls.append((ip, argv[-1]))
        step = self.by_ip.get(ip, _proc(rc=255, stderr="Connection refused"))
        if isinstance(step, Exception):
            raise step
        return step


def _live(d="challenger-hka-777-ab12cd34"):
    return _proc(stdout=f"LIVE:_train_work/_dispatch/{d}\n{PROBE_OK}\n")


def _finished(age, d="challenger-hka-777-ab12cd34"):
    return _proc(stdout=f"FINISHED:{int(age)}:_train_work/_dispatch/{d}\n{PROBE_OK}\n")


def _idle():
    return _proc(stdout=f"{PROBE_OK}\n")


class _Adapter:
    """A provider account: ``pod_id -> PodAddress`` (None = listed, no ssh)."""

    def __init__(self, addrs):
        self.addrs = dict(addrs)

    def get_ip(self, pod_id):
        return self.addrs.get(pod_id)

    def list_tagged(self, prefix):
        return [p for p in self.addrs if p.startswith(prefix)]


class _Runner(_FundedPodMixin):
    """Just enough trainer for the sweep: the real mixin, fake edges."""

    def __init__(self, tmp_path, ssh, *, vault=None):
        self.cfg = types.SimpleNamespace(round=RoundConfig(funded_mode="required", funded_pods="rent"),
                                         subnet=types.SimpleNamespace(netuid=91))
        self.work_root = tmp_path
        self._funded_ledger_lock = threading.Lock()
        self._sweep_ssh_runner = ssh
        self.clock = [NOW]
        self._sweep_clock = lambda: self.clock[0]
        self._vault = vault
        self.revoked: list[str] = []
        self.operator_torn: list[str] = []

    def _funded_pod_profile(self):
        return RemoteHost(name="profile", host="127.0.0.1", user="root", key_path="/k",
                          workdir="/root/cascade", stage="final", profile_only=True)

    def _payer_vault(self):
        return self._vault

    def _revoke_robot(self, pod):
        self.revoked.append(pod.instance_id)

    def _teardown_operator_pod(self, pod):
        self.operator_torn.append(pod.instance_id)
        self._ledger_remove(pod.instance_id)

    def _sweep_operator_orphans(self, keep_ids, keep_payers):
        return None


def _addr(ip):
    return PodAddress(ip=ip, ssh_port=2222)


# ── the probe ────────────────────────────────────────────────────────────────

def test_probe_reads_live_and_finished_runs_and_never_reads_silence_as_idle():
    host = RemoteHost(name="p", host="10.0.0.1", workdir="/root/cascade")
    out = _proc(stdout=("LIVE:_train_work/_dispatch/a\n"
                        "FINISHED:120:_train_work/_dispatch/b\n" + PROBE_OK + "\n"))
    got = probe_pod_runs(host, runner=lambda argv, t, s: out)
    assert got == PodRunsProbe(reachable=True, live=("_train_work/_dispatch/a",),
                               finished_ok=(("_train_work/_dispatch/b", 120.0),))
    # transport failure, a refused host key, a truncated answer: all UNKNOWN
    assert not probe_pod_runs(host, runner=lambda argv, t, s: _proc(rc=255)).reachable
    assert not probe_pod_runs(host, runner=lambda argv, t, s: _proc(stdout="LIVE:x\n")).reachable

    def boom(argv, t, s):
        raise OSError("no route")

    assert not probe_pod_runs(host, runner=boom).reachable


def test_probe_command_checks_pid_liveness_and_exit_code_zero():
    from cascade.trainer.remote import _pod_runs_probe_command

    cmd = _pod_runs_probe_command(RemoteHost(name="p", host="h", workdir="/w"))
    assert "kill -0" in cmd and "exit_code" in cmd and "_train_work/_dispatch" in cmd
    assert cmd.rstrip().endswith(PROBE_OK)


# ── the guard ────────────────────────────────────────────────────────────────

def test_guard_keeps_a_pod_with_a_live_worker(tmp_path):
    r = _Runner(tmp_path, _SSH({"10.0.0.1": _live()}))
    assert r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")


def test_guard_releases_an_idle_reachable_pod(tmp_path):
    r = _Runner(tmp_path, _SSH({"10.0.0.1": _idle()}))
    assert not r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")


def test_guard_keeps_a_finished_run_until_it_is_harvested(tmp_path):
    ssh = _SSH({"10.0.0.1": _finished(600)})
    r = _Runner(tmp_path, ssh)
    adapter = _Adapter({"pod-a": _addr("10.0.0.1")})
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")           # exit 0, never pulled
    # a harvest of this endpoint BEFORE the run ended does not count (reused ip:port)
    r.clock[0] = NOW - 3600
    r._record_pod_harvest("10.0.0.1", 2222)
    r.clock[0] = NOW
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")
    # harvested after the run exited: the pod is free to go
    r._record_pod_harvest("10.0.0.1", 2222)
    assert not r._sweep_keeps_pod(adapter, "pod-a", what="t")


def test_harvest_records_survive_a_restart(tmp_path):
    r = _Runner(tmp_path, _SSH({"10.0.0.1": _finished(60)}))
    r._record_pod_harvest("10.0.0.1", 2222)
    fresh = _Runner(tmp_path, _SSH({"10.0.0.1": _finished(60)}))   # a new process
    assert not fresh._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")


def test_harvest_records_the_pod_endpoint_after_upload(tmp_path):
    from cascade.trainer.loop import _mark_harvested

    r = _Runner(tmp_path, _SSH({}))
    host = RemoteHost(name="funded-hka", host="10.0.0.9", port=2200)
    assert _mark_harvested(r, host, "entry") == "entry"
    assert json.loads((tmp_path / "pod_harvests.json").read_text()) == {"10.0.0.9:2200": NOW}
    assert _mark_harvested(types.SimpleNamespace(), host, "e") == "e"   # no recorder: a no-op


def test_guard_releases_an_unharvested_run_past_its_window(tmp_path):
    age = _FundedPodMixin.SWEEP_UNHARVESTED_KEEP_SECONDS + 60
    r = _Runner(tmp_path, _SSH({"10.0.0.1": _finished(age)}))
    assert not r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")


@pytest.mark.parametrize("addrs", [{"pod-a": _addr("10.0.0.1")}, {"pod-a": None}])
def test_guard_keeps_an_unreachable_pod_through_its_grace_then_releases(tmp_path, addrs):
    # ssh refused (first case) or the pod is listed with no ssh endpoint (second)
    r = _Runner(tmp_path, _SSH({}))
    adapter = _Adapter(addrs)
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")           # first seen unreachable
    r.clock[0] = NOW + _FundedPodMixin.SWEEP_UNREACHABLE_GRACE_SECONDS - 1
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")           # still inside the grace
    r.clock[0] = NOW + _FundedPodMixin.SWEEP_UNREACHABLE_GRACE_SECONDS
    assert not r._sweep_keeps_pod(adapter, "pod-a", what="t")       # unreachable that long


def test_unreachable_grace_survives_restarts(tmp_path):
    """Repeated restarts must not reset the grace forever: the first-seen
    stamp is persisted beside the ledger."""
    adapter = _Adapter({"pod-a": _addr("10.0.0.1")})
    r = _Runner(tmp_path, _SSH({}))
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")           # first seen down at NOW
    assert json.loads((tmp_path / "sweep_unreachable.json").read_text()) == {"pod-a": NOW}
    restarted = _Runner(tmp_path, _SSH({}))                         # a new process
    restarted.clock[0] = NOW + _FundedPodMixin.SWEEP_UNREACHABLE_GRACE_SECONDS - 1
    assert restarted._sweep_keeps_pod(adapter, "pod-a", what="t")
    again = _Runner(tmp_path, _SSH({}))
    again.clock[0] = NOW + _FundedPodMixin.SWEEP_UNREACHABLE_GRACE_SECONDS
    assert not again._sweep_keeps_pod(adapter, "pod-a", what="t")   # grace NOT reset
    assert json.loads((tmp_path / "sweep_unreachable.json").read_text()) == {}


def test_unreachable_stamps_clear_on_contact_and_old_ones_are_forgotten(tmp_path):
    (tmp_path / "sweep_unreachable.json").write_text(json.dumps(
        {"long-gone": NOW - _FundedPodMixin.SWEEP_UNREACHABLE_FORGET_SECONDS - 1,
         "pod-a": NOW - 60}))
    r = _Runner(tmp_path, _SSH({"10.0.0.1": _live()}))
    assert r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")
    assert json.loads((tmp_path / "sweep_unreachable.json").read_text()) == {}


def test_a_reachable_answer_resets_the_unreachable_grace(tmp_path):
    ssh = _SSH({})
    r = _Runner(tmp_path, ssh)
    adapter = _Adapter({"pod-a": _addr("10.0.0.1")})
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")
    ssh.by_ip["10.0.0.1"] = _live()
    r.clock[0] = NOW + 10_000
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")
    del ssh.by_ip["10.0.0.1"]
    assert r._sweep_keeps_pod(adapter, "pod-a", what="t")           # a fresh grace starts


def test_guard_keeps_when_it_cannot_decide(tmp_path):
    r = _Runner(tmp_path, _SSH({}))
    assert r._sweep_keeps_pod(None, "pod-a", what="t")              # no adapter

    def broken():
        raise RuntimeError("no profile")

    r._funded_pod_profile = broken
    assert r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")


def test_probe_never_trusts_the_shared_known_hosts(tmp_path):
    ssh = _SSH({"10.0.0.1": _idle()})
    r = _Runner(tmp_path, ssh)
    r._sweep_keeps_pod(_Adapter({"pod-a": _addr("10.0.0.1")}), "pod-a", what="t")
    # the probe argv — a reused ip:port's stale key must not read as "unreachable"
    host = r._sweep_probe_host("10.0.0.1", 2222)
    assert "UserKnownHostsFile=/dev/null" in host.ssh_options


# ── the sweep ────────────────────────────────────────────────────────────────

def _payer_pod(hk, n=0):
    return PodInstance(provider="lium", instance_id=f"cascade-n91-777-funded-{hk.lower()}-{n}",
                       stage="funded", rented_at_iso="2026-10-31T00:00:00Z",
                       sku="RTX4090", gpus=1, payer_hotkey=hk)


def test_restart_sweep_keeps_live_and_unharvested_pods_of_legs_no_longer_in_flight(
        tmp_path, monkeypatch):
    """The 2026-10-31 cut: three payer pods whose legs had been requeued
    (so none is in keep_payers) — one training, one holding a finished
    checkpoint, one idle. Only the idle one goes; the kept two stay on the
    ledger (else the listing sweep would read them as orphans)."""
    vault = PayerKeyVault(dir=tmp_path / "pv")
    for hk in ("hkA", "hkB", "hkC"):
        vault.insert(hk, f"sk_{hk}")
    ssh = _SSH({"10.0.0.1": _live(), "10.0.0.2": _finished(300), "10.0.0.3": _idle()})
    r = _Runner(tmp_path, ssh, vault=vault)
    pods = {hk: _payer_pod(hk) for hk in ("hkA", "hkB", "hkC")}
    ips = {"hkA": "10.0.0.1", "hkB": "10.0.0.2", "hkC": "10.0.0.3"}
    accounts = {f"sk_{hk}": _Adapter({pods[hk].instance_id: _addr(ips[hk])}) for hk in pods}
    r._funded_provider_factory = lambda key: accounts[key]
    for p in pods.values():
        r._ledger_add(p)
    torn, seen = [], {}
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda ps, v, **kw: (torn.extend(p.instance_id for p in ps), [])[1])
    monkeypatch.setattr(funded_mod, "reconcile_funded",
                        lambda owned, v, **kw: (seen.update(kw), [])[1])
    r._reconcile_funded_pods(keep_round_ids=("777",), keep_payers=set())
    assert torn == [pods["hkC"].instance_id]
    assert r.revoked == [pods["hkC"].instance_id]                   # kept pods keep their robot
    assert sorted(x.instance_id for x in r._load_funded_ledger()) == sorted(
        [pods["hkA"].instance_id, pods["hkB"].instance_id])
    # the listing sweep (off-ledger pods) carries the same guard
    keep = seen["keep"]
    assert keep(_Adapter({"x": _addr("10.0.0.1")}), "x", "hkA") is True
    assert keep(_Adapter({"x": _addr("10.0.0.3")}), "x", "hkC") is False


def test_operator_ledger_pod_with_a_live_worker_is_kept(tmp_path, monkeypatch):
    from cascade.provision.core import LiumProvider

    king = PodInstance(provider="lium", instance_id="cascade-n91-555-funded-king-0",
                       stage="funded", rented_at_iso="2026-10-31T00:00:00Z", payer_hotkey="")
    r = _Runner(tmp_path, _SSH({"10.0.0.5": _live("king-x-555-ab")}))
    r._ledger_add(king)
    monkeypatch.setattr(LiumProvider, "get_ip",
                        lambda self, pid: _addr("10.0.0.5") if pid == king.instance_id else None)
    r._reconcile_funded_pods(keep_round_ids=("777",), keep_payers=set())
    assert r.operator_torn == []
    assert [x.instance_id for x in r._load_funded_ledger()] == [king.instance_id]


def test_payer_account_listing_sweep_honours_the_keep_guard(tmp_path, monkeypatch):
    from cascade.provision.payer_pool import reconcile_payer_accounts

    class _Account:
        def __init__(self):
            self.terminated = []

        def list_named(self, prefix):
            return {"pod-live": "cascade-n91-777-funded-hka-0",
                    "pod-idle": "cascade-n91-778-funded-hka-0"}

        def terminate(self, pod_id):
            self.terminated.append(pod_id)

    acct = _Account()
    vault = PayerKeyVault(dir=tmp_path / "pv")
    vault.insert("hkA", "sk_a")
    monkeypatch.setattr(funded_mod, "terminate_verified",
                        lambda adapter, pid: (adapter.terminate(pid), True)[1])
    killed = reconcile_payer_accounts(
        [], vault, netuid=91, provider_factory=lambda key: acct, hotkeys=None, exclude=(),
        keep=lambda adapter, pod_id, hk: pod_id == "pod-live")
    assert killed == ["pod-idle"] and acct.terminated == ["pod-idle"]


# ── coldkey identity: persisted, and never blocks a re-attach ────────────────

def _admission_runner(tmp_path=None, limit=1):
    failures = []
    r = types.SimpleNamespace(
        cfg=types.SimpleNamespace(round=replace(RoundConfig(), ops_max_active_per_coldkey=limit)),
        _record_funded_failure=lambda *a, **kw: failures.append(a))
    if tmp_path is not None:
        r.work_root = tmp_path
    return r, failures


def test_identity_cache_persists_across_a_restart(tmp_path):
    r, _ = _admission_runner(tmp_path)
    A.remember_owners(r, [types.SimpleNamespace(hotkey="hk", coldkey="owner", commit_block=1)])
    assert (tmp_path / A.IDENTITY_FILE).is_file()
    restarted, failures = _admission_runner(tmp_path)               # a new process, cold memory
    assert A.owner_of(restarted, "hk") == "owner"
    with A.owner_admission(restarted, "hk"):
        assert restarted._owner_slots.active == 1
    assert failures == []


def test_persisted_identity_still_expires(tmp_path):
    (tmp_path / A.IDENTITY_FILE).write_text(
        json.dumps({"hk": ["owner", 1.0]}))                          # resolved long ago
    r, _ = _admission_runner(tmp_path)
    assert A.owner_of(r, "hk") == ""


def test_a_corrupt_identity_file_starts_cold_never_crashes(tmp_path):
    (tmp_path / A.IDENTITY_FILE).write_text("{not json")
    r, _ = _admission_runner(tmp_path)
    assert A.owner_of(r, "hk") == ""
    A.remember_owners(r, [types.SimpleNamespace(hotkey="hk", coldkey="o", commit_block=1)])
    assert A.owner_of(r, "hk") == "o"


def test_new_admission_still_needs_an_identity_but_a_reattach_does_not():
    from cascade.trainer.faults import _FundedLegSkip

    r, failures = _admission_runner()
    with pytest.raises(_FundedLegSkip), A.owner_admission(r, "hk"):
        pass
    assert failures and "coldkey identity is unavailable" in failures[0][1]
    A.mark_admitted(r, "hk")
    entered = []
    with A.owner_admission(r, "hk"), A.owner_admission(r, "hk"):    # outer + the leg's nested gate
        entered.append(True)
    assert entered == [True]
    A.clear_admitted(r, "hk")
    with pytest.raises(_FundedLegSkip), A.owner_admission(r, "hk"):
        pass


def test_a_reattached_leg_takes_its_slot_over_the_owner_limit():
    r, _ = _admission_runner(limit=1)
    A.remember_owners(r, [types.SimpleNamespace(hotkey=h, coldkey="same", commit_block=1)
                          for h in ("hk1", "hk2")])
    A.mark_admitted(r, "hk2")
    hold, done = threading.Event(), threading.Event()

    def first():
        with A.owner_admission(r, "hk1"):
            hold.set()
            done.wait(5)

    t = threading.Thread(target=first)
    t.start()
    assert hold.wait(5)
    with A.owner_admission(r, "hk2", deadline=0.0):                 # never waits, never refused
        assert r._owner_slots.owners["same"] == 2
    done.set()
    t.join(5)
    assert r._owner_slots.active == 0 and r._owner_slots.owners == {}


# ── the restart order ────────────────────────────────────────────────────────

def test_restart_sweep_runs_after_reattach_and_reattached_legs_are_admitted(cfg, tmp_path):
    from tests.unit.test_rolling import (
        EB,
        REF,
        Clock,
        FakeClient,
        FakeOps,
        _armed,
        _commit,
        _join,
        _sched,
    )

    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    sched.tick(client, b0)
    _join(sched)
    ops.commits.append(_commit("BRAV", REF["BRAV"], b0))
    q.add("BRAV", REF["BRAV"], reveal_block=b0)
    q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=era_start + 2 * EB,
                     era_index=sched.state.current.index, started_block=b0 + 1)

    events: list = []

    class _Ops(FakeOps):
        def commitments(self, client):
            events.append("commitments")
            return super().commitments(client)

        def sweep_pods(self, *, keep_round_ids, keep_payers):
            events.append(("sweep", "BRAV" in sched2._threads, set(keep_payers)))
            super().sweep_pods(keep_round_ids=keep_round_ids, keep_payers=keep_payers)

        def train_challenger(self, gen, era, block, *, end_wall):
            events.append(("leg", gen.hotkey, A.is_admitted(sched2.r, gen.hotkey)))
            return super().train_challenger(gen, era, block, end_wall=end_wall)

    ops2 = _Ops(tmp_path, clock)
    ops2.commits = list(ops.commits)
    gate = threading.Event()
    ops2.hold["BRAV"] = gate                                         # the leg stays running
    sched2, _ = _sched(armed, tmp_path, clock, ops=ops2)
    sched2.tick(client, b0 + 3)
    gate.set()
    _join(sched2)
    sweeps = [e for e in events if isinstance(e, tuple) and e[0] == "sweep"]
    # the startup sweep ran with BRAV's re-attached leg thread already live
    assert sweeps[0][1] is True and "BRAV" in sweeps[0][2]
    assert events.index("commitments") < events.index(sweeps[0])    # identities warmed first
    assert ("leg", "BRAV", True) in events                           # re-attach = admitted
    assert not A.is_admitted(sched2.r, "BRAV")                       # cleared when the leg ends
