"""2026-09-21 incident, round 9281816817567242348: hand-rented operator lanes
joined the final's pool blind. A 4x pod whose sshd had died took the king's
ONE retry (the retry then went to the very lane it had just died on) and the
round aborted; a 6x pod at 183M tokens/s (floor 500M) trained six legs to
~20 %; the failure tore down three complete payer pods the retry needed for
its bench. Now: a join gate with a re-probe cooldown, dead-pod draining +
host quarantine on a transport failure, the retry never on the failed pod,
and kept payer pods that survive a mid-epoch round failure."""
from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace

from cascade.trainer import loop as L
from cascade.trainer.loop import TrainerRunner, _FinalLanePool
from cascade.trainer.remote import RemoteDispatchError


class _Host:
    def __init__(self, name, host="10.0.0.1", port=22, stage="final"):
        self.name, self.host, self.port, self.stage = name, host, port, stage


# ── the join gate ────────────────────────────────────────────────────────────

def test_gate_rejects_a_lane_and_regates_it_after_the_cooldown():
    verdict = {"b": "ssh rc=255: refused"}
    gate = lambda h: verdict.get(h.name)                      # noqa: E731
    fresh = [_Host("a"), _Host("b", host="10.0.0.2")]
    pool = _FinalLanePool([_Host("a")], lambda: fresh, gate_fn=gate)
    pool.REJECT_RETRY_S = 0.05
    pool._absorb_new()
    assert [h.name for h in pool.known_hosts()] == ["a"]      # b not admitted
    pool._absorb_new()
    assert [h.name for h in pool.known_hosts()] == ["a"]      # not re-gated yet
    verdict.clear()                                            # the pod came back
    time.sleep(0.06)
    pool._absorb_new()
    assert sorted(h.name for h in pool.known_hosts()) == ["a", "b"]
    assert pool.get(block=False).name == "a" and pool.get(block=False).name == "b"


def test_gate_crash_is_a_rejection_not_a_wedge():
    def boom(h):
        raise RuntimeError("probe exploded")
    pool = _FinalLanePool([], lambda: [_Host("a")], gate_fn=boom)
    pool._absorb_new()
    assert pool.known_hosts() == [] and pool.empty()


def test_round_start_lanes_are_taken_as_sized_only_joins_are_gated():
    # the round's geometry was computed on the hosts file; the gate guards
    # the mid-final joins (the blind spot), never the sized fleet
    gate = lambda h: "slow host" if h.name == "slow" else None  # noqa: E731
    pool = _FinalLanePool([_Host("ok"), _Host("slow", host="10.0.0.9")], lambda: [], gate_fn=gate)
    assert [h.name for h in pool.known_hosts()] == ["ok", "slow"]


# ── dead-pod draining ───────────────────────────────────────────────────────

def test_mark_dead_drains_every_lane_of_the_pod_and_keeps_the_rest():
    a0, a1, b0 = _Host("a-g0", port=3039), _Host("a-g1", port=3039), _Host("b-g0", host="10.0.0.2")
    pool = _FinalLanePool([a0, a1, b0], lambda: [a0, a1, b0])
    pool.REJECT_RETRY_S = 0.05
    assert pool.get(block=False) is a0                          # a0 checked out, then dies
    assert sorted(pool.mark_dead(a0, "ssh refused")) == ["a-g0", "a-g1"]
    assert [h.name for h in pool.known_hosts()] == ["b-g0"]
    assert pool.get(block=False) is b0 and pool.empty()
    pool._absorb_new()                                          # hosts file still lists them
    assert [h.name for h in pool.known_hosts()] == ["b-g0"]     # cooldown holds
    time.sleep(0.06)
    pool._absorb_new()
    assert sorted(h.name for h in pool.known_hosts()) == ["a-g0", "a-g1", "b-g0"]


# ── the free-lane dispatch after a transport failure ─────────────────────────

class _Disp:
    def __init__(self, fail_on: dict[str, Exception]):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def dispatch(self, host, **kw):
        self.calls.append(host.name)
        exc = self.fail_on.pop(host.name, None)
        if exc is not None:
            raise exc
        return SimpleNamespace(host=host.name)


def test_transport_failure_drains_the_pod_quarantines_it_and_retries_elsewhere(monkeypatch):
    quarantined = []
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine",
                        lambda ip, reason, **kw: quarantined.append((ip, reason)))
    a0, a1, b0 = (_Host("a-g0", host="66.153.184.218", port=3039),
                  _Host("a-g1", host="66.153.184.218", port=3039),
                  _Host("b-g0", host="85.242.110.88", port=40100))
    hosts = [a0, a1, b0]
    pool = _FinalLanePool(hosts, lambda: hosts)
    pool.REJECT_RETRY_S = 60.0
    d = _Disp({"a-g0": RemoteDispatchError("remote king on a-g0: pod unreachable for 907s "
                                            "while the detached worker ran — leg lost",
                                            returncode=255)})
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="king", role="king")
    assert out.host == "b-g0" and d.calls == ["a-g0", "b-g0"]
    assert [h.name for h in pool.known_hosts()] == ["b-g0"]
    assert quarantined and quarantined[0][0] == "66.153.184.218"


def test_prepare_step_ssh_255_counts_as_transport_failure(monkeypatch):
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine", lambda *a, **k: None)
    a0, b0 = _Host("a-g0", host="66.153.184.218", port=3039), _Host("b-g0", host="10.0.0.2")
    hosts = [a0, b0]
    pool = _FinalLanePool(hosts, lambda: hosts)
    err = subprocess.CalledProcessError(255, ["ssh", "-p", "3039", "root@66.153.184.218",
                                              "mkdir -p /root/cascade/_vault_stage"])

    def prepare(h):
        if h.name == "a-g0":
            raise err
        return h
    d = _Disp({})
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x", prepare=prepare,
                                               role="challenger")
    assert out.host == "b-g0" and d.calls == ["b-g0"]
    assert [h.name for h in pool.known_hosts()] == ["b-g0"]


def test_miner_fault_keeps_the_lane_and_the_old_retry(monkeypatch):
    called = []
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine",
                        lambda *a, **k: called.append(a))
    a0 = _Host("a-g0", host="10.0.0.1")
    pool = _FinalLanePool([a0], lambda: [a0])
    d = _Disp({"a-g0": RemoteDispatchError("miner submission rejected: training diverged "
                                            "(nan)", returncode=3)})
    out = TrainerRunner._dispatch_on_free_lane(d, pool, [a0], describe="x", role="challenger")
    assert out.host == "a-g0" and d.calls == ["a-g0", "a-g0"]   # same silicon, still good
    assert [h.name for h in pool.known_hosts()] == ["a-g0"] and called == []


def test_transport_failure_classification():
    assert L._transport_failure(RemoteDispatchError("x", returncode=255))
    assert L._transport_failure(RemoteDispatchError("leg: detached launch failed (rc=1): boom",
                                                    returncode=1))
    assert L._transport_failure(subprocess.CalledProcessError(255, ["ssh"]))
    assert not L._transport_failure(subprocess.CalledProcessError(3, ["ssh"]))
    assert not L._transport_failure(RemoteDispatchError("training diverged", returncode=3))
    assert not L._transport_failure(RuntimeError("torn dispatch"))


def test_loopback_hosts_are_never_quarantined(monkeypatch):
    called = []
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine",
                        lambda *a, **k: called.append(a))
    L._quarantine_lane_host(_Host("profile", host="127.0.0.1", port=2222), "x")
    L._quarantine_lane_host(_Host("lane", host="", port=22), "x")
    assert called == []


# ── the operator lane gate's three checks ────────────────────────────────────

def _runner_for_gate(monkeypatch, *, rc=0, out="LANE_OK\n", runtime="", slow=""):
    monkeypatch.setattr("cascade.trainer.remote.run_ssh",
                        lambda argv, timeout=None, stdin_text=None: SimpleNamespace(
                            returncode=rc, stdout=out, stderr="refused" if rc else ""))
    monkeypatch.setattr("cascade.trainer.remote.probe_worker_runtime", lambda h, **k: runtime)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(funded_pod_sku="RTX4090")),
                           _funded_round_sku="",
                           _host_bench_below_floor=lambda host, sku, label: slow)
    return TrainerRunner._operator_lane_gate.__get__(fake)


def test_operator_lane_gate_checks_ssh_runtime_and_floor(monkeypatch):
    quarantined = []
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine",
                        lambda ip, reason, **kw: quarantined.append(ip))
    from cascade.trainer.remote import RemoteHost

    lane = RemoteHost(name="lane", host="66.153.184.218", port=3039, user="root",
                      key_path="~/.ssh/k", workdir="/root/cascade")
    assert _runner_for_gate(monkeypatch)(lane) is None
    assert _runner_for_gate(monkeypatch, rc=255, out="")(lane).startswith("ssh rc=255")
    assert quarantined == ["66.153.184.218"]
    assert _runner_for_gate(monkeypatch, runtime="worker lacks --local-only")(lane) == \
        "worker lacks --local-only"
    assert _runner_for_gate(monkeypatch, slow="slow host: 183M < 500M")(lane).startswith("slow host")


# ── kept payer pods across a round failure ───────────────────────────────────

def test_kept_pods_survive_a_failure_inside_the_epoch_only():
    now = 1_000_000.0
    assert L._kept_pods_survive_failure(now + 3600.0, now)
    assert not L._kept_pods_survive_failure(now - 1.0, now)
    assert not L._kept_pods_survive_failure(None, now)


def test_dispatch_retry_skips_a_sibling_lane_handed_back_by_a_peer(monkeypatch):
    """A sibling lane of the dead pod re-enters the rotation between the
    drain and the retry (a peer's non-transport failure put it back): the
    retry still refuses it."""
    monkeypatch.setattr("cascade.provision.core.record_host_quarantine", lambda *a, **k: None)
    a0, a1, b0 = (_Host("a-g0", host="66.153.184.218", port=3039),
                  _Host("a-g1", host="66.153.184.218", port=3039),
                  _Host("b-g0", host="10.0.0.2"))
    hosts = [a0, a1, b0]
    pool = _FinalLanePool(hosts, lambda: [])

    class _Pool(type(pool)):
        pass
    real_mark_dead = pool.mark_dead

    def mark_dead_then_peer_returns_sibling(host, reason):
        dead = real_mark_dead(host, reason)
        pool.put(a1)                                   # the peer hands a1 back
        return dead
    pool.mark_dead = mark_dead_then_peer_returns_sibling
    d = _Disp({"a-g0": RemoteDispatchError("pod unreachable for 900s", returncode=255)})
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x", role="challenger")
    assert out.host == "b-g0" and d.calls == ["a-g0", "b-g0"]
