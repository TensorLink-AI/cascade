"""After a trainer restart, an era whose king leg already completed must
re-find its operator king pod (2026-10-10).

The king pod's host lived only in memory, learned when the pod was rented or
its leg ran. The 02:59 restart came after era 2569's king leg had finished, so
the trainer forgot the pod. Queued payer verifications then said "wait" forever,
and the idle release kept the pod for that queue: an H100 sat at 0 % GPU for
hours. The pod name is deterministic, so the trainer looks it up again.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.trainer import loop as L
from cascade.trainer.loop import TrainerRunner, king_pod_name_prefix, rediscover_king_host
from cascade.trainer.remote import RemoteHost

from .test_rolling import Clock, FakeClient, FakeOps, _armed, _join, _sched  # noqa: F401

ERA = SimpleNamespace(index=2569, base_seed=13733685250937628265, king_hotkey="KING")
POD = king_pod_name_prefix(91, str(ERA.base_seed)) + "-0"
ADDR = SimpleNamespace(ip="162.243.203.115", ssh_port=20310)


class _Provider:
    def __init__(self, live: dict):
        self.live = live
        self.calls: list[str] = []

    def live_pod_address(self, name):
        self.calls.append(name)
        return self.live.get(name)


def _runner(pinned="ssh-ed25519 AAAA", pin_exc: Exception | None = None):
    contract = SimpleNamespace(arch_preset="toto2-4m")
    profile = SimpleNamespace(user="root", key_path="~/.ssh/k", remote_python="/py",
                              workdir="/root/cascade", chain_toml="chain.toml",
                              ssh_options=())
    r = SimpleNamespace(
        cfg=SimpleNamespace(subnet=SimpleNamespace(netuid=91), round=SimpleNamespace(),
                            throne_contracts=lambda: [contract]),
        _final_role_hosts={},
        cascade_bench_plan=None,
        _funded_pod_profile=lambda: profile,
        _funded_price_caps=lambda: {},
    )

    def _pin(pod_id, addr):
        if pin_exc is not None:
            raise pin_exc
        return pinned
    r._pin_king_host_key = _pin
    return r


@pytest.fixture
def provider(monkeypatch):
    p = _Provider({POD: ADDR})
    monkeypatch.setattr(L, "_lium_provider", lambda rnd, **kw: p)
    return p


def test_restart_rediscovers_the_finished_eras_king_pod(provider):
    r = _runner()
    host = rediscover_king_host(r, ERA)
    assert isinstance(host, RemoteHost)
    assert (host.host, host.port, host.pinned_host_key) == (ADDR.ip, ADDR.ssh_port, "ssh-ed25519 AAAA")
    assert host.isolated and host.forward_env == ()          # credential-free like a fresh rent
    assert r._rolling_king_hosts[ERA.index] is host
    assert r._final_role_hosts[("king", "toto2-4m", "KING")] is host
    assert provider.calls == [POD]
    # the verification path and the idle release now see the pod
    king = SimpleNamespace(miner_hotkey="KING")
    assert TrainerRunner._rolling_verify_host(r, king, ERA) is host
    assert TrainerRunner._rolling_king_pod_can_bench(r, ERA) is True
    assert rediscover_king_host(r, ERA) is host and provider.calls == [POD]   # cached


def test_missing_pod_is_not_usable_so_the_idle_release_proceeds(monkeypatch):
    p = _Provider({})
    monkeypatch.setattr(L, "_lium_provider", lambda rnd, **kw: p)
    r = _runner()
    assert rediscover_king_host(r, ERA) is None
    king = SimpleNamespace(miner_hotkey="KING")
    assert TrainerRunner._rolling_verify_host(r, king, ERA) is None      # verify waits
    assert TrainerRunner._rolling_king_pod_can_bench(r, ERA) is False    # release, don't hold
    assert len(p.calls) == 1                                             # backoff: no hammering


def test_unreachable_pod_is_treated_as_no_king_pod(provider):
    r = _runner(pin_exc=RuntimeError("sshd down"))
    assert rediscover_king_host(r, ERA) is None
    assert ERA.index not in r._rolling_king_hosts


def test_released_pod_is_never_revived(provider):
    r = _runner()
    r._rolling_king_released = {ERA.index}
    assert rediscover_king_host(r, ERA) is None
    assert provider.calls == []


def test_known_host_needs_no_lookup(provider):
    r = _runner()
    known = RemoteHost(name="funded-king", host="1.2.3.4", workdir="/w", cuda_device="0")
    r._rolling_king_hosts = {ERA.index: known}
    assert rediscover_king_host(r, ERA) is known
    assert TrainerRunner._rolling_king_pod_can_bench(r, ERA) is True
    assert provider.calls == []


def test_partial_runner_fails_safe():
    assert rediscover_king_host(SimpleNamespace(), ERA) is None


class _RediscoverOps(FakeOps):
    def __init__(self, tmp_path, clock):
        super().__init__(tmp_path, clock)
        self.rediscovered: list[int] = []

    def rediscover_king_pod(self, era):
        self.rediscovered.append(era.index)
        return None


def test_scheduler_restart_rediscovers_eras_with_a_finished_king_leg(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    client = FakeClient()
    sched, ops = _sched(armed, tmp_path, clock)
    b0 = armed.round.rolling_from_block + 5
    sched.tick(client, b0)
    _join(sched)
    sched.tick(client, b0 + 1)
    _join(sched)
    idx = sched.state.current.index
    assert sched.state.current.king_entry is not None
    ops2 = _RediscoverOps(tmp_path, clock)
    sched2, _ = _sched(armed, tmp_path, clock, ops=ops2)       # the restart
    sched2.tick(client, b0 + 2)
    _join(sched2)
    assert idx in ops2.rediscovered
