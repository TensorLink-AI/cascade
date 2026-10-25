"""Operator lanes stage a vault/direct challenger's ZIP on the CHOSEN lane
before each dispatch attempt (first try and the retry)."""
from __future__ import annotations

import queue
from dataclasses import replace
from types import SimpleNamespace

from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost


class _Disp:
    def __init__(self, fail_first: bool):
        self.fail_first = fail_first
        self.seen: list[RemoteHost] = []

    def dispatch(self, host, **kw):
        self.seen.append(host)
        if self.fail_first and len(self.seen) == 1:
            raise RuntimeError("lane A hiccup")
        return SimpleNamespace(host=host.name)


def _lanes():
    a = RemoteHost(name="a", host="10.0.0.1")
    b = RemoteHost(name="b", host="10.0.0.2")
    q = queue.Queue()
    q.put(a)
    q.put(b)
    return [a, b], q


def test_free_lane_dispatch_prepares_the_chosen_lane_on_both_attempts(monkeypatch):
    monkeypatch.setattr(loop_mod, "STORAGE_RETRY_BACKOFF_SECONDS", 0, raising=False)
    hosts, free = _lanes()
    disp = _Disp(fail_first=True)
    staged = []

    def prepare(host):
        staged.append(host.name)
        return replace(host, static_env=(("CASCADE_VAULT_DIR", "/root/cascade/_vault_stage"),))

    entry = TrainerRunner._dispatch_on_free_lane(disp, free, hosts, describe="x",
                                                 prepare=prepare, hotkey="h")
    assert entry.host == "b"
    assert staged == ["a", "b"]                            # staged before EACH attempt
    for h in disp.seen:
        assert dict(h.static_env)["CASCADE_VAULT_DIR"] == "/root/cascade/_vault_stage"


def test_free_lane_dispatch_without_prepare_is_unchanged():
    hosts, free = _lanes()
    disp = _Disp(fail_first=False)
    entry = TrainerRunner._dispatch_on_free_lane(disp, free, hosts, describe="x", hotkey="h")
    assert entry.host == "a" and disp.seen[0].static_env == ()
