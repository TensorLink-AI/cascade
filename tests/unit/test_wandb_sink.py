"""Optional wandb training mirror: gating, fan-out, and best-effort safety.

No real wandb is imported — a fake module is injected into ``sys.modules`` so
the tests run anywhere (the integration is an optional extra).
"""

from __future__ import annotations

import re
import sys
import types

from cascade.shared.config import WandbConfig
from cascade.trainer.wandb_sink import WandbSink, open_wandb_run, wandb_run_id


class _FakeRun:
    def __init__(self):
        self.logged: list[dict] = []
        self.finished = False

    def log(self, record):
        self.logged.append(record)

    def finish(self):
        self.finished = True


def _install_fake_wandb(monkeypatch, *, run=None, init_raises=False):
    """Put a stand-in ``wandb`` module on sys.modules; return the captured run."""
    captured = {"kwargs": None, "run": run or _FakeRun()}

    def init(**kwargs):
        captured["kwargs"] = kwargs
        if init_raises:
            raise RuntimeError("boom")
        return captured["run"]

    fake = types.ModuleType("wandb")
    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return captured


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    sink = open_wandb_run(
        WandbConfig(enabled=False), round_id="7", role="king",
        hotkey="hk", uid=1, size="toto2-4m",
    )
    assert sink is None


def test_online_without_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    _install_fake_wandb(monkeypatch)
    sink = open_wandb_run(
        WandbConfig(enabled=True, mode="online"), round_id="7", role="king",
        hotkey="hk", uid=1, size="toto2-4m",
    )
    assert sink is None  # missing key ⇒ skip, never crash


def test_offline_mode_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cap = _install_fake_wandb(monkeypatch)
    sink = open_wandb_run(
        WandbConfig(enabled=True, mode="offline", project="cascade-test", entity="team"),
        round_id="42", role="challenger", hotkey="hk9", uid=5, size="toto2-22m",
        config={"corpus_mode": "stream_cpu", "token_budget": 1000},
    )
    assert isinstance(sink, WandbSink)
    kw = cap["kwargs"]
    assert kw["project"] == "cascade-test"
    assert kw["entity"] == "team"
    assert kw["mode"] == "offline"
    assert kw["group"] == "42"
    assert "hotkey:hk9" in kw["tags"] and "size:toto2-22m" in kw["tags"]
    # The miner-facing metadata is on the run config so a miner can filter to it.
    assert kw["config"]["miner_hotkey"] == "hk9"
    assert kw["config"]["miner_uid"] == 5
    assert kw["config"]["corpus_mode"] == "stream_cpu"


def test_run_id_is_stable_and_unique_per_competitor():
    # Same (round, role, hotkey) ⇒ same id, so a retried round resumes its run
    # instead of minting a duplicate; a different competitor/size/stage differs.
    a = wandb_run_id("2", "king-toto2-4m", "hkA")
    assert a == wandb_run_id("2", "king-toto2-4m", "hkA")
    assert a != wandb_run_id("2", "challenger-toto2-4m", "hkB")  # role differs
    assert a != wandb_run_id("2", "king-toto2-22m", "hkA")       # size differs
    assert a != wandb_run_id("3", "king-toto2-4m", "hkA")        # round differs


def test_run_id_is_wandb_safe_for_odd_inputs():
    # Non [A-Za-z0-9_-] chars are sanitised; a digest still keeps distinct keys
    # apart even when the sanitised slugs would collide.
    rid = wandb_run_id("2", "heat-a/b c", "hk:1")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", rid)
    assert wandb_run_id("2", "heat-a/b/c", "x") != wandb_run_id("2", "heat-a-b-c", "x")


def test_retry_resumes_the_same_run(monkeypatch):
    # Two inits for the same (round, role, hotkey) carry the SAME id and
    # resume="allow" — wandb then resumes one run rather than piling up empties.
    monkeypatch.setenv("WANDB_API_KEY", "k")
    cap = _install_fake_wandb(monkeypatch)
    kw = dict(round_id="2", role="king-toto2-4m", hotkey="hkA", uid=0, size="toto2-4m")
    open_wandb_run(WandbConfig(enabled=True, mode="online"), **kw)
    first = cap["kwargs"]
    assert first["resume"] == "allow"
    assert first["id"] == wandb_run_id("2", "king-toto2-4m", "hkA")
    open_wandb_run(WandbConfig(enabled=True, mode="online"), **kw)
    assert cap["kwargs"]["id"] == first["id"]   # retry reuses the id ⇒ resume


def test_missing_wandb_package_returns_none(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb ⇒ ImportError
    sink = open_wandb_run(
        WandbConfig(enabled=True, mode="online"), round_id="7", role="king",
        hotkey="hk", uid=1, size="toto2-4m",
    )
    assert sink is None


def test_init_failure_degrades_to_none(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    _install_fake_wandb(monkeypatch, init_raises=True)
    sink = open_wandb_run(
        WandbConfig(enabled=True, mode="online"), round_id="7", role="king",
        hotkey="hk", uid=1, size="toto2-4m",
    )
    assert sink is None  # a broken wandb backend never blocks a round


def test_emit_and_finish_forward_to_run():
    run = _FakeRun()
    sink = WandbSink(run)
    sink.emit({"step": 50, "loss": 0.3})
    sink.emit({"event": "done", "final_loss": 0.1})
    sink.finish()
    assert run.logged == [{"step": 50, "loss": 0.3}, {"event": "done", "final_loss": 0.1}]
    assert run.finished is True


def test_emit_swallows_log_errors():
    class _Boom:
        def log(self, record):
            raise RuntimeError("network down")

        def finish(self):
            raise RuntimeError("still down")

    sink = WandbSink(_Boom())
    # Neither call raises — logging must never abort a training run.
    sink.emit({"step": 1, "loss": 1.0})
    sink.finish()


def test_config_section_loads(cfg):
    # chain.toml ships [wandb] ENABLED (owner 2026-07-18) — observability only,
    # and still gated at runtime on WANDB_API_KEY being present. The safety
    # property this file guards is the no-key/no-package/error no-op behaviour
    # (tests above), not the shipped toggle value.
    assert cfg.wandb.enabled is True
    assert cfg.wandb.project
    assert cfg.wandb.mode in ("online", "offline", "disabled")


def test_config_missing_section_defaults():
    # WandbConfig defaults stand alone (a chain.toml with no [wandb] block loads).
    w = WandbConfig()
    assert w.enabled is False and w.project == "cascade" and w.mode == "online"
