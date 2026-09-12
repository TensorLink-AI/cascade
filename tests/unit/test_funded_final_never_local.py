"""Round 9046800 (2026-09-11/12) regressions — funded finals and operator lanes.

* A ``funded_mode = "required"`` final NEVER trains on the orchestrator: with an
  empty hosts.toml the trainer used to wait ``hosts_wait_seconds`` and then take
  the LOCAL branch, which skips the JIT king rent and the per-payer funded legs
  (and can only abort on ``assert_train_image``, since the box has no digest env).
* A final that rents its own pods does not wait for a hosts.toml fleet at all.
* No lanes + no rent ⇒ refuse loudly (the round retries), never block or go local.
* Operator lanes stage a vault/direct challenger's ZIP on the CHOSEN lane before
  each dispatch attempt (first try and the retry).
* A settled-round retry re-derives seating when the fleet grew and entries
  waited behind the smaller one, instead of restoring the capped field.
"""
from __future__ import annotations

import json
import queue
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from cascade.trainer.remote import RemoteHost
from tests.unit.test_trainer_round import _FakeBaseTrainer

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64
VAULT_REF = "vault/direct@sha256:" + "d" * 64


def _funded_cfg(cfg, *, pods: str, king_rent: bool):
    return replace(cfg, round=replace(cfg.round, funded_mode="required", funded_pods=pods,
                                      funded_king_rent=king_rent,
                                      funded_activation_block=0))


def _runner(cfg, tmp_path, **kw):
    import threading

    kw.setdefault("hosts_wait_seconds", 0)
    r = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                      trainer_spec="m:C", **kw)
    # The per-round funded state run_round initialises (these tests enter the
    # helpers below run_round's entry).
    r._funded_field = {}
    r._funded_leg_failures = {}
    r._funded_claimed_execs = set()
    r._funded_exec_lock = threading.Lock()
    r._funded_admission_info = {}
    r._funded_roster = {"seated": [], "waiting": [], "terminal": [], "outcomes": []}
    r._funded_round_sku = cfg.round.funded_pod_sku
    r._funded_king_host = None
    r._funded_king_lock = threading.Lock()
    return r


# ── empty hosts.toml under funded_mode="required" ───────────────────────────


def test_required_mode_empty_hosts_keeps_the_remote_branch_not_local(cfg, tmp_path):
    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text("", encoding="utf-8")            # the deployed final: 0 pods
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path,
                     remote_hosts_path=hosts_path)
    runner._reload_remote_hosts(require_stage="final")
    # [] (remote branch, no lanes yet) — NOT None (the local branch).
    assert runner.remote_hosts == []


def test_unfunded_round_still_falls_back_to_local(cfg, tmp_path):
    hosts_path = tmp_path / "hosts.toml"
    hosts_path.write_text("", encoding="utf-8")
    runner = _runner(replace(cfg, round=replace(cfg.round, funded_mode="off")), tmp_path,
                     remote_hosts_path=hosts_path)
    runner._reload_remote_hosts()
    assert runner.remote_hosts is None                     # pre-existing behaviour kept


def test_final_that_rents_its_pods_does_not_wait_for_a_fleet(cfg, tmp_path):
    hosts_path = tmp_path / "hosts.toml"                   # never written
    runner = _runner(_funded_cfg(cfg, pods="rent", king_rent=True), tmp_path,
                     remote_hosts_path=hosts_path, hosts_wait_seconds=600)
    t0 = time.monotonic()
    runner._reload_remote_hosts(require_stage="final")
    assert time.monotonic() - t0 < 5.0                      # no 600 s hold
    assert runner.remote_hosts == []


def test_train_final_takes_the_remote_branch_with_no_hosts_under_required(cfg, tmp_path,
                                                                            monkeypatch):
    runner = _runner(_funded_cfg(cfg, pods="rent", king_rent=True), tmp_path)
    runner.remote_hosts = []
    calls = []
    runner._train_remote = lambda *a, **k: calls.append("remote") or []
    runner._train_local = lambda *a, **k: calls.append("local") or []
    # The local branch's runtime check must not even be consulted.
    monkeypatch.setattr(loop_mod, "assert_train_image",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("local branch")))
    king = ResolvedGenerator(hotkey="k", uid=0, ref=REF_A)
    runner._train_final([(king, "king")], SimpleNamespace(base_seed=1), block=10)
    assert calls and set(calls) == {"remote"}


def test_no_lanes_and_no_rent_refuses_instead_of_blocking(cfg, tmp_path):
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    runner.remote_hosts = []
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="k", uid=0, ref=REF_A)
    with pytest.raises(RuntimeError, match="refusing to train the final on the orchestrator"):
        runner._train_remote([(king, "king")], SimpleNamespace(base_seed=1), 10, contract,
                             contract.train_tokens)


# ── vault ZIP staging on operator lanes ──────────────────────────────────────


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


def test_vault_challenger_on_operator_lane_is_staged_before_dispatch(cfg, tmp_path,
                                                                     monkeypatch):
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    lane = RemoteHost(name="lane", host="10.0.0.9", stage="final", workdir="/root/cascade")
    runner.remote_hosts = [lane]
    staged = []

    def fake_stage(host, digest):
        staged.append((host.name, digest))
        return replace(host, static_env=(*host.static_env,
                                         ("CASCADE_VAULT_DIR", f"{host.workdir}/_vault_stage")))

    runner._stage_vault_zip_on = fake_stage
    seen = []

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            seen.append((host, kw["role"]))
            return SimpleNamespace(hotkey=kw["hotkey"], role=kw["role"])

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="k", uid=0, ref=REF_A)
    chal = ResolvedGenerator(hotkey="c", uid=1, ref=VAULT_REF)
    runner._train_remote([(king, "king"), (chal, "challenger")],
                         SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)
    assert staged == [("lane", "d" * 64)]                  # only the vault ref stages
    chal_host = next(h for h, role in seen if role == "challenger")
    assert dict(chal_host.static_env)["CASCADE_VAULT_DIR"] == "/root/cascade/_vault_stage"
    king_host = next(h for h, role in seen if role == "king")
    assert "CASCADE_VAULT_DIR" not in dict(king_host.static_env)


# ── retry re-derives seating when the fleet grew ─────────────────────────────


def _marker(tmp_path, *, lanes, waiting):
    d = tmp_path / "1"
    d.mkdir()
    (d / "heat_complete.json").write_text(json.dumps({
        "round_id": "1", "screened": 2, "finalists": ["a"],
        "funded": {"field": {"a": REF_A}, "round_sku": "", "admission": {},
                   "roster": {"seated": ["a"], "waiting": waiting, "terminal": []},
                   "lanes": lanes},
    }), encoding="utf-8")


def _final_lanes(n):
    return [RemoteHost(name=f"l{i}", host=f"10.0.0.{i}", stage="final") for i in range(n)]


def test_retry_re_derives_when_lanes_grew_and_entries_waited(cfg, tmp_path):
    _marker(tmp_path, lanes=1, waiting=["b"])
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    runner.remote_hosts = _final_lanes(3)
    chal = [ResolvedGenerator("a", 1, REF_A), ResolvedGenerator("b", 2, REF_B)]
    assert runner._settled_finalists(1, chal) is None      # ⇒ seat from scratch


def test_retry_restores_the_settled_field_when_lanes_did_not_grow(cfg, tmp_path):
    _marker(tmp_path, lanes=3, waiting=["b"])
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    runner.remote_hosts = _final_lanes(3)
    chal = [ResolvedGenerator("a", 1, REF_A), ResolvedGenerator("b", 2, REF_B)]
    got = runner._settled_finalists(1, chal)
    assert [c.hotkey for c in got] == ["a"]
    assert runner._funded_field == {"a": REF_A}


def test_retry_restores_when_nobody_waited_even_if_lanes_grew(cfg, tmp_path):
    _marker(tmp_path, lanes=1, waiting=[])
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    runner.remote_hosts = _final_lanes(3)
    chal = [ResolvedGenerator("a", 1, REF_A)]
    got = runner._settled_finalists(1, chal)
    assert [c.hotkey for c in got] == ["a"]


def test_snapshot_records_the_lanes_the_seating_saw(cfg, tmp_path):
    runner = _runner(_funded_cfg(cfg, pods="off", king_rent=False), tmp_path)
    runner.remote_hosts = _final_lanes(2)
    runner._funded_field = {"a": REF_A}
    runner._mark_heat_complete(1, [ResolvedGenerator("a", 1, REF_A)],
                               [ResolvedGenerator("a", 1, REF_A)])
    raw = json.loads((tmp_path / "1" / "heat_complete.json").read_text())
    assert raw["funded"]["lanes"] == 2
