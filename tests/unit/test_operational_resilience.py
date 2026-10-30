"""Recovery decisions use durable state, known owners and bounded provider effects."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config import RoundConfig
from cascade.shared.config.reliability import parse_reliability
from cascade.trainer.admission import FairSlots, owner_admission, remember_owners
from cascade.trainer.gpu_health import GpuHealth, GuardedProvider


def test_gpu_type_breaker_is_distinct_pod_based_and_survives_restart(tmp_path):
    clock = [1000.0]
    path = tmp_path / "health.json"
    health = GpuHealth(path, clock=lambda: clock[0])
    assert not health.record("H100", "lium:first", kind="lost")
    assert not health.record("H100", "lium:first", kind="lost")
    assert not health.record(
        "H100", "lium:account", kind="boot", reason="HTTP 402 insufficient balance"
    )
    assert not health.record("H100", "lium:rate", kind="boot", reason="HTTP 429")
    assert health.record("H100", "shadeform:second", kind="lost")
    restarted = GpuHealth(path, clock=lambda: clock[0])
    assert restarted.allowed(("H100", "RTX4090")) == ("RTX4090",)
    clock[0] += 10801
    assert restarted.allowed(("H100",)) == ("H100",)


def test_all_types_blocked_refuses_launch_without_provider_effect(tmp_path):
    from cascade.provision.core import LaunchSpec, ProvisionError

    clock = [10.0]
    health = GpuHealth(tmp_path / "h.json", threshold=1, clock=lambda: clock[0])
    health.record("H100", "p1", kind="lost")
    effects = []
    provider = GuardedProvider(
        SimpleNamespace(name="fixture", launch=lambda spec: effects.append(spec)), health
    )
    spec = LaunchSpec(
        sku="H100", count=1, image="fixture", ssh_pubkey="fixture", name_prefix="fixture"
    )
    with pytest.raises(ProvisionError, match="no_capacity"):
        provider.launch(spec)
    assert not effects


def test_owner_gate_skips_blocked_owner_without_starving_other_owners():
    gate = FairSlots()
    entered, release = [], threading.Event()

    def worker(label, owner):
        with gate.slot(3, owner=owner, owner_limit=1, deadline=time.time() + 5) as acquired:
            assert acquired
            entered.append(label)
            if label == "first":
                release.wait(3)

    first = threading.Thread(target=worker, args=("first", "same"))
    first.start()
    deadline = time.time() + 2
    while entered != ["first"] and time.time() < deadline:
        time.sleep(0.005)
    same = threading.Thread(target=worker, args=("same", "same"))
    other = threading.Thread(target=worker, args=("other", "other"))
    same.start()
    other.start()
    other.join(2)
    assert entered == ["first", "other"]
    release.set()
    first.join(2)
    same.join(2)
    assert entered == ["first", "other", "same"]
    assert gate.active == 0 and gate.owners == {}


def test_owner_gate_reentrant_handoff_and_unknown_identity_refusal():
    from cascade.trainer.faults import _FundedLegSkip

    failures = []
    runner = SimpleNamespace(
        cfg=SimpleNamespace(round=replace(RoundConfig(), ops_max_active_per_coldkey=1)),
        _record_funded_failure=lambda *a, **kw: failures.append(a),
    )
    with pytest.raises(_FundedLegSkip), owner_admission(runner, "hk"):
        pass
    remember_owners(runner, [SimpleNamespace(hotkey="hk", coldkey="owner", commit_block=1)])
    with owner_admission(runner, "hk"), owner_admission(runner, "hk"):
        assert runner._owner_slots.active == 1
    assert runner._owner_slots.active == 0
    assert failures


def test_reliability_policy_refuses_unbounded_paid_backups():
    with pytest.raises(ValueError, match="spending cap"):
        parse_reliability({"ops_king_backup_after_seconds": 60})
    with pytest.raises(ValueError):
        parse_reliability({"ops_operator_daily_cap_usd": float("nan")})


def test_host_faults_use_short_quarantine_and_account_faults_never_extend_it(tmp_path):
    import json

    from cascade.provision.core import record_host_quarantine

    path = tmp_path / "hosts.json"
    record_host_quarantine("host", "pod lost", path=path, now=10)
    assert json.loads(path.read_text())["host"]["until"] == 1810
    record_host_quarantine("host", "HTTP 402 insufficient balance", path=path, now=11)
    assert json.loads(path.read_text())["host"]["until"] == 1810
    record_host_quarantine("host", "pod lost again", path=path, now=12)
    assert json.loads(path.read_text())["host"]["until"] == 10812
    record_host_quarantine("stale", "runtime rejected: stale image", path=path, now=12)
    assert json.loads(path.read_text())["stale"]["until"] == 86412


def _monitor(tmp_path, *, clock, deliver=None):
    from cascade.trainer.operator_health import OperatorMonitor

    runner = SimpleNamespace(work_root=tmp_path, cfg=SimpleNamespace(round=RoundConfig()))
    return OperatorMonitor(runner, SimpleNamespace(), clock=clock, deliver=deliver)


def test_watchdog_retries_same_event_after_restart_and_records_resolution(tmp_path):
    import json

    now, attempts = [100.0], []

    def offline(event):
        attempts.append(event["id"])
        raise OSError("fixture endpoint down")

    mon = _monitor(tmp_path, clock=lambda: now[0], deliver=offline)
    snap = {"as_of": now[0], "conditions": [{"code": "low_balance", "scope": "lium"}]}
    mon.record(snap)
    saved = json.loads(mon.path.read_text())
    assert not saved["outbox"][0]["delivered"]
    delivered = []
    restarted = _monitor(tmp_path, clock=lambda: now[0], deliver=delivered.append)
    restarted.record({**snap, "as_of": 101})
    assert delivered[0]["id"] == attempts[0]
    restarted.record({"as_of": 102, "conditions": []})
    assert delivered[-1]["status"] == "resolved"
    assert not json.loads(mon.path.read_text())["active"]


def test_watchdog_does_not_resolve_old_conditions_or_duplicate_hung_sampler(tmp_path):
    now = [100.0]
    mon = _monitor(tmp_path, clock=lambda: now[0])
    mon.record({"as_of": now[0], "conditions": [{"code": "low_balance", "scope": "lium"}]})
    entered, release = threading.Event(), threading.Event()

    def hung():
        entered.set()
        release.wait(3)

    mon.sample = hung
    mon.poll()
    assert entered.wait(1)
    first = mon._sample_thread
    now[0] += 121
    for _ in range(4):
        mon.poll()
    assert mon._sample_thread is first
    status = mon.runner._operations_status
    assert status["stale"] and status["observed_as_of"] == 100
    assert {c["code"] for c in status["conditions"]} == {"low_balance", "watchdog_stalled"}
    release.set()
    first.join(1)


def test_watchdog_hold_delay_and_pending_alerts_survive_long_outage(tmp_path):
    mon = _monitor(tmp_path, clock=lambda: 0)
    hold = {"code": "settlement_waiting_for_king", "scope": "settlement"}
    mon.record({"as_of": 1, "conditions": [hold]})
    assert mon._load()["outbox"] == []
    mon.record({"as_of": 901, "conditions": [hold]})
    assert mon._load()["outbox"][0]["event"]["status"] == "active"
    for i in range(300):
        mon.record(
            {
                "as_of": 1000 + i,
                "conditions": [hold]
                + ([{"code": "low_balance", "scope": "lium"}] if i % 2 else []),
            }
        )
    pending = [e for e in mon._load()["outbox"] if not e["delivered"]]
    assert any(e["event"]["condition"] == hold for e in pending)
    assert len(pending) <= 3


def test_paid_fallback_requires_observed_payer_shortage_and_persistent_wait_age(tmp_path):
    from cascade.funding.queue import FundedQueue
    from cascade.shared.config import load_chain_config
    from cascade.trainer.loop import TrainerRunner

    cfg = load_chain_config("chain.toml")
    cfg = replace(
        cfg,
        round=replace(
            cfg.round,
            ops_paid_fallback_after_seconds=3600,
            ops_operator_daily_cap_usd=25,
            ops_operator_leg_cap_usd=5,
        ),
    )
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    queue = FundedQueue(tmp_path / "queue.json")
    queue.add("miner", "fixture-ref", 1)
    entry = queue.get("miner")
    runner._funded_queue = lambda: queue
    runner._funded_rent_wait_deadline = lambda: entry.funded_at + 7200
    runner._rent_wait_now = lambda: entry.funded_at + 4000
    runner._operator_fallback_lanes = lambda: []
    runner._funded_wait_abort = None
    assert (
        runner._wait_for_funded_capacity(
            ("H100",),
            describe="fixture",
            hotkey="miner",
            capacity_probe=lambda *a: 0,
            allow_paid_fallback=True,
        )
        == "operator_rent"
    )
    runner._king_pending = lambda: False
    assert (
        runner._wait_for_funded_capacity(
            ("H100",),
            describe="fixture",
            hotkey="miner",
            capacity_probe=lambda *a: 1,
            allow_paid_fallback=True,
        )
        is True
    )


def test_advancing_but_old_chain_head_is_reconnected_and_refused(cfg, tmp_path):
    from cascade.trainer.loop import TrainerRunner

    reconnects = []
    head = [1000]
    client = SimpleNamespace(
        current_block=lambda: head[0],
        reconnect=lambda: reconnects.append(True),
        block_timestamp=lambda block: time.time() - 1200,
    )
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    for block in (1000, 1001):
        head[0] = block
        with pytest.raises(RuntimeError, match="timestamp is stale"):
            runner._block_with_freeze_guard(client)
    assert len(reconnects) == 2
    client.block_timestamp = lambda block: time.time()
    assert runner._block_with_freeze_guard(client) == 1001


def test_owner_busy_sees_other_lanes_from_the_shared_scheduler():
    from cascade.trainer.admission import owner_busy

    runner = SimpleNamespace(
        cfg=SimpleNamespace(round=replace(RoundConfig(), ops_max_active_per_coldkey=1))
    )
    remember_owners(
        runner,
        [
            SimpleNamespace(hotkey=h, coldkey="same-owner", commit_block=1)
            for h in ("lane-a-miner", "lane-c-miner")
        ],
    )
    runner._rolling_sched = SimpleNamespace(
        _threads={}, lane_b=None, lane_c=SimpleNamespace(live_hotkeys=lambda: {"lane-c-miner"})
    )
    assert owner_busy(runner, "lane-a-miner")
    assert not owner_busy(runner, "lane-c-miner")


def test_rotation_releases_only_the_failed_allocation(cfg, tmp_path, monkeypatch):
    from cascade.provision.state import PodInstance
    from cascade.trainer.loop import TrainerRunner
    from cascade.trainer.remote import RemoteHost

    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    failed = RemoteHost("failed", "127.0.0.1", allocation_id="failed-id", sku="H100")
    healthy = RemoteHost("healthy", "127.0.0.2", allocation_id="healthy-id", sku="A100")
    runner._rolling_king_hosts = {1: failed, 2: healthy}
    runner._funded_king_host = healthy
    pods = [PodInstance("lium", name, "funded", "") for name in ("failed-id", "healthy-id")]
    removed = []
    monkeypatch.setattr(runner, "_load_funded_ledger", lambda: pods)
    monkeypatch.setattr(
        runner, "_teardown_operator_pod", lambda pod: removed.append(pod.instance_id)
    )
    runner._rolling_rotate_king_pod(SimpleNamespace(index=1, base_seed=77), "pod unreachable")
    assert removed == ["failed-id"]
    assert runner._rolling_king_hosts == {2: healthy}
    assert runner._funded_king_host is healthy


def test_backlog_alert_includes_in_flight_workers_waiting_for_gpu():
    from cascade.trainer.operator_health import inspect_health

    cfg = RoundConfig()
    old = SimpleNamespace(status="in_flight", phase="waiting_gpu", funded_at=1)
    training = SimpleNamespace(status="in_flight", phase="training", funded_at=1)
    report = inspect_health(
        now=10000,
        current_block=100,
        chain_timestamp=10000,
        era=None,
        last_settled_boundary=0,
        settlement_boundary=0,
        queues={"a": [old, training]},
        balances={},
        cfg=cfg,
    )
    assert report["queues"]["a"]["count"] == 1
    assert any(c["code"] == "queue_backlog" for c in report["conditions"])


def test_alert_redirect_is_not_recorded_as_delivered(tmp_path, monkeypatch):
    mon = _monitor(tmp_path, clock=lambda: 100)
    mon.runner.cfg.round = replace(RoundConfig(), ops_alert_webhook_env="FIXTURE_ALERT_URL")
    monkeypatch.setenv("FIXTURE_ALERT_URL", "https://alerts.invalid/fixture")
    calls = []

    def redirect(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=302, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.post", redirect)
    mon.record({"as_of": 100, "conditions": [{"code": "low_balance", "scope": "lium"}]})
    assert calls[0]["allow_redirects"] is False
    assert not mon._load()["outbox"][0]["delivered"]


def test_reused_pod_name_counts_new_allocations_but_not_repeated_failure_callbacks(tmp_path):
    from cascade.provision.core import LaunchSpec

    health = GpuHealth(tmp_path / "gpu.json")
    provider = SimpleNamespace(
        name="fixture", launch=lambda spec: ["reused-name"], wait_ready=lambda *a, **kw: False
    )
    guarded = GuardedProvider(provider, health)
    spec = LaunchSpec(
        sku="H100", count=1, image="fixture", ssh_pubkey="fixture", name_prefix="fixture"
    )
    guarded.launch(spec)
    guarded.wait_ready("reused-name", timeout=1)
    guarded.wait_ready("reused-name", timeout=1)
    assert not health.blocked()
    guarded.launch(spec)
    guarded.wait_ready("reused-name", timeout=1)
    assert "H100" in health.blocked()


def test_king_failure_keeps_the_exact_rental_credential_digest_after_scope_exits():
    from cascade.trainer.runner_funded_legs import funded_leg_scope

    runner = SimpleNamespace()
    with (
        pytest.raises(RuntimeError) as failure,
        funded_leg_scope(runner, None, object(), king_leg=True),
    ):
        runner._leg_local_obj.funded_key_digest = "exact-rental-credential-digest"
        raise RuntimeError("fixture provider rejected original key after rotation")
    assert failure.value.payer_key_digest == "exact-rental-credential-digest"
    assert not hasattr(runner._leg_local_obj, "funded_key_digest")


def test_settlement_hold_is_alerted_even_after_king_training_finishes():
    from cascade.trainer.operator_health import inspect_health

    cfg = RoundConfig()
    report = inspect_health(
        now=10000,
        current_block=100,
        chain_timestamp=10000,
        era=SimpleNamespace(king_entry={"trained": "complete"}),
        last_settled_boundary=0,
        settlement_boundary=100,
        queues={},
        balances={},
        cfg=cfg,
    )
    assert any(c["code"] == "settlement_held" for c in report["conditions"])


def test_a_timed_out_chain_read_rebuilds_the_connection(cfg, tmp_path):
    """Testnet 2026-10-29: one unanswered reply left the trainer's websocket
    hanging every later read until a restart. A timed-out read — in the
    scheduler's chain_read or anywhere in the tick — now reconnects."""
    from cascade.trainer.loop import TrainerRunner
    from cascade.trainer.rolling import LegOps

    reconnects = []
    client = SimpleNamespace(reconnect=lambda: reconnects.append(True))
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    runner.__dict__["_chain_client"] = client

    def hung(fn, seconds):
        raise TimeoutError()

    runner._with_deadline = hung
    with pytest.raises(TimeoutError):
        LegOps(runner).chain_read(lambda: 1)
    assert reconnects == [True]
    runner._with_deadline = lambda fn, seconds: fn()
    assert LegOps(runner).chain_read(lambda: 7) == 7
    assert reconnects == [True]                     # a healthy read never reconnects


def test_a_tick_that_times_out_reconnects_before_the_next_poll(cfg, tmp_path, monkeypatch):
    from cascade.trainer import loop as loop_mod
    from cascade.trainer.loop import TrainerRunner

    reconnects = []
    client = SimpleNamespace(reconnect=lambda: reconnects.append(True))
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    from cascade.trainer import operator_health

    monkeypatch.setattr(operator_health, "OperatorMonitor",
                        lambda *a: SimpleNamespace(start=lambda: None))
    runner._upkeep = lambda: None
    ticks = []

    def guard(c):
        ticks.append(1)
        if len(ticks) == 1:
            raise TimeoutError()
        raise SystemExit                           # stop the service loop
    runner._block_with_freeze_guard = guard
    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        runner.run_forever(client)
    assert reconnects == [True]
