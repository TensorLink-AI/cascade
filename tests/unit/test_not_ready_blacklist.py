"""A Lium executor that repeatedly never becomes ready lands in the persistent
blacklist (incident 2026-11-03: executor 3cfaf487… on 8.28.35.210 was "not
ready within 420s" for 5G6jNi6Y at 07:11 and again at 10:19 — the lemon
path's 3 h bad mark and 0.5 h host quarantine had expired, so the same lemon
was re-rented). Two host-side not-ready strikes within 7 d block it with the
short-pod rule's durations and IP escalation.

Every provider call is mocked: no rentals, no network."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from cascade.provision import core as core_mod
from cascade.provision import funded as funded_mod
from cascade.provision import short_pods
from cascade.provision.funded import LEMON_CLASS, record_not_ready_strike
from tests.unit.test_funded_provision import FakeProvider, _rent

EX = "3cfaf487-1eaa-4131-8770-b005aad23a0d"
OTHER = "0a0cd0c1-ccbf-40ea-afe9-74939ba87009"
DAY = 86400.0


@pytest.fixture
def blacklist(tmp_path, monkeypatch):
    f = tmp_path / "_train_work" / short_pods.FILE_NAME
    monkeypatch.setenv(core_mod.LIUM_QUARANTINE_FILE_ENV, str(tmp_path / "quarantine.json"))
    monkeypatch.setenv(core_mod.LIUM_COOLDOWN_FILE_ENV, str(tmp_path / "cooldown.json"))
    short_pods.set_path(f)
    yield f
    short_pods.set_path(None)


# ── the rule ─────────────────────────────────────────────────────────────────


def test_one_not_ready_strike_alone_blocks_nothing(blacklist):
    assert short_pods.not_ready_strike(executor_id=EX, ip="8.28.35.210",
                                       reason="not ready within 420s", now=0.0) == {}
    assert short_pods.blocked_ids(now=10.0) == (set(), set())
    assert short_pods.active(now=10.0) == {}


def test_two_not_ready_strikes_within_7d_block_with_the_short_pod_durations(blacklist):
    short_pods.not_ready_strike(executor_id=EX, ip="8.28.35.210",
                                reason="not ready within 420s", now=0.0)
    out = short_pods.not_ready_strike(executor_id=EX, ip="8.28.35.210",
                                      reason="not ready within 420s", now=3 * 3600.0)
    assert out[f"executor:{EX}"] == pytest.approx(3 * 3600.0 + short_pods.FIRST_STRIKE_SECONDS)
    ids, ips = short_pods.blocked_ids(now=4 * 3600.0)
    assert ids == {EX} and ips == set()         # one executor never blocks its NAT IP
    (rec,) = short_pods.active(now=4 * 3600.0).values()
    assert "not-ready" in rec["reason"] and "never ready 2x" in rec["reason"]
    # expired after 72 h; the next not-ready inside 7 d is a repeat block: 7 d
    t = 3 * 3600.0 + short_pods.FIRST_STRIKE_SECONDS + 60.0
    assert short_pods.blocked_ids(now=t) == (set(), set())
    short_pods.not_ready_strike(executor_id=EX, reason="not ready within 420s", now=t + 600)
    (rec,) = short_pods.active(now=t + 700).values()
    assert rec["until"] == pytest.approx(t + 600 + short_pods.REPEAT_STRIKE_SECONDS)


def test_not_ready_strikes_older_than_7d_do_not_count(blacklist):
    short_pods.not_ready_strike(executor_id=EX, reason="not ready", now=0.0)
    short_pods.not_ready_strike(executor_id=EX, reason="not ready", now=7 * DAY + 60.0)
    assert short_pods.blocked_ids(now=7 * DAY + 120.0) == (set(), set())
    short_pods.not_ready_strike(executor_id=EX, reason="not ready", now=8 * DAY)
    assert short_pods.blocked_ids(now=8 * DAY + 1.0)[0] == {EX}


def test_the_same_reject_seen_twice_is_one_strike(blacklist):
    short_pods.not_ready_strike(executor_id=EX, reason="not ready", now=0.0)
    short_pods.not_ready_strike(executor_id=EX, reason="not ready", now=5.0)
    assert short_pods.blocked_ids(now=10.0) == (set(), set())


def test_the_ip_escalates_only_over_two_distinct_blocked_executors(blacklist):
    for eid in (EX, OTHER):
        short_pods.not_ready_strike(executor_id=eid, ip="8.28.35.210", reason="nr", now=0.0)
    assert short_pods.blocked_ids(now=1.0) == (set(), set())
    short_pods.not_ready_strike(executor_id=EX, ip="8.28.35.210", reason="nr", now=3600.0)
    assert short_pods.blocked_ids(now=3601.0) == ({EX}, set())
    short_pods.not_ready_strike(executor_id=OTHER, ip="8.28.35.210", reason="nr", now=7200.0)
    assert short_pods.blocked_ids(now=7201.0) == ({EX, OTHER}, {"8.28.35.210"})


def test_a_hand_unblock_also_forgets_the_not_ready_count(blacklist):
    short_pods.not_ready_strike(executor_id=EX, reason="nr", now=0.0)
    short_pods.not_ready_strike(executor_id=EX, reason="nr", now=3600.0)
    assert short_pods.unblock(EX) == 2                      # the block + the count
    short_pods.not_ready_strike(executor_id=EX, reason="nr", now=7200.0)
    assert short_pods.blocked_ids(now=7201.0) == (set(), set())


# ── the listing filter (the #419 pattern) ────────────────────────────────────


def test_two_not_ready_rejections_filter_the_executor_from_the_lium_listing(
        blacklist, monkeypatch):
    rows = [{"id": EX, "huid": "lemon-host", "gpu_count": 8, "price_per_hour": 20.0},
            {"id": OTHER, "huid": "brave-eagle-3a", "gpu_count": 8, "price_per_hour": 21.0}]
    prov = core_mod.LiumProvider()
    monkeypatch.setattr(prov, "_cli", lambda args: SimpleNamespace(stdout=json.dumps(rows)))
    monkeypatch.setattr(prov, "_steer_hosts", lambda sku, shaped: shaped)
    monkeypatch.setattr(core_mod, "cooling_executors", lambda: set())
    now = time.time()
    short_pods.not_ready_strike(executor_id=EX, reason="not ready within 420s",
                                now=now - 3 * 3600.0)
    assert len(prov._list_executors("8xH200")) == 2          # one strike: still listed
    short_pods.not_ready_strike(executor_id=EX, reason="not ready within 420s", now=now)
    assert [e["id"] for e in prov._list_executors("8xH200")] == [OTHER]


# ── the funded rent path ─────────────────────────────────────────────────────


class _LemonProvider(FakeProvider):
    up_tail: str = ""

    def host_of_pod(self, pod_id: str) -> str:
        return "8.28.35.210"

    def listed_executor_of(self, pod_id: str) -> str:
        return EX

    def _up_log_tail(self, pod_id: str) -> str:
        return self.up_tail


def test_never_ready_funded_rents_strike_and_the_second_blocks(blacklist, monkeypatch):
    monkeypatch.setattr(short_pods, "RESTRIKE_IGNORE_SECONDS", 0.0)   # two rents, one test
    for _ in range(2):
        res = _rent(_LemonProvider(ready=False), ready_timeout=1.0)
        assert not res.ok and res.error_class == LEMON_CLASS and res.burn_attempt is False
    ids, ips = short_pods.blocked_ids()
    assert ids == {EX} and ips == set()


def test_a_miner_side_not_ready_never_strikes(blacklist, monkeypatch):
    monkeypatch.setattr(short_pods, "RESTRIKE_IGNORE_SECONDS", 0.0)
    for _ in range(2):
        p = _LemonProvider(ready=False)
        p.up_tail = "Error: 402 Payment Required — insufficient balance"
        res = _rent(p, ready_timeout=1.0)
        assert not res.ok and res.error_class != LEMON_CLASS
    assert short_pods.blocked_ids() == (set(), set())
    assert not blacklist.exists()


def test_a_ghost_create_or_a_provider_without_the_seam_never_strikes(blacklist):
    ghost = SimpleNamespace(listed_executor_of=lambda pod_id: "")
    assert record_not_ready_strike(ghost, "p-0", "not ready") == ""
    assert record_not_ready_strike(SimpleNamespace(), "p-0", "not ready") == ""

    def boom(pod_id):
        raise core_mod.ProvisionError("lium ps failed")

    assert record_not_ready_strike(SimpleNamespace(listed_executor_of=boom), "p-0", "x") == ""
    assert not blacklist.exists()
    assert record_not_ready_strike(SimpleNamespace(listed_executor_of=lambda p: EX),
                                   "p-0", "not ready", ip="8.28.35.210") == EX
    assert funded_mod.record_not_ready_strike is record_not_ready_strike
