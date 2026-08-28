"""cascade.funding: vault custody, fault taxonomy, and queue lifecycle."""

from __future__ import annotations

import json
import os
import stat

import pytest

from cascade.funding import (
    FundedQueue,
    PayerKeyVault,
    classify_rent_failure,
    is_no_capacity,
    parse_retry_secs,
    rounds_needed,
    select_field,
    should_recover,
)
from cascade.funding.queue import FundedEntry

HK_A = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
HK_B = "5FakeHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


# ── vault ────────────────────────────────────────────────────────────────────


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_vault_memory_roundtrip_and_expiry():
    clock = FakeClock()
    v = PayerKeyVault(dir=None, ttl_seconds=100.0, clock=clock)
    v.insert(HK_A, "sk-live-secret")
    assert v.get(HK_A) == "sk-live-secret"
    clock.t += 101.0
    assert v.get(HK_A) is None          # expired ⇒ purged
    assert not v.has(HK_A)


def test_vault_persists_0600_and_hydrates(tmp_path):
    clock = FakeClock()
    v = PayerKeyVault(dir=tmp_path / "vault", ttl_seconds=1000.0, clock=clock)
    v.insert(HK_A, "sk-live-secret")
    path = tmp_path / "vault" / f"{HK_A}.json"
    assert path.is_file()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    # Fresh process, same dir: hydrate recovers the key (the teardown path).
    v2 = PayerKeyVault(dir=tmp_path / "vault", ttl_seconds=1000.0, clock=clock)
    assert v2.hydrate() == 1
    assert v2.get(HK_A) == "sk-live-secret"


def test_vault_hydrate_sweeps_expired_files(tmp_path):
    clock = FakeClock()
    v = PayerKeyVault(dir=tmp_path / "vault", ttl_seconds=10.0, clock=clock)
    v.insert(HK_A, "sk-old")
    clock.t += 11.0
    v2 = PayerKeyVault(dir=tmp_path / "vault", ttl_seconds=10.0, clock=clock)
    assert v2.hydrate() == 0
    assert not (tmp_path / "vault" / f"{HK_A}.json").exists()


def test_vault_refresh_restamps_ttl():
    clock = FakeClock()
    v = PayerKeyVault(dir=None, ttl_seconds=100.0, clock=clock)
    v.insert(HK_A, "sk")
    clock.t += 90.0
    assert v.refresh(HK_A)
    clock.t += 90.0                      # 180 total, but re-stamped at 90
    assert v.get(HK_A) == "sk"


def test_vault_rejects_traversal_hotkey(tmp_path):
    v = PayerKeyVault(dir=tmp_path / "vault")
    with pytest.raises(ValueError):
        v.insert("../../etc/passwd", "sk")


def test_vault_repr_never_carries_keys():
    v = PayerKeyVault(dir=None)
    v.insert(HK_A, "sk-live-secret")
    assert "sk-live-secret" not in repr(v)


# ── faults ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("msg,expected", [
    ("HTTP 401 unauthorized", "auth"),
    ("invalid api key supplied", "auth"),
    ("HTTP 429 too many requests", "rate_limited"),
    ("no lium offer matches the request", "no_capacity"),
    ("SKU sold out in every region", "no_capacity"),
    ("pod never appeared in `lium ps`", "infra"),
    ("ssh transport closed rc=255", "infra"),
])
def test_classify_rent_failure(msg, expected):
    assert classify_rent_failure(msg) == expected


def test_auth_short_circuits_capacity():
    # A permission error that also mentions offers must read as auth, never
    # as sold-out (sold-out waits forever; auth needs the miner).
    msg = "forbidden: no offer visible to this api key"
    assert classify_rent_failure(msg) == "auth"
    assert not is_no_capacity(msg)


def test_should_recover_rules():
    now = 10_000.0
    assert should_recover("sold out of L40S", now - 100_000, now)          # no time bound
    assert should_recover("429 too many requests", now - 60, now)
    assert not should_recover("429 too many requests", now - 7 * 3600, now)  # window over
    assert not should_recover("401 unauthorized", now - 60, now)            # miner's fix
    assert not should_recover("huggingface 429 rate limit", now - 60, now)  # CDN, not rent


def test_parse_retry_secs():
    assert parse_retry_secs("try again in 37 seconds") == 37
    assert parse_retry_secs("limit per 1 hour") == 120
    assert parse_retry_secs("allowed per 5 seconds") == 5
    assert parse_retry_secs("retry after 999999 seconds") == 7200  # capped
    assert parse_retry_secs("nothing numeric here") is None


# ── queue ────────────────────────────────────────────────────────────────────


def test_queue_add_idempotent_and_replace(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    assert q.add(HK_A, "repo-a@sha256:aa", reveal_block=100) == "queued"
    assert q.add(HK_A, "repo-a@sha256:aa", reveal_block=100) == "already-queued"
    assert q.add(HK_A, "repo-a2@sha256:ab", reveal_block=120) == "replaced"
    q.mark_in_round([HK_A])
    # A live round's entry is frozen — a re-fund cannot swap its ref.
    assert q.add(HK_A, "repo-a3@sha256:ac", reveal_block=130) == "already-queued"
    assert q.get(HK_A).ref == "repo-a2@sha256:ab"


def test_queue_persists_across_reload(tmp_path):
    path = tmp_path / "queue.json"
    q = FundedQueue(path, clock=FakeClock())
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    q2 = FundedQueue(path, clock=FakeClock())
    assert q2.get(HK_A).status == "queued"
    assert q2.get(HK_A).reveal_block == 100


def test_select_field_orders_by_reveal_block(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    q.add(HK_B, "repo-b@sha256:bb", reveal_block=90)
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    field = select_field(q.entries(), cap=1)
    assert [e.hotkey for e in field] == [HK_B]   # earliest reveal wins the slot
    assert len(select_field(q.entries(), cap=3)) == 2


def test_select_field_skips_non_queued():
    entries = [
        FundedEntry(HK_A, "r@sha256:aa", 100, 0.0, status="in_round"),
        FundedEntry(HK_B, "r@sha256:bb", 200, 0.0, status="queued"),
    ]
    assert [e.hotkey for e in select_field(entries, cap=5)] == [HK_B]


def test_requeue_infra_burns_attempt_and_exhausts(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    q.mark_in_round([HK_A])
    for _ in range(3):
        assert q.requeue(HK_A, error="pod dud", error_class="infra",
                         burn_attempt=True, max_attempts=3)
        q.mark_in_round([HK_A])
    assert not q.requeue(HK_A, error="pod dud", error_class="infra",
                         burn_attempt=True, max_attempts=3)
    assert q.get(HK_A).status == "failed"
    # A terminal entry frees the slot for a fresh fund.
    assert q.add(HK_A, "repo-a@sha256:aa", reveal_block=100) == "queued"
    assert q.get(HK_A).attempts == 0


def test_requeue_capacity_never_burns(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    for _ in range(10):
        q.mark_in_round([HK_A])
        assert q.requeue(HK_A, error="sold out", error_class="no_capacity",
                         burn_attempt=False)
    assert q.get(HK_A).status == "queued"
    assert q.get(HK_A).attempts == 0


def test_withdraw_only_while_queued(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    q.mark_in_round([HK_A])
    assert not q.withdraw(HK_A)
    q.requeue(HK_A, error="x", error_class="infra", burn_attempt=True)
    assert q.withdraw(HK_A)


def test_public_view_carries_no_key_shaped_fields(tmp_path):
    q = FundedQueue(tmp_path / "queue.json", clock=FakeClock())
    q.add(HK_A, "repo-a@sha256:aa", reveal_block=100)
    blob = json.dumps(q.public_view())
    assert "api_key" not in blob and "lium" not in blob.lower()


def test_rounds_needed_clamps():
    assert rounds_needed(0, cap=3) == 1
    assert rounds_needed(3, cap=3) == 1
    assert rounds_needed(4, cap=3) == 2
    assert rounds_needed(100, cap=3) == 4
    assert rounds_needed(100, cap=3, max_rounds=2) == 2
    with pytest.raises(ValueError):
        rounds_needed(1, cap=0)
