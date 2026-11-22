"""W3: the persistent queue (atomic writes, survives restart, rolls back on a
failed write) and secret hygiene (grants never logged or persisted).
Contract: README.md "Queue" and "Secrets"."""

from __future__ import annotations

import json
import logging
import os

from .helpers import (
    GRANT_SECRET,
    QUEUE,
    Env,
    Miner,
    call,
    checkpoint,
    digest_of,
    lane_d_cfg,
    manifest,
)


def _queue_file(env: Env):
    from cascade.lane_d import service

    return env.state_dir / service.QUEUE_FILENAME


def test_queue_file_is_json_in_the_state_dir(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "q-a")
    assert m.enter(checkpoint("a")).status == 200
    p = _queue_file(env)
    assert p.is_file(), p
    assert isinstance(json.loads(p.read_text(encoding="utf-8")), dict)


def test_queue_survives_restart(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(max_open_per_coldkey=2))
    a = Miner(env, "q-b", coldkey="ck-r")
    b = Miner(env, "q-c", coldkey="ck-r")
    c = Miner(env, "q-d", coldkey="ck-r")
    fa, fb, fc = checkpoint("b"), checkpoint("c"), checkpoint("d")
    ra = a.enter(fa)
    assert ra.status == 200 and b.enter(fb).status == 200
    assert c.submit(fc).status == 201
    c.upload(fc)
    before = env.queue()
    env.restart()
    after = env.queue()
    key = lambda e: (e["hotkey"], e["digest"])  # noqa: E731
    assert sorted(after, key=key) == sorted(before, key=key)
    # the restarted intake still enforces the coldkey cap and the open slot
    assert c.ready(fc).code == "coldkey_cap_open"
    assert a.submit(checkpoint("b2")).code == "hotkey_entry_open"
    # and still knows who owns what
    again = a.ready(fa)
    assert again.status == 200 and again.json["ready_block"] == ra.json["ready_block"]


def test_failed_state_write_changes_nothing(tmp_path, monkeypatch):
    """Fail closed: a ready whose state write fails is 503, the file on disk
    is the previous one, and memory is rolled back too."""
    env = Env(tmp_path)
    m = Miner(env, "q-e")
    files = checkpoint("e")
    dig = digest_of(manifest(files))
    assert m.submit(files).status == 201
    m.upload(files)
    before = _queue_file(env).read_bytes()

    def boom(*_a, **_k):
        raise OSError("w3: disk full")

    monkeypatch.setattr(os, "replace", boom)
    r = m.ready(files)
    assert r.status == 503 and r.code == "state_unavailable", r
    monkeypatch.undo()
    assert _queue_file(env).read_bytes() == before
    json.loads(before.decode("utf-8"))
    assert env.entry(m.hotkey, dig)["state"] == "submitted"
    env.restart()
    assert env.entry(m.hotkey, dig)["state"] == "submitted"


def test_a_torn_queue_file_never_starts_empty(tmp_path):
    """A corrupt queue file is a refusal to start, never a silent empty
    queue (an empty queue would forget every open slot)."""
    import pytest

    env = Env(tmp_path)
    m = Miner(env, "q-f")
    assert m.enter(checkpoint("f")).status == 200
    p = _queue_file(env)
    for name in os.listdir(env.state_dir):
        q = env.state_dir / name
        if q.is_file():
            q.write_text("{torn", encoding="utf-8")
    assert p.read_text(encoding="utf-8") == "{torn"
    with pytest.raises(Exception):  # noqa: B017 — any refusal to start
        env.restart()


def test_grants_never_reach_logs_or_state(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    env = Env(tmp_path)
    m = Miner(env, "q-g")
    files = checkpoint("g")
    r = m.submit(files)
    assert r.status == 201 and GRANT_SECRET in r.raw.decode()
    m.upload(files)
    assert m.ready(files).status == 200
    assert GRANT_SECRET not in caplog.text
    assert GRANT_SECRET not in env.files_text()
    q = call(env.app, "GET", QUEUE, ip="192.0.2.9")
    assert GRANT_SECRET not in q.raw.decode()


def test_queue_view_rows_carry_the_pinned_fields(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "q-h")
    files = checkpoint("h")
    assert m.enter(files).status == 200
    rows = env.queue()
    assert len(rows) == 1
    row = rows[0]
    for k in ("hotkey", "digest", "state", "context", "coldkey_at_ready", "ready_block",
              "ready_expires_block", "weights_sha256", "licence",
              "licence_attestation_sha256"):
        assert k in row, k
    assert row["state"] in ("submitted", "ready", "revealed", "ready_expired", "closed")
