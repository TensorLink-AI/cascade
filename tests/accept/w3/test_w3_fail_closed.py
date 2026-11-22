"""W3: storage, chain and grant errors fail closed with a typed 503 and
change nothing (plan §13: a transient fetch error is never read as
"absent"). Contract: README.md "Fail closed"."""

from __future__ import annotations

from .helpers import Env, Miner, checkpoint, digest_of, failing_storage, manifest


def _env_with_failing_storage(tmp_path):
    from cascade.lane_d.storage import FilesystemStorage

    inner = FilesystemStorage(tmp_path / "vault")
    store = failing_storage(inner)
    env = Env(tmp_path, storage=store)
    return env, store


def test_storage_down_at_submit_is_503(tmp_path):
    env, store = _env_with_failing_storage(tmp_path)
    m = Miner(env, "fc-a")
    store.failing = True
    r = m.submit(checkpoint("a"))
    assert r.status == 503 and r.code == "storage_unavailable", r
    store.failing = False
    assert env.queue() == []


def test_storage_down_at_ready_is_503_not_missing_object(tmp_path):
    env, store = _env_with_failing_storage(tmp_path)
    m = Miner(env, "fc-b")
    files = checkpoint("b")
    assert m.submit(files).status == 201
    m.upload(files)
    store.failing = True
    r = m.ready(files)
    assert r.status == 503 and r.code == "storage_unavailable", r
    store.failing = False
    assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "submitted"
    assert m.ready(files).status == 200


def test_chain_down_is_503(tmp_path):
    for op in ("current_block", "coldkey_of"):
        env = Env(tmp_path / op)
        m = Miner(env, f"fc-{op}")
        files = checkpoint(op)
        assert m.submit(files).status == 201
        m.upload(files)
        env.reads.fail.add(op)
        r = m.ready(files)
        assert r.status == 503 and r.code == "chain_unavailable", (op, r)
        env.reads.fail.clear()
        assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "submitted"


def test_registration_read_down_at_submit_is_503_not_unregistered(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "fc-c")
    env.reads.fail.add("coldkey_of")
    r = m.submit(checkpoint("c"))
    assert r.status == 503 and r.code == "chain_unavailable", r


def test_reveal_read_down_never_expires_an_entry(tmp_path):
    from .helpers import BLOCK, TTL

    env = Env(tmp_path)
    m = Miner(env, "fc-d")
    files = checkpoint("d")
    assert m.enter(files).status == 200
    env.reads.block = BLOCK + TTL + 1
    env.reads.fail.add("reveal_block")
    r = m.submit(checkpoint("d2"))
    assert r.status in (409, 503), r
    assert r.code in ("hotkey_entry_open", "chain_unavailable"), r
    env.reads.fail.clear()
    env.reads.reveals[(m.hotkey, digest_of(manifest(files)))] = BLOCK + 2
    assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "revealed"


def test_minter_down_is_503_and_nothing_recorded(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "fc-e")
    env.minter.fail = True
    r = m.submit(checkpoint("e"))
    assert r.status == 503 and r.code == "grants_unavailable", r
    assert env.queue() == []


def test_finalize_down_at_ready_is_503(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "fc-f")
    files = checkpoint("f")
    assert m.submit(files).status == 201
    m.upload(files)
    env.minter.fail = True
    r = m.ready(files)
    assert r.status == 503 and r.code == "grants_unavailable", r
    assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "submitted"
