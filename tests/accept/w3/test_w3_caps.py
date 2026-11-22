"""W3: open-entry caps, the advisory ``coldkey_cap_open`` (A8), the
``[lane_d_ops] queue_max_pending`` backpressure and ``ready_ttl_blocks``
(B10). Contract: README.md "Caps" and "TTL"."""

from __future__ import annotations

from .helpers import (
    BLOCK,
    TTL,
    Env,
    Miner,
    checkpoint,
    digest_of,
    lane_d_cfg,
    manifest,
    ops_cfg,
)

# ── A8: coldkey cap at ready (advisory) ─────────────────────────────────────


def test_ready_over_the_coldkey_cap_is_refused_advisory(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(max_open_per_coldkey=2))
    miners = [Miner(env, f"cap-{i}", coldkey="ck-shared") for i in range(3)]
    files = [checkpoint(f"cap{i}") for i in range(3)]
    assert miners[0].enter(files[0]).status == 200
    assert miners[1].enter(files[1]).status == 200
    r = miners[2].enter(files[2])
    assert r.status == 409 and r.code == "coldkey_cap_open", r
    assert r.json["advisory"] is True
    assert r.json["open"] == 2 and r.json["cap"] == 2
    row = env.entry(miners[2].hotkey, digest_of(manifest(files[2])))
    assert row["state"] == "submitted"
    assert "commit_payload" not in r.json


def test_coldkey_cap_counts_only_this_coldkey(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(max_open_per_coldkey=1))
    a = Miner(env, "capb-a", coldkey="ck-1")
    b = Miner(env, "capb-b", coldkey="ck-2")
    assert a.enter(checkpoint("ba")).status == 200
    assert b.enter(checkpoint("bb")).status == 200


def test_coldkey_cap_frees_when_an_entry_expires(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(max_open_per_coldkey=1))
    a = Miner(env, "capc-a", coldkey="ck-c")
    b = Miner(env, "capc-b", coldkey="ck-c")
    fa, fb = checkpoint("ca"), checkpoint("cb")
    assert a.enter(fa).status == 200
    r = b.enter(fb)
    assert r.code == "coldkey_cap_open", r
    env.reads.block = BLOCK + TTL + 1
    r = b.ready(fb)
    assert r.status == 200, r
    assert env.entry(a.hotkey, digest_of(manifest(fa)))["state"] == "ready_expired"


def test_coldkey_cap_frees_on_close(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(max_open_per_coldkey=1))
    a = Miner(env, "capd-a", coldkey="ck-d")
    b = Miner(env, "capd-b", coldkey="ck-d")
    fa, fb = checkpoint("da"), checkpoint("db")
    assert a.enter(fa).status == 200
    assert b.enter(fb).code == "coldkey_cap_open"
    env.app.close(a.hotkey, digest_of(manifest(fa)), "judged")
    assert env.entry(a.hotkey, digest_of(manifest(fa)))["state"] == "closed"
    assert b.ready(fb).status == 200


def test_queue_full_is_global_backpressure(tmp_path):
    env = Env(tmp_path, ops=ops_cfg(queue_max_pending=2))
    miners = [Miner(env, f"qf-{i}", coldkey=f"ck-q{i}") for i in range(3)]
    files = [checkpoint(f"qf{i}") for i in range(3)]
    for m, f in zip(miners, files, strict=True):
        assert m.submit(f).status == 201
        m.upload(f)
    assert miners[0].ready(files[0]).status == 200
    assert miners[1].ready(files[1]).status == 200
    r = miners[2].ready(files[2])
    assert r.status == 503 and r.code == "queue_full", r


# ── B10: ready_ttl_blocks ───────────────────────────────────────────────────


def test_ready_expires_after_ttl_without_a_reveal(tmp_path):
    """T18d: readied but not revealed within ready_ttl_blocks ⇒ ready_expired,
    the hotkey's open slot freed."""
    env = Env(tmp_path)
    m = Miner(env, "ttl-a")
    f1, f2 = checkpoint("t1"), checkpoint("t2")
    d1 = digest_of(manifest(f1))
    assert m.enter(f1).status == 200
    env.reads.block = BLOCK + TTL          # the last block inside the window
    assert env.entry(m.hotkey, d1)["state"] == "ready"
    assert m.submit(f2).code == "hotkey_entry_open"
    env.reads.block = BLOCK + TTL + 1
    assert env.entry(m.hotkey, d1)["state"] == "ready_expired"
    r = m.ready(f1)
    assert r.status == 409 and r.code == "ready_expired", r
    assert m.submit(f2).status == 201


def test_a_revealed_entry_does_not_expire(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "ttl-b")
    f1 = checkpoint("tb")
    d1 = digest_of(manifest(f1))
    assert m.enter(f1).status == 200
    env.reads.reveals[(m.hotkey, d1)] = BLOCK + 5
    env.reads.block = BLOCK + TTL + 50
    assert env.entry(m.hotkey, d1)["state"] == "revealed"
    assert m.submit(checkpoint("tb2")).code == "hotkey_entry_open"


def test_a_reveal_after_the_ttl_does_not_revive(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "ttl-c")
    f1 = checkpoint("tc")
    d1 = digest_of(manifest(f1))
    assert m.enter(f1).status == 200
    env.reads.reveals[(m.hotkey, d1)] = BLOCK + TTL + 3
    env.reads.block = BLOCK + TTL + 10
    assert env.entry(m.hotkey, d1)["state"] == "ready_expired"


def test_ttl_comes_from_the_lane_d_config(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(ready_ttl_blocks=7))
    m = Miner(env, "ttl-d")
    r = m.enter(checkpoint("td"))
    assert r.json["ready_expires_block"] == BLOCK + 7
    env.reads.block = BLOCK + 8
    assert env.entry(m.hotkey, r.json["digest"])["state"] == "ready_expired"


def test_expiry_is_persisted_across_restart(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "ttl-e")
    f1 = checkpoint("te")
    assert m.enter(f1).status == 200
    env.reads.block = BLOCK + TTL + 1
    assert env.entry(m.hotkey, digest_of(manifest(f1)))["state"] == "ready_expired"
    env.reads.block = BLOCK      # a lagging chain read after restart cannot revive it
    env.restart()
    assert env.entry(m.hotkey, digest_of(manifest(f1)))["state"] == "ready_expired"
