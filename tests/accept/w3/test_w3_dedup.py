"""W3: exact-digest dedup on the weights sha256, before any eval spend
(plan §1.2 "Ownership and dedup", T2, T16). Contract: README.md "Dedup"."""

from __future__ import annotations

from .helpers import Env, Miner, checkpoint, make_reads, weights_sha


def test_exact_copy_of_a_published_checkpoint_is_refused_at_submit(tmp_path):
    """T2 / T16: the house or a published king resubmitted."""
    house = checkpoint("house", weights=b"HOUSE-WEIGHTS" * 64)
    env = Env(tmp_path, reads=make_reads(published={weights_sha(house)}))
    m = Miner(env, "dup-a")
    copy = checkpoint("copy", weights=house["weights.safetensors"])
    r = m.submit(copy)
    assert r.status == 409 and r.code == "duplicate_weights", r
    assert env.minter.minted == []
    assert env.queue() == []


def test_a_one_byte_perturbation_is_not_an_exact_duplicate(tmp_path):
    """T16: the perturbation is the margin's job, not dedup's."""
    house = checkpoint("house2", weights=b"HOUSE-WEIGHTS" * 64)
    env = Env(tmp_path, reads=make_reads(published={weights_sha(house)}))
    m = Miner(env, "dup-b")
    nudged = checkpoint("nudged", weights=house["weights.safetensors"][:-1] + b"!")
    assert m.submit(nudged).status == 201


def test_published_set_is_rechecked_at_ready(tmp_path):
    """Front-run: the weights became a published king between submit and
    ready (the crowning settlement)."""
    env = Env(tmp_path)
    m = Miner(env, "dup-c")
    files = checkpoint("c")
    assert m.submit(files).status == 201
    m.upload(files)
    env.reads.published.add(weights_sha(files))
    r = m.ready(files)
    assert r.status == 409 and r.code == "duplicate_weights", r


def test_same_weights_as_an_open_entry_are_refused(tmp_path):
    env = Env(tmp_path)
    a = Miner(env, "dup-d", coldkey="ck-1")
    b = Miner(env, "dup-e", coldkey="ck-2")
    w = b"SHARED-WEIGHTS" * 32
    assert a.enter(checkpoint("d", weights=w)).status == 200
    # different manifest (config differs) but the same weights bytes
    r = b.submit(checkpoint("e", weights=w))
    assert r.status == 409 and r.code == "duplicate_weights", r


def test_same_weights_after_the_open_entry_expires_are_accepted(tmp_path):
    from .helpers import BLOCK, TTL

    env = Env(tmp_path)
    a = Miner(env, "dup-f", coldkey="ck-1")
    b = Miner(env, "dup-g", coldkey="ck-2")
    w = b"EXPIRING-WEIGHTS" * 32
    assert a.enter(checkpoint("f", weights=w)).status == 200
    env.reads.block = BLOCK + TTL + 1
    assert b.submit(checkpoint("g", weights=w)).status == 201


def test_dedup_source_unavailable_fails_closed(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "dup-h")
    env.reads.fail.add("published_weights")
    r = m.submit(checkpoint("h"))
    assert r.status == 503 and r.code == "dedup_unavailable", r
    assert env.minter.minted == []
