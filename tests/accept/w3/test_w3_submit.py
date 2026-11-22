"""W3: ``POST /v1/lane-d/submit`` (plan §1.2 "Flow" step 1). Contract:
README.md "Submit"."""

from __future__ import annotations

import json

from .helpers import (
    ALLOWED_FILES,
    CONTEXT,
    GRANT_SECRET,
    SUBMIT,
    Env,
    Miner,
    call,
    canonical,
    checkpoint,
    digest_of,
    manifest,
    ops_cfg,
)


def test_submit_grants_the_entry_prefix_through_the_chosen_minter(tmp_path):
    from cascade.lane_d import service

    env = Env(tmp_path)
    m = Miner(env, "sub-a")
    files = checkpoint("a")
    dig = digest_of(manifest(files))
    r = m.submit(files)
    assert r.status == 201, r
    prefix = f"lane-d/{m.hotkey}/{dig}/"
    body = r.json
    assert body["status"] == "granted"
    assert body["upload_path"] == service.UPLOAD_PATH
    assert body["hotkey"] == m.hotkey and body["digest"] == dig
    assert body["context"] == CONTEXT
    assert body["prefix"] == prefix
    # the grants are exactly what the minter minted, for the four files
    assert env.minter.minted == [(prefix, {n: len(files[n]) for n in ALLOWED_FILES})]
    assert body["grants"] == env.minter.mint(prefix, {n: len(files[n]) for n in ALLOWED_FILES})
    # the intake stored the canonical manifest itself (vault credential)
    assert env.store.get(prefix + "manifest.json") == canonical(manifest(files))
    row = env.entry(m.hotkey, dig)
    assert row is not None and row["state"] == "submitted"
    assert row["context"] == CONTEXT


def test_submit_uses_the_ops_vault_prefix(tmp_path):
    env = Env(tmp_path, ops=ops_cfg(vault_prefix="lane-d-testnet"))
    m = Miner(env, "sub-pfx")
    files = checkpoint("pfx")
    r = m.submit(files)
    assert r.status == 201, r
    assert r.json["prefix"] == f"lane-d-testnet/{m.hotkey}/{digest_of(manifest(files))}/"


def test_body_must_hash_to_the_declared_digest(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-b")
    files = checkpoint("b")
    body = canonical(manifest(files))
    wrong = digest_of(manifest(checkpoint("other")))
    r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body,
             headers=m.headers("submit", wrong, body))
    assert r.status == 400 and r.code == "digest_mismatch", r
    assert env.minter.minted == []


def test_manifest_must_name_exactly_the_four_files(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-c")
    for files in (checkpoint("c1", extra={"extra.py": b"# extra\n"}),
                  checkpoint("c2", drop=("model.py",)),
                  checkpoint("c3", extra={"weights.bin": b"x" * 16})):
        r = m.submit(files)
        assert r.status == 400 and r.code == "file_not_allowed", r
    assert env.minter.minted == []


def test_unparseable_or_malformed_manifest_is_bad_manifest(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-d")
    good = manifest(checkpoint("d"))
    bad_size = json.loads(json.dumps(good))
    bad_size["files"]["model.py"]["size"] = -1
    zero = json.loads(json.dumps(good))
    zero["files"]["config.json"]["size"] = 0
    bad_sha = json.loads(json.dumps(good))
    bad_sha["files"]["model.py"]["sha256"] = "zz"
    bodies = [b"{not json", b"[]", canonical(bad_size), canonical(zero), canonical(bad_sha)]
    for body in bodies:
        import hashlib

        dig = hashlib.sha256(body).hexdigest()
        r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body,
                 headers=m.headers("submit", dig, body))
        assert r.status == 400 and r.code == "bad_manifest", (body, r)


def test_context_must_be_armed(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-e")
    r = m.submit(checkpoint("e"), context=5000)
    assert r.status == 400 and r.code == "context_not_armed", r
    for raw in ("", "abc", "4096.0"):
        files = checkpoint("e2")
        body = canonical(manifest(files))
        h = m.headers("submit", digest_of(manifest(files)), body)
        h["X-Context-Length"] = raw
        r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body, headers=h)
        assert r.status == 400 and r.code in ("bad_context", "bad_signature"), r


def test_digest_header_must_be_64_hex(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-f")
    files = checkpoint("f")
    body = canonical(manifest(files))
    for dig in ("", "sha256:" + digest_of(manifest(files)), "ABC", "a" * 63):
        r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body,
                 headers=m.headers("submit", dig, body))
        assert r.status == 400 and r.code == "bad_digest", (dig, r)


def test_missing_hotkey_is_typed(tmp_path):
    env = Env(tmp_path)
    r = call(env.app, "POST", SUBMIT, body=b"{}")
    assert r.status == 400 and r.code == "missing_hotkey", r


def test_unregistered_hotkey_is_refused_before_any_grant(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-unreg", coldkey=None)
    r = m.submit(checkpoint("g"))
    assert r.status == 403 and r.code == "not_registered", r
    assert env.minter.minted == []


def test_resubmit_before_ready_replaces_the_hotkeys_entry(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-h")
    f1, f2 = checkpoint("h1"), checkpoint("h2")
    d1, d2 = digest_of(manifest(f1)), digest_of(manifest(f2))
    assert m.submit(f1).status == 201
    # the same digest again (a retry) is granted again, not a duplicate
    assert m.submit(f1).status == 201
    assert m.submit(f2).status == 201
    rows = [e for e in env.queue() if e["hotkey"] == m.hotkey]
    assert [r["digest"] for r in rows] == [d2], rows
    m.upload(f1)
    r = m.ready(f1)
    assert r.status == 404 and r.code == "unknown_entry", r
    assert env.entry(m.hotkey, d1) is None


def test_second_entry_while_one_is_open_is_refused(tmp_path):
    """T6: one open entry per hotkey (structural)."""
    env = Env(tmp_path)
    m = Miner(env, "sub-i")
    assert m.enter(checkpoint("i1")).status == 200
    r = m.submit(checkpoint("i2"))
    assert r.status == 409 and r.code == "hotkey_entry_open", r
    assert len(env.minter.minted) == 1


def test_another_hotkey_cannot_take_a_submitted_digest(tmp_path):
    """DigestOwned at the intake: a digest belongs to its first submitter."""
    env = Env(tmp_path)
    a, b = Miner(env, "own-a"), Miner(env, "own-b", coldkey="ck-b")
    files = checkpoint("shared")
    assert a.submit(files).status == 201
    r = b.submit(files)
    assert r.status == 409 and r.code == "digest_owned", r


def test_submit_response_carries_grants_but_queue_view_does_not(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sub-j")
    r = m.submit(checkpoint("j"))
    assert r.status == 201 and GRANT_SECRET in r.raw.decode(), r
    assert GRANT_SECRET not in json.dumps(env.queue())
