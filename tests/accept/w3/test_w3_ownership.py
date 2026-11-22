"""W3: ownership proof — the hotkey's sr25519 signature over the canonical
request (plan §1.2 "Ownership and dedup"; the ``/v1/submit`` rules reused via
``cascade.shared.sr25519``). Contract: README.md "Ownership"."""

from __future__ import annotations

import json

from .helpers import (
    CONTEXT,
    READY,
    SUBMIT,
    Env,
    Miner,
    call,
    canonical,
    checkpoint,
    digest_of,
    manifest,
    message,
)


def test_canonical_message_is_the_pinned_bytes():
    from cascade.lane_d import service

    body = b'{"x":1}'
    for action in ("submit", "ready"):
        got = service.canonical_message(action, "5Hk", "ab" * 32, 8192, "1800000000", body)
        assert got == message(action, "5Hk", "ab" * 32, 8192, "1800000000", body)
    assert service.FRESHNESS_SECONDS == 300


def test_signature_by_another_key_is_refused(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-a")
    Miner(env, "sig-mallory", coldkey="ck-m")
    r = m.submit(checkpoint("a"), signer="sig-mallory")
    assert r.status == 400 and r.code == "bad_signature", r
    assert env.minter.minted == []


def test_missing_or_garbage_signature_is_refused(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-b")
    files = checkpoint("b")
    body = canonical(manifest(files))
    dig = digest_of(manifest(files))
    for sig in ("", "zz", "00" * 64):
        h = m.headers("submit", dig, body)
        h["X-Signature"] = sig
        r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body, headers=h)
        assert r.status == 400 and r.code == "bad_signature", (sig, r)


def test_signature_binds_the_body(tmp_path):
    """A ready body changed after signing (e.g. the licence) is refused."""
    env = Env(tmp_path)
    m = Miner(env, "sig-c")
    files = checkpoint("c")
    assert m.submit(files).status == 201
    m.upload(files)
    from cascade.lane_d import service

    signed = json.dumps({"licence": "Apache-2.0",
                         "attestation": service.LICENCE_ATTESTATION}).encode()
    sent = json.dumps({"licence": "OpenMDW-1.1",
                       "attestation": service.LICENCE_ATTESTATION}).encode()
    h = m.headers("ready", digest_of(manifest(files)), signed)
    r = call(env.app, "POST", READY, ip=m.ip, body=sent, headers=h)
    assert r.status == 400 and r.code == "bad_signature", r


def test_signature_binds_the_context(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-d")
    files = checkpoint("d")
    body = canonical(manifest(files))
    h = m.headers("submit", digest_of(manifest(files)), body, context=CONTEXT)
    h["X-Context-Length"] = "8192"
    r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body, headers=h)
    assert r.status == 400 and r.code == "bad_signature", r


def test_signature_binds_the_action(tmp_path):
    """A captured submit signature cannot be replayed as a ready."""
    env = Env(tmp_path)
    m = Miner(env, "sig-e")
    files = checkpoint("e")
    assert m.submit(files).status == 201
    m.upload(files)
    from cascade.lane_d import service

    body = json.dumps({"licence": "Apache-2.0",
                       "attestation": service.LICENCE_ATTESTATION}).encode()
    h = m.headers("submit", digest_of(manifest(files)), body)
    r = call(env.app, "POST", READY, ip=m.ip, body=body, headers=h)
    assert r.status == 400 and r.code == "bad_signature", r


def test_stale_timestamp_is_refused(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-f")
    for ts in (str(int(env.clock.t) - 301), str(int(env.clock.t) + 301), "nan", "abc", ""):
        r = m.submit(checkpoint("f"), ts=ts)
        assert r.status == 400 and r.code == "stale_timestamp", (ts, r)


def test_replayed_timestamp_is_refused(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-g")
    files = checkpoint("g")
    ts = m.next_ts()
    assert m.submit(files, ts=ts).status == 201
    r = m.submit(files, ts=ts)
    assert r.status == 409 and r.code == "replayed_timestamp", r
    older = str(int(ts) - 5)
    r = m.submit(files, ts=older)
    assert r.status == 409 and r.code == "replayed_timestamp", r


def test_ready_by_a_hotkey_that_did_not_submit_is_not_owner(tmp_path):
    """The uploader must be the committer: B cannot ready A's entry, even
    though B signs correctly for B."""
    env = Env(tmp_path)
    a, b = Miner(env, "own-ra"), Miner(env, "own-rb", coldkey="ck-b")
    files = checkpoint("ra")
    assert a.submit(files).status == 201
    a.upload(files)
    r = b.ready(files)
    assert r.status == 403 and r.code == "not_owner", r
    assert env.entry(a.hotkey, digest_of(manifest(files)))["state"] == "submitted"


def test_header_hotkey_must_be_the_signer(tmp_path):
    env = Env(tmp_path)
    a, b = Miner(env, "own-sa"), Miner(env, "own-sb", coldkey="ck-b")
    files = checkpoint("sa")
    assert a.submit(files).status == 201
    a.upload(files)
    # B signs a message naming A's hotkey and sends it as A
    r = a.ready(files, signer=b.name)
    assert r.status == 400 and r.code == "bad_signature", r


def test_verifier_unavailable_is_503_never_bad_signature(tmp_path, monkeypatch):
    """DEC-CA-0051: a broken SDK is 'cannot verify', not a verdict."""
    import cascade.shared.sr25519 as sr

    env = Env(tmp_path)
    m = Miner(env, "sig-h")
    files = checkpoint("h")

    def broken():
        raise sr.SignatureVerifierUnavailable("w3: sdk unavailable")

    monkeypatch.setattr(sr, "keypair_class", broken)
    r = m.submit(files)
    assert r.status == 503 and r.code == "signature_unavailable", r
    assert env.minter.minted == []


def test_unsigned_digest_header_cannot_be_swapped(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "sig-i")
    f1, f2 = checkpoint("i1"), checkpoint("i2")
    body2 = canonical(manifest(f2))
    h = m.headers("submit", digest_of(manifest(f1)), canonical(manifest(f1)))
    h["X-Content-Digest"] = digest_of(manifest(f2))
    r = call(env.app, "POST", SUBMIT, ip=m.ip, body=body2, headers=h)
    assert r.status == 400 and r.code == "bad_signature", r
    assert env.minter.minted == []
