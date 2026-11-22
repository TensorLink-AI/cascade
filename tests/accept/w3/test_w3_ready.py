"""W3: ``POST /v1/lane-d/ready`` (plan §1.2 "Flow" step 3, A8 coldkey at
ready, B10 TTL, B13 licence attestation). Contract: README.md "Ready"."""

from __future__ import annotations

import hashlib
import json

import pytest

from .helpers import (
    ALLOWED_FILES,
    BLOCK,
    CONTEXT,
    READY,
    TTL,
    Env,
    Miner,
    call,
    checkpoint,
    digest_of,
    manifest,
    ss58,
    weights_sha,
)


def test_ready_records_the_entry(tmp_path):
    from cascade.lane_d import service

    env = Env(tmp_path)
    m = Miner(env, "rdy-a", coldkey="ck-a")
    files = checkpoint("a")
    dig = digest_of(manifest(files))
    r = m.enter(files, licence="OpenMDW-1.1")
    assert r.status == 200, r
    rec = r.json
    assert rec["status"] == "ready"
    assert rec["hotkey"] == m.hotkey and rec["digest"] == dig and rec["context"] == CONTEXT
    assert rec["coldkey_at_ready"] == ss58("ck-a")
    assert rec["ready_block"] == BLOCK
    assert rec["ready_expires_block"] == BLOCK + TTL
    assert rec["weights_sha256"] == weights_sha(files)
    assert rec["licence"] == "OpenMDW-1.1"
    want_att = hashlib.sha256(service.LICENCE_ATTESTATION.encode("utf-8")).hexdigest()
    assert rec["licence_attestation_sha256"] == want_att
    payload = rec["commit_payload"]
    assert payload.startswith(f"metro-v1:weights:c{CONTEXT}:hippius:"), payload
    assert dig in payload
    # the minter finalised the upload once, for the four files
    prefix = f"lane-d/{m.hotkey}/{dig}/"
    assert env.minter.finalized == [(prefix, {n: len(files[n]) for n in ALLOWED_FILES})]
    row = env.entry(m.hotkey, dig)
    for k in ("coldkey_at_ready", "ready_block", "ready_expires_block", "licence",
              "licence_attestation_sha256", "weights_sha256", "context"):
        assert row[k] == rec[k], k
    assert row["state"] == "ready"


def test_coldkey_is_read_from_the_chain_at_ready_not_at_submit(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-b", coldkey="ck-old")
    files = checkpoint("b")
    assert m.submit(files).status == 201
    m.upload(files)
    env.reads.coldkeys[m.hotkey] = ss58("ck-new")
    env.reads.block = BLOCK + 7
    r = m.ready(files)
    assert r.status == 200, r
    assert r.json["coldkey_at_ready"] == ss58("ck-new")
    assert r.json["ready_block"] == BLOCK + 7


def test_hotkey_deregistered_before_ready_is_refused(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-c")
    files = checkpoint("c")
    assert m.submit(files).status == 201
    m.upload(files)
    del env.reads.coldkeys[m.hotkey]
    r = m.ready(files)
    assert r.status == 403 and r.code == "not_registered", r


def test_repeated_ready_never_extends_the_ttl(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-d")
    files = checkpoint("d")
    first = m.enter(files)
    assert first.status == 200, first
    env.reads.block = BLOCK + 50
    again = m.ready(files)
    assert again.status == 200, again
    assert again.json["ready_block"] == BLOCK
    assert again.json["ready_expires_block"] == BLOCK + TTL
    assert again.json["commit_payload"] == first.json["commit_payload"]


@pytest.mark.parametrize("case,code", [
    ("missing_object", "missing_object"),
    ("size_mismatch", "size_mismatch"),
    ("stray_object", "stray_object"),
])
def test_object_list_is_checked_without_downloading(tmp_path, case, code):
    env = Env(tmp_path)
    m = Miner(env, f"rdy-{case}")
    files = checkpoint(case)
    dig = digest_of(manifest(files))
    prefix = f"lane-d/{m.hotkey}/{dig}/"
    assert m.submit(files).status == 201
    m.upload(files)
    if case == "missing_object":
        (env.tmp / "vault" / prefix / "model.py").unlink()
    elif case == "size_mismatch":
        env.store.put(prefix + "weights.safetensors", files["weights.safetensors"] + b"!")
    else:
        env.store.put(prefix + "train.pkl", b"\x80\x04N.")
    reads = []
    real_get = env.store.get

    def spy_get(k):
        reads.append(k)
        return real_get(k)

    env.store.get = spy_get
    r = m.ready(files)
    assert r.status == 409 and r.code == code, r
    assert not any(k.endswith("weights.safetensors") for k in reads), reads
    assert env.entry(m.hotkey, dig)["state"] == "submitted"


def test_ready_never_downloads_the_weights(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-e")
    files = checkpoint("e")
    assert m.submit(files).status == 201
    m.upload(files)
    reads = []
    real_get = env.store.get

    def spy_get(k):
        reads.append(k)
        return real_get(k)

    env.store.get = spy_get
    assert m.ready(files).status == 200
    assert not any(not k.endswith("manifest.json") for k in reads), reads


def test_ready_for_a_never_submitted_digest_is_unknown_entry(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-f")
    files = checkpoint("f")
    m.upload(files)
    r = m.ready(files)
    assert r.status == 404 and r.code == "unknown_entry", r


def test_ready_context_must_match_the_submit(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "rdy-g")
    files = checkpoint("g")
    assert m.submit(files).status == 201
    m.upload(files)
    r = m.ready(files, context=8192)
    assert r.status == 400 and r.code == "context_mismatch", r


# ── B13 licence attestation ──────────────────────────────────────────────────


def test_licence_constants_follow_d13():
    from cascade.lane_d import service

    assert tuple(service.LICENCES) == ("Apache-2.0", "OpenMDW-1.1")
    text = service.LICENCE_ATTESTATION
    assert isinstance(text, str) and len(text) >= 80
    for needle in ("Apache-2.0", "OpenMDW-1.1", "training data", "teacher"):
        assert needle in text, needle


@pytest.mark.parametrize("licence", ["MIT", "CC-BY-NC-4.0", "", "apache-2.0"])
def test_a_non_open_licence_is_refused(tmp_path, licence):
    env = Env(tmp_path)
    m = Miner(env, "lic-a")
    files = checkpoint("la")
    assert m.submit(files).status == 201
    m.upload(files)
    r = m.ready(files, licence=licence)
    assert r.status == 400 and r.code == "licence_not_open", r
    assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "submitted"


@pytest.mark.parametrize("attestation", ["", "I agree", "x" * 200])
def test_the_attestation_must_be_the_exact_statement(tmp_path, attestation):
    env = Env(tmp_path)
    m = Miner(env, "lic-b")
    files = checkpoint("lb")
    assert m.submit(files).status == 201
    m.upload(files)
    r = m.ready(files, attestation=attestation)
    assert r.status == 400 and r.code == "licence_attestation_missing", r


@pytest.mark.parametrize("body", [b"", b"{not json", b"[]", b'{"licence": 5}'])
def test_unparseable_ready_body_is_typed(tmp_path, body):
    env = Env(tmp_path)
    m = Miner(env, "lic-c")
    files = checkpoint("lc")
    assert m.submit(files).status == 201
    m.upload(files)
    r = m.ready(files, body=body)
    assert r.status == 400 and r.code in ("bad_body", "licence_not_open"), r


def test_licence_is_recorded_and_survives_restart(tmp_path):
    env = Env(tmp_path)
    m = Miner(env, "lic-d")
    files = checkpoint("ld")
    assert m.enter(files).status == 200
    env.restart()
    row = env.entry(m.hotkey, digest_of(manifest(files)))
    assert row["licence"] == "Apache-2.0"
    assert len(row["licence_attestation_sha256"]) == 64


def test_ready_route_needs_post(tmp_path):
    env = Env(tmp_path)
    r = call(env.app, "GET", READY)
    assert r.status == 405 and r.code == "method_not_allowed", r
    r = call(env.app, "POST", "/v1/lane-d/elsewhere", body=json.dumps({}).encode())
    assert r.status == 404 and r.code == "not_found", r
