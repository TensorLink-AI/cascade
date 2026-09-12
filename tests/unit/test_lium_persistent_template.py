"""Lium: one persistent template per image, rented by id (2026-09-12).

Lium caps ``POST /templates`` at 20 requests / hour per client IP and
``lium up --image`` creates a throw-away template on every call, so a round's
launches silently exhausted the hour (no pod, "Rate limit exceeded" only in a
buffered log). The provider now resolves a content-addressed persistent
template once and passes ``--template_id``.
"""
from __future__ import annotations

import types

import pytest

from cascade.provision.core import (
    LaunchSpec,
    LiumProvider,
    ProvisionError,
    lium_get_or_create_template,
    lium_template_name,
    lium_template_payload,
)

DIGEST = "sha256:" + "9d" * 32
IMG = f"ghcr.io/tensorlink-ai/cascade-worker:worker-v0.8.0@{DIGEST}"
PUB = "ssh-ed25519 AAAAkey cascade-orchestrator"


def test_payload_splits_repo_and_tag_and_pins_the_digest_in_env():
    p = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    assert p["docker_image"] == "ghcr.io/tensorlink-ai/cascade-worker"
    assert p["docker_image_tag"] == "worker-v0.8.0"
    # The digest pins the PULL (2026-09-12: tag-only templates let a host serve
    # a stale image under the pinned tag); the env is the gate's byte-compare.
    assert p["docker_image_digest"] == DIGEST
    assert p["environment"] == {"SSH_PUBKEY": PUB, "CASCADE_TRAIN_IMAGE_DIGEST": DIGEST}
    assert p["internal_ports"] == [22, 2222]
    assert p["one_time_template"] is False and p["is_private"] is True
    assert p["name"].startswith("cascade-worker-worker-v0.8.0-")


def test_payload_handles_registry_port_and_missing_tag():
    p = lium_template_payload("localhost:5000/worker@" + DIGEST, ssh_pubkey=PUB, ssh_port=22)
    assert (p["docker_image"], p["docker_image_tag"]) == ("localhost:5000/worker", "latest")
    assert p["internal_ports"] == [22]


def test_template_name_is_content_addressed():
    a = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    b = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    c = lium_template_payload(IMG, ssh_pubkey=PUB + "x", ssh_port=2222)
    assert lium_template_name(a) == lium_template_name(b) == a["name"]
    assert lium_template_name(c) != a["name"]               # a different pubkey ⇒ new template


class _Resp:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.headers = headers or {}
        self.text = text or (str(body) if body is not None else "")

    def json(self):
        return self._body


def _fake_requests(monkeypatch, *, listing, post=None):
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(("GET", url, headers.get("X-API-KEY")))
        return _Resp(200, listing)

    def post_fn(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url, json["name"]))
        return post

    fake = types.SimpleNamespace(get=get, post=post_fn, RequestException=Exception)
    monkeypatch.setitem(__import__("sys").modules, "requests", fake)
    return calls


def test_get_or_create_reuses_an_existing_template_by_name(monkeypatch):
    p = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    calls = _fake_requests(monkeypatch, listing=[
        {"id": "other", "name": "x", "docker_image": p["docker_image"], "docker_image_tag": "v0"},
        {"id": "T1", "name": p["name"], "docker_image": p["docker_image"],
         "docker_image_tag": p["docker_image_tag"]},
    ])
    assert lium_get_or_create_template(p, api_key="sk_x") == "T1"
    assert [c[0] for c in calls] == ["GET"]                  # no creation
    assert calls[0][2] == "sk_x"                            # the payer's/operator's key


def test_get_or_create_creates_once_when_absent(monkeypatch):
    p = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    calls = _fake_requests(monkeypatch, listing=[], post=_Resp(201, {"id": "T2"}))
    assert lium_get_or_create_template(p, api_key="sk_x") == "T2"
    assert [c[0] for c in calls] == ["GET", "POST"] and calls[1][2] == p["name"]


def test_get_or_create_surfaces_the_rate_limit(monkeypatch):
    p = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    _fake_requests(monkeypatch, listing=[],
                   post=_Resp(429, {"error": "rate_limited"}, headers={"retry-after": "1158"},
                              text="You can make 20 requests per 1 hour"))
    with pytest.raises(ProvisionError, match="rate-limited.*retry after 1158s"):
        lium_get_or_create_template(p, api_key="sk_x")


def test_get_or_create_needs_a_key():
    p = lium_template_payload(IMG, ssh_pubkey=PUB, ssh_port=2222)
    with pytest.raises(ProvisionError, match="LIUM_API_KEY"):
        lium_get_or_create_template(p, api_key="")


def test_provider_resolves_the_template_once_per_batch_and_rents_by_id():
    spawned, resolved = [], []

    def _run(argv):
        out = '[{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda a: spawned.append(a),
                        _template=lambda p: resolved.append(p["name"]) or "T9")
    names = prov.launch(LaunchSpec(sku="L40S", count=3, image=IMG, ssh_pubkey=PUB,
                                   ssh_port=2222, name_prefix="cascade-n91-r1-funded"))
    assert names == [f"cascade-n91-r1-funded-{i}" for i in range(3)]
    assert len(resolved) == 1                                # one lookup for three pods
    for argv in spawned:
        assert argv[argv.index("--template_id") + 1] == "T9"
        assert "--image" not in argv and "-e" not in argv    # nothing minted per pod


def test_provider_template_failure_aborts_the_launch_before_any_pod():
    spawned = []

    def _run(argv):
        out = '[{"id": "e1"}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    def _boom(payload):
        raise ProvisionError("lium: template creation rate-limited (20 requests/hour per client IP)")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda a: spawned.append(a), _template=_boom)
    with pytest.raises(ProvisionError, match="rate-limited"):
        prov.launch(LaunchSpec(sku="L40S", count=1, image=IMG, ssh_pubkey=PUB))
    assert spawned == []                                     # infra path, no half-launch
