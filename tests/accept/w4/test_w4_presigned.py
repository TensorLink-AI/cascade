"""W4 presigned fetch (plan §1.2: the eval pod gets presigned GETs only,
never a credential): ``cascade.lane_d.storage.fetch_presigned`` against a
localhost presigned-URL server (the only mocked service, the vault bucket)."""

from __future__ import annotations

import hashlib
import os
import time

import pytest

from .helpers import CREDENTIAL_ENV, PROXY_ENV, VaultServer

KEY = "lane-d/w4-hk/0123abcd/weights.safetensors"


@pytest.fixture
def vault(monkeypatch):
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = VaultServer()
    yield server
    server.close()


def _leftovers(directory) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_fetch_presigned_needs_and_sends_no_credentials(tmp_path, vault, monkeypatch):
    from cascade.lane_d.storage import fetch_presigned

    data = os.urandom(300_000)
    vault.objects[KEY] = data
    sha = hashlib.sha256(data).hexdigest()
    # 1) no credential anywhere in the process: the URL alone suffices.
    for name in list(os.environ):
        if name.startswith(("HIPPIUS_", "AWS_")) or name in CREDENTIAL_ENV:
            monkeypatch.delenv(name, raising=False)
    out = fetch_presigned(vault.url(KEY), tmp_path / "a" / "w", sha256=sha, size=len(data))
    assert (tmp_path / "a" / "w").read_bytes() == data and str(out).endswith("w")
    # 2) sentinel credentials present (as on a misconfigured host): none of
    #    them is ever sent, and no Authorization header is built.
    sentinels = {name: f"w4-sentinel-{i}-{name.lower()}" for i, name in enumerate(CREDENTIAL_ENV)}
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)
    fetch_presigned(vault.url(KEY), tmp_path / "b" / "w", sha256=sha, size=len(data))
    assert (tmp_path / "b" / "w").read_bytes() == data
    assert len(vault.requests) == 2
    for req in vault.requests:
        assert "authorization" not in req["headers"], req["headers"]
        assert "x-amz-security-token" not in req["headers"]
        blob = req["path"] + repr(sorted(req["headers"].items()))
        for value in sentinels.values():
            assert value not in blob


def test_fetch_presigned_sha256_mismatch_refused(tmp_path, vault):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    data = os.urandom(50_000)
    vault.objects[KEY] = data
    wrong = hashlib.sha256(data + b"x").hexdigest()
    with pytest.raises(FetchError) as e:
        fetch_presigned(vault.url(KEY), tmp_path / "d" / "w", sha256=wrong, size=len(data))
    assert e.value.code == "digest_mismatch"
    assert _leftovers(tmp_path / "d") == []


def test_fetch_presigned_short_body_refused(tmp_path, vault):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    data = os.urandom(50_000)
    vault.objects[KEY] = data[:-100]
    with pytest.raises(FetchError) as e:
        fetch_presigned(vault.url(KEY), tmp_path / "d" / "w",
                        sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    assert e.value.code == "size_mismatch"
    assert _leftovers(tmp_path / "d") == []


def test_fetch_presigned_oversize_declared_length_refused_before_download(tmp_path, vault):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    total = 64 << 20
    vault.declared[KEY] = total                 # honest Content-Length, far over budget
    t0 = time.monotonic()
    with pytest.raises(FetchError) as e:
        fetch_presigned(vault.url(KEY), tmp_path / "d" / "w",
                        sha256=hashlib.sha256(b"").hexdigest(), size=4096)
    assert e.value.code == "oversize"
    assert time.monotonic() - t0 < 30
    time.sleep(0.5)
    assert vault.sent.get(KEY, 0) < total // 2, "the oversize body was downloaded"
    assert _leftovers(tmp_path / "d") == []


def test_fetch_presigned_unbounded_stream_stops_at_size(tmp_path, vault):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    cap = 256 << 20
    vault.endless[KEY] = cap                    # no Content-Length, close-delimited
    t0 = time.monotonic()
    with pytest.raises(FetchError) as e:
        fetch_presigned(vault.url(KEY), tmp_path / "d" / "w",
                        sha256=hashlib.sha256(b"").hexdigest(), size=1 << 20)
    assert e.value.code == "oversize"
    assert time.monotonic() - t0 < 60
    time.sleep(0.5)
    assert vault.sent.get(KEY, 0) < (1 << 20) + (48 << 20), "kept reading past the size"
    assert _leftovers(tmp_path / "d") == []


def test_fetch_presigned_rejected_signature_fails_closed(tmp_path, vault):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    data = os.urandom(1000)
    vault.objects[KEY] = data
    with pytest.raises(FetchError) as e:
        fetch_presigned(vault.url(KEY, valid=False), tmp_path / "d" / "w",
                        sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    assert e.value.code == "fetch_failed"
    assert _leftovers(tmp_path / "d") == []


@pytest.mark.parametrize("url", ["file:///etc/hostname", "ftp://127.0.0.1/x", "/etc/hostname"])
def test_fetch_presigned_refuses_non_http_urls(tmp_path, url):
    from cascade.lane_d.storage import FetchError, fetch_presigned

    with pytest.raises(FetchError) as e:
        fetch_presigned(url, tmp_path / "d" / "w", sha256=hashlib.sha256(b"").hexdigest(),
                        size=64)
    assert e.value.code == "fetch_failed"
    assert _leftovers(tmp_path / "d") == []
