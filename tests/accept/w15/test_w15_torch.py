"""W15 checks that need torch (marked ``torch``: CI only, never the 7 GB box):
the full prefetch path. A miss ingests through W4's ``ingest_presigned`` (step
8, the finite check, imports torch), admits the entry under its digest, and a
corrupt entry is refused on read and refetched from the vault."""

from __future__ import annotations

import pytest

from .helpers import PROXY_ENV, Entry, Vault, fixture_checkpoint, fixture_expectation, read_tree

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)


@pytest.fixture
def vault(monkeypatch):
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = Vault()
    yield server
    server.close()


def _cache(root, budget, **kw):
    from cascade.eval.lane_d_cache import PrefetchCache

    return PrefetchCache(root, budget_bytes=budget, **kw)


def test_prefetch_miss_ingests_admits_and_then_hits(tmp_path, vault):
    from cascade.eval.lane_d_cache import prefetch_grant

    files = fixture_checkpoint()
    grant = vault.publish("w15-hk", files)
    c = _cache(tmp_path / "c", 10 ** 7)
    out = prefetch_grant(c, grant, expect=fixture_expectation())
    assert out == tmp_path / "c" / grant["digest"]
    assert read_tree(out) == files
    assert c.entries() == [grant["digest"]]
    assert c.total_bytes() == sum(len(v) for v in files.values())
    n = len(vault.requests)
    assert n == 1 + len(files), "the manifest plus each allowlisted object, once"
    assert prefetch_grant(c, grant, expect=fixture_expectation()) == out
    assert len(vault.requests) == n, "the second prefetch is a hit"


def test_prefetch_refetches_a_corrupt_entry(tmp_path, vault):
    from cascade.eval.lane_d_cache import prefetch_grant

    files = fixture_checkpoint()
    grant = vault.publish("w15-hk", files)
    c = _cache(tmp_path / "c", 10 ** 7, king_d=grant["digest"])
    out = prefetch_grant(c, grant, expect=fixture_expectation())
    w = out / "weights.safetensors"
    data = bytearray(w.read_bytes())
    data[-1] ^= 0x01
    w.write_bytes(bytes(data))
    n = len(vault.requests)
    out2 = prefetch_grant(c, grant, expect=fixture_expectation())
    assert len(vault.requests) > n, "a corrupt entry is fetched again"
    assert read_tree(out2) == files


def test_prefetched_king_survives_lru_pressure(tmp_path, vault):
    from cascade.eval.lane_d_cache import prefetch_grant

    files = fixture_checkpoint()
    grant = vault.publish("w15-hk", files)
    size = sum(len(v) for v in files.values())
    c = _cache(tmp_path / "c", size + 2000, king_d=grant["digest"])
    prefetch_grant(c, grant, expect=fixture_expectation())
    for i in range(10, 20):
        x = Entry(i)
        c.put(x.digest, x.fill)
    assert grant["digest"] in c.entries()
    assert read_tree(c.get(grant["digest"])) == files
