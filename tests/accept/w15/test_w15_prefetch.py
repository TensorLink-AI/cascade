"""W15 prefetch on reveal through W4's presigned ingest (plan §1.2 step 5:
"prefetches the entry into a sha-keyed, resumable cache, using presigned
GETs"; §10 "Data paths" 2). ``prefetch_grant`` fills the cache through
``cascade.eval.lane_d_ingest.ingest_presigned``, so every W4 guard runs before
an entry is admitted.

The cases here never reach W4's step 8 (the only torch step): a cache hit
makes no request at all, and the refusals stop at steps 1-5. The full
miss -> ingest -> admit path needs torch and lives in test_w15_torch.py."""

from __future__ import annotations

import pytest

from .helpers import (
    PROXY_ENV,
    Entry,
    Vault,
    digest_of,
    fixture_checkpoint,
    fixture_expectation,
    read_tree,
)


@pytest.fixture
def vault(monkeypatch):
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = Vault()
    yield server
    server.close()


def _cache(root, budget=10 ** 7, **kw):
    from cascade.eval.lane_d_cache import PrefetchCache

    return PrefetchCache(root, budget_bytes=budget, **kw)


class _Seed:
    """Fills the cache with given bytes, as an earlier prefetch would have."""

    def __init__(self, files):
        self.files = files

    def fill(self, directory):
        for name, data in self.files.items():
            (directory / name).write_bytes(data)


def test_prefetch_hit_makes_no_request(tmp_path, vault):
    from cascade.eval.lane_d_cache import prefetch_grant

    files = fixture_checkpoint()
    grant = vault.publish("w15-hk", files)
    assert grant["digest"] == digest_of(files)
    c = _cache(tmp_path / "c")
    c.put(grant["digest"], _Seed(files).fill)
    out = prefetch_grant(c, grant, expect=fixture_expectation())
    assert out == tmp_path / "c" / grant["digest"]
    assert read_tree(out) == files
    assert vault.requests == [], "a verified cache hit never goes to the vault"


@pytest.mark.parametrize("case", ["extra_file", "object_contradicts_manifest", "edited_model_py"])
def test_prefetch_refused_by_ingest_admits_nothing(tmp_path, vault, case):
    from cascade.eval.lane_d_cache import prefetch_grant
    # isort: split
    from cascade.eval.lane_d_ingest import IngestError

    files = fixture_checkpoint()
    if case == "extra_file":
        bad = {**files, "evil.py": b"import os\nos.mkdir('/tmp/w15-never')\n"}
        grant, code = vault.publish("w15-hk", bad), "file_not_allowed"
    elif case == "object_contradicts_manifest":
        served = dict(files)
        served["weights.safetensors"] = b"\0" + files["weights.safetensors"][1:]
        grant = vault.publish("w15-hk", served, manifest_files=files)
        code = "digest_mismatch"
    else:
        bad = dict(files)
        bad["model.py"] = b"import os\n" + files["model.py"]
        grant, code = vault.publish("w15-hk", bad), "code_mismatch"
    keep = Entry(1)
    c = _cache(tmp_path / "c")
    c.put(keep.digest, keep.fill)
    with pytest.raises(IngestError) as err:
        prefetch_grant(c, grant, expect=fixture_expectation())
    assert err.value.code.value == code
    assert c.entries() == [keep.digest] and c.total_bytes() == keep.size
    assert sorted(p.name for p in (tmp_path / "c").iterdir() if p.is_dir()) == [keep.digest], \
        "a refused prefetch leaves no entry and no staging directory"
    assert c.get(grant["digest"]) is None


def test_prefetch_refuses_a_grant_with_a_bad_digest_before_any_request(tmp_path, vault):
    from cascade.eval.lane_d_cache import CacheError, prefetch_grant

    grant = vault.publish("w15-hk", fixture_checkpoint())
    grant = {**grant, "digest": "../" + grant["digest"][3:]}
    c = _cache(tmp_path / "c")
    with pytest.raises(CacheError) as err:
        prefetch_grant(c, grant, expect=fixture_expectation())
    assert err.value.code == "bad_digest"
    assert vault.requests == [] and c.entries() == []
