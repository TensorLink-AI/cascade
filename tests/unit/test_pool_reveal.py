"""Exact-bytes reveal — retiree selection, digest verification, idempotence.

The reveal's whole contract is "public copy == scored tar": these tests pin the
safety guard (never reveal the newest two real snapshots), the sentinel filter
(smoke-test publishes can't consume a guard slot or leak), and the hard failure
on a bucket/index digest mismatch (an integrity break must never be published).
"""

from __future__ import annotations

import io
import sys
import tarfile
import types

import pytest

from cascade.pool.reveal import (
    SHA_MARKER,
    reveal_dir_name,
    reveal_snapshots,
    select_reveals,
)
from cascade.shared.hippius import PoolSnapshotMeta, StorageError, tar_cid_digest


def _meta(block: int, sha: str = "", as_of: str = "2026-07-31") -> PoolSnapshotMeta:
    return PoolSnapshotMeta(
        effective_block=block,
        key=f"pool/snapshots/block-{block}.tar",
        sha256=sha or f"sha-{block}",
        size_bytes=1,
        as_of=as_of,
        n_series=3,
        context_length=4096,
        horizon=64,
    )


# ── select_reveals: the safety guard ─────────────────────────────────────────


def test_select_withholds_newest_two_and_orders_oldest_first():
    index = [_meta(b) for b in (300, 100, 200, 400)]
    picked = select_reveals(index)
    assert [m.effective_block for m in picked] == [100, 200]


def test_select_sentinels_neither_revealed_nor_counted():
    # A smoke-test sentinel sorts as "newest"; if it consumed a guard slot the
    # actually-active snapshot would be revealed. It must be invisible entirely.
    index = [_meta(999_999_999_999), _meta(300), _meta(200), _meta(100)]
    picked = select_reveals(index)
    assert [m.effective_block for m in picked] == [100]


def test_select_small_and_empty_indexes_reveal_nothing():
    assert select_reveals([]) == []
    assert select_reveals([_meta(100)]) == []
    assert select_reveals([_meta(100), _meta(200)]) == []
    assert select_reveals([_meta(999_999_999_999)]) == []


def test_select_keep_newest_zero_reveals_everything_real():
    index = [_meta(200), _meta(999_999_999_999), _meta(100)]
    picked = select_reveals(index, keep_newest=0)
    assert [m.effective_block for m in picked] == [100, 200]


# ── reveal_snapshots: IO paths with a fake store + stubbed HfApi ─────────────


def _tar_bytes(names_to_payload: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in names_to_payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _FakeStore:
    def __init__(self, index_doc: str, tars: dict[str, bytes]):
        self._index = index_doc
        self._tars = tars

    def get_text(self, key: str) -> str:
        assert key == "pool/index.json"
        return self._index

    def get_bytes(self, key: str) -> bytes:
        if key not in self._tars:
            raise StorageError(f"missing {key}")
        return self._tars[key]


class _FakeApi:
    """Stands in for HfApi: records uploads, reports pre-existing markers."""

    def __init__(self, token=None, existing: set[str] | None = None):
        self.existing = existing if existing is not None else set()
        self.uploaded_dirs: list[str] = []
        self.uploaded_files: dict[str, list[str]] = {}
        self.fail_next = 0
        self.card_updates = 0

    def create_repo(self, **kw):
        pass

    def file_exists(self, repo_id, path, repo_type):
        return path in self.existing

    def upload_folder(self, *, folder_path, path_in_repo, **kw):
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("AccessDenied: Request has expired")
        from pathlib import Path

        self.uploaded_dirs.append(path_in_repo)
        self.uploaded_files[path_in_repo] = sorted(
            p.name for p in Path(folder_path).iterdir() if p.is_file())

    def upload_file(self, *, path_in_repo, **kw):
        assert path_in_repo == "README.md"
        self.card_updates += 1


@pytest.fixture()
def fake_hub(monkeypatch):
    api = _FakeApi()
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = lambda token=None: api
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return api


def _index_doc(metas: list[PoolSnapshotMeta]) -> str:
    import dataclasses
    import json

    return json.dumps(
        {"schema": 2, "snapshots": [dataclasses.asdict(m) for m in metas]}
    )


def test_reveal_uploads_retired_verified_tars(tmp_path, fake_hub):
    tar = _tar_bytes({"a.npy": b"aa", "metadata.json": b"{}"})
    metas = [_meta(100, sha=tar_cid_digest(tar)), _meta(200), _meta(300)]
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})

    n = reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path)
    assert n == 1
    assert fake_hub.uploaded_dirs == [reveal_dir_name(metas[0])]
    assert fake_hub.card_updates == 1
    stage = tmp_path / "block-100"
    assert (stage / "block-100.tar").read_bytes() == tar
    assert (stage / SHA_MARKER).read_text().strip() == tar_cid_digest(tar)
    # ONE commit of the exact tar + marker + small sidecars, never the per-series
    # .npy (thousands of LFS uploads outlived HF's 900 s presigned URLs).
    assert fake_hub.uploaded_files[reveal_dir_name(metas[0])] == [
        SHA_MARKER, "block-100.tar", "metadata.json"]
    assert not (stage / "a.npy").exists()


def test_reveal_digest_mismatch_is_fatal(tmp_path, fake_hub):
    tar = _tar_bytes({"a.npy": b"aa"})
    metas = [_meta(100, sha="0" * 64), _meta(200), _meta(300)]
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})

    with pytest.raises(StorageError, match="reveal_digest_mismatch"):
        reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path)
    assert fake_hub.uploaded_dirs == []  # nothing published on integrity failure


def test_reveal_skips_already_revealed_and_pruned(tmp_path, fake_hub):
    tar = _tar_bytes({"a.npy": b"aa"})
    metas = [
        _meta(100, sha=tar_cid_digest(tar)),  # already on HF -> skip
        _meta(150),                            # tar pruned from bucket -> skip
        _meta(200),
        _meta(300),
    ]
    fake_hub.existing.add(f"{reveal_dir_name(metas[0])}/{SHA_MARKER}")
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})

    n = reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path)
    assert n == 0
    assert fake_hub.uploaded_dirs == []
    assert fake_hub.card_updates == 1  # card still refreshed to newest-done


# ── one reveal per day (owner 2026-11-24, option b) ──────────────────────────


def _index_of(blocks, tars):
    metas = [_meta(b, sha=tar_cid_digest(tars[b]) if b in tars else "") for b in blocks]
    store = _FakeStore(_index_doc(metas),
                       {f"pool/snapshots/block-{b}.tar": t for b, t in tars.items()})
    return metas, store


def test_newest_only_reveals_the_newest_retired_and_never_the_newest_three(tmp_path, fake_hub):
    tars = {b: _tar_bytes({f"s{b}.npy": b"x", "metadata.json": b"{}"})
            for b in (100, 200, 300, 400, 500, 600)}
    metas, store = _index_of([100, 200, 300, 400, 500, 600], tars)

    n = reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path,
                         keep_newest=3, max_reveal=1, newest_only=True)
    # 400/500/600 = active + two pending: withheld. 300 is the newest retiree;
    # 100 and 200 stay in the bucket only.
    assert n == 1
    assert fake_hub.uploaded_dirs == [reveal_dir_name(metas[2])]


def test_newest_only_never_backfills_older_retirees(tmp_path, fake_hub):
    tars = {b: _tar_bytes({"a.npy": b"x"}) for b in (100, 200, 300, 400, 500)}
    metas, store = _index_of([100, 200, 300, 400, 500], tars)
    fake_hub.existing.add(f"{reveal_dir_name(metas[1])}/{SHA_MARKER}")  # 200 already on HF

    n = reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path,
                         keep_newest=3, max_reveal=1, newest_only=True)
    assert n == 0
    assert fake_hub.uploaded_dirs == []  # block 100 is NOT backfilled


def test_upload_retries_then_marker_rides_the_same_commit(tmp_path, fake_hub, monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    tar = _tar_bytes({"a.npy": b"aa", "provenance.json": b"{}"})
    metas = [_meta(100, sha=tar_cid_digest(tar)), _meta(200), _meta(300)]
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})
    fake_hub.fail_next = 1  # HF presigned URL expired on the first attempt

    assert reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path) == 1
    files = fake_hub.uploaded_files[reveal_dir_name(metas[0])]
    assert files == [SHA_MARKER, "block-100.tar", "provenance.json"]


def test_upload_failure_after_retries_is_loud_and_leaves_no_marker(tmp_path, fake_hub,
                                                                   monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    tar = _tar_bytes({"a.npy": b"aa"})
    metas = [_meta(100, sha=tar_cid_digest(tar)), _meta(200), _meta(300)]
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})
    fake_hub.fail_next = 99

    with pytest.raises(RuntimeError, match="expired"):
        reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path)
    assert fake_hub.uploaded_dirs == []  # no marker on HF => next run retries it


def test_revealed_tar_is_what_the_miner_verifier_accepts(tmp_path, fake_hub):
    """The public reader (cascade.miner.hf_pool) checks block-<N>.tar against
    POOL_SHA256 and extracts the tar — exactly the two files staged here."""
    import hashlib

    tar = _tar_bytes({"a.npy": b"aa", "metadata.json": b"{}"})
    metas = [_meta(100, sha=tar_cid_digest(tar)), _meta(200), _meta(300)]
    store = _FakeStore(_index_doc(metas), {"pool/snapshots/block-100.tar": tar})
    reveal_snapshots(store, "org/repo", hf_token="t", work_dir=tmp_path)
    stage = tmp_path / "block-100"
    want = (stage / SHA_MARKER).read_text().split()[0]
    assert hashlib.sha256((stage / "block-100.tar").read_bytes()).hexdigest() == want
