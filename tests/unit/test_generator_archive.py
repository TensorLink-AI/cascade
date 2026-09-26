"""Resolve-time generator audit archive (OPSLOG 2026-08-25).

Miners delete repos post-round; the trainer tars each fetched generator tree
to the manifest bucket at the dedup screen — the one guaranteed moment the
code exists. The archive must be idempotent, content-keyed by ref, and NEVER
able to disturb the round it records.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from cascade.shared.hippius import GENERATOR_ARCHIVE_PREFIX, generator_archive_key


def test_key_is_prefixed_and_filesystem_safe() -> None:
    k = generator_archive_key("makesomething/c@sha256:16d5440dabc")
    assert k.startswith(GENERATOR_ARCHIVE_PREFIX)
    assert k.endswith(".tar")
    assert "/" not in k[len(GENERATOR_ARCHIVE_PREFIX):]
    assert ":" not in k and "@" not in k


def test_key_distinguishes_hf_and_hub_refs() -> None:
    a = generator_archive_key("robert131OO4/cascade-vanta@hf:e40e3826")
    b = generator_archive_key("robert131OO4/cascade-vanta@hf:8f12c865")
    c = generator_archive_key("robert131OO4/cascade-vanta@sha256:e40e3826")
    assert len({a, b, c}) == 3


def test_key_bounds_length() -> None:
    k = generator_archive_key("x/" + "y" * 500 + "@sha256:" + "f" * 64)
    assert len(k) <= len(GENERATOR_ARCHIVE_PREFIX) + 204


class _StubStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts = 0

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]

    def put_bytes(self, key: str, data: bytes, **kw) -> None:
        self.puts += 1
        self.objects[key] = data


class _FailingStore:
    def get_bytes(self, key: str) -> bytes:
        raise KeyError(key)

    def put_bytes(self, key: str, data: bytes, **kw) -> None:
        raise OSError("bucket down")


def _loop_with_store(store):
    from cascade.trainer.loop import TrainerRunner
    loop = TrainerRunner.__new__(TrainerRunner)
    loop._manifest_store = store  # manifest_store() returns the cached attr
    return loop


def _make_tree(tmp_path: Path) -> Path:
    d = tmp_path / "gen"
    d.mkdir()
    (d / "generator.py").write_text("class Generator: ...\n")
    (d / "config.json").write_text("{}")
    return d


def test_archives_tree_as_readable_tar(tmp_path) -> None:
    store = _StubStore()
    loop = _loop_with_store(store)
    d = _make_tree(tmp_path)
    ref = "someone/gen@sha256:abc123"
    loop._archive_generator_tree(ref, d)
    key = generator_archive_key(ref)
    assert key in store.objects
    with tarfile.open(fileobj=io.BytesIO(store.objects[key])) as tar:
        names = {m.name.lstrip("./") for m in tar.getmembers()}
    assert "generator.py" in names and "config.json" in names


def test_skip_if_already_archived(tmp_path) -> None:
    store = _StubStore()
    loop = _loop_with_store(store)
    d = _make_tree(tmp_path)
    ref = "someone/gen@sha256:abc123"
    loop._archive_generator_tree(ref, d)
    loop._archive_generator_tree(ref, d)
    assert store.puts == 1


def test_store_failure_never_raises(tmp_path) -> None:
    loop = _loop_with_store(_FailingStore())
    d = _make_tree(tmp_path)
    loop._archive_generator_tree("someone/gen@sha256:abc123", d)  # must not raise


# ── private (vault/direct) submissions: operator-only archive, never the shared bucket ──

def _loop_with_stores(shared, private):
    loop = _loop_with_store(shared)
    loop._private_archive = private if private is not None else False
    return loop


def test_private_submission_archives_to_private_store_only(tmp_path) -> None:
    from cascade.shared.hippius import PRIVATE_ARCHIVE_PREFIX, private_archive_key

    shared, private = _StubStore(), _StubStore()
    loop = _loop_with_stores(shared, private)
    d = _make_tree(tmp_path)
    digest = "a" * 64
    loop._archive_generator_tree("vault/direct@sha256:" + digest, d)
    assert shared.puts == 0 and shared.objects == {}
    key = private_archive_key(digest)
    assert key.startswith(PRIVATE_ARCHIVE_PREFIX) and set(private.objects) == {key}
    with tarfile.open(fileobj=io.BytesIO(private.objects[key])) as tar:
        names = {m.name.lstrip("./") for m in tar.getmembers()}
    assert "generator.py" in names and "config.json" in names
    loop._archive_generator_tree("vault/direct@sha256:" + digest, d)
    assert private.puts == 1  # idempotent, like the public archive


def test_private_submission_without_private_store_touches_nothing(tmp_path) -> None:
    shared = _StubStore()
    loop = _loop_with_stores(shared, None)
    d = _make_tree(tmp_path)
    loop._archive_generator_tree("vault/direct@sha256:" + "b" * 64, d)
    assert shared.puts == 0 and shared.objects == {}


def test_public_ref_never_reaches_the_private_store(tmp_path) -> None:
    shared, private = _StubStore(), _StubStore()
    loop = _loop_with_stores(shared, private)
    d = _make_tree(tmp_path)
    loop._archive_generator_tree("someone/gen@sha256:abc123", d)
    assert private.puts == 0
    assert set(shared.objects) == {generator_archive_key("someone/gen@sha256:abc123")}


def test_private_store_resolution_failure_is_remembered(monkeypatch) -> None:
    from cascade.trainer.loop import TrainerRunner

    loop = TrainerRunner.__new__(TrainerRunner)
    loop.cfg = type("C", (), {"storage": object()})()  # no endpoint ⇒ config raises
    assert loop._private_archive_store() is None
    assert loop._private_archive is False
    assert loop._private_archive_store() is None
