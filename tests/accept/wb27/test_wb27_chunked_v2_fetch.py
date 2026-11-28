"""2026-11-28: a Hippius chunked-v2 checkpoint is unverifiable on the release.

The lane B rebased candidate c2787703… was pushed (from an eval pod whose venv
resolved a newer hippius_hub) as a chunked-v2 object: a JSON pointer layer
(``application/vnd.hippius.pointer.v2``) plus untitled pack layers. The pinned
hippius_hub 0.5.1 saved the 196 KB pointer AS ``weights.safetensors`` and the
fetch wrote ``.fetch_complete`` over it, so every validator failed
``checkpoint_tampered`` and the catch-up walk re-judged it each settlement.

Contract pinned here (fake registry, no network, torch-free):

* ``fetch_from_hub`` rebuilds a chunked-v2 file from its packs, verified
  against the pinned manifest digest → pointer digest → pack + chunk digests
  → whole-file size + sha256, before the marker;
* a tampered chunk / pack is refused (StorageError, no marker, no snapshot);
* a cached ``.fetch_complete`` over an index is incomplete: refetched, healed;
* every cascade Hub push runs with ``HIPPIUS_CHUNKED_WRITE=0`` (hippius_hub's
  own single-blob switch), so the release can read what we push.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import struct
import sys
import types
from pathlib import Path

import pytest

from tests.accept.wb27.helpers import torch_blocked

POINTER_MT = "application/vnd.hippius.pointer.v2"
PACK_MT = "application/vnd.hippius.pack.v1"
REPO = "cascade/ckpt-test-expert-u1"


def _h():
    with torch_blocked():
        return importlib.import_module("cascade.shared.hippius")


def _sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _safetensors_bytes() -> bytes:
    """A minimal, valid safetensors file (one f32 tensor), built by hand."""
    data = struct.pack("<16f", *range(16)) * 64
    header = json.dumps({"w": {"dtype": "F32", "shape": [64, 16],
                               "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


def _chunked_object(file_bytes: bytes, *, lie_chunk: int | None = None):
    """(manifest bytes, pointer bytes, packs) the way hippius_hub >= 0.6 lays
    a file out: chunks in file order, spread over two packs, the second pack
    holding its chunks in reverse order (offsets are pack-relative)."""
    cuts = [0, 700, 1500, 2600, 3100, len(file_bytes)]
    pieces = [file_bytes[a:b] for a, b in itertools.pairwise(cuts)]
    groups = [[0, 1], [2, 3, 4]]
    packs, chunk_rows = {}, [None] * len(pieces)
    for g in groups:
        order = g if g[0] == 0 else list(reversed(g))
        blob, offs = b"", {}
        for i in order:
            offs[i] = len(blob)
            blob += pieces[i]
        pd = _sha(blob)
        packs[pd] = blob
        for i in g:
            chunk_rows[i] = {"digest": _sha(pieces[i]), "offset": offs[i],
                             "pack": pd, "size": len(pieces[i])}
    if lie_chunk is not None:
        chunk_rows[lie_chunk]["digest"] = _sha(b"not the bytes")
    pointer = json.dumps({"chunks": chunk_rows,
                          "file": {"digest": _sha(file_bytes), "size": len(file_bytes)},
                          "version": "chunked-v2"},
                         separators=(",", ":"), sort_keys=True).encode()
    small = {"config.json": b'{"d_model": 8}'}
    layers = [{"mediaType": "application/octet-stream", "size": len(v), "digest": _sha(v),
               "annotations": {"org.opencontainers.image.title": k}}
              for k, v in small.items()]
    layers.append({"mediaType": POINTER_MT, "size": len(pointer), "digest": _sha(pointer),
                   "annotations": {"org.opencontainers.image.title": "weights.safetensors",
                                   "com.hippius.file.size": str(len(file_bytes)),
                                   "com.hippius.file.digest": _sha(file_bytes),
                                   "com.hippius.chunk.count": str(len(pieces))}})
    layers += [{"mediaType": PACK_MT, "size": len(b), "digest": d} for d, b in packs.items()]
    manifest = json.dumps({"schemaVersion": 2,
                           "mediaType": "application/vnd.oci.image.manifest.v1+json",
                           "annotations": {"com.hippius.layout": "chunked-v2"},
                           "layers": layers}).encode()
    files = dict(small, **{"weights.safetensors": pointer})
    return manifest, files, packs


class FakeRegistry:
    def __init__(self, manifest: bytes, packs: dict, *, serve: dict | None = None):
        self.manifest, self.packs, self.serve = manifest, packs, serve or {}
        self.blob_gets: list[str] = []

    def manifest_bytes(self, digest):
        assert digest == _sha(self.manifest)
        return self.manifest

    def blob_to(self, digest, fh):
        self.blob_gets.append(digest)
        fh.write(self.serve.get(digest, self.packs[digest]))


def _install(monkeypatch, files: dict, registry: FakeRegistry | None):
    """A fake hippius_hub 0.5.1: snapshot_download writes the titled layers'
    blobs (the POINTER under the file's title, packs skipped)."""
    calls = {"snap": 0, "readers": 0}

    def snapshot_download(*, repo_id, revision, local_dir, **_):
        calls["snap"] += 1
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (d / name).write_bytes(data)

    mod = types.ModuleType("hippius_hub")
    mod.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "hippius_hub", mod)
    for k in ("HIPPIUS_HUB_TOKEN", "HIPPIUS_TOKEN", "CASCADE_HIPPIUS_TOKEN",
              "HIPPIUS_HUB_USERNAME", "HIPPIUS_HUB_PASSWORD"):
        monkeypatch.delenv(k, raising=False)

    def reader(ref):
        calls["readers"] += 1
        assert registry is not None, "no registry read expected for a plain snapshot"
        return registry

    monkeypatch.setattr(_h(), "_chunked_registry_reader", reader, raising=False)
    return calls


def test_a_chunked_v2_index_is_reassembled_and_verified(tmp_path, monkeypatch):
    h = _h()
    weights = _safetensors_bytes()
    manifest, files, packs = _chunked_object(weights)
    reg = FakeRegistry(manifest, packs)
    _install(monkeypatch, files, reg)
    ref = f"{REPO}@{_sha(manifest)}"
    dest = h.fetch_from_hub(ref, tmp_path / "snap")
    got = (dest / "weights.safetensors").read_bytes()
    assert got == weights, "weights.safetensors must be the reassembled file, not the index"
    (hlen,) = struct.unpack("<Q", got[:8])
    assert json.loads(got[8:8 + hlen])["w"]["shape"] == [64, 16]  # header in bounds
    assert (dest / "config.json").read_bytes() == files["config.json"]
    assert (dest / h.FETCH_COMPLETE_MARKER).is_file()
    assert sorted(reg.blob_gets) == sorted(packs), "each pack fetched once"
    assert not [p.name for p in dest.iterdir() if p.name.startswith(".") and
                p.name != h.FETCH_COMPLETE_MARKER], "no temp files left behind"


@pytest.mark.parametrize("how", ["lying_chunk_digest", "registry_serves_wrong_pack"])
def test_b_a_tampered_chunk_is_refused(tmp_path, monkeypatch, how):
    h = _h()
    weights = _safetensors_bytes()
    if how == "lying_chunk_digest":
        manifest, files, packs = _chunked_object(weights, lie_chunk=3)
        reg = FakeRegistry(manifest, packs)
    else:
        manifest, files, packs = _chunked_object(weights)
        pd = next(iter(packs))
        bad = bytearray(packs[pd])
        bad[10] ^= 0xFF
        reg = FakeRegistry(manifest, packs, serve={pd: bytes(bad)})
    _install(monkeypatch, files, reg)
    dest = tmp_path / "snap"
    with pytest.raises(h.StorageError) as ei:
        h.fetch_from_hub(f"{REPO}@{_sha(manifest)}", dest)
    assert "chunked_v2" in str(ei.value)
    assert not dest.exists(), "a refused reassembly leaves no snapshot (and no marker)"


def test_c_stale_fetch_complete_over_an_index_is_refetched(tmp_path, monkeypatch):
    h = _h()
    weights = _safetensors_bytes()
    manifest, files, packs = _chunked_object(weights)
    reg = FakeRegistry(manifest, packs)
    calls = _install(monkeypatch, files, reg)
    dest = tmp_path / "sha256-stale"
    dest.mkdir()                       # what a pre-fix validator cached
    for name, data in files.items():
        (dest / name).write_bytes(data)
    (dest / h.FETCH_COMPLETE_MARKER).touch()
    out = h.fetch_from_hub(f"{REPO}@{_sha(manifest)}", dest)
    assert calls["snap"] == 1, "a marker over a chunk index must not be reused"
    assert (out / "weights.safetensors").read_bytes() == weights
    assert (out / h.FETCH_COMPLETE_MARKER).is_file()
    # healed: the next fetch reuses it without touching the registry
    h.fetch_from_hub(f"{REPO}@{_sha(manifest)}", dest)
    assert calls["snap"] == 1


def test_d_plain_snapshot_never_reads_the_registry(tmp_path, monkeypatch):
    h = _h()
    files = {"config.json": b'{"version": "chunked-v2"}',   # JSON, not an index
             "weights.safetensors": _safetensors_bytes()}
    calls = _install(monkeypatch, files, None)
    dest = h.fetch_from_hub(f"{REPO}@sha256:{'b' * 64}", tmp_path / "plain")
    assert (dest / "weights.safetensors").read_bytes() == files["weights.safetensors"]
    assert calls["readers"] == 0
    # the lane B pod cache shares the completeness rule
    with torch_blocked():
        off = importlib.import_module("cascade.trainer.lane_b_offload")
    assert off._complete(dest)
    (dest / "weights.safetensors").write_bytes(_chunked_object(_safetensors_bytes())[1]
                                               ["weights.safetensors"])
    assert not off._complete(dest), "a marker over an index is incomplete on the pod too"


def test_e_malformed_index_shape_is_refused(tmp_path, monkeypatch):
    h = _h()
    weights = _safetensors_bytes()
    manifest, files, packs = _chunked_object(weights)
    doc = json.loads(files["weights.safetensors"])
    doc["file"]["size"] += 1                        # sizes no longer sum
    files = dict(files, **{"weights.safetensors": json.dumps(doc).encode()})
    _install(monkeypatch, files, FakeRegistry(manifest, packs))
    with pytest.raises(h.StorageError):
        h.fetch_from_hub(f"{REPO}@{_sha(manifest)}", tmp_path / "snap")
    assert not (tmp_path / "snap").exists()


def test_f_hub_push_forces_the_single_blob_layout(tmp_path, monkeypatch):
    h = _h()
    seen = {}

    def upload_folder(**kw):
        seen["chunked_write"] = __import__("os").environ.get("HIPPIUS_CHUNKED_WRITE")
        return types.SimpleNamespace(oid="sha256:" + "c" * 64)

    mod = types.ModuleType("hippius_hub")
    mod.upload_folder = upload_folder
    monkeypatch.setitem(sys.modules, "hippius_hub", mod)
    monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "test-token")
    monkeypatch.setenv("HIPPIUS_CHUNKED_WRITE", "1")   # an operator default; restored after
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    up = h.upload_dir_to_hub(d, REPO)
    assert up.ref.digest == "sha256:" + "c" * 64
    assert seen["chunked_write"] == "0", \
        "a push must not write chunked-v2 (the pinned hippius_hub 0.5.1 cannot read it)"
