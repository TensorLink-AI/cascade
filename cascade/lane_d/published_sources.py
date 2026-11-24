"""Fetchers for the published-weights registry sources (W3; LANE_D_PLAN §1.2
dedup, D47).

:mod:`cascade.lane_d.published_weights` writes the intake's dedup registry
from LOCAL inputs. This module fetches them:

* ``mirror-index`` (box-safe: kilobytes): the signed
  ``champions/lane_d/index.json`` from its public URL, verified against the
  pinned signer, and every king's ``manifest.json`` from a public mirror
  (``<mirror-url>/lane-d-kings/<digest>/manifest.json``, the R2 public or the
  HF dataset ``resolve/main`` URL), each re-hashed to its signed digest,
  written as the ``--mirror-dir`` layout ``published_weights`` reads. Plain
  unauthenticated GETs, size-capped; no credential.
* ``checkpoint-shas`` (POD / builder side: it downloads checkpoints and casts
  them, never the production box): every published lane A leg and king ref
  (``--ref``, or every ``metro-v1:trained:hippius:<repo>@<digest>`` pointer
  found in the ``--refs-from`` JSON documents, e.g. the round manifests),
  fetched with :func:`cascade.shared.hippius.fetch_from_hub`; writes
  ``laned_published_shas/1`` with each checkpoint's ``weights.safetensors``
  sha256 AND the sha256 of its D47 BF16 cast
  (:func:`cascade.lane_d.house_source.cast_safetensors_bf16`, deterministic),
  so a byte copy or a cast copy of any published checkpoint (the house
  included: the house IS the cast of KING_A) dedups. ``published_weights
  --shas-file`` reads it.

    python -P -m cascade.lane_d.published_sources mirror-index \\
        --index-url URL --mirror-url URL --signer-ss58 SS58 --dest DIR
    python -P -m cascade.lane_d.published_sources checkpoint-shas \\
        (--ref REPO@DIGEST ... | --refs-from FILE ...) --work-dir DIR --out FILE

Exit 0 written, 1 a source failed (nothing written), 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from .intake import MANIFEST_OBJECT, manifest_digest
from .publish import INDEX_KEY, mirror_prefix, verify_index

SHAS_SCHEMA = "laned_published_shas/1"
WEIGHTS_FILE = "weights.safetensors"
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
_TRAINED = re.compile(r"metro-v1:trained:hippius:(\S+@\S+)")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class SourceFetchError(ValueError):
    """A source cannot be fetched or does not verify; nothing is written."""


def _join(base: str, key: str) -> str:
    return base.rstrip("/") + "/" + key


def http_get(url: str, *, max_size: int) -> bytes:
    """One unauthenticated, size-capped GET (the presigned-GET reader: no
    credential header)."""
    from .storage import FetchError, fetch_presigned_bytes

    try:
        return fetch_presigned_bytes(url, max_size=max_size)
    except FetchError as e:
        raise SourceFetchError(f"GET failed: {e}") from None


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def mirror_index(*, index_url: str, mirror_url: str, signer_ss58: str, dest,
                 get: Callable[..., bytes] = http_get) -> dict:
    """Fetch and verify the signed index and every king manifest into
    ``dest`` (``dest/champions/lane_d/index.json``,
    ``dest/lane-d-kings/<digest>/manifest.json``). Verified in full before
    anything is written."""
    raw = get(index_url, max_size=MAX_INDEX_BYTES)
    try:
        index = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SourceFetchError(f"{index_url} is not JSON") from None
    if not verify_index(index, signer_ss58):
        raise SourceFetchError(f"{index_url} does not verify under {signer_ss58}")
    manifests: dict[str, bytes] = {}
    for king in index["kings"]:
        digest = king["digest"]
        key = mirror_prefix(digest) + MANIFEST_OBJECT
        blob = get(_join(mirror_url, key), max_size=MAX_MANIFEST_BYTES)
        try:
            ok = manifest_digest(json.loads(blob.decode("utf-8"))) == digest
        except (UnicodeDecodeError, ValueError):
            ok = False
        if not ok:
            raise SourceFetchError(f"king {digest}: the mirror's {MANIFEST_OBJECT} does not "
                                   "hash to the signed digest")
        manifests[key] = blob
    root = Path(dest)
    for key, blob in sorted(manifests.items()):
        _write(root / key, blob)
    _write(root / INDEX_KEY, raw)
    return {"index": str(root / INDEX_KEY), "kings": len(manifests), "mirror_dir": str(root)}


def trained_refs(docs: Iterable) -> list[str]:
    """Every Hub ``repo@digest`` of a ``metro-v1:trained:hippius:`` pointer
    anywhere in the JSON values ``docs`` (sorted, unique)."""
    from ..shared.hippius import is_hub_ref

    out: set[str] = set()

    def walk(v) -> None:
        if isinstance(v, str):
            for m in _TRAINED.finditer(v):
                ref = m.group(1).rstrip("\"',;)")
                if is_hub_ref(ref):
                    out.add(ref)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list | tuple):
            for x in v:
                walk(x)

    for d in docs:
        walk(d)
    return sorted(out)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checkpoint_shas(refs: Iterable[str], *, work_dir, fetch: Callable | None = None) -> dict:
    """``laned_published_shas/1`` over ``refs``: each fetched checkpoint's
    weights sha256 and its BF16-cast sha256. Each checkpoint is removed
    after it is hashed."""
    from .house_source import HouseSourceError, cast_safetensors_bf16

    if fetch is None:
        from ..shared.hippius import fetch_from_hub as fetch
    sources = []
    for i, ref in enumerate(sorted(set(refs))):
        d = Path(work_dir) / f"ckpt-{i}"
        try:
            try:
                fetch(ref, d)
            except Exception as e:  # noqa: BLE001 — StorageError, an auth or transport error
                raise SourceFetchError(f"{ref}: fetch failed ({type(e).__name__}: {e})") from None
            p = d / WEIGHTS_FILE
            if not p.is_file():
                raise SourceFetchError(f"{ref} has no {WEIGHTS_FILE}")
            raw = p.read_bytes()
            try:
                cast = _sha(cast_safetensors_bf16(raw))
            except HouseSourceError as e:
                raise SourceFetchError(f"{ref}: {e}") from None
            sources.append({"ref": ref, "weights_sha256": _sha(raw), "bf16_weights_sha256": cast})
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return {"schema": SHAS_SCHEMA, "sources": sources}


def read_shas_file(path) -> list[str]:
    """Every sha in a ``laned_published_shas/1`` file (raw and cast)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise SourceFetchError(f"{path} is unreadable ({type(e).__name__})") from None
    if not isinstance(doc, dict) or doc.get("schema") != SHAS_SCHEMA \
            or not isinstance(doc.get("sources"), list):
        raise SourceFetchError(f"{path} is not a {SHAS_SCHEMA} file")
    out = []
    for i, s in enumerate(doc["sources"]):
        for k in ("weights_sha256", "bf16_weights_sha256"):
            v = s.get(k) if isinstance(s, dict) else None
            if not isinstance(v, str) or not _HEX64.fullmatch(v):
                raise SourceFetchError(f"{path}: sources[{i}].{k} is not a 64-hex sha256")
            out.append(v)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -P -m cascade.lane_d.published_sources",
                                description="Fetch the published-weights registry sources.")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mirror-index", help="the signed index + king manifests (box-safe)")
    m.add_argument("--index-url", required=True, help="public URL of champions/lane_d/index.json")
    m.add_argument("--mirror-url", required=True,
                   help="public base URL of a mirror holding lane-d-kings/<digest>/")
    m.add_argument("--signer-ss58", required=True, help="the pinned index signer")
    m.add_argument("--dest", type=Path, required=True, help="the --mirror-dir to write")
    c = sub.add_parser("checkpoint-shas", help="lane A checkpoint shas + BF16 casts (pod side)")
    c.add_argument("--ref", action="append", default=[], help="a published repo@digest")
    c.add_argument("--refs-from", type=Path, action="append", default=[],
                   help="a JSON document whose trained pointers are published checkpoints")
    c.add_argument("--work-dir", type=Path, required=True, help="scratch for one checkpoint")
    c.add_argument("--out", type=Path, required=True, help="the laned_published_shas/1 file")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_arg_parser().parse_args(argv)
    try:
        if a.command == "mirror-index":
            out = mirror_index(index_url=a.index_url, mirror_url=a.mirror_url,
                               signer_ss58=a.signer_ss58, dest=a.dest)
        else:
            docs = []
            for f in a.refs_from:
                try:
                    docs.append(json.loads(f.read_text(encoding="utf-8")))
                except (OSError, UnicodeDecodeError, ValueError) as e:
                    raise SourceFetchError(f"{f} is unreadable ({type(e).__name__})") from None
            refs = sorted(set(a.ref) | set(trained_refs(docs)))
            if not refs:
                print("published_sources: no --ref and no trained pointer in --refs-from",
                      file=sys.stderr)
                return 2
            doc = checkpoint_shas(refs, work_dir=a.work_dir)
            _write(a.out, (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode())
            out = {"out": str(a.out), "checkpoints": len(doc["sources"])}
    except (SourceFetchError, OSError) as e:
        print(f"published_sources: {e}; nothing written", file=sys.stderr)
        return 1
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SHAS_SCHEMA", "SourceFetchError", "checkpoint_shas", "http_get", "mirror_index",
           "read_shas_file", "trained_refs"]
