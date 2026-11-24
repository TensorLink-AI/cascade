"""Producer of the published-weights registry the W3 intake reads for dedup.

The intake (:mod:`cascade.lane_d.service`, read through
:class:`cascade.lane_d.chain_reads.PublishedWeightsFile`) refuses every submit
with ``503 dedup_unavailable`` until ``<state-dir>/published_weights.json``
exists. This module writes that file::

    {"schema": "lane-d-published-weights/1", "weights_sha256": [<64-hex>, ...]}

from published sources only:

* **lane D kings**: the signed ``champions/lane_d/index.json``
  (:func:`cascade.lane_d.publish.verify_index` against the pinned signer),
  then each king's ``manifest.json`` read from a public mirror
  (``lane-d-kings/<digest>/manifest.json``). The manifest must re-hash to the
  signed digest, so the ``weights.safetensors`` sha256 it lists is bound to
  the signature. Taken-down kings stay listed: a takedown removes copies, not
  the dedup.
* **lane A legs and kings, and the current house**: a checkpoint directory
  fetched from its published ref (``cascade.shared.hippius.fetch_from_hub``
  verifies the layers on fetch); the sha256 of its ``weights.safetensors``.
  For the house, pass the checkpoint the house is scored as (D47: its BF16
  cast).

The write is atomic (temp file in the same directory, fsync, ``os.replace``,
directory fsync). By default the shas already in the file are kept, so a
published checkpoint never leaves the registry; an existing file that is not
a valid registry is refused rather than overwritten.

    python -P -m cascade.lane_d.published_weights --out <state-dir>/published_weights.json \\
        [--lane-d-index index.json --index-signer SS58 --mirror-dir DIR] \\
        [--checkpoint-dir DIR ...] [--shas-file FILE ...] [--replace] [--dry-run]

The sources are fetched by :mod:`cascade.lane_d.published_sources`:
``mirror-index`` writes the ``--lane-d-index`` / ``--mirror-dir`` pair
(box-safe) and ``checkpoint-shas`` (pod side) the ``--shas-file`` of lane A
checkpoints with their D47 BF16-cast shas.

Recurring (``deploy/laned/systemd/cascade-laned-published-weights.*``): an
unchanged registry is not rewritten (``current``); ``--dry-run`` reads and
verifies every source and writes nothing. Exit 0 written / current /
dry-run, 1 a source failed (nothing written), 2 usage.

Torch-free: stdlib plus the lane D intake/publish helpers.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .chain_reads import PUBLISHED_WEIGHTS_SCHEMA
from .intake import MANIFEST_OBJECT, manifest_digest
from .publish import mirror_prefix, verify_index

WEIGHTS_FILE = "weights.safetensors"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PublishedWeightsSourceError(ValueError):
    """A source can't be read or doesn't verify; nothing is written."""


def _check_sha(value, where: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PublishedWeightsSourceError(f"{where}: {value!r} is not a 64-hex sha256")
    return value


def registry_doc(shas: Iterable[str]) -> dict:
    """The registry document: sorted, de-duplicated, every value checked."""
    return {"schema": PUBLISHED_WEIGHTS_SCHEMA,
            "weights_sha256": sorted({_check_sha(s, "weights_sha256") for s in shas})}


def read_registry(path: str | os.PathLike) -> set[str]:
    """The shas in an existing registry; ``set()`` when ``path`` is absent.
    A file that exists but isn't a valid registry raises."""
    p = Path(path)
    if not p.exists():
        return set()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise PublishedWeightsSourceError(f"{p} is unreadable ({type(e).__name__})") from None
    if not isinstance(doc, dict) or doc.get("schema") != PUBLISHED_WEIGHTS_SCHEMA \
            or not isinstance(doc.get("weights_sha256"), list):
        raise PublishedWeightsSourceError(f"{p} is not a {PUBLISHED_WEIGHTS_SCHEMA} file")
    return {_check_sha(s, str(p)) for s in doc["weights_sha256"]}


def write_registry(path: str | os.PathLike, shas: Iterable[str]) -> dict:
    """Write the registry atomically; returns the document written."""
    p = Path(path)
    doc = registry_doc(shas)
    data = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    try:
        dfd = os.open(p.parent, os.O_RDONLY)
    except OSError:
        return doc
    try:
        with contextlib.suppress(OSError):
            os.fsync(dfd)
    finally:
        os.close(dfd)
    return doc


def lane_d_king_weights(index: Mapping, signer_ss58: str, mirror) -> list[str]:
    """The ``weights.safetensors`` sha256 of every king in a signed lane D
    index. ``mirror.get(key)`` returns bytes (``KeyError`` when absent)."""
    try:
        verified = verify_index(index, signer_ss58)
    except Exception as e:  # noqa: BLE001 — SignatureVerifierUnavailable and kin
        raise PublishedWeightsSourceError(
            f"the lane D index signature can't be checked ({type(e).__name__})") from None
    if not verified:
        raise PublishedWeightsSourceError(
            f"the lane D index does not verify against the pinned signer {signer_ss58}")
    out = []
    for king in index["kings"]:
        digest = king["digest"]
        try:
            raw = bytes(mirror.get(mirror_prefix(digest) + MANIFEST_OBJECT))
        except KeyError:
            raise PublishedWeightsSourceError(
                f"king {digest}: {MANIFEST_OBJECT} is missing on the mirror") from None
        try:
            manifest = json.loads(raw.decode("utf-8"))
            ok = manifest_digest(manifest) == digest
            sha = manifest["files"][WEIGHTS_FILE]["sha256"]
        except (UnicodeDecodeError, ValueError, KeyError, TypeError):
            raise PublishedWeightsSourceError(
                f"king {digest}: {MANIFEST_OBJECT} is not a checkpoint manifest") from None
        if not ok:
            raise PublishedWeightsSourceError(
                f"king {digest}: {MANIFEST_OBJECT} does not hash to the signed digest")
        out.append(_check_sha(sha, f"king {digest} {WEIGHTS_FILE}"))
    return out


def checkpoint_weights(directory: str | os.PathLike) -> str:
    """The sha256 of a fetched published checkpoint's ``weights.safetensors``."""
    p = Path(directory) / WEIGHTS_FILE
    h = hashlib.sha256()
    try:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise PublishedWeightsSourceError(f"{p} is unreadable ({type(e).__name__})") from None
    return h.hexdigest()


class DirMirror:
    """A local copy of a public mirror: ``key`` -> ``<root>/<key>``."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()

    def get(self, key: str) -> bytes:
        p = (self.root / key).resolve()
        if self.root not in p.parents or not p.is_file():
            raise KeyError(key)
        return p.read_bytes()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -P -m cascade.lane_d.published_weights",
        description="Write the lane D intake's published-weights registry from published sources.")
    p.add_argument("--out", type=Path, required=True,
                   help="the registry file (the intake's <state-dir>/published_weights.json)")
    p.add_argument("--lane-d-index", type=Path, help="the signed champions/lane_d/index.json")
    p.add_argument("--index-signer", help="the pinned ss58 signer of the lane D index")
    p.add_argument("--mirror-dir", type=Path,
                   help="a local copy of a public mirror holding lane-d-kings/<digest>/")
    p.add_argument("--checkpoint-dir", type=Path, action="append", default=[],
                   help="a published lane A leg/king or house checkpoint, fetched from its "
                        "ref (repeatable)")
    p.add_argument("--shas-file", type=Path, action="append", default=[],
                   help="a laned_published_shas/1 file (published_sources checkpoint-shas): "
                        "lane A checkpoint shas and their BF16 casts (repeatable)")
    p.add_argument("--replace", action="store_true",
                   help="drop the shas already in --out (default: keep them)")
    p.add_argument("--dry-run", action="store_true",
                   help="read and verify every source, print the counts, write nothing")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_arg_parser().parse_args(argv)
    index_args = (a.lane_d_index, a.index_signer, a.mirror_dir)
    if any(x is not None for x in index_args) and not all(x is not None for x in index_args):
        print("published_weights: --lane-d-index, --index-signer and --mirror-dir go together",
              file=sys.stderr)
        return 2
    try:
        shas = set() if a.replace else read_registry(a.out)
        kept = len(shas)
        if a.lane_d_index is not None:
            try:
                index = json.loads(a.lane_d_index.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as e:
                raise PublishedWeightsSourceError(
                    f"{a.lane_d_index} is unreadable ({type(e).__name__})") from None
            shas.update(lane_d_king_weights(index, a.index_signer, DirMirror(a.mirror_dir)))
        for d in a.checkpoint_dir:
            shas.add(checkpoint_weights(d))
        for f in a.shas_file:
            from .published_sources import SourceFetchError, read_shas_file

            try:
                shas.update(read_shas_file(f))
            except SourceFetchError as e:
                raise PublishedWeightsSourceError(str(e)) from None
        doc = registry_doc(shas)
        previous = None
        if a.out.exists():
            try:
                previous = read_registry(a.out)
            except PublishedWeightsSourceError:
                if not a.replace:      # --replace may overwrite an invalid file, as before
                    raise
        if previous is not None and set(doc["weights_sha256"]) == previous:
            # Idempotent: an unchanged registry is not rewritten.
            print(f"published_weights: current, {len(previous)} sha256 in {a.out}")
            return 0
        added = len(set(doc["weights_sha256"]) - (previous or set()))
        if a.dry_run:
            print(f"published_weights: dry-run, would write {len(doc['weights_sha256'])} sha256 "
                  f"to {a.out} ({added} new, {kept} kept); nothing written")
            return 0
        doc = write_registry(a.out, shas)
    except PublishedWeightsSourceError as e:
        print(f"published_weights: {e}; nothing written", file=sys.stderr)
        return 1
    print(f"published_weights: wrote {len(doc['weights_sha256'])} sha256 to {a.out} "
          f"({added} new, {kept} kept from the previous file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
