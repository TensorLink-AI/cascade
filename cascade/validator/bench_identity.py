"""Cheap data-pin discovery for benchmark cache invalidation (no model load).

This reads the operator's declared pins. Fresh sweeps still verify the actual
FEV bytes in the isolated benchmark environment before producing any report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MARKER = "_cascade_revision.json"


def listing_digest(root) -> tuple[int, str]:
    """``(n_files, sha256)`` of a data dir's file listing (sorted relative
    paths + byte sizes; marker and ``.cache`` excluded). MUST stay identical
    to ``cascade_benchmark.datasets.listing_digest`` (tests pin both)."""
    root = Path(root)
    entries = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if rel.parts[0] == ".cache" or rel.as_posix() == MARKER or not p.is_file():
            continue
        entries.append([rel.as_posix(), p.stat().st_size])
    entries.sort()
    blob = json.dumps(entries, separators=(",", ":")).encode()
    return len(entries), hashlib.sha256(blob).hexdigest()


def current_revisions(data_dir=None, *, require_fev=True):
    revisions = {}
    for suite, env in (("gift-eval", "GIFT_EVAL"), ("boom", "BOOM"),
                       ("time", "CASCADE_BENCH_TIME_DATASET")):
        root = Path(data_dir) / suite if data_dir else Path(os.environ.get(env) or ".")
        try:
            marker = json.loads((root / MARKER).read_text())
            revision = marker["revision"]
            if (not isinstance(revision, str) or not revision or marker.get("patterns")
                    or "unknown" in revision or "partial" in revision):
                return None
            # A marker that recorded its download listing must still match it:
            # a pull that lost files (or was never complete) is not that revision.
            if "listing_sha256" in marker and listing_digest(root) != (
                    marker.get("n_files"), marker["listing_sha256"]):
                return None
            revisions[suite] = revision
        except (OSError, ValueError, KeyError, TypeError):
            return None
    if require_fev:
        path = (os.environ.get("CASCADE_BENCH_FEV_MANIFEST")
                or (str(Path(data_dir) / "fev/manifest.json") if data_dir else None))
        if not path:
            return None
        try:
            manifest = json.loads(Path(path).read_text())
            payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
            revisions["fev"] = hashlib.sha256(payload.encode()).hexdigest()
        except (OSError, ValueError, TypeError):
            return None
    return revisions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir")
    p.add_argument("--legacy-three", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(current_revisions(args.data_dir, require_fev=not args.legacy_three)))


if __name__ == "__main__":
    main()
