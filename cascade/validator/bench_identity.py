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


def current_revisions(data_dir=None, *, require_fev=True):
    revisions = {}
    for suite, env in (("gift-eval", "GIFT_EVAL"), ("boom", "BOOM"),
                       ("time", "CASCADE_BENCH_TIME_DATASET")):
        root = Path(data_dir) / suite if data_dir else Path(os.environ.get(env) or ".")
        try:
            marker = json.loads((root / "_cascade_revision.json").read_text())
            revision = marker["revision"]
            if (not isinstance(revision, str) or not revision or marker.get("patterns")
                    or "unknown" in revision or "partial" in revision):
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
