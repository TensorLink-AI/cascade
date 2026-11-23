#!/usr/bin/env python3
"""Copy the signed lane D index into the repo and write the king cards.

Run by ``.github/workflows/laned-king-sync.yml`` (LANE_D_PLAN §11.1, W13).
It reads the public signed index (``champions/lane_d/index.json`` in the
``[storage]`` public manifest bucket, the key
``cascade.lane_d.publish.INDEX_KEY``), checks it under the pinned validator
hotkey (``[manifest] validator_hotkey``) and writes
``champions/lane_d/index.json`` plus ``champions/lane_d/<digest>/
{PROVENANCE.json, MODEL_CARD.md}`` through
:func:`cascade.lane_d.publish.sync_cards`. No weights land in git, and the
script holds no storage or HF credential: the index is public-read.

Nothing published yet (HTTP 404: lane D unarmed or no king crowned) is a
no-op that leaves the tree byte-identical. Any other fetch failure, a
malformed document or an index that does not verify exits non-zero and
writes nothing.

Exit codes: 0 synced or nothing to do, 1 fetch/verify failure, 2 bad config.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cascade.lane_d.publish import INDEX_KEY, PublishError, sync_cards  # noqa: E402

MAX_INDEX_BYTES = 8 * 1024 * 1024


class FetchFailed(RuntimeError):
    """The published index could not be read (anything but a 404)."""


def index_url(cfg) -> str:
    """``<[storage] s3_endpoint>/<manifest_bucket>/champions/lane_d/index.json``."""
    endpoint = str(getattr(cfg.storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(cfg.storage, "manifest_bucket", "") or "")
    if not endpoint.startswith("https://") or not bucket:
        raise ValueError("[storage] s3_endpoint must be https:// and manifest_bucket set "
                         f"(got {endpoint!r}, {bucket!r})")
    return f"{endpoint}/{bucket}/{INDEX_KEY}"


def fetch_index(url: str, *, timeout: float) -> dict | None:
    """The published index, or ``None`` when nothing is published (404)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https only
            raw = resp.read(MAX_INDEX_BYTES + 1)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 404:
            return None
        raise FetchFailed(f"HTTP {e.code} for {url}") from None
    except (OSError, ValueError) as e:
        raise FetchFailed(f"{type(e).__name__} for {url}: {e}") from None
    if len(raw) > MAX_INDEX_BYTES:
        raise FetchFailed(f"{url} is larger than {MAX_INDEX_BYTES} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FetchFailed(f"{url} is not JSON: {e}") from None
    if not isinstance(doc, dict):
        raise FetchFailed(f"{url} is a JSON {type(doc).__name__}, not an object")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--repo-root", type=Path, default=REPO,
                    help="Repository root the cards are written under.")
    ap.add_argument("--chain-toml", type=Path, default=None,
                    help="chain.toml to read [storage] and [manifest] from (default: shipped).")
    ap.add_argument("--timeout", type=float, default=30.0, help="Fetch timeout in seconds.")
    args = ap.parse_args(argv)

    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(args.chain_toml)
    signer = str(getattr(cfg.manifest, "validator_hotkey", "") or "").strip()
    if not signer:
        print("laned-king-sync: [manifest] validator_hotkey is unpinned; the index's own "
              "signer is never trusted", file=sys.stderr)
        return 2
    try:
        url = index_url(cfg)
    except ValueError as e:
        print(f"laned-king-sync: {e}", file=sys.stderr)
        return 2
    try:
        index = fetch_index(url, timeout=args.timeout)
    except FetchFailed as e:
        print(f"laned-king-sync: could not read the lane D index: {e}", file=sys.stderr)
        return 1
    if index is None:
        print(f"laned-king-sync: no lane D index published at {url}; nothing to sync")
        return 0
    try:
        written = sync_cards(args.repo_root, index, signer)
    except PublishError as e:
        print(f"laned-king-sync: refused: {e}", file=sys.stderr)
        return 1
    kings = len(index.get("kings") or [])
    if not written:
        print(f"laned-king-sync: {kings} lane D king(s); cards unchanged")
        return 0
    print(f"laned-king-sync: {kings} lane D king(s); wrote {len(written)} file(s):")
    for path in written:
        print(f"  {Path(path).resolve().relative_to(args.repo_root.resolve()).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
